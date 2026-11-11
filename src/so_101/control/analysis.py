"""Read a recorded `so101 serve` run back into attempts, interruptions, suspect frames and training cuts.

`so101 runs`, `so101 review --runs` and `so101 build --with` all read runs through `analyse`, which
is two steps. `read_run` reads the run's files once into a `RunRecord`: its `events.jsonl`, and the
`control.phase`, `control.policy_action` and `control.leader` columns of its dataset with its fps.
`analyse_record` derives everything below from that record, the run's `labels.toml` and `blame_s`,
and reads no file, so `review --runs` keeps the record across its decisions and the rules can be
run on small inputs. Nothing here modifies the recorded dataset, `events.jsonl` or `run.json`. The one file of a run that changes is
`labels.toml`, which `save_labels` writes for `so101 review --runs`. Each file's place in the run's
folder comes from `data.Run`, which the functions here take.

Each Parquet data file of the dataset is read on its own. One that can't be read, such as the file a
hard kill during a save leaves without its footer, is skipped and named in `notes` and `skipped`, and
only the episodes in it are lost: an attempt whose episode was in it counts as not saved. The fps
comes from `meta/info.json`, or from the recorder's copy of it when a kill during a save tore it
(`data.Run.info`).

Frame numbers are an attempt's episode frames, and every range is half-open, `[start, stop)`, like
the `rewind` event's.

- **Segments:** each run of one `control.phase` in an attempt, with who drove: the policy
  (Autonomous), a human (Teleop), or the script (Starting, Paused, Handover, Rewinding, Returning,
  in which the controller moves or holds the arm itself).
- **Waiting frames:** Autonomous frames whose `control.policy_action` is all NaN: the policy sent no
  action that tick, as while a cloud GPU's next chunk is late. They are script time, not policy
  time, in no training cut, count toward no blame window, and the policy hasn't started driving
  until its first action. A Teleop frame with no leader pose (`control.leader` all NaN, the tick the
  leader was lost) is in no cut either.
- **Interruptions:** each Autonomous segment that an operator's Pause, Rewind or Teleop ended. Reset
  and Quit end the attempt rather than interrupt it, and a time limit, fault or lost link isn't an
  operator's judgement of the policy. Each has the reason from a `label` event, if the operator gave
  one, and what came next: resume, rewind, teleop, how the attempt was asked to end (reset, quit or
  time_limit, from the `state` event into Returning), or end (the episode ended there). A Pause
  and Resume applied in the same tick leave no Paused frame, but their `state` events still make
  an interruption.
- **Suspect frames:** the frames that led to an interruption, never used as training targets. When
  the operator rewound right after it (only pauses in between), they are the range the rewinds
  undid. Otherwise they are the last `blame_s` seconds of policy frames before it, counting back
  through pauses and waiting frames but not through anything else. The frames any other rewind
  undid, such as one from Teleop or after a lost link, are suspect too.
- **Corrections:** the Teleop segments after the policy had driven in the attempt (sent its first
  action), a human stepping in on the policy. Only their frames are interventions (`intervention_mask`). Teleop before the
  policy first drove, as in a human attempt, corrects no policy: it is a human driving, like a demo.
- **Failure-fix pairs:** a rewind that undid policy frames, followed by Teleop from the pose it went
  back to (only pauses and the handover in between).
- **Outcomes:** from the `outcome` events, overridden by `labels.toml`. Ignored attempts count in no
  metric and give no training cut.
- **Dropped video frames:** an attempt whose `attempt_end` has `dropped_frames` gives no training cut,
  with a note: its video and its other columns don't line up. It still counts in the metrics.

`labels.toml` holds review decisions, which `so101 review --runs` writes:

    [[attempt]]
    attempt = 3
    outcome = "failed"        # succeeded, failed, ignored, or unlabelled

    [[interruption]]
    attempt = 3
    frame = 412               # the interruption's frame: the first frame after the policy stopped
    suspect = [380, 412]      # the suspect frames [start, stop); [] means none

An entry that matches no attempt or interruption of the run, and a top-level key other than these
two, change nothing and are named in `notes`.
"""

import json
import math
import tomllib
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from so_101.control.recorder import OUTCOMES, Phase
from so_101.control.states import STEP_INS
from so_101.data import LABELS_FILE, Run, whole
from so_101.settings import ConfigError
UNLABELLED = "unlabelled"

POLICY, HUMAN, SCRIPT = "policy", "human", "script"
ACTORS = {Phase.AUTONOMOUS: POLICY, Phase.TELEOP: HUMAN}

# What came after an interruption, from the first segment that wasn't Paused. Returning becomes how
# the attempt was asked to end: reset, quit or time_limit (ENDINGS).
RESET = "reset"
ENDINGS = (RESET, "quit", "time_limit")
NEXT = {
    Phase.AUTONOMOUS: "resume",
    Phase.REWINDING: "rewind",
    Phase.HANDOVER: "teleop",
    Phase.TELEOP: "teleop",
    Phase.RETURNING: RESET,
    Phase.STARTING: "resume",
}

# Training sources the analysis can cut, besides `demos`, which build reads from the sessions.
CORRECTIONS, HUMAN_ATTEMPTS, POLICY_SUCCESSES = "corrections", "human-attempts", "policy-successes"
SOURCES = (CORRECTIONS, HUMAN_ATTEMPTS, POLICY_SUCCESSES)

# The version of the rules that cut and label a run's frames, which build.json records. Raise it when
# `training_cuts`, `intervention_mask` or `frame_labels` change what a build holds, so builds made
# under the old rules aren't current.
RULES_VERSION = 5

Range = tuple[int, int]


class FrameLabel(IntEnum):
    """`control.label` in a build that keeps labels."""

    SUSPECT = -1
    NEUTRAL = 0
    CORRECTION = 1


@dataclass(frozen=True)
class Segment:
    phase: Phase
    start: int
    stop: int

    @property
    def actor(self) -> str:
        return ACTORS.get(self.phase, SCRIPT)

    @property
    def frames(self) -> int:
        return self.stop - self.start

    @property
    def range(self) -> Range:
        return (self.start, self.stop)


@dataclass(frozen=True)
class Interruption:
    segment: Segment  # the Autonomous segment it ended
    command: str  # pause, rewind or teleop
    reason: str | None  # why the operator paused, from a `label` event
    next: str  # resume, rewind, teleop, an ending (reset, quit or time_limit), or end
    rewound: Range | None  # the frames the rewinds right after it undid
    suspect: Range | None  # the suspect frames, after labels.toml
    suspect_source: str  # rewind, blame or labels

    @property
    def frame(self) -> int:
        """The first frame after the policy stopped: what labels.toml names it by."""
        return self.segment.stop

    @property
    def then(self) -> str:
        """What came next, in words: `resume`, or for an ending, `ended by quit`."""
        return f"ended by {self.next}" if self.next in ENDINGS else self.next


@dataclass(frozen=True)
class FailureFix:
    """A rewound policy segment and the human's fix from the pose the rewind went back to."""

    attempt: int
    episode: int
    failure: Range  # the frames the rewind undid
    fix: Segment  # the Teleop segment after it


@dataclass(frozen=True)
class Cut:
    """Frames `[start, stop)` of a run's episode that a build may train on."""

    attempt: int
    episode: int
    start: int
    stop: int
    source: str  # corrections, human-attempts or policy-successes

    @property
    def frames(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Attempt:
    number: int
    episode: int | None  # None when it wasn't saved, as when the process quit away from rest
    ended_by: str | None
    recorded_outcome: str | None  # from the run's `outcome` events
    outcome: str | None  # after labels.toml; None is unlabelled
    phases: np.ndarray = field(repr=False, compare=False)  # control.phase per frame
    # Per frame: an Autonomous frame where the policy sent no action, and a Teleop frame with no leader pose.
    waiting: np.ndarray = field(repr=False, compare=False)
    leaderless: np.ndarray = field(repr=False, compare=False)
    segments: tuple[Segment, ...]
    interruptions: tuple[Interruption, ...]
    rewinds: tuple[Range, ...]
    failure_fixes: tuple[FailureFix, ...]
    dropped_frames: dict[str, int] = field(default_factory=dict)  # video frames dropped per camera, from attempt_end

    @property
    def frames(self) -> int:
        return len(self.phases)

    @property
    def ignored(self) -> bool:
        return self.outcome == "ignored"

    @property
    def policy_start(self) -> int:
        """The first frame the policy drove (sent an action), or the attempt's length if it never did."""
        driven = np.flatnonzero((self.phases == Phase.AUTONOMOUS) & ~self.waiting)
        return int(driven[0]) if len(driven) else self.frames

    def seconds(self, fps: float) -> dict[str, float]:
        """Time per actor: policy, human and script. Waiting frames are script time: the arm held while
        the policy sent nothing."""
        totals = dict.fromkeys((POLICY, HUMAN, SCRIPT), 0.0)
        for s in self.segments:
            totals[s.actor] += s.frames / fps
        waited = int(self.waiting.sum()) / fps
        totals[POLICY] -= waited
        totals[SCRIPT] += waited
        return totals

    def suspect_mask(self) -> np.ndarray:
        """Each interruption's suspect frames, and every other rewind's undone frames."""
        mask = np.zeros(self.frames, dtype=bool)
        for i in self.interruptions:
            if i.suspect:
                mask[i.suspect[0] : i.suspect[1]] = True
        # The interruption a rewind directly follows decides about its frames, labels.toml included.
        # Any other rewind, from Teleop or after a lost link, still undid motion: never train on it.
        claimed = [i.rewound for i in self.interruptions if i.rewound and i.next == "rewind"]
        for start, stop in self.rewinds:
            if not any(a <= start and stop <= b for a, b in claimed):
                mask[start:stop] = True
        return mask

    def intervention_mask(self) -> np.ndarray:
        """`intervention` per frame: a Teleop frame after the policy had driven in this attempt, a
        human stepping in on the policy. A human attempt, or Teleop before the policy first drove,
        corrects no policy, so it isn't one. Derived from the phases, not the recorded column, so
        runs recorded before this rule are labelled the same way."""
        return (self.phases == Phase.TELEOP) & (np.arange(self.frames) >= self.policy_start)

    def frame_labels(self) -> np.ndarray:
        """`control.label` per frame: 1 an intervention frame (see `intervention_mask`), -1 suspect,
        0 neither."""
        labels = np.where(self.intervention_mask(), FrameLabel.CORRECTION, FrameLabel.NEUTRAL).astype(np.int8)
        labels[self.suspect_mask()] = FrameLabel.SUSPECT
        return labels


@dataclass(frozen=True)
class RunAnalysis:
    run: str  # the run's folder name, YYYYmmdd-HHMMSS
    folder: Run  # where the run's files are
    fps: float
    attempts: tuple[Attempt, ...]
    notes: tuple[str, ...]  # what couldn't be read, one line each
    skipped: tuple[Path, ...] = ()  # data files that couldn't be read, whose episodes are left out

    @property
    def counted(self) -> tuple[Attempt, ...]:
        """Saved attempts that aren't ignored: the ones metrics count."""
        return tuple(a for a in self.attempts if a.episode is not None and not a.ignored)

    @property
    def failure_fixes(self) -> tuple[FailureFix, ...]:
        return tuple(p for a in self.counted for p in a.failure_fixes)


@dataclass(frozen=True)
class Metrics:
    attempts: int  # saved and not ignored
    ignored: int
    unsaved: int
    succeeded: int
    failed: int
    unlabelled: int
    interruptions: int
    policy_s: float
    human_s: float

    @property
    def policy_share(self) -> float | None:
        """The share of driving time the policy drove, rather than a human."""
        driving = self.policy_s + self.human_s
        return self.policy_s / driving if driving else None

    @property
    def success_rate(self) -> float | None:
        """Successes among attempts with an outcome."""
        labelled = self.succeeded + self.failed
        return self.succeeded / labelled if labelled else None

    @property
    def success_interval(self) -> tuple[float, float] | None:
        """The 95% Wilson interval around `success_rate`: where the policy's true rate likely lies.
        20 attempts can't tell 50% from 70% apart (10/20 is 30-70%, 14/20 is 48-85%), so compare
        policies by these, not by the rates alone."""
        n = self.succeeded + self.failed
        if not n:
            return None
        z, p = 1.959964, self.succeeded / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, centre - half), min(1.0, centre + half)

    @property
    def interruptions_per_attempt(self) -> float | None:
        return self.interruptions / self.attempts if self.attempts else None

    @classmethod
    def of(cls, analyses: list[RunAnalysis]) -> "Metrics":
        counted = [(a, run.fps) for run in analyses for a in run.counted]
        seconds = [a.seconds(fps) for a, fps in counted]
        outcomes = [a.outcome for a, _ in counted]
        return cls(
            attempts=len(counted),
            ignored=sum(a.episode is not None and a.ignored for run in analyses for a in run.attempts),
            unsaved=sum(a.episode is None for run in analyses for a in run.attempts),
            succeeded=outcomes.count("succeeded"),
            failed=outcomes.count("failed"),
            unlabelled=sum(o not in ("succeeded", "failed") for o in outcomes),
            interruptions=sum(len(a.interruptions) for a, _ in counted),
            policy_s=sum(s[POLICY] for s in seconds),
            human_s=sum(s[HUMAN] for s in seconds),
        )


# --- labels.toml ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Labels:
    """Review decisions for one run."""

    outcomes: dict[int, str | None] = field(default_factory=dict)  # attempt -> outcome; None is unlabelled
    suspects: dict[tuple[int, int], Range | None] = field(default_factory=dict)  # (attempt, frame) -> range
    unknown: tuple[str, ...] = ()  # top-level keys of labels.toml that mean nothing here


LABELS_HEADER = """\
# Review decisions for this run, written by `uv run so101 review --runs`; you can also edit it by hand.
# The run's dataset, events.jsonl and run.json are never changed. Frames are the attempt's episode
# frames, and ranges are [start, stop), stop not included.
#   [[attempt]]        attempt, outcome: succeeded, failed, ignored or unlabelled
#   [[interruption]]   attempt, frame (the first frame after the policy stopped),
#                      suspect = [start, stop] frames, or [] for none
"""


def load_labels(run: Run) -> Labels:
    path = run.labels_path
    if not path.is_file():
        return Labels()
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    outcomes, suspects = {}, {}
    for table in ("attempt", "interruption"):
        if not isinstance(data.get(table, []), list):
            raise ConfigError(f"{path}: {table} must be a list of [[{table}]] tables, not {data[table]!r}.")
    for i, entry in enumerate(data.get("attempt", [])):
        try:
            attempt, outcome = whole(entry["attempt"]), str(entry["outcome"])
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"{path}: [[attempt]] number {i + 1} is malformed ({e!r}).") from e
        if outcome not in (*OUTCOMES, UNLABELLED):
            raise ConfigError(f"{path}: attempt {attempt}: outcome {outcome!r} isn't one of "
                              f"{', '.join((*OUTCOMES, UNLABELLED))}.")
        outcomes[attempt] = None if outcome == UNLABELLED else outcome
    for i, entry in enumerate(data.get("interruption", [])):
        try:
            key = (whole(entry["attempt"]), whole(entry["frame"]))
            frames = [whole(f) for f in entry["suspect"]]
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"{path}: [[interruption]] number {i + 1} is malformed ({e!r}).") from e
        if frames and (len(frames) != 2 or not 0 <= frames[0] < frames[1]):
            raise ConfigError(f"{path}: attempt {key[0]} frame {key[1]}: suspect needs [start, stop] with "
                              "0 <= start < stop, or [] for none.")
        suspects[key] = (frames[0], frames[1]) if frames else None
    return Labels(outcomes, suspects, tuple(key for key in data if key not in ("attempt", "interruption")))


def save_labels(run: Run, labels: Labels) -> None:
    """Write labels.toml, the only file of a run that changes after recording."""
    lines = [LABELS_HEADER]
    for attempt, outcome in sorted(labels.outcomes.items()):
        lines.append(f'\n[[attempt]]\nattempt = {attempt}\noutcome = "{outcome or UNLABELLED}"\n')
    for (attempt, frame), suspect in sorted(labels.suspects.items()):
        frames = f"[{suspect[0]}, {suspect[1]}]" if suspect else "[]"
        lines.append(f"\n[[interruption]]\nattempt = {attempt}\nframe = {frame}\nsuspect = {frames}\n")
    path = run.labels_path
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text("".join(lines))
    tmp.replace(path)


# --- Reading a run --------------------------------------------------------------------------------


def read_events(run: Run) -> list[dict]:
    """The run's events, skipping any line that isn't a JSON object with a string `type`, such as a
    last line a crash cut short."""
    path = run.events_path
    if not path.is_file():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            events.append(event)
    return events


def read_data(root: Path, columns: list[str]) -> tuple["pd.DataFrame | None", list[tuple[Path, str]]]:
    """`columns` of every Parquet file of the dataset at `root`, and the files that couldn't be read,
    each with why. Each file is read on its own, so one a hard kill left without its footer costs
    only its own episodes. The table is None when no file could be read."""
    import pandas as pd

    tables, skipped = [], []
    for f in sorted(Path(root).glob("data/*/*.parquet")):
        try:
            tables.append(pd.read_parquet(f, columns=columns))
        except Exception as e:  # pyarrow raises several types for a cut-off or malformed file
            reason = (str(e).strip().splitlines() or [""])[0]
            skipped.append((f, f"{e.__class__.__name__}: {reason}" if reason else e.__class__.__name__))
    return (pd.concat(tables) if tables else None), skipped


@dataclass(frozen=True)
class Frames:
    """One episode's columns, in frame order."""

    phases: np.ndarray  # control.phase
    waiting: np.ndarray  # an Autonomous frame whose control.policy_action is all NaN
    leaderless: np.ndarray  # a Teleop frame whose control.leader is all NaN

    @classmethod
    def empty(cls) -> "Frames":
        return cls(np.zeros(0, dtype=np.int8), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))


def _no_values(column) -> np.ndarray:
    """Per row, whether the vector is all NaN: nothing was recorded there."""
    return np.array([bool(np.isnan(np.asarray(v, dtype=float)).all()) for v in column], dtype=bool)


def read_frames(run: Run) -> tuple[float | None, dict[int, Frames], list[tuple[Path, str]]]:
    """The dataset's fps, each episode's frames, and the data files that couldn't be read, each with
    why."""
    if (info := run.info()) is None:
        return None, {}, []
    fps = float(info["fps"])
    table, skipped = read_data(run.dataset_root, ["episode_index", "frame_index", "control.phase",
                                                  "control.policy_action", "control.leader"])
    if table is None:
        return fps, {}, skipped
    episodes = {}
    for episode, rows in table.groupby("episode_index"):
        rows = rows.sort_values("frame_index")
        phases = np.array([int(np.ravel(p)[0]) for p in rows["control.phase"]], dtype=np.int8)
        episodes[int(episode)] = Frames(
            phases,
            (phases == Phase.AUTONOMOUS) & _no_values(rows["control.policy_action"]),
            (phases == Phase.TELEOP) & _no_values(rows["control.leader"]),
        )
    return fps, episodes, skipped


@dataclass(frozen=True)
class RunRecord:
    """A run's files, read once (`read_run`): everything `analyse_record` derives from, besides the
    labels and `blame_s`."""

    folder: Run  # where the run's files are
    events: tuple[dict, ...]
    fps: float | None  # None without a readable meta/info.json, nor a copy of one
    episodes: dict[int, Frames]  # each episode's frames, by episode index
    skipped: tuple[tuple[Path, str], ...]  # data files that couldn't be read, each with why


def read_run(run: Run) -> RunRecord:
    """Read a run's events and its dataset's frames, once."""
    events = read_events(run)
    fps, episodes, skipped = read_frames(run)
    return RunRecord(run, tuple(events), fps, episodes, tuple(skipped))


def segments_of(phases: np.ndarray) -> tuple[Segment, ...]:
    if not len(phases):
        return ()
    edges = np.flatnonzero(np.diff(phases)) + 1
    starts = [0, *edges.tolist()]
    stops = [*edges.tolist(), len(phases)]
    return tuple(Segment(Phase(int(phases[a])), a, b) for a, b in zip(starts, stops, strict=True))


def check_blame_s(blame_s: object) -> float:
    """`[control] blame_s`, which `settings.load()` doesn't check: a finite number of seconds, 0 or
    more. Otherwise a one-line ConfigError naming it and the fix: NaN or a string would crash the
    analysis, and a negative number would silently mark no frame as suspect."""
    if isinstance(blame_s, bool) or not isinstance(blame_s, int | float) or not math.isfinite(blame_s) or blame_s < 0:
        raise ConfigError(f"[control] blame_s in so101.toml is {blame_s!r}: set it to a number of seconds, 0 or more, "
                          "such as 2.5 (0 marks no frames as suspect).")
    return float(blame_s)


def analyse(run: Run, blame_s: float) -> RunAnalysis:
    """Read a run and its labels.toml, and analyse them (`analyse_record`). `blame_s` is checked
    before anything is read."""
    blame_s = check_blame_s(blame_s)
    labels = load_labels(run)
    return analyse_record(read_run(run), labels, blame_s)


def analyse_record(record: RunRecord, labels: Labels, blame_s: float) -> RunAnalysis:
    """The analysis of a run's record with its review decisions, `labels`. Reads no file. `blame_s`:
    the seconds of policy frames before an interruption that are suspect when the operator didn't
    rewind, `[control] blame_s`, which the caller checked before reading the run (`check_blame_s`)."""
    run, events, fps, episodes, skipped = record.folder, record.events, record.fps, record.episodes, record.skipped
    notes = []
    for path, why in skipped:
        notes.append(f"skipped {path.relative_to(run.path)}, which can't be read ({why}): its episodes are left out")
    if fps is None:
        fps = 30.0
        if any(e["type"] == "attempt_end" and e.get("saved") for e in events):
            notes.append("its dataset has no readable meta/info.json, nor a copy of one, so its attempts have no frames")

    by_attempt: dict[int, list[dict]] = {}
    for e in events:
        if (n := e.get("attempt")) is not None:
            by_attempt.setdefault(int(n), []).append(e)

    attempts = []
    for number, evs in sorted(by_attempt.items()):
        end = next((e for e in reversed(evs) if e["type"] == "attempt_end"), None)
        episode = end.get("episode") if end else None
        frames = episodes.get(episode) if episode is not None else None
        if episode is not None and frames is None:
            notes.append(f"attempt {number}'s episode {episode} isn't in the dataset")
            episode = None
        if frames is None:
            frames = Frames.empty()
        if episode is not None and (dropped := (end or {}).get("dropped_frames")):
            counts = ", ".join(f"{camera} {n}" for camera, n in dropped.items())
            notes.append(f"attempt {number} dropped video frames when it was recorded ({counts}), so its video and "
                         "its other columns don't line up: it gives no training cut")
        recorded = next((e["outcome"] for e in reversed(evs) if e["type"] == "outcome"), None)
        outcome = labels.outcomes[number] if number in labels.outcomes else recorded
        attempts.append(_attempt(number, episode, end, recorded, outcome, frames, evs, fps, blame_s, labels))
    numbers = {a.number for a in attempts}
    interruptions = {(a.number, i.frame) for a in attempts for i in a.interruptions}
    notes += [f"{LABELS_FILE} has the unknown key {key!r}, which changes nothing" for key in labels.unknown]
    notes += [f"{LABELS_FILE} names attempt {n}, which isn't in this run" for n in sorted(labels.outcomes)
              if n not in numbers]
    notes += [f"{LABELS_FILE} names an interruption at attempt {a} frame {f}, which this run doesn't have"
              for a, f in sorted(labels.suspects) if (a, f) not in interruptions]
    return RunAnalysis(run.name, run, fps, tuple(attempts), tuple(notes),
                       tuple(path for path, _ in skipped))


def _attempt(number, episode, end, recorded, outcome, frames, events, fps, blame_s, labels) -> Attempt:
    phases = frames.phases
    segments = segments_of(phases)
    starts = {s.start: i for i, s in enumerate(segments)}
    rewinds = [e for e in events if e["type"] == "rewind" and "start_frame" in e]
    states = [e for e in events if e["type"] == "state" and e.get("previous") == "autonomous"]
    reasons = [e for e in events if e["type"] == "label"]

    # Each interruption: the policy's segment up to it, the segments after it, and the command.
    found: list[tuple[Segment, tuple[Segment, ...], str]] = []
    stepped_in = sorted((e["frame"], e["cause"]) for e in states
                        if isinstance(e.get("frame"), int) and e.get("cause") in STEP_INS)
    for i, segment in enumerate(segments):
        if segment.phase is not Phase.AUTONOMOUS:
            continue
        # A Pause and a Resume applied in the same tick leave no Paused frame, so the policy's
        # segment runs on through them: find them by their `state` events inside the segment.
        start = segment.start
        for frame, cause in stepped_in:
            if segment.start < frame < segment.stop:
                rest = (Segment(Phase.AUTONOMOUS, frame, segment.stop), *segments[i + 1 :])
                found.append((Segment(Phase.AUTONOMOUS, start, frame), rest, cause))
                start = frame
        if i + 1 == len(segments):
            continue  # the episode ended on the policy: a fault or a crash, not an operator
        ended = next((e for e in states if e.get("frame") == segment.stop), None)
        command = ended.get("cause") if ended else None
        if command in STEP_INS:
            found.append((Segment(Phase.AUTONOMOUS, start, segment.stop), segments[i + 1 :], command))

    interruptions = []
    for segment, following, command in found:
        after = [s for s in following if s.phase is not Phase.PAUSED]
        next_ = NEXT[after[0].phase] if after else "end"
        if next_ == RESET:
            next_ = _ending(after[0], segments, events, end)

        # Rewinds right after: only pauses and rewinds between the interruption and them.
        window_end = next((s.start for s in following if s.phase not in (Phase.PAUSED, Phase.REWINDING)),
                          len(phases))
        undone = [(e["start_frame"], e["end_frame"]) for e in rewinds
                  if segment.stop <= e.get("frame", -1) <= window_end]
        rewound = (min(a for a, _ in undone), max(b for _, b in undone)) if undone else None
        if next_ == "rewind" and rewound:
            suspect, source = rewound, "rewind"
        else:
            suspect, source = _blame(phases, frames.waiting, segment, round(blame_s * fps)), "blame"
        key = (number, segment.stop)
        if key in labels.suspects:
            suspect, source = labels.suspects[key], "labels"
            if suspect:
                suspect = (min(suspect[0], len(phases)), min(suspect[1], len(phases)))

        # The reason the operator gave before the policy drove again.
        resumed = next((s.start for s in following if s.phase is Phase.AUTONOMOUS), len(phases))
        said = [e for e in reasons if segment.stop <= e["frame"] <= resumed]
        reason = said[-1].get("reason") if said else None
        interruptions.append(Interruption(segment, command, reason, next_, rewound, suspect, source))

    # Failure-fix pairs: rewinds in a row that undid policy frames, then Teleop with only pauses and
    # the handover between. Rewinds in a row undo adjacent ranges, each ending where the next began.
    pairs = []
    if episode is not None:
        for k, e in enumerate(rewinds):
            failure = (e["start_frame"], e["end_frame"])
            for earlier in reversed(rewinds[:k]):
                if earlier["start_frame"] != failure[1]:
                    break
                failure = (failure[0], earlier["end_frame"])
            if not (phases[failure[0] : failure[1]] == Phase.AUTONOMOUS).any():
                continue
            j = starts.get(e.get("frame", -1))
            while j is not None and j < len(segments) and segments[j].phase in (Phase.PAUSED, Phase.HANDOVER):
                j += 1
            if j is not None and j < len(segments) and segments[j].phase is Phase.TELEOP:
                pairs.append(FailureFix(number, episode, failure, segments[j]))

    return Attempt(
        number=number, episode=episode, ended_by=end.get("ended_by") if end else None,
        recorded_outcome=recorded, outcome=outcome, phases=phases, waiting=frames.waiting,
        leaderless=frames.leaderless, segments=segments,
        interruptions=tuple(interruptions), rewinds=tuple((e["start_frame"], e["end_frame"]) for e in rewinds),
        failure_fixes=tuple(pairs), dropped_frames=dict((end or {}).get("dropped_frames") or {}),
    )


def _ending(returning: Segment, segments: tuple[Segment, ...], events: list[dict], end: dict | None) -> str:
    """How the attempt was asked to end when it went to Returning: reset, quit or time_limit, from
    the `state` event into Returning. Without one, the attempt's `attempt_end` says, if that
    Returning ran to the episode's end; otherwise reset."""
    into = next((e for e in events if e["type"] == "state" and e.get("state") == "returning"
                 and e.get("frame") == returning.start and e.get("ended_by")), None)
    if into:
        return into["ended_by"]
    if returning is segments[-1] and end and end.get("ended_by"):
        return end["ended_by"]
    return RESET


def _blame(phases: np.ndarray, waiting: np.ndarray, segment: Segment, frames: int) -> Range | None:
    """The last `frames` policy frames up to the end of `segment`, counting back through pauses and
    `waiting` frames only."""
    if frames <= 0:
        return None
    start, counted = segment.stop, 0
    while start > 0 and counted < frames:
        phase = phases[start - 1]
        if phase == Phase.AUTONOMOUS and not waiting[start - 1]:
            counted += 1
        elif phase != Phase.AUTONOMOUS and phase != Phase.PAUSED:
            break
        start -= 1
    while start < segment.stop and (phases[start] != Phase.AUTONOMOUS or waiting[start]):
        start += 1  # don't begin on a pause or a wait
    return (start, segment.stop) if counted else None


# --- Training cuts --------------------------------------------------------------------------------


def _pieces(mask: np.ndarray, within: Range) -> list[Range]:
    """The runs of True in `mask[within]`."""
    a, b = within
    sub = mask[a:b].astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], sub, [0]])))
    return [(a + int(s), a + int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def training_cuts(analysis: RunAnalysis, sources: set[str] | tuple[str, ...]) -> list[Cut]:
    """The frames of a run a build may train on, from `sources` (any of `SOURCES`).

    - corrections: each Teleop segment after the policy drove in its attempt (the same rule as
      `intervention_mask`) is its own cut, whatever the attempt's outcome;
    - human-attempts: each Teleop segment before the policy first drove, all of them in an attempt
      the human drove alone, when the attempt succeeded. Otherwise that Teleop is in no cut;
    - policy-successes: the Autonomous segments of attempts with outcome succeeded.

    Paused, handover, starting, rewinding and returning frames, suspect and waiting frames, Teleop
    frames without a leader pose, ignored attempts and attempts that dropped video frames are never
    in a cut. Where suspect frames fall inside a segment, its frames on either side are
    separate cuts, since a training episode can't skip frames. Each segment has at most one source,
    so a frame is in at most one cut.
    """
    cuts = []
    for attempt in analysis.attempts:
        if attempt.episode is None or attempt.ignored or attempt.dropped_frames:
            continue
        usable = ~attempt.suspect_mask() & ~attempt.waiting & ~attempt.leaderless
        succeeded = attempt.outcome == "succeeded"
        for segment in attempt.segments:
            after = segment.start >= attempt.policy_start
            if segment.phase is Phase.TELEOP and after and CORRECTIONS in sources:
                source = CORRECTIONS
            elif segment.phase is Phase.TELEOP and not after and succeeded and HUMAN_ATTEMPTS in sources:
                source = HUMAN_ATTEMPTS
            elif segment.phase is Phase.AUTONOMOUS and succeeded and POLICY_SUCCESSES in sources:
                source = POLICY_SUCCESSES
            else:
                continue
            for start, stop in _pieces(usable, segment.range):
                cuts.append(Cut(attempt.number, attempt.episode, start, stop, source))
    return cuts


# --- Failure-fix pairs ----------------------------------------------------------------------------


def export_failure_fixes(analyses: list[RunAnalysis], path: Path) -> int:
    """Write every failure-fix pair of `analyses` to `path`, one JSON object per line, and return how
    many. Each has the run, attempt, episode, the dataset it is in, and for the failure and the fix,
    their frames `[start, stop)` with `observation.state` and `action` per frame. The file is written
    even with no pairs, empty, so pairs of attempts since ignored aren't left in it. Reads the runs
    only, skipping the data files `analyse` skipped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as out:
        for analysis in analyses:
            if not analysis.failure_fixes:
                continue
            root = analysis.folder.dataset_root
            table, _ = read_data(root, ["episode_index", "frame_index", "observation.state", "action"])
            if table is None:
                continue
            for pair in analysis.failure_fixes:
                rows = table[table["episode_index"] == pair.episode].sort_values("frame_index")
                state = np.stack(rows["observation.state"].to_numpy()).round(3)
                action = np.stack(rows["action"].to_numpy()).round(3)

                def part(start: int, stop: int) -> dict:
                    return {"frames": [start, stop], "seconds": round((stop - start) / analysis.fps, 2),
                            "observation.state": state[start:stop].tolist(), "action": action[start:stop].tolist()}

                record = {"run": analysis.run, "attempt": pair.attempt, "episode": pair.episode,
                          "dataset": str(root), "fps": analysis.fps,
                          "failure": part(*pair.failure), "fix": part(*pair.fix.range)}
                out.write(json.dumps(record) + "\n")
                count += 1
    tmp.replace(path)
    return count
