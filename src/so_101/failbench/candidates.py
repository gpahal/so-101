"""Candidate samples from `so101 serve` and `so101 run` runs, for labelling into a labels file.

The grasp detector (`grasp.detector`) replays each saved attempt as the perception saw it live, through
`grasp.replay` (`live_masks`, `LiveFeed`): only the Autonomous frames the model drove or waited on
(`control.driver` POLICY or WAITING), on the model's own clock, with `resumed` called where the model drives
again after anything else held or moved the arm (a pause, a person's Teleop or Rewind, a layer's playback
or hold, or a Pause and Resume taken in one tick, `analysis.Attempt.handbacks`). Each moment the
supervisor would have checked becomes a candidate row: `after_close` and `after_release` at the program's
delays, `stuck` where a rule fired, `approach` halfway from the policy's first frame to its first close
(3 s in without one), and `end` at the episode's last frame.

The after_close and after_release checks follow the supervisor's own rules (`supervisor.schedule.Schedule`):
each is due at its delay on the model's clock; one not yet sent is dropped when the controller took the arm
first (any frame out of Autonomous, or a Pause and Resume in one tick; the layer's own playback or hold drops
nothing), and an after_close one when a newer close came first. A check the supervisor sent and then
dropped as stale (too old, or a carry or release since) stays a candidate: its frames were judged.

Rows have the columns `dataset.read_labels` reads, with `label` and `category` empty. A row's id names its
run, episode, frame and moment, since the moment is part of the question its label answers: two moments at
one frame are two rows. `write` merges into the file it writes: a row whose id is already there keeps its
label, category and note, whatever they hold, and a row the rules don't pick stays when its label or
category is filled in, so running it again after more runs never loses a person's work. `src` is the
run's folder name under the runs folder.
"""

import csv
from pathlib import Path

import numpy as np

from so_101.settings import replace_file

AUTONOMOUS = 1  # control.phase, as `control.recorder.Phase`
COLUMNS = ("id", "src", "ep", "t", "moment", "label", "category", "gripper", "note")
KEPT = ("label", "category", "note")  # what a person fills in, kept by id when the file is written again


def _worked(row: dict) -> bool:
    """A person filled in the row's label or category. The note alone doesn't count: the rules write one."""
    return bool((row.get("label") or "").strip() or (row.get("category") or "").strip())
HEADER = """# Candidate samples from so101 serve and run runs, picked by the grasp detector (so101-failbench candidates).
# Fill in label (ok or wrong) and category (a key of vision/prompt.py's CATEGORIES, none for ok) from the
# frames, and say in note how each label was reached. Leave label empty where you can't decide: build
# leaves those rows out. Running candidates again keeps every label, category and note, by id (run, episode,
# frame and moment).
"""


def pick(run: str, episode: int, phases: np.ndarray, states: np.ndarray, names: list[str],
         fps: float, program, thresholds, drivers: np.ndarray,
         handbacks: np.ndarray | None = None) -> list[dict]:
    """The candidate rows of one attempt. `phases`, `states` (frames x joints, named `names`) and `drivers`
    (`control.driver`) are the episode's columns in frame order; `handbacks` marks the frames a Pause and Resume in one tick came before
    (`analysis.Attempt.handbacks`). `program` gives the supervisor's check delays, `thresholds` the detector's."""
    from so_101.grasp.replay import LiveFeed, live_masks
    from so_101.supervisor.schedule import Schedule

    feed = LiveFeed(thresholds, fps)
    fed, drove, moved = live_masks(phases, drivers)
    gripper = names.index("gripper.pos")
    frames, rows = [], []  # (the model's seconds, frame) of each frame it drove
    picked: list[tuple[int, str, str]] = []  # (frame, moment, note)
    schedule = Schedule(program.after_close_s, program.after_release_s)
    first_close = None
    # Since the frame fed last, whether the controller had the arm (any frame out of Autonomous, or a Pause and
    # Resume in one tick), which drops the checks not yet sent, as `Supervisor.pause` and `run` do.
    paused = False
    for i, phase in enumerate(phases):
        if handbacks is not None and handbacks[i]:
            feed.handback(False)
            paused = True
        if not fed[i]:
            feed.handback(bool(moved[i]))
            paused = paused or phase != AUTONOMOUS
            continue
        if paused:
            schedule.drop()
        paused = False
        pose = dict(zip(names, map(float, states[i]), strict=True))
        # This tick's signals, then the checks due, in the order `Supervisor._watch_model` takes them.
        _, signals = feed.feed(pose, bool(drove[i]))
        t = feed.t
        frames.append((t, i))
        for s in signals:
            schedule.signal(s.kind, s.t)
            if s.kind == "close" and first_close is None:
                first_close = (t, i)
            elif s.kind == "rule":
                picked.append((i, "stuck", f"rule {s.name}"))
        picked += [(i, moment, f"close {close}" if moment == "after_close" else "release")
                   for moment, close, stale in schedule.due(t) if not stale]
    if not frames:
        return []  # what's still scheduled was never sent: the policy stopped driving before then
    start_t = frames[0][0]
    approach_t = (start_t + first_close[0]) / 2 if first_close else start_t + 3.0
    times = np.array([f[0] for f in frames])
    if (k := int(np.searchsorted(times, approach_t - 1e-9))) < len(frames):
        picked.append((frames[k][1], "approach", "halfway to the first close" if first_close else "3 s in, no close"))
    picked.append((len(phases) - 1, "end", "last frame"))
    seen = set()  # two picks of one moment at one frame are one question
    for frame, moment, note in picked:
        key = f"{run}-e{episode}-t{frame}-{moment}"  # the moment is part of the question a label answers
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": key, "src": run, "ep": episode, "t": frame, "moment": moment, "label": "", "category": "",
                     "gripper": round(float(states[frame][gripper]), 1), "note": note})
    return rows


def from_runs(runs: Path, program, thresholds) -> list[dict]:
    """Every saved attempt's candidates, from each run folder under `runs`."""
    from so_101.control import analysis
    from so_101.control.analysis import read_data
    from so_101.data import Run

    rows = []
    for folder in sorted(p for p in Path(runs).iterdir() if (p / "dataset").is_dir()):
        run = Run.at(folder)
        if (info := run.info()) is None:
            continue
        names = list(info["features"]["observation.state"]["names"])
        table, _ = read_data(run.dataset_root, ["episode_index", "frame_index", "observation.state", "control.phase",
                                                "control.driver"])
        if table is None:
            continue
        # Each saved episode's same-tick handbacks, from the run's events. Blame plays no part in them.
        record = analysis.read_run(run)
        handbacks = {a.episode: a.handbacks for a in analysis.analyse_record(record, analysis.load_labels(run), 0.0).attempts
                     if a.episode is not None}
        for episode, ep in table.groupby("episode_index"):
            ep = ep.sort_values("frame_index")
            phases = np.array([int(np.ravel(p)[0]) for p in ep["control.phase"]])
            states = np.stack([np.asarray(v, dtype=float) for v in ep["observation.state"]])
            drivers = np.array([int(np.ravel(d)[0]) for d in ep["control.driver"]])
            marks = handbacks.get(int(episode))
            rows += pick(folder.name, int(episode), phases, states, names, float(info["fps"]), program,
                         thresholds, drivers=drivers,
                         handbacks=marks if marks is not None and len(marks) == len(phases) else None)
    return rows


def _existing(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def unpicked(rows: list[dict], path: Path) -> int:
    """How many rows in the file at `path` with a label or category filled in aren't among `rows`: picked
    under other rules, or from a run that's gone. `write` keeps them; a person decides whether they count."""
    picked = {row["id"] for row in rows}
    return sum(1 for row in _existing(path) if _worked(row) and row["id"] not in picked)


def write(rows: list[dict], path: Path) -> int:
    """Write `rows` to `path`, merged by id with what's there: an existing row keeps its label, category
    and note, whatever they hold, and a row not in `rows` with a label or category stays, after them. Returns
    how many rows with a label or category were kept."""
    import io

    old = {row["id"]: row for row in _existing(path)}
    merged, kept = [], 0
    for row in rows:
        before = old.pop(row["id"], None)
        if before is not None:
            row = {**row, **{k: before.get(k) or "" for k in KEPT}}
            kept += _worked(before)
        merged.append(row)
    for row in old.values():
        if _worked(row):
            merged.append({k: row.get(k) or "" for k in COLUMNS})
            kept += 1
    out = io.StringIO()
    out.write(HEADER)
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, out.getvalue())
    return kept
