"""`so101 build`: turn recorded sessions plus edits.toml into a training dataset.

Two kinds (`data.DATASETS`): `whole`, each kept demo as one episode, and `skills` (`--skills`), each kept
demo's skills as episodes in order, cut where `so_101.skills` finds their events, each frame with its
skill's prompt as `task` and its one-hot as `observation.environment_state`. A demo the skills can't be
cut from is left out, with a note. Either takes runs with --with; in a skills build each run cut is split
where the skill changes, each frame labelled by what happened (`run_skills`, `skills.label_run`): its
joint readings replayed through the detector as a demo's are, the skill a skills model was told on the
frames it drove, and each skill's `to` bound.

Raw sessions are never changed. Each kept episode is read back (joint rows and decoded video
frames), cut to its `keep` range, and written into a fresh dataset with LeRobot's own writer, so
frame indices, timestamps, video files and normalization stats are all consistent. Decoding and
re-encoding loses a little image quality, so the build encodes at a higher quality (lower CRF)
than recording does.

`--with` adds frames of `so101 serve` and `so101 run` runs, from `control.analysis.training_cuts`:
each cut is its own episode, after the demos. The cuts already leave out scripted, suspect and waiting
frames and ignored attempts (`analysis.training_cuts` has the full list). A cut whose episode the
run's episode index doesn't list, as when a kill came after its data file was finished and before its
index row was written, is left out with a note.
Only the demos' columns are written, unless `keep_labels` is set. A cut's `action` is what a demo's
is: what the arm was asked for, before the follower's `max_relative_target` cap (lerobot-record saves
the leader's pose, not the capped goal). So a Teleop cut takes the run's `control.leader`, and a
policy-success cut its `control.policy_action`, rather than the run's `action`, the capped goal
(`ACTION_COLUMN`).
With either, build.json records `analysis.RULES_VERSION`, so a build made under other cut or label
rules isn't current. A skills build also records `skills.contract` (`data.build_inputs`): the skills,
the `[gripper]` thresholds' hash, `skills.CUT_RULES_VERSION` and, when it takes runs,
`skills.RUN_LABEL_VERSION` (how runs were labelled, the grasp detector included), which a skills model
is checked against before it runs (`commands.policy.require_skills_contract`).

build.json sits inside the dataset and names no folder of this computer (`path_free`), since `train`
uploads it with the dataset. The new build replaces the old by `swap`.
"""

import contextlib
import json
import shutil
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from so_101 import data
from so_101.control import analysis
from so_101.control.recorder import Phase
from so_101.dataset_io import column, episode_frames, episode_table
from so_101.report import Report
from so_101.settings import ConfigError, Settings, replace_file
from so_101.tasks import Task
from so_101.validate import validate_dataset

BUILD_CRF = 23
# SVT-AV1 has no preset above 10 here: LeRobot's default of 12 runs as 10, with a warning per episode.
BUILD_PRESET = 10
MIN_EPISODE_S = 0.5
DEMOS = "demos"
# Where each run cut's `action` comes from: what was asked for, before the cap, as in a demo.
ACTION_COLUMN = {
    analysis.CORRECTIONS: "control.leader",
    analysis.HUMAN_ATTEMPTS: "control.leader",
    analysis.POLICY_SUCCESSES: "control.policy_action",
}

# Columns `--keep-labels` adds: control.phase is the run's own, and the rest come from the analysis.
LABEL_FEATURES = {
    "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    "control.phase": {"dtype": "int8", "shape": (1,), "names": None},
    "control.label": {"dtype": "int8", "shape": (1,), "names": None},  # 1 intervention, 0 neutral, -1 suspect
    "control.attempt_success": {"dtype": "bool", "shape": (1,), "names": None},
}


@dataclass(frozen=True)
class Plan:
    """Frames start to stop of one episode: a demo of a session, or a run's training cut, with the
    attempt it's from, whose frames label it, and the cut's source."""

    recording: data.Session | data.Run
    episode: int
    start: int  # first frame kept
    stop: int  # one past the last frame kept
    attempt: analysis.Attempt | None = None
    source: str | None = None
    skill: int | None = None  # in a skills build: the skill's index in `task.skills`

    @property
    def meta(self):
        return self.recording.meta

    @cached_property
    def frame_labels(self) -> np.ndarray:
        return self.attempt.frame_labels()

    @cached_property
    def intervention_mask(self) -> np.ndarray:
        return self.attempt.intervention_mask()

    def labels(self, i: int) -> dict:
        if self.attempt is None:
            # A demo is a human driving with the leader, as in Teleop, but no policy ran: it corrects
            # nothing, so it isn't an intervention.
            return _labels(False, Phase.TELEOP, analysis.FrameLabel.NEUTRAL, True)
        # From the analysis's rule, so the label follows `analysis`, whatever the recorded column says.
        return _labels(bool(self.intervention_mask[i]), int(self.attempt.phases[i]),
                       int(self.frame_labels[i]), self.attempt.outcome == "succeeded")

    def entry(self, task: Task | None = None) -> dict:
        skill = {} if self.skill is None else {"skill": task.skills[self.skill].name}
        if self.attempt is None:
            return {"session": self.recording.name, "episode": self.episode, "frames": [self.start, self.stop], **skill}
        return {"run": self.recording.name, "attempt": self.attempt.number, "episode": self.episode,
                "frames": [self.start, self.stop], "source": self.source, **skill}

    def damaged(self, task: Task, e: Exception) -> str:
        name = self.recording.name
        if self.attempt is None:
            return (f"Session {name} episode {self.episode} is damaged ({e!r}). Delete it with "
                    f"`uv run so101 review {task.name} --episode {name}/{self.episode}` and build again.")
        return (f"Run {name} attempt {self.attempt.number} (episode {self.episode}) is damaged ({e!r}). "
                f"Ignore the attempt in {self.recording.labels_path} (outcome = \"ignored\") and build again.")


def _labels(intervention: bool, phase: int, label: int, success: bool) -> dict:
    return {
        "intervention": np.array([intervention], dtype=bool),
        "control.phase": np.array([phase], dtype=np.int8),
        "control.label": np.array([label], dtype=np.int8),
        "control.attempt_success": np.array([success], dtype=bool),
    }


def parse_sources(text: str | None) -> tuple[str, ...]:
    """`--with corrections,human-attempts`: the run sources besides the demos, in a fixed order."""
    names = {name.strip() for name in (text or "").split(",") if name.strip()}
    if unknown := names - {DEMOS, *analysis.SOURCES}:
        raise ConfigError(f"Unknown training sources: {', '.join(sorted(unknown))}. "
                          f"Use any of {', '.join(analysis.SOURCES)}; demos are always in.")
    return tuple(name for name in analysis.SOURCES if name in names)


def demo_columns(features: dict) -> dict:
    """The columns a demo has and a training episode needs: joints and cameras, not LeRobot's own."""
    from lerobot.utils.constants import DEFAULT_FEATURES

    return {
        key: {"dtype": f["dtype"], "shape": tuple(f["shape"]), "names": f.get("names")}
        for key, f in features.items()
        if key not in DEFAULT_FEATURES
    }


def plan(s: Settings, task: Task) -> tuple[list[Plan], list[str]]:
    """Which frames of which episodes go into the build, and notes on what was left out."""
    edits = data.load_edits(s, task)
    plans, notes, recorded = [], [], set()
    reference = None
    for session in data.sessions(s, task):
        meta = session.meta
        if meta is None:
            why = session.meta_error or "empty or interrupted"
            notes.append(f"session {session.name}: no usable dataset ({why}), skipped")
            continue
        shape = (meta.fps, meta.robot_type, {k: (f["dtype"], tuple(f["shape"])) for k, f in meta.features.items()})
        if reference is None:
            reference = (session.name, shape)
        elif shape != reference[1]:
            raise ConfigError(
                f"Session {session.name} was recorded with different settings (fps, cameras or joints) from "
                f"session {reference[0]}, so they can't go into one dataset. Move one of them out of "
                f"{session.path.parent}."
            )
        for ep in range(meta.total_episodes):
            recorded.add((session.name, ep))
            length = int(meta.episodes[ep]["length"])
            edit = edits.get((session.name, ep))
            if edit and edit.delete is not None:
                notes.append(f"session {session.name} episode {ep}: deleted ({edit.delete or 'no reason given'})")
                continue
            start, stop = 0, length
            if edit and edit.keep:
                start = min(length, round(edit.keep[0] * meta.fps))
                stop = min(length, round(edit.keep[1] * meta.fps))
            if stop - start < MIN_EPISODE_S * meta.fps:
                raise ConfigError(
                    f"Session {session.name} episode {ep}: keeping frames {start}-{stop} of {length} leaves "
                    f"less than {MIN_EPISODE_S} s. Fix it in {data.edits_path(s, task)}."
                )
            plans.append(Plan(session, ep, start, stop))
    unknown = sorted(f"{session}/{episode}" for session, episode in set(edits) - recorded)
    if unknown:
        notes.append(f"edits.toml mentions episodes that don't exist: {', '.join(unknown)}")
    return plans, notes


def plan_runs(s: Settings, task: Task, sources: tuple[str, ...], reference, skills: bool = False) -> tuple[list[Plan], list[str]]:
    """The training cuts of every run from `sources`, each its own episode. `reference` is the
    demos' metadata: a run must have the same fps, robot and demo columns to join them. With `skills`,
    each cut is split where the skill changes (`run_skills`), each piece an episode of its skill."""
    plans, notes = [], []
    if not sources:
        return plans, notes
    grasp = None
    if skills:
        from so_101.grasp import thresholds

        grasp = thresholds.load(s.perception.thresholds)
    want = demo_columns(reference.features)
    for run in data.runs(s, task):
        result = analysis.analyse(run, s.control.blame_s)
        notes += [f"run {run.name}: {note}" for note in result.notes]
        cuts = analysis.training_cuts(result, sources)
        if not cuts:
            continue
        meta = run.meta
        if meta is None:
            notes.append(f"run {run.name}: no usable dataset, skipped")
            continue
        have = {k: (f["dtype"], tuple(f["shape"])) for k, f in meta.features.items()}
        differ = [
            f"{key} is missing" if key not in have else f"{key} is {have[key][0]} {list(have[key][1])}, "
            f"the demos' is {f['dtype']} {list(f['shape'])}"
            for key, f in want.items()
            if have.get(key) != (f["dtype"], f["shape"])
        ]
        if meta.fps != reference.fps:
            differ.insert(0, f"it ran at {meta.fps} fps, the demos at {reference.fps}")
        if meta.robot_type != reference.robot_type:
            differ.insert(0, f"its robot is {meta.robot_type}, the demos' {reference.robot_type}")
        if differ:
            raise ConfigError(
                f"Run {run.name} was recorded with different settings from the demos, so it can't go into one "
                f"dataset: {'; '.join(differ)}. Set every attempt of it to outcome = \"ignored\" in "
                f"{run.labels_path}, or move the run out of {run.path.parent}."
            )
        attempts = {a.number: a for a in result.attempts}
        indexed = {int(e) for e in meta.episodes["episode_index"]}
        labelled: dict[int, tuple[np.ndarray | None, str]] = {}  # each attempt's skill per frame, once
        for cut in cuts:
            if cut.episode not in indexed:
                notes.append(f"run {run.name} attempt {cut.attempt}: episode {cut.episode} isn't in the dataset's "
                             "episode index, left out")
                continue
            pieces = [(cut.start, cut.stop, None)]
            if skills:
                if cut.attempt not in labelled:
                    labelled[cut.attempt] = run_skills(run, attempts[cut.attempt], task, grasp)
                per_frame, why = labelled[cut.attempt]
                if per_frame is None:
                    notes.append(f"run {run.name} attempt {cut.attempt}: {why}, so its {cut.source} frames are left out")
                    continue
                pieces = split_by_skill(per_frame, cut.start, cut.stop)
                outside = (per_frame[cut.start:cut.stop] < 0).sum()
                if outside:
                    notes.append(f"run {run.name} attempt {cut.attempt}: {outside} {cut.source} frames are past the "
                                 "end of every skill, left out")
            for start, stop, skill in pieces:
                if stop - start < MIN_EPISODE_S * meta.fps:
                    which = "" if skill is None else f" ({task.skills[skill].name})"
                    notes.append(f"run {run.name} attempt {cut.attempt}: {cut.source} frames {start}-{stop}{which} "
                                 f"are under {MIN_EPISODE_S} s, left out")
                    continue
                plans.append(Plan(run, cut.episode, start, stop, attempts[cut.attempt], cut.source, skill=skill))
    return plans, notes


def run_skills(run: data.Run, attempt: analysis.Attempt, task: Task, grasp) -> tuple[np.ndarray | None, str]:
    """Each frame's skill, as an index in `task.skills` or -1 for none, for one attempt of a run, or None
    and why not: what happened, by `skills.label_run`, from the attempt's joint readings and the grasp
    thresholds `grasp`.

    Frames a script moved (the move to the start pose, a Rewind, the return to rest, a layer's playback)
    aren't replayed, as the live detector sees none of them. Of the rest, the replay feeds (`fed`) Teleop
    frames, on purpose, since a person's grasp and release count (a re-grip is rolled back), and the frames
    the live detector sees: Autonomous frames the model drove or waited on (`control.driver` POLICY or
    WAITING). Paused and Handover frames and a layer's hold aren't, and count on no clock, and only frames
    the model drove, or Teleop frames, count as driven for the stuck rules (`drove`). The masks are
    `grasp.replay.live_masks`'s, and a Pause and Resume taken in one tick (`attempt.handbacks`) reach the
    replay as the live detector heard them.
    In a skills run, Autonomous frames take the skill the model was told (control.skill,
    indexed in its own run.json's policy.skills and mapped onto the task's by name; of those, the ones the
    model didn't drive, waiting or a layer's hold, are in no cut anyway, `control.analysis`); every other frame,
    such as a correction in Teleop, takes the skill the replay finds, so a person who grasps during pick
    is in place from the carry on, whatever the model was told, and a person who opens the jaws on a
    carried block and closes them again is still in place."""
    from so_101 import skills
    from so_101.grasp.replay import live_masks

    names = list(run.meta.features["observation.state"]["names"])
    table = episode_table(run.meta, attempt.episode)
    state = column(table, "observation.state")
    phases = attempt.phases
    if len(state) != len(phases):
        return None, f"its {len(state)} frames of joint readings don't match its {len(phases)} phases"
    driver = np.array([int(np.ravel(v)[0]) for v in table["control.driver"]])
    # What the live detector saw (`grasp.replay.live_masks`), plus Teleop frames, fed on purpose.
    live, model_drove, moved = live_masks(phases, driver)
    teleop = phases == Phase.TELEOP
    scripted = np.isin(phases, (Phase.STARTING, Phase.REWINDING, Phase.RETURNING)) | (moved & (phases == Phase.AUTONOMOUS))
    fed, drove = teleop | live, teleop | model_drove
    told = None
    recorded = (run.manifest().get("policy") or {}).get("skills")
    if recorded and (attempt.skills >= 0).any():
        index = {n: [s.name for s in task.skills].index(r["name"]) for n, r in enumerate(recorded)
                 if r.get("name") in [s.name for s in task.skills]}
        if len(index) != len(recorded):
            return None, f"its skills ({', '.join(r.get('name', '?') for r in recorded)}) aren't all the task's"
        told = np.array([index.get(int(k), -1) if p == Phase.AUTONOMOUS else -1
                         for k, p in zip(attempt.skills, phases, strict=True)], dtype=np.int8)
    return skills.label_run(task.skills, state, names, run.meta.fps, grasp, scripted, told,
                            teleop=teleop, fed=fed, drove=drove, handbacks=attempt.handbacks), ""


def split_by_skill(per_frame: np.ndarray, start: int, stop: int) -> list[tuple[int, int, int]]:
    """Frames [start, stop) as runs of one skill: (start, stop, skill) each. Frames in no skill (-1) are
    left out."""
    pieces, a = [], start
    for i in range(start + 1, stop + 1):
        if i == stop or per_frame[i] != per_frame[a]:
            if per_frame[a] >= 0:
                pieces.append((a, i, int(per_frame[a])))
            a = i
    return pieces


def skill_plans(s: Settings, task: Task) -> tuple[list[Plan], list[str], list[dict]]:
    """Each kept demo's skills as plans, in order; the notes, with a note per demo that couldn't be
    cut; and every demo's cuts for build.json."""
    from so_101 import skills
    from so_101.grasp import thresholds

    cuts, notes = skills.cut_demos(s, task, thresholds.load(s.perception.thresholds))
    plans = []
    for c in cuts:
        p = c.plan
        if c.error:
            notes.append(f"session {p.recording.name} episode {p.episode}: left out, its skills can't be cut ({c.error})")
            continue
        for n, skill in enumerate(task.skills):
            start, stop = c.spans[skill.name]
            plans.append(Plan(p.recording, p.episode, p.start + start, p.start + stop, skill=n))
    return plans, notes, [c.record() for c in cuts]


def build(s: Settings, task: Task, sources: tuple[str, ...] = (), keep_labels: bool = False,
          kind: str = data.WHOLE) -> Report:
    """Build the `kind` dataset (data.dataset_root) from the sessions and edits, plus, for `whole`, the
    runs' training cuts from `sources`. `keep_labels` adds the LABEL_FEATURES columns. Returns the
    validation report."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    from so_101 import skills

    if kind == data.SKILLS:
        task.require_skills()
        if keep_labels:
            raise ConfigError("A skills build takes no --keep-labels yet.")
    # Taken before reading anything, so an edit or a run recorded during the build rebuilds next time.
    inputs = data.build_inputs(s, task, sources, keep_labels, kind)
    cuts = None
    if kind == data.SKILLS:
        plans, notes, cuts = skill_plans(s, task)
    else:
        plans, notes = plan(s, task)
    if not plans:
        for note in notes:
            print(f"  {note}")
        if cuts:
            raise ConfigError(
                f"Nothing to build for {task.name}: none of its {len(cuts)} kept demos could be cut into skills, for the "
                f"reasons above. `uv run so101 skills {task.name}` shows each cut; check [{task.name}.skills] in tasks.toml.")
        raise ConfigError(f"Nothing to build for {task.name}: record some demos first (`uv run so101 record {task.name}`).")
    source = plans[0].meta
    run_plans, run_notes = plan_runs(s, task, sources, source, skills=kind == data.SKILLS)
    notes += run_notes
    plans = [*plans, *run_plans]
    for note in notes:
        print(f"  {note}")

    features = demo_columns(source.features)
    if keep_labels:
        features |= LABEL_FEATURES
    if kind == data.SKILLS:
        features[skills.STATE_KEY] = skills.state_feature(task.skills)
    repo_id = f"local/so101_{task.name}{'' if kind == data.WHOLE else '_' + kind}"
    root = data.dataset_root(s, task, kind)
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = root.with_name(f"{root.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    out = LeRobotDataset.create(
        repo_id=repo_id,
        fps=source.fps,
        features=features,
        root=tmp,
        robot_type=source.robot_type,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(crf=BUILD_CRF, preset=BUILD_PRESET),
        # Streaming drops frames when its encoder falls behind, which suits a live recording but not
        # this: nothing is waiting on a build, so write every frame and encode each episode after.
        streaming_encoding=False,
        image_writer_threads=8,
    )

    video_keys = source.video_keys
    started = time.perf_counter()
    total = sum(p.stop - p.start for p in plans)
    done = 0
    mapping = []
    try:
        for n, p in enumerate(plans):
            meta = p.meta
            table = episode_table(meta, p.episode)
            state, action = column(table, "observation.state"), column(table, ACTION_COLUMN.get(p.source, "action"))
            streams = [episode_frames(meta, p.episode, key, p.start, p.stop) for key in video_keys]
            try:
                prompt = task.prompt if p.skill is None else task.skills[p.skill].prompt
                extra = {} if p.skill is None else {skills.STATE_KEY: skills.one_hot(task.skills, p.skill)}
                for i in range(p.start, p.stop):
                    frame = {"observation.state": state[i], "action": action[i], "task": prompt, **extra}
                    for key, stream in zip(video_keys, streams, strict=True):
                        index, image = next(stream)
                        assert index == i, f"{key}: got frame {index}, expected {i}"
                        frame[key] = image
                    if keep_labels:
                        frame |= p.labels(i)
                    out.add_frame(frame)
            except (ValueError, IndexError, StopIteration, AssertionError) as e:
                raise ConfigError(p.damaged(task, e)) from e
            # Serial: parallel encoding starts a process pool per episode, each worker a fresh Python importing
            # torch and LeRobot. A 300-frame, two-camera episode takes 4.7-4.9 s serial against 5.7-6.1 s.
            out.save_episode(parallel_encoding=False)
            done += p.stop - p.start
            mapping.append(p.entry(task))
            elapsed = time.perf_counter() - started
            print(
                f"\r  episode {n + 1}/{len(plans)}: {done}/{total} frames, {done / max(elapsed, 1e-6):.0f} frames/s ",
                end="",
                flush=True,
            )
        print()
        out.finalize()
    except BaseException:
        with contextlib.suppress(Exception):
            out.finalize()  # stop the video encoder threads before deleting their output
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    report = Report()
    # Tracking only warns here: the demos are already recorded, and the dry run is where it should block.
    validate_dataset(tmp, repo_id, report, group=f"{kind} dataset", strict_tracking=False)
    if report.failures:
        report.print("Validation of the new build")
        raise ConfigError(f"The build failed validation; it was left at {tmp} for inspection. The old build is unchanged.")
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": inputs,
        "episodes": mapping,
        "frames": total,
        "notes": [path_free(note, s) for note in notes],
        **({"cuts": cuts} if cuts is not None else {}),
    }
    # build.json goes inside the new dataset, so the rename swaps both at once: a kill leaves the old
    # build with its own record, or no build at all, never a dataset beside another build's record.
    replace_file(tmp / "build.json", json.dumps(manifest, indent=2) + "\n")
    swap(tmp, root)
    return report


def path_free(text: str, s: Settings) -> str:
    """`text` with this computer's data folder and home folder written as <data_dir> and ~: build.json is
    uploaded with the dataset, and a note can quote a path, as an error reading a session's metadata does."""
    for folder, name in ((s.data_dir, "<data_dir>"), (s.data_dir.resolve(), "<data_dir>"), (Path.home(), "~")):
        text = text.replace(str(folder), name)
    return text


def swap(new, root) -> None:
    """Put the dataset at `new` in place of the one at `root`: rename the old one aside, rename the new
    one in, then delete the old. The old build is deleted only once the new one is in place, so a
    failure leaves one whole build at `root`, and a kill between the renames leaves none there, never
    half of one. An old build set aside by a killed swap is deleted first."""
    old = root.with_name(f"{root.name}.old")
    if old.exists():
        shutil.rmtree(old)
    if root.exists():
        root.rename(old)
    try:
        new.rename(root)
    except BaseException:
        if old.exists() and not root.exists():
            old.rename(root)
        raise
    shutil.rmtree(old, ignore_errors=True)
