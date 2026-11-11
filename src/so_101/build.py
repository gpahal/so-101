"""`so101 build`: turn recorded sessions plus edits.toml into one training dataset.

Raw sessions are never changed. Each kept episode is read back (joint rows and decoded video
frames), cut to its `keep` range, and written into a fresh dataset with LeRobot's own writer, so
frame indices, timestamps, video files and normalization stats are all consistent. Decoding and
re-encoding loses a little image quality, so the build encodes at a higher quality (lower CRF)
than recording does.

`--with` adds frames of `so101 serve` runs, from `control.analysis.training_cuts`: each cut is its
own episode, after the demos. The cuts already leave out paused and scripted frames, suspect
frames and ignored attempts. A cut whose episode the run's episode index doesn't list, as when a kill
came after its data file was finished and before its index row was written, is left out with a note.
Only the demos' columns are written, unless `keep_labels` is set.
With either, build.json records `analysis.RULES_VERSION`, so a build made under older cut or label
rules isn't current.
"""

import contextlib
import json
import shutil
import time
from dataclasses import dataclass
from functools import cached_property

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
        # From the analysis's rule, not the recorded column, which older runs set in every Teleop frame.
        return _labels(bool(self.intervention_mask[i]), int(self.attempt.phases[i]),
                       int(self.frame_labels[i]), self.attempt.outcome == "succeeded")

    def entry(self) -> dict:
        if self.attempt is None:
            return {"session": self.recording.name, "episode": self.episode, "frames": [self.start, self.stop]}
        return {"run": self.recording.name, "attempt": self.attempt.number, "episode": self.episode,
                "frames": [self.start, self.stop], "source": self.source}

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


def plan_runs(s: Settings, task: Task, sources: tuple[str, ...], reference) -> tuple[list[Plan], list[str]]:
    """The training cuts of every run from `sources`, each its own episode. `reference` is the
    demos' metadata: a run must have the same fps, robot and demo columns to join them."""
    plans, notes = [], []
    if not sources:
        return plans, notes
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
        for cut in cuts:
            if cut.episode not in indexed:
                # A kill after the episode's data file was finished and before its index row was written.
                notes.append(f"run {run.name} attempt {cut.attempt}: episode {cut.episode} isn't in the dataset's "
                             "episode index, left out")
                continue
            if cut.frames < MIN_EPISODE_S * meta.fps:
                notes.append(f"run {run.name} attempt {cut.attempt}: {cut.source} frames {cut.start}-{cut.stop} "
                             f"are under {MIN_EPISODE_S} s, left out")
                continue
            plans.append(Plan(run, cut.episode, cut.start, cut.stop, attempts[cut.attempt], cut.source))
    return plans, notes


def build(s: Settings, task: Task, sources: tuple[str, ...] = (), keep_labels: bool = False) -> Report:
    """Build data/<task>/train from the sessions and edits, plus the runs' training cuts from
    `sources`. `keep_labels` adds the LABEL_FEATURES columns. Returns the validation report."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    # Taken before reading anything, so an edit or a run recorded during the build rebuilds next time.
    inputs = data.build_inputs(s, task, sources, keep_labels)
    plans, notes = plan(s, task)
    if not plans:
        for note in notes:
            print(f"  {note}")
        raise ConfigError(f"Nothing to build for {task.name}: record some demos first (`uv run so101 record {task.name}`).")
    source = plans[0].meta
    run_plans, run_notes = plan_runs(s, task, sources, source)
    notes += run_notes
    plans = [*plans, *run_plans]
    for note in notes:
        print(f"  {note}")

    features = demo_columns(source.features)
    if keep_labels:
        features |= LABEL_FEATURES
    root = data.train_root(s, task)
    tmp = root.with_name("train.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    out = LeRobotDataset.create(
        repo_id=f"local/so101_{task.name}",
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
            state, action = column(table, "observation.state"), column(table, "action")
            streams = [episode_frames(meta, p.episode, key, p.start, p.stop) for key in video_keys]
            try:
                for i in range(p.start, p.stop):
                    frame = {"observation.state": state[i], "action": action[i], "task": task.prompt}
                    for key, stream in zip(video_keys, streams, strict=True):
                        index, image = next(stream)
                        assert index == i, f"{key}: got frame {index}, expected {i}"
                        frame[key] = image
                    if keep_labels:
                        frame |= p.labels(i)
                    out.add_frame(frame)
            except (ValueError, IndexError, StopIteration, AssertionError) as e:
                raise ConfigError(p.damaged(task, e)) from e
            # Parallel encoding starts a process pool per episode, and on macOS each
            # worker is a fresh Python that imports torch and LeRobot. Serial encoding
            # took one 300-frame, two-camera episode from 5.7-6.1 s to 4.7-4.9 s, and
            # the build tests from 59.7 s to 7.6 s.
            out.save_episode(parallel_encoding=False)
            done += p.stop - p.start
            mapping.append(p.entry())
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
    validate_dataset(tmp, f"local/so101_{task.name}", report, group="train dataset", strict_tracking=False)
    if report.failures:
        report.print("Validation of the new build")
        raise ConfigError(f"The build failed validation; it was left at {tmp} for inspection. The old build is unchanged.")
    # build.json goes first: a kill during the swap then leaves no build that claims to be current.
    manifest_path = data.build_manifest_path(s, task)
    manifest_path.unlink(missing_ok=True)
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": inputs,
        "episodes": mapping,
        "frames": total,
        "notes": notes,
    }
    replace_file(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return report
