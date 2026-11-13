"""Run analysis and `so101 build --with`, through their public interfaces: read a recorded run back
into training cuts, build a task's training dataset from demo sessions plus the run, then read the
built dataset and build.json back.

The run is recorded through the controller harness in test_controller, so it has the events and
columns a real `so101 serve` run has. Expected frames come from the run's own files: the
`control.phase` column and the `rewind` events. The demo session is a small LeRobot dataset with
the same robot and cameras. Cameras show mid-grey, since the build's validation fails a mostly black
video.
"""

import contextlib
import dataclasses
import hashlib
import io
import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from fakes import FakeCamera, FakeClock, make_follower
from test_controller import FPS, GRIPPER_ACTIONS, REST_STEPS, Rig

from so_101 import data, settings
from so_101.build import build
from so_101.control import analysis
from so_101.control.recorder import Phase, features
from so_101.data import Run
from so_101.tasks import Task

TASK = Task(name="test", prompt="test", cameras=("top", "wrist"), fps=FPS,
            episodes=1, episode_time_s=10, reset_time_s=5, props=[], setup=[])
BLAME_S = 2.5


def phases(run_dir: Path) -> dict[int, np.ndarray]:
    """Each episode's control.phase column, read straight from the dataset."""
    files = sorted((run_dir / "dataset").glob("data/*/*.parquet"))
    frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "control.phase"]) for f in files)
    return {
        int(ep): np.array([int(np.ravel(p)[0]) for p in rows.sort_values("frame_index")["control.phase"]])
        for ep, rows in frames.groupby("episode_index")
    }


def all_nan(run_dir: Path, name: str) -> dict[int, np.ndarray]:
    """Per episode, whether each frame's `name` column is all NaN, read straight from the dataset."""
    files = sorted((run_dir / "dataset").glob("data/*/*.parquet"))
    frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", name]) for f in files)
    return {
        int(ep): np.array([bool(np.isnan(np.asarray(v, dtype=float)).all()) for v in rows.sort_values("frame_index")[name]])
        for ep, rows in frames.groupby("episode_index")
    }


def runs_of(column: np.ndarray, phase: Phase) -> list[tuple[int, int]]:
    """The [start, stop) ranges where `column` is `phase`."""
    out, i = [], 0
    for value, group in itertools.groupby(column):
        n = len(list(group))
        if value == phase:
            out.append((i, i + n))
        i += n
    return out


def digest(run_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != analysis.LABELS_FILE):
        h.update(str(f.relative_to(run_dir)).encode() + f.read_bytes())
    return h.hexdigest()


def grey(self, *args, **kwargs) -> np.ndarray:
    return np.full((self.height, self.width, 3), 128, dtype=np.uint8)


def make_demo_session(root: Path, frames: int) -> None:
    """One demo episode, with only the columns `lerobot-record` writes."""
    from lerobot.datasets import LeRobotDataset

    robot, _ = make_follower(FakeClock(), REST_STEPS)
    try:
        demo = {k: f for k, f in features(robot).items() if k.startswith(("observation.", "action"))}
        dataset = LeRobotDataset.create("local/demo", FPS, demo, root=root, robot_type=robot.robot_type,
                                        streaming_encoding=False)
    finally:
        robot.inner.disconnect()
    for i in range(frames):
        state = np.full(6, i, dtype=np.float32)
        image = np.full((48, 64, 3), 128, dtype=np.uint8)
        dataset.add_frame({"observation.state": state, "action": state, "task": "test",
                           "observation.images.top": image, "observation.images.wrist": image})
    dataset.save_episode()
    dataset.finalize()


def table(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("data/*/*.parquet"))
    return pd.concat(pd.read_parquet(f) for f in files).sort_values("index").reset_index(drop=True)


class BuildTest(unittest.TestCase):
    def setUp(self) -> None:
        for name in ("read_latest", "read", "async_read"):
            patch = mock.patch.object(FakeCamera, name, grey)
            patch.start()
            self.addCleanup(patch.stop)
        temp = tempfile.TemporaryDirectory(prefix="so101-build-")
        self.addCleanup(temp.cleanup)
        loaded = settings.load()
        self.s = dataclasses.replace(loaded, data_dir=Path(temp.name),
                                     control=dataclasses.replace(loaded.control, blame_s=BLAME_S))
        self.task_dir = data.task_dir(self.s, TASK)
        make_demo_session(self.task_dir / "sessions" / "20260101-100000" / "dataset", FPS)
        self.rig = Rig(self.task_dir / "runs" / "20260101-110000", actions=GRIPPER_ACTIONS)
        self.addCleanup(self.rig.close)

    def pause(self) -> None:
        self.rig.command("pause")
        self.rig.tick(5)

    def test_cuts_and_corrections_leave_out_paused_scripted_suspect_waiting_and_ignored_frames(self) -> None:
        # 1: the policy, paused and rewound, then a correction; succeeded.
        self.rig.policy(3)
        self.pause()
        self.rig.command("rewind")
        self.rig.tick_until("paused", limit=10 * FPS)
        self.rig.teleop(1)
        self.rig.reset("succeeded")
        # 2: a correction in an ignored attempt.
        self.rig.policy(2)
        self.rig.teleop(1)
        self.rig.reset("ignored")
        # 3: two corrections with a pause between them; failed.
        self.rig.policy(3)
        self.rig.teleop(1)
        self.pause()
        self.rig.teleop(1)
        self.rig.reset("failed")
        # 4: a human drives before the policy, then steps in on it; failed.
        self.rig.teleop(1)
        self.rig.policy(2)
        self.rig.teleop(1)
        self.rig.reset("failed")
        # 5: the policy, paused without a rewind and resumed; succeeded.
        self.rig.policy(3)
        self.pause()
        self.rig.command("resume")
        self.rig.tick(FPS)
        self.rig.reset("succeeded")
        # 6: the policy, whose next chunk (from the cloud GPU) is late by 10 ticks; succeeded.
        self.rig.policy(1)
        self.rig.engine.waiting = 10
        self.rig.tick(10 + FPS)
        self.rig.reset("succeeded")
        self.rig.controller.close()

        run_dir = self.rig.run_dir
        before = digest(run_dir)
        columns = phases(run_dir)
        self.assertEqual(len(columns), 6)
        [rewind] = self.rig.events("rewind")
        run = analysis.analyse(Run.at(run_dir), BLAME_S)
        cuts = analysis.training_cuts(run, analysis.SOURCES)

        def cut_ranges(episode: int, source: str) -> list[tuple[int, int]]:
            return [(c.start, c.stop) for c in cuts if c.episode == episode and c.source == source]

        # No cut has a paused or scripted frame, a suspect frame, or a frame of the ignored attempt.
        suspect = {a.episode: a.suspect_mask() for a in run.attempts}
        for cut in cuts:
            frames = columns[cut.episode][cut.start : cut.stop]
            self.assertTrue(np.isin(frames, [Phase.AUTONOMOUS, Phase.TELEOP]).all(), cut)
            self.assertFalse(suspect[cut.episode][cut.start : cut.stop].any(), cut)
        self.assertNotIn(1, {c.episode for c in cuts}, "attempt 2 was ignored")
        self.assertEqual(len({(c.episode, f) for c in cuts for f in range(c.start, c.stop)}),
                         sum(c.frames for c in cuts), "no frame is in two cuts")

        # The rewound policy frames are suspect, so only the policy frames before them are a cut.
        [(policy_start, _)] = runs_of(columns[0], Phase.AUTONOMOUS)
        self.assertEqual(cut_ranges(0, analysis.POLICY_SUCCESSES), [(policy_start, rewind["start_frame"])])
        # Attempt 5's last blame_s of policy frames before its pause are left out.
        first, second = runs_of(columns[4], Phase.AUTONOMOUS)
        blamed = round(BLAME_S * FPS)
        self.assertEqual(cut_ranges(4, analysis.POLICY_SUCCESSES), [(first[0], first[1] - blamed), second])

        # Attempt 6's waiting frames, where the policy sent no action, are in no cut and aren't policy time.
        [(start, stop)] = runs_of(columns[5], Phase.AUTONOMOUS)
        no_action = all_nan(run_dir, "control.policy_action")[5]
        [waiting] = runs_of(no_action & (columns[5] == Phase.AUTONOMOUS), True)
        self.assertEqual(waiting[1] - waiting[0], 10)
        self.assertEqual(cut_ranges(5, analysis.POLICY_SUCCESSES), [(start, waiting[0]), (waiting[1], stop)])
        self.assertAlmostEqual(run.attempts[5].seconds(FPS)[analysis.POLICY], (stop - start - 10) / FPS)

        # The corrections: attempt 4's Teleop before the policy corrects nothing, and it failed, so only
        # its second is in.
        driving, stepping_in = runs_of(columns[3], Phase.TELEOP)
        self.assertLessEqual(driving[1], runs_of(columns[3], Phase.AUTONOMOUS)[0][0])
        corrections = [(ep, a, b) for ep in (0, 2) for a, b in runs_of(columns[ep], Phase.TELEOP)]
        corrections.append((3, *stepping_in))
        self.assertEqual(len(corrections), 4)
        self.assertTrue(runs_of(columns[1], Phase.TELEOP), "the ignored attempt has a correction")

        build(self.s, TASK, (analysis.CORRECTIONS,), keep_labels=True)

        # build.json: the demo, then each correction as its own episode, and none of the ignored attempt.
        manifest = json.loads(data.build_manifest_path(self.s, TASK).read_text())
        self.assertEqual(manifest["episodes"][0], {"session": "20260101-100000", "episode": 0, "frames": [0, FPS]})
        self.assertEqual([(e["episode"], *e["frames"]) for e in manifest["episodes"][1:]], corrections)
        self.assertEqual({e["source"] for e in manifest["episodes"][1:]}, {analysis.CORRECTIONS})
        self.assertEqual(manifest["inputs"]["sources"], ["demos", analysis.CORRECTIONS])
        self.assertEqual(manifest["inputs"]["rules"], analysis.RULES_VERSION)
        # The start pose comes from the demo alone, whose first state is all 0, not from the corrections.
        self.assertEqual(set(data.demo_start_pose(self.s, TASK).values()), {0.0})

        # The built episodes hold exactly the correction frames: Teleop only, none suspect.
        built = table(data.train_root(self.s, TASK))
        recorded = table(run_dir / "dataset")
        # A demo corrects no policy: not an intervention, label 0.
        demo = built[built["episode_index"] == 0]
        self.assertFalse(np.ravel(np.stack(demo["intervention"])).any())
        self.assertTrue((np.ravel(np.stack(demo["control.label"])) == analysis.FrameLabel.NEUTRAL).all())
        self.assertEqual(built["episode_index"].nunique(), 1 + len(corrections))
        for n, (ep, a, b) in enumerate(corrections, start=1):
            rows = built[built["episode_index"] == n]
            source = recorded[recorded["episode_index"] == ep].sort_values("frame_index").iloc[a:b]
            np.testing.assert_array_equal(np.stack(rows["observation.state"]), np.stack(source["observation.state"]))
            # The action asked for, before the cap, as a demo's is: the leader's pose in Teleop.
            np.testing.assert_array_equal(np.stack(rows["action"]), np.stack(source["control.leader"]))
            self.assertTrue((np.ravel(np.stack(rows["control.phase"])) == Phase.TELEOP).all())
            self.assertTrue((np.ravel(np.stack(rows["control.label"])) == analysis.FrameLabel.CORRECTION).all())
            self.assertTrue(np.ravel(np.stack(rows["intervention"])).all())
            success = np.ravel(np.stack(rows["control.attempt_success"]))
            self.assertTrue((success == (ep == 0)).all(), "attempt 1 succeeded; attempts 3 and 4 failed")

        # A policy success's action is the policy's own, before the cap.
        build(self.s, TASK, (analysis.POLICY_SUCCESSES,))
        built = table(data.train_root(self.s, TASK))
        entries = json.loads(data.build_manifest_path(self.s, TASK).read_text())["episodes"][1:]
        self.assertTrue(entries)
        for n, entry in enumerate(entries, start=1):
            self.assertEqual(entry["source"], analysis.POLICY_SUCCESSES)
            ep, (a, b) = entry["episode"], entry["frames"]
            rows = built[built["episode_index"] == n]
            source = recorded[recorded["episode_index"] == ep].sort_values("frame_index").iloc[a:b]
            np.testing.assert_array_equal(np.stack(rows["action"]), np.stack(source["control.policy_action"]))

        # Neither reading nor building changed the run.
        self.assertEqual(digest(run_dir), before)

    # --- Kills and torn metadata --------------------------------------------------------------------

    def built_runs(self) -> list[tuple[str, int, int, int]]:
        """(run, episode, start, stop) of each run cut in the last build."""
        manifest = json.loads(data.build_manifest_path(self.s, TASK).read_text())
        return [(e["run"], e["episode"], *e["frames"]) for e in manifest["episodes"] if "run" in e]

    def build_quietly(self) -> str:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            build(self.s, TASK, (analysis.CORRECTIONS,))
        return printed.getvalue()

    def test_a_kill_or_torn_metadata_keeps_every_earlier_attempt(self) -> None:
        # Two saved corrections, then a third, copied as a hard kill inside its save's finalize would
        # leave the run. The recorder never closed the copy.
        for attempt in (1, 2, 3):  # the policy drives for 2 s, then a human corrects for 1 s
            self.rig.policy(2)
            self.rig.teleop(1)
            if attempt < 3:
                self.rig.reset("failed")
        held = self.task_dir / "held"
        dataset = self.rig.controller.recorder.dataset
        finalize = dataset.finalize

        def kill(*args, **kwargs):
            shutil.copytree(self.rig.run_dir, held / "save")
            return finalize(*args, **kwargs)

        with mock.patch.object(dataset, "finalize", side_effect=kill):
            self.rig.reset("failed")
        self.rig.controller.close()
        columns = phases(self.rig.run_dir)
        saved = [(ep, *runs_of(columns[ep], Phase.TELEOP)[0]) for ep in (0, 1)]
        shutil.move(self.rig.run_dir, held / "closed")
        run = self.task_dir / "runs" / "20260101-120000"

        def tear(name: str):
            def torn(run: Path) -> None:
                path = run / "dataset" / "meta" / name
                path.write_text(path.read_text()[: len(path.read_text()) // 2])  # the write was cut short
            return torn

        def footerless(run: Path) -> None:  # a data file the kill left without its footer costs only itself
            first = sorted((run / "dataset").glob("data/*/*.parquet"))[0]
            self.bad = first.with_name("file-999.parquet")
            self.bad.write_bytes(first.read_bytes()[: first.stat().st_size // 2])

        def unindexed(run: Path) -> None:  # a kill between finishing the data file and writing its index row
            sorted((run / "dataset" / "meta" / "episodes").glob("*/*.parquet"))[-1].unlink()

        rows = [("save", "save", footerless), ("torn info.json", "save", tear("info.json")),
                ("missing from the episode index", "closed", unindexed)]
        for case, source, damage in rows:
            with self.subTest(case):
                shutil.copytree(held / source, run)
                try:
                    damage(run)
                    printed = self.build_quietly()
                    self.assertEqual(self.built_runs(), [(run.name, *cut) for cut in saved])
                    if case == "save":
                        self.assertEqual(len([line for line in printed.splitlines() if self.bad.name in line]), 1)
                        manifest = json.loads(data.build_manifest_path(self.s, TASK).read_text())
                        self.assertEqual(len([note for note in manifest["notes"] if self.bad.name in note]), 1)
                    if case == "missing from the episode index":
                        [note] = [line for line in printed.splitlines() if f"run {run.name} attempt 3" in line]
                        self.assertIn("episode 2", note)
                finally:
                    shutil.rmtree(run)


if __name__ == "__main__":
    unittest.main()
