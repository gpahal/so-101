"""`so101 build --with`, through its public interface: build a task's training dataset from demo
sessions plus a recorded run, then read the built dataset and build.json back.

The run is recorded through the controller harness in test_controller. The demo session is a small
LeRobot dataset with the same robot and cameras. Cameras show mid-grey, since the build's validation
fails a mostly black video.
"""

import contextlib
import dataclasses
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from fakes import FakeCamera, FakeClock, make_follower
from test_analysis import BLAME_S, add_footerless_file, phases, runs_of
from test_controller import FPS, GRIPPER_ACTIONS, REST_STEPS, Rig

from so_101 import data, settings
from so_101.build import build
from so_101.control import analysis
from so_101.control.recorder import Phase, features
from so_101.tasks import Task

TASK = Task(name="test", prompt="test", cameras=("front", "wrist"), fps=FPS,
            episodes=1, episode_time_s=10, reset_time_s=5, props=[], setup=[])


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
                           "observation.images.front": image, "observation.images.wrist": image})
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

    # --- Test case 11, the build half ---------------------------------------------------------------

    def test_each_correction_is_its_own_episode_without_paused_scripted_suspect_or_ignored_frames(self) -> None:
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
        self.rig.controller.close()

        run_dir = self.rig.run_dir
        columns = phases(run_dir)
        driving, stepping_in = runs_of(columns[3], Phase.TELEOP)
        self.assertLessEqual(driving[1], runs_of(columns[3], Phase.AUTONOMOUS)[0][0])
        # Attempt 4's Teleop before the policy corrects nothing, and it failed, so only its second is in.
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
            np.testing.assert_array_equal(np.stack(rows["action"]), np.stack(source["action"]))
            self.assertTrue((np.ravel(np.stack(rows["control.phase"])) == Phase.TELEOP).all())
            self.assertTrue((np.ravel(np.stack(rows["control.label"])) == analysis.FrameLabel.CORRECTION).all())
            self.assertTrue(np.ravel(np.stack(rows["intervention"])).all())
            success = np.ravel(np.stack(rows["control.attempt_success"]))
            self.assertTrue((success == (ep == 0)).all(), "attempt 1 succeeded, attempt 3 failed")

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
        # Two saved corrections, then a third, copied as a hard kill mid-attempt and inside its save's
        # finalize would leave the run. The recorder never closed either copy.
        for attempt in (1, 2, 3):  # the policy drives for 2 s, then a human corrects for 1 s
            self.rig.policy(2)
            self.rig.teleop(1)
            if attempt < 3:
                self.rig.reset("failed")
        held = self.task_dir / "held"
        shutil.copytree(self.rig.run_dir, held / "mid-attempt")
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
            self.bad = add_footerless_file(run)

        def unindexed(run: Path) -> None:  # a kill after the data file was finished and before its index row was
            sorted((run / "dataset" / "meta" / "episodes").glob("*/*.parquet"))[-1].unlink()

        rows = [("mid-attempt", "mid-attempt", lambda run: None), ("save", "save", footerless),
                ("torn stats.json", "save", tear("stats.json")), ("torn info.json", "save", tear("info.json")),
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
