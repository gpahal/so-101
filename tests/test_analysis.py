"""Run analysis, through its public interface: read a run directory back into attempts, suspect
frames and training cuts.

The runs are recorded through the controller harness in test_controller, so they have the events
and columns a real `so101 serve` run has. Expected frames come from the run's own files: the
`control.phase` column and the `rewind` events.
"""

import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from test_controller import FPS, GRIPPER_ACTIONS, Rig

from so_101.control import analysis
from so_101.control.recorder import Phase
from so_101.data import Run

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


def no_policy_action(run_dir: Path) -> dict[int, np.ndarray]:
    return all_nan(run_dir, "control.policy_action")


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


class AnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="so101-run-")
        self.addCleanup(temp.cleanup)
        self.rig = Rig(Path(temp.name) / "run", actions=GRIPPER_ACTIONS)
        self.addCleanup(self.rig.close)

    def rewind(self) -> None:
        self.rig.command("rewind")
        self.rig.tick_until("paused", limit=10 * FPS)

    def finish(self) -> None:
        self.rig.controller.close()

    def test_training_cuts_exclude_paused_scripted_suspect_waiting_and_ignored_frames(self) -> None:
        # 1: the policy, paused and rewound, then a human's correction; succeeded.
        self.rig.policy(3)
        self.rig.command("pause")
        self.rewind()
        self.rig.teleop(2)
        self.rig.reset("succeeded")
        # 2: ignored.
        self.rig.policy(2)
        self.rig.reset("ignored")
        # 3: the policy, paused without a rewind and resumed; succeeded.
        self.rig.policy(3)
        self.rig.command("pause")
        self.rig.tick(5)
        self.rig.command("resume")
        self.rig.tick(FPS)
        self.rig.reset("succeeded")
        # 4: the policy, whose next chunk (from the cloud GPU) is late by 10 ticks; succeeded.
        self.rig.policy(1)
        self.rig.engine.waiting = 10
        self.rig.tick(10 + FPS)
        self.rig.reset("succeeded")
        self.finish()

        before = digest(self.rig.run_dir)
        columns = phases(self.rig.run_dir)
        self.assertEqual(len(columns), 4)
        [rewind] = self.rig.events("rewind")
        run = analysis.analyse(Run.at(self.rig.run_dir), BLAME_S)
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
        # Attempt 3's last blame_s of policy frames before its pause are left out.
        first, second = runs_of(columns[2], Phase.AUTONOMOUS)
        blamed = round(BLAME_S * FPS)
        self.assertEqual(cut_ranges(2, analysis.POLICY_SUCCESSES), [(first[0], first[1] - blamed), second])

        # Attempt 4's waiting frames, where the policy sent no action, are in no cut and aren't policy time.
        [(start, stop)] = runs_of(columns[3], Phase.AUTONOMOUS)
        [waiting] = runs_of(no_policy_action(self.rig.run_dir)[3] & (columns[3] == Phase.AUTONOMOUS), True)
        self.assertEqual(waiting[1] - waiting[0], 10)
        self.assertEqual(cut_ranges(3, analysis.POLICY_SUCCESSES), [(start, waiting[0]), (waiting[1], stop)])
        self.assertAlmostEqual(run.attempts[3].seconds(FPS)[analysis.POLICY], (stop - start - 10) / FPS)

        # Reading never changed the run.
        self.assertEqual(digest(self.rig.run_dir), before)


def add_footerless_file(run_dir: Path) -> Path:
    """A copy of the run's first data file cut off before its footer, as `file-999.parquet`."""
    first = sorted((run_dir / "dataset").glob("data/*/*.parquet"))[0]
    whole = first.read_bytes()
    bad = first.with_name("file-999.parquet")
    bad.write_bytes(whole[: len(whole) // 2])
    return bad


if __name__ == "__main__":
    unittest.main()
