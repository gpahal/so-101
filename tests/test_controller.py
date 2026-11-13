"""The controller, driven through its public interface: submit a command, advance a tick, read the status.

Assertions are on the goals the fake follower and leader receive, the torque the fake leader's bus
sees, the status, command results and the files a run writes.
"""

import contextlib
import errno
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from fakes import (MOTORS, DeadStream, FakeCamera, FakeClock, FullDisk, NoRecorder, control, make_engine, make_follower,
                   make_leader)

from so_101 import rollout
from so_101.control.commands import Result
from so_101.control.controller import Controller
from so_101.control.follower import pose_from_steps
from so_101.control.policy import EngineDriver
from so_101.control.recorder import Phase, Recorder
from so_101.control.terminal import KeyReader
from so_101.data import Run
from so_101.report import Report

FPS = 30
# The arm starts at rest, folded: raw motor steps, as `so101 pose --save-rest` saves them.
REST_STEPS = {
    "shoulder_pan": 2048, "shoulder_lift": 900, "elbow_flex": 3100,
    "wrist_flex": 2600, "wrist_roll": 2048, "gripper": 1000,
}
# The demos' start pose, in degrees (gripper 0-100), as `data.demo_start_pose` gives it.
START_POSE = {
    "shoulder_pan.pos": 0.0, "shoulder_lift.pos": -40.0, "elbow_flex.pos": 40.0,
    "wrist_flex.pos": 30.0, "wrist_roll.pos": 0.0, "gripper.pos": 10.0,
}
# The policy sweeps the pan and wrist from the start pose, half a degree per action.
POLICY_ACTIONS = [
    [START_POSE[k] + (0.5 * i if k in ("shoulder_pan.pos", "wrist_flex.pos") else 0.0) for k in START_POSE]
    for i in range(1, 300)
]
# For rewinding: the policy sweeps the pan and wrist 0.3 degrees per action, and opens the gripper
# 0.15 units per action, so a rewind has the gripper to retrace too.
SWEEP = {"shoulder_pan.pos": 0.3, "wrist_flex.pos": 0.3, "gripper.pos": 0.15}
GRIPPER_ACTIONS = [[START_POSE[k] + SWEEP.get(k, 0.0) * i for k in START_POSE] for i in range(1, 300)]


check = unittest.TestCase()  # its assertions, for the rig's own steps


class Rig:
    def __init__(self, run_dir: Path | None = None, actions: list[list[float]] = POLICY_ACTIONS, **options) -> None:
        """With `run_dir`, the run is recorded there, and `disk.full` makes every flush of its event log
        fail as on a full disk; without, nothing is recorded (`fakes.NoRecorder`). The policy sends
        `actions` in order. `options` go to the Controller. The `[control]` settings are so101.toml's
        defaults, without camera previews, unless `options` has `control`. A leader is plugged in, folded
        at rest like the follower; it connects on the first tick. While `presence_error` is set, the
        device watch raises it."""
        self.clock = FakeClock()
        self.robot, self.motors = make_follower(self.clock, REST_STEPS)
        make, self.leader_motors = make_leader(self.clock, REST_STEPS)
        self.engine = make_engine(actions)
        driver = EngineDriver(self.engine, list(self.robot.action_features))
        self.run_dir = run_dir
        self.missing: dict[str, str] = {}  # what the device watch reports missing, with its reason
        self.presence_error: Exception | None = None
        self.reader: KeyReader | None = None
        recorder = NoRecorder(self.clock)
        if run_dir:
            open_file = Path.open

            def open_events(path, *args, **kwargs):  # the Recorder's one way to its event log
                file = open_file(path, *args, **kwargs)
                if path.name == "events.jsonl":
                    file = self.disk = FullDisk(file)
                return file

            with mock.patch.object(Path, "open", open_events):
                recorder = Recorder(Run.at(run_dir), self.robot, fps=FPS, task="test", clock=self.clock)
        self.controller = Controller(
            self.robot, driver, fps=FPS, rest_pose=pose_from_steps(self.robot.inner.bus, REST_STEPS),
            start_pose=START_POSE, clock=self.clock, recorder=recorder, presence=self.presence,
            leader=make, checks=lambda buses, cameras: Report(), **{"control": control(preview_fps=0), **options},
        )

    def presence(self) -> dict[str, str]:
        if self.presence_error is not None:
            raise self.presence_error
        return dict(self.missing)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.clock.advance(1 / FPS)
            self.controller.tick()

    def command(self, name: str, source: str = "terminal", **options) -> Result:
        """Submit a command and advance one tick, which applies it."""
        future = self.controller.submit(name, source, **options)
        self.tick()
        return future.result(timeout=0)

    def status(self) -> dict:
        return self.controller.status()

    def tick_until(self, state: str, limit: int = 20 * FPS, client: str | None = None) -> None:
        """Tick until the state is `state`, with the page `client`, if any, polling before each tick."""
        for _ in range(limit):
            if self.status()["state"] == state:
                return
            if client is not None:
                self.controller.heartbeat(client)
            self.tick()
        raise AssertionError(f"still {self.status()['state']} after {limit} ticks, not {state}")

    def press(self, key: str) -> str:
        """Press `key` at the robot's terminal and advance one tick, which applies it. Returns what it printed."""
        self.reader = self.reader or KeyReader(self.controller.submit, self.status)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.reader.on_key(key)
            self.tick()
        return out.getvalue()

    def start_policy(self, ticks: int = 10) -> None:
        """Start an attempt, and let the policy drive for `ticks`."""
        check.assertEqual(self.command("resume"), Result("resume", True))
        self.tick_until("autonomous")
        self.tick(ticks)

    def policy(self, seconds: float) -> None:
        self.start_policy(round(seconds * FPS))

    def teleop(self, seconds: float) -> None:
        self.command("teleop")
        self.tick_until("teleop")
        self.tick(round(seconds * FPS))

    def reset(self, outcome: str) -> None:
        self.command("reset", outcome=outcome)
        self.tick_until("resting")

    def fault(self) -> None:
        """The follower stops answering for a tick, and answers again a second later, still in Fault."""
        self.motors.failing = True
        self.tick()
        self.motors.failing = False
        self.tick(FPS)
        check.assertEqual(self.status()["state"], "fault")

    def in_fault(self) -> None:
        self.start_policy()
        self.fault()

    def short_of_rest(self) -> None:
        """A quit's drive to rest stopped short: something blocks the elbow."""
        self.start_policy()
        self.motors.stops = {"elbow_flex": self.motors.present()["elbow_flex"]}
        check.assertEqual(self.command("quit"), Result("quit", True))
        self.tick_until("resting", limit=10 * FPS)
        check.assertTrue(self.status()["confirm_quit"], "it warned that the arm isn't at rest")
        self.clock.advance(31)  # the warning lapses after 30 s
        self.tick()

    def resting_away_from_rest(self) -> None:
        """After a fault, Reset, then Pause on the way to rest."""
        self.in_fault()
        check.assertEqual(self.command("reset"), Result("reset", True))
        self.tick(5)
        check.assertEqual(self.command("pause"), Result("pause", True))
        check.assertEqual(self.status()["state"], "resting")
        check.assertFalse(self.controller.at_rest)

    def goals(self) -> list[dict[str, int]]:
        return self.motors.goals

    def holds(self, ticks: int = FPS, message: str = "the arm holds") -> None:
        """Advance `ticks` and check that each sent the follower its last goal again, and nothing else."""
        held, sent = self.goals()[-1], len(self.goals())
        self.tick(ticks)
        check.assertEqual(self.goals()[sent:], [held] * ticks, message)

    def units(self, steps: dict[str, int]) -> dict[str, float]:
        """Raw motor steps in the arm's units (degrees, gripper 0-100). Both arms share a calibration."""
        return pose_from_steps(self.robot.inner.bus, steps)

    def events(self, type_: str | None = None) -> list[dict]:
        lines = (self.run_dir / "events.jsonl").read_text().splitlines()
        return [e for e in map(json.loads, lines) if type_ is None or e["type"] == type_]

    def episodes(self) -> int:
        info = self.run_dir / "dataset" / "meta" / "info.json"
        return json.loads(info.read_text())["total_episodes"] if info.is_file() else 0

    def frames(self):
        """The recorded frames, and the action's joint names, read straight from the dataset."""
        root = self.run_dir / "dataset"
        names = json.loads((root / "meta" / "info.json").read_text())["features"]["action"]["names"]
        return pd.concat(pd.read_parquet(f) for f in sorted(root.glob("data/*/*.parquet"))), names

    def close(self) -> None:
        self.motors.failing, self.motors.powered = False, True  # so the motors turn off on disconnect
        if self.run_dir is not None:
            self.disk.full = False
        if self.robot.inner.is_connected:
            self.controller.close()
            self.robot.inner.disconnect()


class RigTest(unittest.TestCase):
    """A test on `self.rig`: recorded in a temporary folder when `recorded`, with `options` for the Rig."""

    recorded = False
    options: dict = {}

    def setUp(self) -> None:
        self.rig = self.new_rig(**self.options)

    def new_rig(self, name: str = "run", recorded: bool | None = None, **options) -> Rig:
        """Another rig, recorded as `name` when `recorded`, which defaults to the test's."""
        run_dir = None
        if self.recorded if recorded is None else recorded:
            if not hasattr(self, "temp"):
                temp = tempfile.TemporaryDirectory(prefix="so101-run-")
                self.addCleanup(temp.cleanup)
                self.temp = Path(temp.name)
            run_dir = self.temp / name
        rig = Rig(run_dir, **options)
        self.addCleanup(rig.close)
        return rig


class ControllerTest(RigTest):
    def test_space_and_toggle_pause_but_never_start_resume_or_restart_the_arm(self) -> None:
        # At the terminal, Space pauses but never starts or resumes the arm.
        self.rig.tick(2 * FPS)  # the arm has rested for 2 s, past the 1 s that lets the page's Space resume
        self.rig.press("space")
        self.assertEqual(self.rig.status()["state"], "resting", "Space never starts an attempt at the terminal")

        self.rig.press("r")
        self.rig.tick_until("autonomous")
        self.rig.tick(10)
        self.rig.press("space")
        self.assertEqual(self.rig.status()["state"], "paused")
        held = self.rig.goals()[-1]
        sent = len(self.rig.goals())
        self.rig.tick(2 * FPS)  # past the 1 s that lets the page's Toggle resume
        self.rig.press("space")
        self.assertEqual(self.rig.status()["state"], "paused")
        self.assertEqual(self.rig.goals()[sent:], [held] * (len(self.rig.goals()) - sent), "the arm holds throughout")

        self.rig.press("r")
        self.assertEqual(self.rig.status()["state"], "autonomous")
        self.rig.tick(10)

        # The page's Toggle: a double or held press pauses and never restarts the arm.
        self.assertEqual(self.rig.command("toggle"), Result("pause", True))
        held = self.rig.goals()[-1]
        sent = len(self.rig.goals())
        for _ in range(2 * FPS):  # a double press, then the key held: it repeats every tick for 2 s
            again = self.rig.command("toggle")
            self.assertEqual((again.command, again.accepted), ("resume", False))
        self.assertEqual(self.rig.status()["state"], "paused")
        self.assertEqual(self.rig.goals()[sent:], [held] * (len(self.rig.goals()) - sent), "the arm holds throughout")

        self.rig.tick(FPS + 1)  # let go for a second
        self.assertEqual(self.rig.command("toggle"), Result("resume", True))
        self.assertEqual(self.rig.status()["state"], "autonomous")

    def test_a_late_answer_to_the_terminals_prompt_never_ends_a_newer_attempt(self) -> None:
        self.rig.start_policy()
        self.assertIn("How did it go?", self.rig.press("right"))
        page = {"source": "page", "client": "phone", "who": "100.64.0.2"}
        self.assertEqual(self.rig.command("reset", outcome="succeeded", for_attempt=1, **page), Result("reset", True))
        self.rig.tick_until("resting")
        self.rig.tick(2 * FPS)
        self.assertEqual(self.rig.command("resume", **page), Result("resume", True))
        self.rig.tick_until("autonomous")

        printed = self.rig.press("f")  # the answer meant for attempt 1

        self.assertIn("Reset refused", printed)
        self.assertIn("attempt 1", printed)
        status = self.rig.status()
        self.assertEqual((status["state"], status["attempt"], status["attempt_open"]), ("autonomous", 2, True))
        self.assertIsNone(status["ending"])

        self.rig.press("right")
        self.rig.press("f")
        self.assertEqual((self.rig.status()["state"], self.rig.status()["ending"]), ("returning", "reset"))

    def test_pause_during_returning_cancels_the_reset(self) -> None:
        self.rig.start_policy()
        self.rig.command("reset")
        self.rig.tick(5)
        self.assertEqual(self.rig.status()["state"], "returning")

        self.assertEqual(self.rig.command("pause"), Result("pause", True))

        status = self.rig.status()
        self.assertEqual(status["state"], "paused")
        self.assertTrue(status["attempt_open"])
        self.assertIsNone(status["ending"])
        self.assertIsNone(status["last_attempt"])
        held = self.rig.goals()[-1]
        self.rig.tick(3 * FPS)
        self.assertTrue(all(goal == held for goal in self.rig.goals()[-3 * FPS:]), "the arm holds where it stopped")
        self.assertNotEqual(held, REST_STEPS)

        self.assertEqual(self.rig.command("resume"), Result("resume", True))
        self.assertEqual(self.rig.status()["state"], "autonomous")
        self.assertEqual(self.rig.status()["attempt"], 1, "the same attempt continues")

    def test_a_missing_follower_or_camera_refuses_resume_rewind_and_teleop(self) -> None:
        self.rig.tick()
        self.assertEqual(self.rig.status()["devices"]["leader"], "ok", "the leader is there, so only the device refuses")
        reason = "The follower isn't connected. Plug it in."
        self.rig.missing = {"follower": reason}
        sent = len(self.rig.goals())
        leader_sent = len(self.rig.leader_motors.goals)

        self.assertEqual(self.rig.command("resume"), Result("resume", False, reason))

        status = self.rig.status()
        self.assertEqual(status["state"], "resting")
        self.assertEqual(status["devices"]["follower"], "missing")
        self.assertFalse(status["allowed"]["resume"])
        self.assertEqual(status["refusals"]["resume"], reason)
        self.assertEqual(self.rig.goals()[sent:], [self.rig.goals()[sent - 1]], "the arm holds")
        self.assertEqual(self.rig.leader_motors.goals[leader_sent:], [], "the leader isn't driven")
        self.assertFalse(self.rig.leader_motors.torqued())

        # From a paused attempt too, where Rewind has motion to undo, and Resume, Rewind and Teleop
        # work again once the device is back.
        self.rig.missing = {}
        self.rig.start_policy()
        self.rig.command("pause")
        self.assertGreater(self.rig.status()["rewind_available_s"], 0)
        reason = "The top camera isn't connected. Plug it in."
        self.rig.missing = {"top": reason}
        sent = len(self.rig.goals())

        self.assertEqual(self.rig.command("rewind"), Result("rewind", False, reason))

        status = self.rig.status()
        self.assertEqual(status["state"], "paused")
        self.assertEqual(status["refusals"]["rewind"], reason)
        self.assertEqual(self.rig.goals()[sent:], [self.rig.goals()[sent - 1]], "the arm holds")
        self.rig.missing = {}
        self.assertEqual(self.rig.command("rewind"), Result("rewind", True))
        self.rig.tick_until("paused")
        self.assertEqual(self.rig.command("teleop"), Result("teleop", True))
        self.assertEqual(self.rig.status()["state"], "handover")
        self.rig.command("pause")
        self.assertEqual(self.rig.command("resume"), Result("resume", True))
        self.assertEqual(self.rig.status()["devices"], {"follower": "ok", "top": "ok", "wrist": "ok", "leader": "ok"})

    def test_teleop_hands_over_at_the_measured_pose_without_jumping(self) -> None:
        self.rig.start_policy()
        # Something blocks the pan, so the follower lags the policy's goal by more than the tolerance.
        self.rig.motors.stops = {"shoulder_pan": self.rig.motors.present()["shoulder_pan"]}
        self.rig.tick(FPS)
        commanded = self.rig.units(self.rig.goals()[-1])
        self.assertGreater(rollout.farthest(self.rig.units(self.rig.motors.present()), commanded)[1], rollout.TOLERANCE)
        self.assertFalse(self.rig.leader_motors.torqued(), "the leader hangs free until a handover")
        torque_changes = len(self.rig.leader_motors.torque)

        self.assertEqual(self.rig.command("teleop"), Result("teleop", True))
        self.assertEqual(self.rig.status()["state"], "handover")
        self.rig.tick_until("teleop", limit=10 * FPS)

        # Motors on, then off once: and only when every joint was within the tolerance of the
        # follower's measured pose, not of the goal it was last sent.
        on, off = self.rig.leader_motors.torque[torque_changes:]
        self.assertEqual((on[0], off[0]), (True, False))
        leader_at_release = self.rig.units(off[1])
        follower = self.rig.units(self.rig.motors.present())
        self.assertLessEqual(rollout.farthest(leader_at_release, follower)[1], rollout.TOLERANCE)
        self.assertGreater(rollout.farthest(leader_at_release, commanded)[1], rollout.TOLERANCE)
        self.assertGreater(len(self.rig.leader_motors.goals), 1, "the leader was driven there step by step")

        # The first Teleop goal is where the follower is, so it doesn't jump.
        sent = len(self.rig.goals())
        self.rig.tick()
        first = self.rig.units(self.rig.goals()[sent])
        self.assertLessEqual(rollout.farthest(first, follower)[1], rollout.TOLERANCE)

        # From then on the follower follows the leader, wherever a hand puts it.
        moved = {**self.rig.leader_motors.present(), "shoulder_pan": self.rig.leader_motors.present()["shoulder_pan"] - 40}
        self.rig.leader_motors.move_by_hand(moved)
        self.rig.tick()
        self.assertEqual(self.rig.goals()[-1], moved)
        self.assertFalse(self.rig.leader_motors.torqued())

        # Without a leader, Teleop is refused and says why.
        self.rig.command("pause")
        self.assertTrue(self.rig.leader_motors.torqued(), "Pause holds the leader with its motors on")
        self.rig.missing = {"leader": "The leader's board isn't connected."}
        self.rig.tick()
        refused = self.rig.command("teleop")
        self.assertFalse(refused.accepted)
        self.assertTrue(refused.reason.startswith("Leader arm not connected."), refused.reason)
        status = self.rig.status()
        self.assertEqual((status["state"], status["devices"]["leader"]), ("paused", "missing"))
        self.assertFalse(status["allowed"]["teleop"])
        self.assertTrue(status["allowed"]["resume"], "the leader is optional")

    def test_reset_sends_the_leader_to_rest_with_the_arm_and_then_turns_its_motors_off(self) -> None:
        rig = self.new_rig("leader-rest", recorded=True)
        rig.start_policy()
        rig.command("pause")
        rig.teleop(0.5)
        moved = {**rig.leader_motors.present(), "shoulder_lift": rig.leader_motors.present()["shoulder_lift"] + 400,
                 "wrist_flex": rig.leader_motors.present()["wrist_flex"] - 300}
        rig.leader_motors.move_by_hand(moved)  # a hand has the leader away from rest
        rig.tick()

        rig.command("reset", outcome="succeeded")
        self.assertTrue(rig.leader_motors.torqued(), "its motors turn on to drive it")
        rig.tick_until("resting")
        rig.tick(3 * FPS)  # it may arrive after the follower
        for motor, steps in REST_STEPS.items():
            self.assertAlmostEqual(rig.leader_motors.present()[motor], steps, delta=25, msg=f"leader {motor}")
        self.assertFalse(rig.leader_motors.torqued(), "at rest it hangs free, as when it connects")
        self.assertEqual(rig.events("leader_torque"), [])

        # A Pause on the way stops it where it is, motors on, as it stops the arm.
        rig.start_policy()
        rig.command("pause")
        rig.teleop(0.5)
        rig.leader_motors.move_by_hand(moved)
        rig.tick()
        rig.command("reset", outcome="failed")
        rig.tick(3)
        rig.command("pause")
        stopped = rig.leader_motors.present()
        rig.tick(FPS)
        self.assertTrue(rig.leader_motors.torqued())
        for motor, steps in stopped.items():
            self.assertAlmostEqual(rig.leader_motors.present()[motor], steps, delta=25, msg=f"leader {motor}")

    def test_a_quit_exits_only_once_the_leader_is_at_rest_too(self) -> None:
        # Exiting turns the leader's motors off: one still on its way to rest would fall from there.
        for how in ("quit", "esc_at_rest"):
            with self.subTest(how):
                rig = self.new_rig(f"leader-quit-{how}", recorded=True)
                rig.start_policy()
                rig.command("pause")
                rig.teleop(0.5)
                present = rig.leader_motors.present()
                rig.leader_motors.move_by_hand({**present, "shoulder_lift": present["shoulder_lift"] + 400,
                                                "wrist_flex": present["wrist_flex"] - 300})
                rig.tick()
                if how == "quit":
                    rig.command("quit")
                else:
                    rig.command("reset", outcome="succeeded")
                    rig.tick_until("resting")
                    self.assertTrue(rig.leader_motors.torqued(), "the leader is still on its way")
                    rig.command("quit")
                    # Nothing starts while the quit waits, or every later Esc would be refused as a quit under way.
                    for command in ("resume", "teleop"):
                        self.assertFalse(rig.command(command).accepted, command)
                    self.assertFalse(rig.status()["allowed"]["resume"])
                for _ in range(20 * FPS):
                    if rig.controller.done:
                        break
                    self.assertFalse(rig.controller.done)
                    rig.tick()
                self.assertTrue(rig.controller.done)
                rig.controller.close()
                for motor, steps in REST_STEPS.items():
                    self.assertAlmostEqual(rig.leader_motors.present()[motor], steps, delta=25,
                                           msg=f"leader {motor} reached rest before its motors turned off")
                self.assertFalse(rig.leader_motors.torqued())

    def test_an_attempt_a_fault_saved_with_its_outcome_is_never_labelled_twice(self) -> None:
        from so_101.commands.serve import print_serve_summary

        rig = self.new_rig("labelled-twice", recorded=True)
        rig.start_policy()
        rig.command("reset", outcome="succeeded")
        rig.robot.inner.cameras["top"].stalled()  # a fault on the way to rest saves it as it is
        rig.tick(2)
        self.assertEqual(rig.status()["state"], "fault")
        self.assertEqual([e["outcome"] for e in rig.events("outcome")], ["succeeded"])

        printed = rig.press("left")  # the terminal's ignore, which the page doesn't send in Fault
        self.assertIn("already labelled succeeded", printed)
        self.assertEqual(rig.status()["state"], "fault")
        self.assertEqual([e["outcome"] for e in rig.events("outcome")], ["succeeded"])
        self.assertEqual(rig.command("reset"), Result("reset", True), "Reset without an outcome still goes to rest")
        rig.tick_until("resting")
        rig.controller.close()

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print_serve_summary(Run.at(rig.run_dir))
        self.assertIn("Attempts saved: 1 (1 succeeded, 0 failed, 0 ignored, 0 unlabelled)", out.getvalue())

    def test_a_key_that_ends_the_attempt_instead_of_answering_never_says_it_goes_on(self) -> None:
        rig = self.new_rig("not-ended")
        rig.start_policy()
        rig.press("right")
        self.assertIn("Not ended", rig.press("p"), "Pause leaves the attempt open")
        rig.press("right")
        printed = rig.press("esc")
        self.assertNotIn("Not ended", printed)
        self.assertEqual(rig.status()["state"], "returning", "Esc ended the attempt to quit")

    def test_a_hot_or_alarmed_leader_never_connects_so_teleop_stays_unavailable(self) -> None:
        # Preflight's WARN says Teleop won't be available: a Handover would turn those motors on.
        rig = self.new_rig("leader-unfit")
        elbow = rig.leader_motors.registers[MOTORS.index("elbow_flex") + 1]
        elbow["present_temperature"] = 65
        rig.tick()
        status = rig.status()
        self.assertEqual(status["devices"]["leader"], "missing")
        self.assertIn("elbow_flex 65 °C", status["device_problems"]["leader"])
        self.assertFalse(rig.leader_motors.torqued())

        elbow["present_temperature"] = 40
        rig.leader_motors.status["wrist_roll"] = 4  # overheat alarm, until its power is cycled
        rig.tick(31 * FPS)
        self.assertIn("protection alarm", rig.status()["device_problems"]["leader"])

        rig.leader_motors.status.clear()
        rig.tick(31 * FPS)
        self.assertEqual(rig.status()["devices"]["leader"], "ok", "fit again: it connects")

    def test_a_leader_motor_left_on_in_teleop_is_logged_and_turned_off(self) -> None:
        rig = self.new_rig("stuck", recorded=True)
        rig.start_policy()
        rig.command("pause")
        rig.leader_motors.ignore_torque_off = {"wrist_flex": 1}  # it acknowledges the torque-off, and stays on
        rig.teleop(0.1)
        self.assertFalse(rig.leader_motors.torqued(), "turned off again at once")
        event = rig.events("leader_torque")[-1]
        self.assertEqual((event["why"], event["stuck"]), ("teleop", ["wrist_flex"]))
        self.assertEqual(event["motors"]["wrist_flex"], {"torque": 1, "status": "0x00"})

        # Turned on again later in Teleop: found within a second.
        rig.leader_motors.registers[4]["torque_enable"] = 1
        rig.tick(FPS + 1)
        self.assertEqual(rig.events("leader_torque")[-1]["stuck"], ["wrist_flex"])
        self.assertFalse(rig.leader_motors.torqued())
        self.assertEqual(rig.status()["state"], "teleop")

    def test_a_follower_error_goes_to_fault_and_holds_where_the_arm_is_found_when_it_answers(self) -> None:
        self.rig.start_policy()
        # Something blocks the pan, so the arm is short of its last goal when the bus fails.
        self.rig.motors.stops = {"shoulder_pan": self.rig.motors.present()["shoulder_pan"]}
        self.rig.tick(5)
        last_goal = self.rig.goals()[-1]

        self.rig.motors.failing = True
        self.rig.tick()

        status = self.rig.status()
        self.assertEqual(status["state"], "fault")
        self.assertEqual(status["devices"]["follower"], "missing")
        self.assertEqual(status["last_attempt"], {"attempt": 1, "ended_by": "fault", "outcome": None, "saved": False})
        self.assertFalse(status["attempt_open"])
        [fault] = [e for e in self.rig.controller.events(0, 500) if e["type"] == "fault"]
        self.assertEqual(fault["device"], "follower")
        self.assertIn("TimeoutError", fault["traceback"])

        # Only Reset and Run checks are allowed, and Reset not until the follower answers again.
        self.assertEqual({c for c, ok in status["allowed"].items() if ok}, {"check"})
        self.assertFalse(self.rig.command("pause").accepted)
        refused = self.rig.command("reset")
        self.assertIn("The follower stopped answering", refused.reason)
        self.rig.motors.failing = False
        sent = len(self.rig.goals())
        found = self.rig.motors.present()
        self.assertNotEqual(found, last_goal, "the pan never reached its last goal")
        self.rig.tick(FPS)
        self.assertEqual(self.rig.goals()[sent:], [found] * FPS, "the arm holds where it's found, never the old goal")
        self.assertEqual(self.rig.status()["allowed"],
                         {"resume": False, "pause": False, "rewind": False, "teleop": False, "reset": True,
                          "check": True, "pose": False})

        # A confirmation lapses on a state change.
        self.assertEqual(self.rig.command("quit"), Result("quit", True))
        self.assertTrue(self.rig.status()["confirm_quit"])
        self.assertEqual(self.rig.command("reset"), Result("reset", True))
        self.assertFalse(self.rig.status()["confirm_quit"], "going to rest starts the quit over")


class RecordingTest(RigTest):
    recorded = True

    def test_an_attempt_that_couldnt_be_saved_isnt_counted_and_refuses_new_attempts(self) -> None:
        rig = self.rig
        rig.start_policy()
        disk_error = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(rig.controller.recorder.dataset, "finalize", side_effect=disk_error):
            rig.command("reset", outcome="succeeded")
            rig.tick_until("resting")

        status = rig.status()
        self.assertIsNone(status["prompt"], "no outcome is asked for an attempt that wasn't saved")
        self.assertEqual(status["last_attempt"], {"attempt": 1, "ended_by": "reset", "saved": False,
                                                  "outcome": "succeeded"})
        self.assertEqual(status["summary"]["attempts"], 0, "the summary counts only saved attempts")
        self.assertFalse(rig.command("outcome", outcome="failed").accepted)
        self.assertEqual(rig.events("outcome"), [])
        [end] = rig.events("attempt_end")
        self.assertEqual((end["attempt"], end["ended_by"], end["saved"], end["episode"]), (1, "reset", False, None))
        [failed] = rig.events("recording_failed")
        self.assertIn("No space left on device", failed["error"])
        refused = rig.command("resume")
        self.assertFalse(refused.accepted)
        self.assertEqual(refused.reason, failed["error"])
        self.assertEqual(rig.status()["state"], "resting")
        self.assertEqual(rig.status()["refusals"]["resume"], refused.reason)

    def test_an_attempt_pauses_saves_at_rest_and_records_the_states_it_went_through(self) -> None:
        rig = self.rig
        rig.start_policy()
        driving = rig.goals()[-10:]
        self.assertGreater(len({tuple(g.values()) for g in driving}), 1, "the policy should be moving the arm")
        last_policy_goal = rig.goals()[-1]
        sent = len(rig.goals())

        # Pause holds and sends no policy action.
        self.assertEqual(rig.command("pause"), Result("pause", True))
        self.assertEqual(rig.status()["state"], "paused")
        self.assertEqual(rig.goals()[sent], last_policy_goal, "the next tick sends the hold target")
        rig.holds(FPS)
        rig.command("resume")
        rig.tick(5)

        # Reset saves the attempt only at rest, with how it ended and its outcome.
        self.assertEqual(rig.command("reset", outcome="failed"), Result("reset", True))
        rig.tick(3)
        self.assertEqual(rig.status()["state"], "returning")
        self.assertEqual(rig.status()["ending"], "reset")
        self.assertEqual(rig.episodes(), 0, "not saved before the arm is at rest")
        self.assertEqual(rig.events("attempt_end"), [])
        rig.tick_until("resting")

        self.assertEqual(rig.episodes(), 1)
        [end] = rig.events("attempt_end")
        self.assertEqual((end["attempt"], end["ended_by"], end["saved"], end["episode"]), (1, "reset", True, 0))
        [outcome] = rig.events("outcome")
        self.assertEqual((outcome["attempt"], outcome["outcome"], outcome["episode"]), (1, "failed", 0))
        for motor, steps in REST_STEPS.items():
            self.assertAlmostEqual(rig.goals()[-1][motor], steps, delta=1, msg=motor)
        status = rig.status()
        self.assertFalse(status["attempt_open"])
        self.assertEqual(status["last_attempt"], {"attempt": 1, "ended_by": "reset", "outcome": "failed", "saved": True})
        refused = rig.command("reset")
        self.assertFalse(refused.accepted)
        self.assertIn("already at rest", refused.reason)

        # The columns match the states the attempt went through. The recorder finalizes the dataset
        # after every saved attempt, so it reads while the run is still open.
        features = json.loads((rig.run_dir / "dataset" / "meta" / "info.json").read_text())["features"]
        self.assertEqual(
            sorted(key for key in features if key.startswith("observation.")),
            ["observation.images.top", "observation.images.wrist", "observation.state"],
        )
        frames, _ = rig.frames()
        runs = [(phase, len(list(group))) for phase, group in itertools.groupby(frames["control.phase"])]
        self.assertEqual(
            [phase for phase, _ in runs],
            [Phase.STARTING, Phase.AUTONOMOUS, Phase.PAUSED, Phase.AUTONOMOUS, Phase.RETURNING],
        )
        self.assertEqual(runs[2][1], FPS + 1, "the pause tick and every tick after it, until Resume")
        self.assertEqual(runs[3][1], 6, "the resume tick and the next 5, until Reset")
        self.assertFalse(frames["intervention"].any(), "no Teleop, so no intervention frames")
        self.assertFalse(np.isnan(np.stack(frames["control.leader"].to_numpy())).any(),
                         "the leader is connected, so its pose is in every frame")
        # The state events start each run of phases at the frame they say.
        starts = [e["frame"] for e in rig.events("state") if e["state"] != "resting"]
        self.assertEqual(starts, [sum(n for _, n in runs[:i]) for i in range(len(runs))])
        self.assertEqual(end["frames"], len(frames))
        self.assertEqual(end["seconds"]["paused"], round((FPS + 1) / FPS, 2))

        # Teleop after the policy drove is recorded as an intervention.
        rig.tick(2 * FPS)  # past the toggle's rule, so a new attempt may start
        rig.start_policy()
        rig.teleop(3 / FPS)
        rig.reset("succeeded")
        frames, _ = rig.frames()
        corrected = frames[frames["episode_index"] == 1]
        teleop = corrected["control.phase"] == Phase.TELEOP
        self.assertTrue(teleop.any())
        self.assertTrue((corrected["intervention"] == teleop).all(), "Teleop after the policy drove is a correction")

        # Ticks with no policy action don't make a following Teleop an intervention.
        rig.tick(2 * FPS)
        rig.engine.waiting = 10**6  # the policy never sends an action, as a cloud GPU that never answers
        rig.start_policy()
        rig.teleop(3 / FPS)
        rig.reset("failed")
        frames, _ = rig.frames()
        waited = frames[frames["episode_index"] == 2]
        autonomous = waited[waited["control.phase"] == Phase.AUTONOMOUS]
        self.assertEqual(len(autonomous), 10)
        self.assertTrue(np.isnan(np.stack(autonomous["control.policy_action"].to_numpy())).all(),
                        "a waiting tick records no policy action")
        self.assertTrue((waited["control.phase"] == Phase.TELEOP).any())
        self.assertFalse(waited["intervention"].any(), "a policy that only waited drove nothing to step in on")

    def test_frames_the_video_encoder_dropped_are_on_the_saved_attempts_attempt_end(self) -> None:
        """LeRobot keeps the frames its streaming encoder dropped only in the encoder's private
        `_dropped_frames`, per video feature and cleared as each episode starts. This fails if a
        LeRobot upgrade moves them, which would leave builds training on misaligned attempts."""
        rig = self.rig
        rig.start_policy()
        writer = rig.controller.recorder.dataset.writer
        self.assertTrue(hasattr(writer, "_streaming_encoder"), "LeRobot's dataset writer moved its streaming encoder")
        encoder = writer._streaming_encoder
        self.assertIsNotNone(encoder, "the run records with streaming encoding")
        self.assertTrue(hasattr(encoder, "_dropped_frames"), "LeRobot's streaming encoder moved its dropped frames")
        self.assertEqual(encoder._dropped_frames, {}, "the attempt's episode started with no frames dropped")
        encoder._dropped_frames["observation.images.top"] = 3  # as `feed_frame` counts a frame its full queue refused
        with self.assertLogs("lerobot.datasets.video_utils", "WARNING") as logs:
            rig.reset("succeeded")
        self.assertIn("3 dropped frame(s) for observation.images.top", "\n".join(logs.output),
                      "LeRobot reads the count as its own as it finishes the episode")

        [end] = rig.events("attempt_end")
        self.assertEqual((end["saved"], end.get("dropped_frames")), (True, {"top": 3}))

        # The next attempt's episode starts its count again.
        rig.tick(2 * FPS)  # past the toggle's rule, so a new attempt may start
        rig.start_policy()
        rig.reset("failed")
        self.assertNotIn("dropped_frames", rig.events("attempt_end")[1])


class RewindTest(RigTest):
    options = {"actions": GRIPPER_ACTIONS}

    def test_rewind_retraces_the_follower_and_the_leader_never_past_the_attempt_start(self) -> None:
        rig = self.rig
        rig.policy(9)

        self.assertEqual(rig.command("rewind"), Result("rewind", True))
        self.assertEqual(rig.status()["state"], "rewinding")
        rig.tick_until("paused", limit=4 * FPS)
        self.assertTrue(rig.leader_motors.torqued(), "the leader's motors are on, following the follower")
        while rig.command("rewind").accepted:
            rig.tick_until("paused", limit=10 * FPS)

        refused = rig.command("rewind")
        self.assertEqual(refused, Result("rewind", False, "Nothing left to rewind: the arm is back where attempt 1 started."))
        status = rig.status()
        self.assertEqual((status["state"], status["rewind_available_s"]), ("paused", 0.0))
        self.assertFalse(status["allowed"]["rewind"])
        for motor, steps in REST_STEPS.items():  # the attempt started at rest
            self.assertAlmostEqual(rig.goals()[-1][motor], steps, delta=2, msg=motor)
            self.assertAlmostEqual(rig.leader_motors.goals[-1][motor], steps, delta=2, msg=f"leader {motor}")


class TimeLimitTest(RigTest):
    LIMIT_S = 2  # attempt_limit_s, short so the test records few frames; the fake clock takes no real time
    recorded = True
    options = {"control": control(preview_fps=0, attempt_limit_s=float(LIMIT_S))}

    def test_only_policy_time_counts_and_the_limit_ends_the_attempt_like_reset(self) -> None:
        self.rig.policy(self.LIMIT_S / 2)
        self.rig.command("pause")
        self.rig.tick(2 * self.LIMIT_S * FPS)

        status = self.rig.status()
        self.assertEqual(status["state"], "paused", "a long pause doesn't end the attempt")
        self.assertAlmostEqual(status["policy_s"], self.LIMIT_S / 2, delta=0.1)
        self.assertEqual(status["limit_s"], self.LIMIT_S)

        self.rig.command("resume")
        self.rig.tick_until("returning", limit=self.LIMIT_S * FPS)
        self.assertEqual(self.rig.status()["ending"], "time_limit")
        self.rig.tick_until("resting")

        [end] = self.rig.events("attempt_end")
        self.assertEqual(
            (end["ended_by"], end["limit"], end["limit_s"], end["saved"], end["episode"]),
            ("time_limit", "attempt_limit_s", self.LIMIT_S, True, 0),
        )
        self.assertAlmostEqual(end["seconds"]["autonomous"], self.LIMIT_S, delta=0.1)
        self.assertGreaterEqual(end["seconds"]["paused"], 2 * self.LIMIT_S)
        self.assertEqual(self.rig.status()["prompt"], 1)
        self.assertEqual(self.rig.events("outcome"), [], "unlabelled until someone answers")

        self.assertEqual(self.rig.command("outcome", outcome="failed"), Result("outcome", True))
        [outcome] = self.rig.events("outcome")
        self.assertEqual((outcome["attempt"], outcome["outcome"], outcome["episode"]), (1, "failed", 0))
        self.assertIsNone(self.rig.status()["prompt"])

    def test_an_answer_labels_only_the_attempt_it_was_given_for(self) -> None:
        for _ in range(2):  # attempt 1 is left unanswered, and attempt 2 reaches its limit too
            self.rig.policy(self.LIMIT_S)
            self.rig.tick_until("resting")
        self.assertEqual(self.rig.status()["prompt"], 2)

        refused = self.rig.command("outcome", outcome="succeeded", for_attempt=1)
        self.assertFalse(refused.accepted)
        self.assertIn("attempt 2 is the one waiting", refused.reason)
        self.assertFalse(self.rig.command("outcome", outcome="succeeded", for_attempt=None).accepted,
                         "sent before attempt 2 asked")
        self.assertEqual(self.rig.events("outcome"), [])

        self.assertEqual(self.rig.press("f"), "", "the terminal answers for the attempt its status shows")
        [outcome] = self.rig.events("outcome")
        self.assertEqual((outcome["attempt"], outcome["outcome"]), (2, "failed"))


class LinkLossTest(RigTest):
    PHONE = "phone-page"  # the client id the controlling page made for itself
    TIMEOUT_S = 3  # the default link_timeout_s

    def page(self, name: str, client: str = PHONE) -> Result:
        return self.rig.command(name, source="page", client=client, who="100.64.0.2")

    def polled_ticks(self, n: int) -> None:
        """Ticks with the phone polling /status before each, as it does 4 times a second."""
        for _ in range(n):
            self.rig.controller.heartbeat(self.PHONE)
            self.rig.tick()

    def events(self, type_: str) -> list[dict]:
        return [e for e in self.rig.controller.events(0, 500) if e["type"] == type_]

    def test_link_loss_or_a_hidden_page_pauses_a_running_policy_or_the_move_to_the_start_pose(self) -> None:
        self.assertEqual(self.page("resume"), Result("resume", True), "the first page to act takes control")
        self.assertEqual(self.rig.controller.status(self.PHONE)["you_have_control"], True)
        self.rig.tick_until("autonomous", client=self.PHONE)
        self.polled_ticks(2 * self.TIMEOUT_S * FPS)
        self.assertEqual(self.rig.status()["state"], "autonomous", "polls keep the policy running")
        self.assertEqual(self.events("link_lost"), [])

        self.rig.tick(int((self.TIMEOUT_S - 0.5) * FPS))  # the phone stops polling
        self.assertEqual(self.rig.status()["state"], "autonomous", "a gap under link_timeout_s is no loss")
        self.rig.tick(FPS)

        self.assertEqual(self.rig.status()["state"], "paused")
        [lost] = self.events("link_lost")
        self.assertGreaterEqual(lost["seconds"], self.TIMEOUT_S)
        self.assertEqual((lost["on_link_loss"], lost["driving"]), ("pause", True))
        paused = [e for e in self.events("state") if e["state"] == "paused"]
        self.assertEqual([e["cause"] for e in paused], ["link_lost"])
        held = len(self.rig.goals())
        self.rig.tick(2 * FPS)
        self.assertEqual(self.rig.goals()[held:], [self.rig.goals()[held - 1]] * (len(self.rig.goals()) - held),
                         "the arm holds: no policy action after the pause")

        self.assertFalse(self.page("resume", client="laptop-page").accepted, "another page only views")
        self.polled_ticks(1)
        self.assertEqual(len(self.events("link_restored")), 1, "the next poll restores the link")
        self.assertEqual(self.rig.status()["state"], "paused", "it stays paused until someone resumes")

        # Hidden on the way to the start pose: the policy would start unwatched there, so the move stops.
        self.page("reset")
        self.rig.tick_until("resting", client=self.PHONE)
        self.page("resume")
        self.assertEqual(self.rig.status()["state"], "starting")
        held = len(self.rig.goals())
        self.assertEqual(self.page("page_hidden"), Result("page_hidden", True))
        self.assertEqual(self.rig.status()["state"], "paused", "paused on the tick the page was hidden")
        self.rig.tick(FPS)
        self.assertEqual(self.rig.goals()[held:], [self.rig.goals()[held - 1]] * (len(self.rig.goals()) - held),
                         "the arm holds where the move was: no goal after the loss")
        self.assertEqual(self.page("resume"), Result("resume", True), "Resume moves on to the start pose")
        self.assertEqual(self.rig.status()["state"], "starting")
        self.rig.tick_until("autonomous", client=self.PHONE)


class LoopSafetyTest(RigTest):
    """No error in a tick drops the arm, and no restarted motor is sent toward its old goal."""

    def test_a_dead_terminal_and_a_full_disk_during_a_fault_never_escape_the_tick(self) -> None:
        rig = self.new_rig(recorded=True)
        rig.start_policy()
        rig.disk.full = True
        with mock.patch.object(sys, "stderr", DeadStream()):
            rig.presence_error = RuntimeError("injected")
            rig.tick()  # raises out of the loop if a warning or a traceback can't be printed
            rig.presence_error = None
            self.assertEqual(rig.status()["state"], "fault")
            rig.holds(FPS)
        self.assertTrue(rig.motors.torqued())

    def test_a_tripped_gripper_only_warns_and_its_load_is_logged_but_other_alarms_fault(self) -> None:
        rig = self.new_rig("grasp", recorded=True)
        rig.start_policy()
        gripper = rig.motors.registers[6]
        gripper["present_load"] = 0x400 | 497  # 49.7%, closing: stalled at its 50% torque limit
        rig.tick(FPS + 1)
        self.assertEqual(rig.events("gripper_load")[-1]["load"], 49.7)
        self.assertEqual(rig.events("gripper_load")[-1]["attempt"], 1)
        rig.motors.status["gripper"] = 32  # the grasp held 2 s past the 25% threshold: overload protection trips
        rig.tick(2 * FPS)
        status = rig.status()
        self.assertEqual(status["state"], "autonomous", "a firm grasp carries on, as in so101 run")
        self.assertIn("overload protection tripped", " ".join(status["warnings"]))
        self.assertEqual(len(rig.events("gripper_overload")), 1, "logged once while it stands")
        gripper["present_load"] = 120  # 12%: released
        rig.motors.status["gripper"] = 0
        logged = len(rig.events("gripper_load"))
        rig.tick(2 * FPS)
        self.assertEqual(len(rig.events("gripper_load")), logged, "a light load isn't logged")
        self.assertEqual(rig.status()["warnings"], [])

        # Any other bit on the gripper, and an overload on any other joint, still faults.
        for motor, bits, named in (("gripper", 32 | 4, "gripper (overheat)"), ("elbow_flex", 32, "elbow_flex (overload)")):
            with self.subTest(motor):
                rig = self.new_rig(motor)
                rig.start_policy()
                rig.motors.status[motor] = bits
                rig.tick(FPS + 1)
                self.assertEqual(rig.status()["state"], "fault")
                self.assertIn(named, rig.status()["fault"])

    def test_an_alarm_bit_feetech_doesnt_name_is_logged_not_a_fault(self) -> None:
        rig = self.new_rig("unnamed", recorded=True)
        rig.start_policy()
        rig.motors.status["shoulder_lift"] = 0x10  # as on the arm: no LED, no torque cut
        rig.tick(FPS + 1)
        rig.motors.status["shoulder_lift"] = 0
        rig.tick(FPS)
        status = rig.status()
        self.assertEqual(status["state"], "autonomous")
        self.assertEqual(rig.events("motor_status")[-1]["bits"], {"shoulder_lift": "0x10"})
        self.assertIn("doesn't name 1 time this run, last shoulder_lift 0x10", " ".join(status["warnings"]))

        # A named bit with it still faults, naming only the protection.
        rig.motors.status["shoulder_lift"] = 0x10 | 4
        rig.tick(FPS + 1)
        self.assertEqual(rig.status()["state"], "fault")
        self.assertIn("shoulder_lift (overheat)", rig.status()["fault"])

    def test_a_persistent_motor_alarm_gets_a_hold_on_every_fault_tick(self) -> None:
        rig = self.rig
        rig.start_policy()
        rig.motors.status["elbow_flex"] = 32  # overload: every read of the motors, once a second, reports it
        rig.tick(FPS + 1)
        self.assertEqual(rig.status()["state"], "fault")

        rig.holds(3 * FPS, "one hold a tick, the ticks that read the alarm too")

        self.assertTrue(rig.motors.torqued())

    def test_a_motor_restart_or_a_power_cut_never_sends_the_arm_toward_its_old_goal(self) -> None:
        rig = self.rig
        rig.start_policy()
        last_goal = rig.goals()[-1]
        fell = {motor: steps - 300 for motor, steps in last_goal.items()}
        rig.motors.power_cycle(fell)  # a brownout: back before any read failed
        sent = len(rig.goals())

        rig.tick()
        status = rig.status()
        self.assertEqual(status["state"], "fault")
        self.assertIn("lost power or restarted", status["fault"])
        self.assertEqual(rig.goals()[sent:], [], "no goal on the tick that found the restart")
        rig.tick(3)
        self.assertEqual(rig.goals()[sent:], [fell] * 3, "it holds where it fell")
        self.assertEqual(rig.motors.present(), fell, "the arm never moved toward the old goal")
        rig.tick(FPS)
        self.assertEqual(rig.status()["state"], "fault", "the new mark doesn't clear the fault: Reset does")

        # A power cut while already in Fault for something else.
        rig = self.new_rig("another-fault")
        rig.start_policy()
        rig.engine.error = "the policy failed"  # a fault the follower isn't to blame for
        rig.tick()
        self.assertEqual(rig.status()["state"], "fault")
        last_goal = rig.goals()[-1]
        sent = len(rig.goals())
        rig.tick(3)
        self.assertEqual(rig.goals()[sent:], [last_goal] * 3, "it holds its last goal, keeping any grip")

        fell = {motor: steps - 300 for motor, steps in last_goal.items()}
        rig.motors.power_off()  # now the power goes, while it's already in Fault
        sent = len(rig.goals())
        rig.tick(3)
        self.assertEqual(rig.goals()[sent:], [], "no goal while the follower is silent")
        rig.motors.power_on(fell)
        written = len(rig.motors.writes)
        rig.tick()
        self.assertEqual(rig.motors.writes[written:], [("goal_position", fell), ("torque_enable", True)],
                         "aimed where it fell before its torque is on")
        rig.tick(2)

        self.assertEqual(rig.goals()[sent:], [fell] * 3, "it holds where it fell, never the old goal")

    def test_a_power_cut_the_servos_answer_through_never_pulls_the_arm_back(self) -> None:
        # The servos' chips keep answering at 4 V while their motors are limp and the arm falls. When
        # the power returns, each servo drives to the goal it last got: that must be where the arm fell.
        def paused(rig: Rig) -> None:
            rig.start_policy()
            rig.command("pause")

        def in_alarm_fault(rig: Rig) -> None:
            rig.start_policy()
            rig.motors.status["elbow_flex"] = 4  # overheat: a fault that holds the last goal
            rig.tick(FPS + 1)
            self.assertEqual(rig.status()["state"], "fault")

        for name, setup in (("autonomous", Rig.start_policy), ("paused", paused), ("alarm", in_alarm_fault)):
            with self.subTest(name):
                rig = self.new_rig(f"power-{name}", recorded=True)
                setup(rig)
                held = rig.motors.present()
                rig.motors.cut_motor_power()
                fell = {motor: steps - 300 for motor, steps in held.items()}
                rig.motors.move_by_hand(fell)
                rig.tick(2)
                status = rig.status()
                self.assertEqual(status["state"], "fault")
                if name != "alarm":
                    self.assertIn("power dropped", status["fault"])
                self.assertIn("power dropped", status["device_problems"]["follower"])
                self.assertEqual([e["change"] for e in rig.events("power")], ["lost"])
                self.assertFalse(rig.command("reset").accepted, "no Reset while the motors have no power")

                sent = len(rig.goals())
                rig.tick(FPS)
                for goal in rig.goals()[sent:]:
                    for motor, steps in fell.items():
                        self.assertAlmostEqual(goal[motor], steps, delta=2, msg=f"{motor}: aimed where the arm is")

                # The supply comes back lower than at start, under load (87.5%), and a loaded joint settles
                # 3 steps below its goal: aiming at where it settles would lower the arm every tick.
                rig.motors.sag = {"shoulder_lift": 3, "elbow_flex": 3}
                rig.motors.restore_motor_power(volts=10.5)
                rig.tick(2 * FPS)
                for motor, steps in fell.items():
                    self.assertAlmostEqual(rig.motors.present()[motor], steps, delta=4,
                                           msg=f"{motor} stays where it fell when the power returns")
                self.assertEqual([e["change"] for e in rig.events("power")], ["lost", "back"])
                self.assertTrue(rig.motors.torqued(), "its torque is on again, where the arm is")
                if name != "alarm":
                    self.assertTrue(rig.command("reset").accepted)
                    rig.tick_until("resting")

    def test_one_low_voltage_read_changes_nothing_and_a_voltage_alarm_is_a_power_loss(self) -> None:
        rig = self.new_rig("dip", recorded=True)
        rig.start_policy()
        rig.motors.cut_motor_power(volts=8.0)  # one read under 9 V of a 12 V supply, as a dip might be
        rig.tick()
        rig.motors.restore_motor_power()
        rig.tick(FPS)
        self.assertEqual(rig.status()["state"], "autonomous")
        self.assertEqual(rig.events("power"), [])

        rig.motors.status["wrist_roll"] = 1  # the servo's own voltage alarm, on two replies in a row
        rig.tick()
        self.assertEqual(rig.status()["state"], "autonomous", "one reply with the bit, as a garbled one, changes nothing")
        rig.tick()
        self.assertEqual(rig.status()["state"], "fault")
        self.assertIn("power dropped", rig.status()["fault"])
        self.assertTrue(rig.events("power")[0]["alarm"])


class QuitTest(RigTest):
    """Where the arm can't get to rest, only y at the terminal, within 30 s of the warning, exits: no
    repeated Esc and no signal does. The test runs in two places the arm can't get to rest, and in
    `short_of_rest` the quit's own warning lapses first."""

    recorded = True
    PLACES = ("in_fault", "short_of_rest")

    def setUp(self) -> None:
        pass  # each place gets its own rig

    def warnings(self) -> int:
        return sum("press y" in message for message in self.said)

    def test_only_y_at_the_terminal_within_30_s_of_the_warning_exits_away_from_rest(self) -> None:
        for place in self.PLACES:
            with self.subTest(place):
                self.said: list[str] = []
                rig = self.new_rig(place, notify=lambda message, state: self.said.append(message))
                getattr(rig, place)()
                printed = rig.press("y")
                self.assertIn("Exit here refused", printed, "nothing asked yet, or the request lapsed")
                self.assertFalse(rig.status()["confirm_quit"])

                # No signal exits, and Exit here is taken only from the terminal.
                for _ in range(5):
                    quit = rig.controller.submit("quit", "signal")
                    rig.tick()
                    self.assertTrue(quit.result(timeout=0).accepted)
                    rig.clock.advance(0.5)
                rig.tick(FPS)
                self.assertFalse(rig.controller.done)
                self.assertTrue(rig.motors.torqued())
                for source in ("signal", "page", "unknown"):
                    refused = rig.command("exit_here", source=source, client="phone", who="100.64.0.2")
                    self.assertFalse(refused.accepted, source)
                    self.assertFalse(rig.controller.done, source)

                # A repeated Esc only repeats the warning.
                warned = self.warnings()
                rig.press("esc")
                self.assertFalse(rig.controller.done)
                rig.clock.advance(0.5)  # the held key's first repeat
                rig.press("esc")
                self.assertFalse(rig.controller.done, "a second Esc never exits away from rest")
                self.assertEqual(self.warnings(), warned + 2, "each Esc gives the warning")
                self.assertTrue(rig.status()["confirm_quit"])
                self.assertTrue(rig.motors.torqued())

                rig.tick(29 * FPS)
                rig.press("y")
                self.assertTrue(rig.controller.done)
                self.assertEqual(rig.episodes(), 1, "the attempt is saved where the arm stopped")
                self.assertEqual(rig.events("attempt_end")[-1]["ended_by"],
                                 "quit" if place == "short_of_rest" else "fault")
                rig.controller.close()
                self.assertEqual(rig.events()[-1]["type"], "run_end")


    def test_a_quit_while_the_follower_is_silent_exits_at_once_and_leaves_its_motors(self) -> None:
        # The arm's power is switched off mid-attempt: exiting can't turn its motors off, so no y is needed.
        for source in ("signal", "terminal"):
            with self.subTest(source):
                self.said = []
                rig = self.new_rig(f"silent-{source}", notify=lambda message, state: self.said.append(message))
                rig.start_policy()
                rig.motors.power_off()
                rig.tick(3)
                self.assertEqual(rig.status()["state"], "fault")

                quit = rig.controller.submit("quit", source)
                rig.tick()
                self.assertTrue(quit.result(timeout=0).accepted)
                self.assertTrue(rig.controller.done, "a silent follower needs no confirmation")
                self.assertTrue(rig.controller.leaves_motors)
                self.assertTrue(any("leaves its motors as they are" in message for message in self.said))
                self.assertEqual(rig.episodes(), 1, "the attempt was saved at the fault")
                rig.controller.close()
                self.assertEqual(rig.events()[-1]["type"], "run_end")

        # A motor restart its marks found answered: the follower may hold a raised arm, so a quit only asks.
        rig = self.new_rig("restarted")
        rig.start_policy()
        rig.motors.power_cycle({motor: steps - 300 for motor, steps in rig.motors.present().items()})
        rig.tick()
        self.assertEqual(rig.status()["state"], "fault")
        quit = rig.controller.submit("quit", "signal")
        rig.tick()
        self.assertTrue(quit.result(timeout=0).accepted)
        self.assertFalse(rig.controller.done, "a follower that answers never exits on a signal")
        self.assertFalse(rig.controller.leaves_motors)

        # Once it answers again, the arm may be powered and raised: a quit only asks.
        rig = self.new_rig("answering-again")
        rig.start_policy()
        rig.motors.power_off()
        rig.tick(3)
        rig.motors.power_on(rig.motors.present())
        rig.tick(2)
        rig.controller.submit("quit", "signal")
        rig.tick(FPS)
        self.assertFalse(rig.controller.done)
        self.assertFalse(rig.controller.leaves_motors)
        self.assertTrue(rig.status()["confirm_quit"])
        self.assertTrue(rig.motors.torqued(), "it holds where it fell")


class ResetTest(RigTest):
    """Reset always has a way to send the arm to rest, and an outcome given with it is recorded or refused."""

    recorded = True

    def test_reset_after_a_paused_return_from_a_fault_drives_to_rest_without_an_attempt(self) -> None:
        self.rig.resting_away_from_rest()
        self.assertTrue(self.rig.status()["allowed"]["reset"])

        self.assertEqual(self.rig.command("reset"), Result("reset", True))
        self.assertEqual(self.rig.status()["state"], "returning")
        self.rig.tick_until("resting")

        for motor, steps in REST_STEPS.items():
            self.assertAlmostEqual(self.rig.goals()[-1][motor], steps, delta=1, msg=motor)
        status = self.rig.status()
        self.assertEqual((status["attempt"], status["attempt_open"]), (1, False), "no attempt opened")
        self.assertEqual(len(self.rig.events("attempt_start")), 1)
        refused = self.rig.command("reset")
        self.assertFalse(refused.accepted)
        self.assertIn("already at rest", refused.reason)

    def assert_refused_and_held(self, **options) -> Result:
        """Reset with `options` in Fault is refused, and the arm holds where it is."""
        held = self.rig.goals()[-1]
        sent = len(self.rig.goals())
        refused = self.rig.command("reset", **options)
        self.assertFalse(refused.accepted, options)
        self.rig.tick(FPS)
        self.assertEqual(self.rig.status()["state"], "fault")
        self.assertEqual(self.rig.goals()[sent:], [held] * (FPS + 1), "no goal changes")
        return refused

    def test_an_outcome_given_with_reset_in_fault_labels_the_attempt_the_fault_saved(self) -> None:
        self.rig.start_policy()
        self.rig.fault()
        self.assert_refused_and_held(outcome="succeeded", for_attempt=2)  # a sheet for another attempt
        self.assertEqual(self.rig.events("outcome"), [])

        self.assertEqual(self.rig.command("reset", outcome="succeeded", for_attempt=1), Result("reset", True))
        [outcome] = self.rig.events("outcome")
        self.assertEqual((outcome["attempt"], outcome["outcome"], outcome["episode"]), (1, "succeeded", 0))
        self.assertEqual(self.rig.status()["last_attempt"],
                         {"attempt": 1, "ended_by": "fault", "outcome": "succeeded", "saved": True})
        self.rig.tick_until("resting")

        # A fault with no attempt open saves none, so there is nothing to label.
        self.rig.fault()
        refused = self.assert_refused_and_held(outcome="ignored")
        self.assertIn("no attempt", refused.reason)
        self.assertEqual(len(self.rig.events("outcome")), 1)
        self.assertEqual(self.rig.command("reset"), Result("reset", True), "Reset without an outcome goes to rest")

    def test_a_fault_on_the_tick_an_attempt_opens_saves_nothing_to_label(self) -> None:
        camera = self.rig.robot.inner.cameras["top"]
        camera.is_connected = False  # the first observation of the attempt fails
        self.addCleanup(setattr, camera, "is_connected", True)  # before the rig closes
        self.assertEqual(self.rig.command("resume"), Result("resume", True))

        status = self.rig.status()
        self.assertEqual(status["state"], "fault")
        self.assertFalse(status["last_attempt"]["saved"])
        self.assertEqual(status["summary"]["attempts"], 0, "the summary counts only saved attempts")
        refused = self.assert_refused_and_held(outcome="succeeded", for_attempt=1)
        self.assertIn("The fault saved no attempt", refused.reason)
        self.assertEqual(self.rig.events("outcome"), [])


class ArmsTest(RigTest):
    """Goals without cameras, and the follower's restart marks."""

    def test_a_lost_camera_faults_naming_it_and_reset_reaches_rest(self) -> None:
        def disconnected(camera) -> None:
            camera.is_connected = False

        # It reports itself disconnected, or, as an unplugged camera usually does, stays "connected"
        # while its reads fail: no new frame, then LeRobot's read thread giving up.
        for case, lose in (("disconnected", disconnected), ("stalled", FakeCamera.stalled),
                           ("read thread died", FakeCamera.thread_died)):
            with self.subTest(case):
                rig = self.new_rig()
                rig.start_policy()
                camera = rig.robot.inner.cameras["top"]
                lose(camera)
                self.addCleanup(camera.connect)  # before the rig closes
                sent = len(rig.goals())

                rig.tick()
                status = rig.status()
                self.assertEqual(status["state"], "fault")
                self.assertIn("top camera", status["fault"])
                self.assertEqual(status["devices"]["top"], "missing")
                rig.tick(FPS)
                self.assertGreaterEqual(len(rig.goals()) - sent, FPS, "the arm keeps getting its hold every tick")
                self.assertEqual(len({tuple(g.values()) for g in rig.goals()[sent:]}), 1, "and holds where it was")

                self.assertEqual(rig.command("reset"), Result("reset", True))
                rig.tick_until("resting")
                for motor, steps in REST_STEPS.items():
                    self.assertAlmostEqual(rig.goals()[-1][motor], steps, delta=1, msg=motor)

    def test_a_torque_limit_changed_to_another_value_is_its_own_fault_and_left_as_found(self) -> None:
        rig = self.rig
        rig.start_policy()
        gripper = rig.motors.registers[MOTORS.index("gripper") + 1]
        gripper["torque_limit"] = 250  # something other than a restart lowered it, such as the servo's protection
        last_goal = rig.goals()[-1]
        sent = len(rig.goals())

        rig.tick()
        status = rig.status()
        self.assertEqual(status["state"], "fault")
        self.assertIn("gripper", status["fault"])
        self.assertIn("250", status["fault"])
        self.assertNotIn("lost power", status["fault"])
        rig.tick(FPS)
        self.assertEqual(rig.goals()[sent:], [last_goal] * (FPS + 1), "it holds the last goal every tick")
        self.assertEqual(gripper["torque_limit"], 250, "never marked again")
        self.assertEqual([e["type"] for e in rig.controller.events(0, 500) if e["type"] == "fault"], ["fault"])

        # Another value, found while already in Fault, is reported too, and the arm keeps holding.
        gripper["torque_limit"] = 300
        sent = len(rig.goals())
        rig.tick(FPS)
        self.assertIn("gripper reads 300", rig.status()["fault"])
        self.assertEqual(rig.goals()[sent:], [last_goal] * FPS)
        self.assertEqual(len([e for e in rig.controller.events(0, 500) if e["type"] == "fault"]), 2)
        self.assertEqual(gripper["torque_limit"], 300)

        self.assertEqual(rig.command("reset"), Result("reset", True))
        rig.tick_until("resting")
        self.assertEqual(gripper["torque_limit"], 300, "still as found")

        # A read timeout makes the follower lost; when it answers again, only a restarted motor is
        # marked again, and the changed value stays as found.
        rig.motors.failing = True
        rig.tick(2)
        rig.motors.failing = False
        rig.tick(3)
        self.assertEqual(gripper["torque_limit"], 300, "a timeout never re-marks a changed limit")
        elbow = rig.motors.registers[MOTORS.index("elbow_flex") + 1]
        self.assertEqual(elbow["torque_limit"], elbow["max_torque_limit"] - 1, "the others keep their mark")


if __name__ == "__main__":
    unittest.main()
