"""Connecting the follower holds it where it is, whatever goal its servos last got (`so_101.connect`)."""

import json
import tempfile
import unittest
from pathlib import Path

from fakes import FakeClock, FakeMotors, calibration

from so_101 import connect

REST = {"shoulder_pan": 2048, "shoulder_lift": 900, "elbow_flex": 3100,
        "wrist_flex": 2600, "wrist_roll": 2048, "gripper": 1000}


def follower(motors: FakeMotors, calib: dict | None = None):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    directory = Path(tempfile.mkdtemp(prefix="so101-test-"))
    (directory / "follower.json").write_text(json.dumps(calib or calibration()))
    robot = SO101Follower(SO101FollowerConfig(port="/dev/fake", id="follower", calibration_dir=directory,
                                              use_degrees=True, max_relative_target=10.0))
    robot.bus._open_bus = lambda: motors
    robot.cameras = {}
    return robot


class ConnectTest(unittest.TestCase):
    def test_connecting_never_drives_the_arm_to_the_goal_its_servos_last_got(self) -> None:
        # `serve`'s y turned the torque off with the arm raised, and a hand lowered it 400 steps: each
        # servo still has the raised pose as its goal.
        clock = FakeClock()
        motors = FakeMotors(clock, calibration(), REST)
        for i, register in motors.registers.items():
            register["goal_position"] = REST[motors.names[i]] + 400

        connect.install()
        connect.install()  # a second install changes nothing
        robot = follower(motors)
        robot.connect(calibrate=False)
        clock.advance(2.0)

        self.assertTrue(motors.torqued(), "the torque is on, as LeRobot's connect leaves it")
        self.assertEqual(motors.present(), REST, "the arm holds where it was, never driven to the old goal")
        last_goal = [value for name, value in motors.writes if name == "goal_position"][-1]
        self.assertEqual(last_goal, REST)
        self.assertEqual(motors.writes[-1], ("torque_enable", True), "aimed where it is before the torque is on")
        robot.disconnect()

    def test_a_lost_goal_write_is_sent_again_and_one_that_never_lands_leaves_the_torque_off(self) -> None:
        # A goal write gets no reply, so a garbled one would leave a servo its old goal.
        for dropped, connects in ((2, True), (connect.GOAL_TRIES, False)):
            with self.subTest(dropped=dropped):
                clock = FakeClock()
                motors = FakeMotors(clock, calibration(), REST)
                for i, register in motors.registers.items():
                    register["goal_position"] = REST[motors.names[i]] + 400
                connect.install()
                robot = follower(motors)
                motors.drop_goal_writes = dropped
                if connects:
                    robot.connect(calibrate=False)
                    clock.advance(2.0)
                    self.assertEqual(motors.present(), REST)
                    self.assertTrue(motors.torqued())
                    robot.disconnect()
                else:
                    with self.assertRaisesRegex(ConnectionError, "torque stays off"):
                        robot.connect(calibrate=False)
                    clock.advance(2.0)
                    self.assertFalse(any(r["torque_enable"] for r in motors.registers.values()))
                    self.assertEqual(motors.present(), REST, "never driven to the old goal")


    def test_a_joint_resting_just_past_its_limit_is_pulled_back_to_the_limit(self) -> None:
        # On 2026-11-13 the folded elbow read 3035 against a Max_Position_Limit of 3031: the servo stored
        # a goal of 3035 as 3031, so the read-back never matched and every connect refused.
        for side, offset in (("max", connect.OVERSHOOT_MAX_STEPS), ("min", -connect.OVERSHOOT_MAX_STEPS)):
            with self.subTest(side=side):
                clock = FakeClock()
                calib = calibration()
                calib["elbow_flex"]["range_max"] = REST["elbow_flex"]
                calib["elbow_flex"]["range_min"] = REST["elbow_flex"]
                rest = dict(REST, elbow_flex=REST["elbow_flex"] + offset)
                motors = FakeMotors(clock, calib, rest)
                for i, register in motors.registers.items():
                    register["goal_position"] = REST[motors.names[i]] - 400  # old goals, inside every limit

                connect.install()
                robot = follower(motors, calib)
                robot.connect(calibrate=False)
                clock.advance(2.0)

                self.assertTrue(motors.torqued())
                self.assertEqual(motors.present(), REST, "only the elbow moved, back to its limit")
                self.assertEqual(motors.writes[-1], ("torque_enable", True))
                robot.disconnect()

    def test_a_joint_further_past_its_limit_leaves_the_torque_off_and_says_so(self) -> None:
        clock = FakeClock()
        calib = calibration()
        calib["elbow_flex"]["range_max"] = REST["elbow_flex"]
        rest = dict(REST, elbow_flex=REST["elbow_flex"] + connect.OVERSHOOT_MAX_STEPS + 1)
        motors = FakeMotors(clock, calib, rest)
        connect.install()
        robot = follower(motors, calib)
        with self.assertRaisesRegex(ConnectionError, r"elbow_flex at 3121, 21 steps \(1.8°\) past its limit 3100.*by hand"):
            robot.connect(calibrate=False)
        clock.advance(2.0)
        self.assertFalse(any(r["torque_enable"] for r in motors.registers.values()))
        self.assertEqual(motors.goals, [], "no goal written, which would turn the torque on")
        self.assertEqual(motors.present(), rest)


class CheckLimitsTest(unittest.TestCase):
    """`so101 check` reports what connecting will do with a follower joint past its limits."""

    def report_for(self, elbow: int):
        from types import SimpleNamespace

        from lerobot.motors import Motor, MotorCalibration, MotorNormMode, SerialMotorsBus

        from so_101 import check
        from so_101.report import Report

        calib = calibration()
        calib["elbow_flex"]["range_max"] = REST["elbow_flex"]
        motors = FakeMotors(FakeClock(), calib, dict(REST, elbow_flex=elbow))
        file = Path(tempfile.mkdtemp(prefix="so101-test-")) / "follower.json"
        file.write_text(json.dumps(calib))
        bus = SerialMotorsBus("/dev/fake", {n: Motor(c["id"], "sts3215", MotorNormMode.DEGREES) for n, c in calib.items()},
                              calibration={n: MotorCalibration(**c) for n, c in calib.items()})
        bus._open_bus = lambda: motors
        bus._connect(handshake=False)
        report = Report()
        check.check_calibration(report, SimpleNamespace(name="follower", calibration_file=file), bus)
        self.assertEqual(motors.goals, [], "check writes nothing to the motors")
        return [(r.status, r.detail, r.fix) for r in report.results]

    def test_inside_the_range_passes(self) -> None:
        [(status, detail, _)] = self.report_for(REST["elbow_flex"])
        self.assertEqual((status, detail), ("PASS", "calibration matches the motors (follower.json)"))

    def test_a_joint_just_past_its_limit_warns_of_the_pull_back(self) -> None:
        [(status, detail, fix)] = self.report_for(REST["elbow_flex"] + 4)
        self.assertEqual(status, "WARN")
        self.assertIn("elbow_flex at 3104, 4 steps (0.4°) past its limit 3100", detail)
        self.assertIn("pulls each back to its limit", fix)

    def test_a_joint_too_far_past_its_limit_fails_as_connecting_would(self) -> None:
        [(status, detail, fix)] = self.report_for(REST["elbow_flex"] + connect.OVERSHOOT_MAX_STEPS + 1)
        self.assertEqual(status, "FAIL")
        self.assertIn("elbow_flex at 3121, 21 steps", detail)
        self.assertIn("by hand", fix)


if __name__ == "__main__":
    unittest.main()
