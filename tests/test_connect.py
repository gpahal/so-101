"""Connecting the follower holds it where it is, whatever goal its servos last got (`so_101.connect`)."""

import json
import tempfile
import unittest
from pathlib import Path

from fakes import FakeClock, FakeMotors, calibration

from so_101 import connect

REST = {"shoulder_pan": 2048, "shoulder_lift": 900, "elbow_flex": 3100,
        "wrist_flex": 2600, "wrist_roll": 2048, "gripper": 1000}


def follower(motors: FakeMotors):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    directory = Path(tempfile.mkdtemp(prefix="so101-test-"))
    (directory / "follower.json").write_text(json.dumps(calibration()))
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


if __name__ == "__main__":
    unittest.main()
