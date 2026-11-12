"""`settings.load()` on a so101.toml in a temporary folder."""

import unittest

from lerobot.robots.utils import ensure_safe_goal_position
from test_serve_preflight import ARMS, ConfigTest

from so_101 import settings


class TeleopTest(ConfigTest):
    def test_a_whole_number_max_relative_target_is_one_lerobot_accepts(self) -> None:
        # TOML reads `10` as an int, and LeRobot's cap takes only a float or a dict: it raised
        # TypeError(10) on the live test's first step, which passes the value to LeRobot directly.
        self.config.write_text(ARMS + "\n[teleop]\nmax_relative_target = 10\n")
        cap = settings.load().teleop.max_relative_target
        self.assertEqual(ensure_safe_goal_position({"elbow_flex": (25.0, 0.0)}, cap), {"elbow_flex": 10.0})

    def test_a_negative_or_unusable_max_relative_target_is_refused(self) -> None:
        # LeRobot clamps a goal to within the cap of where the arm is: -10 moved an unchanged goal 10.
        self.assertEqual(ensure_safe_goal_position({"elbow_flex": (20.0, 20.0)}, -10.0), {"elbow_flex": 30.0})
        for value in ("-10", "nan", "inf", '"ten"', "true"):
            with self.subTest(value):
                self.config.write_text(ARMS + f"\n[teleop]\nmax_relative_target = {value}\n")
                with self.assertRaisesRegex(settings.ConfigError, "max_relative_target"):
                    settings.load()
        self.config.write_text(ARMS + "\n[teleop]\nmax_relative_target = 0\n")
        self.assertEqual(settings.load().teleop.max_relative_target, 0.0, "0 turns the limit off")


if __name__ == "__main__":
    unittest.main()
