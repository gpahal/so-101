"""`settings.load()` on a so101.toml in a temporary folder."""

import unittest

from lerobot.robots.utils import ensure_safe_goal_position
from test_serve_preflight import ARMS, ConfigTest

from so_101 import settings, tasks


class TeleopTest(ConfigTest):
    def test_a_whole_number_max_relative_target_is_one_lerobot_accepts(self) -> None:
        # TOML reads `10` as an int, and LeRobot's cap takes only a float or a dict: an int raises
        # TypeError at the live test's first step, which passes the value to LeRobot directly.
        self.config.write_text(ARMS + "\n[teleop]\nmax_relative_target = 10\n")
        cap = settings.load().teleop.max_relative_target
        self.assertEqual(ensure_safe_goal_position({"elbow_flex": (25.0, 0.0)}, cap), {"elbow_flex": 10.0})

    def test_a_negative_or_unusable_max_relative_target_is_refused(self) -> None:
        # LeRobot clamps a goal to within the cap of where the arm is: a cap of -10 moves an unchanged goal by 10.
        self.assertEqual(ensure_safe_goal_position({"elbow_flex": (20.0, 20.0)}, -10.0), {"elbow_flex": 30.0})
        for value in ("-10", "nan", "inf", '"ten"', "true"):
            with self.subTest(value):
                self.config.write_text(ARMS + f"\n[teleop]\nmax_relative_target = {value}\n")
                with self.assertRaisesRegex(settings.ConfigError, "max_relative_target"):
                    settings.load()
        self.config.write_text(ARMS + "\n[teleop]\nmax_relative_target = 0\n")
        self.assertEqual(settings.load().teleop.max_relative_target, 0.0, "0 turns the limit off")


class RunTest(ConfigTest):
    def test_episodes_is_a_whole_number_of_one_or_more(self) -> None:
        self.config.write_text(ARMS + "\n[run]\nepisodes = 5\n")
        self.assertEqual(settings.load().run.episodes, 5)
        self.config.write_text(ARMS)
        self.assertEqual(settings.load().run.episodes, 1)
        for value in ("0", "-1", "1.5", '"3"', "true"):
            with self.subTest(value):
                self.config.write_text(ARMS + f"\n[run]\nepisodes = {value}\n")
                with self.assertRaisesRegex(settings.ConfigError, r"\[run\] episodes"):
                    settings.load()


class TasksTest(ConfigTest):
    TASK = '[test]\nprompt = "Put the block in the jar"\n'

    def test_a_reset_time_that_is_not_finite_seconds_of_0_or_more_is_refused(self) -> None:
        # `so101 run` waits reset_time_s between attempts: nan or inf would break the control page's strict JSON status.
        for value in ("nan", "inf", "-1", '"ten"', "true"):
            with self.subTest(value):
                (self.root / "tasks.toml").write_text(self.TASK + f"reset_time_s = {value}\n")
                with self.assertRaisesRegex(settings.ConfigError, "reset_time_s"):
                    tasks.load()

    def test_a_reset_time_of_0_or_more_is_kept(self) -> None:
        for value, expected in (("0", 0), ("12.5", 12.5)):
            (self.root / "tasks.toml").write_text(self.TASK + f"reset_time_s = {value}\n")
            self.assertEqual(tasks.load()["test"].reset_time_s, expected)
        (self.root / "tasks.toml").write_text(self.TASK)
        self.assertEqual(tasks.load()["test"].reset_time_s, 10, "the default")


class RemoteTest(ConfigTest):
    def test_pi05_pins_its_unused_action_columns_unless_the_file_turns_it_off(self) -> None:
        section = '\n[remote.pi05]\njob_target = "l4x1"\njob_timeout = "1h"\n'
        self.config.write_text(ARMS + section)
        self.assertTrue(settings.load().remote["pi05"].pin_unused_actions)
        self.config.write_text(ARMS + section + "pin_unused_actions = false\n")
        self.assertFalse(settings.load().remote["pi05"].pin_unused_actions)

    def test_a_pin_setting_that_is_not_true_or_false_is_refused(self) -> None:
        for value in ('"yes"', "1", "0"):
            with self.subTest(value):
                self.config.write_text(
                    ARMS + f'\n[remote.pi05]\njob_target = "l4x1"\njob_timeout = "1h"\npin_unused_actions = {value}\n')
                with self.assertRaisesRegex(settings.ConfigError, "pin_unused_actions"):
                    settings.load()


if __name__ == "__main__":
    unittest.main()
