"""`so101 run`'s move to the demos' start pose before each attempt (`rollout.ensure_at_start`)."""

import unittest

from so_101 import rollout


class Arm:
    """An arm that reaches each goal at once."""

    def __init__(self, pose: dict[str, float]) -> None:
        self.pose = dict(pose)

    def get_observation(self) -> dict[str, float]:
        return dict(self.pose)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.pose.update(action)
        return action


class StartPoseTest(unittest.TestCase):
    def test_esc_on_the_way_to_the_start_pose_stops_the_run_instead_of_starting_an_attempt(self) -> None:
        # LeRobot's Esc sets both flags. Its policy loop reads only exit_early, so clearing that flag
        # on arrival without a stop ran a whole attempt.
        arm = Arm({"elbow_flex.pos": 0.0})
        events = {"stop_recording": False, "exit_early": False, "rerecord_episode": False}

        def sleep(_: float) -> None:
            events["stop_recording"] = events["exit_early"] = True  # Esc during the first step

        with self.assertRaises(rollout.StopRun):
            rollout.ensure_at_start(arm, {"elbow_flex.pos": 40.0}, events, lambda: events["stop_recording"],
                                    sleep=sleep)
        self.assertLess(arm.pose["elbow_flex.pos"], 40.0, "the move stopped where it was")

    def test_without_a_stop_it_arrives_and_clears_the_flags(self) -> None:
        arm = Arm({"elbow_flex.pos": 0.0})
        events = {"stop_recording": False, "exit_early": True, "rerecord_episode": True}
        rollout.ensure_at_start(arm, {"elbow_flex.pos": 40.0}, events, lambda: False, sleep=lambda _: None)
        self.assertEqual(arm.pose["elbow_flex.pos"], 40.0)
        self.assertEqual((events["exit_early"], events["rerecord_episode"]), (False, False))


if __name__ == "__main__":
    unittest.main()
