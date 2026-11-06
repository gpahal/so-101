"""`lerobot-rollout`, with every attempt starting from the pose the demos start from.

LeRobot's return between attempts is a fixed 1-second move. `max_relative_target` caps each command
to a few degrees from where the arm is, so long moves stop short and nothing resends them, and the
next attempt then starts from that pose. Here the return continues until the arm has arrived. If it
can't arrive, the attempt waits while you move the arm by hand, because a policy started from the
wrong pose is an invalid test of it.
"""

import json
import math
import os
import sys
import time

START_POSE_ENV = "SO101_START_POSE"
TOLERANCE = 4.0  # per joint, in the arm's units: about the spread of the demos' start poses
SPEED = 45.0  # units per second, about half the demos' fast moves
SETTLE_S = 5.0
FPS = 30
HOLD_ARM = "Support the arm first: it goes limp and will drop."


class StopRun(Exception):
    """The operator stopped the run while it waited at the start pose."""


def positions(robot, keys) -> dict[str, float]:
    observation = robot.get_observation()
    return {key: float(observation[key]) for key in keys}


def farthest(present: dict[str, float], target: dict[str, float]) -> tuple[str, float]:
    key = max(target, key=lambda k: abs(present[k] - target[k]))
    return key, abs(present[key] - target[key])


def drive_to(robot, target: dict[str, float], sleep=time.sleep, clock=time.monotonic) -> bool:
    """Move smoothly to `target` and keep commanding it until every joint is within TOLERANCE."""
    start = positions(robot, target)
    _, distance = farthest(start, target)
    steps = max(math.ceil(distance / SPEED * FPS), 1)
    for step in range(1, steps + 1):
        robot.send_action({k: start[k] + (target[k] - start[k]) * step / steps for k in target})
        sleep(1 / FPS)
    deadline = clock() + SETTLE_S
    while True:
        # A cap on each command means one send may not reach the target, so keep resending it.
        robot.send_action(target)
        if farthest(positions(robot, target), target)[1] <= TOLERANCE:
            return True
        if clock() > deadline:
            return False
        sleep(1 / FPS)


def wait_for_key(events: dict[str, bool], stopping, sleep=time.sleep) -> None:
    """Wait for the right arrow. Esc or Ctrl+C stops the run instead."""
    events["exit_early"] = False
    while not events["exit_early"]:
        if stopping():
            raise StopRun
        sleep(0.05)
    events["exit_early"] = False
    if stopping():
        raise StopRun


def release(robot) -> None:
    robot.inner.bus.disable_torque()


def engage(robot) -> None:
    """Turn the motors back on where the arm is now: they'd otherwise jump to their last goal."""
    bus = robot.inner.bus
    bus.sync_write("Goal_Position", bus.sync_read("Present_Position", num_retry=2))
    bus.enable_torque()


def ensure_at_start(robot, target: dict[str, float], events: dict[str, bool], stopping, sleep=time.sleep) -> None:
    """Return only with the arm at `target`, moving it there or waiting while you do."""
    while not drive_to(robot, target, sleep=sleep):
        key, off = farthest(positions(robot, target), target)
        print(
            f"\nThe arm can't reach its start pose: {key} is {off:.0f} away. Right arrow turns the motors off so "
            f"you can move it by hand, Esc stops.\n{HOLD_ARM}",
            flush=True,
        )
        wait_for_key(events, stopping, sleep)
        release(robot)
        print("Motors off. Move the arm to its start pose, then press right arrow.", flush=True)
        wait_for_key(events, stopping, sleep)
        engage(robot)
    events["exit_early"] = False
    events["rerecord_episode"] = False


def install() -> None:
    from lerobot.rollout.strategies.core import RolloutStrategy
    from lerobot.rollout.strategies.episodic import EpisodicStrategy

    configured = json.loads(os.environ[START_POSE_ENV]) if os.environ.get(START_POSE_ENV) else None

    def target_for(hw) -> dict[str, float] | None:
        return configured or hw.initial_position

    def return_to_initial_position(hw, duration_s: float = 3.0, fps: int = 50) -> bool:
        target = target_for(hw)
        if target is None or hw.robot_wrapper.hardware_failure is not None:
            return False
        try:
            arrived = drive_to(hw.robot_wrapper, target)
        except Exception as e:
            print(f"Could not return to the start pose: {e}", flush=True)
            return False
        if not arrived:
            print("The arm stopped short of its start pose.", flush=True)
        return arrived

    run = EpisodicStrategy.run
    start_episode = EpisodicStrategy._start_episode

    def run_stoppable(self, ctx) -> None:
        self._so101_ctx = ctx
        try:
            run(self, ctx)
        except StopRun:
            print("Stopped at the start pose.", flush=True)

    def start_at_pose(self, robot, timer) -> None:
        ctx = self._so101_ctx
        target = target_for(ctx.hardware)
        if target is not None:
            events = self._events
            print("Moving to the start pose...", flush=True)
            ensure_at_start(
                robot, target, events, lambda: events["stop_recording"] or ctx.runtime.shutdown_event.is_set()
            )
            print("At the start pose. Starting the attempt.", flush=True)
        start_episode(self, robot, timer)

    RolloutStrategy.return_to_initial_position = staticmethod(return_to_initial_position)
    EpisodicStrategy.run = run_stoppable
    EpisodicStrategy._start_episode = start_at_pose


def main() -> None:
    from lerobot.scripts.lerobot_rollout import main as rollout

    install()
    sys.exit(rollout())
