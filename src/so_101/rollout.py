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


class Move:
    """A smooth move to `target`, one command per control tick.

    It steps from where the arm is at about SPEED, then keeps commanding `target` until every joint
    is within TOLERANCE. `drive_to` runs one to the end; `so101 serve` steps one per tick of its
    control loop, so a pause can cut it short.
    """

    def __init__(self, target: dict[str, float], fps: float = FPS, clock=time.monotonic) -> None:
        self.target = target
        self.fps = fps
        self.clock = clock
        self.start: dict[str, float] | None = None
        self.steps = 0
        self.step_index = 0
        self.deadline: float | None = None

    def step(self, robot, present: dict[str, float] | None = None) -> bool | None:
        """Send the next command. Returns None while moving, True once arrived, False if stopped short.

        `present` is the arm's position measured this tick; without it, the robot is read.
        """
        if self.start is None:
            self.start = present if present is not None else positions(robot, self.target)
            _, distance = farthest(self.start, self.target)
            self.steps = max(math.ceil(distance / SPEED * self.fps), 1)
        if self.step_index < self.steps:
            self.step_index += 1
            fraction = self.step_index / self.steps
            robot.send_action({k: self.start[k] + (self.target[k] - self.start[k]) * fraction for k in self.target})
            return None
        if self.deadline is None:
            self.deadline = self.clock() + SETTLE_S
        # A cap on each command means one send may not reach the target, so keep resending it.
        robot.send_action(self.target)
        if farthest(present if present is not None else positions(robot, self.target), self.target)[1] <= TOLERANCE:
            return True
        if self.clock() > self.deadline:
            return False
        return None


def drive_to(robot, target: dict[str, float], sleep=time.sleep, clock=time.monotonic) -> bool:
    """Move smoothly to `target` and keep commanding it until every joint is within TOLERANCE."""
    move = Move(target, clock=clock)
    while (arrived := move.step(robot)) is None:
        sleep(1 / FPS)
    return arrived


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
    from so_101 import svtshim

    svtshim.ensure()  # `so101 run` starts this with the fix loaded; run directly, it restarts itself
    svtshim.require()  # it records the attempts with SVT-AV1
    from lerobot.scripts.lerobot_rollout import main as rollout

    install()
    sys.exit(rollout())
