"""Scripted moves of the follower, one control tick at a time, which the control center steps.

`max_relative_target` caps each command to a few degrees from where the arm is, so one long command
stops short. A `Move` steps from where the arm is and keeps commanding its target until the arm has
arrived, so `so101 serve` and `run` reach the demos' start pose and the rest pose, and a Pause can cut
the move short.
"""

import math
import time

TOLERANCE = 4.0  # per joint, in the arm's units: about the spread of the demos' start poses
SPEED = 45.0  # units per second, about half the demos' fast moves
SETTLE_S = 5.0
FPS = 30


def positions(robot, keys) -> dict[str, float]:
    observation = robot.get_observation()
    return {key: float(observation[key]) for key in keys}


def farthest(present: dict[str, float], target: dict[str, float]) -> tuple[str, float]:
    key = max(target, key=lambda k: abs(present[k] - target[k]))
    return key, abs(present[key] - target[key])


class Move:
    """A smooth move to `target`, one command per control tick.

    It steps from where the arm is at about SPEED, then keeps commanding `target` until every joint
    is within TOLERANCE. The control loop steps one per tick, so a pause can cut it short.
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
