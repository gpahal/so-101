"""Rewind: the open attempt's history of measured follower poses, played backwards.

The control loop adds the follower's measured pose to the open attempt's `History` on every tick in
which something drives it (`DRIVEN`). `Rewind` plays that history back newest first, at its original
timing, one goal per tick; the controller mirrors each goal to a connected leader (`leader.Mirror`).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from so_101 import rollout
from so_101.control.states import State

REWIND_SETTLE_S = 1.0  # at a rewind's end, how long the loop waits for the joints to arrive
TIMING_SLACK_S = 1e-3  # tick timing jitter that still counts as a whole tick of the history
# The states in which something drives the follower. Their ticks go into the rewind history. Paused
# and Handover ticks, where the follower holds, don't, so a long pause never uses up a rewind; nor do
# Rewinding's own ticks, so a second Rewind goes further back instead of undoing the first.
DRIVEN = frozenset({State.STARTING, State.AUTONOMOUS, State.TELEOP, State.RETURNING})


@dataclass(frozen=True)
class Moment:
    seconds: float  # since the moment before it in the history; 0 for the first
    pose: dict[str, float]  # the follower's measured pose, gripper included
    frame: int  # the episode frame it was measured in


@dataclass
class History:
    """The follower's measured poses in the open attempt, oldest first: what Rewind plays back."""

    moments: list[Moment] = field(default_factory=list)
    seconds: float = 0.0  # the motion it holds: the time from the oldest moment to the newest
    rewound: bool = False  # the newest moment is where a rewind ended, with no driving since

    def add(self, seconds: float, pose: dict[str, float], frame: int) -> None:
        seconds = seconds if self.moments else 0.0
        self.moments.append(Moment(seconds, pose, frame))
        self.seconds += seconds
        self.rewound = False

    def pop(self) -> Moment:
        moment = self.moments.pop()
        self.seconds = self.seconds - moment.seconds if len(self.moments) > 1 else 0.0
        return moment

    @property
    def newest(self) -> Moment:
        return self.moments[-1]


class Rewind:
    """Play the history backwards, newest first, at its original timing, one goal per control tick.

    Each tick it drops the moments whose time has come and sends the pose of the one left on top, so
    the arm retraces its measured path, gripper included. It stops after `seconds` of motion, or at
    the attempt's first moment. Then it keeps sending that pose until every joint is within
    `rollout.TOLERANCE`, or REWIND_SETTLE_S passes. The moment on top stays in the history: it is
    where the arm is, and where the next rewind starts.
    """

    def __init__(self, history: History, seconds: float, clock: Callable[[], float]) -> None:
        self.history = history
        self.budget = seconds  # seconds of motion still to undo
        self.owed = 0.0  # seconds played since the last moment dropped
        self.undone_s = 0.0
        self.clock = clock
        self.deadline: float | None = None
        # Frames before this one had their actions undone, from `start_frame` on. The newest moment's
        # own action moved the arm on from it, unless a rewind left the arm there.
        self.end_frame = history.newest.frame + (0 if history.rewound else 1)

    def extend(self, seconds: float) -> None:
        self.budget += seconds
        self.deadline = None

    @property
    def target(self) -> dict[str, float]:
        return self.history.newest.pose

    @property
    def start_frame(self) -> int:
        """The frame whose measured pose the arm goes back to."""
        return self.history.newest.frame

    def _due(self) -> bool:
        """Whether the moment on top is still to be undone: it isn't the first, and the budget covers it."""
        moments = self.history.moments
        return len(moments) > 1 and self.budget >= moments[-1].seconds - TIMING_SLACK_S

    def step(self, robot, present: dict[str, float], seconds: float) -> bool | None:
        """`seconds` have passed since the last tick. Returns None while rewinding, True once
        arrived, False if the arm stopped short."""
        if self._due():
            self.owed += seconds
            while self._due() and self.owed >= self.history.newest.seconds - TIMING_SLACK_S:
                moment = self.history.pop()
                self.owed -= moment.seconds
                self.budget -= moment.seconds
                self.undone_s += moment.seconds
            robot.send_action(self.target)
            return None
        self.owed = 0.0
        if self.deadline is None:
            self.deadline = self.clock() + REWIND_SETTLE_S
        robot.send_action(self.target)
        if rollout.farthest(present, self.target)[1] <= rollout.TOLERANCE:
            return True
        if self.clock() > self.deadline:
            return False
        return None


def available(history: History, rewind: Rewind | None) -> float:
    """Seconds of motion a Rewind now could still undo, beyond what `rewind`, if running, already will."""
    scheduled = rewind.budget if rewind is not None else 0.0
    return max(history.seconds - scheduled, 0.0)


def refusal(history: History, rewind: Rewind | None, attempt: int) -> str | None:
    """Why Rewind has nothing to play back in attempt number `attempt`, or None when it has."""
    if available(history, rewind) > TIMING_SLACK_S:
        return None
    if rewind is not None and history.seconds > TIMING_SLACK_S:
        return f"The rewind already goes back to where attempt {attempt} started."
    return f"Nothing left to rewind: the arm is back where attempt {attempt} started."


def finish(rewind: Rewind, present: dict[str, float], cause: str) -> dict:
    """End `rewind`, with the arm at `present`, for `cause`: the history's newest moment is where it
    ended. Returns the `rewind` event's fields: what it undid."""
    rewind.history.rewound = True
    joint, gap = rollout.farthest(present, rewind.target)
    return {"start_frame": rewind.start_frame, "end_frame": rewind.end_frame, "seconds": round(rewind.undone_s, 3),
            "cause": cause, "farthest": joint.removesuffix(".pos"), "gap": round(gap, 2)}
