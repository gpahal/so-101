"""The leader arm as the control loop uses it: connected while the arm rests or is paused, driven to
the follower's pose before Teleop, and mirroring the follower's goals during a rewind.

The leader is optional. `Leader` connects it when the device watch sees its board and its
calibration file exists, retries a failed connect with a growing wait, and turns any error from its
bus into `LeaderLost`. `Handover` drives it to the follower's measured pose one step per tick, so
Pause can cut it short; the controller frees it only once every joint is within `rollout.TOLERANCE`.
`Mirror` sends it the follower's goals during a rewind. `Leader` keeps the handover or mirror running, and
dropping the leader ends either.

Losing the leader in Handover or Teleop pauses the follower where it is (`states.Event.LEADER_LOST`).
Its pose is recorded every tick of an attempt while it's connected; a read that fails elsewhere only
drops it (`Leader.pose_for_frame`).

Every goal the leader gets through `send_feedback` is capped by the follower's `max_relative_target`
against the leader's pose measured that tick (`capped`), as LeRobot caps the follower's. A leader
held still by a hand otherwise falls behind a goal that keeps moving, and jumps to it when let go.
"""

from collections.abc import Callable

from so_101 import rollout
from so_101.control.states import LEADER_MISSING
from so_101.control.watch import INTERVAL_S

# Each failed leader connect doubles the wait before the next try, from the watch's 2 s (so 4 s,
# 8 s, 16 s), up to this. A connect blocks the loop for up to a second, so a leader that's listed but
# broken mustn't be retried every 2 s. Unplugging it starts the wait over.
LEADER_RETRY_MAX_S = 30.0


def capped(goal: dict[str, float], measured: dict[str, float],
           max_step: float | dict[str, float] | None) -> dict[str, float]:
    """`goal`, with each joint at most `max_step` from `measured`. `max_step` is the follower's
    `max_relative_target`: one number, or one per motor name, or None for no cap."""
    if max_step is None:
        return dict(goal)
    sent = {}
    for key, value in goal.items():
        step = max_step[key.removesuffix(".pos")] if isinstance(max_step, dict) else max_step
        sent[key] = measured[key] + max(-step, min(step, value - measured[key]))
    return sent


class LeaderLost(Exception):
    """The leader stopped answering, or was never there, while it was needed. The message is the reason."""


class Leader:
    """The leader's connection: the connected LeRobot leader (`arm`), or why there is none, and the
    handover or rewind mirror driving it, if one is.

    `make()` makes a new, unconnected LeRobot leader (SO101Leader) in the follower's units.
    `say(message)` tells the operator. `max_step` caps each goal sent to it (see `capped`)."""

    def __init__(self, make: Callable[[], object], clock: Callable[[], float],
                 say: Callable[[str], None], max_step: float | dict[str, float] | None) -> None:
        self.make = make
        self.clock = clock
        self.say = say
        self.max_step = max_step
        self.arm = None  # the connected leader, or None
        self.problem: str | None = None  # why it isn't connected, when known
        self.tried_at: float | None = None
        self.wait_s = INTERVAL_S  # how long after a try the next one may come
        self.handover: Handover | None = None  # the handover running, if any
        self.mirror: Mirror | None = None  # the rewind's goals, mirrored to the leader while it runs

    @property
    def connected(self) -> bool:
        return self.arm is not None

    def call(self, call: Callable[[], object]):
        """Run one leader bus call. Any error means the leader is lost."""
        if self.arm is None:
            raise LeaderLost(self.problem or LEADER_MISSING)
        try:
            return call()
        except Exception as error:
            raise LeaderLost(f"{LEADER_MISSING} It stopped answering: {error}") from error

    def read(self) -> dict[str, float]:
        pose = self.call(lambda: self.arm.get_action())
        return {key: float(value) for key, value in pose.items()}

    def hold(self) -> dict[str, float]:
        """Turn the leader's motors on where it is, so it stays put when let go. Returns that pose.

        The motors are aimed where the leader is before they turn on: they'd otherwise jump to the last
        goal they were sent, which may be from long ago."""
        pose = self.read()
        self.call(lambda: self.arm.send_feedback(pose))
        self.call(lambda: self.arm.enable_torque())
        return pose

    def free(self) -> None:
        """Turn the leader's motors off, so a human can move it."""
        self.call(lambda: self.arm.disable_torque())

    def send(self, goal: dict[str, float], measured: dict[str, float]) -> dict[str, float]:
        """Send `goal`, capped against `measured`, the leader's pose this tick. Returns the goal sent."""
        sent = capped(goal, measured, self.max_step)
        self.call(lambda: self.arm.send_feedback(sent))
        return sent

    def watch(self, unplugged: str | None, may_connect: bool) -> str | None:
        """Connect the leader when the device watch sees it, `may_connect` (the arm is resting or
        paused), and the wait since its last try is over (see LEADER_RETRY_MAX_S). `unplugged` is
        the watch's reason when it doesn't see it. Returns why the connected leader must be dropped,
        when the watch has lost it."""
        if self.arm is not None:
            if unplugged:
                return f"{LEADER_MISSING} Plug it back in: it connects again while the arm rests or is paused."
            return None
        if unplugged:
            self.problem = f"{LEADER_MISSING} Plug it in: it connects while the arm rests or is paused."
            self.wait_s = INTERVAL_S  # a replugged leader may well connect: try it soon
            return None
        if not may_connect:
            return None
        if self.tried_at is not None and self.clock() - self.tried_at < self.wait_s:
            return None
        self.tried_at = self.clock()
        if self.connect():
            self.wait_s = INTERVAL_S
        else:
            self.wait_s = min(self.wait_s * 2, LEADER_RETRY_MAX_S)
        return None

    def connect(self) -> bool:
        """Connect the leader. Returns whether it connected."""
        try:
            leader = self.make()  # finds its port, which fails when it's unplugged
        except Exception as error:
            self.problem = f"{LEADER_MISSING} {error}"
            return False
        if not leader.calibration_fpath.is_file():
            self.problem = (f"{LEADER_MISSING} It has no calibration file at {leader.calibration_fpath}: "
                            "run `uv run so101 calibrate leader`.")
            return False
        try:
            leader.connect(calibrate=False)  # turns its motors off: it hangs free until a handover
            if not leader.is_calibrated:
                raise RuntimeError(f"its motors' calibration doesn't match {leader.calibration_fpath}; "
                                   "run `uv run so101 calibrate leader`")
        except Exception as error:
            try:
                leader.disconnect()
            except Exception:
                pass
            self.problem = f"{LEADER_MISSING} Connecting failed: {error}."
            return False
        self.arm, self.problem = leader, None
        self.say("Leader arm connected: Teleop can take over.")
        return True

    def ok(self, call: Callable[[], object]) -> bool:
        """Run `call`, which uses the leader. If the leader is lost meanwhile, drop it if it's still
        connected and return False: whatever needed it carries on without it."""
        try:
            call()
        except LeaderLost as lost:
            if self.connected:
                self.drop(str(lost))
            return False
        return True

    def pose_for_frame(self) -> dict[str, float] | None:
        """The leader's pose, for the frame, outside Handover and Teleop. A leader that fails to
        answer is dropped as a lost one is, but nothing it drives pauses: the frame records NaN."""
        try:
            return self.read()
        except LeaderLost as lost:
            self.drop(str(lost))
            return None

    def start_mirror(self, follower_goal: dict[str, float], fps: float) -> None:
        """At a rewind's start, turn the leader's motors on where it is, if it's connected."""
        if not self.connected:
            return
        mirror = Mirror(self, fps)
        self.mirror = mirror if self.ok(lambda: mirror.start(follower_goal)) else None

    def follow_mirror(self, follower_goal: dict[str, float]) -> dict[str, float] | None:
        """Read the leader, and send it toward the follower's goal this tick (see `Mirror`). Returns
        its pose, for the frame, or None when it isn't mirroring. A leader that fails is dropped; the
        rewind carries on without it."""
        if not self.connected or self.mirror is None:
            return None
        try:
            pose = self.read()
            self.mirror.follow(follower_goal, pose)
        except LeaderLost as lost:
            self.drop(str(lost))
            return None
        return pose

    def drop(self, reason: str | None) -> None:
        """Forget the leader and close its port, ending any handover or mirror. Its motors turn off
        if it still answers."""
        self.handover = self.mirror = None
        leader, self.arm = self.arm, None
        self.problem = reason
        self.tried_at = self.clock()
        if leader is None:
            return
        try:
            leader.disconnect()
        except Exception:
            pass  # the port is closed even when turning the motors off fails
        if reason:
            self.say(reason)

    def reason(self, may_connect: bool) -> str | None:
        """Why Teleop can't use the leader, or None when it is connected. `may_connect` is whether
        the arm rests or is paused, when the leader connects."""
        if self.arm is not None:
            return None
        if self.problem:
            return self.problem
        if not may_connect:
            return f"{LEADER_MISSING} It connects while the arm rests or is paused: pause first."
        return LEADER_MISSING


class Handover:
    """Drive the leader to the follower's measured pose, one step per control tick.

    It does what LeRobot's `teleop_smooth_move_to` does (motors on, then a straight line from where
    the leader is), but one step per tick instead of sleeping between steps, so the control loop keeps
    holding the follower and Pause can cut it short. The line runs at `rollout.Move`'s speed. Then it
    keeps commanding the follower's pose, measured afresh every tick, until every joint is within
    `rollout.TOLERANCE`, or `rollout.SETTLE_S` passes. The caller frees the leader on arrival.

    Each goal is capped (`Leader.send`) against the leader's pose measured that tick, so a leader held
    still never gets a goal more than one step away, however far the line has run on. Any error from
    the leader's bus raises LeaderLost.
    """

    def __init__(self, leader: Leader, fps: float, clock: Callable[[], float]) -> None:
        self.leader = leader
        self.fps = fps
        self.clock = clock
        self.move: rollout.Move | None = None
        self.pose: dict[str, float] | None = None  # the leader's pose measured this tick

    def step(self, follower: dict[str, float]) -> bool | None:
        """`follower` is the follower's pose measured this tick. Returns None while moving, True once
        the leader is there, False if it stopped short."""
        if self.move is None:
            self.pose = pose = self.leader.hold()
            self.move = rollout.Move(dict(follower), self.fps, self.clock)
        else:
            self.pose = pose = self.leader.read()
            if self.move.step_index >= self.move.steps:
                self.move.target = dict(follower)  # the follower's pose now, not when the handover began
        return self.move.step(self, pose)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """`rollout.Move` sends each goal here: on to the leader, capped against its pose this tick."""
        return self.leader.send(action, self.pose)

    def gap(self, follower: dict[str, float]) -> tuple[str, float]:
        """The joint farthest from the follower's pose, and how far."""
        return rollout.farthest(self.pose or follower, follower)


class Mirror:
    """A rewind's follower goals, sent on to the leader one per tick, so a hand on it feels the rewind.

    Each joint moves as far as the follower's goal did, plus `rollout.SPEED`, so a leader in line
    with the follower tracks it exactly and one that starts away from it closes the gap at a scripted
    move's speed. The leader's cap (`Leader.send`) keeps each goal near where the leader is."""

    def __init__(self, leader: Leader, fps: float) -> None:
        self.leader = leader
        self.extra = rollout.SPEED / fps
        self.goal: dict[str, float] | None = None  # the last goal sent to the leader
        self.mirrored: dict[str, float] | None = None  # the follower goal that goal mirrored

    def start(self, follower_goal: dict[str, float]) -> None:
        """Turn the leader's motors on where it is. Raises LeaderLost."""
        self.goal = self.leader.hold()
        self.mirrored = dict(follower_goal)

    def follow(self, follower_goal: dict[str, float], measured: dict[str, float]) -> None:
        """Send the leader toward the follower's goal this tick. `measured` is the leader's pose this
        tick. Raises LeaderLost."""
        mirrored = {}
        for key, last in self.goal.items():
            limit = abs(follower_goal[key] - self.mirrored[key]) + self.extra
            mirrored[key] = last + max(-limit, min(limit, follower_goal[key] - last))
        self.goal, self.mirrored = self.leader.send(mirrored, measured), dict(follower_goal)
