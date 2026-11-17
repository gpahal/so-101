"""The stuck and event detector: the arm's own readings, one policy tick at a time, into gripper events and
stuck rules. Pure: it keeps its own counters and touches nothing, so tests, `so101 skills` and
`so101-failbench candidates` replay recorded readings through the same code every run's perception runs.

Feed it only the ticks the policy drives (`step`), with each tick's time, pose and whether the policy's
action reached the arm. A tick where the policy sent nothing (a cloud GPU's late chunk) still moves the
clock, but counts toward neither the hover and frozen rules nor the policy's driving time. Call
`resumed(moved, pose)` when the policy drives again after anything else (a pause, a playback, Teleop),
with whether the arm may have moved meanwhile and where it is now: the speed window and the rules'
running timers start again, because the pose jumped. After a move with the jaws open (at `closed_below`
or above, or above `hold_band`'s top after a carry or release), whatever was closed, carried or released is
undone (a playback opened the jaws, or a person put the block back), so
the next close starts a new grasp and nothing is inferred across the gap. When that undid a carry or
release, `no_carry` counts the policy's driving time from the handback, as from a new start.

The signals, in the order they can happen in one attempt:
- `close`: the gripper fell below `closed_below`.
- `carry`: after a close, the elbow moved `carry_elbow_deg` with the gripper still above
  `carry_open_above` (`how="elbow"`), or the gripper held steady in `hold_band` for `hold_s` (`how="hold"`).
- `regrasp`: a close reopened without a carry.
- `release`: after a carry, the gripper opened `release_above` past where it held and above `hold_band`'s
  top. Where it held is the median of its readings over the last `hold_s` in which they all stayed in
  `hold_band` and within `hold_steady` of each other (`_steady`), or its reading at the carry until it has
  held that long. A block settling in the jaws moves that, and jaws that open quickly don't, so a release
  that opens only a little past the band still counts. Jaws that open slower than about `hold_steady` per
  `hold_s` (1.3 a second) carry the hold with them, as a settling block does, so such a release counts
  only once the reading passes `release_above` over where the hold stopped. A reading inside the band,
  where a held block keeps the gripper, is never a release. Nothing after a release is a grasp: the arm is
  on its way back to rest.
- `rule`: a stuck rule fired, once per attempt each. All but `not_released` apply only before a carry.
"""

from collections import deque
from statistics import median
from dataclasses import dataclass, field

from so_101.grasp.thresholds import Thresholds

GRIPPER = "gripper.pos"
ELBOW = "elbow_flex.pos"
ARM = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos")

# A change to how the detector finds events from the same thresholds raises `skills.RUN_LABEL_VERSION`, since
# it can move a run's labels, and `skills.CUT_RULES_VERSION` too when it moves a demo's cut.


@dataclass(frozen=True)
class Signal:
    kind: str  # close, carry, regrasp, release or rule
    t: float  # the attempt's policy-clock seconds when it happened
    name: str | None = None  # the rule's name, or how a carry was seen
    detail: dict = field(default_factory=dict)


class Detector:
    def __init__(self, thresholds: Thresholds) -> None:
        self.p = thresholds
        self.fired: set[str] = set()
        self.closes = self.regrasps = 0
        self.carried = self.released = False
        self.policy_s = 0.0  # seconds the policy's action reached the arm
        self._no_carry_from = 0.0  # policy_s when no_carry began counting: 0, or a handback that undid a carry
        self._start: dict[str, float] | None = None  # the attempt's first pose
        self._last_t: float | None = None
        self._last_g: float | None = None
        self._window: deque[tuple[float, dict[str, float]]] = deque()
        self._crossings: deque[float] = deque()
        self._close: dict | None = None  # the open close: when, and the elbow then
        self._hold_value: float | None = None  # where the gripper is holding steady, and since when
        self._hold_since: float | None = None
        self._carry_hold: float | None = None  # where the gripper held the carried block
        self._held: deque[tuple[float, float]] = deque()  # the gripper's readings over the last hold_s
        self._hover_s = self._frozen_s = 0.0
        self.speed: float | None = None

    # --- Feeding it ------------------------------------------------------------------------------

    def resumed(self, moved: bool, pose: dict[str, float]) -> None:
        """The policy drives again after something else held the arm, or moved it (`moved`), leaving
        it at `pose`. With the jaws open after a move, the open close, the carry and the release are
        forgotten, and the gripper's last reading is this one, so reopened jaws aren't a regrasp. Open means
        at `closed_below` or above, or, after a carry or release, above `hold_band`'s top, where no block is
        held: the line a release crosses, so a narrow release (18.9 to 23.8) followed by a person undoing it
        re-arms too. The next close then needs the jaws opened past `closed_below` first, which a policy does
        anyway to fit the block. With them closed, the last reading stays the policy's: jaws a person closed
        count as a close.

        Forgetting a carry or a release also starts `no_carry` again, from the policy's driving time now:
        the block is back to be picked, as at the attempt's start, and the time before the handback was
        spent carrying, which the rule never counted, so it would otherwise fire on the first tick. A
        handback that undid only a close (a playback of a miss) leaves it counting from the start, or a
        policy that keeps missing would never trip it."""
        self._last_t = None
        self._window.clear()
        self._crossings.clear()
        self._hold_value = self._hold_since = None
        self._held.clear()
        self._hover_s = self._frozen_s = 0.0
        g, gp = pose[GRIPPER], self.p.gripper
        # Jaws above the hold band hold no block (a release crosses the same line), so a carry or release is
        # undone. With nothing carried, jaws a person closed below closed_below still count as a close.
        held = self.carried or self.released
        if moved and (g >= gp.closed_below or (held and g > gp.hold_band[1])):
            if held:
                self._no_carry_from = self.policy_s
            self.carried = self.released = False
            self._close = self._carry_hold = None
            self._last_g = g

    def step(self, t: float, pose: dict[str, float], drove: bool = True) -> list[Signal]:
        """One policy tick at `t` seconds with the arm at `pose`. Returns what happened on it."""
        p, g = self.p, pose[GRIPPER]
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if self._start is None:
            self._start = dict(pose)
        if drove:
            self.policy_s += dt
        self._window.append((t, pose))
        while len(self._window) > 2 and self._window[1][0] <= t - p.rules.speed_window_s:
            self._window.popleft()
        first_t, first = self._window[0]
        span = t - first_t
        self.speed = (max(abs(pose[j] - first[j]) for j in ARM) / span) if span >= p.rules.speed_window_s / 2 else None
        signals: list[Signal] = []
        closed_below = p.gripper.closed_below
        prev = self._last_g
        self._last_g = g
        if prev is not None and (prev < closed_below) != (g < closed_below):
            self._crossings.append(t)
        while self._crossings and self._crossings[0] < t - p.rules.chatter_within_s:
            self._crossings.popleft()
        if not self.released:
            signals += self._gripper(t, pose, g, prev)
        if not self.released:
            signals += self._rules(t, pose, g, dt, drove)
        return signals

    # --- Gripper events --------------------------------------------------------------------------

    def _gripper(self, t: float, pose: dict, g: float, prev: float | None) -> list[Signal]:
        gp, out = self.p.gripper, []
        if prev is not None and prev >= gp.closed_below > g and not self.carried:
            self.closes += 1
            self._close = {"t": t, "elbow": pose[ELBOW], "travel": 0.0}
            out.append(Signal("close", t, detail={"close": self.closes, "gripper": round(g, 1)}))
        lo, hi = gp.hold_band
        if lo <= g <= hi:
            if self._hold_value is None or abs(g - self._hold_value) > gp.hold_steady:
                self._hold_value, self._hold_since = g, t
        else:
            self._hold_value = self._hold_since = None
        steady = self._steady(t, g)
        if self._close is not None and not self.carried:
            self._close["travel"] = max(self._close["travel"], abs(pose[ELBOW] - self._close["elbow"]))
            if g > gp.carry_open_above and self._close["travel"] >= gp.carry_elbow_deg:
                out.append(self._carry(t, g, "elbow"))
            elif self._hold_since is not None and t - self._hold_since >= gp.hold_s:
                out.append(self._carry(t, g, "hold"))
            elif prev is not None and prev < gp.closed_below <= g:
                self.regrasps += 1
                travel = round(self._close["travel"], 1)
                self._close = None
                out.append(Signal("regrasp", t, detail={"regrasps": self.regrasps, "elbow_travel": travel}))
        elif self.carried:
            if steady is not None:
                self._carry_hold = steady
            if g > self._carry_hold + gp.release_above and g > hi:  # inside the band, a held block keeps it
                self.released = True
                out.append(Signal("release", t, detail={"gripper": round(g, 1), "held_at": round(self._carry_hold, 1)}))
        return out

    def _steady(self, t: float, g: float) -> float | None:
        """Where the gripper has held steady, with this tick's reading `g`: the median of its readings
        over the last `hold_s`, when every one is in `hold_band` and within `hold_steady` of the others.
        None while it hasn't held that long. The median, because the first readings of an opening are still
        within `hold_steady` of the hold: the reading itself would follow the jaws open, as far as the
        band's top, and an opening that stopped under `release_above` past that was never a release.
        Jaws that open slower than about `hold_steady` per `hold_s` still move it, as a settling block
        does. A release also needs the reading above the band's top (`_gripper`)."""
        gp = self.p.gripper
        self._held.append((t, g))
        while len(self._held) > 1 and self._held[1][0] <= t - gp.hold_s:
            self._held.popleft()
        if t - self._held[0][0] < gp.hold_s:
            return None
        readings = [reading for _, reading in self._held]
        low, high = min(readings), max(readings)
        if low < gp.hold_band[0] or high > gp.hold_band[1] or high - low > gp.hold_steady:
            return None
        return median(readings)

    def _carry(self, t: float, g: float, how: str) -> Signal:
        self.carried = True
        self._carry_hold = min(max(g, self.p.gripper.hold_band[0]), self.p.gripper.hold_band[1])
        travel = round(self._close["travel"], 1) if self._close else None
        self._close = None
        return Signal("carry", t, how, {"gripper": round(g, 1), "elbow_travel": travel})

    # --- Stuck rules -----------------------------------------------------------------------------

    def _rules(self, t: float, pose: dict, g: float, dt: float, drove: bool) -> list[Signal]:
        r, out = self.p.rules, []
        slow = self.speed is not None
        if not self.carried:
            far = max(abs(pose[j] - self._start[j]) for j in ARM)
            hovering = g > r.hover_open_above and slow and self.speed < r.hover_speed_below and far > r.hover_from_start_deg
            if drove:  # a tick the policy sent nothing neither counts nor resets: the arm stands still waiting
                self._hover_s = self._hover_s + dt if hovering else 0.0
                self._frozen_s = self._frozen_s + dt if slow and self.speed < r.frozen_speed_below else 0.0
        progress = self.progress()
        for name, value in progress.items():
            if value >= 1.0 and name not in self.fired:
                self.fired.add(name)
                out.append(Signal("rule", t, name, {"progress": round(value, 2), "policy_s": round(self.policy_s, 1),
                                                    "closes": self.closes, "regrasps": self.regrasps}))
        return out

    def progress(self) -> dict[str, float]:
        """How close each rule that still applies is to firing: 1 fires it. Rules already fired, and
        rules that no longer apply (before-carry rules after a carry, all of them after a release), are left out."""
        if self.released:
            return {}
        r = self.p.rules
        if self.carried:
            held = 0.0 if self._hold_since is None else self._last_t - self._hold_since
            values = {"not_released": held / r.not_released_s}
        else:
            values = {
                "regrasp_loop": self.regrasps / r.regrasps,
                "hover": self._hover_s / r.hover_s,
                "chatter": len(self._crossings) / r.chatter_crossings,
                "frozen": self._frozen_s / r.frozen_s,
                "no_carry": (self.policy_s - self._no_carry_from) / r.no_carry_s,
            }
        return {name: value for name, value in values.items() if name not in self.fired}
