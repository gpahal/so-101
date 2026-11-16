"""Perception: the gripper rules over each tick the model drove, for every layer above to read.

One `supervisor.detector.Detector` per attempt, with the program's thresholds, fed the measured joints on
each tick the model drove (its Output's driver is POLICY or WAITING), at the model's own clock: seconds
it drove, so a pause or a layer's playback never counts. Each tick's signals and the detector's state go
up the stack in the Output (`signals`, `readings`), so the layers above read one detector and can't
disagree about whether a carry happened.

It logs a `grasp` event for each close, carry, regrasp and release, and a `rule` event when a stuck rule
fires, each with the model's seconds, so every run's events mark its grasps whatever model ran. On a
handback it restarts the detector's timers, since something else held or moved the arm meanwhile.
"""

from dataclasses import replace

from so_101.policy.base import POLICY, WAITING, Handback, Layer, LayerContext, Output, Readings, Tick

MAX_STEP_S = 0.5  # a longer gap between ticks (a stalled loop) counts as this much of the model's time


class Perception(Layer):
    def __init__(self, inner, ctx: LayerContext) -> None:
        super().__init__(inner, ctx, "perception")
        self.detector = None
        self.t = 0.0
        self._last: float | None = None

    def begin_episode(self, number: int) -> None:
        from so_101.supervisor.detector import Detector

        self.detector, self.t, self._last = Detector(self.ctx.program), 0.0, None
        self.inner.begin_episode(number)

    def end_episode(self) -> None:
        self.inner.end_episode()
        self.detector = None

    def run(self, handback: Handback) -> None:
        self.inner.run(handback)
        self._last = None
        if self.detector is not None:
            self.detector.resumed()

    def pause(self) -> None:
        self.inner.pause()
        self._last = None

    def step(self, robot, tick: Tick) -> Output:
        out = self.inner.step(robot, tick)
        if self.detector is None or out.driver not in (POLICY, WAITING) or self.broken is not None:
            return out
        found = []
        self.safely(self._watch, tick, out.driver == POLICY, found)
        return out if self.broken is not None else replace(out, signals=tuple(found), readings=self.readings())

    def _watch(self, tick: Tick, drove: bool, found: list) -> None:
        dt = 0.0 if self._last is None else min(max(tick.now - self._last, 0.0), MAX_STEP_S)
        self._last = tick.now
        self.t += dt
        for signal in self.detector.step(self.t, tick.pose, drove):
            found.append(signal)
            kind = "rule" if signal.kind == "rule" else "grasp"
            self.ctx.event(kind, **({"name": signal.name} if signal.name else {}),
                           **({} if kind == "rule" else {"grasp": signal.kind}), policy_t=round(signal.t, 2),
                           **signal.detail)

    def readings(self) -> Readings:
        d = self.detector
        return Readings(round(self.t, 3), tuple(d.near()), d.closes, d.regrasps, d.carried, d.released,
                        tuple(sorted(d.fired)))
