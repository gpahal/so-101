"""Perception: the grasp detector over each tick the model drove, for every layer above to read.

One `grasp.detector.Detector` per attempt, with `ctx.grasp`'s thresholds, fed the measured joints on each
tick whose Output's driver is POLICY or WAITING, through `grasp.replay.LiveFeed`, the feed the offline
replays use: 1/`ctx.fps` per tick fed, the first after a handback adding nothing, so a pause, a playback
or a slow loop never counts. Each tick's signals and the detector's state go up in the Output (`signals`,
`readings`), so the layers above read one detector.

It logs a `grasp` event for each close, carry, regrasp and release, and a `rule` event when a stuck rule
fires, each at the model's seconds. After a handback it tells the detector, on the next tick fed, whether
the arm moved (`Detector.resumed`), so a close a playback undid is never counted as a regrasp.
"""

from dataclasses import replace

from so_101.policy.base import POLICY, WAITING, Handback, Layer, LayerContext, Output, Readings, Tick


class Perception(Layer):
    def __init__(self, inner, ctx: LayerContext) -> None:
        super().__init__(inner, ctx, "perception")
        self.feed = None  # grasp.replay.LiveFeed: the attempt's detector, its clock and the handback to tell it
        self.detector = None

    def begin_episode(self, number: int) -> None:
        self.inner.begin_episode(number)
        self.feed = self.detector = None
        self.safely(self._begin)

    def _begin(self) -> None:
        from so_101.grasp.replay import LiveFeed

        self.feed = LiveFeed(self.ctx.grasp, self.ctx.fps)
        self.detector = self.feed.detector

    def end_episode(self) -> None:
        self.inner.end_episode()
        self.feed = self.detector = None

    def run(self, handback: Handback) -> None:
        self.inner.run(handback)
        if self.feed is not None:  # held, a move outweighing a still handback, until the next tick fed (`LiveFeed.handback`)
            self.feed.handback(handback.moved)

    @property
    def t(self) -> float:
        """The model's seconds at the tick fed last."""
        return self.feed.t if self.feed is not None else 0.0

    def step(self, robot, tick: Tick) -> Output:
        out = self.inner.step(robot, tick)
        if self.detector is None or out.driver not in (POLICY, WAITING) or self.broken is not None:
            return out
        seen: list = []
        if not self.safely(self._watch, tick, out.driver == POLICY, seen):
            return out
        signals, readings = seen
        return replace(out, signals=signals, readings=readings)

    def _watch(self, tick: Tick, drove: bool, seen: list) -> None:
        found = []
        _, signals = self.feed.feed(tick.pose, drove)
        for signal in signals:
            found.append(signal)
            kind = "rule" if signal.kind == "rule" else "grasp"
            self.ctx.event(kind, **({"name": signal.name} if signal.name else {}),
                           **({} if kind == "rule" else {"grasp": signal.kind}), policy_t=round(signal.t, 2),
                           **signal.detail)
        seen += [tuple(found), self.readings()]

    def readings(self) -> Readings:
        d = self.detector
        # Unrounded: the supervisor compares it with due times the detector's signals set, and rounding sends a
        # check a tick late (`tests/test_supervisor.py`'s SchedulesMatchTest).
        return Readings(t=self.t, closes=d.closes, regrasps=d.regrasps, carried=d.carried, released=d.released,
                        fired=tuple(sorted(d.fired)))
