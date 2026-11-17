"""SkillSequencer: runs a skills model one skill at a time, switching at each handover.

It starts each attempt on the task's first skill (`so_101.skills`), conditioning the model on its prompt
and one-hot. When the perception below reports the next skill's handover signal (the carry for
block-container's place, the release for its return), it conditions the model on that skill, which
clears the model's queued actions in the leaf, and logs a `skill` event. Each tick's Output keeps the
current skill's index in `skill`, which the recording writes as control.skill.

A handback with `moved` (a layer above played motion back, or a person rewound or teleoperated) means
what was held or let go may have been undone. The sequencer follows the perception below: on the first
tick it feeds after the handback, the perception tells its detector the arm moved
(`Detector.resumed`), and the Output's readings say whether a carry or a release still stands. If
neither does, whatever the current skill, the sequencer starts again on the first skill, as for a
missed grasp: a block dropped outside the jar and rewound to before the close is picked again. With
the jaws closed after the move (inside `hold_band` once a block was carried or let go, `Detector.resumed`)
the detector keeps what it had, so a block still held keeps its skill,
and a block a person closed the jaws on counts as a close, whose carry switches to the next skill as
usual. A person who teleoperates after a real release, with the jaws open, also sends the attempt back
to the first skill: the perception can't tell a release from a block put back. The model drove that
first tick on the skill it had; the switch conditions it from the next, and the frame records the new
skill, as at a handover.

Deciding to switch is the layer's own logic and runs under `safely`; conditioning the model below runs
outside it, so an error from the model reaches the loop as the model's.
"""

from dataclasses import replace

from so_101.policy.base import Conditioning, Handback, Layer, LayerContext, Output, Readings, Tick


class SkillSequencer(Layer):
    def __init__(self, inner, ctx: LayerContext, skills: tuple) -> None:
        super().__init__(inner, ctx, "skills")
        self.skills = skills
        self.current = 0
        self._recheck = False  # a handback moved the arm: check the perception's next readings
        self._tell: Conditioning | None = None  # what the next public call tells the model, outside `safely`

    @property
    def skill(self):
        return self.skills[self.current]

    def begin_episode(self, number: int) -> None:
        self.inner.begin_episode(number)
        self._recheck, self._tell, self.current = False, None, 0  # even if `_move` fails: never the last attempt's
        self.safely(self._move, 0, "start")
        if (told := self._tell) is not None:  # outside `safely`: the model's error is the model's
            self._tell = None
            self.inner.condition(told)

    def run(self, handback: Handback) -> None:
        self.inner.run(handback)
        self._recheck = self._recheck or handback.moved

    def step(self, robot, tick: Tick) -> Output:
        out = self.inner.step(robot, tick)
        if self._recheck and out.readings is not None:  # the perception fed this tick, after `resumed`
            self._recheck = False
            self.safely(self._after_handback, out.readings)
        self.safely(self._handover, out)
        if (told := self._tell) is not None:
            self._tell = None
            self.inner.condition(told)
        return replace(out, skill=self.current)

    def _move(self, index: int, cause: str) -> bool:
        """Switch to skill `index` and log it; the public method that called this conditions the model."""
        previous, self.current = self.skill.name, index
        self._tell = Conditioning(self.skill.prompt, tuple(s.name for s in self.skills), index)
        self.ctx.event("skill", skill=self.skill.name, index=index, cause=cause,
                       **({"previous": previous} if cause != "start" else {}))
        return True

    def _after_handback(self, readings: Readings) -> bool:
        undone = not (readings.carried or readings.released)
        return bool(self.current and undone and self._move(0, "reopened"))

    def _handover(self, out: Output) -> bool:
        nxt = self.current + 1
        return (nxt < len(self.skills) and any(s.kind == self.skills[nxt].handover for s in out.signals)
                and self._move(nxt, self.skills[nxt].handover))
