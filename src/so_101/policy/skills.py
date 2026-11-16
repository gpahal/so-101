"""SkillSequencer: runs a skills model one skill at a time, switching at each handover.

It starts each attempt on the task's first skill (`so_101.skills`), conditioning the model on its prompt
and one-hot. When the perception below reports the next skill's handover signal (the carry for
block-container's place, the release for its return), it conditions the model on that skill, which
clears the model's queued actions in the leaf, and logs a `skill` event. Each tick's Output keeps the
current skill's index in `skill`, which the recording writes as control.skill.

A handback with `moved` (a layer above played motion back, or a person rewound or teleoperated) means
what was held may have been let go or undone: on its next tick, in a skill that holds the block (one
handed over on a carry), a gripper reading open sends it back to the first skill. A gripper still closed
keeps the skill it was on, and a block a person closed the jaws on counts as a close in the perception
below, so its carry switches to the next skill as usual. A skill handed over on a release expects the
jaws open, so a handback never moves it.

Deciding to switch is the layer's own logic and runs under `safely`; conditioning the model below runs
outside it, so an error from the model reaches the loop as the model's.
"""

from dataclasses import replace

from so_101.policy.base import Conditioning, Handback, Layer, LayerContext, Output, Tick

GRIPPER = "gripper.pos"


class SkillSequencer(Layer):
    def __init__(self, inner, ctx: LayerContext, skills: tuple) -> None:
        super().__init__(inner, ctx, "skills")
        self.skills = skills
        self.current = 0
        self._recheck = False
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
        self._recheck = handback.moved

    def step(self, robot, tick: Tick) -> Output:
        if self._recheck:
            self._recheck = False
            self.safely(self._after_handback, tick)
            if (told := self._tell) is not None:
                self._tell = None
                self.inner.condition(told)
        out = self.inner.step(robot, tick)
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

    def _after_handback(self, tick: Tick) -> bool:
        holds = self.current and self.skill.handover == "carry"
        return bool(holds and tick.pose.get(GRIPPER, 0.0) >= self.ctx.grasp.gripper.closed_below
                    and self._move(0, "reopened"))

    def _handover(self, out: Output) -> bool:
        nxt = self.current + 1
        return (nxt < len(self.skills) and any(s.kind == self.skills[nxt].handover for s in out.signals)
                and self._move(nxt, self.skills[nxt].handover))
