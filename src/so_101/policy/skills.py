"""SkillSequencer: runs a skills model one skill at a time, switching at each handover.

It starts each attempt on the task's first skill (`so_101.skills`), conditioning the model on its prompt
and one-hot. When the perception below reports the next skill's handover signal (the carry, for
block-container's place), it conditions the model on that skill, which clears the model's queued
actions in the leaf, and logs a `skill` event. Each tick's Output keeps the current skill's index in
`skill`, which the recording writes as control.skill.

A handback with `moved` (a layer above played motion back, or a person rewound or teleoperated) means
what was held may have been let go or undone: on its next tick, a gripper reading open sends it back to
the first skill. A gripper still closed keeps the skill it was on, and a block a person closed the jaws
on counts as a close in the perception below, so its carry switches to the next skill as usual.
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

    @property
    def skill(self):
        return self.skills[self.current]

    def _switch(self, index: int, cause: str, frame: int | None = None) -> None:
        previous, self.current = self.skill.name, index
        self.inner.condition(Conditioning(self.skill.prompt, tuple(s.name for s in self.skills), index))
        self.ctx.event("skill", skill=self.skill.name, index=index, cause=cause,
                       **({"previous": previous} if cause != "start" else {}))

    def begin_episode(self, number: int) -> None:
        self.inner.begin_episode(number)
        self._recheck = False
        self._switch(0, "start")

    def run(self, handback: Handback) -> None:
        self.inner.run(handback)
        self._recheck = handback.moved

    def step(self, robot, tick: Tick) -> Output:
        if self._recheck:
            self._recheck = False
            self.safely(self._after_handback, tick)
        out = self.inner.step(robot, tick)
        self.safely(self._handover, out)
        return replace(out, skill=self.current)

    def _after_handback(self, tick: Tick) -> None:
        if self.current and tick.pose.get(GRIPPER, 0.0) >= self.ctx.grasp.gripper.closed_below:
            self._switch(0, "reopened")

    def _handover(self, out: Output) -> None:
        nxt = self.current + 1
        if nxt < len(self.skills) and any(s.kind == self.skills[nxt].handover for s in out.signals):
            self._switch(nxt, self.skills[nxt].handover)
