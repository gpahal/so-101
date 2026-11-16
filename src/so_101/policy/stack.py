"""Compose the policy stack (docs/policy-layers.md): the one place that knows the layers' order.

    EngineDriver -> Perception -> SkillSequencer (a skills run) -> Supervisor ([supervisor] mode shadow or act)

Perception is always there, so every run logs its grasps; it reads the program's gripper and rule
thresholds, which therefore load for every run.
"""

from so_101.policy.base import LayerContext, Policy
from so_101.policy.perception import Perception


def build(driver, ctx: LayerContext, *, skills: tuple = (), judges=None, mode: str = "off", cameras=()) -> Policy:
    """The stack over `driver`: perception, then a skill sequencer over `skills` when there are any,
    then the supervisor with `judges` when `mode` is shadow or act."""
    policy: Policy = Perception(driver, ctx)
    if skills:
        from so_101.policy.skills import SkillSequencer

        policy = SkillSequencer(policy, ctx, skills)
    if mode in ("shadow", "act"):
        from so_101.supervisor.layer import Supervisor

        policy = Supervisor(policy, ctx, judges, mode=mode, cameras=cameras)
    return policy
