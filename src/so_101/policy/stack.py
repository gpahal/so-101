"""Compose the policy stack (docs/policy-layers.md): the one place that knows the layers' order.

    EngineDriver -> Perception -> SkillSequencer (a skills run) -> Supervisor ([supervisor] mode shadow or act)

Perception is always there, so every run logs its grasps; it reads the gripper and rule thresholds
(`grasp.thresholds`), which therefore load for every run. The supervisor's program loads only when the
supervisor runs.
In a skills run the perception and the sequencer are essential (`Layer.essential`): the skill changes
only on what the perception sees, so either one broken refuses Resume rather than leave the model on its
first skill.
"""

from so_101.policy.base import LayerContext, Policy
from so_101.policy.perception import Perception


def build(driver, ctx: LayerContext, *, skills: tuple = (), judges=None, mode: str = "off", cameras=()) -> Policy:
    """The stack over `driver`: perception, then a skill sequencer over `skills` when there are any,
    then the supervisor with `judges` when `mode` is shadow or act."""
    policy: Policy = Perception(driver, ctx)
    if skills:
        from so_101.policy.skills import SkillSequencer

        policy.essential = True
        policy = SkillSequencer(policy, ctx, skills)
        policy.essential = True
    if mode in ("shadow", "act"):
        from so_101.supervisor.layer import Supervisor

        policy = Supervisor(policy, ctx, judges, mode=mode, cameras=cameras)
    return policy
