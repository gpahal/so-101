"""The policy stack the control loop steps (docs/policy-layers.md): the model, and layers over it.

A policy is anything the loop steps (`base.Policy`); a layer is a policy that wraps one (`base.Layer`),
forwarding every method it doesn't change. `stack.build` composes the stack, in one fixed order:

- `base`        Policy, Layer, and the Tick, Output, Handback and Conditioning passed between them
- `engine`      EngineDriver: LeRobot's inference engine, local or remote; the only LeRobot code
- `perception`  Perception: the gripper rules over each tick the model drove, logged as `grasp` and `rule` events
- `skills`      SkillSequencer: which skill the model is conditioned on, switched at each handover (--skills)
- (`supervisor.layer.Supervisor`, when [supervisor] mode isn't off)
- `stack`       build: composes them; the only place that knows the order
"""
