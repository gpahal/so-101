"""The supervisor (docs/supervisor.md): the policy stack's top layer, which watches each attempt and steps in.

`[supervisor] mode` in so101.toml turns it on: `off` (the default), `shadow` (decides and logs, changes
nothing) or `act` (plays back a missed grasp and retries, holds the arm on a stuck rule, and asks for a
person when the cameras don't clear it or the task looks done). `policy.stack.build` puts it at the top
of the stack `serve` and `run` step (docs/policy-layers.md); it sends them no commands.

Modules:
- `program`   its timing, judges, responses and budget, from `program.toml` or the file `[supervisor] program`
              names. The gripper and rule thresholds, and the detector the perception steps, are in `grasp/`.
- `judges`    the vision checks through `vision`'s adapters (event, watch, escalation), within a cost cap
- `schedule`  when the after_close and after_release checks fall due, which `failbench.candidates` shares
- `layer`     `Supervisor`: reads the perception's signals, schedules checks, decides, plays back, logs
"""
