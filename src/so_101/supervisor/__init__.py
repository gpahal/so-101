"""The supervisor (docs/supervisor.md): a policy over the policy, which watches each attempt and steps in.

`[supervisor] mode` in so101.toml turns it on: `off` (the default), `shadow` (decides and logs, changes
nothing) or `act` (rewinds and retries a missed grasp, holds the arm on a stuck rule, and asks for a
person when the cameras don't clear it or the task looks done). `so101 serve` steps it as its policy
driver, in place of the one it wraps; it sends serve no commands.

Modules:
- `program`   the thresholds, judges and responses, from `program.toml` or the file `[supervisor] program` names
- `detector`  the arm's readings into gripper events and stuck rules; pure, so recorded runs replay through it
- `judges`    the vision checks through failbench's adapters (event, watch, escalation), within a cost cap
- `layer`     `SupervisedPolicy`: steps the wrapped policy, schedules checks, decides, plays back, logs
"""
