"""The longer command implementations, kept apart from `cli.py`, which parses the arguments and dispatches.

- `policy`: the checks and LeRobot arguments `run` and `serve` share.
- `serve`: `so101 serve` and `so101 run`, which run the control center as `so101-control`.
- `runs`: `so101 runs`, which summarises a task's runs.
- `sim`: `so101 sim`, which builds a task's simulated twin from its demos and runs a policy in it.
- `skills`: `so101 skills`, which checks each skill's cuts and draws contact sheets.
"""
