# so-101

`so101` is a thin CLI over LeRobot for one SO-101 leader and follower arm pair with two cameras, on macOS. `README.md` is the user guide: every command, the hardware setup, and the record, review, train and run loop. Read it before changing a command's behavior, and update it in the same change.

## Commands

- `uv sync` sets up `.venv`. Python is pinned to 3.13: LeRobot's argument parser (draccus) breaks on 3.14.
- `uv run so101 <command> --help` lists a command's options.
- There is no test suite, linter or formatter. To verify a change without hardware:
  - `uv run python -m compileall -q src`
  - `uv run so101 <command> --help`
  - `uv run python -c "from so_101 import settings; print(settings.load())"` for config changes
  - Call the changed function directly with fake inputs. For motor code, run LeRobot's motor code against an in-memory `rustypot.Bus` (already a dependency).
- Say in the PR when a change hasn't run on the real arm or a remote server.

## Commands that touch hardware or bill

The arms may be plugged in. Run only read-only commands unless the user asks: `info`, `env`, `tasks`, `check` (without `--live`), `scan`, `motors`, `pose`, and `--help` on any command.

- These move the arm: `teleop`, `record`, `run`, `check --live`, `record --dry-run --replay`.
- `setup-motors` and `calibrate` write to the motors.
- These bill the user's Hugging Face credit: `train` (unless `--target local`) and `server start`. A started server keeps billing until `server stop`.
- `train` and `backup` upload to the Hub.

## Architecture

- **Two config files, one loader each.** `so101.toml` holds hardware and run settings, loaded by `settings.load()` into frozen dataclasses (`Settings`, `Run`, `Train`, ...). `tasks.toml` holds tasks, loaded by `tasks.py`. `SO101_*` environment variables override `so101.toml`. Ports and ids are written only in `so101.toml`. A new setting needs a dataclass field, a default in `load()`, and a comment in `so101.toml` when the file sets it.
- **LeRobot runs as a child process.** `cli.py` builds a LeRobot CLI command (`lerobot-record`, `lerobot-teleoperate`, ...) from the settings and runs it with `console.run_lerobot`. That function prints the full command first, and filters known-harmless stderr noise such as the duplicate FFmpeg class warning. Unknown CLI arguments pass straight through to LeRobot, which is why each `cmd_*` takes an `extra` list.
- **`so101-rollout`** (`rollout.py`) replaces `lerobot-rollout` for `so101 run`. It moves the arm to the demos' start pose before each attempt and waits until the arm gets there. The CLI passes that pose in through the `SO101_START_POSE` environment variable.
- **Preflight gate.** `teleop`, `record` and `run` call `check.preflight()`, which runs the `so101 check` checks and exits on any FAIL. There is deliberately no override. Checks report PASS/WARN/FAIL with a fix through `report.Report`.
- **Cameras are found by USB id.** `devices.find_camera` maps each camera's `usb_id` to its current OpenCV index, because indices change on replug. `devices.apply_camera_controls` sets UVC exposure and anti-flicker with the external `uvc-util` binary, since macOS gives OpenCV no access to those controls.
- **Demos are append-only.** Each `record` run is a new session, a separate LeRobot dataset under `data/<task>/sessions/` that is never modified. `review` writes trims and deletions only to `edits.toml`. `build.py` makes `train/` from the sessions plus the edits and rebuilds when either changes. The layout is in the `data.py` docstring.
- **Remote inference.** `remote.py` starts a Hugging Face Jobs GPU running LeRobot's policy server, labels the job with what it serves, and reaches it over an SSH tunnel. `run --remote` refuses a server whose labels don't match the run. The server's client rejects `--policy.*` flags, so policy options go to the server config.
- **Units.** `use_degrees` in `tasks.toml` says whether a dataset is in degrees (LeRobot 0.5 and later) or the older -100..100 range. Joint comparisons such as `pose` and the rollout start pose must use the dataset's units.

## Ignored and local files

- `data/` (demos) and `outputs/` (policies, rollouts, reports) are git-ignored, and `data/` holds hours of the user's recordings. Back demos up with `so101 backup`. Never delete anything under either folder.
- `.scratch/` is git-ignored working space: handoff notes, research and reviews.
- `src/so_101/sim/` is git-ignored work in progress from the `sim-twin` branch (PR #13). Keep it out of commits on other branches.

## Writing style

The README, the `so101.toml` comments, help text and printed messages use plain, specific sentences. They give measured numbers with units and the reason behind each setting (for example, why a camera's exposure is 114). Match that style, and keep comments true when a setting changes.

Commit and PR titles are imperative sentences without a prefix, for example "Match the run limit to recording and add an action steps option". PRs are squash-merged, with the PR number appended to the title. PR bodies have `## Summary` and `## Testing` sections, and the Testing section uses a table of cases and results.
