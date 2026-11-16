# `so101 run` on the control center: serve plus an autopilot

Status: built, tested on the fake arm, not yet run on the arm. Replaced `run`'s LeRobot rollout loop
(`so101-rollout`). How to use it is in the README, under Training your own task (step 6).

`run` and `serve` become one program, `so101-control`, with one difference: in `run` an **autopilot**
starts and ends the attempts. Everything else is shared: the policy stack (skills, supervisor,
perception, [`policy-layers.md`](policy-layers.md)), the control page and keys, the safety rules, and the
recording into `data/<task>/runs/`, so `review --runs`, `runs`, `build --with` and the skills builds read
`run` sessions as they read `serve` ones.

## The command

```
so101 run [task] [--episodes N] [--duration S] [--reset-time S] [--operator NAME]
          [--policy act|pi05] [--remote] [--revision TAG] [--checkpoint DIR] [--action-steps N] [--skills]
```

- `--episodes`: attempts to make (default `[run] episodes`).
- `--duration`: seconds of policy time per attempt; overrides `[control] attempt_limit_s` for this run.
  `[run] duration_s` is removed: one setting for both commands.
- `--reset-time`: seconds between attempts to reset the scene (default the task's `reset_time_s`).
- `--display` is removed: the control page shows both cameras.
- Everything else as `serve`. `run.json` records `"mode": "run"` and the autopilot's settings.

## The autopilot (`control/autopilot.py`)

A small piece of the control process that submits commands with source `autopilot`, as the terminal and
the page submit theirs, so the state machine still decides what's allowed. Once a tick it looks at the
controller's status:

1. **Between attempts**, at rest with no attempt open: after `reset_time_s` (none before the first
   attempt) it submits Resume, which opens an attempt, moves to the demos' start pose and starts the policy.
2. **An attempt ends** at its time limit (unlabelled, without the outcome prompt waiting for an answer),
   or when the policy stack asks for a person: the supervisor sure the task is done, or stuck and not
   cleared. After an ask, the autopilot waits `ASK_GRACE_S` (10 s) for a person to act; if nobody does, it
   ends the attempt unlabelled with Reset. The supervisor's decisions and advice are in the run's events
   for review.
3. **A person's command** during an attempt (Pause, Rewind, Teleop, Reset, from the page or the keys)
   hands that attempt to them: the autopilot doesn't resume or end it, and carries on with the next
   attempt once it has ended.
4. **It stops** after `--episodes` attempts have ended, at Esc (Quit, with the usual confirmation away from
   rest), or when an attempt can't start (a fault, a full disk): it says why and waits for a person.

The arrows mean what they mean in `serve`: the right arrow ends the attempt and asks how it went, the left
arrow ends it as one to ignore.

## What goes away

- `so101-rollout`, its LeRobot patches and `run --display`. `rollout.py` keeps the helpers the control
  center uses (`Move`, `TOLERANCE`, `farthest`).
- `outputs/rollouts/`: there are no recordings in it on this Mac.

## Tests

On the controller's fake arm: the autopilot makes N attempts with resets between them and stops; a time
limit ends an attempt unlabelled; an ask ends it after the grace period, and a person's command within it
keeps it open; a person's Pause hands that attempt over and the next still starts; Esc stops it; `run.json`
says `run`. `run` and `serve` share `commands/policy.py`'s setup, so their preflight tests cover both.
