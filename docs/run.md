# `so101 run` on the control center: serve plus an autopilot

Status: built, tested on the fake arm, not yet run on the arm or shown in a browser. Replaced `run`'s
LeRobot rollout loop (`so101-rollout`). How to use it is in the README, under Training your own task
(step 6).

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
the page submit theirs, so the state machine still decides what's allowed. The controller steps it once a
tick, after publishing the status, with that tick's facts (`autopilot.Facts`): the state, whether the arm
is at rest, the attempt and whether it's open, why the arm is paused (the policy stack asked, with the
ask's kind; the control page's link was lost; a move stopped short; or a person), the commands people gave
that tick, whether a person is answering how the attempt went, and why Resume would be refused. It reads
no event log and runs no thread of its own, and it has one command in flight at a time, so a slow tick
never makes it send a Resume twice.

1. **Between attempts**, resting at the rest pose with no attempt open: `reset_time_s` after the arm
   arrived at rest (none before the first attempt) it submits Resume, which opens an attempt, moves to the
   demos' start pose and starts the policy. A person's command between attempts restarts that wait. It
   starts nothing while the arm rests away from the rest pose (a move to rest paused or stopped short), in
   Fault, or while the page's Hold is on (`autopilot_hold`, allowed only at rest): it says why once, in
   the terminal and on the page, and waits.
2. **An attempt ends** at its time limit (unlabelled, without the outcome prompt waiting for an answer),
   or after a pause the autopilot didn't ask for: the policy stack asked for a person (the supervisor
   stuck and not cleared), the control page's link was lost, or the move to the start pose or to rest
   stopped short. It says why once, waits `ASK_GRACE_S` (10 s) for a person to act, and if nobody does,
   ends the attempt unlabelled with Reset. A `done` ask, the supervisor sure the task is done once the
   arm is home, ends the attempt at once; the verdict is in the run's events. While a person is saying
   how the attempt went (the terminal's right-arrow question, or the page's Reset sheet, which renews
   this every 10 s and lapses after 30 s), the grace Reset waits, so it can't take their label.
3. **A person's Pause, Resume, Rewind, Teleop, Reset or Quit** during an attempt, from the page, the
   keys or a signal, hands that attempt to them: the autopilot doesn't resume or end it, and carries on
   with the next attempt once it has ended. Check pose, Run checks, a step-in label and taking control on
   the page don't hand it over.
4. **It stops** after `--episodes` attempts have ended, by submitting Quit from rest at the rest pose;
   with the arm anywhere else, it says the run is over and waits for a person's Reset. A person's Quit
   (Esc, with the usual confirmation away from rest, or a signal) stops it too. The state machine refuses
   the autopilot a Quit that would exit leaving a silent follower's motors as they are: that's a person's
   call.

The page's status has `autopilot`: `made` (attempts ended) of `of`, `next_in_s` while it counts down to
the next attempt, `waiting` (why it waits, or null) and `held`. The page shows them in a card with a
Hold or Continue button. With `--duration 0` the status's `limit_s` is null and the page shows no limit.

The arrows mean what they mean in `serve`: the right arrow ends the attempt and asks how it went, the left
arrow ends it as one to ignore.

## What goes away

- `so101-rollout`, its LeRobot patches and `run --display`. `rollout.py` keeps the helpers the control
  center uses (`Move`, `TOLERANCE`, `farthest`).
- `outputs/rollouts/`: there are no recordings in it on this Mac.

## Tests

On the controller's fake arm (`tests/test_autopilot.py`): the autopilot makes N attempts with resets
between them and stops at rest; an unanswered ask ends an attempt after the grace period, a `done` ask at
once; a person's command within the grace keeps it open; Check pose, a label and taking control don't
hand it over; the terminal's outcome question and the page's Reset sheet keep the grace Reset off the
label; a lost link and a move to the start pose that stopped short are ended after the grace; a person's
Pause on the way to rest hands that attempt over; a fault on the last attempt waits for a Reset and quits
only at rest; resting away from rest starts nothing; Esc stops it; a refused start is said once; Hold
keeps the next attempt from starting, even one the autopilot already sent. `tests/test_process.py` runs
`so101-control` in run mode and checks `run.json` and `--duration 0`'s strict JSON status. `run` and
`serve` share `commands/policy.py`'s setup, so their preflight tests cover both.
