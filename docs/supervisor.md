# Supervisor: catching stuck and failed attempts while a policy runs

Status: built through shadow mode, as a policy layer (`src/so_101/supervisor/`; how to use it is in the README, under The supervisor), tested only on the fake arm with stand-in models. Nothing here has run on the arm. The stuck rules below were measured offline on the first 150 recorded attempts, which were deleted for a fresh start; the numbers need refitting on `serve` runs of the policies trained on the new demos. Not built: the compile loop (step 1 below), the layer in `so101 run` and `sim eval`, and acting on Reset, which stays the operator's.

The supervisor watches a policy attempt and acts when it fails or gets stuck. It's a policy over the policy (System 2 over System 1): it wraps the trained policy's driver, and each tick either passes the tick to the policy or drives the arm itself, to play back a missed grasp or to hold. Code decides most cases from the arm's own readings. A fast decision model (Jev) checks the camera frames at gripper events and about once a second while the arm seems stuck. Gemini 3.8 Flash answers when Jev is unsure, and a frontier model is asked only to advise a person once the supervisor has stopped. It sends `serve` no commands: `serve` is the operator's, and the supervisor is part of the policy it runs.

## Why stuck comes first

Of the 150 recorded attempts with an agreed outcome label (ACT and π0.5, across the reviews in `results/`), 83 placed the block. 56 got stuck: the block was never picked up (52) or never let go (4). Only 11 failed at the jar. **Stuck is 84% of failures**, and nothing ends a stuck attempt today except the operator or the 45 s `attempt_limit_s` backstop.

The reviews show five ways of getting stuck:

| Pattern | Seen in | What the arm does |
|---|---|---|
| Regrasp loop | ACT, mostly | Closes on nothing or on the block's edge, opens, and tries again from the same place, 2 to 6 times |
| Hover | π0.5 | Stops with the block at the tip of the left jaw and creeps forward for 10 s or more with the jaws open, never closing |
| Chatter | π0.5 | The gripper flickers around half-open, crossing the open threshold 4 or more times in 3 s |
| Frozen | Both | The arm barely moves for seconds before it has carried anything |
| Not released | π0.5 | Holds the block and never lets go, for example over the jar's old place |

Vision models can't see the first four in a single frame, and none of them produces a clean gripper event. That's why the supervisor needs both rules over time and a periodic check, not only checks at events.

## What code can tell from the arm alone

Replaying every recorded attempt tick by tick, with only what a supervisor would know so far, gives these signals:
- **A close** is the gripper falling below 25.
- **A carry** is either a close followed by the elbow moving 25° or more with the gripper still above 12, or the gripper holding steady (within 2) in 16.5 to 22 for 1.5 s. A miss reopens after a median of 5° of elbow travel; a real grasp moves a median of 36°, or holds steady at 18 to 19.
- **A regrasp** is a close that reopens without a carry.

The rules and their results on the 56 stuck and 83 successful attempts:

| Rule (before any carry unless noted) | Threshold | Stuck caught, ACT | Stuck caught, π0.5 | False alarms on successes |
|---|---|---|---|---|
| Regrasp loop | 2 regrasps | 11/26 | 15/30 | 3/83 |
| Hover | Open, under 10°/s, over 15° from the start pose, for 5 s | 0/26 | 14/30 | 1/83 |
| Chatter | 4 crossings of 25 within 3 s | 2/26 | 11/30 | 1/83 |
| Frozen | Under 3°/s for 4 s | 0/26 | 14/30 | 1/83 |
| No carry yet | 14 s of attempt time | 16/26 | 22/30 | 5/83 |
| Not released (after a carry) | Holding steady 10 s | 0/26 | 2/30 | 1/83 |
| **Any rule** | | **16/26** | **25/30** | **5/83** |

All rules together catch 41 of 56 stuck attempts, at a median of 14 s into the attempt, against a median of 18.9 s at which those attempts actually ended and the 45 s backstop. Of the 15 they miss:
- **11 were ended before 14.5 s** by the operator or the run, before any rule could fire.
- **The other 4 looked like a carry but were empty**, as was one of the 11. The jaws pinched the block's edge, the arm lifted, and the block slipped out. Only the wrist camera shows this, so the event check after each close handles it.

Successful attempts sometimes miss once or twice and recover: 76 of 83 had no regrasp, 4 had one, 2 had two and 1 had three. One regrasp alone is not stuck.

The offline replay scripts are in `.scratch/stuck/` (git-ignored); the recordings they read are gone. The rules now live in `grasp/detector.py`, their thresholds in `grasp/grasp.toml`, and `so101-failbench candidates` replays `serve` runs through them. Thresholds were chosen on the same attempts they're scored on, so expect somewhat worse on new runs.

## Who does what

| Part | What | When | Where |
|---|---|---|---|
| Controller | The existing 30 Hz loop and state machine, unchanged | Every tick | `control/controller.py` |
| Stuck and event detector | Code: the rules above, the gripper bands, and the events (after a close, after a release) | Every tick the policy drives | The layer, on the loop's thread |
| Event check | One Jev request with every question for the moment, plus one Gemini 3.8 Flash request sent at the same time | At after_close and after_release | Supervisor's worker pool |
| Watch check | One Jev request with two frames (now and 2 s ago) and narrow questions | About 1 Hz, only while a stuck rule is close to firing or has fired | Supervisor's worker pool |
| Escalation | A frontier model reads a contact sheet of the attempt's closes and the log, and picks from a fixed list | Only when the program has no rule or both judges are unsure | Supervisor's worker pool |
| Program | The questions, thresholds and rule table, written offline by a frontier model and tested against failbench | Read at start | A versioned file |
| Operator | Takes over with Teleop, confirms Reset and unsure successes | When asked | The page |

### Why a layer, not a second operator

The first build ran the supervisor as a thread in `so101-control` that sent Pause, Rewind and Resume through `Controller.submit`, as a person does. That mixed two things `serve` keeps apart:

- **Training data.** `control/analysis.py` reads an operator's Pause or Rewind as a person judging the policy. Supervisor commands looked the same, so a supervised retry counted as a human step-in.
- **Who is in charge.** Two senders of commands had to watch each other: the thread read the event log to back off once a person acted, and needed guards so a refused Rewind never later turned a Pause into a Resume.
- **Where it runs.** It could only run inside `serve`, never in `so101 run` or the sim.

So the supervisor is now the top layer of the policy stack ([`policy-layers.md`](policy-layers.md)): `supervisor/layer.py`'s `Supervisor`, over the perception (and the skill sequencer, for a skills model), with the interface every policy in the stack has. It reads the perception's signals rather than keeping a detector of its own, asks for a person with `Output.asks`, which the state machine takes as the `policy_asked` event (Autonomous to Paused, like a lost link), and hands the arm back after a playback with a `Handback` that says the arm moved. It shares code with `serve` rather than calling it: its playback is `control/rewind.py`'s `History` and `Rewind`. Its own ticks are recorded with `control.driver` playback or hold, never as the model's. Each playback's `playback` event has the frames it undid, which analysis treats as suspect, and a `policy_asked` pause counts as an interruption. Neither is ever trained on as policy data.

Everything runs on the control loop's thread except the judges, whose calls go to a worker pool and are collected on later ticks, so a step never waits. An exception in the layer's own logic is logged, and from then on every tick goes straight to the policy. An exception from the arm or the policy reaches `serve` as it would without the layer. Late answers carry the attempt and event they were asked about, and are dropped if a newer event has happened since. A verdict that arrives while a person has the arm (between `pause` and `run`) is logged and not acted on, and after each `run` the playback history starts again, so the layer never drives back past a pose a person set.

## How it acts

Responses get more expensive only as evidence grows, because a false alarm on a cheap action costs seconds while one on Reset costs the attempt:

1. **First miss** (after_close says wrong): play back 2.5 s of the policy's own motion, newest first, and hand the arm back to the policy with its queue cleared (`miss_rewind_s`).
2. **Second miss:** play back 5 s, so the policy approaches along a different path. ACT has no randomness at inference, so a retry from the same pose would very likely repeat the miss.
3. **A stuck rule fires, or a third miss:** hold the arm. The watch check confirms with two frames, which cuts false alarms. A sure ok after a stuck rule hands the arm back. Otherwise the layer asks for a person, `serve` pauses, and one escalation follows: the frontier model gets every close of the attempt and picks the advice for that person:
   - take over with Teleop (block in a pose the demos don't cover);
   - check the gripper (it isn't following its commands);
   - Reset as a failure (the scene changed).
   It never orders another identical retry.
4. **After a release:** when Jev and Flash are both sure (each under 0.1) that the block is in the container, the task is done: the layer asks for a person to Reset and label it. A wrong release is told to the operator. The supervisor never labels or ends an attempt itself.

A Teleop takeover from a stuck pause is recorded as a correction from the failure state, which is the training data the policies lack.

## Building it

1. **Offline, no arm.**
   - Built: the stuck detector as a pure module (`grasp/detector.py`), tested on made-up traces. The recorded attempts it should have matched are deleted, so it's checked against new `serve` runs instead.
   - Built: a `stuck` moment in failbench, and `so101-failbench candidates`, which picks frames from `serve` runs where each rule fires and at each check, for labelling.
   - Built: the program format, its loader and the rule engine (`supervisor/program.toml`, `program.py`).
   - Not built: the compile loop, in which a frontier model proposes questions and a change is kept only if held-out failbench accuracy rises. It needs a labelled benchmark from new runs first.
2. **In `serve`, off by default** (`[supervisor] mode = "off"` in `so101.toml`, or `shadow` or `act`):
   - Built: the layer (`supervisor/layer.py`), the three policy hooks in the controller and the `policy_asked` event, and analysis of its playbacks and pauses, with tests on the fake arm.
   - Decided: the layer sends `serve` no commands. Teleop, Reset and labels stay the operator's; the layer only asks.
3. **On the arm:** measure stuck attempts caught, false alarms, event-to-command latency and the cost of each false alarm. Run `shadow` first, then `act`.

## Open questions

- Whether gripper load, logged in `serve` runs as `gripper_load` about once a second, separates an empty edge pinch from a held block. The 5 empty carries need it or the wrist camera.
- How well the watch check works on mid-approach frames. Failbench has no such frames yet.
- Whether a 5 s playback gives ACT a different enough approach to break a regrasp loop.
- Whether the layer belongs in `so101 run` and `sim eval` too. Both run LeRobot's own rollout loops, so each needs a wrapper around LeRobot's policy rather than this driver. The sim would let the rules and the playbacks be tested without the arm, though its rendered frames may not suit the judges.
- All thresholds are fitted on the same 150 attempts. They need checking on new runs.
