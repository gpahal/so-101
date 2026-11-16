# Supervisor: catching stuck and failed attempts while a policy runs

Status: built, as a policy layer in `so101 serve` and `so101 run` (`src/so_101/supervisor/`, program version 5; how to use it is in the README, under The supervisor), tested only on the fake arm with stand-in models. Nothing here has run on the arm, and no judge has answered from it. The stuck rules below were measured offline on the first 150 recorded attempts, which were deleted for a fresh start; the numbers need refitting on `serve` and `run` runs of the policies trained on the new demos. The judges' thresholds (`sure_ok_below`, `sure_wrong_above`, `success_p_wrong`) were tuned with the shared prompt as it was then, which told every model it got two frames, the top camera then the wrist camera; every request now says which frames it carries (`vision.prompt.describe`), so they need checking with `so101-failbench` too. Not built: the compile loop (step 1 below), the layer in `sim eval`, and acting on Reset, which stays the operator's.

The supervisor watches a policy attempt and acts when it fails or gets stuck. It's a policy over the policy (System 2 over System 1): it wraps the trained policy's driver, and each tick either passes the tick to the policy or drives the arm itself, to play back a missed grasp or to hold. Code decides most cases from the arm's own readings. A fast decision model (Jev) and Gemini 3.8 Flash check the camera frames at gripper events, and Jev checks again when a stuck rule holds the arm. A missed grasp is played back only when both are sure it's wrong, and a frontier model is asked only to advise a person once the supervisor has stopped. It sends the control center no commands: `serve` and `run` belong to the operator (and to `run`'s autopilot), and the supervisor is part of the policy they run.

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

Vision models can't see the first four in a single frame, and none of them produces a clean gripper event. That's why the supervisor needs rules over the arm's readings over time, not only checks at events. The camera check on a stuck attempt (the watch check) runs once a rule has fired and the arm holds; an earlier design also checked about once a second while a rule was close to firing, which version 4 of the program dropped, because nothing acted on those answers.

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

The offline replay scripts are in `.scratch/stuck/` (git-ignored); the recordings they read are gone. The rules now live in `grasp/detector.py`, their thresholds in `grasp/grasp.toml`, and `so101-failbench candidates` replays `serve` and `run` runs through them. Thresholds were chosen on the same attempts they're scored on, so expect somewhat worse on new runs.

## Who does what

| Part | What | When | Where |
|---|---|---|---|
| Controller | The existing 30 Hz loop and state machine, unchanged | Every tick | `control/controller.py` |
| Stuck and event detector | Code: the rules above, the gripper bands, and the events (after a close, after a release) | Every tick the policy drives | The layer, on the loop's thread |
| Event check | One Jev request with every question for the moment, plus one Gemini 3.8 Flash request sent at the same time, each its own call; settled as soon as the answers in can't be overturned, or after `answer_within_s` (5 s) | At after_close and after_release | Supervisor's worker pool |
| Watch check | One Jev request with four frames, each camera now and 2 s before (`lookback_s`), and the `stuck` moment | Once per hold, when a stuck rule or a third miss holds the arm; a hold it hasn't cleared in `hold_deadline_s` (5 s) asks for a person | Supervisor's worker pool |
| Escalation | A frontier model reads a contact sheet of the attempt's closes and the log, and picks from a fixed list | Once an attempt, when a hold asks for a person | Supervisor's worker pool |
| Program | The timing, judges, responses and budget (`supervisor/program.toml`); the gripper and stuck-rule thresholds are `grasp/grasp.toml`'s | Read at start, every version recorded in the run's `run.json` | Two versioned files |
| Operator | Takes over with Teleop, confirms Reset and unsure successes | When asked | The page |

### Why a layer, not a second operator

The first build ran the supervisor as a thread in `so101-control` that sent Pause, Rewind and Resume through `Controller.submit`, as a person does. That mixed two things `serve` keeps apart:

- **Training data.** `control/analysis.py` reads an operator's Pause or Rewind as a person judging the policy. Supervisor commands looked the same, so a supervised retry counted as a human step-in.
- **Who is in charge.** Two senders of commands had to watch each other: the thread read the event log to back off once a person acted, and needed guards so a refused Rewind never later turned a Pause into a Resume.
- **Where it runs.** It could only run inside `serve`, never in `so101 run` or the sim.

So the supervisor is now the top layer of the policy stack ([`policy-layers.md`](policy-layers.md)): `supervisor/layer.py`'s `Supervisor`, over the perception (and the skill sequencer, for a skills model), with the interface every policy in the stack has. It reads the perception's signals rather than keeping a detector of its own, asks for a person with `Output.asks`, which the state machine takes as the `policy_asked` event (Autonomous to Paused, like a lost link), and hands the arm back after a playback with a `Handback` that says the arm moved. It shares code with `serve` rather than calling it: its playback is `control/rewind.py`'s `History` and `Rewind`. Its own ticks are recorded with `control.driver` playback or hold, never as the model's. Each playback's `playback` event has the frames it undid, which analysis treats as suspect, and a `policy_asked` pause counts as an interruption. Neither is ever trained on as policy data.

Everything runs on the control loop's thread except the judges, whose calls the loop sends to a worker pool, one call per judge, and collects on later ticks, so a step never waits. The pool's workers are daemon threads, so a call that hangs never keeps `serve` or `run` from exiting, and closing it cancels the calls still queued. Each judge's request times out at `answer_within_s` (5 s), when the layer stops waiting for it, rather than at failbench's 180 s, and a check that settles cancels its calls that haven't started: a slow provider would otherwise keep all 8 workers busy within about four checks, later checks would queue and come back stale, and the queued calls would still bill. Program version 5 made `answer_within_s` the request timeout. A judge whose key or SDK is missing is left out and named once; once the run has spent `max_cost_usd` (2 USD), the checks stop calling and the operator is told once. An exception in the layer's own logic is logged and announced once, and from then on every tick goes straight to the policy; the layer's own logic runs in `Layer.safely`, and every call down the stack outside it, so an exception from the arm or the policy reaches the control loop as it would without the layer.

Every check carries the generation it was asked in, which `pause`, `run` and each hold raise. A watch verdict settles only the hold it was asked for. A person's Pause during a hold ends it (a `hold_ended` event): whoever paused has the arm, so the hold's deadline asks nobody, no escalation is sent, and its watch verdict is stale. The rule that held stays spent for the attempt, because the detector fires each rule once, so after Resume the policy drives and the other rules and the miss ladder still watch: the person saw the arm stuck and handed it back. An after_close verdict is dropped after a pause or resume, a newer close, a carry or release since it was asked, or once it is `after_close_answer_s` (1.5 s) of the policy's time old: by then the first playback, 2.5 s, would no longer reach back past the close, which the program's loader checks. Each dropped check is logged with `stale` saying why, and an answer that comes in after its check settled is logged as `late_answer`. After each `run` the playback history starts again, so the layer never drives back past a pose a person set.

## How it acts

Responses get more expensive only as evidence grows, because a false alarm on a cheap action costs seconds while one on Reset costs the attempt:

1. **First miss** (after_close, both Jev and Flash sure it's wrong): play back 2.5 s of the policy's own motion, newest first, and hand the arm back to the policy with its queue cleared (`miss_rewind_s`). The handback says the arm moved, so the perception forgets the close it undid (`Detector.resumed`): it never counts as a regrasp, and the next close starts a new grasp. One judge alone never plays anything back: a false miss costs a retry, and a wrong Jev answer on a real grasp would undo it.
2. **Second miss:** play back 5 s, so the policy approaches along a different path. ACT has no randomness at inference, so a retry from the same pose would very likely repeat the miss. While the ladder has a rung left and both judges answer, a `regrasp_loop` rule is logged as `ignored` and left to it: the ladder sees each miss, and the rule would hold the arm a rung early.
3. **A stuck rule fires, or a third miss:** hold the arm. The watch check looks at each camera now and 2 s before, which cuts false alarms. A sure ok after a stuck rule hands the arm back. Otherwise, or with no answer within `hold_deadline_s` (5 s), the layer asks for a person with an `Ask` of kind `stuck`, the control center pauses, and one escalation follows: the frontier model gets every close of the attempt and picks the advice for that person:
   - take over with Teleop (block in a pose the demos don't cover);
   - check the gripper (it isn't following its commands);
   - Reset as a failure (the scene changed).
   It never orders another identical retry.
4. **After a release:** when Jev and Flash are both sure (each under 0.1) that the block is in the container, the task is done. The layer logs it and lets the policy drive the arm home, then asks for a person with an `Ask` of kind `done` once `Tick.at_rest` says the arm is at rest, or `home_within_s` (10 s) of the policy's time after the release, so Reset never cuts the return to rest short. A wrong release is told to the operator. The supervisor never labels or ends an attempt itself; in `so101 run` the autopilot ends the attempt at a `done` ask.

A Teleop takeover from a stuck pause is recorded as a correction from the failure state, which is the training data the policies lack.

## Building it

1. **Offline, no arm.**
   - Built: the stuck detector as a pure module (`grasp/detector.py`), tested on made-up traces. The recorded attempts it should have matched are deleted, so it's checked against new `serve` and `run` runs instead.
   - Built: a `stuck` moment in failbench, and `so101-failbench candidates`, which picks frames from `serve` and `run` runs where each rule fires and at each check, for labelling.
   - Built: the program format and its loader (`supervisor/program.toml`, `program.py`), and the thresholds' (`grasp/grasp.toml`, `grasp/thresholds.py`).
   - Not built: the compile loop, in which a frontier model proposes questions and a change is kept only if held-out failbench accuracy rises. It needs a labelled benchmark from new runs first.
2. **In `serve` and `run`, off by default** (`[supervisor] mode = "off"` in `so101.toml`, or `shadow` or `act`; `--supervisor` sets it for one run, and `run.json` records it):
   - Built: the layer (`supervisor/layer.py`), the policy stack's interface the controller steps it through and the `policy_asked` event, and analysis of its playbacks and pauses, with tests on the fake arm.
   - Decided: the layer sends the control center no commands. Teleop, Reset and labels stay the operator's, and in `run` the autopilot's; the layer only asks.
3. **On the arm:** measure stuck attempts caught, false alarms, event-to-command latency and the cost of each false alarm. Run `shadow` first, then `act`.

## Open questions

- Whether gripper load, logged in `serve` and `run` runs as `gripper_load` about once a second, separates an empty edge pinch from a held block. The 5 empty carries need it or the wrist camera.
- How well the watch check works on mid-approach frames. Failbench has no such frames yet.
- Whether a 5 s playback gives ACT a different enough approach to break a regrasp loop.
- Whether the layer belongs in `sim eval` too. It runs LeRobot's own rollout loop, so it needs a wrapper around LeRobot's policy rather than the policy stack. The sim would let the rules and the playbacks be tested without the arm, though its rendered frames may not suit the judges.
- All thresholds are fitted on the same 150 attempts, and the judges' with the old two-frame prompt. They need checking on new runs.
- The time limits (`after_close_answer_s`, `hold_deadline_s`, `home_within_s`, `answer_within_s`) are unmeasured: no judge has answered from the arm yet.
