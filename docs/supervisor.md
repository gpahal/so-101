# Supervisor: catching stuck and failed attempts while a policy runs

Status: built, as a policy layer in `so101 serve` and `so101 run` (`src/so_101/supervisor/`, program version 2; how to use it is in the README, under [The supervisor](../README.md#the-supervisor)). On the arm it has run only in `shadow`, on 12 attempts of ACT and π0.5 that all placed the block ([`results/act-pi05-block-container-supervisor-shadow.md`](../results/act-pi05-block-container-supervisor-shadow.md)); its playbacks, holds and asks are tested only on the fake arm with stand-in models, and no judge has answered on a failure from the arm. The stuck rules' thresholds haven't been scored on `serve` and `run` runs, and the judges' `sure_ok_below` and `sure_wrong_above` are unscored until `so101-failbench` scores them on the prompt they're asked with. Not built: the compile loop (step 1 below), the layer in `sim eval`, and acting on Reset, which stays the operator's.

The supervisor watches a policy attempt and acts when it fails or gets stuck. It's a policy over the policy (System 2 over System 1): it wraps the trained policy's driver, and each tick either passes the tick to the policy or drives the arm itself, to play back a missed grasp or to hold. Code decides most cases from the arm's own readings. A fast decision model (Jev) and Gemini 3.8 Flash check the camera frames at gripper events, and Jev checks again when a stuck rule holds the arm. A missed grasp is played back only when both are sure it's wrong, and a frontier model is asked only to advise a person once the supervisor has stopped. It sends the control center no commands: `serve` and `run` belong to the operator (and to `run`'s autopilot), and the supervisor is part of the policy they run.

## Why stuck comes first

In the policy reviews in `results/`, most failed attempts got stuck rather than failing at the jar: the block was never picked up, or never let go. Without the supervisor only the operator or the 45 s `attempt_limit_s` ends a stuck attempt.

The reviews show five ways of getting stuck:

| Pattern | Seen in | What the arm does |
|---|---|---|
| Regrasp loop | ACT, mostly | Closes on nothing or on the block's edge, opens, and tries again from the same place, 2 to 6 times |
| Hover | π0.5 | Stops with the block at the tip of the left jaw and creeps forward for 10 s or more with the jaws open, never closing |
| Chatter | π0.5 | The gripper flickers around half-open, crossing the open threshold 4 or more times in 3 s after its first close |
| Frozen | Both | The arm barely moves for seconds before it has carried anything |
| Not released | π0.5 | Holds the block and never lets go, for example over where the jar stood in the demos |

Vision models can't see the first four in a single frame, and none of them produces a clean gripper event. That's why the supervisor needs rules over the arm's readings over time, not only checks at events. The camera check on a stuck attempt (the watch check) runs once a rule has fired and the arm holds.

## What code can tell from the arm alone

Tick by tick, with only what a supervisor would know so far:
- **A close** is the gripper falling below 25.
- **A carry** is either a close followed by the elbow moving 25° or more with the gripper still above 12, or the gripper holding steady (within 2) in 16.5 to 22 for 1.5 s.
- **A regrasp** is a close that reopens without a carry.

The rules, as `grasp/grasp.toml` sets them:

| Rule (before any carry unless noted) | Threshold |
|---|---|
| Regrasp loop | 2 regrasps |
| Hover | Open, under 10°/s, over 15° from the start pose, for 5 s |
| Chatter | 4 crossings of 25 within 3 s, counted from the attempt's first close |
| Frozen | Under 3°/s for 4 s |
| No carry yet | 14 s of attempt time |
| Not released (after a carry) | Holding steady 10 s |

One regrasp alone is not stuck: successful attempts sometimes miss once or twice and recover. A carry can be empty: the jaws pinch the block's edge, the arm lifts, and the block slips out. Only the wrist camera shows this, so the event check after each close handles it.

The rules live in `grasp/detector.py`, their thresholds in `grasp/grasp.toml`, and `so101-failbench candidates` replays `serve` and `run` runs through them.

## Who does what

| Part | What | When | Where |
|---|---|---|---|
| Controller | The 30 Hz loop and state machine | Every tick | `control/controller.py` |
| Stuck and event detector | Code: the rules above, the gripper bands, and the events (after a close, after a release) | Every tick the policy drives | The layer, on the loop's thread |
| Event check | One Jev request with every question for the moment, plus one Gemini 3.8 Flash request sent at the same time, each its own call; settled as soon as the answers in can't be overturned, or after `answer_within_s` (8 s) | At after_close and after_release | Supervisor's worker pool |
| Watch check | One Jev request with four frames, each camera now and 2 s before (`lookback_s`), and the `stuck` moment | Once per hold, when a stuck rule or a third miss holds the arm; a hold it hasn't cleared in `hold_deadline_s` (5 s) asks for a person | Supervisor's worker pool |
| Escalation | A frontier model reads a contact sheet of the attempt's closes and the log, and picks from a fixed list | Once an attempt, when a hold asks for a person | Supervisor's worker pool |
| Program | The timing, judges, responses and budget (`supervisor/program.toml`); the gripper and stuck-rule thresholds are `grasp/grasp.toml`'s | The thresholds read at start by every run, the program only with the supervisor on; each version read recorded in the run's `run.json` | Two versioned files |
| Operator | Takes over with Teleop, confirms Reset and unsure successes | When asked | The page |

### Why a layer, not a second operator

A thread in `so101-control` sending Pause, Rewind and Resume through `Controller.submit`, as a person does, would mix things `serve` keeps apart:

- **Training data.** `control/analysis.py` reads an operator's Pause or Rewind as a person judging the policy, so a supervised retry would count as a human step-in.
- **Who is in charge.** Two senders of commands would have to watch each other: back off once a person acts, and never let a refused Rewind later turn a Pause into a Resume.
- **Where it runs.** It could run only inside `so101-control`, never in the sim.

So the supervisor is the top layer of the policy stack ([`policy-layers.md`](policy-layers.md)): `supervisor/layer.py`'s `Supervisor`, over the perception (and the skill sequencer, for a skills model), with the interface every policy in the stack has. It reads the perception's signals rather than keeping a detector of its own, asks for a person with `Output.asks`, which the state machine takes as the `policy_asked` event (Autonomous to Paused, like a lost link), and hands the arm back after a playback that arrives with a `Handback` that says the arm moved (one that stops short holds and asks instead). It shares code with `serve` rather than calling it: its playback is `control/rewind.py`'s `History` and `Rewind`. Its own ticks are recorded with `control.driver` playback or hold, never as the model's. Each playback's `playback` event has the frames it undid, which analysis treats as suspect, and a `policy_asked` pause counts as an interruption. Neither is ever trained on as policy data.

Everything runs on the control loop's thread except the judges, whose calls the loop sends to a worker pool, one call per judge, and collects on later ticks, so a step never waits. The pool's workers are daemon threads, so a call that hangs never keeps `serve` or `run` from exiting, and closing it cancels the calls still queued. Each judge's request times out at `answer_within_s` (8 s), when the layer stops waiting for it, rather than at failbench's 180 s, and a check that settles cancels its calls that haven't started: a slow provider would otherwise keep all 8 workers busy within about four checks, later checks would queue and come back stale, and the queued calls would still bill. A judge's probability counts only when it is a finite number: a NaN or infinite one is no answer, and so is a chat model's answer whose `wrong` contradicts its `p_wrong` (wrong with `p_wrong` under 0.5, or not wrong with it over 0.5), whose check event's `error` says so (`vision.providers.probability` and `parse_answer`). No answer plays nothing back, never ends a hold (it goes to a person) and never calls a release done. A judge whose key or SDK is missing is left out and named once. Spending is capped at `max_cost_usd` (2 USD) by `vision/budget.py`'s `Budget`, which `so101-failbench run` uses too: each call reserves its model's `estimate` and starts only if that fits, settles to the real cost after, and keeps the estimate if it raises, since it may have billed. Once the cap is reached the checks stop calling and the operator is told once. An exception in the layer's own logic is logged and announced once, and from then on every tick goes straight to the policy; the layer's own logic runs in `Layer.safely`, and every call down the stack outside it, so an exception from the arm or the policy reaches the control loop as it would without the layer.

Every check carries the generation it was asked in, which `pause`, `run` and each hold raise. A watch verdict settles only the hold it was asked for. A person's Pause during a hold ends it (a `hold_ended` event): whoever paused has the arm, so the hold's deadline asks nobody, no escalation is sent, and its watch verdict is stale. The rule that held stays spent for the attempt, because the detector fires each rule once, so after Resume the policy drives and the other rules and the miss ladder still watch: the person saw the arm stuck and handed it back. The same goes after a person resumes from any of the layer's asks: the rules not yet fired, the miss ladder and the done check watch again (`_Attempt.waiting`, which an ask sets and the controller's handback clears), so a resumed attempt can hold once more for each rule not yet fired, and once for each further miss past the ladder: a `miss N` hold always asks, and is no detector rule. The escalation model is still asked once an attempt, so a later ask reads "Paused for you." with no advice, and `attempt_end`'s `asked` still says the layer asked at least once. An after_close verdict is dropped after a pause or resume, a newer close, a carry or release since it was asked, or once it is `after_close_answer_s` (4.5 s) of the policy's time old (the ticks the model drove, at the task's fps), so the playback of even the latest miss starts soon enough to undo the close: the program's loader requires every `miss_rewind_s` to be at least `after_close_s + after_close_answer_s` + 0.5 s (`program.REACH_MARGIN_S`), which covers the ticks between the check being due, sent and acted on, and the jaws' closing motion before the close is seen. Without the margin, a playback exactly as long as the wait, answered at the last moment, stopped with the gripper at 26, one tick short of the close at 25; so the first playback is 6 s against a 1.0 + 4.5 s wait. A check that was due later, such as an after_close check scheduled 1 s after its close, is dropped at the pause or resume too, before it is sent: it would judge the scene after the handback as if it were the one the close made, and could play back motion made since. An after_close check still waiting to be sent when a newer close comes is dropped too, unsent: every check keeps the close it was scheduled for, so two closes 0.4 s apart give one check, `after_close_s` after the second, and at most one playback, never both rungs of the ladder from one grasp. These scheduling rules are one module, `supervisor/schedule.py`, which `so101-failbench candidates` picks its frames by too, so the benchmark scores the moments the live supervisor checks; a test runs both on one attempt and compares the frames. Each check sends the tick's own frames, the ones the benchmark scores for that moment, never one up to 0.2 s older. Each dropped check is logged with `stale` saying why (one never sent also has `sent: false`), and an answer that comes in after its check settled is logged as `late_answer`. After each `run` the playback history starts again, so the layer never drives back past a pose a person set. The history counts each tick the model drove as 1/fps of the task's fps, as the perception's clock does, so the reach rule above holds in ticks at any loop speed: a playback replays one recorded pose a tick, newest first. Were only one of them on ticks, a loop slower than the fps would let a playback stop short of the close. The watch check's lookback ring (one frame every 0.2 s, `FRAME_EVERY_S`), `hold_deadline_s` and `answer_within_s` stay on the loop's clock, since they're about a person's and a provider's real time. A release the judges confirmed stays done across a pause and resume that moved nothing, but a handback that says the arm moved (a Teleop, a Rewind) forgets it, so the layer never asks `done` for a block a person may have taken back out.

## How it acts

Responses get more expensive only as evidence grows, because a false alarm on a cheap action costs seconds while one on Reset costs the attempt:

1. **First miss** (after_close, both Jev and Flash sure it's wrong): play back 6 s of the policy's own motion (the ticks it drove, at the task's fps), newest first, and hand the arm back to the policy with its queue cleared (`miss_rewind_s`). The handback says the arm moved, so the perception forgets the close it undid (`Detector.resumed`): it never counts as a regrasp, and the next close starts a new grasp. One judge alone never plays anything back: a false miss costs a retry, and a wrong Jev answer on a real grasp would undo it. A playback that stops short, with a joint still more than `rollout.TOLERANCE` (4) off `REWIND_SETTLE_S` (1 s) after it should have arrived, hands nothing back: the layer holds where it stopped and asks for a person with an `Ask` of kind `stuck` ("the playback stopped short: gripper is 30 away from the pose it went back to"), and asks the escalation model as in 3, as a person's Rewind that stops short pauses and holds. The `playback` event's cause is `stopped_short`. In `so101 run` the autopilot leaves that pause to a person: nothing ends it on a timer.
2. **Second miss:** play back 8 s, so the policy approaches along a different path; one that stops short holds and asks, as in 1. ACT has no randomness at inference, so a retry from the same pose would very likely repeat the miss. While the ladder has a rung left and both judges answer, a `regrasp_loop` rule is logged as `ignored` and left to it: the ladder sees each miss, and the rule would hold the arm a rung early.
3. **A stuck rule fires, or a third miss:** hold the arm. The watch check looks at each camera now and 2 s before, which cuts false alarms. A sure ok after a stuck rule hands the arm back. Otherwise, or with no answer within `hold_deadline_s` (5 s), the layer asks for a person with an `Ask` of kind `stuck`, the control center pauses, and one escalation follows: the frontier model gets every close of the attempt and picks the advice for that person:
   - take over with Teleop (block in a pose the demos don't cover);
   - check the gripper (it isn't following its commands);
   - Reset as a failure (the scene changed).
   It never orders another identical retry.
4. **After a release:** when Jev (under 0.3) and Flash (under 0.1) both say the block is in the container, the task is done. The layer logs it and lets the policy drive the arm home, then asks for a person with an `Ask` of kind `done` once `Tick.at_rest` says the arm is at rest, or `home_within_s` (10 s) of the policy's time (the ticks it drove, at the task's fps) after the release, so Reset never cuts the return to rest short. A wrong release is told to the operator. The supervisor never labels or ends an attempt itself; in `so101 run` the autopilot ends the attempt at a `done` ask.

A Teleop takeover from a stuck pause is recorded as a correction from the failure state, which is the training data the policies lack.

## Building it

1. **Offline, no arm.**
   - Built: the stuck detector as a pure module (`grasp/detector.py`), tested on made-up traces.
   - Built: a `stuck` moment in failbench, and `so101-failbench candidates`, which picks frames from `serve` and `run` runs where each rule fires and at each check, for labelling.
   - Built: the program format and its loader (`supervisor/program.toml`, `program.py`), and the thresholds' (`grasp/grasp.toml`, `grasp/thresholds.py`).
   - Not built: the compile loop, in which a frontier model proposes questions and a change is kept only if held-out failbench accuracy rises. It needs a labelled benchmark of `serve` and `run` attempts first.
2. **In `serve` and `run`, off by default** (`[supervisor] mode = "off"` in `so101.toml`, or `shadow` or `act`; `--supervisor` sets it for one run, and `run.json` records it):
   - Decided: it supervises only `block-container`, the task `vision.prompt.TASK_NAME` names, because the prompt, its moments and its failure categories describe the block and the jar. `check.check_supervisor` fails the preflight of `serve` and `run` with `shadow` or `act` on another task, judging the mode `--supervisor` sets for the run, and `so101-control` refuses the same; `so101 check` warns.
   - Built: the layer (`supervisor/layer.py`), the policy stack's interface the controller steps it through and the `policy_asked` event, and analysis of its playbacks and pauses, with tests on the fake arm. A playback that stops short holds where it stopped and asks for a person, as a person's Rewind does.
   - Decided: the layer sends the control center no commands. Teleop, Reset and labels stay the operator's, and in `run` the autopilot's; the layer only asks.
3. **On the arm:** measure stuck attempts caught, false alarms, event-to-command latency and the cost of each false alarm. `shadow` has run ([below](#what-the-shadow-runs-measured)); `act` hasn't.

## Open questions

- Whether gripper load, logged in `serve` and `run` runs as `gripper_load` about once a second, separates an empty edge pinch from a held block. Empty carries need it or the wrist camera.
- How well the watch check works on mid-approach frames. Failbench has no such frames yet.
- Whether an 8 s playback gives ACT a different enough approach to break a regrasp loop, and whether 6 s undoes more of a good approach than a retry needs.
- Whether the layer belongs in `sim eval` too. It runs LeRobot's own rollout loop, so it needs a wrapper around LeRobot's policy rather than the policy stack. The sim would let the rules and the playbacks be tested without the arm, though its rendered frames may not suit the judges.
- `hold_deadline_s` and `home_within_s` are unmeasured: no stuck rule held the arm and nothing was called done in the shadow runs. `after_close_answer_s` and `answer_within_s` are set from 24 answers per judge, all on successes.
- Whether a faster chat judge than Gemini 3.8 Flash would allow an after_close wait of about 1.5 s, and playbacks of 3 and 5 s.

## What the shadow runs measured

Four `so101 run --supervisor shadow` runs on the arm, 3 attempts each, of ACT and π0.5, whole-task and skills models, all 12 placing the block ([`results/act-pi05-block-container-supervisor-shadow.md`](../results/act-pi05-block-container-supervisor-shadow.md)). The program's timing and done cutoffs are set from them:

- **Jev** answered all 24 checks in 0.3 to 0.6 s, with p_wrong 0.16 to 0.47 on 12 good grasps and 12 blocks plainly in the jar; 5 of the good grasps were 0.21 to 0.25, just over `sure_ok_below`. `success_decision_below` is 0.3, which passes 9 of the 12 releases.
- **Gemini 3.8 Flash** answered 14 of 24 checks, each 0.00 to 0.02 and right, in 3.0 to 5.2 s; 9 timed out at 5 s and one came back as JSON cut off mid-number. So `answer_within_s` is 8 s, and `after_close_answer_s` 4.5 s, which takes 12 of its 14 answers, with playbacks of 6 and 8 s to reach back past it.
- **The stuck rules** fired once in the 24 attempts of all eight runs (the perception runs with the supervisor off too): `chatter`, on a π0.5 attempt that approached with the jaws shut, opened them, missed once and then placed the block. That's why the rule counts crossings from the attempt's first close.
- **Carries** were all the steady-hold kind, at 18, with 1 to 14° of elbow travel: none of these policies lifts with the elbow before the hold is seen.
- **Cost** was about $0.02 per 3-attempt run.

All of it was measured on successes only.
