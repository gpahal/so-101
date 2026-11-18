# `so101 run` on the control center: serve plus an autopilot

Status: built. `so101 run` has made 24 attempts on the arm
([`results/act-pi05-block-container-supervisor-shadow.md`](../results/act-pi05-block-container-supervisor-shadow.md)),
all started and ended at the terminal with no page in control, so the autopilot has started and ended none
there; its rules are tested on the fake arm. How to use it is in the README, under
[Training your own task](../README.md#training-your-own-task) (step 6).

`run` and `serve` are one program, `so101-control`, with one difference: in `run` an **autopilot**
starts and ends the attempts. Everything else is shared: the policy stack (skills, supervisor,
perception, [`policy-layers.md`](policy-layers.md)), the control page and keys, the safety rules, and the
recording into `data/<task>/runs/`, so `review --runs`, `runs`, `build --with` and the skills builds read
`run` sessions as they read `serve` ones.

## The command

- `--duration`: seconds of policy time per attempt (default `[control] attempt_limit_s`, which `serve` uses too).
- `--duration` and `--reset-time` must be finite numbers, 0 or more, and so must a task's `reset_time_s`
  (`tasks.load`): `run` stops before its checks, naming the option. NaN passes a `< 0` check and would put NaN
  in the page's strict JSON status. `--duration 0` runs until you end the attempt.
- Everything else as `serve`, `--supervisor` included (it sets `[supervisor] mode` for one run). `run.json`
  records `"mode": "run"`, the supervisor's mode and the autopilot's settings.
- Like `serve`, it passes on to `so101-control` only LeRobot's `--inference.*`, `--policy.*`, `--robot.*` and
  `--rename_map`, and stops on any other argument, naming it.

## The autopilot (`control/autopilot.py`)

A small piece of the control process whose commands carry source `autopilot`, as the terminal's and the
page's carry theirs, so the state machine still decides what's allowed. Each tick, the controller first
applies the commands people sent since the last one, in the order they came, and checks the page's link.
Then it steps the autopilot with that tick's facts (`autopilot.Facts`): the state, whether the arm is at
rest, the attempt and whether it's open, the kind of the ask the arm paused on (when the policy stack
asked), why no page in control watches the arm (`unwatched`, None while one does), the commands people
sent that tick from the terminal, a signal or the page in control (accepted or refused, a Quit only when
accepted), each with the attempt that was open when it arrived, why Resume would be refused, and whether
the arm was in Fault since the last attempt opened (`after_fault`). The autopilot's command is carried
out at once, in the same tick (`Controller.pilot_command`), before the arm is driven. So a person's
command always comes first, and the autopilot decides from facts that already include it: a person's
Resume in the tick a `done` ask would end the attempt keeps it, and a Hold sent just before the next
start is due holds it. It reads no event log and runs no thread of its own. A command it sent that was refused is said once, and not sent again until a person
acts, the state changes, or an attempt opens or ends.

1. **The first attempt is a person's start.** The autopilot never sends the first Resume: it says once
   that attempt 1 waits for you, and `r` at the terminal or Resume on the page starts it. `so101 run` can
   take minutes to reach its loop (the model's download, a remote server, the cameras), and this way
   someone is at the arm when it first moves. The autopilot then runs that attempt like any other: the
   command that starts an attempt doesn't hand it over (item 4).
   **So is the first attempt after any fault.** After a fault (an overload, lost power, a lost camera, a
   policy error), a person's Reset drives the arm to rest, and the autopilot then waits for `r` or Resume,
   saying once that the next attempt waits for you, so the policy never drives the arm again without
   someone choosing to restart it. The controller tells it whether the arm was in Fault since the last
   attempt opened (`Facts.after_fault`, from `Controller.fault_since_attempt`, set as the state becomes
   Fault and cleared as an attempt opens), because a fault at the end of one tick and a Reset at the start
   of the next take the state from Fault to Returning before the autopilot steps, so it never sees Fault.
   A finished run still quits at rest after a fault (item 5).
2. **Between attempts**, resting at the rest pose with no attempt open: `reset_time_s` after the arm
   arrived at rest it submits Resume, which opens an attempt, moves to the
   demos' start pose and starts the policy. A person's command between attempts restarts that wait. It
   starts nothing while the arm rests away from the rest pose (a move to rest paused or stopped short), in
   Fault, after a fault until a person starts an attempt (item 1), or while the page's Hold is on
   (`autopilot_hold`): it says why once, in the terminal and on the page, and waits. Hold and Continue are taken any time the autopilot runs, so they work with
   `--reset-time 0`, where the arm rests for a tick or two between attempts. A Hold during an attempt
   stops the next attempt, not this one: the arm isn't paused, the attempt isn't taken over, and it ends
   as it would have. A held autopilot still quits at rest once `--episodes` attempts have ended.
   **It starts the arm moving only while a page watches it.** With no page in control, or the page in control
   out of reach (no poll for `link_timeout_s`, or hidden), whatever `on_link_loss` says, the state machine
   refuses the autopilot's Resume and Reset (`Facts.unwatched`, `states.step`, beside the missing-device
   check, so a Fault or a missing device is still the reason given first): it says the refusal once ("No
   control page is in control, so the autopilot moves nothing unwatched..." or "The controlling page (who)
   is out of reach, so the autopilot starts nothing unwatched...") and starts nothing. Its Quit, from rest
   at the rest pose, moves nothing and isn't refused. When a page in control watches again (a page takes
   control, or the controlling page's next visible poll, or a command from it other than `answering`,
   which isn't proof the page is in sight), the wait starts again: the next Resume is `reset_time_s`, and at
   least `link_timeout_s` + 0.5 s (`autopilot.RELINK_MARGIN_S`), after that, so unlocking the phone moves the
   arm only that long after. The floor is for a restore no poll follows: a command or a visible poll that
   reaches the server after `page_hidden`, or a `take_control` that races a hide, hears the page. Every
   command from the page in control must hear it, since that re-arms the timeout: without it a page's
   Resume sent after a hide would run the policy with the link lost and never pause again. Such a link is
   lost again `link_timeout_s` after it came back, and the link check runs before the autopilot in each
   tick, so that loss comes before the next start; the 0.5 s covers the two reading the clock separately.
   `r` at the terminal, or a page's Resume, still starts an attempt early. `so101 run` refuses
   `on_link_loss = "continue"` before its checks (`commands.serve.cmd_run`), since an attempt the autopilot
   started would keep driving after the page is lost. The state machine's refusal stays as a second guard:
   `Link.lost` is set whatever `on_link_loss` says, so even with `"continue"` the autopilot starts nothing
   while the link is lost. A page's first command takes control when no page has it
   (`commands._control`), and nothing gives it back, so a page used once and then closed holds every later
   start until it, or another page, is back; `r` at the terminal starts one attempt each time.
3. **An attempt ends** at its time limit (the controller's, unlabelled, without the outcome prompt
   waiting for an answer), or on a `done` ask: the supervisor sure the task is done once the arm is home,
   and the autopilot ends the attempt unlabelled with Reset at once; the verdict is in the run's events.
   Every other pause it leaves to a person, saying once in the terminal and on the page that it does and
   sending nothing: the policy stack asked for a person (the supervisor stuck and not cleared), a move to
   the start pose or back to rest stopped short (a Reset would drive the arm at whatever blocked it), or
   the link was lost. An unattended run stalls there until someone answers, which is the point: nothing
   moves the arm on a timer. `r` resumes the attempt; the right arrow or Reset ends it, and the autopilot
   carries on once it has.
   **An attempt ever unwatched while open is a person's for good.** Once no page in control watches the
   arm while an attempt is open (none is in control, as when `r` at the terminal starts attempt 1 with no
   page open, or the controlling page's link is lost), the autopilot says so once, with why, and sends
   nothing more for that attempt, a `done` ask's Reset included, even after a page watches again: it never
   starts the arm moving (Resume, or Reset from a pause) unless a page in control watches. A return to rest
   it already started finishes even if the page is lost, as every return to rest does, because stopping
   halfway would leave the arm held in mid-air.
4. **A person's Pause, Resume, Rewind, Teleop, Reset or Quit**, sent while an attempt is open, from the
   keys, a signal or the page in control, hands that attempt to them, whether the state machine accepted
   it or not: a Pause refused because the arm is already paused still says a person is at the arm and
   wants it held. A Quit counts only when it's accepted, so a page's Quit, which is always refused, never
   stops the autopilot, and nothing from a page that isn't in control counts: a forgotten tab can't take
   attempts over or push back the next start. The command that starts an attempt doesn't hand it over, but
   one after it does, even in the same tick: each command carries the attempt open when it arrived
   (`commands.intake`), so a Resume, Pause and Resume queued together hand the attempt the first Resume
   opened to the person. So does their start at saying how the attempt went: the terminal's right-arrow question, or the page's
   Reset sheet opening, each of which sends `answering` once, whether the person then answers or cancels.
   The autopilot doesn't resume or end that attempt, and carries on with the next once it has ended.
   Check pose, Run checks, a step-in label and taking control on the page don't hand it over.
5. **It stops** after `--episodes` attempts have ended, by submitting Quit from rest at the rest pose;
   with the arm anywhere else, it says the run is over and waits for a person's Reset. Any Quit the control
   center takes from a person (Esc or a signal) stops it for good too, including one that only asks to
   confirm, away from rest, and one a Pause then cancels: whoever pressed Esc wanted the run to stop, so
   it fails safe, and restarting `so101 run` carries on. A page's Quit is always refused and doesn't stop
   it. The state machine refuses the autopilot a Quit that would exit leaving a silent follower's motors
   as they are: that's a person's call.

The page's status has `autopilot`: `made` (attempts ended) of `of`, `next_in_s` while it counts down to
the next attempt, `waiting` (why it waits, or null) and `held`. The page shows them in a card with a
Hold or Continue button. With `--duration 0` the status's `limit_s` is null and the page shows no limit.

The arrows mean what they mean in `serve`: the right arrow ends the attempt and asks how it went, the left
arrow ends it as one to ignore.

## Tests

- `tests/test_autopilot.py`: every rule above on the controller's fake arm, with a page in control unless a
  test says otherwise.
- `tests/test_controller.py`'s `LinkLossTest`: the state machine refuses the autopilot's Resume and Reset
  while no page in control watches, with either `on_link_loss`, accepts the same commands from `terminal` and
  `page`, never refuses its Quit at rest, and still gives a Fault first.
- `tests/test_cli.py` and `tests/test_settings.py`: the finite-number checks on `--duration`, `--reset-time`
  and `reset_time_s`, and the refusal of `on_link_loss = "continue"`.
- `tests/test_process.py` and `tests/test_serve_preflight.py`: `so101-control` in run mode, `run.json`, and
  `--duration 0` through the settings check and the strict JSON status.
