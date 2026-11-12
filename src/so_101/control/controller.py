"""The control loop: the only code that touches the follower, its cameras, the leader and the policy engine.

Other threads (the terminal reader and the control page's web server) only
`submit` commands, report page polls with `heartbeat`, and read `status`. Each `tick` first applies
the commands waiting since the last tick, then reads the arm and sends exactly one goal: the
policy's action, the leader's pose in Teleop, the next step of a scripted move, or the hold target.

The controller keeps the loop, the commands and how the state machine's transitions are carried
out. The parts it drives sit beside it, and are used only through it:

- `follower`: reading the follower and its cameras, its motors' power marks, and holding it in Fault.
- `leader`: the optional leader, connected while the arm rests or is paused, with Handover and the
  rewind's mirror.
- `rewind`: the open attempt's history of follower poses, played backwards.
- `attempt`: one attempt, from when it opens until it closes, and the run's summary.
- `commands`: taking commands in: what each takes, page control, dispatch and Space's toggle.
- `tools`: Run checks and Check pose.
- `status`: the status the page and the terminal read, built from the controller's state.
- `wording`: what the operator is told as the arm moves between states.

The follower, the policy driver and the clock are passed in, so tests run the controller on
LeRobot's motor code over a fake bus, with a scripted engine. The recorder is passed in too, so
tests write a run to a temporary directory, and so is `presence`, the device watch's latest answer.

Page control and the link are `link`'s; the loop judges the controlling page's polls with its own
clock. The terminal is never refused for lack of control. The loop's rate and the previews it sheds
are `shedding`'s. Faults and quitting follow AGENTS.md's invariants: `states` alone decides what is
allowed, from the facts the controller gives it (`_facts`).
"""

import contextlib
import queue
import shutil
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from typing import NamedTuple

from so_101 import rollout
from so_101.check import DISK_FAIL_GB, DISK_WARN_GB
from so_101.report import Report
from so_101.settings import Control
from so_101.control import rewind, status, wording
from so_101.control.attempt import COUNTS, LIMITED, Attempt
from so_101.control.commands import LABELS, Result, Toggle, intake, stale_prompt
from so_101.control.follower import Failure, Follower
from so_101.control.leader import Handover, Leader, LeaderLost
from so_101.control.link import Link
from so_101.control.previews import Previews
from so_101.control.recorder import OUTCOMES, Phase, Recorder
from so_101.control.rewind import DRIVEN, Rewind
from so_101.control.shedding import Pace
from so_101.control.states import (
    CONFIRM_S,
    STEP_INS,
    Action,
    Command,
    Event,
    Facts,
    State,
    Tool,
    Transition,
    allowed,
    step,
    tool_refusal,
)
from so_101.control.tools import pose_report, run_checks


DISK_READ_S = 1.0  # how often the loop reads the free disk space, before a new attempt is allowed


class Driven(NamedTuple):
    """What `Controller._drive` did this tick."""

    policy_action: dict | None = None  # the policy's raw action, if one reached the arm this tick
    leader_pose: dict | None = None  # the leader's pose, if it was read
    event: Event | None = None  # what became of a scripted move, applied once the frame is recorded


def _move_event(arrived: bool | None) -> Event | None:
    """What a scripted move's step returned, as the state machine's event: None while it still moves."""
    if arrived is None:
        return None
    return Event.ARRIVED if arrived else Event.STOPPED_SHORT


def _last_line(failure: str) -> str:
    """The line of a failure's traceback that names the error."""
    return failure.strip().splitlines()[-1]


class Controller:
    def __init__(
        self,
        robot,
        policy,
        *,
        fps: float,
        rest_pose: dict[str, float],
        control: Control,
        leader: Callable[[], object],
        checks: Callable[[dict, dict], Report],
        start_pose: dict[str, float] | None = None,
        policy_name: str = "",
        clock: Callable[[], float] = time.monotonic,
        recorder: Recorder,
        presence: Callable[[], dict[str, str | None]],
        notify: Callable[[str, str], None] | None = None,
        operator: str | None = None,
    ) -> None:
        """`robot` is LeRobot's robot wrapper (ThreadSafeRobot) with hold configured. `policy` is an
        EngineDriver. Poses are in the arm's units, keyed like its actions. `rest_pose` is the
        `[control]` rest pose, already converted from raw steps with `pose_from_steps`. Without
        `start_pose` (no build), attempts start from wherever the arm is. `notify(message,
        state)` hears every message, with the new state when it announces one. `recorder` records the
        run. `control` is so101.toml's `[control]` section, as `settings.load()` gives it; so101.toml
        says what each setting does.

        `leader()` makes a new, unconnected LeRobot leader (SO101Leader) in the follower's units. It
        is called whenever the leader is to be connected: while the arm rests or is paused, the watch
        reports it present (or doesn't watch it) and its calibration file exists.

        `operator` is the name given with `so101 serve --operator`, or None. It is recorded on every
        event a person's action causes: command, label, outcome, control_taken, check and pose.

        `presence()` returns the device watch's latest answer: for each watched device, None when it
        is connected, or why it is missing and the fix. It must not block. The follower and every
        camera, by the robot's camera keys (so101.toml's camera names), are required.

        `checks(buses, cameras)` runs `so101 check`'s read-only checks on the devices the loop has
        open: the arms' buses by arm name (the follower, and the leader when it's connected) and the
        cameras by camera name. It returns their Report.

        `rest_pose` must name exactly the follower's motors, or `at_rest` would never check a joint
        it left out: `check.check_control` refuses any other before `so101-control` connects."""
        self.recorder = recorder
        self.operator = operator
        self.policy = policy
        self.fps = fps
        self.rest_pose = rest_pose
        self.start_pose = start_pose
        self.policy_name = policy_name
        self.control = control
        self.clock = clock
        self.notify = notify or (lambda message, state: None)
        self.done = False
        # Set with `done` when the quit found the follower silent: the exit leaves its motors as they are
        # (`main.disconnect_follower`), so a follower that answers again just then is never turned off.
        self.leaves_motors = False
        self.presence = presence
        inner = robot.inner
        self.required = ("follower", *inner.cameras)
        self.previews = Previews(control.preview_fps, inner.cameras, clock) if control.preview_fps else None
        self._pace = Pace(fps, self.previews, clock, self.recorder.event, self._say)

        self._commands: queue.SimpleQueue[tuple[str, str, str | None, str | None, dict, Future]] = queue.SimpleQueue()
        self.link = Link(control, clock, self.recorder.event, self._say, operator)
        self._state = State.RESTING
        self._quitting = False
        # Until when a confirmation to exit where the arm is stays pending, or None: `exit_here` exits only before then.
        self._confirm_until: float | None = None
        self._attempt = 0  # the number of the attempt opened last
        self._open: Attempt | None = None  # the open attempt: from OPEN_ATTEMPT until it is saved or discarded
        # The attempt closed last, kept so the page still shows how it went and a later outcome can label it.
        self._closed: Attempt | None = None
        self._attempts: list[Attempt] = []  # the closed attempts that were saved, for the run's summary
        self._move: rollout.Move | None = None
        self._rewind: Rewind | None = None
        self._fault_error: str | None = None
        self._presence: dict[str, str | None] = {}
        self._disk: str | None = None  # why a new attempt can't be recorded, as last read
        self._disk_read_at: float | None = None
        self._disk_warned = False
        self._prompt: int | None = None  # the attempt a time limit ended, while it waits for its outcome
        self._told_dropped = False  # the operator has been told the video encoder dropped frames
        # The last fault saved an attempt (`_closed`), until Reset leaves Fault: an outcome given with
        # that Reset labels it.
        self._faulted = False
        self._checks = checks
        # The follower's step cap, which caps the leader's goals too.
        self.leader = Leader(leader, clock, self._say, inner.config.max_relative_target)
        self._toggle = Toggle(clock)
        self._last_tick = clock()
        self._follower = Follower(robot, rollout.positions(robot, rest_pose), clock, self._say,
                                  self.previews, log=self.recorder.event)
        self._follower.mark()
        # The status as of the last tick, and the page in control then: replaced whole, in one
        # assignment, so another thread never reads one without the other.
        self._status: tuple[dict, str | None] = (self._build_status(), None)

    # --- The interface other threads use --------------------------------------------------------

    def submit(self, command: str, source: str, *, client: str | None = None,
               who: str | None = None, **options) -> Future:
        """Queue a command for the next tick. The future resolves to its Result once applied.
        `source` (terminal, page or signal) is recorded with it. A page's command needs `client`, the
        id the page made for itself, and `who`, how the status and events name its caller (a
        Tailscale login, "this Mac" or an IP address). The commands, their options, page control and
        `toggle` are in `commands`; when a Reset's outcome is refused is `states._reset`'s.
        """
        future: Future = Future()
        self._commands.put((command, source, client, who, options, future))
        return future

    def heartbeat(self, client: str, visible: bool = True) -> None:
        """A page polled the status, and said whether it's in sight. The next tick hears it, if it's
        the page in control and in sight (link.py)."""
        self.link.heartbeat(client, visible)

    def events(self, after: int, limit: int) -> list[dict]:
        """The run's events numbered above `after`, oldest first; at most the newest `limit`."""
        return self.recorder.tail.since(after, limit)

    def status(self, client: str | None = None) -> dict:
        """The state as of the last tick, shaped like the control page's /status, as the page
        `client` sees it: a page that doesn't control the robot has every command refused, with the
        reason. Without `client`, `you_have_control` is None."""
        built, controller = self._status
        return status.seen_by(built, controller, client)

    def run(self, sleep: Callable[[float], None] = time.sleep) -> None:
        """Tick at `fps` until a quit finishes.

        Ticks are due on fixed deadlines, so a sleep that overshoots (about 7 ms on macOS) is taken
        out of the next wait instead of slowing every tick: sleeping for the period minus the tick's
        own time would run the loop at about 24 Hz instead of 30. A loop more than one period behind (a
        slow tick, or Run checks) starts counting again from now rather than ticking in a burst."""
        period = 1 / self.fps
        due = self.clock()
        while not self.done:
            self.tick()
            due += period
            now = self.clock()
            if now - due > period:
                due = now
            sleep(max(due - now, 0))

    def close(self) -> None:
        """Finish the run's files, and disconnect the leader. An attempt still open is not saved.

        After a quit (`done`), the event log ends with `run_end`: when the run finished, how many
        attempts it opened and how many episodes it saved. A run that stopped any other way has none."""
        try:
            self.recorder.close(run_end={"attempts": self._attempt} if self.done else None)
        finally:
            if self.leader.connected:
                self.leader.drop(None)

    @property
    def at_rest(self) -> bool:
        """Every joint is within rollout.TOLERANCE of the rest pose, as last measured."""
        return rollout.farthest(self._follower.present, self.rest_pose)[1] <= rollout.TOLERANCE

    # --- The loop --------------------------------------------------------------------------------

    def tick(self) -> None:
        """One tick of the loop. It never raises: any error in it is a fault, which holds the arm,
        and a hold that fails too is tried again on the next tick. An error in a step before the
        commands or after the goal faults without skipping the rest of the tick, so commands such
        as Reset and Quit, and the hold in Fault, still get through while it lasts.

        The status is published once, last, and only then are the commands taken this tick answered,
        so a page's reply sees its command's effect. The answers go out whatever happens in the tick,
        so no command's caller is left waiting."""
        answers: list[tuple[Future, Result]] = []
        try:
            try:
                self._tick(answers)
            except Exception as error:
                self._fault(error)
            self._guard(self._publish)
        finally:
            for future, result in answers:
                future.set_result(result)

    def _tick(self, answers: list[tuple[Future, Result]]) -> None:
        self._guard(self._read_presence)
        self._guard(self._watch_leader)
        self._guard(self._maintain_policy)
        self._guard(self._read_disk)
        while True:
            try:
                command, source, client, who, options, future = self._commands.get_nowait()
            except queue.Empty:
                break
            try:
                result = intake(self, command, source, client, who, options)
            except Exception as error:  # carrying out the command failed: as any failure, a fault
                self._fault(error)
                result = Result(command, False, f"{command.capitalize()} failed, so the arm holds in Fault: {self._fault_error}")
            answers.append((future, result))
        if not self.done:
            self._guard(self._check_link)  # may pause the policy
        previous, now = self._last_tick, self.clock()
        self._last_tick = now
        if not self.done:
            if self._state is State.FAULT:
                self._guard(lambda: self._follower.hold_in_fault(self.policy))
            else:
                self._guard(lambda: self._drive_tick(now - previous))
            self._guard(self._pace.tick)

    def _guard(self, step: Callable[[], object]) -> None:
        """Run one step of the tick. An error in it is a fault."""
        try:
            step()
        except Exception as error:
            self._fault(error)

    def _held_up(self) -> None:
        """A step known to hold up the loop just ran: the time it took isn't motion, nor the loop's
        own rate. The next tick counts from now, and the rate starts a new window rather than log a
        slow loop and shed previews."""
        self._last_tick = self.clock()
        self._pace.restart()

    def _read_presence(self) -> None:
        self._presence = self.presence()

    def _watch_leader(self) -> None:
        """Drop the leader when the device watch loses it; connect it when it's back and the arm is
        resting or paused. A try to connect holds up the loop for up to a second."""
        tried_at = self.leader.tried_at
        try:
            lost = self.leader.watch(self._presence.get("leader"), self._state in (State.RESTING, State.PAUSED))
        finally:
            if self.leader.tried_at != tried_at:
                self._held_up()
        if lost:
            self.leader.drop(lost)
            self._apply(Event.LEADER_LOST)

    def _maintain_policy(self) -> None:
        """Let the policy driver reconnect a failed remote engine; announce what it says."""
        try:
            if (message := self.policy.maintain()) is not None:
                self._say(message)
        except Exception as error:  # reconnecting must never stop the loop
            self._say(f"Reconnecting the policy failed: {type(error).__name__}: {error}")

    def _read_disk(self) -> None:
        """Read the free disk space where the run is recorded, at most once a second, into `_disk`:
        why a new attempt can't be recorded, or None. Under DISK_FAIL_GB new attempts are refused;
        under DISK_WARN_GB the operator is told once. A disk that can't be read refuses them too."""
        run_dir = self.recorder.run_dir
        now = self.clock()
        if run_dir is None or (self._disk_read_at is not None and now - self._disk_read_at < DISK_READ_S):
            return
        self._disk_read_at = now
        try:
            free_gb = shutil.disk_usage(run_dir).free / 1e9
        except OSError as error:
            self._disk = (f"Can't read the free disk space where this run is recorded ({error}): new attempts are "
                          "refused until it can be read. Is the disk still connected?")
            return
        self._disk = (f"Only {free_gb:.1f} GB free for this run's recording: a new attempt needs "
                      f"{DISK_FAIL_GB} GB. Free up disk space.") if free_gb < DISK_FAIL_GB else None
        if self._disk is None and free_gb < DISK_WARN_GB and not self._disk_warned:
            self._disk_warned = True
            self._say(f"Only {free_gb:.1f} GB free for this run's recording: free up disk space before it "
                      f"falls under {DISK_FAIL_GB} GB, where new attempts are refused.")

    def _check_link(self) -> None:
        """Hear the controlling page's polls since the last tick, or lose the link after
        `link_timeout_s` without one. With on_link_loss "pause", what the state machine pauses on a
        lost link (`states._link_lost`) pauses."""
        if self.link.check(step(self._state, Event.LINK_LOST, self._facts()).accepted):
            self._apply(Event.LINK_LOST)

    def _drive_tick(self, seconds: float) -> None:
        observation = self._follower.observe(self._open is not None)
        state, arm = self._state, self._follower.arm
        try:
            policy_action, leader_pose, event = self._drive(observation, seconds)
        except LeaderLost as lost:
            self.leader.drop(str(lost))
            if state is State.TELEOP:  # in Handover, `_drive` already held the follower this tick
                self.policy.hold(arm)  # the follower holds rather than chase a stale pose
            policy_action, leader_pose, event = Driven(event=Event.LEADER_LOST)
        if state is State.AUTONOMOUS and (failure := self.policy.failure):
            raise Failure(f"The policy failed: {_last_line(failure)}", detail=failure)
        if (attempt := self._open) is not None:
            if leader_pose is None and self.leader.connected:
                leader_pose = self.leader.pose_for_frame()
            # The policy drove only on a tick its action reached the arm: one waiting for an action (a
            # cloud GPU's chunk, say) drove nothing, though the phase is Autonomous.
            attempt.policy_drove |= policy_action is not None
            frame = self.recorder.frames
            self.recorder.add_frame(observation, arm.goal, Phase[state.name], policy_action=policy_action,
                                    intervention=state is State.TELEOP and attempt.policy_drove, leader=leader_pose)
            if state in DRIVEN:
                attempt.history.add(seconds, dict(self._follower.present), frame)
            attempt.extend_timeline(state, 1 / self.fps)
        if event is None and (limit := LIMITED.get(state)):
            limit_s = getattr(self.control, limit)
            if state is State.AUTONOMOUS and policy_action is None:
                self._open.waited_s += seconds
            if self._open.count(limit, seconds, limit_s):
                self._say(f"Attempt {self._attempt} reached {limit_s:g} s of {COUNTS[limit]} ({limit}).")
                event = Event.TIME_LIMIT
        if event is not None:
            self._apply(event)

    def _drive(self, observation: dict, seconds: float) -> Driven:
        """Send this tick's goal, `seconds` after the last tick's. Raises LeaderLost when the leader
        fails in Handover or Teleop."""
        arm, present = self._follower.arm, self._follower.present
        if self._state is State.AUTONOMOUS:
            try:
                return Driven(policy_action=self.policy.step(arm, observation))
            except Exception as error:
                if self._follower.blame(error) is None:  # the policy's side failed, not a goal to the arm
                    self.policy.raised = traceback.format_exc()  # Resume is refused from now on
                raise
        if self._state is State.TELEOP:
            pose = self.leader.read()
            arm.send_action(pose)  # through max_relative_target, like every goal
            return Driven(leader_pose=pose)
        if self._state in (State.STARTING, State.RETURNING):
            return Driven(event=_move_event(self._move.step(arm, present)))
        if self._state is State.REWINDING:
            arrived = self._rewind.step(arm, present, seconds)
            return Driven(leader_pose=self.leader.follow_mirror(arm.goal), event=_move_event(arrived))
        self.policy.hold(arm)
        if self._state is State.HANDOVER:
            arrived = self.leader.handover.step(present)
            return Driven(leader_pose=self.leader.handover.pose, event=_move_event(arrived))
        return Driven()

    def _fault(self, error: BaseException) -> None:
        """Hold the arm, save any open attempt where it is as `fault`, and allow only Reset and Run checks.

        The hold goes first, before any file is written (`Follower.fault`: a follower that may have
        lost power gets no goal). Already in Fault, it doesn't hold again: `hold_in_fault` holds once
        every tick. The same error again while in Fault isn't logged again.

        Nothing here raises. When handling the fault fails (a bug, since the event log and the save
        don't raise), it's announced, and if the state didn't get to Fault, it's forced there: the
        policy pauses, the open attempt is dropped unsaved, and a quit waits for y at the robot's
        terminal, since the arm can't be driven to rest now."""
        try:
            device, message = self._follower.fault(error, self.policy, hold=self._state is not State.FAULT)
            repeated = self._state is State.FAULT and message == self._fault_error
            self._fault_error = message
            if not repeated:
                detail = error.detail if isinstance(error, Failure) else "".join(traceback.format_exception(error))
                self.recorder.event("fault", error=message, device=device, **({"traceback": detail} if detail else {}))
            self._apply(Event.FAULT)
        except Exception as failure:
            self._force_fault(error, failure)

    def _force_fault(self, error: BaseException, failure: Exception) -> None:
        """Handling the fault `error` failed with `failure`: get to Fault anyway. Each step is tried
        on its own, and none raises."""
        with contextlib.suppress(OSError, ValueError):  # stderr is gone with a closed terminal
            traceback.print_exception(failure)
        if self._state is not State.FAULT:
            with contextlib.suppress(Exception):
                self.policy.pause()
            self._state = State.FAULT
            self._fault_error = self._fault_error or f"{type(error).__name__}: {error}"
            # The arm can't be driven to rest now, so a quit waits for y at the terminal.
            self._confirm_until = self.clock() + CONFIRM_S if self._quitting else None
            self._quitting = False
            self._move = self._rewind = None
            self._faulted = False  # the open attempt, if any, is dropped unsaved
            self.leader.handover = self.leader.mirror = None
            if (attempt := self._open) is not None:
                self._open, self._closed = None, attempt
                with contextlib.suppress(Exception):
                    self.recorder.discard_attempt("fault")
        with contextlib.suppress(Exception):
            confirm = f" {wording.SUPPORT_AND_QUIT}" if self._confirm_pending else ""  # a quit was under way
            self._say(f"Fault: {self._fault_error} Handling the fault failed too: {type(failure).__name__}: "
                      f"{failure}. Only Reset and Run checks are allowed.{confirm}", self._state.value)

    # --- What the state machine needs to know ---------------------------------------------------

    @property
    def _confirm_pending(self) -> bool:
        """A Quit asked to confirm exiting where the arm is under CONFIRM_S ago, with no state change since."""
        return self._confirm_until is not None and self.clock() < self._confirm_until

    def _facts(self, source: str | None = None, outcome: str | None = None) -> Facts:
        attempt, devices = self._open, self._devices()
        return Facts(
            underway=attempt is not None and attempt.underway,
            quitting=self._quitting,
            confirm_pending=self._confirm_pending,
            source=source,
            at_rest=self.at_rest,
            attempt_open=attempt is not None,
            follower_missing=devices["follower"],
            follower_silent=self._follower.lost,
            device_missing=next((reason for reason in devices.values() if reason), None),
            fault=self._fault_error,
            leader=self._leader_reason(),
            teleop_from_start_pose=self.control.teleop_from_rest != "rest",
            teleop_next=attempt is not None and attempt.teleop_next,
            rewind=self._rewind_reason(),
            policy=self._policy_reason(),
            recording=self.recorder.failure or self._disk,
            outcome=outcome is not None,
            fault_saved=self._faulted,
        )

    def _devices(self) -> dict[str, str | None]:
        """Each required device: None when it is ok, or why it is missing and the fix."""
        read_errors = self._follower.read_errors
        problems = {device: self._presence.get(device) or read_errors.get(device) for device in self.required}
        problems["follower"] = problems["follower"] or self._follower.alarm
        return problems

    def _leader_reason(self) -> str | None:
        """Why Teleop can't use the leader, or None when it is connected."""
        return self.leader.reason(self._state in (State.RESTING, State.PAUSED))

    def _rewind_available(self) -> float:
        """Seconds of motion a Rewind now could still undo."""
        return rewind.available(self._open.history, self._rewind) if self._open is not None else 0.0

    def _rewind_reason(self) -> str | None:
        """Why Rewind has nothing to play back, or None when it has."""
        if self._open is None:
            return "No attempt is open, so there is nothing to rewind."
        return rewind.refusal(self._open.history, self._rewind, self._attempt)

    def _policy_reason(self) -> str | None:
        """Why the policy can't run now: it failed, for good when it's local, or until it has
        reconnected when it's remote. None when it can run."""
        if not (failure := self.policy.failure):
            return None
        if retrying := self.policy.retrying:
            return f"The policy failed: {_last_line(failure)}. {retrying}"
        return f"The policy failed and can't run again until so101 serve restarts: {_last_line(failure)}"

    def _label_reason(self) -> str | None:
        """Why `label` is refused now, or None when it would be accepted."""
        if self._open is None:
            return "No attempt is open to label."
        if self._open.step_ins == 0:
            return "Nothing to label: nobody has stepped in on the policy in this attempt."
        if self._state is State.AUTONOMOUS:
            return "The policy is driving again. Label a step-in while the arm is paused, rewinding or in Teleop."
        return None

    # --- Commands, as `commands.intake` hands them on ----------------------------------------------

    def _tool(self, tool: Tool) -> Result:
        """Run checks or Check pose."""
        if reason := tool_refusal(self._state, tool, self._facts()):
            return Result(tool.value, False, reason)
        if tool is Tool.CHECK:
            # Inside this tick: the arm holds its last goal meanwhile.
            report = run_checks(self._checks, self._follower, self.leader, self._leader_reason())
            self._held_up()
        else:
            report = pose_report(self._follower.present, self.start_pose)
        problems = [{"group": r.group, "status": r.status, "detail": r.detail}
                    for r in report.results if r.status != "PASS"]
        self.recorder.event(tool.value, summary=report.summary(), fails=len(report.failures),
                            warns=len(report.warnings), problems=problems, operator=self.operator)
        self._say(f"{'Checks' if tool is Tool.CHECK else 'Pose'}: {report.summary()}")
        return Result(tool.value, True, report=report)

    def _label(self, outcome: str) -> None:
        """Record `outcome` for the attempt closed last: an `outcome` event."""
        closed = self._closed
        closed.outcome = outcome
        self.recorder.event("outcome", attempt=closed.number, outcome=outcome, episode=closed.episode,
                            operator=self.operator)
        self._say(f"Attempt {closed.number} labelled {outcome}.")

    def _answer(self, options: dict) -> Result:
        """Label the attempt a time limit ended, once it is at rest."""
        outcome = options.get("outcome")
        if outcome not in OUTCOMES:
            return Result("outcome", False, f"Unknown outcome {outcome!r}: use {', '.join(OUTCOMES)}.")
        if self._prompt is None:
            return Result("outcome", False, "No attempt is waiting for an outcome.")
        if "for_attempt" in options and (reason := stale_prompt(options["for_attempt"], self._prompt)):
            return Result("outcome", False, reason)
        self._label(outcome)
        self._prompt = None
        return Result("outcome", True)

    def _label_step_in(self, options: dict) -> Result:
        """Record why the operator stepped in last: a `label` event."""
        label = options.get("label")
        if label not in LABELS:
            return Result("label", False, f"Unknown label {label!r}: use {', '.join(LABELS)}.")
        if reason := self._label_reason():
            return Result("label", False, reason)
        self._open.label = label
        self.recorder.event("label", reason=label, operator=self.operator)
        self._say(f"Step-in labelled: {label.replace('_', ' ')}.")
        return Result("label", True)

    # --- Applying the state machine's transitions -----------------------------------------------

    def _apply(self, given: Command | Event, outcome: str | None = None, source: str | None = None) -> Transition:
        """Take the state machine's step for `given`, a command from `source` or an event. The state
        changes first, and then every action is carried out even when one fails, so a failed write or
        save can't leave the transition half done. The first failure is raised once the transition is
        complete."""
        transition = step(self._state, given, self._facts(source, outcome))
        if not transition.accepted:
            return transition
        if given is Event.FAULT:
            self._faulted = False  # set again if this fault saves an attempt
        before, shortfall = self._state, wording.shortfall(self) if given is Event.STOPPED_SHORT else None
        no_attempt = self._open is None  # asked before the actions, which may close the one open
        self._state = transition.state
        if before is State.AUTONOMOUS and given in STEP_INS:
            self._open.step_in()
        if transition.state is not before:
            self._confirm_until = None  # the arm moves on, or faults: a pending confirmation lapses
        if self._state is not before and self._state in (State.PAUSED, State.RESTING):
            self._toggle.stopped()  # TOGGLE waits TOGGLE_RESUME_AFTER_S from here to resume
        steps = [lambda action=action: self._carry_out(action, transition, outcome) for action in transition.actions]
        if before is State.REWINDING and transition.state is not State.REWINDING:
            steps.insert(0, lambda: self._end_rewind(given.value))
        failure = None
        for carry_out in steps:
            try:
                carry_out()
            except Exception as error:
                failure = failure or error
        ended_by = self._open.ended_by if self._open is not None else None
        if self._state is not before:
            ending = {"ended_by": ended_by} if self._state is State.RETURNING and ended_by else {}
            self.recorder.event("state", state=self._state.value, previous=before.value, cause=given.value,
                                **ending, **({"note": shortfall} if shortfall else {}))
        if shortfall:
            self._say(shortfall)
        if self._state is not before:
            self._say(wording.announcement(self, ended_by, no_attempt), self._state.value)
        if self._prompt is not None and Action.CLOSE_ATTEMPT in transition.actions:
            self._say(wording.ASK_OUTCOME.format(attempt=self._prompt))
        if Action.CONFIRM_QUIT in transition.actions:
            self._say(wording.confirm_message(self, announced=self._state is not before))
        if failure is not None:
            raise failure
        return transition

    def _carry_out(self, action: Action, transition: Transition, outcome: str | None = None) -> None:
        match action:
            case Action.OPEN_ATTEMPT:
                self._attempt += 1
                self._open = Attempt(self._attempt, self.clock())
                self._prompt = None  # a new attempt leaves the last one unlabelled
                self.recorder.start_attempt(self._attempt)
            case Action.MOVE_TO_REST:
                self._move = rollout.Move(self.rest_pose, self.fps, self.clock)
                self._faulted = False
                if outcome is not None:  # `states._reset` allows one only in Fault, which saved this attempt
                    self._label(outcome)
            case Action.MOVE_TO_START:
                self._move = rollout.Move(self.start_pose or dict(self._follower.present), self.fps, self.clock)
                self._open.teleop_next = transition.then_teleop
            case Action.RUN_POLICY:
                self.policy.run()
                self._open.underway = True
            case Action.PAUSE_POLICY:
                self.policy.pause()
            case Action.START_REWIND:
                self._rewind = Rewind(self._open.history, self.control.rewind_s, self.clock)
                self.leader.start_mirror(self._follower.arm.goal or self._follower.present, self.fps)
            case Action.EXTEND_REWIND:
                self._rewind.extend(self.control.rewind_s)
                self._say(f"Rewinding another {self.control.rewind_s:g} s.")
            case Action.STOP_MOVE:
                self._move = None
                self.leader.handover = None  # the leader holds where it was last sent, motors on
            case Action.START_HANDOVER:
                self._move = None
                self.leader.handover = Handover(self.leader, self.fps, self.clock)
            case Action.FREE_LEADER:
                self._log_handover()
                self.leader.handover = None
                self.leader.ok(self.leader.free)
                self._open.underway = True
            case Action.HOLD_LEADER:
                self.leader.ok(self.leader.hold)
            case Action.END_ATTEMPT:
                self.end_attempt(transition.ended_by, outcome)
            case Action.CANCEL_END:
                if self._open is not None:
                    self._open.ended_by = self._open.outcome = None  # the attempt stays open, with no outcome yet
                self._quitting = False
            case Action.QUIT_AT_REST:
                self._quitting = True
            case Action.CONFIRM_QUIT:
                self._quitting = False
                self._confirm_until = self.clock() + CONFIRM_S  # asked again: CONFIRM_S from now
            case Action.CLOSE_ATTEMPT:
                self._close_attempt()
            case Action.EXIT | Action.EXIT_LEAVING_MOTORS:
                # No attempt is open here: it was saved at rest, where the arm stopped short, or at a fault.
                if action is Action.EXIT_LEAVING_MOTORS:
                    self.leaves_motors = True
                    self._say(wording.LEAVING_MOTORS)
                self.done = True
                try:
                    self.policy.pause()
                except Exception:
                    pass  # a failed policy may fail to pause too; the process exits either way

    def end_attempt(self, ended_by: str, outcome: str | None = None) -> None:
        """End the open attempt. Every ending comes through here: the state machine's END_ATTEMPT.

        Stops the policy, records why the attempt ended, and drives the arm to rest (Returning). The
        leader's torque stays as it is: a free leader (Teleop) stays free, and one held with its motors
        on, which the operator was told they could let go of, stays held until the next handover
        arrives or serve exits.
        The attempt closes when the arm arrives, and is saved then with its outcome, if it has one.
        A time limit's ending has none, so it asks for one then, as the status's `prompt`.
        A Pause before then cancels the ending and the outcome with it.

        A fault is the exception: the arm holds where it is and the attempt is saved there at once,
        keeping an outcome a Reset already gave it.
        """
        attempt = self._open
        self.leader.handover = None
        if ended_by == "fault":
            try:
                self.policy.pause()
            except Exception:
                pass  # a failed policy may fail to pause too; it no longer drives either way
            attempt.ended_by, attempt.outcome = ended_by, outcome or attempt.outcome
            self._close_attempt()
            return
        attempt.ended_by, attempt.outcome = ended_by, outcome
        self._move = rollout.Move(self.rest_pose, self.fps, self.clock)
        self.policy.pause()

    def _close_attempt(self) -> None:
        """Save the open attempt, if one is, and drop it. It is dropped even when the save fails:
        the attempt is closed either way. Saving holds up the loop, for about 0.1 s and more on a
        slow disk."""
        attempt = self._open
        try:
            if attempt is not None:  # not after a fault's Reset, whose attempt was saved at the fault
                self._save_attempt(attempt)
        finally:
            if attempt is not None:
                self._open, self._closed = None, attempt
                attempt.history = None  # nothing rewinds a closed attempt
            self._held_up()

    def _save_attempt(self, attempt: Attempt) -> None:
        """A save that fails sets the recorder's `failure`, which refuses new attempts: say why. An
        attempt with no episode, because its save failed or it had no frames, is lost: the status shows
        it as not saved, the run's summary leaves it out, and no outcome is asked or taken for it."""
        failure = self.recorder.failure
        limit = {"limit": attempt.limit_hit, "limit_s": getattr(self.control, attempt.limit_hit)} \
            if attempt.ended_by == "time_limit" else {}
        attempt.episode = episode = self.recorder.save_attempt(attempt.ended_by, attempt.outcome,
                                                               operator=self.operator, **limit)
        if episode is not None and (dropped := self.recorder.dropped) and not self._told_dropped:
            self._told_dropped = True
            counts = ", ".join(f"{camera} {n}" for camera, n in dropped.items())
            self._say(f"The video encoder fell behind and dropped frames in attempt {attempt.number} ({counts}), so "
                      "its video doesn't line up with the arm's data. It's saved, but builds leave it out, and any "
                      "later attempt that drops frames too. Lighten the Mac's load: close other apps, or lower "
                      "preview_fps under [control] in so101.toml. This is said once per run.")
        if (now_failing := self.recorder.failure) and now_failing != failure:
            self._say(now_failing)  # with an episode, it was saved, and only reopening the dataset failed
        if episode is None:
            return
        self._attempts.append(attempt)
        if attempt.ended_by == "fault":
            self._faulted = True
        if attempt.ended_by == "time_limit" and attempt.outcome is None:
            self._prompt = attempt.number

    def _end_rewind(self, cause: str) -> None:
        """Log what a rewind undid, and leave the history's newest moment as where it ended."""
        running, self._rewind = self._rewind, None
        self.leader.mirror = None
        if running is not None:
            self.recorder.event("rewind", **rewind.finish(running, self._follower.present, cause))

    def _log_handover(self) -> None:
        handover = self.leader.handover
        joint, gap = handover.gap(self._follower.present)
        self.recorder.event("handover", leader=handover.pose, follower=self._follower.present,
                            farthest=joint.removesuffix(".pos"), gap=round(gap, 2))

    # --- What the operator is told (the wording is in `wording`) ---------------------------------

    def _say(self, message: str, state: str = "") -> None:
        """Tell the operator. `state` is set when the message announces a new state."""
        self.notify(message, state)

    # --- Status (built in `status`) ----------------------------------------------------------------

    def _build_status(self) -> dict:
        return status.build(self, allowed(self._state, self._facts()))

    def _publish(self) -> None:
        self._status = (self._build_status(), self.link.client)
