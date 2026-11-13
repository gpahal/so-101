"""The control center's state machine: `step(state, command or event, facts) -> Transition`.

It is pure: it decides the next state and the actions the controller must carry out, and refuses a
command that isn't allowed with a reason the operator can act on. It touches nothing.

Starting, Handover, Rewinding and Returning are moves the controller runs itself, one goal per tick,
so Pause can cut them short.

Fault is where a hardware, camera, motor-alarm or policy failure leaves the arm: it holds where it
is, the attempt is saved there, and only Reset and Run checks are allowed.

Quitting never drops the arm mid-air. Quit ends the attempt and drives the arm to rest, then exits
with the motors off; further Quits on the way are refused. At rest, Quit exits at once. Either way it
waits for the leader's own move to rest, if one is running (LEADER_RESTED), since exiting turns its
motors off and a leader dropped part of the way falls. Where the
arm can't get to rest (in Fault, short of rest, or when that drive stops short) Quit never exits: it
asks for confirmation, and only EXIT_HERE (y at the robot's terminal), sent while that confirmation
is pending, exits where the arm is. The controller says whether one is pending: it lapses
CONFIRM_S after the last ask and on any state change, and a Quit while one is pending asks again.

    Quit,      can't reach rest, none pending  -> ask (pending for CONFIRM_S)
    Quit,      pending                          -> ask again (CONFIRM_S from now)
    exit_here, pending, source terminal         -> EXIT
    exit_here, otherwise                        -> refused, with what to press

No signal can confirm: Ctrl+C, SIGTERM and SIGHUP submit Quit with source `signal`.

Run checks and Check pose are read-only tools, not commands: they change no state. `tool_refusal`
says when they're allowed.
"""

from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    RESTING = "resting"
    STARTING = "starting"  # moving to the demos' start pose before the policy starts
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    HANDOVER = "handover"  # driving the leader to the follower's measured pose before Teleop
    TELEOP = "teleop"  # a human drives the follower with the leader
    REWINDING = "rewinding"  # playing the attempt's recent motion backwards, then pausing
    RETURNING = "returning"  # moving to the rest pose at the end of an attempt, or after a fault
    FAULT = "fault"  # a failure: the arm holds where it is until Reset


class Command(StrEnum):
    """What an operator asks for."""

    RESUME = "resume"
    PAUSE = "pause"
    REWIND = "rewind"
    TELEOP = "teleop"
    RESET = "reset"
    QUIT = "quit"
    EXIT_HERE = "exit_here"  # y at the terminal: confirm exiting where the arm can't get to rest


# The commands that interrupt a running policy: each is a step-in on the attempt's timeline, and an
# interruption in the run's analysis. Reset and Quit end the attempt instead.
STEP_INS = frozenset({Command.PAUSE, Command.REWIND, Command.TELEOP})


class Event(StrEnum):
    """What happens without an operator's command: a move ends, a time limit runs out, the leader or
    the controlling page is lost, or a fault."""

    ARRIVED = "arrived"
    STOPPED_SHORT = "stopped_short"
    TIME_LIMIT = "time_limit"  # the attempt has used its policy or its Teleop seconds
    LEADER_LOST = "leader_lost"  # the leader was unplugged or stopped answering
    LINK_LOST = "link_lost"  # the controlling page stopped polling or was hidden, with on_link_loss = "pause"
    LEADER_RESTED = "leader_rested"  # the leader's move to rest ended: it arrived, stopped short or was lost
    FAULT = "fault"  # a hardware, camera, motor-alarm or policy failure


class Action(StrEnum):
    OPEN_ATTEMPT = "open_attempt"  # number a new attempt and start its clock
    MOVE_TO_START = "move_to_start"
    RUN_POLICY = "run_policy"  # reset the engine, so no queued actions remain, and resume it
    PAUSE_POLICY = "pause_policy"
    STOP_MOVE = "stop_move"  # cut a scripted move or a handover short; the arm holds where it was commanded
    START_HANDOVER = "start_handover"  # turn the leader's motors on and drive it to the follower's measured pose
    FREE_LEADER = "free_leader"  # the leader arrived: its motors turn off, so a human can move it
    HOLD_LEADER = "hold_leader"  # leaving Teleop: the leader's motors turn on where it is
    START_REWIND = "start_rewind"  # play back the last rewind_s of the follower's measured motion, newest first
    EXTEND_REWIND = "extend_rewind"  # a running rewind goes back another rewind_s
    END_ATTEMPT = "end_attempt"  # the one way an attempt ends; see Transition.ended_by
    MOVE_TO_REST = "move_to_rest"  # Reset with no attempt open, after a fault or away from rest: the arm goes to rest
    CANCEL_END = "cancel_end"  # a pause during Returning: the attempt stays open, and so does the process
    QUIT_AT_REST = "quit_at_rest"  # exit once the arm, and the leader if it's moving there, arrive at rest
    # The arm can't get to rest: hold it, and ask for EXIT_HERE, pending for CONFIRM_S from now.
    CONFIRM_QUIT = "confirm_quit"
    CLOSE_ATTEMPT = "close_attempt"  # the arm arrived at rest, or a quit's drive stopped short: save the attempt
    EXIT = "exit"
    # The follower doesn't answer: exit now, leaving its motors as they are, never turning them off.
    EXIT_LEAVING_MOTORS = "exit_leaving_motors"


# Teleop's refusal while no leader is connected: unplugged, without a calibration file, or lost.
LEADER_MISSING = "Leader arm not connected."
# How long a request to confirm exiting away from rest stays pending.
CONFIRM_S = 30.0


@dataclass(frozen=True)
class Facts:
    """What the state alone doesn't say."""

    underway: bool  # the attempt got past Starting (the policy or a human drove), so Resume continues it
    quitting: bool  # a Quit is under way (Esc or a signal), so the process exits at rest
    # A Quit asked to confirm exiting where the arm is, under CONFIRM_S ago, and the state hasn't changed since.
    confirm_pending: bool
    source: str | None  # where the command came from: terminal, page or signal
    at_rest: bool  # every joint is within rollout.TOLERANCE of the rest pose
    leader_to_rest: bool  # the leader is on its way to rest, motors on: exiting now would drop it
    attempt_open: bool
    # Why a required device is missing, naming it and the fix: the follower, and the first of any
    # (follower first), or None when it's there.
    follower_missing: str | None
    device_missing: str | None
    # The follower doesn't answer: its last read failed (`Follower.silent`). Exiting can't turn its
    # motors off, so it changes nothing about the arm. A motor restart its marks found answered, so
    # it isn't silent, and a quit then asks for y.
    follower_silent: bool
    fault: str | None  # the error that caused the Fault state
    leader: str | None  # why the leader can't take over, or None when it is connected
    teleop_from_start_pose: bool  # Teleop from rest first drives to the demos' start pose
    teleop_next: bool  # Teleop started this move to the start pose, so the leader takes over there
    # Why Rewind has nothing to play back, or None when the attempt's history has motion left to undo.
    rewind: str | None
    policy: str | None  # why the policy can't run now (it failed; a remote one may reconnect), or None
    recording: str | None  # why a new attempt can't be recorded (little free disk, a failed dataset)
    outcome: bool  # the command came with an outcome to record (Reset's)
    fault_saved: bool  # the last fault saved an attempt, which an outcome given with Reset in Fault labels
    fault_labelled: str | None  # that attempt's outcome, when it already has one: Reset in Fault won't relabel it


@dataclass(frozen=True)
class Transition:
    state: State
    actions: tuple[Action, ...] = ()
    refused: str | None = None
    ended_by: str | None = None  # with END_ATTEMPT: reset, time_limit, quit or fault
    then_teleop: bool = False  # with MOVE_TO_START: hand over to the leader at the start pose, not the policy

    @property
    def accepted(self) -> bool:
        return self.refused is None


def _refuse(state: State, reason: str) -> Transition:
    return Transition(state, refused=reason)


def _fault_refusal(facts: Facts) -> str:
    return f"After a fault only Reset and Run checks are allowed: {facts.fault or 'see the event log'}"


# Commands that start the arm moving on its own, under the policy, a human or a rewind. A missing
# required device refuses them.
NEEDS_DEVICES = frozenset({"resume", "rewind", "teleop"})


def _resume(state: State, facts: Facts) -> Transition:
    if state is State.FAULT:
        return _refuse(state, _fault_refusal(facts))
    if facts.policy is not None and state in (State.RESTING, State.PAUSED, State.TELEOP):
        return _refuse(state, facts.policy)
    if state is State.RESTING:
        return Transition(State.STARTING, (Action.OPEN_ATTEMPT, Action.MOVE_TO_START))
    if state is State.PAUSED:
        if facts.underway:
            return Transition(State.AUTONOMOUS, (Action.RUN_POLICY,))
        # The arm never reached the start pose in this attempt, so nobody has driven yet.
        return Transition(State.STARTING, (Action.MOVE_TO_START,))
    if state is State.TELEOP:
        # The leader holds where the human left it, and the policy carries on from there.
        return Transition(State.AUTONOMOUS, (Action.HOLD_LEADER, Action.RUN_POLICY))
    if state is State.AUTONOMOUS:
        return _refuse(state, "The policy is already running.")
    if state is State.STARTING:
        return _refuse(state, "Already moving to the start pose.")
    if state is State.HANDOVER:
        return _refuse(state, "Handing over to the leader. Pause first.")
    if state is State.REWINDING:
        return _refuse(state, "Rewinding: the arm pauses when it's done. Pause first to stop it sooner.")
    return _refuse(state, "Returning to rest. Pause first to keep the attempt open.")


def _pause(state: State, facts: Facts) -> Transition:
    if state is State.FAULT:
        return _refuse(state, f"The arm already holds. {_fault_refusal(facts)}")
    if state is State.AUTONOMOUS:
        return Transition(State.PAUSED, (Action.PAUSE_POLICY,))
    if state in (State.STARTING, State.HANDOVER, State.REWINDING):
        return Transition(State.PAUSED, (Action.STOP_MOVE,))
    if state is State.TELEOP:
        return Transition(State.PAUSED, (Action.HOLD_LEADER,))
    if state is State.RETURNING:
        if not facts.attempt_open:  # a move to rest with no attempt open: there is none to keep open
            return Transition(State.RESTING, (Action.STOP_MOVE, Action.CANCEL_END))
        return Transition(State.PAUSED, (Action.STOP_MOVE, Action.CANCEL_END))
    if state is State.PAUSED:
        return _refuse(state, "Already paused.")
    return _refuse(state, "Nothing to pause: the arm is resting.")


def _teleop(state: State, facts: Facts) -> Transition:
    if state is State.FAULT:
        return _refuse(state, _fault_refusal(facts))
    if state is State.TELEOP:
        return _refuse(state, "Already in Teleop.")
    if state is State.HANDOVER:
        return _refuse(state, "Already handing over to the leader.")
    if state is State.STARTING:
        return _refuse(state, "Moving to the start pose. Pause first, then Teleop.")
    if state is State.REWINDING:
        return _refuse(state, "Rewinding: the arm pauses when it's done. Then press Teleop.")
    if state is State.RETURNING:
        return _refuse(state, "Returning to rest. Pause first to keep the attempt open.")
    if facts.leader is not None:
        return _refuse(state, facts.leader)
    if state is State.RESTING:
        if facts.teleop_from_start_pose:
            return Transition(State.STARTING, (Action.OPEN_ATTEMPT, Action.MOVE_TO_START), then_teleop=True)
        return Transition(State.HANDOVER, (Action.OPEN_ATTEMPT, Action.START_HANDOVER))
    if state is State.AUTONOMOUS:
        return Transition(State.HANDOVER, (Action.PAUSE_POLICY, Action.START_HANDOVER))
    return Transition(State.HANDOVER, (Action.START_HANDOVER,))  # Paused


def _rewind(state: State, facts: Facts) -> Transition:
    if state is State.FAULT:
        return _refuse(state, _fault_refusal(facts))
    if state is State.STARTING:
        return _refuse(state, "Moving to the start pose. Pause first, then Rewind.")
    if state is State.HANDOVER:
        return _refuse(state, "Handing over to the leader. Pause first, then Rewind.")
    if state is State.RETURNING:
        return _refuse(state, "Returning to rest. Pause first to keep the attempt open, then Rewind.")
    if facts.rewind is not None:
        return _refuse(state, facts.rewind)
    if state is State.REWINDING:
        return Transition(state, (Action.EXTEND_REWIND,))
    if state is State.AUTONOMOUS:
        return Transition(State.REWINDING, (Action.PAUSE_POLICY, Action.START_REWIND))
    # Paused, or Teleop: the rewind turns the leader's motors on, and it follows the follower.
    return Transition(State.REWINDING, (Action.START_REWIND,))


def _reset(state: State, facts: Facts) -> Transition:
    if state in (State.AUTONOMOUS, State.PAUSED, State.STARTING, State.HANDOVER, State.TELEOP, State.REWINDING):
        return Transition(State.RETURNING, (Action.END_ATTEMPT,), ended_by="reset")
    if state is State.RETURNING:
        return _refuse(state, "Already returning to rest.")
    if state is State.FAULT or (state is State.RESTING and not facts.at_rest):
        # After a fault, the attempt was saved when it happened. Resting away from rest (a move to
        # rest paused or stopped short), none is open. Either way the arm goes to rest with no
        # attempt open, and only if the follower answers. An outcome labels the attempt the fault
        # saved; with nothing to label, the Reset is refused rather than the outcome dropped.
        if facts.outcome and state is State.FAULT and not facts.fault_saved:
            return _refuse(state, "The fault saved no attempt, so there is nothing to label: Reset without an "
                                  "outcome drives the arm to rest.")
        if facts.outcome and state is State.FAULT and facts.fault_labelled:
            return _refuse(state, f"The attempt the fault saved is already labelled {facts.fault_labelled}: Reset "
                                  "without an outcome drives the arm to rest. To change the label, use "
                                  "`uv run so101 review --runs`.")
        if facts.outcome and state is State.RESTING:
            return _refuse(state, "No attempt is open to label: Reset without an outcome drives the arm to rest.")
        if facts.follower_missing:
            return _refuse(state, facts.follower_missing)
        return Transition(State.RETURNING, (Action.MOVE_TO_REST,))
    return _refuse(state, "No attempt is open: the arm is already at rest.")


def _quit(state: State, facts: Facts) -> Transition:
    # Where the arm can't get to rest, a Quit only asks, and asks again while one is pending: a held
    # Esc, a doubled signal or a closed terminal must never turn the motors off with the arm raised.
    # The one exception is a silent follower in Fault, whose motors exiting leaves as they are.
    if facts.quitting:
        return _refuse(state, "Already quitting. The arm is on its way to rest, and serve exits once it's there; "
                              "Pause cancels the quit.")
    if state is State.RESTING:
        if not facts.at_rest:
            return Transition(state, (Action.CONFIRM_QUIT,))
        return Transition(state, (Action.QUIT_AT_REST,) if facts.leader_to_rest else (Action.EXIT,))
    if state is State.FAULT:
        # The follower may not answer, so the arm can't be driven to rest. One that doesn't, as when its
        # power is off, can't have its motors turned off either: exiting leaves the arm as it is, so no
        # confirmation is needed, and a signal exits too.
        if facts.follower_silent:
            return Transition(state, (Action.EXIT_LEAVING_MOTORS,))
        return Transition(state, (Action.CONFIRM_QUIT,))
    if state is State.RETURNING:
        return Transition(state, (Action.QUIT_AT_REST,))
    return Transition(State.RETURNING, (Action.QUIT_AT_REST, Action.END_ATTEMPT), ended_by="quit")


# What EXIT_HERE's refusal says to press.
EXIT_HERE_HOW = (f"Press Esc first. Where the arm can't get to rest, y at the robot's terminal within "
                 f"{CONFIRM_S:g} s of the warning exits where it is.")


def _exit_here(state: State, facts: Facts) -> Transition:
    if facts.source != "terminal":
        return _refuse(state, "Only y at the robot's terminal exits where the arm is: whoever turns the motors off "
                              "should be beside the arm.")
    if not facts.confirm_pending:
        return _refuse(state, f"No exit is waiting for y. {EXIT_HERE_HOW}")
    # Someone at the terminal confirmed, within CONFIRM_S of the warning: exit, and the motors turn off.
    return Transition(state, (Action.EXIT,))


def _arrived(state: State, facts: Facts) -> Transition:
    if state is State.STARTING:
        if facts.teleop_next:
            return Transition(State.HANDOVER, (Action.START_HANDOVER,))
        return Transition(State.AUTONOMOUS, (Action.RUN_POLICY,))
    if state is State.HANDOVER:
        return Transition(State.TELEOP, (Action.FREE_LEADER,))
    if state is State.REWINDING:
        return Transition(State.PAUSED, (Action.STOP_MOVE,))
    if state is State.RETURNING:
        exits = facts.quitting and not facts.leader_to_rest  # otherwise LEADER_RESTED exits
        return Transition(State.RESTING, (Action.CLOSE_ATTEMPT, *((Action.EXIT,) if exits else ())))
    return _refuse(state, f"No move is running in {state}.")


def _stopped_short(state: State, facts: Facts) -> Transition:
    if state is State.STARTING:
        # Matches `so101 run`: never start the policy from the wrong pose.
        return Transition(State.PAUSED)
    if state is State.HANDOVER:
        # The leader couldn't line up with the follower: it holds, motors on, rather than be freed
        # away from the follower's pose, which would make the follower jump.
        return Transition(State.PAUSED, (Action.STOP_MOVE,))
    if state is State.REWINDING:
        # The arm holds where the rewind left it; the controller reports how far off it is.
        return Transition(State.PAUSED, (Action.STOP_MOVE,))
    if state is State.RETURNING:
        if facts.quitting:
            # The arm holds where it stopped and the attempt is saved there; only y at the
            # terminal, pressed once someone supports the arm, exits.
            return Transition(State.RESTING, (Action.STOP_MOVE, Action.CLOSE_ATTEMPT, Action.CONFIRM_QUIT))
        if not facts.attempt_open:
            return Transition(State.RESTING, (Action.STOP_MOVE,))
        return Transition(State.PAUSED, (Action.CANCEL_END,))
    return _refuse(state, f"No move is running in {state}.")


def _time_limit(state: State, facts: Facts) -> Transition:
    if state in (State.AUTONOMOUS, State.TELEOP):
        return Transition(State.RETURNING, (Action.END_ATTEMPT,), ended_by="time_limit")
    return _refuse(state, f"No time limit applies in {state}.")


def _leader_lost(state: State, facts: Facts) -> Transition:
    if state in (State.HANDOVER, State.TELEOP):
        # The follower must not chase a stale pose: it holds where it is.
        return Transition(State.PAUSED, (Action.STOP_MOVE,))
    return _refuse(state, f"The leader isn't driving in {state}.")


def _link_lost(state: State, facts: Facts) -> Transition:
    if state is State.AUTONOMOUS:
        # Nobody is watching the policy: it stops, and the arm holds where it is.
        return Transition(State.PAUSED, (Action.PAUSE_POLICY,))
    if state is State.STARTING and not facts.teleop_next:
        # The policy would start unwatched on arrival: the move stops, and the arm holds where it was
        # commanded. Resume moves on to the start pose.
        return Transition(State.PAUSED, (Action.STOP_MOVE,))
    return _refuse(state, f"The policy isn't driving in {state}.")


def _leader_rested(state: State, facts: Facts) -> Transition:
    if state is State.RESTING and facts.quitting and facts.at_rest:
        return Transition(state, (Action.EXIT,))
    return _refuse(state, "No quit is waiting for the leader.")


def _fault(state: State, facts: Facts) -> Transition:
    if state is State.FAULT:
        return _refuse(state, "Already in Fault.")
    # The arm holds where it is, and an open attempt is saved there, not driven to rest.
    actions = (Action.STOP_MOVE,)
    if facts.attempt_open:
        actions += (Action.END_ATTEMPT,)
    if facts.quitting:  # the arm can't be driven to rest now, so exiting waits for y at the terminal
        actions += (Action.CONFIRM_QUIT,)
    return Transition(State.FAULT, actions, ended_by="fault" if facts.attempt_open else None)


_RULES = {
    Command.RESUME: _resume,
    Command.PAUSE: _pause,
    Command.REWIND: _rewind,
    Command.TELEOP: _teleop,
    Command.RESET: _reset,
    Command.QUIT: _quit,
    Command.EXIT_HERE: _exit_here,
    Event.ARRIVED: _arrived,
    Event.STOPPED_SHORT: _stopped_short,
    Event.TIME_LIMIT: _time_limit,
    Event.LEADER_LOST: _leader_lost,
    Event.LINK_LOST: _link_lost,
    Event.LEADER_RESTED: _leader_rested,
    Event.FAULT: _fault,
}


def step(state: State, given: Command | Event, facts: Facts) -> Transition:
    """The next state and the actions to carry out, or the same state with a refusal."""
    transition = _RULES[given](state, facts)
    if transition.accepted and given in NEEDS_DEVICES and facts.device_missing:
        return _refuse(state, facts.device_missing)
    if transition.accepted and Action.OPEN_ATTEMPT in transition.actions and facts.recording:
        return _refuse(state, facts.recording)
    return transition


class Tool(StrEnum):
    """Read-only tools: they read the open devices and change nothing."""

    CHECK = "check"  # Run checks: `so101 check` on the devices the server has open
    POSE = "pose"  # Check pose: the follower's pose against the demos' start pose


# Run checks samples the cameras for check.CAMERA_SAMPLE_S (2.5 s) inside one tick, so the loop
# sends no goal for about 3 s. The arm holds its last goal on its own meanwhile, which is safe only
# when nothing is driving it.
CHECKS_HOLD = ("Run checks holds the control loop for about 3 s, so it runs only while the arm holds still: "
               "Pause first.")


def tool_refusal(state: State, tool: Tool, facts: Facts) -> str | None:
    """Why `tool` is refused now, or None when it would run."""
    if tool is Tool.CHECK:
        return None if state in (State.RESTING, State.PAUSED, State.FAULT) else CHECKS_HOLD
    # Check pose reads the follower's last measured pose, which costs the loop nothing.
    return _fault_refusal(facts) if state is State.FAULT else None


def allowed(state: State, facts: Facts) -> dict[str, str | None]:
    """Each command's and tool's refusal reason now, or None when it would be accepted."""
    commands = (Command.RESUME, Command.PAUSE, Command.REWIND, Command.TELEOP, Command.RESET)
    refusals = {command.value: step(state, command, facts).refused for command in commands}
    return refusals | {tool.value: tool_refusal(state, tool, facts) for tool in Tool}
