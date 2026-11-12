"""What the controller tells the operator as the arm moves between states, as functions of its state.

Each new state is announced (`announcement`); a move that stops short says where and why
(`shortfall`); and a Quit where the arm can't get to rest says why and how to exit there
(`confirm_message`). These are the controller's own, kept in a file beside `controller.py`: they read the
controller's state and change nothing.
"""

from so_101 import rollout
from so_101.control.rewind import REWIND_SETTLE_S
from so_101.control.states import CONFIRM_S, State

# Where the arm can't get to rest, each Quit asks for y at the terminal. With no terminal, nothing can
# confirm, so the arm holds until its power is switched off.
SUPPORT_AND_QUIT = (f"To exit here, support the arm, then press y at this terminal within {CONFIRM_S:g} s: the "
                    "motors turn off, so lower it to rest by hand. With no terminal, the arm keeps holding: support "
                    "it, then switch the arm's power off.")
# A quit in Fault while the follower doesn't answer exits at once (`states._quit`).
LEAVING_MOTORS = ("The follower isn't answering, so serve exits now and leaves its motors as they are: without "
                  "power they stay off, and if only its USB cable came out, the arm keeps holding until you switch "
                  "its power off.")
# The announcement when a quit's drive to rest stops short, before SUPPORT_AND_QUIT.
SHORT_OF_REST = "The arm holds where it stopped, short of rest."
# The time-limit prompt's question; the terminal answers it with s, f or i at rest.
ASK_OUTCOME = "How did attempt {attempt} go? s succeeded, f failed, i ignore it. No answer leaves it unlabelled."

SAYS = {
    State.RESTING: "Attempt {attempt} is over: the arm is at rest.",
    State.STARTING: "Attempt {attempt}: moving to the demos' start pose.",
    State.AUTONOMOUS: "Attempt {attempt}: the policy is driving.",
    State.PAUSED: "Attempt {attempt} paused: the arm holds with the motors on.",
    State.HANDOVER: "Attempt {attempt}: moving the leader to the follower's pose. Let go of the leader until it stops.",
    State.TELEOP: "Attempt {attempt}: Teleop. The leader's motors are off: move it to drive the follower.",
    State.REWINDING: "Attempt {attempt}: rewinding the arm's last moves, newest first. It pauses when done.",
    State.RETURNING: "Attempt {attempt} ended ({ended_by}): returning to rest.",
    State.FAULT: "Fault: the arm holds where it is. {fault} Only Reset and Run checks are allowed.",
}
# Reset's move to rest with no attempt open (after a fault, or from a stop away from rest) is announced
# differently. RESTING is said where the move ends, at rest or, paused or stopped short, away from it.
SAYS_NO_ATTEMPT = {
    State.RETURNING: "Returning to rest, with no attempt open.",
    State.RESTING: "The arm is at rest, with no attempt open.",
}
AWAY_FROM_REST = "The arm holds away from rest, with no attempt open: Reset drives it to rest."


def announcement(controller, ended_by: str | None, no_attempt: bool) -> str:
    """What the state the controller just entered is called, for the operator. `ended_by` is why
    the open attempt is ending, if it is. `no_attempt` is whether no attempt was open as it entered it."""
    c = controller
    if c._state is State.RESTING and c._confirm_pending:
        return SHORT_OF_REST
    if no_attempt and c._state in SAYS_NO_ATTEMPT:
        return SAYS_NO_ATTEMPT[c._state] if c._state is State.RETURNING or c.at_rest else AWAY_FROM_REST
    return SAYS[c._state].format(attempt=c._attempt, ended_by=ended_by, fault=c._fault_error)


def confirm_message(controller, announced: bool) -> str:
    """Why the arm can't go to rest before exiting, and what to do. `announced` is whether a new
    state was just announced, which said where the arm is."""
    c = controller
    if c._state is State.FAULT:
        return (f"The arm can't drive to rest after a fault: the follower may not answer. Reset drives it to rest "
                f"once it does. {SUPPORT_AND_QUIT}")
    if announced:
        return f"Reset drives it to rest. {SUPPORT_AND_QUIT}"
    joint, off = rollout.farthest(c._follower.present, c.rest_pose)
    return (f"The arm isn't at rest: {joint.removesuffix('.pos')} is {off:.0f} away. Reset drives it to rest. "
            f"{SUPPORT_AND_QUIT}")


def shortfall(controller) -> str:
    """Where the move the controller is running stopped short, and what holds now."""
    c = controller
    present = c._follower.present
    if c._state is State.HANDOVER:
        key, off = c.leader.handover.gap(present)
        return (f"The leader couldn't line up with the follower: {key.removesuffix('.pos')} is {off:.0f} away. "
                "Paused, with both arms held. Let go of the leader, then press Teleop again.")
    if c._state is State.REWINDING:
        key, off = rollout.farthest(present, c._rewind.target)
        return (f"The rewind stopped short: {key.removesuffix('.pos')} is {off:.0f} away from the pose it went "
                f"back to, after {REWIND_SETTLE_S:g} s of trying. Paused there.")
    key, off = rollout.farthest(present, c._move.target)
    joint = key.removesuffix(".pos")
    if c._state is State.STARTING:
        return (f"The arm can't reach the demos' start pose: {joint} is {off:.0f} away. Paused rather than "
                "starting the policy from the wrong pose.")
    if c._open is None:  # a move to rest with no attempt open: after a fault, or from a stop away from rest
        return f"The arm stopped short of rest: {joint} is {off:.0f} away. It holds there, with no attempt open."
    if c._quitting:
        return f"The arm stopped short of rest: {joint} is {off:.0f} away. Attempt {c._attempt} is saved there."
    return f"The arm stopped short of rest: {joint} is {off:.0f} away. The attempt stays open."
