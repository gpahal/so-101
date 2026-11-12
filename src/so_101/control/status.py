"""The controller's status, shaped like the control page's /status, as a function of its state.

The controller builds it at the end of each tick, and the web server's threads read the copy it
kept (`Controller.status`). It is the controller's own, kept in a file beside `controller.py`: it reads the
controller's state and changes nothing.
"""

from so_101.control.attempt import summary
from so_101.control.link import NOT_IN_CONTROL
from so_101.control.states import State


def build(controller, refusals: dict[str, str | None]) -> dict:
    """The status as of now. `refusals` is the state machine's answer for each command: None when
    it's allowed, else why not (`states.allowed`)."""
    c = controller
    attempt = c._open
    closed = c._closed
    shown = attempt or closed  # the page shows how the attempt closed last went, until the next opens
    devices = {**c._devices(), "leader": c._leader_reason()}  # the leader is optional
    return {
        "state": c._state.value,
        "attempt": c._attempt,
        "attempt_open": attempt is not None,
        "elapsed_s": round(c.clock() - attempt.started, 1) if attempt is not None else None,
        "policy_s": round(attempt.spent["attempt_limit_s"], 1) if attempt is not None else None,
        "limit_s": c.control.attempt_limit_s,
        "teleop_s": round(attempt.spent["human_limit_s"], 1) if attempt is not None else None,
        "rewind_available_s": round(c._rewind_available(), 1) if attempt is not None else None,
        "rewind_s": c.control.rewind_s,
        "human_limit_s": c.control.human_limit_s,
        "policy": c.policy_name,
        "allowed": {command: reason is None for command, reason in refusals.items()},
        "refusals": {command: reason for command, reason in refusals.items() if reason is not None},
        "ending": attempt.ended_by if attempt is not None else None,
        "confirm_quit": c._confirm_pending,  # y at the terminal now exits where the arm is
        "last_attempt": {"attempt": closed.number, "ended_by": closed.ended_by, "outcome": closed.outcome,
                         "saved": closed.episode is not None} if closed else None,
        "devices": {device: "missing" if reason else "ok" for device, reason in devices.items()},
        "device_problems": {device: reason for device, reason in devices.items() if reason},
        "fault": c._fault_error if c._state is State.FAULT else None,
        "warnings": [warning for warning in (c._follower.hot, c._follower.gripper_tripped, c._pace.behind) if warning],
        "previews": {"fps": c.previews.fps if c.previews else 0, "note": c._pace.preview_note},
        "prompt": c._prompt,  # the attempt a time limit ended, waiting for its outcome, or None
        "operator": c.operator,
        "controller": c.link.who if c.link.client is not None else None,
        "you_have_control": None,  # per page: `seen_by` answers it
        "link_lost": c.link.lost,
        "last_event": c.recorder.tail.last_id,
        "timeline": [{**run, "s": round(run["s"], 1)} for run in (shown.timeline if shown else [])],
        "label": {"current": shown.label if shown else None, "refusal": c._label_reason()},
        "summary": summary(c._attempts),
    }


def seen_by(status: dict, controller: str | None, client: str | None) -> dict:
    """`status` as the page `client` sees it, when the page `controller` (or none) has control: a
    page that doesn't control the robot has every command refused, with the reason. Without
    `client`, `you_have_control` is None."""
    if client is None:
        return status
    if controller is None or controller == client:
        return {**status, "you_have_control": True}
    reason = NOT_IN_CONTROL.format(who=status["controller"])
    return {
        **status,
        "you_have_control": False,
        "allowed": dict.fromkeys(status["allowed"], False),
        "refusals": dict.fromkeys(status["allowed"], reason),
        "label": {**status["label"], "refusal": reason},
    }
