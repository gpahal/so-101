"""What the controller's commands take, what they answer, and how they're taken in.

Other threads `submit` commands by name; the control loop takes each in with `intake`, on its own
thread, and answers it with a `Result`. COMMANDS is the one table of every command and the options
it takes. Options are checked once, there: a command with any other option is refused, and the
option isn't logged. Then page control (a page's command needs control, `link`), then the command
is carried out.

The page's Space sends TOGGLE: it pauses whatever is moving, and otherwise
resumes. The loop decides which against the state it has, never a page's older copy.
A toggle resumes only once the arm has been stopped, and no toggle has come, for
TOGGLE_RESUME_AFTER_S, so a double press or a held key, meant to stop the arm, can't start it again.
The terminal's Space sends only Pause, because a terminal sees no key-up (terminal.py).

The intake is the controller's own, kept in a file beside `controller.py`: it reads the controller's state
and carries out commands through it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from so_101.control.link import NOT_IN_CONTROL, ONLY_PAGES
from so_101.control.recorder import OUTCOMES
from so_101.control.states import Command, Event, State, Tool, step
from so_101.report import Report


@dataclass(frozen=True)
class Result:
    """What became of a submitted command."""

    command: str
    accepted: bool
    reason: str | None = None
    report: Report | None = None  # what Run checks or Check pose found


TOGGLE = "toggle"
TOGGLE_RESUME_AFTER_S = 1.0
TAKE_CONTROL = "take_control"
# The control page reports when it goes out of sight (screen locked, another app in front). Its first
# poll in sight brings the link back (`link`).
PAGE_HIDDEN = "page_hidden"
# Why the operator stepped in, as the page's pause reason chips offer them. `label` records one.
LABELS = ("missed_grasp", "about_to_collide", "wrong_object", "dropped_it", "stuck", "other")
# A person started saying how the open attempt went: the terminal sends it once with the right arrow's
# question and the page once as its Reset sheet opens. Accepted for the open attempt only, it hands that
# attempt to the person in `so101 run` (`autopilot.HANDS_OVER`), as a Pause does, so the autopilot neither
# resumes nor ends it, whether the person then answers or cancels. Logged like any command.
ANSWERING = "answering"
# `so101 run`'s Hold and Continue on the page, which stop or let the next attempt start (`Controller.hold_autopilot`).
AUTOPILOT_HOLD = "autopilot_hold"
# The sources a person's commands come from; `so101 run`'s autopilot's carry source "autopilot"
# (`Controller.pilot_command`).
PEOPLE = ("terminal", "page", "signal")


class Toggle:
    """When the arm last stopped and the toggle was last pressed, which decide what a toggle means."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.stopped_at = -math.inf  # when the arm last paused or came to rest; at start, long ago
        self.last = -math.inf  # when the page's Space was last pressed

    def stopped(self) -> None:
        """The arm just paused or came to rest: a toggle waits TOGGLE_RESUME_AFTER_S from now to resume."""
        self.stopped_at = self.clock()

    def press(self, pause_accepted: bool, still: bool) -> tuple[str, Result | None]:
        """What a toggle means now: pause when Pause would be accepted, otherwise resume. `still` is
        whether the arm is paused or resting. Returns the command, and a refusal when a resume comes
        too soon after the arm stopped."""
        if pause_accepted:
            self.last = self.clock()
            return Command.PAUSE.value, None
        # A held key repeats about 30 times a second: counting from the last toggle as well as from
        # the stop keeps every repeat refused, so the arm starts again only after the key is let go.
        now, last, self.last = self.clock(), self.last, self.clock()
        quiet_for = now - max(self.stopped_at, last)
        if still and quiet_for < TOGGLE_RESUME_AFTER_S:
            return Command.RESUME.value, Result(
                Command.RESUME.value, False,
                f"Let go, wait {TOGGLE_RESUME_AFTER_S:g} s, then press again to resume, or use Resume on the page, "
                "or r at the terminal: a double press or a held key never starts the arm again.")
        return Command.RESUME.value, None


def stale_sheet(attempt, current: int) -> str | None:
    """Why a Reset from the page's sheet, opened for `attempt`, is refused when attempt `current` is
    the one now, or None. The sheet may end only the attempt it was opened for: the one open, or the
    one a fault just ended. A sheet left open on a page while another page or the terminal started a
    new attempt would otherwise label and end that one."""
    if type(attempt) is not int:
        return f"Reset's for_attempt must be an attempt number, not {attempt!r}. Reload the page."
    if attempt != current:
        return (f"That Reset was for attempt {attempt}, and attempt {current} is the one now: nothing was "
                "reset. Press Reset again to end this one.")
    return None


def stale_prompt(attempt, prompt: int) -> str | None:
    """Why an answer to the time-limit prompt, sent for `attempt`, is refused while attempt `prompt`
    is the one waiting, or None. `attempt` is None when the sender saw no prompt. A card left on a page,
    or a key pressed as a later attempt reached its limit, would otherwise label the later attempt."""
    if attempt is None:
        return f"Attempt {prompt} reached its time limit after that answer was sent: nothing was labelled. Answer again."
    if type(attempt) is not int:
        return f"An outcome's for_attempt must be an attempt number, not {attempt!r}. Reload the page."
    if attempt != prompt:
        return (f"That answer was for attempt {attempt}, and attempt {prompt} is the one waiting: nothing was "
                "labelled. Answer again.")
    return None


# --- The table of commands ------------------------------------------------------------------------


def _transition(controller, name: str, options: dict, source: str) -> Result:
    """A state machine command. Only Reset takes options: an outcome, and the attempt its sheet was for."""
    outcome = options.get("outcome")
    if outcome is not None and outcome not in OUTCOMES:
        return Result(name, False, f"Unknown outcome {outcome!r}: use {', '.join(OUTCOMES)}.")
    if "for_attempt" in options and (reason := stale_sheet(options["for_attempt"], controller.attempt_number)):
        return Result(name, False, reason)
    transition = controller.apply(Command(name), outcome=outcome, source=source)
    return Result(name, transition.accepted, transition.refused)


# Every command, with the options it takes. `outcome` and `label` come first, so OPTIONS lists them in
# that order.
COMMANDS: dict[str, tuple[str, ...]] = {
    "outcome": ("outcome", "for_attempt"),
    "label": ("label",),
    **{command.value: ("outcome", "for_attempt") if command is Command.RESET else () for command in Command},
    **{tool.value: () for tool in Tool},
    TAKE_CONTROL: (),
    TOGGLE: (),
    PAGE_HIDDEN: (),
    ANSWERING: ("for_attempt",),
    AUTOPILOT_HOLD: ("hold",),
}
# Every option some command takes. The web server refuses a request carrying any other key before it
# reaches the loop. for_attempt: the attempt the Reset sheet was opened for, or the time-limit prompt asked
# about, or the one a person started answering for; hold: Hold (true) or Continue.
OPTIONS = tuple(dict.fromkeys(option for options in COMMANDS.values() for option in options))
# The names of the state machine's commands, and of the read-only tools.
TRANSITIONS = frozenset(command.value for command in Command)
TOOLS = frozenset(tool.value for tool in Tool)


# --- Taking a command in --------------------------------------------------------------------------


def intake(controller, name: str, source: str, client: str | None, who: str | None, options: dict) -> Result:
    """Take in one command on the loop's thread: check its options, then page control, then carry
    it out, and log it as a `command` event (page_hidden is logged only as the link's `page_hidden` event,
    when it loses the link). Every command a person sends from the terminal, a signal or the page in control
    goes on the controller's `people` for this tick, which `so101 run`'s autopilot reads, accepted or refused:
    a refused Pause, Resume or Reset still says a person is at the arm and wants it. Each goes on with the
    attempt open when it arrived, before it ran, so a Pause right after the Resume that opened an attempt, in
    the same tick, hands that attempt over, and the Resume doesn't. A Quit goes on it only when accepted, and
    nothing from a page that isn't in control does. page_hidden returns before that, since the page sends it
    by itself.
    The autopilot's commands come through here too, in the same tick, after people's
    (`Controller.pilot_command`), so it decides from facts that already include them.

    Options are checked before anything else, and only known ones are logged: an option named like a
    field of the event (accepted, type, id) must neither crash the loop nor rewrite the log. A Quit
    earlier in this tick that exited refuses everything after it, unread."""
    via: dict = {}
    # The attempt open as the command arrives: only a command sent while an attempt is open hands it over.
    during = controller.attempt_number if controller.open_attempt is not None else None
    if controller.done:
        result = Result(name, False, f"{controller.command} is exiting.")
        options = {}
    elif unknown := set(options) - set(COMMANDS.get(name, ())):
        result = Result(name, False, f"{name.capitalize()} takes no {', '.join(sorted(map(str, unknown)))}.")
        options = {}
    elif name == PAGE_HIDDEN:
        return _hidden(controller, name, source, client)
    elif name == ANSWERING:
        result = _answering(controller, name, source, client, who, options)
    elif name == Command.QUIT and source == "page":
        result = Result(name, False, "Quit only from the robot's terminal (Esc or Ctrl+C): whoever ends the "
                                     "process should see the arm. Reset ends the attempt from here.")
    else:
        result = _control(controller, name, source, client, who)
        if result is None and name == TOGGLE:
            state = controller.state
            name, result = controller.toggle.press(step(state, Command.PAUSE, controller.facts()).accepted,
                                                    state in (State.PAUSED, State.RESTING))
            via = {"via": TOGGLE}  # logged as the pause or resume it became
        if result is None:
            if name == "outcome":
                result = controller.answer(options)
            elif name == "label":
                result = controller.label_step_in(options)
            elif name in TRANSITIONS:
                result = _transition(controller, name, options, source)
            elif name in TOOLS:
                result = controller.run_tool(Tool(name))
            elif name == AUTOPILOT_HOLD:
                result = controller.hold_autopilot(options)
            else:
                result = Result(name, False, f"Unknown command {name!r}.")
    # A person at the arm or on the page in control; a page that isn't in control counts for nothing. A Quit
    # counts only when accepted: a page's, always refused, never stops the autopilot.
    in_control = source in ("terminal", "signal") or (source == "page" and client is not None
                                                     and client == controller.link.client)
    if in_control and (name != Command.QUIT or result.accepted):
        controller.people.append((name, during))
    page = {"who": who} if source == "page" else {}
    controller.recorder.event("command", **options, command=name, accepted=result.accepted, reason=result.reason,
                              source=source, **page, **via, operator=controller.operator)
    return result


def _hidden(controller, name: str, source: str, client: str | None) -> Result:
    """The controlling page went out of sight. Logged as page_hidden, not as a command; from any
    other page, ignored."""
    link = controller.link
    if source != "page" or client is None or client != link.client:
        return Result(name, False, "Only the page in control reports whether it's in sight.")
    if link.hide(step(controller.state, Event.LINK_LOST, controller.facts()).accepted):
        controller.apply(Event.LINK_LOST)
    return Result(name, True)


def _answering(controller, name: str, source: str, client: str | None, who: str | None, options: dict) -> Result:
    """A person started saying how the open attempt went (`ANSWERING`). It changes nothing in the controller:
    being on `people` hands the attempt over (`autopilot.HANDS_OVER`). It never brings a lost link back
    (`hear=False`): it isn't proof the page is in sight, and the page's next poll in sight, every 250 ms
    (page.html's POLL_MS), is."""
    if source not in PEOPLE:
        return Result(name, False, "Only a person answers how an attempt went.")
    if (refused := _control(controller, name, source, client, who, hear=False)) is not None:
        return refused
    attempt = options.get("for_attempt")
    if controller.open_attempt is None or attempt != controller.attempt_number or type(attempt) is not int:
        return Result(name, False, f"Attempt {attempt!r} isn't the one open.")
    return Result(name, True)


def _control(controller, name: str, source: str, client: str | None, who: str | None, *,
             hear: bool = True) -> Result | None:
    """Take control, or refuse a page's command for lack of it. None lets the command through. With
    `hear`, a command from the page in control counts as hearing it, as a poll does."""
    link = controller.link
    if source != "page":
        return Result(name, False, ONLY_PAGES) if name == TAKE_CONTROL else None
    if not client:
        return Result(name, False, "The page didn't say which page it is. Reload it.")
    if name == TAKE_CONTROL:
        if client != link.client:
            link.take(client, who)
        return Result(name, True)
    if link.client is None:
        link.take(client, who, implicit=True)
    elif client != link.client:
        return Result(name, False, NOT_IN_CONTROL.format(who=link.who))
    elif hear:
        link.hear()  # a command shows the page is there as well as a poll does
    return None
