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

The intake is the controller's own, split out of `controller.py`: it reads the controller's state
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
    if "for_attempt" in options and (reason := stale_sheet(options["for_attempt"], controller._attempt)):
        return Result(name, False, reason)
    transition = controller._apply(Command(name), outcome=outcome, source=source)
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
}
# Every option some command takes. The web server refuses a request carrying any other key before it
# reaches the loop. for_attempt: the attempt the Reset sheet was opened for, or the time-limit prompt asked about.
OPTIONS = tuple(dict.fromkeys(option for options in COMMANDS.values() for option in options))
# The names of the state machine's commands, and of the read-only tools.
TRANSITIONS = frozenset(command.value for command in Command)
TOOLS = frozenset(tool.value for tool in Tool)


# --- Taking a command in --------------------------------------------------------------------------


def intake(controller, name: str, source: str, client: str | None, who: str | None, options: dict) -> Result:
    """Take in one command on the loop's thread: check its options, then page control, then carry
    it out, and log it as a `command` event (page_hidden is logged only as the link's `page_hidden` event, when it loses the link).

    Options are checked before anything else, and only known ones are logged: an option named like a
    field of the event (accepted, type, id) must neither crash the loop nor rewrite the log. A Quit
    earlier in this tick that exited refuses everything after it, unread."""
    via: dict = {}
    if controller.done:
        result = Result(name, False, "serve is exiting.")
        options = {}
    elif unknown := set(options) - set(COMMANDS.get(name, ())):
        result = Result(name, False, f"{name.capitalize()} takes no {', '.join(sorted(map(str, unknown)))}.")
        options = {}
    elif name == PAGE_HIDDEN:
        return _hidden(controller, name, source, client)
    elif name == Command.QUIT and source == "page":
        result = Result(name, False, "Quit only from the robot's terminal (Esc or Ctrl+C): whoever ends the "
                                     "process should see the arm. Reset ends the attempt from here.")
    else:
        result = _control(controller, name, source, client, who)
        if result is None and name == TOGGLE:
            state = controller._state
            name, result = controller._toggle.press(step(state, Command.PAUSE, controller._facts()).accepted,
                                                    state in (State.PAUSED, State.RESTING))
            via = {"via": TOGGLE}  # logged as the pause or resume it became
        if result is None:
            if name == "outcome":
                result = controller._answer(options)
            elif name == "label":
                result = controller._label_step_in(options)
            elif name in TRANSITIONS:
                result = _transition(controller, name, options, source)
            elif name in TOOLS:
                result = controller._tool(Tool(name))
            else:
                result = Result(name, False, f"Unknown command {name!r}.")
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
    if link.hide(step(controller._state, Event.LINK_LOST, controller._facts()).accepted):
        controller._apply(Event.LINK_LOST)
    return Result(name, True)


def _control(controller, name: str, source: str, client: str | None, who: str | None) -> Result | None:
    """Take control, or refuse a page's command for lack of it. None lets the command through."""
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
    else:
        link.hear()  # a command shows the page is there as well as a poll does
    return None
