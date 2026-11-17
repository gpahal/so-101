"""The autopilot `so101 run` adds to the control center: it starts and ends the attempts. docs/run.md gives
the reasons for each rule.

It is another operator: its commands carry source `autopilot`, so the state machine decides what's allowed.
The controller steps it once a tick, after people's commands and the link check (`Controller._step_autopilot`),
and carries out its command at once (`Controller.pilot_command`). A refused command is said once (`_refused`).

- A person starts attempt 1, and the first attempt after any fault (`Facts.after_fault`).
- At the rest pose with no attempt open, it submits Resume `reset_s` (`--reset-time`) after the arm arrives
  there or a person's last command (`ready_at`); never in Fault, away from rest, or while held (`hold`).
- It never starts motion (Resume, or Reset from a pause) unless a page in control watches: `states.step`
  refuses both on `Facts.unwatched`.
- After a page in control watches again it waits `--reset-time`, at least `link_timeout_s` + 0.5 s
  (`RELINK_MARGIN_S`), so a restore no poll follows is lost again before it starts anything (`_note`).
- A return to rest it already started finishes even if the page is lost, as every return to rest does,
  because stopping halfway would leave the arm held in mid-air.
- `so101 run` refuses `on_link_loss = "continue"` (`commands.serve.cmd_run`).
- The controller ends an attempt at its time limit; the autopilot only on a `done` ask, with Reset and no
  label. Every other pause is a person's (`_during`).
- An attempt ever unwatched while open is a person's for good (`handed`).
- A person's command in `HANDS_OVER`, sent while an attempt is open, hands it over (`Facts.people`).
- It quits at rest after `episodes` attempts, and stops for good on a person's accepted Quit (`_person_quit`).
"""

from collections.abc import Callable
from dataclasses import dataclass

SOURCE = "autopilot"
# How much longer than link_timeout_s the autopilot waits, after a page in control watches again, before it starts
# an attempt. The link check runs before the autopilot in each tick, but each reads the clock separately, so without
# a margin a restore that no poll follows could time out one tick after a start.
RELINK_MARGIN_S = 0.5
# The commands with which a person takes over from the autopilot the attempt open when the command arrived.
HANDS_OVER = frozenset({"pause", "resume", "rewind", "teleop", "reset", "quit", "answering"})
DONE = "done"  # `policy.base.DONE`: the policy stack judged the task done


@dataclass(frozen=True)
class Facts:
    """What the controller knows at the end of a tick, for the autopilot."""

    now: float  # the control loop's clock
    state: str  # the controller's state, as `states.State`'s values
    at_rest: bool  # every joint within rollout.TOLERANCE of the rest pose
    attempt: int  # the number of the attempt opened last
    attempt_open: bool
    # The kind of the ask the arm paused on (`policy.base.ASK_KINDS`), when the policy stack asked; None unless
    # the state is Paused.
    ask: str | None = None
    # The commands a person sent this tick from the terminal, a signal or the page in control, accepted or refused,
    # and a Quit only when accepted (`commands.intake`), each with the attempt open when it arrived, or None. Only a
    # command sent while an attempt is open hands it over, so the command that starts one (attempt 1's r, say)
    # leaves it to the autopilot like any other, and a Pause in the same tick, after it, doesn't.
    people: tuple[tuple[str, int | None], ...] = ()
    resume_refusal: str | None = None  # why Resume would be refused now, or None
    fault: str | None = None  # the error that caused Fault
    after_fault: bool = False  # the arm was in Fault since the last attempt opened: a person starts the next
    # Why no page watches the arm (none is in control, or its link is lost, whatever on_link_loss says), or None
    # when a page in control watches it. The state machine refuses the autopilot's Resume and Reset meanwhile.
    unwatched: str | None = None


class Autopilot:
    def __init__(self, *, command: Callable, say: Callable[[str], None], episodes: int, reset_s: float,
                 link_timeout_s: float) -> None:
        """`command(name)` carries out a command now and returns its `commands.Result`
        (`Controller.pilot_command`); `say` tells the operator. `link_timeout_s` is `[control] link_timeout_s`."""
        self._command, self._say = command, say
        self.episodes, self.reset_s, self.link_timeout_s = episodes, reset_s, link_timeout_s
        self.ended = 0  # attempts that have ended since the run started
        self.handed: int | None = None  # the attempt a person took over
        self.open: int | None = None  # the attempt open at the last step
        self.ready_at: float | None = None  # when the next attempt may start; None: at once
        self.held = False  # held from the page: no attempt starts until Continue
        self.done = False
        self._person_quit = False  # a person's Quit was accepted, even one that only asks to confirm: it stops for good
        self.waiting: str | None = None  # why it waits, said once and shown on the page
        # The command last refused, with the attempt it was for (None between attempts), and the state then:
        # not sent again until a person acts, the state changes, or an attempt opens or ends.
        self._refused: tuple[str, int | None, str] | None = None
        self._resting: bool | None = None  # resting at the rest pose with no attempt open, at the last step
        self._unwatched = False  # no page in control watched the arm at the last step

    def hold(self, held: bool) -> None:
        """Hold the next attempt back, or let it start: the page's Hold and Continue. Taken any time; during
        an attempt it changes only whether the next one starts."""
        self.held = held

    def status(self, now: float) -> dict:
        """The page's view: attempts made of how many, seconds until the next starts, why it waits, and
        whether it's held."""
        counting = not (self.done or self.held or self.open is not None or self.waiting) and self.ended < self.episodes
        next_in = max(self.ready_at - now, 0.0) if counting and self.ready_at is not None else 0.0 if counting else None
        return {"made": self.ended, "of": self.episodes, "next_in_s": None if next_in is None else round(next_in, 1),
                "waiting": self.waiting, "held": self.held}

    def step(self, f: Facts) -> None:
        self._note(f)
        if self.done:
            return
        if f.state == "fault":
            self._wait(f"The arm holds in Fault: {f.fault or 'see the event log'} Reset drives it to rest once the "
                       "follower answers. Then the next attempt, if the run has one left, waits for you to start it.")
        elif f.attempt_open:
            self._during(f)
        else:
            self._between(f)

    def _note(self, f: Facts) -> None:
        """What changed since the last step: an attempt ended, the arm arrived at rest, a person acted."""
        current = f.attempt if f.attempt_open else None
        if self.open is not None and current != self.open:
            self.ended += 1
        self.open = current
        if current is not None and f.unwatched and self.handed != current:
            self.handed = current
            self._say(f"Autopilot: attempt {current} goes to a person, as {f.unwatched}: the autopilot sends nothing "
                      "more for it. r resumes it; the right arrow or Reset ends it.")
        people = {name for name, _ in f.people}
        if self._refused is not None and (people or self._refused[1:] != (current, f.state)):
            self._refused = None
        if "quit" in people and not self._person_quit:
            self._person_quit = True
            if not self.done:
                self.waiting = None
                self._say("Autopilot: stopped, after your Quit.")
            self.done = True
        if current is not None and any(name in HANDS_OVER and during == current for name, during in f.people):
            self.handed = current
        if current is None and people:  # a person at the arm between attempts: the scene may not be ready
            self.ready_at = f.now + self.reset_s
        resting = f.state == "resting" and f.at_rest and current is None
        if resting and self._resting is False:  # arrived at rest
            self.ready_at = f.now + self.reset_s
        self._resting = resting
        if self._unwatched and not f.unwatched:  # a page watches again: the scene may have changed meanwhile
            # A restore with no poll in sight (a command or a poll that reached the server after page_hidden, or a
            # take_control that raced a hide) is lost again link_timeout_s after it. The link check runs earlier in
            # each tick, so waiting longer than that means such a loss always comes before this start.
            self.ready_at = f.now + max(self.reset_s, self.link_timeout_s + RELINK_MARGIN_S)
        self._unwatched = bool(f.unwatched)

    def _during(self, f: Facts) -> None:
        if self.handed == f.attempt or f.state != "paused":
            self.waiting = None
            return
        if f.ask == DONE:
            self._send(f, "reset", f"attempt {f.attempt} looks done to the supervisor: ending it unlabelled.")
            return
        self._wait(f"Attempt {f.attempt} is paused: the autopilot leaves it to a person. r resumes it; the right "
                   "arrow or Reset ends it.")

    def _between(self, f: Facts) -> None:
        if f.state != "resting":
            self.waiting = None  # on the way to rest
            return
        made = f"{self.ended} attempt{'' if self.ended == 1 else 's'}"
        if self.ended >= self.episodes and not f.at_rest:
            self._wait(f"{made} made, so the run is over, but the arm rests away from the rest pose: the run ends "
                       "once it's at rest. Reset drives it there.")
            return
        if not f.at_rest:
            self._wait("The arm rests away from the rest pose, with no attempt open: Reset drives it to rest, and "
                       f"the autopilot carries on {self.reset_s:g} s after it gets there.")
            return
        if self.ended >= self.episodes:
            self.done = self._send(f, "quit", f"{made} made: the run ends here, at rest.")
            return
        if self.ended == 0:
            self._wait("Attempt 1 waits for you: r at the terminal or Resume on the page starts it, and the autopilot "
                       "makes the rest.")
            return
        if f.after_fault:
            self._wait("The arm was in Fault: the next attempt waits for you. r at the terminal or Resume on the page "
                       "starts it, and the autopilot makes the rest.")
            return
        if self.held:
            self._wait("Held: Continue on the page lets the next attempt start.")
            return
        if self.ready_at is not None and f.now < self.ready_at:
            self.waiting = None
            return
        if f.resume_refusal:
            self._wait(f"The next attempt can't start ({f.resume_refusal}).")
            return
        self._send(f, "resume", f"attempt {self.ended + 1} of {self.episodes}.")

    def _wait(self, why: str) -> None:
        """Wait for `why`: said once, until the autopilot sends a command or waits for something else."""
        if why != self.waiting:
            self.waiting = why
            self._say(f"Autopilot waits. {why}")

    def _send(self, f: Facts, command: str, message: str) -> bool:
        """Carry out `command` now, unless it was refused for this attempt and state already. Returns
        whether it was accepted. A refusal is said once, as a wait."""
        attempt = f.attempt if f.attempt_open else None
        if self._refused == (command, attempt, f.state):
            return False
        self.waiting = None
        self._say(f"Autopilot: {message}")
        result = self._command(command)
        if not result.accepted:
            self._refused = (command, attempt, f.state)
            self._wait(f"{command.capitalize()} was refused: {result.reason}")
        return result.accepted
