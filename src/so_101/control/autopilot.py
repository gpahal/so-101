"""The autopilot `so101 run` adds to the control center: it starts and ends the attempts (docs/run.md).

It is another operator, like the terminal and the page: it submits commands with source `autopilot`,
so the state machine still decides what's allowed and every safety rule holds. The controller steps
it once a tick, after publishing the status, with the tick's `Facts`; it never reads the event log,
and it has one command in flight at a time, so a slow tick can't make it send a command twice. Its
command is taken in on the next tick, after any a person sent meanwhile; once a person's is accepted
there, `commands.intake` refuses the autopilot's (GAVE_WAY), which was decided before it, and the
autopilot decides again from the tick after's facts.

- **Between attempts,** resting at the rest pose with no attempt open, it waits `reset_s` from the
  arm's arrival at rest or a person's last command (none before the first attempt), then submits
  Resume: an attempt opens, the arm moves to the demos' start pose and the policy starts. It starts
  nothing while the arm rests away from the rest pose, in Fault, or while held from the page.
- **An attempt ends** at its time limit (the controller's, unlabelled), or when the arm pauses without
  a person's command: the policy stack asked for a person, the control page's link was lost, or a
  move stopped short. It says why once, and after ASK_GRACE_S with no person taking over submits
  Reset with no outcome. A `done` ask ends the attempt at once. While a person answers how the
  attempt went (the terminal's right-arrow question, the page's Reset sheet), it waits.
- **A person's Pause, Resume, Rewind, Teleop, Reset or Quit** during an attempt hands that attempt to
  them (`HANDS_OVER`): the autopilot neither resumes nor ends it, and carries on once it has ended.
  Check pose, Run checks, a label and taking control don't.
- **It stops** once `episodes` attempts have ended, by submitting Quit from rest at the rest pose;
  anywhere else it says the run is over and waits for a person to Reset. A person's Quit stops it too.
  In Fault it says why once and waits.
"""

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

SOURCE = "autopilot"
ASK_GRACE_S = 10.0  # after a pause the autopilot didn't ask for, how long it waits for a person
# The commands with which a person takes the attempt they're given in over from the autopilot.
HANDS_OVER = frozenset({"pause", "resume", "rewind", "teleop", "reset", "quit"})
# Why the arm may be paused with nobody's command (`Facts.paused_by`), and what the autopilot says.
UNOWNED = {
    "policy_asked": "the policy stack asks for a person",
    "link_lost": "the control page's link was lost",
    "stopped_short": "the move stopped short",
}
DONE = "done"  # `policy.base.DONE`: the policy stack judged the task done
# Why `commands.intake` refuses the autopilot's command once a person's was accepted in the same tick.
GAVE_WAY = ("A person's command came first in this tick, and the autopilot decided this one before it: it decides "
            "again from the next tick's facts.")


@dataclass(frozen=True)
class Facts:
    """What the controller knows at the end of a tick, for the autopilot."""

    now: float  # the control loop's clock
    state: str  # the controller's state, as `states.State`'s values
    at_rest: bool  # every joint within rollout.TOLERANCE of the rest pose
    attempt: int  # the number of the attempt opened last
    attempt_open: bool
    # Why the arm is paused: a command or event (`states`'s values), or "person" for an operator's command or a
    # move they started. None unless the state is Paused.
    paused_by: str | None = None
    ask: str | None = None  # the ask's kind (`policy.base.ASK_KINDS`), when paused_by is policy_asked
    people: tuple[str, ...] = ()  # the commands a person gave this tick that were accepted
    person_answering: bool = False  # a person is saying how the open attempt went
    resume_refusal: str | None = None  # why Resume would be refused now, or None
    fault: str | None = None  # the error that caused Fault


class Autopilot:
    def __init__(self, *, submit: Callable, say: Callable[[str], None], episodes: int, reset_s: float) -> None:
        """`submit(command, source=...)` is the controller's; `say` tells the operator."""
        self._submit, self._say = submit, say
        self.episodes, self.reset_s = episodes, reset_s
        self.ended = 0  # attempts that have ended since the run started
        self.handed: int | None = None  # the attempt a person took over
        self.open: int | None = None  # the attempt open at the last step
        self.ready_at: float | None = None  # when the next attempt may start; None: at once
        self.paused_at: float | None = None  # when the pause it didn't ask for began
        self.held = False  # held from the page: no attempt starts until Continue
        self.done = False
        self._person_quit = False  # a person's Quit was accepted: it stops for good
        self.waiting: str | None = None  # why it waits, said once and shown on the page
        self._pending: Future | None = None
        self._resting: bool | None = None  # resting at the rest pose with no attempt open, at the last step
        self._last_now: float | None = None

    def hold(self, held: bool) -> None:
        """Hold the next attempt back, or let it start: the page's Hold and Continue, between attempts only."""
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
        if self._pending is not None:
            if not self._pending.done():
                return
            result, self._pending = self._pending.result(), None
            if result.accepted:
                pass
            elif result.reason == GAVE_WAY:  # decide again, below, from this tick's facts
                if result.command == "quit" and not self._person_quit:
                    self.done = False
                self._say(f"Autopilot: a person's command came first, so its {result.command.capitalize()} was "
                          "dropped. It decides again from here.")
            else:
                self._wait(f"{result.command.capitalize()} was refused: {result.reason}")
        if self.done:
            return
        if f.state == "fault":
            self._wait(f"The arm holds in Fault: {f.fault or 'see the event log'} Reset drives it to rest once the "
                       "follower answers, and the autopilot carries on from there.")
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
        people = set(f.people)
        if "quit" in people and not self._person_quit:
            self._person_quit = True
            if not self.done:
                self.waiting = None
                self._say("Autopilot: stopped, after your Quit.")
            self.done = True
        if current is not None and people & HANDS_OVER:
            self.handed = current
        if current is None and people:  # a person at the arm between attempts: the scene may not be ready
            self.ready_at = f.now + self.reset_s
        resting = f.state == "resting" and f.at_rest and current is None
        if resting and self._resting is False:  # arrived at rest
            self.ready_at = f.now + self.reset_s
        self._resting = resting
        unowned = f.state == "paused" and f.paused_by in UNOWNED
        if not unowned:
            self.paused_at = None
        elif self.paused_at is None:
            self.paused_at = f.now
            if self.handed != current and not self.done:
                what = "the supervisor judged it done" if f.ask == DONE else UNOWNED[f.paused_by]
                end = "it ends now" if f.ask == DONE else f"it ends unlabelled in {ASK_GRACE_S:g} s unless you take over"
                self._say(f"Autopilot: attempt {current} paused, as {what}: {end} (r resumes, the right arrow or "
                          "Reset ends it with an outcome).")

    def _during(self, f: Facts) -> None:
        if self.handed == f.attempt or self.paused_at is None:
            self.waiting = None
            return
        if f.person_answering:
            self._wait(f"A person is saying how attempt {f.attempt} went: the autopilot leaves it to them.")
            return
        if f.ask == DONE:
            self._send("reset", f"attempt {f.attempt} looks done to the supervisor: ending it unlabelled.")
        elif f.now - self.paused_at >= ASK_GRACE_S:
            self._send("reset", f"nobody took over attempt {f.attempt} in {ASK_GRACE_S:g} s: ending it unlabelled.")

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
            self.done = True
            self._send("quit", f"{made} made: the run ends here, at rest.")
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
        self._send("resume", f"attempt {self.ended + 1} of {self.episodes}.")

    def _wait(self, why: str) -> None:
        """Wait for `why`: said once, until the autopilot sends a command or waits for something else."""
        if why != self.waiting:
            self.waiting = why
            self._say(f"Autopilot waits. {why}")

    def _send(self, command: str, message: str) -> None:
        self.waiting = None
        self._say(f"Autopilot: {message}")
        self._pending = self._submit(command, source=SOURCE)
