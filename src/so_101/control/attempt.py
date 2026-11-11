"""The open attempt: what the control loop keeps about one attempt, from when it opens until it closes.

The controller makes an `Attempt` when the state machine opens one (OPEN_ATTEMPT) and closes it at
rest, where a quit's drive to rest stopped short, or at a fault, saving it there. Nothing
about one attempt carries into the next, because nothing is reset by hand. The attempt closed last is
kept, so the page still shows how it went until the next one opens, and an outcome given later (at
rest after a time limit, or with the Reset after a fault) can label it. Every saved one is kept for
the run's `summary`, without its rewind history.

An attempt ends (END_ATTEMPT) before it closes: its `ended_by` says why, while the arm returns to rest.
A Pause on the way cancels the ending and its outcome, and the attempt stays open.
"""

from dataclasses import dataclass, field

from so_101.control.rewind import History
from so_101.control.states import State

# The states whose time counts toward a limit, and the setting that holds each limit. Paused,
# Starting, Handover, Rewinding, Returning and other scripted time count toward none, so a long pause never ends
# an attempt and a correction never uses up the policy's time.
LIMITED = {State.AUTONOMOUS: "attempt_limit_s", State.TELEOP: "human_limit_s"}
COUNTS = {"attempt_limit_s": "policy time", "human_limit_s": "Teleop time"}


@dataclass
class Attempt:
    number: int
    started: float  # the loop's clock when it opened
    spent: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COUNTS, 0.0))  # seconds toward each limit
    waited_s: float = 0.0  # of the seconds toward attempt_limit_s, those where the policy sent no action
    limit_hit: str | None = None  # the limit it used up last, which a time_limit ending names
    underway: bool = False  # the policy or a human has driven in it
    policy_drove: bool = False  # it has recorded a frame where the policy's action reached the arm
    teleop_next: bool = False  # the move to the start pose hands over to the leader, not the policy
    history: History | None = field(default_factory=History)  # what Rewind plays back; None once closed
    # Runs of one phase, each {"phase", "s", "stepped_in"}.
    timeline: list[dict] = field(default_factory=list)
    stepped_in: bool = False  # the next run of the timeline starts with a step-in
    step_ins: int = 0
    label: str | None = None  # why the operator stepped in last, once labelled
    # Set by END_ATTEMPT: reset, time_limit, quit or fault, and the outcome given, if any. A Pause on
    # the way to rest clears both.
    ended_by: str | None = None
    outcome: str | None = None
    episode: int | None = None  # the run's episode it was saved as; None until saved, or if the save failed

    def count(self, limit: str, seconds: float, limit_s: float) -> bool:
        """Add a tick's seconds toward `limit`. Returns whether the attempt has used up its `limit_s`."""
        self.spent[limit] += seconds
        if self.spent[limit] < limit_s:
            return False
        self.limit_hit = limit
        return True

    def step_in(self) -> None:
        """Someone stepped in on the policy: a new run of the timeline starts with it, not yet labelled."""
        self.stepped_in = True
        self.step_ins += 1
        self.label = None

    def extend_timeline(self, state: State, seconds: float) -> None:
        """Add a tick of `seconds` in `state` to the timeline."""
        if not self.timeline or self.timeline[-1]["phase"] != state.value:
            self.timeline.append({"phase": state.value, "s": 0.0, "stepped_in": self.stepped_in})
            self.stepped_in = False
        self.timeline[-1]["s"] += seconds

    @property
    def policy_s(self) -> float:
        """Seconds the policy drove: its time toward attempt_limit_s, less the ticks it sent no action."""
        return self.spent["attempt_limit_s"] - self.waited_s


def summary(attempts: list[Attempt]) -> dict:
    """The run so far, over its saved attempts: ignored ones count as attempts but not toward the
    policy's share of driving time or step-ins per attempt."""
    counted = [a for a in attempts if a.outcome != "ignored"]
    driven = sum(a.policy_s + a.spent["human_limit_s"] for a in counted)
    return {
        "attempts": len(attempts),
        "succeeded": sum(a.outcome == "succeeded" for a in attempts),
        "ignored": len(attempts) - len(counted),
        # Of the time anyone drove (the policy or a human in Teleop), the share the policy drove. A tick
        # where the policy sent no action is no one driving, as in `so101 runs`.
        "policy_share": round(sum(a.policy_s for a in counted) / driven, 3) if driven else None,
        "step_ins_per_attempt": round(sum(a.step_ins for a in counted) / len(counted), 2) if counted else None,
    }
