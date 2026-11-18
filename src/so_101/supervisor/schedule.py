"""When the supervisor's after_close and after_release checks fall due, on the model's clock.

One copy of the rules, which `supervisor.layer.Supervisor` sends its checks by and `failbench.candidates`
picks its candidate frames by, so the benchmark scores the moments the live supervisor checks:

- A close schedules an after_close check `after_close_s` later, and a release an after_release check
  `after_release_s` later, each about the newest close so far.
- An after_close check that falls due after a newer close came is stale: that close has its own check.
- A pause or resume (a handback) drops the checks not yet sent: each was about the scene before it.
  The layer's own playback or hold drops nothing.

Pure: it keeps its own list and touches nothing.
"""

from dataclasses import dataclass, field

# A check due exactly at a tick is sent on it: without the slack, float rounding in t + after_close_s could put it a tick later.
SLACK_S = 1e-9


@dataclass
class Schedule:
    after_close_s: float
    after_release_s: float
    closes: int = 0  # the closes so far in the attempt: an after_close check is sent only for the newest
    _pending: list[tuple[float, str, int]] = field(default_factory=list)  # (due on the model's clock, moment, close)

    def signal(self, kind: str, t: float) -> None:
        """The detector's signal `kind` at the model's `t`: a close or a release schedules its check."""
        if kind == "close":
            self.closes += 1
            self._pending.append((t + self.after_close_s, "after_close", self.closes))
        elif kind == "release":
            self._pending.append((t + self.after_release_s, "after_release", self.closes))

    def due(self, t: float) -> list[tuple[str, int, bool]]:
        """The checks due by the model's `t`, taken off the list in the order scheduled: (moment, the close it's
        about, stale). Stale: an after_close check for a close a newer one has replaced."""
        ready = [item for item in self._pending if item[0] <= t + SLACK_S]
        self._pending = [item for item in self._pending if item[0] > t + SLACK_S]
        return [(moment, close, moment == "after_close" and close != self.closes) for _, moment, close in ready]

    def drop(self) -> list[str]:
        """A pause or resume: the moments of the checks not yet sent, taken off the list."""
        dropped, self._pending = self._pending, []
        return [moment for _, moment, _ in dropped]
