"""The autopilot `so101 run` adds to the control center: it starts and ends the attempts (docs/run.md).

It is another operator, like the terminal and the page: it reads the controller's status and event
log, and submits commands with source `autopilot`, so the state machine still decides what's allowed
and every safety rule holds. Its own thread calls `step` every POLL_S; tests call `step` themselves.

- **Between attempts,** at rest with no attempt open, it waits `reset_s` (none before the first
  attempt), then submits Resume: an attempt opens, the arm moves to the demos' start pose and the
  policy starts.
- **An attempt ends** at its time limit (the controller's, unlabelled), or after the policy stack asked
  for a person and nobody acted for ASK_GRACE_S: it submits Reset with no outcome.
- **A person's command** during an attempt (from the terminal, the page or a signal) hands that attempt
  to them: the autopilot neither resumes nor ends it, and carries on once it has ended.
- **It stops** once `episodes` attempts have ended, by submitting Quit, which exits at rest; at a
  person's Quit; or when Resume is refused (a fault, a full disk), saying why, until Resume is allowed.
"""

import threading
from collections.abc import Callable

SOURCE = "autopilot"
POLL_S = 0.1
ASK_GRACE_S = 10.0  # after the policy stack asks for a person, how long the autopilot waits for one
PEOPLE = ("terminal", "page", "signal")


class Autopilot:
    def __init__(self, *, submit: Callable, status: Callable[[], dict], events: Callable, say: Callable[[str], None],
                 clock: Callable[[], float], episodes: int, reset_s: float) -> None:
        """`submit(command, source)` and `status()` and `events(after, limit)` are the controller's."""
        self._submit, self._status, self._events, self._say, self.clock = submit, status, events, say, clock
        self.episodes, self.reset_s = episodes, reset_s
        self.ended = 0  # attempts that have ended since the run started
        self.handed: int | None = None  # the attempt a person took over
        self.asked_at: float | None = None  # when the policy stack asked for a person, in the open attempt
        self.ended_at: float | None = None  # when the last attempt ended; None before the first
        self.open: int | None = None  # the attempt open at the last step
        self.refusal: str | None = None
        self.done = False
        self._last_event = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="autopilot", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(POLL_S):
            try:
                self.step()
            except Exception as error:  # the autopilot must never take the run down with it
                self._say(f"The autopilot stopped: {type(error).__name__}: {error}. Run attempts from the page or keys.")
                return

    def step(self) -> None:
        status = self._status()
        self._read_events(status)
        now = self.clock()
        current = status["attempt"] if status["attempt_open"] else None
        if self.open is not None and current != self.open:  # the attempt that was open has ended
            self.ended += 1
            self.ended_at, self.asked_at = now, None
        self.open = current
        if self.done:
            return
        if current is None:
            self._between(status, now)
        elif self.handed != current and self.asked_at is not None and status["state"] == "paused" \
                and now - self.asked_at >= ASK_GRACE_S:
            self.asked_at = None
            self._send("reset", f"Nobody took over attempt {current} in {ASK_GRACE_S:g} s: ending it unlabelled.")

    def _between(self, status: dict, now: float) -> None:
        if self.ended >= self.episodes:
            self.done = True
            self._send("quit", f"{self.ended} attempt{'' if self.ended == 1 else 's'} made: the run ends at rest.")
            return
        if status["state"] != "resting" or (self.ended_at is not None and now - self.ended_at < self.reset_s):
            return
        if not status["allowed"].get("resume"):
            if (reason := status["refusals"].get("resume")) != self.refusal:
                self.refusal = reason
                self._say(f"The autopilot waits: the next attempt can't start ({reason}).")
            return
        self.refusal = None
        self._send("resume", f"Attempt {self.ended + 1} of {self.episodes}.")

    def _read_events(self, status: dict) -> None:
        for event in self._events(self._last_event, 500):
            self._last_event = max(self._last_event, event.get("id", 0))
            kind = event.get("type")
            if kind == "command" and event.get("source") in PEOPLE and event.get("accepted"):
                if event.get("command") == "quit":
                    self.done = True
                elif event.get("attempt") is not None:
                    self.handed = event["attempt"]
            elif kind == "state" and event.get("cause") == "policy_asked":
                self.asked_at = self.clock()

    def _send(self, command: str, message: str) -> None:
        self._say(f"Autopilot: {message}")
        self._submit(command, source=SOURCE)
