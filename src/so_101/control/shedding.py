"""The control loop's own rate, and the optional work it sheds when it falls behind.

`Pace` counts the loop's ticks over each RATE_WINDOW_S. When the rate crosses SLOW_FRACTION of the
task's fps, in either direction, it logs a `loop_rate` event; nothing else changes, and the policy
runs on. Previews are the only optional work: while the loop is slow they drop to half rate, then
stop, and they step back up after PREVIEW_RECOVER_S of keeping up. With nothing left to shed, the
status warns with the measured rate.
"""

from collections.abc import Callable

from so_101.control.previews import Previews

RATE_WINDOW_S = 1.0  # the loop measures its own rate over this long
SLOW_FRACTION = 0.95  # below this share of the task's fps, the loop is falling behind
# Seconds the loop must keep up before shed previews step back up: long enough that turning them
# back on doesn't flip the loop between slow and keeping up every second.
PREVIEW_RECOVER_S = 5


class Pace:
    """`record(type, **fields)` logs an event and `say(message)` tells the operator."""

    def __init__(self, fps: float, previews: Previews | None, clock: Callable[[], float],
                 record: Callable[..., None], say: Callable[[str], None]) -> None:
        self.fps = fps
        self.previews = previews
        self.clock = clock
        self.record = record
        self.say = say
        self.hz: float | None = None  # measured over the last RATE_WINDOW_S
        self.preview_note: str | None = None  # why previews are slower or off
        self.behind: str | None = None  # the loop falls behind with nothing left to shed
        self._started: float | None = None
        self._ticks = 0
        self._slow = False
        self._keeping_up_s = 0.0  # since the loop last fell behind, while previews are shed

    def restart(self) -> None:
        """Start a new window from the next tick: the time since the last one wasn't the loop's."""
        self._started = None

    def tick(self) -> None:
        """Count a tick, and at the end of each window act on the rate."""
        now = self.clock()
        if self._started is None:
            self._started, self._ticks = now, 0
            return
        self._ticks += 1
        elapsed = now - self._started
        if elapsed < RATE_WINDOW_S:
            return
        self.hz = self._ticks / elapsed
        self._started, self._ticks = now, 0
        slow = self.hz < SLOW_FRACTION * self.fps
        self._shed(slow, elapsed)
        if slow is self._slow:
            return
        self._slow = slow
        self.record("loop_rate", hz=round(self.hz, 1), target_hz=self.fps, slow=slow)
        if slow:
            self.say(f"The control loop runs at {self.hz:.1f} Hz, under {SLOW_FRACTION:.0%} of the task's "
                     f"{self.fps:g} fps. The policy keeps running; recording and control keep priority.")
        else:
            self.say(f"The control loop is back to {self.hz:.1f} Hz.")

    def _shed(self, slow: bool, elapsed: float) -> None:
        """Halve the previews' rate, then stop them, while the loop is slow, and step back up after
        PREVIEW_RECOVER_S of keeping up."""
        previews, hz = self.previews, self.hz
        target = f"under {SLOW_FRACTION:.0%} of the task's {self.fps:g} fps"
        if slow:
            self._keeping_up_s = 0.0
            if previews is not None and previews.fps > 0:
                fps = previews.target_fps / 2 if previews.fps == previews.target_fps else 0
                previews.set_fps(fps)
                self.preview_note = (
                    f"Previews at {fps:g} fps: the control loop ran at {hz:.1f} Hz, {target}." if fps
                    else f"Previews off: the control loop ran at {hz:.1f} Hz, {target}. They come back once it keeps up."
                )
                self.say(self.preview_note)
                self.behind = None
            else:
                self.behind = (f"The control loop runs at {hz:.1f} Hz{' with previews off' if previews else ''}, "
                               f"{target}. Recording and control keep priority; recorded frames are further apart.")
            return
        self.behind = None
        if previews is None or previews.fps == previews.target_fps:
            return
        self._keeping_up_s += elapsed
        if self._keeping_up_s < PREVIEW_RECOVER_S:
            return
        self._keeping_up_s = 0.0
        fps = previews.target_fps / 2 if previews.fps == 0 else previews.target_fps
        previews.set_fps(fps)
        self.preview_note = (None if fps == previews.target_fps else
                             f"Previews at {fps:g} fps: the control loop keeps up again, at {hz:.1f} Hz. "
                             f"They go back to {previews.target_fps:g} fps after {PREVIEW_RECOVER_S} s more.")
        self.say(f"Previews back to {fps:g} fps: the control loop runs at {hz:.1f} Hz.")
