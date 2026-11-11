"""Which control page has control, and whether it still polls.

One page controls at a time. The controlling page's polls are the link's heartbeat. The web server's
threads only report polls (`heartbeat`); the control loop judges the gap with its own clock (`check`):
after `link_timeout_s` without one, it logs `link_lost`, and with `on_link_loss = "pause"` the loop
pauses a running policy, or a move to the start pose that would start it. The next poll logs
`link_restored`.

A phone's browser slows or stops a hidden page's timers, so the page also says when it's hidden
(the screen locked, or another app or tab in front): the link is lost at once, logged as
`page_hidden`, and the polls heard before the report are dropped. Each poll says whether the page is
visible, and only a visible page's polls count, so the link comes back with the first poll after the
page is back in sight, even if the page's own "visible" report was lost on the way. A visible poll
that reaches the server after the hide report can bring the link back early; the pause has already
happened, and the next timeout loses the link again.
"""

import threading
from collections.abc import Callable

from so_101.settings import Control

# Why a page that doesn't control the robot can't act. Any page can take control.
NOT_IN_CONTROL = "Another page has control ({who}). Take control to use this page."
ONLY_PAGES = "Only a page takes control: the terminal works whichever page has it."


class Link:
    """`record(type, **fields)` logs an event; `say(message)` tells the operator. `operator` is the
    name given with `so101 serve --operator`, or None: it goes on `control_taken`, which a person
    causes, but not on `link_lost` or `link_restored`."""

    def __init__(self, control: Control, clock: Callable[[], float], record: Callable[..., None],
                 say: Callable[[str], None], operator: str | None = None) -> None:
        self.timeout_s = control.link_timeout_s
        self.on_loss = control.on_link_loss
        self.clock = clock
        self.record = record
        self.say = say
        self.operator = operator
        self.client: str | None = None  # the client id of the page in control, or None
        self.who: str | None = None  # whose that page is, as the status names it
        self._beats: set[str] = set()  # the pages that polled in sight since the last tick
        self._beats_lock = threading.Lock()
        self.heard_at: float | None = None  # when the loop last heard the controlling page
        self.lost = False

    def heartbeat(self, client: str, visible: bool = True) -> None:
        """A page polled the status, and said whether it's in sight. Called from the web server's
        threads. Only a poll in sight counts."""
        if not visible:
            return
        with self._beats_lock:
            self._beats.add(client)

    def take(self, client: str, who: str | None, implicit: bool = False) -> None:
        """Make the page `client` the one in control. `who` names its caller: a Tailscale login,
        "this Mac" or an IP address."""
        previous = self.who if self.client is not None else None
        self.client, self.who = client, who or "an unknown caller"
        self.record("control_taken", who=self.who, previous=previous, implicit=implicit,
                    operator=self.operator)
        self.say(f"A page ({self.who}) has control." if previous is None else
                 f"A page ({self.who}) took control from {previous}.")
        self.hear()

    def hide(self, driving: bool) -> bool:
        """The controlling page went out of sight. Lose the link now, and drop its polls heard before
        the report; returns whether to pause, as `check` does."""
        with self._beats_lock:
            self._beats.discard(self.client)
        if self.lost:
            return False
        return self._lose("page_hidden", f"The controlling page ({self.who}) was hidden (screen locked or another "
                          "app in front).", driving, explain=False)

    def hear(self) -> None:
        """The controlling page polled, or just took control."""
        now = self.clock()
        if self.lost:
            self.lost = False
            self.record("link_restored", who=self.who, seconds=round(now - self.heard_at, 3))
            self.say(f"The controlling page ({self.who}) is back.")
        self.heard_at = now

    def check(self, driving: bool) -> bool:
        """Hear the controlling page's polls since the last tick, or lose the link after
        `link_timeout_s` without one. `driving` is whether the policy drives, or the arm moves to the
        start pose to start it. Returns whether to pause now: with on_link_loss "pause", only when the
        link is lost, so a policy started at the robot's terminal while it stays lost runs on."""
        with self._beats_lock:
            beats, self._beats = self._beats, set()
        if self.client is None:
            return False
        if self.client in beats:  # the page in control, in sight
            self.hear()
            return False
        gap = self.clock() - self.heard_at
        if not self.lost and gap >= self.timeout_s:
            return self._lose("link_lost", f"No poll from the controlling page ({self.who}) for {self.timeout_s:g} s.",
                              driving, explain=True, seconds=round(gap, 3))
        return False

    def _lose(self, event: str, why: str, driving: bool, *, explain: bool, **fields) -> bool:
        """Lose the link: log `event` with `fields`, and tell the operator `why` and what becomes of
        the policy. The one path for a timeout and for a hidden page. With `explain`, the operator
        is also told when a running policy keeps running. Returns whether to pause now."""
        self.lost = True
        pause = self.on_loss == "pause" and driving
        self.record(event, who=self.who, **fields, on_link_loss=self.on_loss)
        if pause:
            self.say(f"{why} Pausing: the policy never runs unsupervised.")
        elif explain and driving:
            self.say(f'{why} The policy keeps running: on_link_loss is "continue".')
        else:
            self.say(why)
        return pause
