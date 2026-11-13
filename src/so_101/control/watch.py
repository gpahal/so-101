"""Watch for devices being unplugged, off the control loop.

Every 2 s a thread asks `probe` which devices are connected, such as `devices.presence`, which lists
USB serial ports and cameras without opening them. The control loop reads the latest answer with
`snapshot`, which never blocks, so a slow USB listing never costs a tick. Only the loop reads the
arm's bus: two readers on one serial port corrupt each other's packets.
"""

import threading
from collections.abc import Callable

INTERVAL_S = 2.0


class DeviceWatch:
    def __init__(self, probe: Callable[[], dict[str, str | None]]) -> None:
        self.probe = probe
        self._latest: dict[str, str | None] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="device-watch", daemon=True)

    def start(self) -> None:
        """Probe once now, so the first tick already knows, then every INTERVAL_S."""
        self._check()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict[str, str | None]:
        """For each watched device: None when connected, or why it isn't and the fix."""
        return self._latest  # replaced whole, never mutated, so reading it needs no lock

    def _check(self) -> None:
        try:
            self._latest = dict(self.probe())
        except Exception:
            pass  # keep the last answer; read errors in the loop still catch a lost device

    def _loop(self) -> None:
        while not self._stop.wait(INTERVAL_S):
            self._check()
