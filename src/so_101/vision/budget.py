"""One spending cap for paid vision calls, shared by `so101-failbench run` and the supervisor's judges.

Each call reserves its estimate before it starts, under the lock, so calls out at once can't all pass
against the same balance, and settles it to its real cost once it's back. A call starts only if its
estimate fits under the cap (`spent + estimate <= max_cost`), so no call is started that would take the
total past it; one whose cost comes back above its estimate can still leave the total a little over.
"""

import threading
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_cost: float
    spent: float = 0.0  # what the calls back have cost, with the estimates of the calls still out
    stopped: bool = False  # a call was refused for want of budget
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, estimate: float) -> bool:
        """Reserve `estimate` for a call about to start. Returns False, reserving nothing and stopping the
        budget, when it wouldn't fit under `max_cost`."""
        with self.lock:
            if self.spent + estimate > self.max_cost:
                self.stopped = True
                return False
            self.spent += estimate
            return True

    def settle(self, estimate: float, actual: float) -> None:
        """Replace a call's reserved `estimate` with what it cost: 0 for a call that never ran."""
        with self.lock:
            self.spent += actual - estimate

    def exhausted(self) -> bool:
        """No call may start any more: one was refused, or the calls have spent the whole cap."""
        with self.lock:
            return self.stopped or self.spent >= self.max_cost
