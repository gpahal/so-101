"""Ping motor ids on a bus at each baud rate the servos support. Read-only.

Asks each id on its own: broadcast ping (what `SerialMotorsBus` scans with) misses motors
intermittently on macOS.

A motor in a protection alarm (blinking red LED) still answers, with error bits set in its status
packet. `SerialMotorsBus.ping` drops those, so this reads through rustypot directly and keeps them.
"""

from dataclasses import dataclass

import rustypot

MOTOR_IDS = range(1, 11)
NUM_TRIES = 3
TIMEOUT_S = 0.05  # longer than the 16 ms latency timer of a USB adapter
DEFAULT_BAUDRATE = 1_000_000

# Status packet error bits of a Feetech servo.
ERROR_BITS = {1: "voltage", 2: "angle sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}


@dataclass(frozen=True)
class Reply:
    model: int
    error: int

    def errors(self) -> list[str]:
        names = [name for bit, name in ERROR_BITS.items() if self.error & bit]
        unknown = self.error & ~sum(ERROR_BITS)
        return names + ([f"0x{unknown:02x}"] if unknown else [])

    def __str__(self) -> str:
        return f"{self.model} (error: {', '.join(self.errors())})" if self.error else str(self.model)


def ping(bus: rustypot.Bus, id_: int) -> Reply | None:
    try:
        return Reply(*bus.read_register_with_error(id_, "model_number", retries=NUM_TRIES - 1))
    except (OSError, RuntimeError, TimeoutError):  # no answer from this id
        return None


def scan(port: str) -> dict[int, dict[int, Reply]]:
    """Return {baud rate: {motor id: reply}} for every baud rate where a motor answered."""
    definition = rustypot.Sts3215PyController.definition()
    bus = rustypot.Bus(port, DEFAULT_BAUDRATE, TIMEOUT_S, dict.fromkeys(MOTOR_IDS, definition))
    found = {}
    try:
        for baudrate in sorted(definition.baudrates):
            bus.set_baudrate(baudrate)
            ids = {id_: reply for id_ in MOTOR_IDS if (reply := ping(bus, id_)) is not None}
            if ids:
                found[baudrate] = ids
    finally:
        bus.close()
    return found
