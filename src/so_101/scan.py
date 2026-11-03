"""Ping motor ids on a bus at each baud rate LeRobot supports. Read-only.

Pings each id on its own: broadcast ping (what `FeetechMotorsBus.scan_port` uses)
misses motors intermittently on macOS.

A motor in a protection alarm (blinking red LED) still answers, with error bits set
in its status packet. `FeetechMotorsBus.ping` drops those replies, so this calls the
packet handler directly and keeps the error.
"""

from dataclasses import dataclass

from lerobot.motors.feetech import FeetechMotorsBus
from scservo_sdk import COMM_SUCCESS

MOTOR_IDS = range(1, 11)
NUM_TRIES = 3

# Status packet error bits, from scservo_sdk.protocol_packet_handler.
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


def ping(bus: FeetechMotorsBus, id_: int) -> Reply | None:
    for _ in range(NUM_TRIES):
        model, comm, error = bus.packet_handler.ping(bus.port_handler, id_)
        if comm == COMM_SUCCESS:
            return Reply(model, error)
    return None


def scan(port: str) -> dict[int, dict[int, Reply]]:
    """Return {baud rate: {motor id: reply}} for every baud rate where a motor answered."""
    bus = FeetechMotorsBus(port, {})
    bus._connect(handshake=False)
    found = {}
    try:
        for baudrate in bus.available_baudrates:
            bus.set_baudrate(baudrate)
            ids = {id_: reply for id_ in MOTOR_IDS if (reply := ping(bus, id_)) is not None}
            if ids:
                found[baudrate] = ids
    finally:
        bus.port_handler.closePort()
    return found
