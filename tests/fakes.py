"""Fakes for the tests: the arms, cameras, policy, recorder, disk and terminal, without hardware.

The follower and leader are LeRobot's real SO101Follower, SO101Leader and Feetech motor code. Only
the layer under them is fake:
`FakeMotors` answers the calls LeRobot's motor bus makes on `rustypot.Bus`, whose 1.13 release has no
in-memory mode, and its motors move toward their goal positions as time passes on a `FakeClock`.
"""

import contextlib
import errno
import json
import logging
import os
import tempfile
import time
from pathlib import Path

# Keep the test output to the tests. Each of these printed hundreds of lines on every run:
# - SVT-AV1 prints a banner of about 20 lines for every encoder, and warnings that the tests' tiny
#   frames are too small for some of its tools. SVT_LOG=1 keeps its errors.
os.environ.setdefault("SVT_LOG", "1")

import av
import datasets.utils.logging
import numpy as np
import rustypot
import torch

from so_101 import svtshim
from so_101.control.recorder import EventTail, jsonable

# The recorder and build tests encode video with SVT-AV1. Restart with its macOS teardown crash fix
# loaded, as the commands do, so a test run can't die when an encoder closes. SVT_LOG carries over
# to the restarted process.
svtshim.ensure()

# The rest of the test output kept quiet, once imported:
# - LeRobot's video writer hands FFmpeg's logging back to FFmpeg's own printer, at its default INFO
#   level, which notes every MP4 it finishes ("Starting second pass: moving the moov atom").
av.logging.set_libav_level(av.logging.ERROR)
# - Progress bars from loading datasets. LeRobot's `SuppressProgressBars` turns them back on after
#   each load it wraps, so that is made a no-op here too.
datasets.utils.logging.disable_progress_bar()
datasets.utils.logging.enable_progress_bar = lambda: None
# - LeRobot warns each time it limits a goal to `max_relative_target`, with every joint's goal. The
#   tests that block a joint or jump the policy's actions do that on every tick, on purpose.
logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)
# - Without a system FFmpeg (CI has none), LeRobot warns on every video decode that torchcodec can't
#   load and that it decodes with PyAV instead.
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

# The recorder's streaming encoder queues 30 frames per camera and drops a frame when the queue stays
# full for 0.1 s. The real loop feeds one frame every 33 ms, so that takes a stall of over a second.
# The tests tick as fast as the CPU allows, so on a busy CI runner a 0.1 s stall of an encoder thread
# dropped a frame, and the build then rightly left that attempt out. An unbounded queue (size 0)
# never drops. The test of dropped frames sets the encoder's count itself.
from lerobot.datasets import video_utils

_encoder_init = video_utils.StreamingVideoEncoder.__init__
video_utils.StreamingVideoEncoder.__init__ = lambda self, *args, **kwargs: _encoder_init(
    self, *args, **{**kwargs, "queue_maxsize": 0})

STEPS_PER_TURN = 4096
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STS3215_MODEL = 777
# The STS3215 register names rustypot knows; the fake bus refuses any other, as the real one does.
STS3215_REGISTERS = frozenset(r.name for r in rustypot.Sts3215PyController.registers())


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeMotors:
    """In-memory stand-in for `rustypot.Bus`. Registers are by motor id and lowercase name.

    Each torqued motor moves toward its goal at `speed` raw steps per second of `clock` time, but
    never past a position in `stops`, as when the arm is blocked by an object. `goals` lists every
    Goal_Position sync write, the way a control loop sends goals, and `torque` every torque change,
    with the positions the motors were at then. `writes` lists both in the order they came, as
    ("goal_position", goal) and ("torque_enable", enabled). While `failing` is set, every read times
    out, as when the bus stops answering. `status` gives a motor's alarm bits, which every reply from it
    carries, as an overload's 32. A motor with its torque off stays where it is put, as a
    hand leaves it: set it with `move_by_hand`.

    The motors behave like the STS3215 around power: `power_off` turns their torque off and makes
    them silent, and writes to them go nowhere; `power_on` brings them back with torque off and
    their RAM at its power-on values. Writing a goal to a powered motor turns its torque on, so a
    restarted motor drives to the first goal it gets. `cut_motor_power` is the other way a power cut
    can go: the servos' chips keep running and answering, at a low voltage, but their motors are
    limp, so the arm stays where it's put (`move_by_hand`, as it falls). Their RAM, goal and torque
    included, is kept, so when `restore_motor_power` brings the motors back, each drives to the goal
    it last got.
    """

    def __init__(self, clock: FakeClock, calibration: dict[str, dict], positions: dict[str, int]):
        self.clock = clock
        self.speed = 3000.0
        self.names = {c["id"]: name for name, c in calibration.items()}
        self.registers: dict[int, dict[str, int]] = {}
        for name, c in calibration.items():
            self.registers[c["id"]] = {
                "model_number": STS3215_MODEL,
                "firmware_major_version": 3,
                "firmware_minor_version": 10,
                "min_position_limit": c["range_min"],
                "max_position_limit": c["range_max"],
                "homing_offset": c["homing_offset"],
                "present_position": positions[name],
                "goal_position": positions[name],
                "torque_enable": 0,
                "max_torque_limit": 1000,  # EEPROM: what torque_limit becomes at power-on
                "torque_limit": 1000,
                "present_load": 0,  # magnitude in 0.1% units, direction in bit 10
                "present_voltage": 120,  # 0.1 V units: a 12 V supply
            }
        self.goals: list[dict[str, int]] = []
        self.torque: list[tuple[bool, dict[str, int]]] = []
        self.stops: dict[str, int] = {}  # motor -> the raw position it can't rise above
        # motor -> how many steps short of its goal it settles under the arm's weight, as a servo with
        # LeRobot's P-only control does when it holds a loaded joint
        self.sag: dict[str, int] = {}
        self.writes: list[tuple[str, object]] = []
        self.status: dict[str, int] = {}  # motor -> the alarm bits in its replies
        self.drop_goal_writes = 0  # how many Goal_Position sync writes are lost on the bus, as a garbled packet
        self.failing = False
        # motor -> how many torque-off writes it acknowledges without carrying out, as a servo may after
        # a garbled packet: its torque stays on and a hand can't move it.
        self.ignore_torque_off: dict[str, int] = {}
        self.powered = True
        self.motor_power = True  # the motors can move: off while `cut_motor_power` holds
        self.open = True
        self._last = clock()

    def _settle(self) -> None:
        elapsed, self._last = self.clock() - self._last, self.clock()
        for i, r in self.registers.items():
            if r["torque_enable"] and self.motor_power:
                gap = r["goal_position"] - self.sag.get(self.names[i], 0) - r["present_position"]
                move = max(-self.speed * elapsed, min(self.speed * elapsed, gap))
                r["present_position"] = round(r["present_position"] + move)
            if (stop := self.stops.get(self.names[i])) is not None:
                r["present_position"] = min(r["present_position"], stop)

    def present(self) -> dict[str, int]:
        self._settle()
        return {self.names[i]: r["present_position"] for i, r in self.registers.items()}

    def torqued(self) -> bool:
        return all(r["torque_enable"] for r in self.registers.values())

    def move_by_hand(self, positions: dict[str, int]) -> None:
        for i, r in self.registers.items():
            if self.names[i] in positions:
                r["present_position"] = positions[self.names[i]]

    def power_off(self) -> None:
        """The motors lose power: they go limp and stop answering."""
        self._settle()
        self.powered = False
        for r in self.registers.values():
            r["torque_enable"] = 0

    def power_on(self, positions: dict[str, int]) -> None:
        """The power comes back: torque off, SRAM back to its power-on values, and the arm wherever
        it fell to (`positions`)."""
        self._settle()
        self.powered = True
        for r in self.registers.values():
            r["torque_enable"] = 0
            r["torque_limit"] = r["max_torque_limit"]
        self.move_by_hand(positions)

    def cut_motor_power(self, volts: float = 4.0) -> None:
        """The supply drops but the servos' chips keep answering, reading `volts`: the motors go limp,
        keeping their goal and torque setting."""
        self._settle()
        self.motor_power = False
        for r in self.registers.values():
            r["present_voltage"] = round(volts * 10)

    def restore_motor_power(self, volts: float = 12.0) -> None:
        """The supply is back: each motor with its torque on drives to the goal it last got."""
        self._settle()
        self.motor_power = True
        for r in self.registers.values():
            r["present_voltage"] = round(volts * 10)

    def power_cycle(self, positions: dict[str, int]) -> None:
        """The motors lose power and get it back, as in a brownout: see `power_on`."""
        self.power_off()
        self.power_on(positions)

    def _answer(self, ids) -> None:
        """Time out, as the bus does, when no motor can answer."""
        if self.failing or not self.powered:
            raise TimeoutError(f"no status packet from ids {ids}")

    # The rustypot.Bus methods LeRobot's SerialMotorsBus calls.

    @staticmethod
    def _register(name: str) -> str:
        """`name` as the STS3215 table spells it. Like rustypot, a name the servo doesn't define raises."""
        if name.lower() not in STS3215_REGISTERS:
            raise ValueError(f"no register named '{name}'")
        return name.lower()

    def read_register_with_error(self, motor_id: int, name: str, retries: int = 0) -> tuple[int, int]:
        name = self._register(name)
        self._answer([motor_id])
        self._settle()
        return self.registers[motor_id].get(name, 0), self.status.get(self.names[motor_id], 0)

    def sync_read_register_with_error(self, ids: list[int], name: str, retries: int = 0) -> list[tuple[int, int]]:
        name = self._register(name)
        self._answer(ids)
        self._settle()
        return [(self.registers[i].get(name, 0), self.status.get(self.names[i], 0)) for i in ids]

    def sync_read_register(self, ids: list[int], name: str, retries: int = 0) -> list[int]:
        return [value for value, _ in self.sync_read_register_with_error(ids, name, retries)]

    def write_register_with_error(self, motor_id: int, name: str, value: int, retries: int = 0) -> int:
        name = self._register(name)
        self._answer([motor_id])  # it waits for the motor's status packet
        self._settle()
        self._write(motor_id, name, value)
        return 0

    def sync_write_register(self, ids: list[int], name: str, values: list[int], retries: int = 0) -> None:
        """A sync write gets no reply, so it never fails: unpowered motors just miss it."""
        name = self._register(name)
        self._settle()
        if name == "goal_position" and self.drop_goal_writes > 0:
            self.drop_goal_writes -= 1
            return
        for i, value in zip(ids, values, strict=True):
            self._write(i, name, value)
        if name == "goal_position":
            goal = {self.names[i]: int(v) for i, v in zip(ids, values, strict=True)}
            self.goals.append(goal)
            self.writes.append((name, goal))

    def _write(self, motor_id: int, name: str, value: int) -> None:
        if not self.powered:
            return
        self.registers[motor_id][name] = int(value)
        if name == "goal_position":
            self.registers[motor_id]["torque_enable"] = 1  # as the STS3215 does

    def set_torque(self, ids: list[int], enabled: bool, retries: int = 0) -> dict:
        if self.failing or not self.powered:
            return {i: "no status packet" for i in ids}  # the motors that didn't answer, as rustypot reports them
        self._settle()
        for i in ids:
            if not enabled and self.ignore_torque_off.get(self.names[i], 0) > 0:
                self.ignore_torque_off[self.names[i]] -= 1
                continue
            self.registers[i]["torque_enable"] = int(enabled)
        self.torque.append((enabled, self.present()))
        self.writes.append(("torque_enable", enabled))
        return {}

    def set_baudrate(self, baudrate: int) -> None:
        pass

    @property
    def is_open(self) -> bool:  # a property on rustypot.Bus
        return self.open

    def close(self) -> None:
        self.open = False


class FullDisk:
    """A run's open event file on a disk that is full while `full` is set.

    As with a real file, a write only fills Python's buffer, and it's the flush that fails: the
    buffer keeps the lines, and a flush once the disk has space writes them. A close while the disk
    is full fails too, and leaves the file closed."""

    def __init__(self, file) -> None:
        self.file = file
        self.full = False

    def _check(self) -> None:
        if self.full:
            raise OSError(errno.ENOSPC, "No space left on device")

    def write(self, text: str) -> int:
        return self.file.write(text)

    def flush(self) -> None:
        self._check()
        self.file.flush()

    def close(self) -> None:
        try:
            self._check()
        finally:
            self.file.close()

    def __getattr__(self, name: str):
        return getattr(self.file, name)


class NoRecorder:
    """The recorder of a controller with no run directory: it records nothing to disk. Events are still
    numbered and kept in memory, so the status and the event list show them."""

    attempt = None
    frames = 0
    run_dir = failure = dropped = None

    def __init__(self, clock) -> None:
        self.clock = clock
        self.started = clock()
        self.tail = EventTail()

    def event(self, type_: str, /, **fields) -> None:
        own = {"type": type_, "t": round(self.clock() - self.started, 3), "wall": round(time.time(), 3)}
        with contextlib.suppress(TypeError, ValueError, RecursionError):  # like `Recorder.event`, it never raises
            self.tail.number(jsonable({**own, **fields, **own}))

    def start_attempt(self, attempt: int) -> None:
        pass

    def add_frame(self, *args, **kwargs) -> None:
        pass

    def save_attempt(self, ended_by: str | None, outcome: str | None = None, **details) -> int | None:
        return None

    def discard_attempt(self, ended_by: str | None, **details) -> None:
        pass

    def close(self, run_end: dict | None = None) -> None:
        if run_end is not None:
            self.event("run_end", **run_end, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), episodes=0)


class FakeCamera:
    """A camera source that always has a black frame ready while connected.

    Set `is_connected` to False to make it report itself disconnected: its reads then raise, and
    LeRobot's follower counts itself as not connected, refusing `get_observation` and `send_action`.
    LeRobot's OpenCV camera does that only if OpenCV closes the device. A camera that's unplugged
    usually stays "connected" and fails its reads instead, as `stalled()` and `thread_died()` do."""

    def __init__(self) -> None:
        self.width, self.height = 64, 48
        self.is_connected = False
        self.failure: Exception | None = None  # what each read raises while connected

    def stalled(self) -> None:
        """No new frame: LeRobot's `read_latest` raises once the last one is over `max_age_ms` old."""
        self.failure = TimeoutError(f"{self} latest frame is too old: 812.4 ms (max allowed: 500 ms).")

    def thread_died(self) -> None:
        """LeRobot's read thread gave up after 10 failed reads in a row, so every read raises."""
        self.failure = RuntimeError(f"{self} read thread is not running.")

    def connect(self, warmup: bool = True) -> None:
        self.is_connected = True
        self.failure = None

    def disconnect(self) -> None:
        self.is_connected = False

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        if not self.is_connected:  # as LeRobot's `check_if_not_connected` on its cameras' reads
            from lerobot.utils.errors import DeviceNotConnectedError

            raise DeviceNotConnectedError(f"{type(self).__name__} is not connected. Run `.connect()` first.")
        if self.failure is not None:
            raise self.failure
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    read = async_read = read_latest


def calibration() -> dict[str, dict]:
    return {
        name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": STEPS_PER_TURN - 1}
        for i, name in enumerate(MOTORS, start=1)
    }


def make_follower(clock: FakeClock, positions: dict[str, int]):
    """LeRobot's SO-101 follower on fake motors and two fake cameras, wrapped and with hold on, as
    `so101-control` sets it up. Returns the wrapper and the motors."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    directory = Path(tempfile.mkdtemp(prefix="so101-test-"))
    (directory / "follower.json").write_text(json.dumps(calibration()))
    robot = SO101Follower(SO101FollowerConfig(
        port="/dev/fake", id="follower", calibration_dir=directory, use_degrees=True,
        max_relative_target=10.0,
    ))
    motors = FakeMotors(clock, calibration(), positions)
    robot.bus._open_bus = lambda: motors
    robot.cameras = {"top": FakeCamera(), "wrist": FakeCamera()}
    robot.connect(calibrate=False)
    wrapper = ThreadSafeRobot(robot)
    wrapper.configure_hold()
    return wrapper, motors


def make_leader(clock: FakeClock, positions: dict[str, int]):
    """LeRobot's SO-101 leader on fake motors, as `so101-control` makes it. Returns a function that
    makes a new, unconnected leader on the same motors each call, as replugging would, and the motors."""
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    directory = Path(tempfile.mkdtemp(prefix="so101-test-"))
    (directory / "leader.json").write_text(json.dumps(calibration()))
    motors = FakeMotors(clock, calibration(), positions)

    def open_bus() -> FakeMotors:
        motors.open = True
        return motors

    def make() -> SO101Leader:
        leader = SO101Leader(SO101LeaderConfig(port="/dev/fake-leader", id="leader", calibration_dir=directory,
                                               use_degrees=True))
        leader.bus._open_bus = open_bus
        return leader

    return make, motors


def control(**overrides):
    """so101.toml's `[control]` settings as `settings.load()` gives them when the file sets none,
    with `overrides`."""
    from so_101 import settings

    defaults = {key: setting.default for key, setting in settings.CONTROL_SETTINGS.items()}
    return settings.Control(**{"rest_pose": None, "rest_pose_calibration": None, **defaults, **overrides})


def make_engine(actions: list[list[float]]):
    """A LeRobot inference engine that returns `actions` in order, then repeats the last one.

    Set `waiting` to give no action for that many more ticks, as a remote engine does while it waits
    for the cloud GPU's next chunk: LeRobot's dispatch gate then sends nothing."""
    from lerobot.inference.base import InferenceEngine

    class ScriptedEngine(InferenceEngine):
        def __init__(self) -> None:
            super().__init__(task="test")
            self.actions = list(actions)
            self.next = 0
            self.error: str | None = None  # set it to fail the engine, as a lost remote server does
            self.waiting = 0

        @property
        def failed(self) -> bool:
            return self.error is not None

        @property
        def failure_traceback(self) -> str | None:
            return self.error

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def reset(self) -> None:
            pass

        @property
        def control_thread_owns_policy(self) -> bool:
            return True  # inference runs inline on the control thread, as LeRobot's sync engine does

        def get_action(self, obs_frame):
            if self.waiting > 0:
                self.waiting -= 1
                return None  # starved: no action this tick
            action = self.actions[min(self.next, len(self.actions) - 1)]
            self.next += 1
            return torch.tensor(action, dtype=torch.float32)

    return ScriptedEngine()



class DeadStream:
    """stdout or stderr once the terminal is closed: every write fails with EIO."""

    def write(self, text: str) -> int:
        raise OSError(errno.EIO, "Input/output error")

    def flush(self) -> None:
        raise OSError(errno.EIO, "Input/output error")
