"""The follower as the control loop reads and drives it: its joints, cameras, motors and power.

`Follower.observe` reads the arm, the cameras and, every `MOTORS_READ_S`, the motors' temperatures
and alarms. During an attempt every camera is required: its frames are recorded and the policy sees
them, so a camera that fails is a fault. With no attempt open only the follower is: a failing
camera is noted in `read_errors`, and the arm can still go to rest after a camera fault.

A motor that loses power comes back with torque off, and its first goal turns the torque on again
and drives it to that goal at full speed. To catch a restart, the loop sets each motor's
torque_limit (SRAM, address 48) one below its max_torque_limit (EEPROM, address 16), which the motor
copies into torque_limit at power-on. A torque_limit back at max_torque_limit means the motor restarted.
Any other value means something else changed it, such as the servo's own protection: that is a
fault of its own, which holds the last goal and leaves the value as found. The loop reads these
marks on every tick, before any goal goes out: about 1.5 ms of a 33 ms tick.

In Fault, a follower that may have lost power (it stopped answering, or a motor restarted) gets no
goal until it answers again, then holds where it's found (`hold_in_fault`). Any other fault, such as
a motor alarm, holds the last goal, which keeps a grip and keeps loaded joints from sagging.

Goals and holds need no camera, so they reach the arm even while a camera reports itself
disconnected (`Arm`), through the same step cap.
"""

import logging
import traceback
from collections.abc import Callable

from so_101.control.previews import Previews
from so_101.scan import Reply

TEMP_WARN_C = 55  # a warning before 60 °C, where `so101 check` fails: a joint is straining or the arm needs a rest
MOTORS_READ_S = 1.0  # how often the loop reads the motors' temperatures and alarms: one bus read
TORQUE_LIMIT_CHANGED = ("The follower's torque_limit changed without a restart: {listed}, where serve set it one below "
                        "max_torque_limit. Something else changed it, such as the servo's own overload protection; it's left "
                        "as found. The arm holds its last goal: look for a joint straining against something, then "
                        "Reset.")
POWER_LOST = ("The follower's {motors} lost power or restarted. The arm now holds where it is, not where it "
              "was told to go. Check the arm's power supply and cables, then Reset.")


class Failure(Exception):
    """A failure the loop found itself, such as a motor alarm or a failed policy. `device` names
    the device to blame, if any."""

    def __init__(self, message: str, device: str | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.device = device
        self.detail = detail  # a traceback worth keeping in the event log


class TorqueLimitChanged(Failure):
    """A motor's torque_limit reads neither the restart mark nor its max_torque_limit. The arm holds its last
    goal, and the value is left as found."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "follower")


class FollowerLost(Failure):
    """The follower may have lost power: it stopped answering a read, or a motor restarted. It gets
    no goal until it answers again, then holds where it's found (`Follower.hold_in_fault`)."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message, "follower", detail)


class Arm:
    """The follower as the loop drives it: remembers the goal sent this tick, after the step cap.

    Every goal goes through `send_action` or `hold`. A hold does what LeRobot's position hold does: it
    resends the last goal sent, or the measured pose before any goal, through `send_action` and so
    through `max_relative_target`. It goes through here rather than `ThreadSafeRobot.hold`, which
    returns nothing, so that `goal` is the capped goal the motors got; a robot that holds by itself
    (`supports_command_hold`) gets no goal. The cap matters when a hand
    pushes the arm more than a step from its goal: the hold then sends a goal a step from where the
    arm is, and that is what's recorded, never the uncapped goal nor the measured pose.

    A goal needs no camera, but LeRobot's follower refuses every goal while any of its cameras
    reports itself disconnected (`check_if_not_connected`, which asks the cameras too). So a goal is
    written through the follower's own `send_action` without that check: under the wrapper's lock,
    through the same step cap, and still refused by the motor bus when its own port is closed. That
    keeps the arm held after a camera fault, and lets Reset drive it to rest.
    """

    def __init__(self, robot) -> None:
        self.robot = robot
        self.goal: dict[str, float] | None = None
        self.present: dict[str, float] = {}

    def send_action(self, action: dict) -> dict:
        sent = self._write(action)
        self.goal = {key: float(value) for key, value in sent.items()}
        return sent

    def _write(self, action: dict) -> dict:
        """Send `action` through the follower's step cap, whether or not its cameras are connected."""
        inner = self.robot.inner
        write = type(inner).send_action.__wrapped__  # under LeRobot's connection check
        with self.robot._lock:  # ThreadSafeRobot's lock on the bus
            return write(inner, action)

    def hold(self) -> None:
        if self.robot.supports_command_hold:  # the robot holds by itself, sending no goal
            self.robot.hold()
            if self.goal is None:
                self.goal = dict(self.present)
            return
        self.send_action(dict(self.goal if self.goal is not None else self.present))

    def __getattr__(self, name: str):
        return getattr(self.robot, name)


def pose_from_steps(bus, steps: dict[str, int]) -> dict[str, float]:
    """A pose in raw motor steps (0-4095), such as `rest_pose`, in the units the bus works in."""
    by_id = {bus.motors[motor].id: value for motor, value in steps.items()}
    return {f"{bus._id_to_name(id_)}.pos": float(value) for id_, value in bus._normalize(by_id).items()}


class Follower:
    """The follower's reads and holds, and what they last found.

    `robot` is LeRobot's robot wrapper (ThreadSafeRobot). `present` is the pose measured last, in
    the arm's units and keyed like its actions; `present` starts it. `say(message)` tells the
    operator. With `previews`, each observation's camera frames are handed to it at its rate."""

    def __init__(self, robot, present: dict[str, float], clock: Callable[[], float],
                 say: Callable[[str], None], previews: Previews | None) -> None:
        self.robot = robot
        self.inner = robot.inner
        self.arm = Arm(robot)
        self.keys = tuple(present)  # the joints, in the order `observe` keeps them
        self.arm.present = present  # what a hold before the first tick keeps
        self.clock = clock
        self.say = say
        self.previews = previews
        self.read_errors: dict[str, str] = {}  # device -> why its last read failed, until one succeeds
        self.lost = False  # the follower stopped answering or lost power: hold where it is found
        self.marks: dict[str, int] = {}  # motor -> the torque_limit value that shows it hasn't restarted
        self.maxima: dict[str, int] = {}  # motor -> its max_torque_limit, which a restart copies into torque_limit
        self.changed: dict[str, int] = {}  # motor -> another torque_limit value, reported once and left as found
        self.alarm: str | None = None  # the follower's motors reported a protection alarm
        self.hot: str | None = None  # the warning about hot motors, while they are
        self.motors_read_at: float | None = None
        self.unblamed: str | None = None  # the last error of a hold that no device explains, logged once

    @property
    def present(self) -> dict[str, float]:
        return self.arm.present

    def cameras(self) -> dict[str, object]:
        """The open cameras, by camera name (the robot's camera keys are so101.toml's camera names)."""
        return dict(self.inner.cameras)

    # --- Reading ---------------------------------------------------------------------------------

    def observe(self, attempt_open: bool) -> dict:
        """Read the arm, its motors' restart marks and the cameras, and every `MOTORS_READ_S` the
        motors' temperatures and alarms. Raises FollowerLost when a motor restarted. With
        `attempt_open` every camera is required, and one that reports itself disconnected is a
        Failure naming it; otherwise a camera's error goes in `read_errors`."""
        if attempt_open:
            for name, camera in self.cameras().items():
                if not camera.is_connected:
                    raise Failure(f"The {name} camera reports itself disconnected. Check its USB cable; if it was "
                                  "unplugged, restart so101 serve.", name)
            observation = self.robot.get_observation()
        else:
            observation = self._read_follower()
        self.arm.present = {key: float(observation[key]) for key in self.keys}
        self.check_marks()  # every tick, before any goal: a restarted motor's first goal re-arms it
        now = self.clock()
        if self.previews is not None and self.previews.due(now):
            self.previews.put({key: observation[key] for key in self.inner.cameras if key in observation}, now)
        if self.motors_read_at is None or self.clock() - self.motors_read_at >= MOTORS_READ_S:
            self.motors_read_at = self.clock()
            self.read_motors()
        if attempt_open:
            self.read_errors.clear()
        else:
            self.read_errors.pop("follower", None)
        return observation

    def read_joints(self) -> None:
        """Read the follower's joints into `present`, and nothing else. Raises while it's silent."""
        self.arm.present = {key: float(value) for key, value in self._read_positions().items()}
        self.read_errors.pop("follower", None)

    def _read_positions(self) -> dict:
        """The follower's joint positions, as `get_observation` reads them. Raises while it's silent."""
        positions = self.inner.bus.sync_read("Present_Position", num_retry=self.inner.config.num_read_retries)
        return {f"{motor}.pos": value for motor, value in positions.items()}

    def _read_follower(self) -> dict:
        """The follower's joint positions, plus a frame from each camera that sends one. A camera's
        error is kept in `read_errors` instead of raised."""
        observation = self._read_positions()
        for name, camera in self.inner.cameras.items():
            try:
                observation[name] = camera.read_latest()
                self.read_errors.pop(name, None)
            except Exception as error:
                self.read_errors[name] = self.describe(error, name)
        return observation

    def read_motors(self) -> None:
        """One bus read for every motor's temperature, with the alarm bits of its status reply."""
        bus = self.inner.bus
        names = list(bus.motors)
        try:
            replies = bus._connected_bus.sync_read_register_with_error(
                [bus.motors[name].id for name in names], "present_temperature", retries=2
            )
        except Exception as error:
            raise FollowerLost(f"Reading the motors' temperatures failed: {error}") from error
        temps = {name: int(value) for name, (value, _) in zip(names, replies, strict=True)}
        alarms = {name: Reply(0, status).errors() for name, (_, status) in zip(names, replies, strict=True) if status}
        if alarms:
            listed = ", ".join(f"{name} ({', '.join(errors)})" for name, errors in alarms.items())
            # Kept until a read shows the alarm cleared, which only the motors' power cycle does.
            # The motors go limp at power-off, so the arm is supported or at rest first.
            self.alarm = (f"Motors in a protection alarm: {listed}. To clear it, support the arm, or Reset it to "
                          "rest if Reset is allowed, then switch the arm's power off and on. If it returns, look for "
                          "a jammed joint or a weak supply.")
            raise Failure(self.alarm, "follower")
        self.alarm = None
        hot = {name: t for name, t in temps.items() if t > TEMP_WARN_C}
        warning = (
            "Hot motors: " + ", ".join(f"{name} {t} °C" for name, t in hot.items())
            + f", above {TEMP_WARN_C} °C. Let the arm cool down, and look for a joint straining against something."
        ) if hot else None
        if warning and not self.hot:
            self.say(warning)
        self.hot = warning

    # --- Power marks -----------------------------------------------------------------------------

    def mark(self) -> None:
        """Set each motor's torque_limit one below its max_torque_limit, the mark `check_marks` looks for,
        and keep each max_torque_limit. A motor restart copies max_torque_limit back, so the mark tells a
        restart apart."""
        bus = self.inner.bus._connected_bus
        ids = {name: motor.id for name, motor in self.inner.bus.motors.items()}
        try:
            maxima = {name: int(value) for name, value in
                      zip(ids, bus.sync_read_register(list(ids.values()), "max_torque_limit", retries=2), strict=True)}
            marks = {name: max(value - 1, 0) for name, value in maxima.items()}
            bus.sync_write_register(list(ids.values()), "torque_limit", [marks[name] for name in ids], retries=2)
        except Exception as error:
            raise FollowerLost(f"Setting the motors' restart marks failed: {error}") from error
        self.marks, self.maxima, self.changed = marks, maxima, {}

    def check_marks(self) -> None:
        """Raise FollowerLost when a motor's torque_limit is back at its max_torque_limit: it lost power or
        restarted. Any other value than the mark is something else, such as the servo's own
        protection: a Failure naming the motor and the value, raised once for that value. The value
        is left as found, never marked again, and the arm holds its last goal."""
        if not self.marks:
            return
        bus = self.inner.bus._connected_bus
        ids = {name: self.inner.bus.motors[name].id for name in self.marks}
        try:
            values = {name: int(value) for name, value in
                      zip(ids, bus.sync_read_register(list(ids.values()), "torque_limit", retries=2), strict=True)}
        except Exception as error:
            raise FollowerLost(f"Reading the motors' restart marks failed: {error}") from error
        if lost := [name for name, value in values.items() if value == self.maxima[name] != self.marks[name]]:
            raise FollowerLost(POWER_LOST.format(motors=", ".join(lost)))
        for name, value in values.items():
            if value == self.marks[name]:
                self.changed.pop(name, None)
        if new := {name: value for name, value in values.items()
                   if value != self.marks[name] and self.changed.get(name) != value}:
            self.changed |= new
            listed = ", ".join(f"{name} reads {value}" for name, value in new.items())
            raise TorqueLimitChanged(TORQUE_LIMIT_CHANGED.format(listed=listed))

    # --- Faults ----------------------------------------------------------------------------------

    def fault(self, error: BaseException, policy, hold: bool = True) -> tuple[str | None, str]:
        """Hold the arm for the fault `error`, unless the follower may have lost power or `hold` is
        False (already in Fault, where `hold_in_fault` holds every tick). Returns the device to blame,
        if any, and the message describing the error, kept in `read_errors` too. When the hold finds
        a restart or a changed torque_limit, the message says so as well."""
        device = self.blame(error)
        self.lost = self.lost or self.is_lost(error)
        found = None
        if hold and not self.lost:
            try:
                self.hold_in_fault(policy)
            except TorqueLimitChanged as changed:  # reported with this fault; the arm has held
                found = str(changed)
            if self.lost:
                found = self.read_errors["follower"]
        message = self.describe(error, device)
        if device is not None:
            self.read_errors[device] = message
        if found is not None:
            message = f"{message} {found}"
        return device, message

    def hold_in_fault(self, policy) -> None:
        """Hold the arm, and keep reading it so the status shows when the follower answers again.

        Every hold in Fault goes through here. The arm and its motors' restart marks are read before the hold,
        so a power cut or a restart is caught before it resends the old goal. A failure that doesn't
        mean the follower is lost, such as a motor alarm, is noted and the arm holds all the same.
        After the follower was lost, it's read and marked again first, then its goal is set to its measured
        pose, and only then is its torque turned on: explicitly, rather than trusting the goal write
        to turn it on. A fault always closes the attempt, so no camera is required here: one that
        fails is only noted in `read_errors`, as in `observe` with no attempt open. Nothing is raised
        but a torque_limit found changed, once the arm has held, so the loop reports it as a fault of
        its own."""
        if self.lost:
            try:
                self.read_joints()
                self.mark()
                self.arm.send_action(dict(self.present))  # where the arm is now, never the old goal
                self.inner.bus.enable_torque(num_retry=2)
                self.lost = False
            except Exception as error:  # the next tick tries again
                self._note(error)
            return
        changed = None
        try:
            try:
                self.observe(attempt_open=False)
            except TorqueLimitChanged:
                raise
            except Exception as error:
                self._note(error)
                if not self.lost:  # `observe` may have stopped before it read the marks
                    self.check_marks()
        except TorqueLimitChanged as found:
            changed = found
        except Exception as error:
            self._note(error)
        if not self.lost:
            try:
                policy.hold(self.arm)
            except Exception as error:  # the next tick tries again
                self._note(error)
        if changed is not None:
            raise changed

    def _note(self, error: BaseException) -> None:
        """Keep what a failed read or hold found: whether the follower may have lost power, and why
        its device failed. An error no device explains is logged, once until a different one comes."""
        self.lost = self.lost or self.is_lost(error)
        if device := self.blame(error):
            self.read_errors[device] = self.describe(error, device)
        elif (text := self.describe(error, None)) != self.unblamed:
            self.unblamed = text
            logging.getLogger(__name__).error("Holding the arm in Fault hit an error: %s", text, exc_info=error)

    def is_lost(self, error: BaseException) -> bool:
        """Whether `error` means the follower may have lost power: it stopped answering, or a motor
        restarted. A motor alarm or a failed policy doesn't: the arm still holds its last goal."""
        if isinstance(error, Failure):
            return isinstance(error, FollowerLost)
        return self.blame(error) == "follower"

    def blame(self, error: BaseException) -> str | None:
        """The device whose code raised `error`, innermost first: a camera, or the follower (its bus,
        LeRobot's robot or the wrapper around it). LeRobot's robot refuses a read when any camera
        reports itself disconnected: that error is the camera's."""
        from lerobot.utils.errors import DeviceNotConnectedError

        if isinstance(error, Failure):
            return error.device
        owners = {id(owner): "follower" for owner in (self.inner.bus, self.inner, self.robot)}
        owners |= {id(camera): name for name, camera in self.cameras().items()}
        device, raiser = None, None
        for frame, _ in traceback.walk_tb(error.__traceback__):
            if (owner := owners.get(id(frame.f_locals.get("self")))) is not None:
                device, raiser = owner, frame.f_locals.get("self")
        if raiser is self.inner and isinstance(error, DeviceNotConnectedError):
            device = next((name for name, camera in self.cameras().items() if not camera.is_connected), device)
        return device

    def describe(self, error: BaseException, device: str | None) -> str:
        if isinstance(error, Failure):
            return str(error)
        if device == "follower":
            return (f"The follower stopped answering: {error}. Check its power and USB cable; "
                    "if it was unplugged, restart so101 serve.")
        if device is not None:
            return (f"The {device} camera stopped sending frames: {error}. Check its USB cable; "
                    "if it was unplugged, restart so101 serve.")
        return f"{type(error).__name__}: {error}"
