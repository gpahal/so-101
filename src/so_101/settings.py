"""Load `so101.toml`, resolve each arm's current serial port, and save ports and the rest pose back into it."""

import hashlib
import json
import math
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from serial.tools import list_ports

ArmName = Literal["follower", "leader"]
ARMS: tuple[ArmName, ...] = ("follower", "leader")

# LeRobot's CLI flag prefix for each arm.
LEROBOT_PREFIX = {"follower": "robot", "leader": "teleop"}

# A training job's timeout, from its expected time plus start-up. Generous, because hitting it loses work.
JOB_TIMEOUT_MARGIN = 2.5
JOB_STARTUP_S = 600
# A calibrated range under this many encoder steps (4096 per turn) means the joint wasn't moved
# through its range during calibration, and ends up nearly locked.
MIN_JOINT_RANGE_STEPS = 500


def config_path() -> Path:
    if env := os.environ.get("SO101_CONFIG"):
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "so101.toml"


@dataclass(frozen=True)
class Arm:
    name: ArmName
    id: str
    port: str
    serial_number: str
    calibration_dir: Path

    @property
    def calibration_file(self) -> Path:
        return self.calibration_dir / f"{self.id}.json"

    def calibration_fingerprint(self) -> str | None:
        """The first FINGERPRINT_DIGITS hex digits of the calibration file's SHA-256, or None without a file.

        `pose --save-rest` saves the follower's beside the rest pose, which is in raw motor steps:
        recalibrating moves where each step count puts a joint, so the saved pose no longer folds the arm.
        """
        if not self.calibration_file.is_file():
            return None
        return hashlib.sha256(self.calibration_file.read_bytes()).hexdigest()[:FINGERPRINT_DIGITS]

    def narrow_joints(self) -> list[str]:
        """Joints whose calibrated range is under MIN_JOINT_RANGE_STEPS encoder steps."""
        if not self.calibration_file.is_file():
            return []
        calibration = json.loads(self.calibration_file.read_text())
        return [joint for joint, c in calibration.items() if c["range_max"] - c["range_min"] < MIN_JOINT_RANGE_STEPS]

    def resolve_port(self) -> str:
        """Return the saved port if present, else find the board by its USB serial number."""
        if self.port and Path(self.port).exists():
            return self.port
        if self.serial_number:
            for p in list_ports.comports():
                if p.serial_number == self.serial_number:
                    return p.device
        raise PortNotFoundError(self)

    def lerobot_args(self) -> list[str]:
        p = LEROBOT_PREFIX[self.name]
        return [
            f"--{p}.type=so101_{self.name}",
            f"--{p}.port={self.resolve_port()}",
            f"--{p}.id={self.id}",
            f"--{p}.calibration_dir={self.calibration_dir}",
        ]

    def lerobot_device(self, port: str, use_degrees: bool = True, max_relative_target: float | None = None):
        """A new, unconnected LeRobot SO101Follower or SO101Leader for this arm on `port`. `use_degrees`
        picks LeRobot's units (its default, degrees, or -100 to 100 of each joint's range without it), and
        `max_relative_target` caps the follower's step per goal (None, the default, for no cap)."""
        if self.name == "follower":
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

            return SO101Follower(SO101FollowerConfig(
                port=port, id=self.id, calibration_dir=self.calibration_dir, use_degrees=use_degrees,
                max_relative_target=max_relative_target,
            ))
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        return SO101Leader(SO101LeaderConfig(
            port=port, id=self.id, calibration_dir=self.calibration_dir, use_degrees=use_degrees,
        ))


class ConfigError(RuntimeError):
    pass


def step_cap(value, where: str) -> float:
    """`[teleop] max_relative_target`, or a forwarded `--robot.max_relative_target` (`where` names
    which), refused unless it's a finite number, 0 or more. LeRobot clamps
    each goal to within the cap of where the arm is, so a negative cap moved every joint by it on
    every goal, holds included: an arm at rest crept 10 degrees a tick."""
    try:
        cap = float(value)
    except (TypeError, ValueError):
        cap = math.nan
    if isinstance(value, bool) or not math.isfinite(cap) or cap < 0:
        raise ConfigError(f"{where} is {value!r}: set it to a number of degrees, 0 or more, such as 10 "
                          "(0 turns the limit off).")
    return cap


class PortNotFoundError(RuntimeError):
    def __init__(self, arm: Arm) -> None:
        super().__init__(
            f"{arm.name} board not found at {arm.port or '(no port set)'}"
            + (f" or by serial number {arm.serial_number}" if arm.serial_number else "")
            + f". Plug it in, or run `uv run so101 find-port {arm.name}` to detect it again."
        )


@dataclass(frozen=True)
class Teleop:
    fps: int
    max_relative_target: float


@dataclass(frozen=True)
class Camera:
    name: str
    index: int
    width: int
    height: int
    fps: int
    rotation: int = 0
    usb_id: str = ""
    controls: dict[str, int] = field(default_factory=dict)

    def lerobot_spec(self, width: int | None = None, height: int | None = None) -> str:
        """One entry of LeRobot's `--robot.cameras` dict, named after this camera."""
        return (
            f"{self.name}: {{type: opencv, index_or_path: {self.index}, width: {width or self.width}, "
            f"height: {height or self.height}, fps: {self.fps}, rotation: {self.rotation}}}"
        )


def cameras_arg(specs: list[str]) -> str:
    return "--robot.cameras={" + ", ".join(specs) + "}"


@dataclass(frozen=True)
class Run:
    duration_s: float
    episodes: int
    device: str


@dataclass(frozen=True)
class Train:
    """One policy's training recipe: a `[train.<policy>]` section of so101.toml."""

    job_target: str
    seconds_per_step: dict[str, float]  # measured speed on each Hugging Face Jobs flavor
    steps_per_demo: int
    batch_size: int
    num_workers: int
    checkpoints: int

    def steps(self, demos: int) -> int:
        return self.steps_per_demo * demos

    def save_freq(self, steps: int) -> int:
        return max(steps // self.checkpoints, 1)

    def expected_hours(self, steps: int, target: str) -> float:
        return (JOB_STARTUP_S + steps * self.seconds_per_step[target]) / 3600

    def job_timeout(self, steps: int, target: str) -> str:
        return f"{math.ceil(self.expected_hours(steps, target) * JOB_TIMEOUT_MARGIN)}h"


@dataclass(frozen=True)
class Remote:
    """A `[remote.<policy>]` section of so101.toml: the job that serves a policy to this Mac."""

    job_target: str
    job_timeout: str
    # pi0.5 only: give the model its unused action columns as training did (`remote.PI05_SERVER`).
    pin_unused_actions: bool


# Raw encoder steps per turn of an STS3215 motor: positions run 0-4095.
MOTOR_STEPS = 4096
# Hex digits of a calibration file's SHA-256 kept as `rest_pose_calibration`: enough to tell two
# calibrations apart, short enough to read in so101.toml.
FINGERPRINT_DIGITS = 12


@dataclass(frozen=True)
class ControlSetting:
    """What `load()` uses when so101.toml leaves a `[control]` setting out, and what `serve`'s
    preflight (`check.check_control`) allows for it."""

    default: float | str
    minimum: float | None = None  # the lowest a number may be
    inclusive: bool = False  # whether `minimum` itself is allowed
    why: str = ""  # why a number can't be lower
    choices: tuple[str, ...] = ()  # the strings it may be


# Every `[control]` setting but the rest pose. `check_control` also checks `port` is a port number.
CONTROL_SETTINGS = {
    "rewind_s": ControlSetting(2.5, minimum=0, why="each Rewind has to go back some time"),
    "attempt_limit_s": ControlSetting(45, minimum=0, why="at 0 s every attempt would end as it starts"),
    "human_limit_s": ControlSetting(180, minimum=0, why="at 0 s every Teleop would end as it starts"),
    "teleop_from_rest": ControlSetting("start-pose", choices=("start-pose", "rest")),
    "blame_s": ControlSetting(2.5, minimum=0, inclusive=True, why="0 marks no frames as suspect"),
    "on_link_loss": ControlSetting("pause", choices=("pause", "continue")),
    "link_timeout_s": ControlSetting(
        3, minimum=1, inclusive=True,
        why="the page polls every 0.25 s, so under 1 s a few slow polls would count as a lost link",
    ),
    "bind": ControlSetting("tailscale", choices=("tailscale", "127.0.0.1")),
    "port": ControlSetting(8101),
    "preview_fps": ControlSetting(5, minimum=0, inclusive=True, why="0 turns the camera previews off"),
}


@dataclass(frozen=True)
class Control:
    """The `[control]` section of so101.toml: settings for `so101 serve`, the control center.

    `load()` keeps each value as the file writes it, without checking it, so a bad value never stops
    the commands that fix it, such as `pose --save-rest`. `serve`'s preflight (`check.check_control`)
    checks every value before the control center starts. Only `blame_s` is read elsewhere: by `runs`,
    `build --with` and `review --runs`, which check it as they read runs (`analysis.check_blame_s`)."""

    # Each motor's raw position in encoder steps (0-4095), independent of calibration and units, so it
    # stays right for every task. None until `so101 pose --save-rest` saves one.
    rest_pose: dict[str, int] | None
    # The first FINGERPRINT_DIGITS hex digits of the follower calibration file's SHA-256 when the rest
    # pose was saved, or None when so101.toml has none.
    rest_pose_calibration: str | None
    rewind_s: float
    attempt_limit_s: float
    human_limit_s: float
    teleop_from_rest: str
    blame_s: float
    on_link_loss: str
    link_timeout_s: float
    bind: str
    port: int
    preview_fps: float


@dataclass(frozen=True)
class Settings:
    follower: Arm
    leader: Arm
    teleop: Teleop
    cameras: dict[str, Camera]
    run: Run
    train: dict[str, Train]
    remote: dict[str, Remote]
    control: Control
    data_dir: Path
    path: Path

    @property
    def outputs_dir(self) -> Path:
        return self.path.parent / "outputs"

    def arm(self, name: ArmName) -> Arm:
        return self.follower if name == "follower" else self.leader

    def follower_args(self) -> list[str]:
        """The follower's LeRobot arguments, with `[teleop]`'s per-step cap when it's set."""
        cap = self.teleop.max_relative_target
        return [*self.follower.lerobot_args(), *([f"--robot.max_relative_target={cap}"] if cap else [])]

    def camera(self, name: str) -> Camera:
        if name not in self.cameras:
            raise ConfigError(f"No camera named {name!r} in {self.path}. Cameras: {', '.join(self.cameras) or 'none'}.")
        return self.cameras[name]


def _control(section: dict) -> Control:
    return Control(
        rest_pose=section.get("rest_pose"),
        rest_pose_calibration=section.get("rest_pose_calibration"),
        **{key: section.get(key, setting.default) for key, setting in CONTROL_SETTINGS.items()},
    )


def _remote(policy: str, section: dict, path: Path) -> Remote:
    pin = section.get("pin_unused_actions", True)
    if not isinstance(pin, bool):
        raise ConfigError(f"[remote.{policy}] pin_unused_actions in {path} is {pin!r}: set it to true or false.")
    return Remote(job_target=section["job_target"], job_timeout=section["job_timeout"], pin_unused_actions=pin)


def load() -> Settings:
    path = config_path()
    try:
        return _parse(path)
    except (KeyError, ValueError, TypeError, tomllib.TOMLDecodeError) as e:
        problem = f"the setting {e.args[0]!r} is missing" if isinstance(e, KeyError) else e
        raise ConfigError(f"{path}: {problem}") from e


def _parse(path: Path) -> Settings:
    data = tomllib.loads(path.read_text())
    calibration_root = path.parent / data.get("calibration_dir", "calibration")

    def arm(name: ArmName) -> Arm:
        section = data[name]
        return Arm(
            name=name,
            id=section["id"],
            port=section.get("port", ""),
            serial_number=section.get("serial_number", ""),
            calibration_dir=calibration_root / name,
        )

    def camera(name: str, section: dict) -> Camera:
        index = section.get("index", 0)
        if type(index) is not int:  # not int(index), which takes 1.5 and true as camera 1
            raise ValueError(f"camera {name}'s index is {index!r}: set it to a whole number.")
        return Camera(
            name=name,
            index=index,
            width=section.get("width", 640),
            height=section.get("height", 480),
            fps=section.get("fps", 30),
            rotation=section.get("rotation", 0),
            usb_id=section.get("usb_id", ""),
            controls=section.get("controls", {}),
        )

    def recipe(policy: str, section: dict) -> Train:
        checkpoints = section["checkpoints"]
        if type(checkpoints) is not int or checkpoints < 1:
            raise ValueError(f"[train.{policy}] checkpoints is {checkpoints!r}: set it to a whole number, 1 or more.")
        return Train(
            job_target=section["job_target"],
            seconds_per_step=section["seconds_per_step"],
            steps_per_demo=section["steps_per_demo"],
            batch_size=section["batch_size"],
            num_workers=section["num_workers"],
            checkpoints=checkpoints,
        )

    teleop = data.get("teleop", {})
    run = data.get("run", {})
    train = data.get("train", {})
    data_dir = Path(os.environ.get("SO101_DATA_DIR", data.get("data", {}).get("dir", "data"))).expanduser()
    return Settings(
        follower=arm("follower"),
        leader=arm("leader"),
        teleop=Teleop(fps=teleop.get("fps", 60),
                      max_relative_target=step_cap(teleop.get("max_relative_target", 0), f"[teleop] max_relative_target in {path}")),
        cameras={name: camera(name, section) for name, section in data.get("cameras", {}).items()},
        run=Run(
            duration_s=run.get("duration_s", 30),
            episodes=run.get("episodes", 1),
            device=run.get("device", "mps"),
        ),
        train={policy: recipe(policy, section) for policy, section in train.items()},
        remote={policy: _remote(policy, section, path) for policy, section in data.get("remote", {}).items()},
        control=_control(data.get("control", {})),
        data_dir=data_dir if data_dir.is_absolute() else path.parent / data_dir,
        path=path,
    )


def replace_file(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file, so a kill leaves the old file or the new one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def save_port(name: ArmName, port: str, serial_number: str) -> bool:
    """Set `port` and `serial_number` in the arm's section, keeping comments and layout. Returns
    whether the file changed: it doesn't when both already hold these values."""
    path = config_path()
    old_text = path.read_text()
    lines = _set_in_section(old_text.splitlines(keepends=True), name, "port", f'port = "{port}"\n')
    lines = _set_in_section(lines, name, "serial_number", f'serial_number = "{serial_number}"\n', after="port")
    new_text = "".join(lines)
    if new_text == old_text:
        return False
    replace_file(path, new_text)
    return True


def _set_in_section(lines: list[str], name: str, key: str, line: str, after: str | None = None) -> list[str]:
    """`lines` with `key = ...` under the section `[name]` set to `line`.

    Replaces an existing `key = ...` line in the section, or a commented-out `# key = ...` placeholder,
    else adds the line right after the `after = ...` line, else right after the section's header,
    adding the section at the end of the file when there is none. A dotted header such as
    `[cameras.top]` is a section of its own."""
    lines = list(lines)
    section = None
    header = existing = placeholder = anchor = None
    for i, current in enumerate(lines):
        if m := re.match(r"\s*\[([^\[\]]+)\]\s*(#.*)?$", current):
            section = m.group(1).strip()
            if section == name:
                header = i
        elif section == name:
            if existing is None and re.match(rf"\s*{key}\s*=", current):
                existing = i
            elif placeholder is None and re.match(rf"\s*#\s*{key}\s*=", current):
                placeholder = i
            elif after is not None and anchor is None and re.match(rf"\s*{after}\s*=", current):
                anchor = i
    if existing is not None:
        lines[existing] = line
    elif placeholder is not None:
        lines[placeholder] = line
    elif anchor is not None:
        lines.insert(anchor + 1, line)
    elif header is not None:
        lines.insert(header + 1, line)
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ["\n", f"[{name}]\n", line]
    return lines


def save_rest_pose(pose: dict[str, int], path: Path | None = None, *, calibration: str) -> Path:
    """Write `rest_pose` under `[control]`, and `rest_pose_calibration` = `calibration`, the follower
    calibration file's fingerprint (`Arm.calibration_fingerprint`), keeping the rest of the file as it is.

    Each line replaces an existing one in the section, or a commented-out placeholder, else goes right
    after the `[control]` header (`rest_pose_calibration` right after `rest_pose`), adding the section
    at the end of the file when there is none. Checks every position is from 0 to 4095, and that the
    result parses to the same values, before writing.
    """
    path = path or config_path()
    # The same rule `serve`'s preflight applies, checked first: a bad value written here would stop `serve`.
    for motor, steps in pose.items():
        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps < MOTOR_STEPS:
            raise ConfigError(
                f"The follower's {motor} read {steps!r}, not a raw position from 0 to {MOTOR_STEPS - 1}, so rest_pose "
                f"wasn't saved and {path} is unchanged. Check the arm's power and cable, then run `uv run so101 scan`."
            )
    pose_line = "rest_pose = { " + ", ".join(f"{motor} = {int(steps)}" for motor, steps in pose.items()) + " }\n"
    lines = _set_in_section(path.read_text().splitlines(keepends=True), "control", "rest_pose", pose_line)
    lines = _set_in_section(lines, "control", "rest_pose_calibration", f'rest_pose_calibration = "{calibration}"\n',
                            after="rest_pose")
    new_text = "".join(lines)
    try:
        saved = tomllib.loads(new_text).get("control", {})
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Saving rest_pose would break {path} ({e}), so it was left unchanged.") from e
    wanted = {"rest_pose": {motor: int(steps) for motor, steps in pose.items()}, "rest_pose_calibration": calibration}
    got = {key: saved.get(key) for key in wanted}
    if got != wanted:
        raise ConfigError(f"Saving rest_pose to {path} didn't round-trip (read back {got!r}), so it was left unchanged.")
    replace_file(path, new_text)
    return path
