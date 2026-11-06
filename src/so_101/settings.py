"""Load `so101.toml` and resolve each arm's current serial port."""

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

# LeRobot's device type and CLI flag prefix for each arm.
LEROBOT_TYPE = {"follower": "so101_follower", "leader": "so101_leader"}
LEROBOT_PREFIX = {"follower": "robot", "leader": "teleop"}

# A training job's timeout, from its expected time plus start-up. Generous, because hitting it loses work.
JOB_TIMEOUT_MARGIN = 2.5
JOB_STARTUP_S = 600


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
    def lerobot_type(self) -> str:
        return LEROBOT_TYPE[self.name]

    @property
    def prefix(self) -> str:
        return LEROBOT_PREFIX[self.name]

    @property
    def calibration_file(self) -> Path:
        return self.calibration_dir / f"{self.id}.json"

    def narrow_joints(self, min_ticks: int = 500) -> list[str]:
        """Joints whose calibrated range is under `min_ticks` encoder steps (4096 per turn).

        A joint that wasn't moved through its range during calibration ends up nearly locked.
        """
        if not self.calibration_file.is_file():
            return []
        calibration = json.loads(self.calibration_file.read_text())
        return [joint for joint, c in calibration.items() if c["range_max"] - c["range_min"] < min_ticks]

    def resolve_port(self) -> str:
        """Return the saved port if present, else find the board by its USB serial number."""
        if Path(self.port).exists():
            return self.port
        if self.serial_number:
            for p in list_ports.comports():
                if p.serial_number == self.serial_number:
                    return p.device
        raise PortNotFoundError(self)

    def lerobot_args(self) -> list[str]:
        p = self.prefix
        return [
            f"--{p}.type={self.lerobot_type}",
            f"--{p}.port={self.resolve_port()}",
            f"--{p}.id={self.id}",
            f"--{p}.calibration_dir={self.calibration_dir}",
        ]


class ConfigError(RuntimeError):
    pass


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
    # True when SO101_CAMERA_<NAME>_INDEX set `index`, which then wins over the USB id lookup.
    index_from_env: bool = False

    def lerobot_spec(self, key: str | None = None, width: int | None = None, height: int | None = None) -> str:
        """One entry of LeRobot's `--robot.cameras` dict, named `key` (default: this camera's name)."""
        return (
            f"{key or self.name}: {{type: opencv, index_or_path: {self.index}, width: {width or self.width}, "
            f"height: {height or self.height}, fps: {self.fps}, rotation: {self.rotation}}}"
        )


def cameras_arg(specs: list[str]) -> str:
    return "--robot.cameras={" + ", ".join(specs) + "}"


@dataclass(frozen=True)
class Run:
    duration_s: float
    episodes: int
    device: str
    max_relative_target: float


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


@dataclass(frozen=True)
class Settings:
    follower: Arm
    leader: Arm
    teleop: Teleop
    cameras: dict[str, Camera]
    run: Run
    train: dict[str, Train]
    remote: dict[str, Remote]
    data_dir: Path
    path: Path

    @property
    def outputs_dir(self) -> Path:
        return self.path.parent / "outputs"

    def arm(self, name: ArmName) -> Arm:
        return self.follower if name == "follower" else self.leader

    def camera(self, name: str) -> Camera:
        if name not in self.cameras:
            raise ConfigError(f"No camera named {name!r} in {self.path}. Cameras: {', '.join(self.cameras) or 'none'}.")
        return self.cameras[name]


def load() -> Settings:
    path = config_path()
    data = tomllib.loads(path.read_text())
    calibration_root = path.parent / data.get("calibration_dir", "calibration")

    def arm(name: ArmName) -> Arm:
        section = data[name]
        env = f"SO101_{name.upper()}_"
        return Arm(
            name=name,
            id=os.environ.get(env + "ID", section["id"]),
            port=os.environ.get(env + "PORT", section.get("port", "")),
            serial_number=section.get("serial_number", ""),
            calibration_dir=calibration_root / name,
        )

    def camera(name: str, section: dict) -> Camera:
        env_index = os.environ.get(f"SO101_CAMERA_{name.upper()}_INDEX")
        return Camera(
            name=name,
            index=int(env_index if env_index is not None else section["index"]),
            width=section.get("width", 640),
            height=section.get("height", 480),
            fps=section.get("fps", 30),
            rotation=section.get("rotation", 0),
            usb_id=section.get("usb_id", ""),
            controls=section.get("controls", {}),
            index_from_env=env_index is not None,
        )

    teleop = data.get("teleop", {})
    run = data.get("run", {})
    train = data.get("train", {})
    data_dir = Path(os.environ.get("SO101_DATA_DIR", data.get("data", {}).get("dir", "data"))).expanduser()
    return Settings(
        follower=arm("follower"),
        leader=arm("leader"),
        teleop=Teleop(fps=teleop.get("fps", 60), max_relative_target=teleop.get("max_relative_target", 0)),
        cameras={name: camera(name, section) for name, section in data.get("cameras", {}).items()},
        run=Run(
            duration_s=run.get("duration_s", 30),
            episodes=run.get("episodes", 1),
            device=run.get("device", "mps"),
            max_relative_target=run.get("max_relative_target", 5),
        ),
        train={
            policy: Train(
                job_target=section["job_target"],
                seconds_per_step=section["seconds_per_step"],
                steps_per_demo=section["steps_per_demo"],
                batch_size=section["batch_size"],
                num_workers=section["num_workers"],
                checkpoints=section["checkpoints"],
            )
            for policy, section in train.items()
        },
        remote={
            policy: Remote(job_target=section["job_target"], job_timeout=section["job_timeout"])
            for policy, section in data.get("remote", {}).items()
        },
        data_dir=data_dir if data_dir.is_absolute() else path.parent / data_dir,
        path=path,
    )


def save_port(name: ArmName, port: str, serial_number: str) -> Path:
    """Rewrite `port` and `serial_number` in the arm's section, keeping comments and layout."""
    path = config_path()
    lines = path.read_text().splitlines(keepends=True)
    section = None
    for i, line in enumerate(lines):
        if m := re.match(r"\s*\[(\w+)\]", line):
            section = m.group(1)
        elif section == name:
            if re.match(r"\s*port\s*=", line):
                lines[i] = f'port = "{port}"\n'
            elif re.match(r"\s*serial_number\s*=", line):
                lines[i] = f'serial_number = "{serial_number}"\n'
    path.write_text("".join(lines))
    return path
