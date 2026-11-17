"""The gripper and stuck-rule thresholds: `grasp.toml` beside this file, or the file `[perception]
thresholds` names.

`load` reads them into frozen dataclasses and checks every value, so a typo stops a command before the
arm moves or a demo is cut, rather than changing what the perception sees. The file's comments say what
each number means and where it came from. `[gripper]` and `[rules]` each carry a `version`.
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from so_101.settings import ConfigError

DEFAULT = Path(__file__).with_name("grasp.toml")
RULES = ("regrasp_loop", "hover", "chatter", "frozen", "no_carry", "not_released")


class ThresholdsError(ConfigError):
    pass


@dataclass(frozen=True)
class Gripper:
    version: int
    closed_below: float
    carry_open_above: float
    carry_elbow_deg: float
    hold_band: tuple[float, float]
    hold_steady: float
    hold_s: float
    release_above: float


@dataclass(frozen=True)
class Rules:
    version: int
    speed_window_s: float
    regrasps: int
    hover_open_above: float
    hover_speed_below: float
    hover_from_start_deg: float
    hover_s: float
    chatter_crossings: int
    chatter_within_s: float
    frozen_speed_below: float
    frozen_s: float
    no_carry_s: float
    not_released_s: float


@dataclass(frozen=True)
class Thresholds:
    path: Path
    gripper: Gripper
    rules: Rules


def number(table: dict, key: str, where: str, *, minimum: float = 0.0, strict: bool = True,
           error: type[Exception] = ThresholdsError) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{where}.{key} is {value!r}: it must be a number.")
    if not math.isfinite(value):
        raise error(f"{where}.{key} is {value!r}: it must be a finite number.")
    if value < minimum or (strict and value == minimum):
        raise error(f"{where}.{key} is {value!r}: it must be {'over' if strict else 'at least'} {minimum:g}.")
    return float(value)


def whole(table: dict, key: str, where: str, minimum: int = 1, error: type[Exception] = ThresholdsError) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise error(f"{where}.{key} is {value!r}: it must be a whole number, {minimum} or more.")
    return value


def section(data: dict, *keys: str, error: type[Exception] = ThresholdsError) -> dict:
    table = data
    for key in keys:
        table = table.get(key)
        if not isinstance(table, dict):
            raise error(f"[{'.'.join(keys)}] is missing.")
    return table


def version(table: dict, where: str, error: type[Exception] = ThresholdsError) -> int:
    value = table.get("version")
    if type(value) is not int:
        raise error(f"{where}version is {value!r}: it must be a whole number.")
    return value


def load(path: Path | None = None) -> Thresholds:
    """Read and check the thresholds at `path`, or the default ones."""
    path = Path(path) if path else DEFAULT
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ThresholdsError(f"Can't read the grasp thresholds {path}: {error}") from error
    try:
        return _parse(data, path)
    except ThresholdsError as error:
        raise ThresholdsError(f"{path}: {error}") from error


def _parse(data: dict, path: Path) -> Thresholds:
    g = section(data, "gripper")
    band = g.get("hold_band")
    if not (isinstance(band, list) and len(band) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in band)
            and 0 < band[0] < band[1]):
        raise ThresholdsError(f"gripper.hold_band is {band!r}: it must be [low, high] with 0 < low < high.")
    gripper = Gripper(
        version=version(g, "gripper."),
        closed_below=number(g, "closed_below", "gripper"),
        carry_open_above=number(g, "carry_open_above", "gripper"),
        carry_elbow_deg=number(g, "carry_elbow_deg", "gripper"),
        hold_band=(float(band[0]), float(band[1])),
        hold_steady=number(g, "hold_steady", "gripper"),
        hold_s=number(g, "hold_s", "gripper"),
        release_above=number(g, "release_above", "gripper"),
    )
    if not gripper.carry_open_above < gripper.hold_band[0] <= gripper.hold_band[1] < gripper.closed_below:
        raise ThresholdsError("gripper: need carry_open_above < hold_band low < hold_band high < closed_below, "
                              "or a held block would count as open or a miss as held.")
    top = section(data, "rules")
    r = {name: section(data, "rules", name) for name in RULES}
    rules = Rules(
        version=version(top, "rules."),
        speed_window_s=number(top, "speed_window_s", "rules"),
        regrasps=whole(r["regrasp_loop"], "regrasps", "rules.regrasp_loop"),
        hover_open_above=number(r["hover"], "open_above", "rules.hover"),
        hover_speed_below=number(r["hover"], "speed_below", "rules.hover"),
        hover_from_start_deg=number(r["hover"], "from_start_deg", "rules.hover"),
        hover_s=number(r["hover"], "seconds", "rules.hover"),
        chatter_crossings=whole(r["chatter"], "crossings", "rules.chatter", minimum=2),
        chatter_within_s=number(r["chatter"], "within_s", "rules.chatter"),
        frozen_speed_below=number(r["frozen"], "speed_below", "rules.frozen"),
        frozen_s=number(r["frozen"], "seconds", "rules.frozen"),
        no_carry_s=number(r["no_carry"], "seconds", "rules.no_carry"),
        not_released_s=number(r["not_released"], "seconds", "rules.not_released"),
    )
    return Thresholds(path=path, gripper=gripper, rules=rules)
