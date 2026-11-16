"""The supervisor's program: `program.toml` beside this file, or the file `[supervisor] program` names.

`load` reads it into frozen dataclasses and checks every value, so a typo stops `serve` before the arm
moves rather than changing what the supervisor does. The file's comments say what each number means and
where it came from.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT = Path(__file__).with_name("program.toml")
RULES = ("regrasp_loop", "hover", "chatter", "frozen", "no_carry", "not_released")


class ProgramError(ValueError):
    pass


@dataclass(frozen=True)
class Gripper:
    closed_below: float
    carry_open_above: float
    carry_elbow_deg: float
    hold_band: tuple[float, float]
    hold_steady: float
    hold_s: float
    release_above: float


@dataclass(frozen=True)
class Rules:
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
class Watch:
    near: float
    period_s: float
    lookback_s: float
    max_per_attempt: int


@dataclass(frozen=True)
class Judges:
    decision: str
    chat: str
    escalation: str
    sure_ok_below: float
    sure_wrong_above: float
    success_p_wrong: float
    max_cost_usd: float


@dataclass(frozen=True)
class Program:
    version: int
    path: Path
    gripper: Gripper
    speed_window_s: float
    rules: Rules
    after_close_s: float
    after_release_s: float
    watch: Watch
    judges: Judges
    miss_rewind_s: tuple[float, ...]


def _number(table: dict, key: str, where: str, *, minimum: float = 0.0, strict: bool = True) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramError(f"{where}.{key} is {value!r}: it must be a number.")
    if value < minimum or (strict and value == minimum):
        raise ProgramError(f"{where}.{key} is {value!r}: it must be {'over' if strict else 'at least'} {minimum:g}.")
    return float(value)


def _whole(table: dict, key: str, where: str, minimum: int = 1) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProgramError(f"{where}.{key} is {value!r}: it must be a whole number, {minimum} or more.")
    return value


def _section(data: dict, *keys: str) -> dict:
    table = data
    for key in keys:
        table = table.get(key)
        if not isinstance(table, dict):
            raise ProgramError(f"[{'.'.join(keys)}] is missing.")
    return table


def load(path: Path | None = None) -> Program:
    """Read and check the program at `path`, or the default one."""
    from so_101.failbench.models import MODELS

    path = Path(path) if path else DEFAULT
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ProgramError(f"Can't read the supervisor's program {path}: {error}") from error
    try:
        return _parse(data, path, MODELS)
    except ProgramError as error:
        raise ProgramError(f"{path}: {error}") from error


def _parse(data: dict, path: Path, models: dict) -> Program:
    g = _section(data, "gripper")
    band = g.get("hold_band")
    if not (isinstance(band, list) and len(band) == 2 and all(isinstance(v, (int, float)) for v in band)
            and 0 < band[0] < band[1]):
        raise ProgramError(f"gripper.hold_band is {band!r}: it must be [low, high] with 0 < low < high.")
    gripper = Gripper(
        closed_below=_number(g, "closed_below", "gripper"),
        carry_open_above=_number(g, "carry_open_above", "gripper"),
        carry_elbow_deg=_number(g, "carry_elbow_deg", "gripper"),
        hold_band=(float(band[0]), float(band[1])),
        hold_steady=_number(g, "hold_steady", "gripper"),
        hold_s=_number(g, "hold_s", "gripper"),
        release_above=_number(g, "release_above", "gripper"),
    )
    if not gripper.carry_open_above < gripper.hold_band[0] <= gripper.hold_band[1] < gripper.closed_below:
        raise ProgramError("gripper: need carry_open_above < hold_band low < hold_band high < closed_below, "
                           "or a held block would count as open or a miss as held.")
    r = {name: _section(data, "rules", name) for name in RULES}
    rules = Rules(
        regrasps=_whole(r["regrasp_loop"], "regrasps", "rules.regrasp_loop"),
        hover_open_above=_number(r["hover"], "open_above", "rules.hover"),
        hover_speed_below=_number(r["hover"], "speed_below", "rules.hover"),
        hover_from_start_deg=_number(r["hover"], "from_start_deg", "rules.hover"),
        hover_s=_number(r["hover"], "seconds", "rules.hover"),
        chatter_crossings=_whole(r["chatter"], "crossings", "rules.chatter", minimum=2),
        chatter_within_s=_number(r["chatter"], "within_s", "rules.chatter"),
        frozen_speed_below=_number(r["frozen"], "speed_below", "rules.frozen"),
        frozen_s=_number(r["frozen"], "seconds", "rules.frozen"),
        no_carry_s=_number(r["no_carry"], "seconds", "rules.no_carry"),
        not_released_s=_number(r["not_released"], "seconds", "rules.not_released"),
    )
    t, w, j = _section(data, "timing"), _section(data, "watch"), _section(data, "judges")
    watch = Watch(
        near=_number(w, "near", "watch"),
        period_s=_number(w, "period_s", "watch"),
        lookback_s=_number(w, "lookback_s", "watch"),
        max_per_attempt=_whole(w, "max_per_attempt", "watch", minimum=0),
    )
    if watch.near > 1:
        raise ProgramError(f"watch.near is {watch.near:g}: a rule's progress runs from 0 to 1.")
    for key in ("decision", "chat", "escalation"):
        if j.get(key) not in models:
            raise ProgramError(f"judges.{key} is {j.get(key)!r}: use a model key from `so101-failbench models`.")
    judges = Judges(
        decision=j["decision"], chat=j["chat"], escalation=j["escalation"],
        sure_ok_below=_number(j, "sure_ok_below", "judges"),
        sure_wrong_above=_number(j, "sure_wrong_above", "judges"),
        success_p_wrong=_number(j, "success_p_wrong", "judges"),
        max_cost_usd=_number(j, "max_cost_usd", "judges", strict=False),
    )
    if not judges.sure_ok_below < judges.sure_wrong_above <= 1:
        raise ProgramError("judges: need sure_ok_below < sure_wrong_above <= 1.")
    if models[judges.escalation].provider != "openrouter":
        raise ProgramError(f"judges.escalation is {judges.escalation!r}: it must be a chat model through OpenRouter.")
    rewinds = _section(data, "responses").get("miss_rewind_s")
    if not (isinstance(rewinds, list) and all(type(n) in (int, float) and n > 0 for n in rewinds)):
        raise ProgramError(f"responses.miss_rewind_s is {rewinds!r}: it must be a list of seconds, each above 0.")
    version = data.get("version")
    if type(version) is not int:
        raise ProgramError(f"version is {version!r}: it must be a whole number.")
    return Program(
        version=version, path=path, gripper=gripper,
        speed_window_s=_number(_section(data, "speed"), "window_s", "speed"),
        rules=rules,
        after_close_s=_number(t, "after_close_s", "timing", strict=False),
        after_release_s=_number(t, "after_release_s", "timing", strict=False),
        watch=watch, judges=judges, miss_rewind_s=tuple(float(n) for n in rewinds),
    )
