"""The supervisor's program: `program.toml` beside this file, or the file `[supervisor] program` names.

It holds the supervisor's own settings: when it checks, its judges, its responses and its budget. The
gripper and stuck-rule thresholds every run's perception reads are `grasp.thresholds`, not here.

`load` reads it into frozen dataclasses and checks every value, so a typo stops `serve` before the arm
moves rather than changing what the supervisor does. The file's comments say what each number means and
where it came from.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from so_101.grasp.thresholds import number, section, whole

DEFAULT = Path(__file__).with_name("program.toml")


class ProgramError(ValueError):
    pass


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
    after_close_s: float
    after_release_s: float
    watch: Watch
    judges: Judges
    miss_rewind_s: tuple[float, ...]


def _number(table: dict, key: str, where: str, *, minimum: float = 0.0, strict: bool = True) -> float:
    return number(table, key, where, minimum=minimum, strict=strict, error=ProgramError)


def _whole(table: dict, key: str, where: str, minimum: int = 1) -> int:
    return whole(table, key, where, minimum, error=ProgramError)


def _section(data: dict, *keys: str) -> dict:
    return section(data, *keys, error=ProgramError)


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
        version=version, path=path,
        after_close_s=_number(t, "after_close_s", "timing", strict=False),
        after_release_s=_number(t, "after_release_s", "timing", strict=False),
        watch=watch, judges=judges, miss_rewind_s=tuple(float(n) for n in rewinds),
    )
