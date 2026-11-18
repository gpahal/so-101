"""The supervisor's program: `program.toml` beside this file, or the file `[supervisor] program` names.

It holds the supervisor's own settings: when it checks, its judges, its responses and its budget. The
gripper and stuck-rule thresholds every run's perception reads are `grasp.thresholds`, not here.

`load` reads it into frozen dataclasses and checks every value, so a typo stops `serve` or `run` before the
arm moves rather than changing what the supervisor does. The file's comments say what each number means and
where it came from. A `ProgramError` is a `settings.ConfigError`.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from so_101.grasp.thresholds import number, section, whole
from so_101.settings import ConfigError

DEFAULT = Path(__file__).with_name("program.toml")
RING_S = 4.0  # the supervisor keeps this many seconds of frames, so the watch check's lookback is at most this
# How much further back than after_close_s + after_close_answer_s every miss playback must reach: the check
# is sent up to a tick after it's due, the playback starts a tick after the verdict, and the jaws close over
# several ticks before the detector sees it. With no margin, a 30 fps test played back to gripper 26
# against closed_below 25, one tick short of the close.
REACH_MARGIN_S = 0.5


class ProgramError(ConfigError):
    pass


@dataclass(frozen=True)
class Watch:
    lookback_s: float


@dataclass(frozen=True)
class Judges:
    decision: str
    chat: str
    escalation: str
    sure_ok_below: float
    sure_wrong_above: float
    success_decision_below: float  # after a release, done needs the decision model's p_wrong under this
    success_chat_below: float  # and the chat model's under this
    answer_within_s: float
    max_cost_usd: float


@dataclass(frozen=True)
class Program:
    version: int
    path: Path
    after_close_s: float
    after_release_s: float
    after_close_answer_s: float
    hold_deadline_s: float
    home_within_s: float
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
    from so_101.vision.models import MODELS

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
    watch = Watch(lookback_s=_number(w, "lookback_s", "watch"))
    if watch.lookback_s > RING_S:
        raise ProgramError(f"watch.lookback_s is {watch.lookback_s:g}: the supervisor keeps only {RING_S:g} s of frames.")
    roles = ("decision", "chat", "escalation")
    for key in roles:
        if j.get(key) not in models:
            raise ProgramError(f"judges.{key} is {j.get(key)!r}: use a model key from `so101-failbench models`.")
    for n, key in enumerate(roles):
        if (other := next((o for o in roles[:n] if j[o] == j[key]), None)) is not None:
            raise ProgramError(f"judges.{key} is {j[key]!r}, the same model as judges.{other}: each role needs its "
                               "own model, or one model's answer would count twice.")
    judges = Judges(
        decision=j["decision"], chat=j["chat"], escalation=j["escalation"],
        sure_ok_below=_number(j, "sure_ok_below", "judges"),
        sure_wrong_above=_number(j, "sure_wrong_above", "judges"),
        success_decision_below=_number(j, "success_decision_below", "judges"),
        success_chat_below=_number(j, "success_chat_below", "judges"),
        answer_within_s=_number(j, "answer_within_s", "judges"),
        max_cost_usd=_number(j, "max_cost_usd", "judges", strict=False),
    )
    if not judges.sure_ok_below < judges.sure_wrong_above <= 1:
        raise ProgramError("judges: need sure_ok_below < sure_wrong_above <= 1.")
    if models[judges.escalation].provider != "openrouter":
        raise ProgramError(f"judges.escalation is {judges.escalation!r}: it must be a chat model through OpenRouter.")
    rewinds = _section(data, "responses").get("miss_rewind_s")
    if not (isinstance(rewinds, list)
            and all(type(n) in (int, float) and n > 0 and n != float("inf") for n in rewinds)):
        raise ProgramError(f"responses.miss_rewind_s is {rewinds!r}: it must be a list of seconds, each above 0.")
    version = data.get("version")
    if type(version) is not int:
        raise ProgramError(f"version is {version!r}: it must be a whole number.")
    program = Program(
        version=version, path=path,
        after_close_s=_number(t, "after_close_s", "timing", strict=False),
        after_release_s=_number(t, "after_release_s", "timing", strict=False),
        after_close_answer_s=_number(t, "after_close_answer_s", "timing"),
        hold_deadline_s=_number(t, "hold_deadline_s", "timing"),
        home_within_s=_number(t, "home_within_s", "timing"),
        watch=watch, judges=judges, miss_rewind_s=tuple(float(n) for n in rewinds),
    )
    reach = program.after_close_s + program.after_close_answer_s
    for n, seconds in enumerate(program.miss_rewind_s):
        if seconds < reach + REACH_MARGIN_S:
            raise ProgramError(
                f"responses.miss_rewind_s[{n}] is {seconds:g} s, under timing.after_close_s + timing.after_close_answer_s "
                f"({program.after_close_s:g} + {program.after_close_answer_s:g} s) + {REACH_MARGIN_S:g} s: a miss answered "
                "that late would be played back to jaws still closed, or past the close not at all.")
    return program
