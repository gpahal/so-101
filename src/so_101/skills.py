"""Skills: named parts of a task's demos, cut where the arm's readings show an event, trained into one model.

A task's `[<task>.skills]` in tasks.toml lists its skills in the order they run. Each skill spans every
kept demo from one event to another, plus or minus seconds:

    pick  = { prompt = "...", from = "start", to = "carry + 0.5" }
    place = { prompt = "...", from = "lift", to = "widest + 0.2", handover = "carry" }

The events (`events`) come from the supervisor's detector over the demo's joint readings, with the
supervisor's program for its thresholds, plus two found here:
- start, end: the kept demo's first and last frames;
- close: the gripper drops below the program's closed_below, on the close that led to the carry;
- lift: shoulder_lift has moved LIFT_DEG since that close;
- carry: the detector's carry, the gripper holding steady in its band or the elbow carrying;
- release: the gripper opens past its hold;
- widest: the gripper's widest opening within WIDEST_WITHIN_S after the release.

A skill's span is inclusive of its `to` frame. Every skill after the first names its `handover`: the
event that switches to it at runtime, one the detector sees live (LIVE). It must fall inside both its
own span and the previous skill's, on every demo (`spans`), so the model has seen every state on both
sides of the switch.

`so101 build --skills` writes each kept demo's skills as episodes, in order, each frame with its skill's
prompt as LeRobot's `task` (for policies that read text, such as pi0.5) and the skill as a one-hot
`observation.environment_state` (for ACT, which takes no text but reads an environment state).

`SkillPolicy` runs a skills model in `so101 serve`: a layer over the policy driver with the driver's
methods, like the supervisor's. It starts each attempt on the first skill and switches at each handover:
the new prompt (`set_task`), the new one-hot, and the policy's queue cleared, so no action of the last
skill reaches the arm after the switch.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EVENTS = ("start", "close", "lift", "carry", "release", "widest", "end")
LIVE = ("close", "carry", "release")  # what the detector sees as the policy drives: the possible handovers
LIFT_DEG = 4.0  # shoulder_lift travel after the close that counts as the block leaving the mat
WIDEST_WITHIN_S = 2.0  # the widest opening is looked for this long after the release
MIN_SPAN_S = 0.5  # a skill's shortest span in any demo
STATE_KEY = "observation.environment_state"
SHOULDER_LIFT, GRIPPER = "shoulder_lift.pos", "gripper.pos"
_POINT = re.compile(r"^\s*([a-z]+)\s*(?:([+-])\s*(\d+(?:\.\d+)?)\s*)?$")


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Point:
    """An event, plus or minus seconds."""

    event: str
    offset_s: float = 0.0

    @classmethod
    def parse(cls, text, where: str) -> "Point":
        m = _POINT.match(text) if isinstance(text, str) else None
        if m is None or m.group(1) not in EVENTS:
            raise SkillError(f"{where} is {text!r}: write an event, optionally plus or minus seconds, such as "
                             f"\"carry + 0.5\". Events: {', '.join(EVENTS)}.")
        offset = float(m.group(3) or 0) * (-1 if m.group(2) == "-" else 1)
        return cls(m.group(1), offset)

    def __str__(self) -> str:
        if not self.offset_s:
            return self.event
        return f"{self.event} {'+' if self.offset_s > 0 else '-'} {abs(self.offset_s):g}"


@dataclass(frozen=True)
class Skill:
    name: str
    prompt: str
    start: Point
    stop: Point
    handover: str | None  # the event that switches to this skill at runtime; None for the first

    def record(self) -> dict:
        return {"name": self.name, "prompt": self.prompt, "from": str(self.start), "to": str(self.stop),
                "handover": self.handover}


def parse(table, task: str) -> tuple[Skill, ...]:
    """`[<task>.skills]` from tasks.toml, in order. Raises SkillError on a bad entry."""
    if not isinstance(table, dict) or not table:
        raise SkillError(f"[{task}.skills] must list skills, such as pick = {{ prompt = ..., from = ..., to = ... }}.")
    skills = []
    for n, (name, entry) in enumerate(table.items()):
        where = f"[{task}.skills] {name}"
        if not isinstance(entry, dict):
            raise SkillError(f"{where} must be a table: {{ prompt = ..., from = ..., to = ... }}.")
        unknown = set(entry) - {"prompt", "from", "to", "handover"}
        if unknown:
            raise SkillError(f"{where} has unknown keys {sorted(unknown)}: use prompt, from, to and handover.")
        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise SkillError(f"{where} needs a prompt: the text a policy that reads text is given for this skill.")
        handover = entry.get("handover")
        if n == 0 and handover is not None:
            raise SkillError(f"{where} is the first skill, which every attempt starts with: it takes no handover.")
        if n > 0 and handover not in LIVE:
            raise SkillError(f"{where} needs a handover, the event that switches to it as the policy runs: one of "
                             f"{', '.join(LIVE)} (got {handover!r}).")
        skills.append(Skill(name, prompt.strip(), Point.parse(entry.get("from"), f"{where} from"),
                            Point.parse(entry.get("to"), f"{where} to"), handover))
    if len({s.prompt for s in skills}) != len(skills):
        raise SkillError(f"[{task}.skills]: each skill needs its own prompt, or a model can't tell them apart.")
    return tuple(skills)


def state_feature(skills: tuple[Skill, ...]) -> dict:
    """The one-hot `observation.environment_state` a skills dataset has, and a skills run adds."""
    return {"dtype": "float32", "shape": (len(skills),), "names": [f"skill.{s.name}" for s in skills]}


def one_hot(skills: tuple[Skill, ...], index: int) -> np.ndarray:
    value = np.zeros(len(skills), dtype=np.float32)
    value[index] = 1.0
    return value


# --- Cutting demos ---------------------------------------------------------------------------------


def events(state: np.ndarray, names: list[str], fps: float, program) -> dict[str, int]:
    """Each event's frame in one demo's joint readings (frames x joints, named `names`). Raises
    SkillError when the demo has no carry or release, or the block never leaves the mat."""
    from so_101.supervisor.detector import Detector

    detector = Detector(program)
    closes, found = [], {}
    for i, row in enumerate(state):
        for signal in detector.step(i / fps, dict(zip(names, map(float, row), strict=True)), True):
            if signal.kind == "close":
                closes.append(i)
            elif signal.kind in ("carry", "release") and signal.kind not in found:
                found[signal.kind] = i
    if "carry" not in found:
        raise SkillError(f"no carry: {len(closes)} close{'' if len(closes) == 1 else 's'}, and the block was never held")
    if "release" not in found:
        raise SkillError("no release: the block was never let go")
    close = max(c for c in closes if c <= found["carry"])
    lift = state[:, names.index(SHOULDER_LIFT)]
    moved = np.where(np.abs(lift[close:] - lift[close]) > LIFT_DEG)[0]
    if not len(moved):
        raise SkillError(f"no lift: shoulder_lift never moved {LIFT_DEG:g} deg after the close")
    release = found["release"]
    gripper = state[release: release + round(WIDEST_WITHIN_S * fps), names.index(GRIPPER)]
    return {"start": 0, "close": close, "lift": close + int(moved[0]), "carry": found["carry"],
            "release": release, "widest": release + int(np.argmax(gripper)), "end": len(state) - 1}


def frame_of(point: Point, at: dict[str, int], fps: float) -> int:
    return at[point.event] + round(point.offset_s * fps)


def spans(skills: tuple[Skill, ...], at: dict[str, int], fps: float, length: int) -> dict[str, tuple[int, int]]:
    """Each skill's frames [start, stop) in a demo of `length` frames with the events `at`. Raises
    SkillError when a span is too short or a handover falls outside the spans it joins."""
    out = {}
    for n, skill in enumerate(skills):
        start = min(max(frame_of(skill.start, at, fps), 0), length)
        stop = min(max(frame_of(skill.stop, at, fps) + 1, 0), length)
        if stop - start < MIN_SPAN_S * fps:
            raise SkillError(f"{skill.name} spans {(stop - start) / fps:.2f} s ({skill.start} to {skill.stop}), "
                             f"under {MIN_SPAN_S:g} s")
        if n:
            handover, previous = at[skill.handover], out[skills[n - 1].name]
            if not (start <= handover < stop and previous[0] <= handover < previous[1]):
                raise SkillError(f"{skill.name}'s handover ({skill.handover}, frame {handover}) isn't inside both "
                                 f"{skills[n - 1].name} (frames {previous[0]}-{previous[1] - 1}) and {skill.name} "
                                 f"(frames {start}-{stop - 1})")
        out[skill.name] = (start, stop)
    return out


@dataclass(frozen=True)
class DemoCuts:
    """One kept demo's events and skill spans, as frames of its session episode."""

    plan: object  # build.Plan: the session, episode and kept frames
    events: dict[str, int] | None
    spans: dict[str, tuple[int, int]] | None
    error: str | None = None

    def record(self) -> dict:
        p = self.plan
        entry = {"session": p.recording.name, "episode": p.episode, "kept": [p.start, p.stop]}
        if self.error:
            return {**entry, "error": self.error}
        return {**entry, "events": {k: p.start + v for k, v in self.events.items()},
                "spans": {k: [p.start + a, p.start + b] for k, (a, b) in self.spans.items()}}


def cut_demos(s, task, program) -> tuple[list[DemoCuts], list[str]]:
    """Every kept demo's events and spans, from the raw sessions as `so101 build` reads them, and the
    build plan's notes. A demo that can't be cut has its error instead."""
    from so_101 import build
    from so_101.dataset_io import column, episode_table

    plans, notes = build.plan(s, task)
    out = []
    for p in plans:
        names = list(p.meta.features["observation.state"]["names"])
        state = column(episode_table(p.meta, p.episode), "observation.state")[p.start:p.stop]
        try:
            at = events(state, names, p.meta.fps, program)
            out.append(DemoCuts(p, at, spans(task.skills, at, p.meta.fps, len(state))))
        except SkillError as error:
            out.append(DemoCuts(p, None, None, str(error)))
    return out, notes


# --- Running a skills model ------------------------------------------------------------------------


class SkillPolicy:
    def __init__(self, inner, skills: tuple[Skill, ...], program, *, log: Callable,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """`inner` is the policy driver it wraps; `log(type, **fields)` writes an event (the recorder's)."""
        from so_101.supervisor.detector import Detector

        self.inner, self.skills, self.program = inner, skills, program
        self._log, self.clock = log, clock
        self._detector_type = Detector
        self.detector = Detector(program)
        self.current = 0
        self._resumed = False
        # A local engine builds the policy's frame from the driver's features and processed observation
        # (`EngineDriver.step`): add the one-hot to both, so ACT reads which skill it's on. A remote
        # engine builds its frame from its own features, and pi0.5 reads the prompt instead.
        feature = state_feature(skills)
        if hasattr(inner, "dataset_features"):
            inner.dataset_features = {**inner.dataset_features, STATE_KEY: feature}
        if hasattr(inner, "observation_processor"):
            process = inner.observation_processor
            inner.observation_processor = lambda observation: {**process(observation), **self.values()}

    @property
    def skill(self) -> Skill:
        return self.skills[self.current]

    def values(self) -> dict[str, float]:
        """The one-hot, keyed as `state_feature` names it."""
        return {f"skill.{s.name}": float(n == self.current) for n, s in enumerate(self.skills)}

    def _set_task(self) -> None:
        engine = getattr(self.inner, "engine", None)
        if engine is not None and hasattr(engine, "set_task"):
            engine.set_task(self.skill.prompt)

    # --- The driver's interface ---------------------------------------------------------------------

    def start(self) -> None:
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def maintain(self) -> str | None:
        return self.inner.maintain()

    def asks(self) -> str | None:
        return self.inner.asks()

    def begin_episode(self, number: int) -> None:
        self.current, self._resumed = 0, False
        self.detector = self._detector_type(self.program)
        self.inner.begin_episode(number)
        self._set_task()
        self._log("skill", skill=self.skill.name, cause="start")

    def end_episode(self) -> None:
        self.inner.end_episode()

    def pause(self) -> None:
        self.inner.pause()

    def run(self) -> None:
        self.inner.run()
        self._resumed = True

    def hold(self, robot) -> None:
        self.inner.hold(robot)

    @property
    def failure(self) -> str | None:
        return self.inner.failure

    @property
    def raised(self) -> str | None:
        return self.inner.raised

    @raised.setter
    def raised(self, value: str | None) -> None:
        self.inner.raised = value

    @property
    def retrying(self) -> str | None:
        return self.inner.retrying

    def step(self, robot, observation: dict) -> dict | None:
        """The current skill's action, then the detector on this tick's pose: at the next skill's
        handover, switch. The switch takes effect from the next tick's inference."""
        action = self.inner.step(robot, observation)
        if self._resumed:  # something else held or moved the arm since the last tick: restart the timers
            self.detector.resumed()
            self._resumed = False
        pose = {k: float(v) for k, v in observation.items() if k.endswith(".pos")}
        nxt = self.current + 1
        for signal in self.detector.step(self.clock(), pose, action is not None):
            if nxt < len(self.skills) and signal.kind == self.skills[nxt].handover:
                previous, self.current = self.skill.name, nxt
                self._set_task()
                self.inner.run()  # a clean slate: no queued action of the last skill reaches the arm
                self._log("skill", skill=self.skill.name, previous=previous, cause=signal.kind)
                break
        return action
