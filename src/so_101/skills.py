"""Skills: named parts of a task's demos, cut where the arm's readings show an event, trained into one model.

A task's `[<task>.skills]` in tasks.toml lists its skills in the order they run. Each skill spans every
kept demo from one event to another, plus or minus seconds:

    pick  = { prompt = "...", from = "start", to = "carry + 0.5" }
    place = { prompt = "...", from = "lift", to = "widest + 0.2", handover = "carry" }

The events (`events`) come from the grasp detector (`grasp.detector`) over the demo's joint readings,
with the grasp thresholds (`grasp.thresholds`), plus two found here:
- start, end: the kept demo's first and last frames;
- close: the gripper drops below the thresholds' closed_below, on the close that led to the carry;
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

`policy.skills.SkillSequencer` runs a skills model: a layer of the policy stack that starts each attempt
on the first skill and switches at each handover, conditioning the model on the next skill's prompt and
one-hot. The recording writes the skill on each frame as control.skill (`policy-layers.md`).
"""

import re
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


def events(state: np.ndarray, names: list[str], fps: float, thresholds) -> dict[str, int]:
    """Each event's frame in one demo's joint readings (frames x joints, named `names`). Raises
    SkillError when the demo has no carry or release, or the block never leaves the mat."""
    from so_101.grasp.detector import Detector

    detector = Detector(thresholds)
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


def cut_demos(s, task, thresholds) -> tuple[list[DemoCuts], list[str]]:
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
            at = events(state, names, p.meta.fps, thresholds)
            out.append(DemoCuts(p, at, spans(task.skills, at, p.meta.fps, len(state))))
        except SkillError as error:
            out.append(DemoCuts(p, None, None, str(error)))
    return out, notes
