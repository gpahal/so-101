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
event that switches to it at runtime, a carry or a release (HANDOVERS), which the detector sees live. A
close isn't one: at runtime a close that misses the block reopens, and a switch on it would already have
told the model to place. The handover must fall inside both its own span and the previous skill's, on
every demo (`spans`), so the model has seen every state on both sides of the switch.

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
HANDOVERS = ("carry", "release")  # what the detector sees as the policy drives and can switch skills on
# Raise with any change to LIFT_DEG, WIDEST_WITHIN_S, MIN_SPAN_S or how `events` and `spans` cut a demo, or
# `label_run` a run: a skills build records it (`data.build_inputs`), so one made under other rules is stale.
CUT_RULES_VERSION = 3
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
        if n > 0 and handover == "close":
            raise SkillError(f"{where} can't hand over on the close: as the policy runs, a close that misses the "
                             f"block reopens, and the switch would already have happened. Use {' or '.join(HANDOVERS)}.")
        if n > 0 and handover not in HANDOVERS:
            raise SkillError(f"{where} needs a handover, the event that switches to it as the policy runs: one of "
                             f"{', '.join(HANDOVERS)} (got {handover!r}).")
        skills.append(Skill(name, prompt.strip(), Point.parse(entry.get("from"), f"{where} from"),
                            Point.parse(entry.get("to"), f"{where} to"), handover))
    if len({s.prompt for s in skills}) != len(skills):
        raise SkillError(f"[{task}.skills]: each skill needs its own prompt, or a model can't tell them apart.")
    return tuple(skills)


def gripper_hash(gripper) -> str:
    """The sha256 of the grasp thresholds' `[gripper]` table (`grasp.thresholds.Gripper`), whose events
    cut the demos and switch skills at runtime. A skills build records it, and a skills model is run
    only with the thresholds its dataset was cut with."""
    import dataclasses
    import hashlib
    import json

    fields = json.dumps(dataclasses.asdict(gripper), sort_keys=True)
    return hashlib.sha256(fields.encode()).hexdigest()


def contract(skills: tuple[Skill, ...], gripper) -> dict:
    """What a skills model is tied to at runtime: each skill's name and prompt, in order, and the
    `[gripper]` thresholds whose events switch them. A skills build records it in build.json."""
    return {"skills": [{"name": s.name, "prompt": s.prompt} for s in skills], "gripper_sha256": gripper_hash(gripper)}


def state_feature(skills: tuple[Skill, ...]) -> dict:
    """The one-hot `observation.environment_state` a skills dataset has, and a skills run adds."""
    return {"dtype": "float32", "shape": (len(skills),), "names": [f"skill.{s.name}" for s in skills]}


def one_hot(skills: tuple[Skill, ...], index: int) -> np.ndarray:
    value = np.zeros(len(skills), dtype=np.float32)
    value[index] = 1.0
    return value


# --- Cutting demos ---------------------------------------------------------------------------------


def _replay(state: np.ndarray, names: list[str], fps: float, thresholds) -> tuple[list[int], dict[str, int]]:
    """Every close's frame, and the first carry's and release's, as the detector sees `state` frame by frame."""
    from so_101.grasp.detector import Detector

    detector = Detector(thresholds)
    closes, found = [], {}
    for i, row in enumerate(state):
        for signal in detector.step(i / fps, dict(zip(names, map(float, row), strict=True)), True):
            if signal.kind == "close":
                closes.append(i)
            elif signal.kind in ("carry", "release") and signal.kind not in found:
                found[signal.kind] = i
    return closes, found


def _derive(state: np.ndarray, names: list[str], fps: float, closes: list[int], found: dict[str, int]) -> dict:
    """Each event's frame from what the detector found, or None for one that didn't happen. The close
    is the one that led to the carry, or the last one without a carry."""
    carry, release = found.get("carry"), found.get("release")
    before = [c for c in closes if carry is None or c <= carry]
    close = max(before) if before else None
    lift = widest = None
    if close is not None:
        column = state[:, names.index(SHOULDER_LIFT)]
        moved = np.where(np.abs(column[close:] - column[close]) > LIFT_DEG)[0]
        lift = close + int(moved[0]) if len(moved) else None
    if release is not None:
        gripper = state[release: release + round(WIDEST_WITHIN_S * fps), names.index(GRIPPER)]
        widest = release + int(np.argmax(gripper))
    return {"start": 0, "close": close, "lift": lift, "carry": carry, "release": release, "widest": widest,
            "end": len(state) - 1}


def events(state: np.ndarray, names: list[str], fps: float, thresholds) -> dict[str, int]:
    """Each event's frame in one demo's joint readings (frames x joints, named `names`). Raises
    SkillError when the demo has no carry or release, or the block never leaves the mat."""
    closes, found = _replay(state, names, fps, thresholds)
    if "carry" not in found:
        raise SkillError(f"no carry: {len(closes)} close{'' if len(closes) == 1 else 's'}, and the block was never held")
    if "release" not in found:
        raise SkillError("no release: the block was never let go")
    at = _derive(state, names, fps, closes, found)
    if at["lift"] is None:
        raise SkillError(f"no lift: shoulder_lift never moved {LIFT_DEG:g} deg after the close")
    return at


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


def label_run(skills: tuple[Skill, ...], state: np.ndarray, names: list[str], fps: float, thresholds,
              scripted: np.ndarray, told: np.ndarray | None = None, teleop: np.ndarray | None = None) -> np.ndarray:
    """Each frame's skill, as an index in `skills`, for one attempt of a run, or -1 for a frame in none:
    what happened, found as the runtime finds it (`policy.skills.SkillSequencer` over the perception).

    The attempt's joint readings (`state`, frames x joints named `names`) go through one detector frame
    by frame, as the perception's does at runtime. Each attempt starts on the first skill and moves to the
    next at its handover event. A frame a script moved (`scripted`: the move to the start pose, a Rewind,
    a layer's playback, the return to rest) isn't fed to the detector, and a frame a person drove in
    Teleop (`teleop`) is fed only while nothing is carried: a person's close and carry count, so a
    correction that starts in pick is place from its carry, but a person opening the jaws on a carried
    block is never a release, because the perception never sees Teleop at runtime. After either, the next
    frame fed tells the detector the arm moved (`Detector.resumed`), as the perception does after a
    handback; if it then shows neither a carry nor a release, what was held or let go was undone, and the
    replay starts again on the first skill, as the sequencer does.

    `told` is, per frame, the skill a skills model was told to do (control.skill, mapped onto `skills`),
    -1 where it wasn't driving: it labels those frames, and the replay labels the rest. Last, each skill's
    `to` bound applies, from the events of the replay its frame belongs to: a frame past it is in no skill,
    as in a demo, where the return to rest after a place that ends at the widest opening is in none. A
    bound whose event didn't happen doesn't apply."""
    from so_101.grasp.detector import Detector

    labels = np.zeros(len(state), dtype=np.int8)
    replays = []  # (first frame, one past the last, closes, first carry and release) of each replay
    current, start, closes, found = 0, 0, [], {}
    detector = Detector(thresholds)
    gap = handback = False  # frames since the last one fed weren't fed; a script or a person had the arm
    for i, row in enumerate(state):
        person = teleop is not None and bool(teleop[i])
        if scripted[i] or (person and detector.carried):
            gap = handback = True
            labels[i] = current
            continue
        pose = dict(zip(names, map(float, row), strict=True))
        if gap or (handback and not person):
            detector.resumed(True, pose)
            gap, handback = False, person
            if not (detector.carried or detector.released) and (current or found):
                replays.append((start, i, closes, found))
                current, start, closes, found = 0, i, [], {}
        for signal in detector.step(i / fps, pose, True):
            if signal.kind == "close":
                closes.append(i - start)
            elif signal.kind in ("carry", "release") and signal.kind not in found:
                found[signal.kind] = i - start
            if current + 1 < len(skills) and signal.kind == skills[current + 1].handover:
                current += 1
        handback |= person
        labels[i] = current
    replays.append((start, len(state), closes, found))
    if told is not None:
        labels = np.where((told >= 0) & ~scripted, told, labels).astype(np.int8)
    for a, b, closes, found in replays:
        at = _derive(state[a:b], names, fps, closes, found)
        for n, skill in enumerate(skills):
            if at[skill.stop.event] is None:
                continue
            bound = a + frame_of(skill.stop, at, fps)
            piece = labels[a:b]
            piece[(piece == n) & (np.arange(a, b) > bound)] = -1
    return labels


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
