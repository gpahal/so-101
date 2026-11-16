"""The interface every policy in the stack has, and the base class layers build on.

The control loop steps the top of the stack. Each tick it drives, it builds one `Tick` and calls
`step`, which returns an `Output`; each tick it doesn't, it calls `hold`. Around attempts it calls
`begin_episode` and `end_episode`; when it hands the arm to the stack, `run` with a `Handback`
saying why; when it takes the arm, `pause`. Every tick, in every state, `maintain`.

A `Layer` wraps a policy and forwards every method to it, so a layer overrides only what it adds.
Outputs flow up the stack: a layer reads the signals and readings of the one below in the Output it
gets back, and returns it, or its own. Only the leaf (`engine.EngineDriver`) acts on a `Conditioning`;
layers pass it down, so where a layer sits never changes what the model is told.

A layer's own exceptions are its own (`Layer.safely`): logged once, after which the layer passes every
call through. An exception from the arm or the model reaches the loop unchanged.
"""

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

# Who drove the arm on a tick, as `Output.driver` and the recording's control.driver column say.
OTHER, POLICY, WAITING, PLAYBACK, HOLD = 0, 1, 2, 3, 4
DRIVERS = {OTHER: "other", POLICY: "policy", WAITING: "waiting", PLAYBACK: "playback", HOLD: "hold"}
# Why the stack gets the arm: an attempt starts, a person resumed, or a layer above gives it back.
START, PERSON, LAYER = "start", "person", "layer"


@dataclass(frozen=True)
class Tick:
    now: float  # the control loop's clock
    frame: int  # the open episode's frame number this tick, or -1 with no attempt open
    observation: dict  # the robot's raw observation: joints ("<joint>.pos") and camera frames
    pose: dict  # the measured joints, keyed like the robot's actions


@dataclass(frozen=True)
class Readings:
    """What the perception has seen this attempt, as of its last tick."""

    t: float = 0.0  # seconds the model has driven
    near: tuple[str, ...] = ()  # stuck rules within the supervisor program's `watch.near` of firing
    closes: int = 0
    regrasps: int = 0
    carried: bool = False
    released: bool = False
    fired: tuple[str, ...] = ()


@dataclass(frozen=True)
class Output:
    action: dict | None = None  # the model's action when it reached the arm, before the step cap
    driver: int = WAITING  # who drove: POLICY, WAITING (the model sent nothing), PLAYBACK or HOLD
    signals: tuple = ()  # the perception's `grasp.detector.Signal`s this tick
    readings: Readings | None = None
    asks: str | None = None  # why a layer wants a person: the loop pauses
    # The index of the skill the model is conditioned on (run.json's policy.skills), or None for no
    # change: the loop keeps the last one for the attempt, so a layer's playback and a person's Teleop
    # are recorded under the skill they interrupted.
    skill: int | None = None


@dataclass(frozen=True)
class Handback:
    why: str  # START, PERSON or LAYER
    moved: bool = False  # the arm may have moved while someone else had it


@dataclass(frozen=True)
class Conditioning:
    """What the model is told: a prompt (pi0.5 reads it) and, for a skills model, which skill as a
    one-hot named `skill.<name>` (ACT reads it as observation.environment_state)."""

    prompt: str
    skills: tuple[str, ...] = ()
    index: int = -1

    def values(self) -> dict[str, float]:
        return {f"skill.{name}": float(n == self.index) for n, name in enumerate(self.skills)}


@dataclass(frozen=True)
class Health:
    failure: str | None = None  # why the model can't drive: Resume is refused while it's set
    retrying: str | None = None  # how reconnecting to a remote model is going, while it is


@dataclass(frozen=True)
class LayerContext:
    """What every layer gets: the run's event log, the loop's clock, the gripper and rule thresholds,
    and the supervisor's program (its `watch.near` says which rules the perception reports as near)."""

    event: Callable[..., None]  # event(type, **fields): the recorder's
    clock: Callable[[], float]
    grasp: object  # grasp.thresholds.Thresholds
    program: object  # supervisor.program.Program


class Policy(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def begin_episode(self, number: int) -> None: ...
    def end_episode(self) -> None: ...
    def run(self, handback: Handback) -> None: ...
    def pause(self) -> None: ...
    def hold(self, robot) -> None: ...
    def step(self, robot, tick: Tick) -> Output: ...
    def condition(self, conditioning: Conditioning) -> None: ...
    def maintain(self) -> list[str]: ...
    def health(self) -> Health: ...
    def record_failure(self, detail: str) -> None: ...


@dataclass
class Layer:
    """A policy over `inner`. Every method forwards; override what the layer adds. `name` labels its
    events and errors."""

    inner: Policy
    ctx: LayerContext
    name: str = "layer"
    broken: str | None = field(default=None, init=False)  # why the layer stopped, after an error of its own

    def start(self) -> None:
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def close(self) -> None:
        self.inner.close()

    def begin_episode(self, number: int) -> None:
        self.inner.begin_episode(number)

    def end_episode(self) -> None:
        self.inner.end_episode()

    def run(self, handback: Handback) -> None:
        self.inner.run(handback)

    def pause(self) -> None:
        self.inner.pause()

    def hold(self, robot) -> None:
        self.inner.hold(robot)

    def step(self, robot, tick: Tick) -> Output:
        return self.inner.step(robot, tick)

    def condition(self, conditioning: Conditioning) -> None:
        self.inner.condition(conditioning)

    def maintain(self) -> list[str]:
        return self.inner.maintain()

    def health(self) -> Health:
        return self.inner.health()

    def record_failure(self, detail: str) -> None:
        self.inner.record_failure(detail)

    def safely(self, fn, *args) -> bool:
        """Run the layer's own logic. An exception is logged once as an `error` event, and from then on
        `safely` runs nothing: the layer passes through. Returns whether `fn` ran without raising."""
        if self.broken is not None:
            return False
        try:
            fn(*args)
            return True
        except Exception as error:
            self.broken = f"{type(error).__name__}: {error}"
            self.ctx.event(self.name, kind="error", error=self.broken, traceback=traceback.format_exc()[-2000:])
            self.failed()
            return False

    def failed(self) -> None:
        """Called once after the layer's own error, to leave the arm in a safe hand. Override to undo a takeover."""
