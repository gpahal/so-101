"""The interface every policy in the stack has, and the base class layers build on.

The control loop steps the top of the stack. Each tick it drives, it builds one `Tick` and calls
`step`, which returns an `Output`; each tick it doesn't, it calls `hold`. Around attempts it calls
`begin_episode` and `end_episode`; when it hands the arm to the stack, `run` with a `Handback`
saying why; when it takes the arm, `pause`. Every tick, in every state, `maintain`.

A `Layer` wraps a policy and forwards every method to it, so a layer overrides only what it adds.
Outputs flow up the stack: a layer reads the signals and readings of the one below in the Output it
gets back, and returns it, or its own. Only the leaf (`engine.EngineDriver`) acts on a `Conditioning`;
layers pass it down, so where a layer sits never changes what the model is told.

A layer's own exceptions are its own (`Layer.safely`): logged once and announced once through
`maintain`, after which the layer passes every call through. `safely` wraps only the layer's own logic:
every call down the stack runs outside it, so an exception from the arm or the model reaches the loop
unchanged. A layer the run can't do without (`essential`: the perception and the skill sequencer of a
skills run, whose skill would otherwise never change) reports itself broken in `health`, which refuses
Resume.
"""

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

# Who drove the arm on a tick, as `Output.driver` and the recording's control.driver column say.
OTHER, POLICY, WAITING, PLAYBACK, HOLD = 0, 1, 2, 3, 4
DRIVERS = {OTHER: "other", POLICY: "policy", WAITING: "waiting", PLAYBACK: "playback", HOLD: "hold"}
# Why the stack gets the arm: an attempt starts, a person resumed, or a layer above gives it back.
START, PERSON, LAYER = "start", "person", "layer"
# Why a layer asks for a person (`Ask.kind`): the attempt is stuck, the task looks done, or the layer
# itself can't go on (it stopped after an error while it held the arm).
STUCK, DONE = "stuck", "done"
ASK_KINDS = (STUCK, DONE, LAYER)


@dataclass(frozen=True)
class Tick:
    now: float  # the control loop's clock
    frame: int  # the open episode's frame number this tick, or -1 with no attempt open
    observation: dict  # the robot's raw observation: joints ("<joint>.pos") and camera frames
    pose: dict  # the measured joints, keyed like the robot's actions
    at_rest: bool = False  # every joint within rollout.TOLERANCE of the rest pose, as the controller measured


@dataclass(frozen=True)
class Ask:
    """A layer asking for a person: the loop pauses, says `text`, and logs `kind` (one of ASK_KINDS)
    in the `state` event for `policy_asked`."""

    kind: str
    text: str


@dataclass(frozen=True)
class Readings:
    """What the perception has seen this attempt, as of its last tick."""

    t: float = 0.0  # seconds the model has driven
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
    asks: Ask | None = None  # a layer wants a person: the loop pauses
    # The index of the skill the model is conditioned on (run.json's policy.skills), or None for no
    # change: the loop keeps the last one for the attempt, so a layer's playback and a person's Teleop
    # are recorded under the skill they interrupted.
    skill: int | None = None
    # On a PLAYBACK tick, the episode frame (`Tick.frame`) its goal pose was measured in: the loop drops
    # every moment after it from the history a person's Rewind plays back. None on every other tick.
    goal_frame: int | None = None


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
    the supervisor's program when the supervisor runs, and the task's fps."""

    event: Callable[..., None]  # event(type, **fields): the recorder's
    clock: Callable[[], float]
    grasp: object  # grasp.thresholds.Thresholds
    program: object  # the supervisor's program (`supervisor.program.Program`), or None when the supervisor is off
    # The task's fps: each tick the model drove counts as 1/fps of its time, in the perception's clock and the
    # supervisor's playback history, as the offline replays count each recorded frame (`grasp.replay.LiveFeed`).
    fps: float


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
    essential: bool = field(default=False, init=False)  # broken, it refuses Resume (`health`): set by stack.build
    _said: list[str] = field(default_factory=list, init=False)  # for the operator, through `maintain`

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
        messages = self.inner.maintain()
        return [*messages, *self.said()]

    def health(self) -> Health:
        health = self.inner.health()
        if self.essential and self.broken is not None and health.failure is None:
            return replace(health, failure=f"the {self.name} layer stopped after an error: {self.broken}")
        return health

    def record_failure(self, detail: str) -> None:
        self.inner.record_failure(detail)

    def say(self, text: str) -> None:
        """Tell the operator `text`, through the next `maintain`."""
        self._said.append(text)

    def said(self) -> list[str]:
        """What the layer has to tell the operator since the last call."""
        said, self._said = self._said, []
        return said

    def safely(self, fn, *args) -> bool:
        """Run the layer's own logic, never a call down the stack. An exception is logged once as an
        `error` event and announced once through `maintain`, and from then on `safely` runs nothing: the
        layer passes through. Returns whether `fn` ran without raising."""
        if self.broken is not None:
            return False
        try:
            fn(*args)
            return True
        except Exception as error:
            self.broken = f"{type(error).__name__}: {error}"
            self.ctx.event(self.name, kind="error", error=self.broken, traceback=traceback.format_exc()[-2000:])
            then = ("Resume is refused until so101 restarts: the run can't go on without it." if self.essential
                    else "The policy runs without it.")
            self.say(f"{self.name.capitalize()} layer: stopped after an error ({self.broken}). {then}")
            self.failed()
            return False

    def failed(self) -> None:
        """Called once after the layer's own error, to leave the arm in a safe hand. Override to undo a takeover."""
