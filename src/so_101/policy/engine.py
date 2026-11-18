"""The leaf of the policy stack: a LeRobot inference engine, one control tick at a time, as LeRobot's
rollout strategies run it. The only module that touches LeRobot's inference (`base.Policy`).

The control loop steps the stack only while it drives, and the stack steps this only while the model
drives. Every action goes through LeRobot's dispatch gate (`next_interpolated_action`, `commit_action`), the robot action processor and
`follower.Arm`, so `max_relative_target` still caps it.

The engine is local (sync) or remote (the cloud GPU's policy server). Both clear their queued actions
on `pause` and `reset`; the remote one also tells the server to drop its own, and holds the arm until
the server confirms.

An exception a step raises from the policy's side (inference, its processors, the engine's gate) is
the driver's `failure` from then on, whatever the engine's own `failed` says: a local policy that
raised once refuses Resume until so101 restarts. The controller tells the two apart and sets
`raised` (`Controller._drive`): one raised by the robot (sending the goal or holding) is the
follower's, not the policy's, and isn't kept.

LeRobot's remote engine never reconnects after a fault. With `reconnect`, the driver builds a new engine
in a background thread once the policy has failed, retrying every RECONNECT_EVERY_S, and swaps it in on
the loop's thread in `maintain`; Resume stays refused until then. The policy server admits one session
at a time: an engine that failed releases its own, but one whose step raised on the policy's side still
holds it, so that thread stops the failed engine first.
"""

import threading
from collections.abc import Callable
from typing import Any

from so_101.policy.base import POLICY, WAITING, Conditioning, Handback, Health, Output, Tick

STATE_KEY = "observation.environment_state"  # a skills model's one-hot input (`so_101.skills`)
# How often to try connecting again after the remote engine failed. The server frees a lost session
# after its idle timeout (10 s by default), so the first tries may be refused as busy.
RECONNECT_EVERY_S = 5.0


def _first(pair: tuple[dict, dict]) -> dict:
    return pair[0]


def _same(observation: dict) -> dict:
    return observation


class EngineDriver:
    def __init__(
        self,
        engine,
        action_keys: list[str],
        *,
        dataset_features: dict | None = None,
        observation_processor: Callable[[dict], dict] = _same,
        action_processor: Callable[[tuple[dict, dict]], dict] = _first,
        interpolation_multiplier: int = 1,
        reconnect: Callable[[], Any] | None = None,
    ) -> None:
        from lerobot.utils.action_interpolator import ActionInterpolator

        self.engine = engine
        self.action_keys = action_keys
        self.dataset_features = dataset_features or {}
        self.observation_processor = observation_processor
        self.action_processor = action_processor
        self.interpolator = ActionInterpolator(multiplier=interpolation_multiplier)
        self._processed: dict | None = None
        self._reconnect = reconnect
        self._reconnecting: threading.Thread | None = None
        self._stopping = threading.Event()
        self._swap_lock = threading.Lock()  # `_connect` hands over its engine, or `stop` stops it: never neither
        self._replacement = None  # a connected engine, waiting for `maintain` to swap it in
        self._tries = 0
        self.retrying: str | None = None  # how reconnecting is going, while it is
        self.raised: str | None = None  # the traceback of an exception a step raised from the policy's side
        self.running = False  # between `run` and `pause`
        # The skill one-hot a skills model reads as observation.environment_state (`condition`), added
        # to each processed observation; the frame it's built into names it in `dataset_features`. A
        # remote skills run's features already have it (`control.main.remote_context`): it starts on
        # the first skill, so no frame is ever built without it.
        names = self.dataset_features.get(STATE_KEY, {}).get("names") or []
        self._skill_values: dict[str, float] = {name: float(n == 0) for n, name in enumerate(names)}
        self._conditioning: Conditioning | None = None  # the last one, for an engine swapped in

    @classmethod
    def from_context(cls, ctx, reconnect: Callable[[], Any] | None = None) -> "EngineDriver":
        """The engine, processors and features LeRobot's `build_rollout_context` made."""
        return cls(
            ctx.policy.inference,
            ctx.data.ordered_action_keys,
            dataset_features=ctx.data.dataset_features,
            observation_processor=ctx.processors.robot_observation_processor,
            action_processor=ctx.processors.robot_action_processor,
            interpolation_multiplier=ctx.runtime.cfg.interpolation_multiplier,
            reconnect=reconnect,
        )

    def start(self) -> None:
        self.engine.start()

    def close(self) -> None:
        """Nothing beyond `stop`: the process stops the driver itself."""

    def stop(self) -> None:
        with self._swap_lock:
            self._stopping.set()
            spare, self._replacement = self._replacement, None
        self.engine.stop()
        if spare is not None:
            spare.stop()

    def maintain(self) -> list[str]:
        """Called by the control loop every tick. After a failure, start reconnecting; once a new
        engine is connected, swap it in. Returns what to tell the operator."""
        message = self._maintain()
        return [message] if message is not None else []

    def _maintain(self) -> str | None:
        if self._reconnect is None or self.failure is None:
            return None
        if (engine := self._replacement) is not None:
            self._replacement, self._reconnecting = None, None
            old, self.engine = self.engine, engine
            self.raised = None
            self._clear()
            threading.Thread(target=self._stop_quietly, args=(old,), daemon=True).start()
            if self._conditioning is not None and (set_task := getattr(engine, "set_task", None)) is not None:
                set_task(self._conditioning.prompt)  # it was made with the run's first prompt
            self.engine.start()
            self.retrying = None
            return "Reconnected to the cloud GPU's policy server: Resume works again."
        if self._reconnecting is None:
            self._tries = 0
            self.retrying = "Reconnecting to the cloud GPU's policy server..."
            holding = None if self.engine.failed else self.engine  # still holds the server's one session
            self._reconnecting = threading.Thread(target=self._connect, args=(holding,), name="so101-reconnect",
                                                  daemon=True)
            self._reconnecting.start()
            return self.retrying
        return None

    def _connect(self, holding=None) -> None:
        """Connect a new engine for `maintain` to swap in, after stopping `holding`, an old engine that
        still holds the server's session. Stopping it takes up to 2 s, so it's done here, off the loop."""
        if holding is not None:
            self._stop_quietly(holding)
        while not self._stopping.is_set():
            self._tries += 1
            try:
                engine = self._reconnect()
            except Exception as error:
                self.retrying = (f"Reconnecting to the cloud GPU's policy server: try {self._tries} failed "
                                 f"({type(error).__name__}: {error}). Trying again every {RECONNECT_EVERY_S:g} s.")
                self._stopping.wait(RECONNECT_EVERY_S)
                continue
            with self._swap_lock:
                stopped = self._stopping.is_set()
                if not stopped:
                    self._replacement = engine
            if stopped:
                self._stop_quietly(engine)
            return

    @staticmethod
    def _stop_quietly(engine) -> None:
        try:
            engine.stop()
        except Exception:
            pass  # a failed engine may not stop cleanly; its worker is gone either way

    def _clear(self) -> None:
        self.interpolator.reset()
        self._processed = None

    def pause(self) -> None:
        """Stop the policy: no action it queued reaches the arm after this."""
        self.running = False
        self.engine.pause()
        self._clear()

    def run(self, handback: Handback | None = None) -> None:
        """Start or continue the policy from a clean slate: reset clears its queued actions."""
        self.engine.reset()
        self._clear()
        self.engine.resume()
        self.running = True

    def condition(self, conditioning: Conditioning) -> None:
        """Tell the model what to do: the prompt as the engine's task, and a skills model's one-hot as
        observation.environment_state in the frame it builds. While the model drives, a change clears
        its queued actions, so none meant for the old skill reaches the arm. An engine reconnected
        later is told the last prompt before it starts."""
        self._conditioning = conditioning
        set_task = getattr(self.engine, "set_task", None)
        changed = bool(set_task(conditioning.prompt)) if set_task is not None else False
        values = conditioning.values()
        if values and STATE_KEY not in self.dataset_features:
            self.dataset_features = {**self.dataset_features,
                                     STATE_KEY: {"dtype": "float32", "shape": (len(values),), "names": list(values)}}
        changed = changed or values != self._skill_values
        self._skill_values = values
        if changed and self.running:
            self.run()

    def hold(self, robot) -> None:
        """Hold the arm at its last target, as LeRobot does while a policy may not move it."""
        self._clear()
        robot.hold()
        self.engine.acknowledge_hold()

    def begin_episode(self, number: int) -> None:
        """An attempt opened. The engine needs nothing: `run` clears it before the policy drives."""

    def end_episode(self) -> None:
        """Nothing to do: the engine keeps no per-attempt state."""

    @property
    def failure(self) -> str | None:
        if self.raised is not None:
            return self.raised
        return (self.engine.failure_traceback or "the policy failed") if self.engine.failed else None

    def health(self) -> Health:
        return Health(self.failure, self.retrying)

    def record_failure(self, detail: str) -> None:
        """A step raised from the policy's side (the loop tells it from the robot's): Resume is refused from now on."""
        self.raised = detail

    def step(self, robot, tick: Tick) -> Output:
        """Send the policy's next action: an Output with the action as the policy gave it, before the
        step cap, when it reached the arm, or WAITING on a tick with none."""
        action = self._step(robot, tick.observation)
        return Output(action, POLICY) if action is not None else Output(None, WAITING)

    def _step(self, robot, observation: dict[str, Any]) -> dict | None:
        from lerobot.inference import commit_action, next_interpolated_action
        from lerobot.utils import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        if self._processed is None or self.interpolator.needs_new_action():
            self._processed = {**self.observation_processor(observation), **self._skill_values}
            self.engine.notify_observation(self._processed)
        processed = self._processed

        def hold() -> None:
            if robot.supports_hold:
                robot.hold()

        if not self.engine.ready:
            hold()
            return None
        target = next_interpolated_action(
            self.engine,
            self.interpolator,
            lambda: build_dataset_frame(self.dataset_features, processed, prefix=OBS_STR),
            hold=hold,
        )
        if target is None:
            return None
        action = {key: target[i].item() for i, key in enumerate(self.action_keys)}
        goal = self.action_processor((action, observation))
        if not commit_action(self.engine, self.interpolator, lambda: robot.send_action(goal), hold=hold):
            return None
        self.engine.pump_query(processed)
        return action
