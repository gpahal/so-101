"""Run a LeRobot inference engine one control tick at a time, as LeRobot's rollout strategies do.

The controller calls `step` only while the policy drives. Every action goes through LeRobot's
dispatch gate (`next_interpolated_action`, `commit_action`), the robot action processor and the
robot wrapper, so `max_relative_target` still caps it.

The engine is local (sync) or remote (the cloud GPU's policy server). Both clear their queued actions
on `pause` and `reset`; the remote one also tells the server to drop its own, and holds the arm until
the server confirms.

An exception a step raises from the policy's side (inference, its processors, the engine's gate) is
the driver's `failure` from then on, whatever the engine's own `failed` says: a local policy that
raised once refuses Resume until `serve` restarts. The controller tells the two apart and sets
`raised` (`Controller._drive`): one raised by the robot (sending the goal or holding) is the
follower's, not the policy's, and isn't kept.

LeRobot's remote engine never reconnects after a fault. With `reconnect`, a function that connects a
new remote engine, the driver builds one in a background thread once the policy has failed, retrying
every RECONNECT_EVERY_S, and swaps it in on the control loop's thread in `maintain`. Nothing moves:
the controller is in Fault, and Resume stays refused until the new engine is in. LeRobot's policy
server admits one session at a time, and an engine that failed releases its own. One that didn't,
because the exception came from a step on the policy's side, still holds it, so that thread stops
the old engine before it connects: otherwise every new session would be refused as busy.
"""

import threading
from collections.abc import Callable
from typing import Any

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

    def stop(self) -> None:
        with self._swap_lock:
            self._stopping.set()
            spare, self._replacement = self._replacement, None
        self.engine.stop()
        if spare is not None:
            spare.stop()

    def maintain(self) -> str | None:
        """Called by the control loop every tick. After a failure, start reconnecting; once a new
        engine is connected, swap it in. Returns what to tell the operator, or None."""
        if self._reconnect is None or self.failure is None:
            return None
        if (engine := self._replacement) is not None:
            self._replacement, self._reconnecting = None, None
            old, self.engine = self.engine, engine
            self.raised = None
            self._clear()
            threading.Thread(target=self._stop_quietly, args=(old,), daemon=True).start()
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
        self.engine.pause()
        self._clear()

    def run(self) -> None:
        """Start or continue the policy from a clean slate: reset clears its queued actions."""
        self.engine.reset()
        self._clear()
        self.engine.resume()

    def hold(self, robot) -> None:
        """Hold the arm at its last target, as LeRobot does while a policy may not move it."""
        self._clear()
        robot.hold()
        self.engine.acknowledge_hold()

    @property
    def failure(self) -> str | None:
        if self.raised is not None:
            return self.raised
        return (self.engine.failure_traceback or "the policy failed") if self.engine.failed else None

    def step(self, robot, observation: dict[str, Any]) -> dict | None:
        """Send the policy's next action. Returns it as the policy gave it, before the step cap, when it
        reached the arm, or None on a tick with none."""
        from lerobot.inference import commit_action, next_interpolated_action
        from lerobot.utils import OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        if self._processed is None or self.interpolator.needs_new_action():
            self._processed = self.observation_processor(observation)
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
