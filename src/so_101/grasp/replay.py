"""The live perception's view of a recorded attempt, for the offline replays (`skills.label_run`, which a skills
build with runs labels by, and `failbench.candidates`): which frames the detector saw, what it was told between
them, and its clock. One copy, so the offline replays can't drift from what ran: `tests/test_replay.py` feeds
`policy.perception.Perception` and `LiveFeed` the same ticks and compares them.

`Perception` runs `LiveFeed` live, over each tick the model drove, and the replays run it offline, over the
recorded frames, as `Controller._drive_tick` decides what the perception sees:

- The detector sees each Autonomous frame the model drove or waited on (`control.driver` POLICY or WAITING,
  `live_masks`).
- After anything else had the arm, it's told `resumed(moved)` once, on the next frame it sees, where `moved`
  is whether anything moved the arm meanwhile: a Teleop, Rewinding or Returning frame, or a layer's playback
  (as the controller sets `attempt.moved`). A Pause and Resume taken in one tick leave no frame out of
  Autonomous, but the perception still hears that handback (`resumed(False)`): `analysis.Attempt.handbacks`
  finds them from the run's `state` events.
- Its clock is the model's: 1/fps per frame it sees, except the first after a handback, which adds nothing.
  The first frame of an attempt is at 0. Live, each tick fed counts 1/fps whatever the loop's real rate, as
  LeRobot stamps each recorded frame at frame_index / fps, so the two clocks agree by construction.
"""

import numpy as np

from so_101.control.recorder import Phase
from so_101.policy.base import PLAYBACK, POLICY, WAITING

# The phases in which something other than the model moved the arm (`Controller._drive_tick`'s `attempt.moved`).
MOVED_PHASES = (Phase.TELEOP, Phase.REWINDING, Phase.RETURNING)


def live_masks(phases: np.ndarray, drivers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: (fed, drove, moved). `fed`: the live detector saw it. `drove`: of those, the model drove it
    rather than waited, which only the stuck rules read. `moved`: a frame the detector didn't see in which
    something else moved the arm. `drivers` is `control.driver`."""
    phases, drivers = np.asarray(phases), np.asarray(drivers)
    autonomous = phases == Phase.AUTONOMOUS
    fed = autonomous & np.isin(drivers, (POLICY, WAITING))
    drove = autonomous & (drivers == POLICY)
    playback = autonomous & (drivers == PLAYBACK)
    moved = ~fed & (np.isin(phases, MOVED_PHASES) | playback)
    return fed, drove, moved


class LiveFeed:
    """One grasp detector, fed recorded frames as `policy.perception.Perception` feeds it live. Copyable with
    `copy.deepcopy`, so a replay can save it and go back (`skills.label_run`'s re-grip)."""

    def __init__(self, thresholds, fps: float) -> None:
        from so_101.grasp.detector import Detector

        self.detector = Detector(thresholds)
        self.fps = fps
        self.t = 0.0  # the model's seconds at the frame stepped last
        self._ticks = 0  # the model's clock in frames: counted, not summed, so 1/fps adds no rounding error
        self._pending: bool | None = None  # a handback to tell with the next frame: whether the arm moved
        self._stepped = False  # a frame was stepped since the last handback (or ever)

    def handback(self, moved: bool) -> None:
        """Something else had the arm before the next frame fed, and moved it if `moved`. Handbacks before one
        frame add up, a move winning, as `Perception.run` keeps `moved` until the detector hears it."""
        self._pending = moved or bool(self._pending)

    def resume(self, pose: dict[str, float]) -> bool | None:
        """Tell the detector the pending handback, if any, with the frame's pose. Returns whether the arm moved,
        or None with no handback pending. Call `step` next, with the same frame."""
        moved, self._pending = self._pending, None
        if moved is not None:
            self.detector.resumed(moved, pose)
            self._stepped = False
        return moved

    def step(self, pose: dict[str, float], drove: bool) -> list:
        """Step the detector on one frame, on the model's clock, and return its signals."""
        if self._stepped:
            self._ticks += 1
        self._stepped = True
        self.t = self._ticks / self.fps
        return self.detector.step(self.t, pose, drove)

    def feed(self, pose: dict[str, float], drove: bool) -> tuple[bool | None, list]:
        """`resume`, then `step`: what was told before the frame, and the frame's signals."""
        return self.resume(pose), self.step(pose, drove)
