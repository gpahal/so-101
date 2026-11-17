"""`grasp.replay`, the offline copy of the live perception's feed, against `policy.perception.Perception` itself.

Both are fed one recorded attempt: the live layer as the controller and a supervisor would drive it (begin, a
START handback, pauses, a Teleop, a layer's playback and hold, a Pause and Resume in one tick), the replay from
the frames' phases and drivers as a run records them. They must tell their detectors the same handbacks at
the same frames, keep the same clock, and find the same signals. At 32 fps, so 1/32 s ticks are exact in
binary and the two clocks can be compared exactly: a drift of even one frame fails.
"""

import unittest

import numpy as np
from test_skills import GRASP, NAMES, PROGRAM, trace

from so_101.control.recorder import Phase
from so_101.grasp.detector import Detector
from so_101.grasp.replay import LiveFeed, live_masks
from so_101.policy.base import HOLD, LAYER, OTHER, PERSON, PLAYBACK, POLICY, START, WAITING, Handback, LayerContext, Output, Tick
from so_101.policy.perception import Perception

FPS = 32  # 1/32 is exact in binary


def timeline(length: int):
    """Per frame: the phase and driver a run records, and the frames a Pause and Resume in one tick came before."""
    phases = np.full(length, Phase.AUTONOMOUS, dtype=np.int8)
    drivers = np.full(length, POLICY, dtype=np.int8)
    phases[0:10], drivers[0:10] = Phase.STARTING, OTHER  # the move to the start pose
    phases[50:55], drivers[50:55] = Phase.PAUSED, OTHER  # a person's pause
    drivers[60] = WAITING  # the model sent nothing this tick
    phases[71:76], drivers[71:76] = Phase.TELEOP, OTHER  # a person's Teleop: the arm moved
    drivers[96:101] = PLAYBACK  # a layer plays back the model's motion, and hands the arm back itself
    drivers[151:156] = HOLD  # a layer holds the arm
    handbacks = np.zeros(length, dtype=bool)
    handbacks[120] = True  # a Pause and Resume taken in the tick that recorded frame 120
    return phases, drivers, handbacks


class Leaf:
    """The model: drives or waits as the script says."""

    def __init__(self) -> None:
        self.driver = POLICY

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def step(self, robot, tick):
        return Output({"x": 0.0} if self.driver == POLICY else None, self.driver)


def spy(detector: Detector, heard: list, at: list) -> None:
    """Record each `resumed` the detector hears, with the frame it heard it on (`at[0]`)."""
    resumed = detector.resumed

    def wrapped(moved, pose):
        heard.append((at[0], moved))
        return resumed(moved, pose)

    detector.resumed = wrapped


def live(states, phases, drivers, handbacks, tick_s: float = 1 / FPS, layers: list | None = None):
    """Drive a real Perception as the controller (and a layer above it) would, with the loop's clock advancing
    `tick_s` a tick, and the layer appended to `layers`. Returns (resumed calls, clock per fed frame, signals)."""
    leaf = Leaf()
    layer = Perception(leaf, LayerContext(lambda *args, **kwargs: None, lambda: 0.0, GRASP, PROGRAM, FPS))
    layer.begin_episode(1)
    if layers is not None:
        layers.append(layer)
    heard, clocks, signals, at = [], {}, [], [0]
    spy(layer.detector, heard, at)
    started, away, moved = False, False, False  # away: the stack doesn't have the arm
    for i, row in enumerate(states):
        at[0] = i
        pose = dict(zip(NAMES, map(float, row), strict=True))
        phase, driver = int(phases[i]), int(drivers[i])
        if phase == Phase.STARTING:
            continue
        if phase != Phase.AUTONOMOUS or driver in (PLAYBACK, HOLD):  # a person, or a layer above, has the arm
            if not away and started:
                layer.pause()
            away = True
            moved = moved or phase == Phase.TELEOP or driver == PLAYBACK
            continue
        if not started:
            layer.run(Handback(START, moved=False))
            started = True
        elif away:  # who handed back doesn't matter to the perception, only whether the arm moved
            layer.run(Handback(LAYER if int(drivers[i - 1]) in (PLAYBACK, HOLD) else PERSON, moved=moved))
        away, moved = False, False
        if handbacks[i]:
            layer.pause()
            layer.run(Handback(PERSON, moved=False))
        leaf.driver = driver
        out = layer.step(None, Tick(i * tick_s, i, pose, pose))
        clocks[i] = layer.t
        signals += [(i, s.kind, s.name) for s in out.signals]
    return heard, clocks, signals


def replayed(states, phases, drivers, handbacks):
    """The same attempt through `grasp.replay`, as the offline replays read a run."""
    feed = LiveFeed(GRASP, FPS)
    fed, drove, moved = live_masks(phases, drivers)
    heard, clocks, signals, at = [], {}, [], [0]
    spy(feed.detector, heard, at)
    for i, row in enumerate(states):
        at[0] = i
        if handbacks[i]:
            feed.handback(False)
        if not fed[i]:
            feed.handback(bool(moved[i]))
            continue
        _, found = feed.feed(dict(zip(NAMES, map(float, row), strict=True)), bool(drove[i]))
        clocks[i] = feed.t
        signals += [(i, s.kind, s.name) for s in found]
    return heard, clocks, signals


class SameTickHandbacksTest(unittest.TestCase):
    def test_only_a_pause_and_resume_in_one_tick_marks_a_frame(self) -> None:
        from so_101.control.analysis import same_tick_handbacks

        phases = np.array([Phase.AUTONOMOUS] * 4 + [Phase.PAUSED] * 2 + [Phase.AUTONOMOUS] * 4, dtype=np.int8)
        events = [
            {"type": "state", "previous": "autonomous", "state": "paused", "frame": 2},  # resumed in the same tick
            {"type": "state", "previous": "paused", "state": "autonomous", "frame": 2},
            {"type": "state", "previous": "autonomous", "state": "paused", "frame": 4},  # a pause that lasted: Paused
            {"type": "state", "previous": "teleop", "state": "autonomous", "frame": 7},  # not out of Autonomous
            {"type": "command", "previous": "autonomous", "frame": 8},  # not a state event
            {"type": "state", "previous": "autonomous", "frame": 10},  # past the last frame
        ]
        np.testing.assert_array_equal(np.flatnonzero(same_tick_handbacks(phases, events)), [2])


class LiveFeedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.states = trace()
        self.phases, self.drivers, self.handbacks = timeline(len(self.states))

    def test_the_replay_hears_what_the_perception_heard_on_the_same_clock(self) -> None:
        live_heard, live_clocks, live_signals = live(self.states, self.phases, self.drivers, self.handbacks)
        heard, clocks, signals = replayed(self.states, self.phases, self.drivers, self.handbacks)
        # The live START handback tells the fresh detector nothing moved, at the first frame the model drives; the
        # replay, which sees only the Starting frames before it, says the same.
        self.assertEqual(live_heard[0], (10, False))
        self.assertEqual(heard, live_heard, "the same handbacks, at the same frames")
        self.assertEqual([moved for _, moved in live_heard], [False, False, True, True, False, False],
                         "start, pause, Teleop, playback, same-tick handback, hold")
        self.assertEqual(clocks, live_clocks, "the same clock on every frame fed")
        self.assertEqual(signals, live_signals, "the same signals")
        kinds = {kind for _, kind, _ in live_signals}
        self.assertIn("close", kinds, "the timeline reaches a close, so signals are compared")

    def test_a_slow_loop_keeps_the_perception_on_the_replays_clock(self) -> None:
        # The loop ticks at 25 Hz while the task's fps is 32: the perception counts each tick fed as 1/fps, as the
        # replay counts each recorded frame (LeRobot stamps frame_index / fps), so neither the clock nor a signal
        # drifts. Summing the loop's real gaps, as the perception once did, made a 14 s rule fire 28% late.
        layers: list = []
        live_heard, live_clocks, live_signals = live(self.states, self.phases, self.drivers, self.handbacks,
                                                     tick_s=1 / 25, layers=layers)
        heard, clocks, signals = replayed(self.states, self.phases, self.drivers, self.handbacks)
        self.assertEqual(heard, live_heard)
        self.assertEqual(clocks, live_clocks, "the same clock on every frame fed")
        self.assertEqual(signals, live_signals, "the same signals")
        self.assertIn("close", {kind for _, kind, _ in live_signals})
        [perception] = layers
        ticks = len(live_clocks) - len(live_heard)  # every frame fed counts, but the first after each handback
        self.assertEqual(perception.feed.t, ticks / FPS)
        self.assertEqual(perception.t, max(clocks.values()))
        self.assertGreater(ticks, 100)

    def test_masks_follow_the_controllers_moved_rule(self) -> None:
        fed, drove, moved = live_masks(self.phases, self.drivers)
        self.assertFalse(fed[:10].any() or moved[:10].any(), "the move to the start pose isn't a move away")
        self.assertTrue(moved[71:76].all() and moved[96:101].all(), "Teleop and a playback moved the arm")
        self.assertFalse(moved[50:55].any() or moved[151:156].any(), "a pause or a layer's hold didn't")
        self.assertTrue(fed[60] and not drove[60], "a waiting tick is seen but not driven")
        old_fed, old_drove, _ = live_masks(self.phases, None, waiting=self.drivers != POLICY)
        np.testing.assert_array_equal(old_fed, self.phases == Phase.AUTONOMOUS)
        self.assertFalse(old_drove[60])

    def test_the_clock_adds_nothing_for_the_first_frame_after_a_handback(self) -> None:
        feed = LiveFeed(GRASP, FPS)
        pose = dict(zip(NAMES, map(float, self.states[0]), strict=True))
        times = []
        for _ in range(3):
            feed.feed(pose, True)
            times.append(feed.t)
        feed.handback(False)
        feed.feed(pose, True)
        times.append(feed.t)
        feed.feed(pose, True)
        times.append(feed.t)
        self.assertEqual(times, [0.0, 1 / FPS, 2 / FPS, 2 / FPS, 3 / FPS])


if __name__ == "__main__":
    unittest.main()
