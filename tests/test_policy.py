"""The policy stack's contract (docs/policy-layers.md), checked on every policy the control loop can step:
the model over a scripted engine, each layer over a stand-in model, and the full stack. Each layer's own
behaviour is tested beside it (test_skills, test_supervisor); this file checks what every one owes the loop.
"""

import types
import unittest

import numpy as np
from fakes import make_engine
from test_skills import FPS, NAMES, PROGRAM, SKILLS, FakeLeaf, trace
from test_supervisor import FakeEscalation, FakeModel, Pool

from so_101.policy import stack
from so_101.policy.base import (HOLD, LAYER, PERSON, PLAYBACK, POLICY, START, WAITING, Conditioning, Handback,
                                LayerContext, Output, Tick)
from so_101.policy.engine import EngineDriver
from so_101.policy.perception import Perception
from so_101.supervisor import judges as J

FRAME = np.zeros((48, 64, 3), np.uint8)


def judges():
    return J.Judges(PROGRAM, adapters={"jev": FakeModel({}), "gemini-3.8-flash": FakeModel({}),
                                       "claude-opus-5.5": FakeEscalation()}, pool=Pool())


def context(events: list) -> LayerContext:
    return LayerContext(lambda type_, **fields: events.append({"type": type_, **fields}), lambda: 0.0, PROGRAM)


def stacks():
    """(name, the policy, the leaf it ends in) for each policy under contract."""
    for name, kwargs in (("perception", {}), ("skills", {"skills": SKILLS}),
                         ("supervisor shadow", {"judges": judges(), "mode": "shadow"}),
                         ("supervisor act", {"judges": judges(), "mode": "act"}),
                         ("full stack", {"skills": SKILLS, "judges": judges(), "mode": "act"})):
        leaf = FakeLeaf()
        yield name, stack.build(leaf, context([]), cameras=("top", "wrist"), **kwargs), leaf


class Robot:
    def __init__(self) -> None:
        self.goals: list[dict] = []
        self.supports_hold = False

    def send_action(self, goal: dict) -> None:
        self.goals.append(dict(goal))


def tick(n: int, row) -> Tick:
    pose = dict(zip(NAMES, map(float, row), strict=True))
    return Tick(n / FPS, n, {**pose, "top": FRAME, "wrist": FRAME}, pose)


class ContractTest(unittest.TestCase):
    def test_every_call_the_loop_makes_reaches_the_model(self) -> None:
        for name, policy, leaf in stacks():
            with self.subTest(name):
                policy.begin_episode(1)
                policy.run(Handback(START))
                self.assertEqual(leaf.handbacks[-1], Handback(START), "a handback reaches the model as given")
                policy.condition(Conditioning("do it"))
                self.assertEqual(leaf.conditions[-1].prompt, "do it")
                policy.record_failure("the policy failed")
                self.assertEqual(policy.health().failure, "the policy failed")
                policy.pause()
                self.assertEqual(leaf.calls[-1], "pause")
                policy.run(Handback(PERSON, moved=True))
                self.assertEqual(leaf.handbacks[-1], Handback(PERSON, moved=True))
                policy.end_episode()

    def test_step_returns_an_output_and_passes_the_model_s_action_through(self) -> None:
        for name, policy, _ in stacks():
            with self.subTest(name):
                policy.begin_episode(1)
                policy.run(Handback(START))
                for n, row in enumerate(trace()[:30]):
                    out = policy.step(Robot(), tick(n, row))
                    self.assertIsInstance(out, Output)
                    self.assertIn(out.driver, (POLICY, WAITING, PLAYBACK, HOLD))
                    self.assertEqual(out.action, {"shoulder_pan.pos": float(row[0])}, "nothing to step in on yet")
                self.assertIsInstance(policy.maintain(), list)

    def test_nothing_reaches_the_arm_after_pause(self) -> None:
        for name, policy, leaf in stacks():
            with self.subTest(name):
                policy.begin_episode(1)
                policy.run(Handback(START))
                robot = Robot()
                for n, row in enumerate(trace()[:30]):
                    policy.step(robot, tick(n, row))
                policy.pause()
                before = len(robot.goals)
                for _ in range(10):
                    policy.hold(robot)  # what the loop does each tick it doesn't drive
                self.assertEqual(len(robot.goals), before, "a hold sends no goal of a layer's own")
                self.assertEqual(leaf.calls[-1], "pause")

    def test_the_layers_are_composed_in_one_order(self) -> None:
        _, top, leaf = list(stacks())[-1]
        order = []
        while hasattr(top, "inner"):
            order.append(top.name)
            top = top.inner
        self.assertEqual(order, ["supervisor", "skills", "perception"])
        self.assertIs(top, leaf)


class PerceptionTest(unittest.TestCase):
    def test_its_own_error_is_logged_once_and_the_model_drives_on(self) -> None:
        events = []
        leaf = FakeLeaf()
        layer = Perception(leaf, context(events))
        layer.begin_episode(1)
        layer.run(Handback(START))

        def broken(*args):
            raise RuntimeError("detector bug")

        layer.detector.step = broken
        outs = [layer.step(Robot(), tick(n, row)) for n, row in enumerate(trace()[:10])]
        self.assertEqual([e["error"] for e in events if e.get("kind") == "error"], ["RuntimeError: detector bug"])
        self.assertTrue(all(o.action is not None for o in outs), "every tick still drove")

    def test_a_layer_s_playback_or_hold_is_never_fed_to_the_detector(self) -> None:
        events = []

        class Holding(FakeLeaf):
            def step(self, robot, tick):
                return Output(None, HOLD)

        layer = Perception(Holding(), context(events))
        layer.begin_episode(1)
        layer.run(Handback(LAYER))
        out = layer.step(Robot(), tick(0, trace()[0]))
        self.assertIsNone(out.readings)
        self.assertEqual(layer.t, 0.0)


class EngineTest(unittest.TestCase):
    def test_the_model_meets_the_contract_too(self) -> None:
        engine = make_engine([[0.0] * 6] * 5)
        driver = EngineDriver(engine, NAMES, dataset_features={
            "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES}})
        robot = types.SimpleNamespace(supports_hold=False, send_action=lambda goal: None, hold=lambda: None)
        driver.begin_episode(1)
        driver.run(Handback(START))
        out = driver.step(robot, tick(0, trace()[0]))
        self.assertEqual(out.driver, POLICY)
        self.assertEqual(driver.maintain(), [])
        driver.record_failure("boom")
        self.assertEqual(driver.health().failure, "boom")
        engine.waiting = 3
        driver.record_failure(None)
        self.assertEqual(driver.step(robot, tick(1, trace()[0])).driver, WAITING, "a tick with no action is waiting")


if __name__ == "__main__":
    unittest.main()
