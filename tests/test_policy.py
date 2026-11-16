"""The policy stack's contract (docs/policy-layers.md), checked on every policy the control loop can step:
the model over a scripted engine, each layer over a stand-in model, and the full stack. Each layer's own
behaviour is tested beside it (test_skills, test_supervisor); this file checks what every one owes the loop.
"""

import types
import unittest

import numpy as np
from fakes import make_engine
from test_skills import FPS, GRASP, NAMES, PROGRAM, SKILLS, FakeLeaf, trace
from test_supervisor import FakeEscalation, FakeModel, Pool

from so_101.policy import stack
from so_101.policy.base import (HOLD, LAYER, PERSON, PLAYBACK, POLICY, START, WAITING, Conditioning, Handback,
                                LayerContext, Output, Tick)
from so_101.policy.engine import EngineDriver
from so_101.policy.perception import Perception
from so_101.supervisor import judges as J

FRAME = np.zeros((48, 64, 3), np.uint8)


def judges(answers: dict | None = None):
    return J.Judges(PROGRAM, adapters={"jev": FakeModel(answers or {}), "gemini-3.8-flash": FakeModel(answers or {}),
                                       "claude-opus-5.5": FakeEscalation()}, pool=Pool())


def context(events: list) -> LayerContext:
    return LayerContext(lambda type_, **fields: events.append({"type": type_, **fields}), lambda: 0.0, GRASP, PROGRAM)


KINDS = (("perception", {}), ("skills", {"skills": SKILLS}),
         ("supervisor shadow", {"mode": "shadow"}),
         ("supervisor act", {"mode": "act"}),
         ("full stack", {"skills": SKILLS, "mode": "act"}))


def stacks(events: list | None = None, answers: dict | None = None):
    """(name, the policy, the leaf it ends in) for each policy under contract. Its events go to `events`;
    both judges answer `answers` (moment: p_wrong)."""
    for name, kwargs in KINDS:
        leaf = Leaf()
        if "mode" in kwargs:
            kwargs = {**kwargs, "judges": judges(answers)}
        yield name, stack.build(leaf, context(events if events is not None else []), cameras=("top", "wrist"),
                                **kwargs), leaf


def layers(policy) -> list:
    """The layers of a stack, top first."""
    found = []
    while hasattr(policy, "inner"):
        found.append(policy)
        policy = policy.inner
    return found


class Leaf(FakeLeaf):
    def stop(self) -> None:
        self.calls.append("stop")


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

    def test_stop_reaches_the_model(self) -> None:
        for name, policy, leaf in stacks():
            with self.subTest(name):
                policy.stop()
                self.assertEqual(leaf.calls[-1], "stop")

    def test_a_layer_s_own_error_never_escapes(self) -> None:
        """Each layer's own methods, one at a time, raise: no call the loop makes raises, the error is logged
        and announced once, and the model is still stepped. The judges call every close a miss, so the
        supervisor's playbacks, holds and asks run too."""
        for name, _, _ in stacks():
            cases = [(n, type(layer), method) for n, layer in enumerate(layers(dict(
                (k, p) for k, p, _ in stacks())[name])) for method in own_methods(type(layer))]
            for depth, kind, method in cases:
                with self.subTest(name, layer=kind.__name__, method=method):
                    events: list = []
                    policy, leaf = next((p, leaf) for k, p, leaf in stacks(events, {"after_close": 0.95,
                                                                                    "stuck": 0.95}) if k == name)
                    broken = layers(policy)[depth]

                    def raises(*args, **kwargs):
                        raise RuntimeError(f"{method} bug")

                    setattr(broken, method, raises)
                    said = drive(policy)
                    errors = [e for e in events if e.get("kind") == "error"]
                    self.assertLessEqual(len(errors), 1)
                    if errors:
                        self.assertEqual(errors[0]["type"], broken.name)
                        self.assertEqual(len([line for line in said if "stopped after an error" in line]), 1)
                    self.assertGreater(len(leaf.calls), 0)

    def test_the_layers_are_composed_in_one_order(self) -> None:
        _, top, leaf = list(stacks())[-1]
        order = []
        while hasattr(top, "inner"):
            order.append(top.name)
            top = top.inner
        self.assertEqual(order, ["supervisor", "skills", "perception"])
        self.assertIs(top, leaf)


def own_methods(kind) -> list[str]:
    """The private methods a layer class defines itself: its own logic."""
    return sorted(name for name, value in vars(kind).items()
                  if name.startswith("_") and not name.startswith("__") and callable(value))


def drive(policy, ticks: int = 300) -> list[str]:
    """Step `policy` through a demo's trace as the loop does, pausing it on an ask and resuming it a few ticks
    later, then end the attempt. Returns what `maintain` said."""
    said: list[str] = []
    robot, rows = Robot(), trace()
    policy.start()
    policy.begin_episode(1)
    policy.run(Handback(START))
    policy.condition(Conditioning("do it"))
    paused = 0
    for n in range(ticks):
        said += policy.maintain()
        policy.health()
        if paused:
            policy.hold(robot)
            paused -= 1
            if not paused:
                policy.run(Handback(PERSON, moved=True))
            continue
        out = policy.step(robot, tick(n, rows[min(n, len(rows) - 1)]))
        if out.asks is not None:
            policy.pause()
            paused = 5
    policy.pause()
    said += policy.maintain()
    policy.end_episode()
    said += policy.maintain()
    policy.stop()
    policy.close()
    return said


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

    def test_a_moved_handback_isnt_lost_to_a_later_one_before_the_next_fed_tick(self) -> None:
        layer = Perception(FakeLeaf(), context([]))
        layer.begin_episode(1)
        layer.run(Handback(START))
        rows = trace()
        for n, row in enumerate(rows[:5]):
            layer.step(Robot(), tick(n, row))
        heard = []
        layer.detector.resumed = lambda moved, pose: heard.append(moved)
        layer.pause()
        layer.run(Handback(PERSON, moved=True))  # a Rewind, then Resume and Pause before any fed tick
        layer.pause()
        layer.run(Handback(PERSON, moved=False))  # the next Resume: nothing moved since the last one
        layer.step(Robot(), tick(5, rows[5]))
        self.assertEqual(heard, [True], "the detector hears the arm moved since it last saw it")

    def test_broken_in_a_skills_run_it_refuses_resume_and_says_so_once(self) -> None:
        for skills in ((), SKILLS):
            with self.subTest(skills=bool(skills)):
                events = []
                policy = stack.build(FakeLeaf(), context(events), skills=skills)
                perception = layers(policy)[-1]
                policy.begin_episode(1)
                policy.run(Handback(START))

                def broken(*args):
                    raise RuntimeError("detector bug")

                perception.detector.step = broken
                policy.step(Robot(), tick(0, trace()[0]))
                said = policy.maintain() + policy.maintain()
                self.assertEqual(len([line for line in said if line.startswith("Perception layer: stopped")]), 1)
                failure = policy.health().failure
                if skills:
                    self.assertIn("perception layer stopped after an error: RuntimeError: detector bug", failure)
                else:
                    self.assertIsNone(failure, "without skills the model drives on without it")

    def test_a_broken_skill_sequencer_refuses_resume(self) -> None:
        policy = stack.build(FakeLeaf(), context([]), skills=SKILLS)
        sequencer = layers(policy)[0]

        def broken(*args):
            raise RuntimeError("handover bug")

        sequencer._handover = broken
        policy.begin_episode(1)
        policy.run(Handback(START))
        policy.step(Robot(), tick(0, trace()[0]))
        self.assertIn("skills layer stopped", policy.health().failure)

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
