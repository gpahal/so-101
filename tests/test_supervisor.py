"""The supervisor without hardware: the program, the detector on made-up traces, the judges with stand-in
models, the layer's decisions in shadow and act mode, the layer as serve's policy on the fake arm, and
failbench's candidates from a run's columns.

The traces are 30 Hz, in the follower's units (degrees, gripper 0 to 100). The thresholds are the
program's own; the tests check what the rules do, not that the numbers are right for the arm.
"""

import dataclasses
import os
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np
from test_serve_preflight import ARMS, ConfigTest

from so_101 import settings
from so_101.failbench import candidates
from so_101.supervisor import judges as J
from so_101.supervisor import program as P
from so_101.policy import stack
from so_101.policy.base import PERSON, PLAYBACK, POLICY, START, WAITING, Handback, Health, LayerContext, Output, Tick
from so_101.grasp import thresholds as T
from so_101.grasp.detector import ARM, GRIPPER, Detector

FPS = 30
PROGRAM = P.load()
GRASP = T.load()
NAMES = [*ARM, GRIPPER]


def pose(gripper=40.0, elbow=0.0, pan=0.0) -> dict[str, float]:
    return {"shoulder_pan.pos": pan, "shoulder_lift.pos": 0.0, "elbow_flex.pos": elbow, "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0, GRIPPER: gripper}


class Trace:
    """Builds a 30 Hz trace segment by segment, each moving linearly from the last pose."""

    def __init__(self, start: dict | None = None) -> None:
        self.poses = [dict(start) if start is not None else pose()]
        self.drove: list[bool] = [True]

    def to(self, seconds: float, drove: bool = True, **target) -> "Trace":
        start = self.poses[-1]
        end = {**start, **{({"gripper": GRIPPER, "elbow": "elbow_flex.pos", "pan": "shoulder_pan.pos"}[k]): v
                           for k, v in target.items()}}
        n = max(1, round(seconds * FPS))
        for i in range(1, n + 1):
            self.poses.append({k: start[k] + (end[k] - start[k]) * i / n for k in start})
            self.drove.append(drove)
        return self

    def run(self, detector: Detector) -> list:
        out = []
        for i, (p, drove) in enumerate(zip(self.poses, self.drove, strict=True)):
            out += detector.step(i / FPS, p, drove)
        return out


def kinds(signals) -> list[str]:
    return [s.kind if s.kind != "rule" else f"rule:{s.name}" for s in signals]


def grasp() -> Trace:
    """Reach for 3 s, close on the block, carry it with the elbow, release over the jar."""
    return Trace().to(3, pan=20).to(1, gripper=18).to(2, elbow=40).to(1, gripper=40)


class ProgramTest(unittest.TestCase):
    def test_the_default_program_and_thresholds_load(self) -> None:
        self.assertEqual(GRASP.rules.regrasps, 2)
        self.assertEqual(GRASP.gripper.version, 1)
        self.assertEqual(PROGRAM.judges.decision, "jev")

    def test_a_bad_threshold_names_itself(self) -> None:
        text = T.DEFAULT.read_text()
        cases = {
            "hold band over the close threshold": (text.replace("hold_band = [16.5, 22.0]", "hold_band = [16.5, 26.0]"),
                                                   "hold_band high < closed_below"),
            "a negative threshold": (text.replace("seconds = 14.0", "seconds = -1"), "rules.no_carry.seconds"),
            "a missing section": (text.replace("[rules.chatter]", "[rules.chattering]"), r"\[rules.chatter\]"),
            "a fraction of a regrasp": (text.replace("regrasps = 2 ", "regrasps = 1.5 "), "regrasps"),
            "no gripper version": (text.replace("[gripper]\nversion = 1", "[gripper]"), "gripper.version"),
            "an infinite threshold": (text.replace("seconds = 14.0", "seconds = inf"), "rules.no_carry.seconds.*finite"),
            "not a number": (text.replace("hold_s = 1.5", "hold_s = nan"), "gripper.hold_s.*finite"),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, (changed, message) in cases.items():
                with self.subTest(name):
                    path = Path(folder) / "grasp.toml"
                    path.write_text(changed)
                    with self.assertRaisesRegex(T.ThresholdsError, message):
                        T.load(path)

    def test_a_bad_program_value_names_itself(self) -> None:
        text = P.DEFAULT.read_text()
        cases = {
            "an unknown judge": (text.replace('decision = "jev"', 'decision = "nope"'), "judges.decision"),
            "a missing section": (text.replace("[watch]", "[watching]"), r"\[watch\]"),
            "a negative lookback": (text.replace("lookback_s = 2.0", "lookback_s = -1"), "watch.lookback_s"),
            "an infinite budget": (text.replace("max_cost_usd = 2.0", "max_cost_usd = inf"), "judges.max_cost_usd.*finite"),
            "one model in two roles": (text.replace('chat = "gemini-3.8-flash"', 'chat = "jev"'),
                                       "judges.chat.*judges.decision"),
            "a lookback longer than the frames kept": (text.replace("lookback_s = 2.0", "lookback_s = 9.0"),
                                                       "watch.lookback_s.*4 s"),
            "a first playback that can't reach the close": (
                text.replace("after_close_answer_s = 1.5", "after_close_answer_s = 3.0"), "miss_rewind_s"),
            "no hold deadline": (text.replace("hold_deadline_s = 5.0", ""), "timing.hold_deadline_s"),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, (changed, message) in cases.items():
                with self.subTest(name):
                    path = Path(folder) / "program.toml"
                    path.write_text(changed)
                    with self.assertRaisesRegex(P.ProgramError, message):
                        P.load(path)


    def test_a_bad_program_or_threshold_is_a_config_error(self) -> None:
        self.assertTrue(issubclass(P.ProgramError, settings.ConfigError))
        self.assertTrue(issubclass(T.ThresholdsError, settings.ConfigError))


class DetectorTest(unittest.TestCase):
    def test_a_clean_grasp_closes_carries_and_releases_with_no_rule(self) -> None:
        detector = Detector(GRASP)
        self.assertEqual(kinds(grasp().run(detector)), ["close", "carry", "release"])
        self.assertEqual(detector.fired, set())

    def test_two_misses_fire_the_regrasp_loop(self) -> None:
        trace = Trace().to(2, pan=20)
        for _ in range(2):  # closes on nothing and opens again, the elbow still, slowly enough not to chatter
            trace.to(0.5, gripper=10, pan=25).to(0.7, pan=30).to(0.5, gripper=40, pan=25).to(0.5, pan=20)
        self.assertEqual(kinds(trace.run(Detector(GRASP))),
                         ["close", "regrasp", "close", "regrasp", "rule:regrasp_loop"])

    def test_one_miss_is_not_stuck(self) -> None:
        trace = Trace().to(2, pan=20).to(0.5, gripper=10).to(0.5, gripper=40).to(1, gripper=18).to(2, elbow=40)
        self.assertEqual(kinds(trace.run(Detector(GRASP))), ["close", "regrasp", "close", "carry"])

    def test_creeping_forward_open_away_from_the_start_is_a_hover(self) -> None:
        trace = Trace().to(2, pan=20).to(6, pan=50)  # 5 deg/s: slower than a reach, faster than frozen
        signals = trace.run(Detector(GRASP))
        self.assertEqual(kinds(signals), ["rule:hover"])
        self.assertAlmostEqual(signals[0].t, 7.0, delta=0.2)

    def test_a_flickering_gripper_is_chatter(self) -> None:
        trace = Trace().to(2, pan=20, gripper=30)
        for _ in range(3):
            trace.to(0.3, gripper=20).to(0.3, gripper=30)
        self.assertIn("rule:chatter", kinds(trace.run(Detector(GRASP))))

    def test_standing_still_is_frozen_but_waiting_for_the_policy_is_not(self) -> None:
        self.assertEqual(kinds(Trace().to(2, pan=20).to(4.5).run(Detector(GRASP))), ["rule:frozen"])
        self.assertEqual(kinds(Trace().to(2, pan=20).to(4.5, drove=False).run(Detector(GRASP))), [])

    def test_nothing_carried_in_14_s_of_driving(self) -> None:
        signals = Trace().to(14.5, pan=200).run(Detector(GRASP))  # 14 deg/s: never slow enough to hover
        self.assertEqual(kinds(signals), ["rule:no_carry"])
        self.assertAlmostEqual(signals[0].t, 14.0, delta=0.1)

    def test_holding_a_carried_block_for_10_s_is_not_released(self) -> None:
        trace = Trace().to(2, pan=20).to(1, gripper=18).to(2, elbow=40).to(11, pan=60)
        self.assertEqual(kinds(trace.run(Detector(GRASP))), ["close", "carry", "rule:not_released"])

    def test_after_a_release_nothing_is_a_grasp_or_stuck(self) -> None:
        trace = grasp().to(1, gripper=10).to(20)  # closes on the way back to rest, then sits
        self.assertEqual(kinds(trace.run(Detector(GRASP))), ["close", "carry", "release"])

    def test_a_resume_restarts_the_running_timers(self) -> None:
        detector = Detector(GRASP)
        trace = Trace().to(2, pan=20).to(3)
        out = trace.run(detector)
        detector.resumed(True, trace.poses[-1])
        t0 = len(trace.poses) / FPS
        for i in range(int(3 * FPS)):
            out += detector.step(t0 + i / FPS, trace.poses[-1])
        self.assertEqual(kinds(out), [], "3 s still, a rewind, then 3 s still: never 4 s in a row")

    def test_waiting_for_the_policy_never_counts_as_a_hover(self) -> None:
        # Reached out with the jaws open, then 6 s with no action from the model (a late remote chunk).
        self.assertEqual(kinds(Trace().to(2, pan=20).to(6, drove=False).run(Detector(GRASP))), [])

    def test_a_close_undone_while_something_else_had_the_arm_is_not_a_regrasp(self) -> None:
        detector = Detector(GRASP)
        trace = Trace().to(2, pan=20).to(0.5, gripper=10).to(1)
        out = trace.run(detector)
        opened = pose(gripper=40.0, pan=15.0)  # a playback opened the jaws and moved back
        detector.resumed(True, opened)
        t0 = len(trace.poses) / FPS
        for i in range(30):
            out += detector.step(t0 + i / FPS, opened)
        self.assertEqual(kinds(out), ["close"])
        self.assertEqual(detector.regrasps, 0)

    def test_a_second_full_grasp_after_a_handback_with_the_jaws_open(self) -> None:
        detector = Detector(GRASP)
        out = grasp().run(detector)
        self.assertEqual(kinds(out), ["close", "carry", "release"])
        detector.resumed(True, pose(gripper=40.0, pan=20.0))  # a person put the block back and handed the arm back
        again = Trace(pose(gripper=40.0, pan=20.0)).to(1, gripper=18).to(2, elbow=40).to(1, gripper=40)
        t0 = len(grasp().poses) / FPS
        for i, (p, drove) in enumerate(zip(again.poses, again.drove, strict=True)):
            out += detector.step(t0 + i / FPS, p, drove)
        self.assertEqual(kinds(out), ["close", "carry", "release", "close", "carry", "release"])

    def test_a_handback_that_moved_nothing_keeps_what_was_seen(self) -> None:
        detector = Detector(GRASP)
        trace = grasp()
        trace.run(detector)
        detector.resumed(False, trace.poses[-1])  # a hold, the arm where it was
        self.assertEqual(kinds(Trace(trace.poses[-1]).to(1, gripper=10).to(1).run(detector)), [],
                         "still released: a close on the way home isn't a grasp")

    def test_a_handback_with_the_jaws_closed_keeps_the_close_to_come(self) -> None:
        detector = Detector(GRASP)
        trace = Trace().to(2, pan=20)
        out = trace.run(detector)
        closed = pose(gripper=18.0, pan=20.0)  # a person closed the jaws on the block in Teleop
        detector.resumed(True, closed)
        t0 = len(trace.poses) / FPS
        for i in range(3):
            out += detector.step(t0 + i / FPS, closed)
        self.assertEqual(kinds(out), ["close"])

    def test_progress_tells_the_watch_when_a_rule_is_near(self) -> None:
        detector = Detector(GRASP)
        Trace().to(2, pan=20).to(3).run(detector)
        self.assertEqual(detector.near(PROGRAM.watch.near), ["frozen"])


class FakeModel:
    """A judge whose p_wrong per moment is scripted; it records what it was asked."""

    def __init__(self, answers: dict[str, float], cost: float = 0.001) -> None:
        self.answers, self.cost, self.asked = answers, cost, []

    def predict(self, images, moment):
        self.asked.append((moment, [name for name, _ in images]))
        return {"p_wrong": self.answers.get(moment, 0.05), "category": "none", "cost_usd": self.cost, "latency_s": 0.1}


class FakeEscalation:
    def __init__(self) -> None:
        self.calls = 0

    def escalate(self, sheet, log):
        self.calls += 1
        return {"choice": "teleop", "reason": "block at the jaw's tip", "observation": "", "cost_usd": 0.02}


class Hang(BaseException):
    """Raised by a stand-in model whose call never answers: `Pool` leaves its future pending."""


class Hung:
    """A judge whose every call hangs."""

    def __init__(self) -> None:
        self.asked = []

    def predict(self, images, moment):
        self.asked.append((moment, [name for name, _ in images]))
        raise Hang


class Raising:
    def predict(self, images, moment):
        raise RuntimeError("the provider's SDK broke")


class Pool:
    """Runs each call when `run` is called, or at once with `now`. A call that raises `Hang` never answers."""

    def __init__(self, now: bool = True) -> None:
        self.now, self.waiting, self.hung = now, [], []
        self.closed = False

    def submit(self, fn, *args):
        future = Future()
        if self.now:
            self._answer(future, fn, args)
        else:
            self.waiting.append((future, fn, args))
        return future

    def _answer(self, future, fn, args) -> None:
        try:
            future.set_result(fn(*args))
        except Hang:
            self.hung.append(future)

    def run(self, n: int | None = None) -> None:
        """Answer the first `n` calls waiting, or all of them."""
        n = len(self.waiting) if n is None else n
        waiting, self.waiting = self.waiting[:n], self.waiting[n:]
        for future, fn, args in waiting:
            self._answer(future, fn, args)

    def shutdown(self, **kwargs) -> None:
        self.closed = True


def judges(decision: dict[str, float], chat: dict[str, float] | None = None, program=PROGRAM, pool=None,
           models: dict | None = None):
    """Stand-in judges: the decision model answers `decision`, the chat model `chat` or, without it, the
    same. `models` replaces any of them by key; None leaves that one out."""
    adapters = {"jev": FakeModel(decision), "gemini-3.8-flash": FakeModel(decision if chat is None else chat),
                "claude-opus-5.5": FakeEscalation(), **(models or {})}
    return J.Judges(program, adapters={k: v for k, v in adapters.items() if v is not None}, pool=pool or Pool())


class JudgesTest(unittest.TestCase):
    def test_the_decision_model_stands_when_sure_else_the_chat_model(self) -> None:
        v = lambda p: J.Verdict("x", p_wrong=p)  # noqa: E731
        self.assertEqual(J.combine(PROGRAM, v(0.9), v(0.1)), J.WRONG)
        self.assertEqual(J.combine(PROGRAM, v(0.5), v(0.1)), J.OK)
        self.assertEqual(J.combine(PROGRAM, v(0.5), v(0.5)), J.UNSURE)
        self.assertEqual(J.combine(PROGRAM, J.Verdict("x", error="down"), v(0.95)), J.WRONG)

    def test_success_needs_both_judges_sure(self) -> None:
        v = lambda p: J.Verdict("x", p_wrong=p)  # noqa: E731
        self.assertTrue(J.success(PROGRAM, v(0.02), v(0.05)))
        self.assertFalse(J.success(PROGRAM, v(0.02), v(0.3)))
        self.assertFalse(J.success(PROGRAM, v(0.02), J.Verdict("x", error="down")))

    def test_checks_stop_at_the_cost_cap(self) -> None:
        program = dataclasses.replace(PROGRAM, judges=dataclasses.replace(PROGRAM.judges, max_cost_usd=0.0025))
        j = judges({}, program=program)
        answers = [j.ask("jev", [("top", b"x")], "end") for _ in range(4)]
        self.assertEqual([a.error is None for a in answers], [True, True, True, False])
        self.assertIn("max_cost_usd", answers[-1].error)

    def test_a_missing_judge_answers_with_an_error_not_an_exception(self) -> None:
        j = J.Judges(PROGRAM, adapters={}, pool=Pool())
        self.assertEqual(j.ask("jev", [], "end").error, "not available")


FRAME = np.zeros((48, 64, 3), np.uint8)


class FakeDriver:
    """The model the stack wraps: each step reports an action, or none on a waiting tick."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.handbacks: list = []
        self.steps = self.holds = 0
        self.failure: str | None = None
        self.drove = True
        self.error: Exception | None = None

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def close(self) -> None:
        pass

    def begin_episode(self, number: int) -> None:
        pass

    def end_episode(self) -> None:
        pass

    def maintain(self) -> list[str]:
        return []

    def pause(self) -> None:
        self.calls.append("pause")

    def run(self, handback) -> None:
        self.calls.append("run")
        self.handbacks.append(handback)

    def hold(self, robot) -> None:
        self.holds += 1

    def condition(self, conditioning) -> None:
        pass

    def health(self) -> Health:
        return Health(self.failure)

    def record_failure(self, detail: str) -> None:
        self.failure = detail

    def step(self, robot, tick) -> Output:
        if self.error is not None:
            raise self.error
        self.steps += 1
        if not self.drove:
            return Output(None, WAITING)
        return Output({"shoulder_pan.pos": tick.observation["shoulder_pan.pos"]}, POLICY)


class FakeRobot:
    def __init__(self) -> None:
        self.goals: list[dict] = []

    def send_action(self, goal: dict) -> None:
        self.goals.append(dict(goal))


class Harness:
    """The layer stepped by hand as serve steps it: maintain, then step, then asks. The trace's poses are
    the policy's, taken only on ticks the policy drives; on the layer's own ticks the arm is where the
    layer last sent it. When the layer asks for a person, the harness pauses it, as serve does."""

    def __init__(self, mode: str, decision: dict[str, float], chat=None, pool=None, program=PROGRAM,
                 models: dict | None = None) -> None:
        self.logged: list[dict] = []
        self.said: list[str] = []
        self.asked: list[str] = []
        self.ask_kinds: list[str] = []
        self.judges = judges(decision, chat, program=program, pool=pool, models=models)
        self.driver, self.robot = FakeDriver(), FakeRobot()
        self.now, self.frame, self.pose = 0.0, -1, pose()
        self.rest: dict | None = None  # where Tick.at_rest is true, within 1 deg
        self.ignore_asks = False  # step on after an ask, as if nobody paused
        ctx = LayerContext(lambda type_, **fields: self.logged.append({"type": type_, **fields}), lambda: self.now,
                           GRASP, program)
        self.layer = stack.build(self.driver, ctx, judges=self.judges, mode=mode, cameras=("top", "wrist"))
        self.layer.begin_episode(1)
        self.layer.run(Handback(START))
        self.paused = False

    def feed(self, trace: Trace, until=None, limit: int = 3000) -> None:
        """Step through `trace`, or until `until()` is true after a step."""
        moves = list(zip(trace.poses, trace.drove, strict=True))
        i = 0
        while i < len(moves) and limit and not self.paused and not (until and until()):
            limit -= 1
            self.now += 1 / FPS
            self.frame += 1
            self.said.extend(self.layer.maintain())
            if self.layer.doing == "pass":
                p, self.driver.drove = moves[i]
            else:
                p = self.robot.goals[-1] if self.robot.goals else self.pose
            steps = self.driver.steps
            rest = self.rest is not None and max(abs(p[k] - self.rest[k]) for k in p) <= 1.0
            out = self.layer.step(self.robot, Tick(self.now, self.frame, {**p, "top": FRAME, "wrist": FRAME}, dict(p),
                                                   at_rest=rest))
            self.pose = p
            if self.driver.steps > steps:
                i += 1
            if out.asks is not None:
                self.asked.append(out.asks.text)
                self.ask_kinds.append(out.asks.kind)
                if not self.ignore_asks:
                    self.layer.pause()
                    self.paused = True

    def settle(self, ticks: int = 3 * FPS) -> None:
        """The policy stands still wherever the arm is, while the layer finishes what it started."""
        for _ in range(ticks):
            if self.paused:
                return
            self.feed(_still(self.pose, 1), limit=1)

    def through_playback(self) -> None:
        """Step until the layer hands the arm back, the policy standing still where the arm is."""
        for _ in range(30 * FPS):
            if self.paused or self.layer.doing == "pass":
                return
            self.feed(_still(self.pose, 1), limit=1)

    def here(self) -> Trace:
        """A trace starting where the arm is now."""
        return Trace(self.pose)

    def rewinding(self) -> bool:
        return self.layer.doing == "rewinding"

    def holding(self) -> bool:
        return self.layer.doing == "holding"

    def resume(self) -> None:
        self.paused = False
        self.layer.run(Handback(PERSON, moved=True))

    def kinds(self, kind: str) -> list[dict]:
        """The supervisor's events of `kind`, or the events of type `kind` (playback, grasp, rule)."""
        return [e for e in self.logged if (e["type"] == "supervisor" and e.get("kind") == kind) or e["type"] == kind]

    def decisions(self) -> list[str]:
        return [d["decision"] for d in self.kinds("decision")]


def _still(p: dict, ticks: int) -> Trace:
    trace = Trace()
    trace.poses, trace.drove = [dict(p)] * ticks, [True] * ticks
    return trace


def miss() -> Trace:
    """Reach, then close on nothing and stay closed long enough for the after_close check."""
    return Trace().to(2, pan=20).to(0.5, gripper=10).to(1.5)


class LayerTest(unittest.TestCase):
    def test_shadow_mode_decides_and_logs_but_every_step_is_the_policy_s(self) -> None:
        h = Harness("shadow", {"after_close": 0.95})
        trace = miss()
        h.feed(trace)
        self.assertEqual(h.decisions(), ["miss 1: rewind 2.5 s and retry"])
        self.assertFalse(h.kinds("decision")[0]["acted"])
        self.assertEqual(h.driver.calls, ["run"], "never paused")
        self.assertEqual(h.driver.steps, len(trace.poses))
        self.assertEqual((h.robot.goals, h.said, h.asked), ([], [], []))

    def test_a_miss_plays_back_the_policy_s_motion_then_hands_the_arm_back(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(miss(), until=h.rewinding)
        h.settle()
        self.assertEqual(h.driver.calls, ["run", "pause", "run"])
        [rewind] = h.kinds("playback")
        self.assertLess(rewind["start_frame"], rewind["end_frame"])
        self.assertAlmostEqual(rewind["seconds"], 2.5, delta=0.05)
        self.assertEqual(rewind["cause"], "arrived")
        self.assertEqual(h.robot.goals[-1][GRIPPER], 40.0, "back to before the close: the jaws open")
        self.assertTrue(any("missed grasp 1" in line for line in h.said))

    def test_a_playback_s_ticks_are_never_the_policy_s_action(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(miss(), until=h.rewinding)
        self.assertEqual(h.layer.doing, "rewinding")
        out = h.layer.step(h.robot, Tick(h.now, h.frame, {**h.robot.goals[-1], "top": FRAME, "wrist": FRAME}, h.robot.goals[-1]))
        self.assertEqual((out.action, out.driver), (None, PLAYBACK))

    def test_the_second_miss_plays_back_further(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(miss(), until=h.rewinding)
        h.settle()
        h.feed(Trace().to(0.5, gripper=10).to(1.5), until=h.rewinding)  # closes on nothing again
        h.settle()
        self.assertEqual([d.get("rewind_s") for d in h.kinds("decision")], [2.5, 5.0])

    def test_an_ok_close_does_nothing(self) -> None:
        h = Harness("act", {"after_close": 0.05})
        h.feed(miss())
        self.assertEqual(h.driver.calls, ["run"])
        self.assertEqual(h.kinds("check")[0]["call"], "ok")

    def test_a_miss_needs_both_judges_sure_it_is_wrong(self) -> None:
        cases = ((0.95, 0.95, "wrong"), (0.95, 0.5, "unsure"), (0.5, 0.95, "unsure"), (0.95, 0.05, "unsure"),
                 (0.5, 0.05, "ok"))
        for decision, chat, call in cases:
            with self.subTest(decision=decision, chat=chat):
                h = Harness("act", {"after_close": decision}, chat={"after_close": chat})
                h.feed(miss(), until=h.rewinding)
                self.assertEqual(h.rewinding(), call == "wrong")
                [check] = [c for c in h.kinds("check") if c["moment"] == "after_close"]
                self.assertEqual(check["call"], call)

    def test_a_judge_that_raises_is_no_answer_and_plays_nothing_back(self) -> None:
        h = Harness("act", {"after_close": 0.95}, models={"gemini-3.8-flash": Raising()})
        h.feed(miss())
        self.assertEqual(h.driver.calls, ["run"])
        [check] = [c for c in h.kinds("check") if c["moment"] == "after_close"]
        self.assertIn("the provider's SDK broke", check["chat"]["error"])

    def test_a_sure_ok_doesn_t_wait_for_a_judge_that_hangs(self) -> None:
        h = Harness("act", {"after_close": 0.05}, models={"gemini-3.8-flash": Hung()})
        h.feed(Trace().to(2, pan=20).to(0.5, gripper=10).to(1.1))
        [check] = [c for c in h.kinds("check") if c["moment"] == "after_close"]
        self.assertEqual((check["call"], check["pending"]), ("ok", ["chat"]))

    def test_a_sure_wrong_waiting_on_a_judge_that_hangs_is_dropped_in_time(self) -> None:
        h = Harness("act", {"after_close": 0.95}, models={"gemini-3.8-flash": Hung()})
        h.feed(miss().to(2))
        self.assertEqual(h.driver.calls, ["run"], "never played back")
        [check] = [c for c in h.kinds("check") if c["moment"] == "after_close"]
        self.assertEqual(check["stale"], "too_old")
        self.assertAlmostEqual(check["waited_s"], PROGRAM.after_close_answer_s, delta=0.1)

    def test_the_third_miss_asks_for_a_person_with_no_rule_on_the_way(self) -> None:
        h = Harness("act", {"after_close": 0.95, "stuck": 0.95})
        for n in range(3):  # reach on, then close on nothing while still creeping, from wherever the arm was left
            h.feed(h.here().to(1, pan=h.pose["shoulder_pan.pos"] + 12).to(0.5, gripper=10, pan=h.pose["shoulder_pan.pos"] + 15)
                   .to(1.5, pan=h.pose["shoulder_pan.pos"] + 22), until=lambda: h.rewinding() or h.holding())
            h.through_playback()
        self.assertEqual(h.decisions()[:3], ["miss 1: rewind 2.5 s and retry", "miss 2: rewind 5 s and retry",
                                             "hold: miss 3"])
        self.assertEqual([e["name"] for e in h.kinds("rule")], [], "no stuck rule fired first")
        self.assertEqual([e["grasp"] for e in h.kinds("grasp")], ["close"] * 3, "a close it undid is no regrasp")
        self.assertEqual(h.asked, ["Supervisor: stuck (miss 3). Paused for you; asking for advice."])

    def test_a_regrasp_loop_is_left_to_the_miss_ladder_while_the_judges_answer(self) -> None:
        def regrasps(h) -> None:
            h.feed(Trace().to(2, pan=20))
            for _ in range(2):  # closes on nothing and the policy opens again itself, the checks unsure
                h.feed(h.here().to(0.5, gripper=10, pan=25).to(1.2, pan=32).to(0.5, gripper=40, pan=26).to(0.5, pan=20))
        h = Harness("act", {"after_close": 0.5, "stuck": 0.95})
        regrasps(h)
        self.assertEqual([e["name"] for e in h.kinds("rule")], ["regrasp_loop"])
        self.assertEqual(h.decisions(), [])
        self.assertEqual([e["rule"] for e in h.kinds("ignored")], ["regrasp_loop"])
        h = Harness("act", {"after_close": 0.5, "stuck": 0.95}, models={"gemini-3.8-flash": None})  # one judge can't answer
        regrasps(h)
        self.assertEqual(h.decisions()[0], "hold: regrasp_loop")

    def test_a_second_full_grasp_after_a_playback(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(miss(), until=h.rewinding)
        h.through_playback()
        h.layer.judges._adapters["jev"].answers["after_close"] = 0.05
        start = len(h.kinds("grasp"))
        h.feed(h.here().to(1, gripper=18).to(2, elbow=40).to(1, gripper=40).to(0.5))
        self.assertEqual([e["grasp"] for e in h.kinds("grasp")[start:]], ["close", "carry", "release"])
        self.assertEqual(h.kinds("grasp")[start]["close"], 2)
        self.assertEqual(h.decisions(), ["miss 1: rewind 2.5 s and retry"])

    def test_a_playback_never_goes_back_past_where_a_person_handed_the_arm_back(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(Trace().to(2, pan=20))
        h.layer.pause()  # serve's Pause: a person may move the arm now
        h.layer.run(Handback(PERSON, moved=True))
        handed_back = h.frame + 1
        h.feed(Trace().to(0.3, gripper=10).to(1.5), until=h.rewinding)  # 1.2 s since, under the 2.5 s asked for
        h.settle()
        [rewind] = h.kinds("playback")
        self.assertEqual(rewind["start_frame"], handed_back)

    def test_a_pause_cuts_a_playback_short_and_logs_what_it_undid(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(miss(), until=h.rewinding)
        h.feed(_still(h.pose, 10), limit=10)  # ten ticks of the playback
        h.layer.pause()
        [rewind] = h.kinds("playback")
        self.assertEqual(rewind["cause"], "paused")
        self.assertLess(rewind["seconds"], 2.5)
        self.assertEqual(h.layer.doing, "pass")

    def test_a_stuck_rule_holds_the_arm_and_a_sure_ok_watch_check_hands_it_back(self) -> None:
        pool = Pool(now=False)  # the watch check takes its time
        h = Harness("act", {"stuck": 0.05}, pool=pool)
        h.feed(Trace().to(2, pan=20).to(4.5), until=lambda: h.layer.doing == "holding")  # frozen
        h.feed(_still(h.pose, 5), limit=5)
        self.assertEqual(h.driver.holds, 6, "the arm holds on every tick until the answer")
        pool.run()
        h.feed(_still(h.pose, 3), limit=3)
        self.assertEqual(h.decisions()[:2], ["hold: frozen", "resume: the watch check saw nothing wrong"])
        self.assertEqual(h.driver.calls, ["run", "pause", "run"])
        self.assertEqual((h.layer.doing, h.kinds("escalation"), h.asked), ("pass", [], []))

    def test_a_confirmed_stuck_rule_asks_for_a_person_and_the_advice_follows(self) -> None:
        h = Harness("act", {"stuck": 0.9})
        h.feed(Trace().to(2, pan=20).to(4.5))
        self.assertEqual(h.asked, ["Supervisor: stuck (frozen). Paused for you; asking for advice."])
        self.assertEqual(h.ask_kinds, ["stuck"])
        [advice] = h.layer.maintain()  # serve keeps calling it while paused
        self.assertIn("Take over with the leader", advice)
        self.assertEqual(h.kinds("escalation")[0]["choice"], "teleop")
        watch = h.judges._adapters["jev"].asked[-1]
        self.assertEqual(watch, ("stuck", ["top", "wrist", "top_2s_ago", "wrist_2s_ago"]))

    def test_the_escalation_model_is_asked_once_an_attempt(self) -> None:
        h = Harness("shadow", {"stuck": 0.9, "after_close": 0.95})
        h.feed(miss())
        h.feed(Trace().to(0.3, gripper=40).to(0.3, gripper=10).to(1.2))
        h.feed(Trace().to(0.3, gripper=40).to(0.3, gripper=10).to(1.2))  # a third miss: stuck, not a rewind
        h.feed(Trace().to(2, pan=20).to(4.5))  # and frozen
        self.assertEqual(h.judges._adapters["claude-opus-5.5"].calls, 1)

    def test_a_late_after_close_verdict_is_dropped_after_a_newer_close(self) -> None:
        pool = Pool(now=False)
        h = Harness("act", {"after_close": 0.95}, pool=pool)
        h.feed(Trace().to(2, pan=20).to(0.5, gripper=10).to(1.2))  # the first check is asked, not answered
        h.feed(Trace().to(0.3, gripper=40).to(0.3, gripper=10).to(0.2))  # a second close before its answer
        pool.run()
        h.layer.maintain()
        checks = [c for c in h.kinds("check") if c["moment"] == "after_close"]
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0]["stale"])
        self.assertEqual(h.driver.calls, ["run"])

    def test_a_done_release_asks_a_person_once_the_arm_is_home(self) -> None:
        h = Harness("act", {"after_release": 0.02}, chat={"after_release": 0.03})
        h.rest = pose(gripper=40.0)
        h.feed(grasp().to(2))
        check = next(c for c in h.kinds("check") if c["moment"] == "after_release")
        self.assertTrue(check["success"])
        self.assertEqual((h.asked, h.decisions()), ([], []), "the policy still drives the arm home")
        h.feed(h.here().to(2, pan=0, elbow=0).to(0.5))
        self.assertEqual(h.asked, ["Supervisor: the block looks in the container. Reset and label the attempt."])
        self.assertEqual(h.ask_kinds, ["done"])
        self.assertLess(h.kinds("decision")[0]["policy_t"], 11.5, "asked on arrival")
        h.layer.end_episode()
        self.assertEqual(h.kinds("attempt_end")[0]["released"], True)

    def test_a_done_release_asks_after_home_within_s_if_the_arm_never_gets_home(self) -> None:
        h = Harness("act", {"after_release": 0.02}, chat={"after_release": 0.03})
        h.feed(grasp().to(12))
        [decision] = h.kinds("decision")
        released = next(e for e in h.kinds("grasp") if e["grasp"] == "release")["policy_t"]
        self.assertAlmostEqual(decision["policy_t"] - released, PROGRAM.home_within_s, delta=0.1)
        self.assertEqual(h.ask_kinds, ["done"])

    def test_a_verdict_while_a_person_has_the_arm_changes_nothing(self) -> None:
        pool = Pool(now=False)
        h = Harness("act", {"after_close": 0.95}, pool=pool)
        h.feed(miss())
        h.layer.pause()
        pool.run()
        h.layer.maintain()
        self.assertEqual(h.decisions(), [])
        self.assertEqual([c["stale"] for c in h.kinds("check")], ["paused_or_resumed"])
        h.layer.run(Handback(PERSON))
        h.feed(_still(h.pose, 5))
        self.assertEqual(h.robot.goals, [], "no playback once it has the arm again")

    def test_a_verdict_arriving_after_resume_is_dropped(self) -> None:
        pool = Pool(now=False)
        h = Harness("act", {"after_close": 0.95}, pool=pool)
        h.feed(Trace().to(2, pan=20).to(0.5, gripper=10).to(1.1))  # asked, not answered
        h.layer.pause()
        h.resume()
        h.feed(_still(h.pose, 3))
        pool.run()
        h.feed(_still(h.pose, 3))
        self.assertEqual(h.decisions(), [])
        self.assertEqual(h.driver.calls, ["run", "pause", "run"], "only the person's pause and resume")

    def test_an_after_close_verdict_is_dropped_after_a_carry(self) -> None:
        pool = Pool(now=False)
        h = Harness("act", {"after_close": 0.95}, pool=pool)
        h.feed(Trace().to(2, pan=20).to(0.5, gripper=18).to(1.1))  # asked, not answered
        h.feed(h.here().to(1, elbow=40))  # then it carries
        pool.run()
        h.feed(_still(h.pose, 3))
        self.assertEqual(h.decisions(), [])
        self.assertEqual([c["stale"] for c in h.kinds("check")], ["carried_or_released"])

    def test_a_watch_verdict_settles_only_the_hold_it_was_asked_for(self) -> None:
        pool = Pool(now=False)
        h = Harness("act", {"stuck": 0.05}, pool=pool)
        h.feed(Trace().to(2, pan=20).to(4.5), until=h.holding)  # frozen: holds, its watch check asked
        h.layer.pause()  # a person steps in, then resumes
        h.resume()
        h.feed(h.here().to(14, pan=220), until=h.holding)  # no carry in 14 s: holds again
        self.assertEqual(h.decisions(), ["hold: frozen", "hold: no_carry"])
        pool.run(1)  # only the first hold's answer comes back: sure nothing is wrong
        h.settle(3)
        self.assertEqual(h.decisions(), ["hold: frozen", "hold: no_carry"], "the second hold still holds")
        self.assertTrue(h.holding())

    def test_a_hold_asks_once_its_deadline_passes(self) -> None:
        h = Harness("act", {}, models={"jev": Hung(), "gemini-3.8-flash": Hung()})
        h.feed(Trace().to(2, pan=20).to(4.5), until=h.holding)
        held = h.now
        h.settle(10 * FPS)
        self.assertEqual(h.ask_kinds, ["stuck"])
        self.assertAlmostEqual(h.kinds("decision")[1]["policy_t"], h.kinds("decision")[0]["policy_t"])
        self.assertAlmostEqual(h.now - held, PROGRAM.hold_deadline_s, delta=0.1)

    def test_no_watch_check_is_asked_while_a_rule_is_only_near(self) -> None:
        h = Harness("act", {"stuck": 0.95})
        h.feed(Trace().to(2, pan=20).to(3.5))  # frozen is near (3.5 of 4 s), never fires
        self.assertEqual(h.judges._adapters["jev"].asked, [])

    def test_the_frames_kept_stay_bounded_while_it_holds(self) -> None:
        h = Harness("act", {}, models={"jev": Hung(), "gemini-3.8-flash": Hung()})
        h.ignore_asks = True
        h.feed(Trace().to(2, pan=20).to(4.5), until=h.holding)
        h.settle(20 * FPS)
        self.assertLessEqual(len(h.layer.attempt.ring), P.RING_S / 0.2 + 2)

    def test_a_judge_without_its_key_is_told_once(self) -> None:
        h = Harness("act", {}, models={"gemini-3.8-flash": None})
        h.layer.start()
        h.feed(Trace().to(1, pan=10))
        told = [line for line in h.said if "gemini-3.8-flash" in line]
        self.assertEqual(len(told), 1)
        self.assertIn("never played back", told[0])

    def test_running_out_of_budget_is_told_once(self) -> None:
        program = dataclasses.replace(PROGRAM, judges=dataclasses.replace(PROGRAM.judges, max_cost_usd=0.0015))
        h = Harness("act", {"after_close": 0.05}, program=program)
        h.feed(miss())  # its after_close check spends the budget
        h.feed(h.here().to(0.3, gripper=40).to(0.3, gripper=10).to(1.5))
        told = [line for line in h.said if "max_cost_usd" in line]
        self.assertEqual(len(told), 1)

    def test_its_own_error_is_logged_once_and_the_policy_drives_on(self) -> None:
        h = Harness("act", {})
        h.feed(Trace().to(0.5, pan=5))

        def broken(*args):
            raise RuntimeError("checks bug")

        h.layer._due_checks = broken
        steps = h.driver.steps
        h.feed(Trace().to(1, pan=10))
        self.assertEqual([e["error"] for e in h.kinds("error")], ["RuntimeError: checks bug"])
        self.assertEqual(h.driver.steps - steps, 31, "every step still went to the policy")
        self.assertTrue(any("stopped after an error" in line for line in h.said))

    def test_the_wrapped_policy_s_errors_reach_the_runner_unchanged(self) -> None:
        h = Harness("act", {})
        h.driver.error = ValueError("inference failed")
        with self.assertRaisesRegex(ValueError, "inference failed"):
            h.layer.step(h.robot, Tick(0.0, 0, {**pose(), "top": FRAME, "wrist": FRAME}, pose()))
        h.layer.record_failure("traceback")
        self.assertEqual(h.layer.health().failure, "traceback", "the runner's note of it lands on the model, and reads back")
        self.assertEqual(h.driver.failure, "traceback")


class DaemonPoolTest(unittest.TestCase):
    def test_a_hung_call_never_holds_up_closing_and_queued_calls_are_cancelled(self) -> None:
        import threading

        started, release = threading.Event(), threading.Event()
        pool = J.DaemonPool(1, "test-judge")
        hung = pool.submit(lambda: (started.set(), release.wait())[1])
        self.assertTrue(started.wait(2))
        queued = pool.submit(lambda: "never")
        began = time.monotonic()
        pool.shutdown(wait=False, cancel_futures=True)
        self.assertLess(time.monotonic() - began, 0.5)
        self.assertTrue(queued.cancelled())
        self.assertTrue(all(t.daemon for t in pool._threads), "Python's exit doesn't wait for a daemon worker")
        self.assertTrue(pool.submit(lambda: 1).cancelled(), "nothing new runs once closed")
        release.set()
        self.assertTrue(hung.result(timeout=2))


class CandidatesTest(unittest.TestCase):
    def test_candidates_come_from_where_the_supervisor_would_check(self) -> None:
        trace = Trace().to(1, pan=5)
        trace.to(2, pan=20).to(1, gripper=18).to(2, elbow=40).to(1, gripper=40).to(1)
        n = len(trace.poses)
        phases = np.array([0] * 10 + [1] * n + [6] * 10)  # starting, the policy, returning
        states = np.array([[p[k] for k in NAMES] for p in [trace.poses[0]] * 10 + trace.poses + [trace.poses[-1]] * 10])
        rows = candidates.pick("run1", 0, phases, np.zeros(len(phases), bool), states, NAMES, FPS, PROGRAM, GRASP)
        self.assertEqual([r["moment"] for r in rows], ["after_close", "after_release", "approach", "end"])
        closed = 10 + next(i for i, p in enumerate(trace.poses) if p[GRIPPER] < GRASP.gripper.closed_below)
        close = next(r for r in rows if r["moment"] == "after_close")
        self.assertEqual(close["t"], closed + round(PROGRAM.after_close_s * FPS))
        self.assertEqual(rows[-1]["t"], len(phases) - 1)
        self.assertTrue(all(r["label"] == "" and r["src"] == "run1" for r in rows))


class SettingsTest(ConfigTest):
    def test_the_supervisor_is_off_unless_the_file_turns_it_on(self) -> None:
        self.config.write_text(ARMS)
        self.assertEqual(settings.load().supervisor, settings.Supervisor("off", None))
        self.config.write_text(ARMS + '\n[supervisor]\nmode = "shadow"\nprogram = "mine.toml"\n')
        loaded = settings.load().supervisor
        self.assertEqual(loaded.mode, "shadow")
        self.assertEqual(loaded.program, settings.load().path.parent / "mine.toml")

    def test_the_perception_reads_the_package_thresholds_unless_the_file_names_others(self) -> None:
        self.config.write_text(ARMS)
        self.assertIsNone(settings.load().perception.thresholds)
        self.config.write_text(ARMS + '\n[perception]\nthresholds = "mine.toml"\n')
        self.assertEqual(settings.load().perception.thresholds, settings.load().path.parent / "mine.toml")
        self.config.write_text(ARMS + '\n[perception]\nthresholds = 3\n')
        with self.assertRaisesRegex(settings.ConfigError, r"\[perception\] thresholds"):
            settings.load()

    def supervisor_check(self, extra: str = "", sdk=None):
        """`check_supervisor` on so101.toml with `extra`, every judge's key set; returns (problems, report)."""
        from so_101 import check
        from so_101.report import Report

        self.config.write_text(ARMS + extra)
        report, problems = Report(), []
        keys = {"TYPESAFE_API_KEY": "x", "OPENROUTER_API_KEY": "x", "GEMINI_API_KEY": "x", "OPENAI_API_KEY": "x"}
        with mock.patch.dict(os.environ, keys), mock.patch("so_101.vision.providers.sdk_missing", sdk or (lambda p: None)):
            check.check_supervisor(report, settings.load(), lambda *args: problems.append(args))
        return problems, report

    def test_the_thresholds_and_program_are_checked_even_with_the_supervisor_off(self) -> None:
        (self.root / "bad.toml").write_text("version = 1\n")
        for table in ("perception]\nthresholds", "supervisor]\nprogram"):
            with self.subTest(table):
                problems, _ = self.supervisor_check(f'\n[{table} = "bad.toml"\n')
                [(group, detail, _fix)] = problems
                self.assertIn("bad.toml", detail)
        problems, report = self.supervisor_check()
        self.assertEqual(problems, [])
        self.assertEqual([r.status for r in report.results], ["PASS"])

    def test_a_judge_whose_sdk_is_missing_is_a_warning(self) -> None:
        sdk = lambda provider: "the typesafe-sdk SDK isn't installed" if provider == "typesafe" else None  # noqa: E731
        problems, report = self.supervisor_check('\n[supervisor]\nmode = "shadow"\n', sdk=sdk)
        self.assertEqual(problems, [])
        [warning] = report.warnings
        self.assertIn("jev: the typesafe-sdk SDK isn't installed", warning.detail)

    def test_an_unknown_mode_is_refused(self) -> None:
        self.config.write_text(ARMS + '\n[supervisor]\nmode = "on"\n')
        with self.assertRaisesRegex(settings.ConfigError, "mode"):
            settings.load()


class InServeTest(unittest.TestCase):
    """The layer as serve's policy on the controller's fake arm, recording a run."""

    def test_misses_play_back_then_serve_pauses_when_it_asks_and_analysis_never_trains_on_either(self) -> None:
        from test_controller import START_POSE, Rig

        from so_101.control import analysis
        from so_101.data import Run

        # Open the jaws while sweeping the pan, then close them on nothing and stay there.
        opened = [{**START_POSE, "shoulder_pan.pos": i / 4, GRIPPER: min(10.0 + 2 * i, 50.0)} for i in range(1, 81)]
        closed = [{**opened[-1], GRIPPER: 50.0 - 4 * i} for i in range(1, 11)]
        actions = [[a[k] for k in START_POSE] for a in opened + closed]
        made = []

        def wrap(driver, recorder, clock):
            made.append(stack.build(driver, LayerContext(recorder.event, clock, GRASP, PROGRAM),
                                    judges=judges({"after_close": 0.95, "stuck": 0.95}), mode="act",
                                    cameras=("top", "wrist")))
            return made[0]

        with tempfile.TemporaryDirectory() as folder:
            rig = Rig(Path(folder) / "run", actions=actions, wrap=wrap)
            rig.start_policy(ticks=1)
            rig.tick_until("paused", limit=40 * FPS)
            rig.reset("failed")
            rig.controller.close()
            playbacks = rig.events("playback")
            self.assertEqual(len(playbacks), 2, "two misses played back; the third asks for a person")
            self.assertAlmostEqual(playbacks[0]["seconds"], 2.5, delta=0.05)
            self.assertGreater(playbacks[1]["seconds"], 2.5, "the second goes further back")
            self.assertEqual([e["cause"] for e in rig.events("state") if e["state"] == "paused"], ["policy_asked"])
            self.assertEqual([e for e in rig.events("command") if e.get("source") not in ("terminal",)], [])
            [attempt] = analysis.analyse(Run.at(rig.run_dir), 1.0).attempts
            suspect = attempt.suspect_mask()
            for e in playbacks:
                self.assertTrue(suspect[e["start_frame"] : e["end_frame"]].all(), "the motion it undid")
            self.assertEqual([i.command for i in attempt.interruptions], ["policy_asked"])


if __name__ == "__main__":
    unittest.main()
