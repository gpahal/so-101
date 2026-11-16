"""The supervisor without hardware: the program, the detector on made-up traces, the judges with stand-in
models, the layer's decisions in shadow and act mode, the layer as serve's policy on the fake arm, and
failbench's candidates from a run's columns.

The traces are 30 Hz, in the follower's units (degrees, gripper 0 to 100). The thresholds are the
program's own; the tests check what the rules do, not that the numbers are right for the arm.
"""

import dataclasses
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from test_serve_preflight import ARMS, ConfigTest

from so_101 import settings
from so_101.failbench import candidates
from so_101.supervisor import judges as J
from so_101.supervisor import program as P
from so_101.supervisor.layer import SupervisedPolicy
from so_101.supervisor.detector import ARM, GRIPPER, Detector

FPS = 30
PROGRAM = P.load()
NAMES = [*ARM, GRIPPER]


def pose(gripper=40.0, elbow=0.0, pan=0.0) -> dict[str, float]:
    return {"shoulder_pan.pos": pan, "shoulder_lift.pos": 0.0, "elbow_flex.pos": elbow, "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0, GRIPPER: gripper}


class Trace:
    """Builds a 30 Hz trace segment by segment, each moving linearly from the last pose."""

    def __init__(self) -> None:
        self.poses = [pose()]
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
    def test_the_default_program_loads(self) -> None:
        self.assertEqual(PROGRAM.rules.regrasps, 2)
        self.assertEqual(PROGRAM.judges.decision, "jev")

    def test_a_bad_value_names_itself(self) -> None:
        text = P.DEFAULT.read_text()
        cases = {
            "hold band over the close threshold": (text.replace("hold_band = [16.5, 22.0]", "hold_band = [16.5, 26.0]"),
                                                   "hold_band high < closed_below"),
            "an unknown judge": (text.replace('decision = "jev"', 'decision = "nope"'), "judges.decision"),
            "a negative threshold": (text.replace("seconds = 14.0", "seconds = -1"), "rules.no_carry.seconds"),
            "a missing section": (text.replace("[rules.chatter]", "[rules.chattering]"), r"\[rules.chatter\]"),
            "a fraction of a regrasp": (text.replace("regrasps = 2 ", "regrasps = 1.5 "), "regrasps"),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, (changed, message) in cases.items():
                with self.subTest(name):
                    path = Path(folder) / "program.toml"
                    path.write_text(changed)
                    with self.assertRaisesRegex(P.ProgramError, message):
                        P.load(path)


class DetectorTest(unittest.TestCase):
    def test_a_clean_grasp_closes_carries_and_releases_with_no_rule(self) -> None:
        detector = Detector(PROGRAM)
        self.assertEqual(kinds(grasp().run(detector)), ["close", "carry", "release"])
        self.assertEqual(detector.fired, set())

    def test_two_misses_fire_the_regrasp_loop(self) -> None:
        trace = Trace().to(2, pan=20)
        for _ in range(2):  # closes on nothing and opens again, the elbow still, slowly enough not to chatter
            trace.to(0.5, gripper=10, pan=25).to(0.7, pan=30).to(0.5, gripper=40, pan=25).to(0.5, pan=20)
        self.assertEqual(kinds(trace.run(Detector(PROGRAM))),
                         ["close", "regrasp", "close", "regrasp", "rule:regrasp_loop"])

    def test_one_miss_is_not_stuck(self) -> None:
        trace = Trace().to(2, pan=20).to(0.5, gripper=10).to(0.5, gripper=40).to(1, gripper=18).to(2, elbow=40)
        self.assertEqual(kinds(trace.run(Detector(PROGRAM))), ["close", "regrasp", "close", "carry"])

    def test_creeping_forward_open_away_from_the_start_is_a_hover(self) -> None:
        trace = Trace().to(2, pan=20).to(6, pan=50)  # 5 deg/s: slower than a reach, faster than frozen
        signals = trace.run(Detector(PROGRAM))
        self.assertEqual(kinds(signals), ["rule:hover"])
        self.assertAlmostEqual(signals[0].t, 7.0, delta=0.2)

    def test_a_flickering_gripper_is_chatter(self) -> None:
        trace = Trace().to(2, pan=20, gripper=30)
        for _ in range(3):
            trace.to(0.3, gripper=20).to(0.3, gripper=30)
        self.assertIn("rule:chatter", kinds(trace.run(Detector(PROGRAM))))

    def test_standing_still_is_frozen_but_waiting_for_the_policy_is_not(self) -> None:
        self.assertEqual(kinds(Trace().to(2, pan=20).to(4.5).run(Detector(PROGRAM))), ["rule:frozen"])
        self.assertEqual(kinds(Trace().to(2, pan=20).to(4.5, drove=False).run(Detector(PROGRAM))), [])

    def test_nothing_carried_in_14_s_of_driving(self) -> None:
        signals = Trace().to(14.5, pan=200).run(Detector(PROGRAM))  # 14 deg/s: never slow enough to hover
        self.assertEqual(kinds(signals), ["rule:no_carry"])
        self.assertAlmostEqual(signals[0].t, 14.0, delta=0.1)

    def test_holding_a_carried_block_for_10_s_is_not_released(self) -> None:
        trace = Trace().to(2, pan=20).to(1, gripper=18).to(2, elbow=40).to(11, pan=60)
        self.assertEqual(kinds(trace.run(Detector(PROGRAM))), ["close", "carry", "rule:not_released"])

    def test_after_a_release_nothing_is_a_grasp_or_stuck(self) -> None:
        trace = grasp().to(1, gripper=10).to(20)  # closes on the way back to rest, then sits
        self.assertEqual(kinds(trace.run(Detector(PROGRAM))), ["close", "carry", "release"])

    def test_a_resume_restarts_the_running_timers(self) -> None:
        detector = Detector(PROGRAM)
        trace = Trace().to(2, pan=20).to(3)
        out = trace.run(detector)
        detector.resumed()
        t0 = len(trace.poses) / FPS
        for i in range(int(3 * FPS)):
            out += detector.step(t0 + i / FPS, trace.poses[-1])
        self.assertEqual(kinds(out), [], "3 s still, a rewind, then 3 s still: never 4 s in a row")

    def test_progress_tells_the_watch_when_a_rule_is_near(self) -> None:
        detector = Detector(PROGRAM)
        Trace().to(2, pan=20).to(3).run(detector)
        self.assertEqual(detector.near(), ["frozen"])


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


class Pool:
    """Runs each call when `run` is called, or at once with `now`."""

    def __init__(self, now: bool = True) -> None:
        self.now, self.waiting = now, []

    def submit(self, fn, *args):
        future = Future()
        if self.now:
            future.set_result(fn(*args))
        else:
            self.waiting.append((future, fn, args))
        return future

    def run(self) -> None:
        """Answer the calls waiting; a call they make in turn (the event check's chat model) runs at once."""
        waiting, self.waiting, self.now = self.waiting, [], True
        for future, fn, args in waiting:
            future.set_result(fn(*args))
        self.now = False

    def shutdown(self, **kwargs) -> None:
        pass


def judges(decision: dict[str, float], chat: dict[str, float] | None = None, program=PROGRAM, pool=None):
    return J.Judges(program, adapters={"jev": FakeModel(decision), "gemini-3.8-flash": FakeModel(chat or {}),
                                       "claude-opus-5.5": FakeEscalation()}, pool=pool or Pool())


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
    """The policy driver the layer wraps. Each step it reports an action, or none on a waiting tick."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.steps = self.holds = 0
        self.raised = self.retrying = self.failure = None
        self.drove = True
        self.error: Exception | None = None

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def maintain(self) -> None:
        return None

    def pause(self) -> None:
        self.calls.append("pause")

    def run(self) -> None:
        self.calls.append("run")

    def hold(self, robot) -> None:
        self.holds += 1

    def step(self, robot, observation):
        if self.error is not None:
            raise self.error
        self.steps += 1
        return {"shoulder_pan.pos": observation["shoulder_pan.pos"]} if self.drove else None


class FakeRobot:
    def __init__(self) -> None:
        self.goals: list[dict] = []

    def send_action(self, goal: dict) -> None:
        self.goals.append(dict(goal))


class Harness:
    """The layer stepped by hand as serve steps it: maintain, then step, then asks. The trace's poses are
    the policy's, taken only on ticks the policy drives; on the layer's own ticks the arm is where the
    layer last sent it. When the layer asks for a person, the harness pauses it, as serve does."""

    def __init__(self, mode: str, decision: dict[str, float], chat=None, pool=None) -> None:
        self.logged: list[dict] = []
        self.said: list[str] = []
        self.asked: list[str] = []
        self.judges = judges(decision, chat, pool=pool)
        self.driver, self.robot = FakeDriver(), FakeRobot()
        self.now, self.frame, self.pose = 0.0, -1, pose()
        self.layer = SupervisedPolicy(self.driver, PROGRAM, self.judges, mode=mode, frame=lambda: self.frame,
                                      log=lambda type_, **fields: self.logged.append(fields), clock=lambda: self.now)
        self.layer.begin_episode(1)
        self.layer.run()
        self.paused = False

    def feed(self, trace: Trace, until=None, limit: int = 3000) -> None:
        """Step through `trace`, or until `until()` is true after a step."""
        moves = list(zip(trace.poses, trace.drove, strict=True))
        i = 0
        while i < len(moves) and limit and not self.paused and not (until and until()):
            limit -= 1
            self.now += 1 / FPS
            self.frame += 1
            if (message := self.layer.maintain()) is not None:
                self.said.append(message)
            if self.layer.driving == "policy":
                p, self.driver.drove = moves[i]
            else:
                p = self.robot.goals[-1] if self.robot.goals else self.pose
            steps = self.driver.steps
            self.layer.step(self.robot, {**p, "top": FRAME, "wrist": FRAME})
            self.pose = p
            if self.driver.steps > steps:
                i += 1
            if (asked := self.layer.asks()) is not None:
                self.asked.append(asked)
                self.layer.pause()
                self.paused = True

    def settle(self, ticks: int = 3 * FPS) -> None:
        """The policy stands still, where the arm is, while the layer finishes what it started."""
        self.feed(_still(self.pose, ticks))

    def rewinding(self) -> bool:
        return self.layer.driving == "rewind"

    def resume(self) -> None:
        self.paused = False
        self.layer.run()

    def kinds(self, kind: str) -> list[dict]:
        return [e for e in self.logged if e["kind"] == kind]

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
        [rewind] = h.kinds("rewind")
        self.assertLess(rewind["start_frame"], rewind["end_frame"])
        self.assertAlmostEqual(rewind["seconds"], 2.5, delta=0.05)
        self.assertEqual(rewind["cause"], "arrived")
        self.assertEqual(h.robot.goals[-1][GRIPPER], 40.0, "back to before the close: the jaws open")
        self.assertTrue(any("missed grasp 1" in line for line in h.said))

    def test_a_playback_s_ticks_are_never_the_policy_s_action(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(miss(), until=h.rewinding)
        self.assertEqual(h.layer.driving, "rewind")
        self.assertIsNone(h.layer.step(h.robot, {**h.robot.goals[-1], "top": FRAME, "wrist": FRAME}))

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

    def test_an_unsure_decision_model_defers_to_the_chat_model(self) -> None:
        h = Harness("act", {"after_close": 0.5}, chat={"after_close": 0.9})
        h.feed(miss(), until=h.rewinding)
        self.assertEqual(h.decisions(), ["miss 1: rewind 2.5 s and retry"])

    def test_a_playback_never_goes_back_past_where_a_person_handed_the_arm_back(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(Trace().to(2, pan=20))
        h.layer.pause()  # serve's Pause: a person may move the arm now
        h.layer.run()
        handed_back = h.frame + 1
        h.feed(Trace().to(0.3, gripper=10).to(1.5), until=h.rewinding)  # 1.2 s since, under the 2.5 s asked for
        h.settle()
        [rewind] = h.kinds("rewind")
        self.assertEqual(rewind["start_frame"], handed_back)

    def test_a_pause_cuts_a_playback_short_and_logs_what_it_undid(self) -> None:
        h = Harness("act", {"after_close": 0.95})
        h.feed(miss(), until=h.rewinding)
        h.feed(_still(h.pose, 10), limit=10)  # ten ticks of the playback
        h.layer.pause()
        [rewind] = h.kinds("rewind")
        self.assertEqual(rewind["cause"], "paused")
        self.assertLess(rewind["seconds"], 2.5)
        self.assertEqual(h.layer.driving, "policy")

    def test_a_stuck_rule_holds_the_arm_and_a_sure_ok_watch_check_hands_it_back(self) -> None:
        pool = Pool(now=False)  # the watch check takes its time
        h = Harness("act", {"stuck": 0.05}, pool=pool)
        h.feed(Trace().to(2, pan=20).to(4.5), until=lambda: h.layer.driving == "hold")  # frozen
        h.feed(_still(h.pose, 5), limit=5)
        self.assertEqual(h.driver.holds, 6, "the arm holds on every tick until the answer")
        pool.run()
        h.feed(_still(h.pose, 3), limit=3)
        self.assertEqual(h.decisions()[:2], ["hold: frozen", "resume: the watch check saw nothing wrong"])
        self.assertEqual(h.driver.calls, ["run", "pause", "run"])
        self.assertEqual((h.layer.driving, h.kinds("escalation"), h.asked), ("policy", [], []))

    def test_a_confirmed_stuck_rule_asks_for_a_person_and_the_advice_follows(self) -> None:
        h = Harness("act", {"stuck": 0.9})
        h.feed(Trace().to(2, pan=20).to(4.5))
        self.assertEqual(h.asked, ["Supervisor: stuck (frozen). Paused for you; asking for advice."])
        advice = h.layer.maintain()  # serve keeps calling it while paused
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

    def test_a_release_both_judges_are_sure_of_asks_a_person_to_end_the_attempt(self) -> None:
        h = Harness("act", {"after_release": 0.02}, chat={"after_release": 0.03})
        h.feed(grasp().to(2))
        check = next(c for c in h.kinds("check") if c["moment"] == "after_release")
        self.assertTrue(check["success"])
        self.assertEqual(h.asked, ["Supervisor: the block looks in the container. Reset and label the attempt."])
        h.layer.end_episode()
        self.assertEqual(h.kinds("attempt_end")[0]["released"], True)

    def test_a_verdict_while_a_person_has_the_arm_changes_nothing(self) -> None:
        pool = Pool(now=False)
        h = Harness("act", {"after_close": 0.95}, pool=pool)
        h.feed(miss())
        h.layer.pause()
        pool.run()
        h.layer.maintain()
        self.assertFalse(h.kinds("decision")[0]["acted"])
        h.layer.run()
        h.feed(_still(h.pose, 5))
        self.assertEqual(h.robot.goals, [], "no playback once it has the arm again")

    def test_its_own_error_is_logged_once_and_the_policy_drives_on(self) -> None:
        h = Harness("act", {})
        h.feed(Trace().to(0.5, pan=5))

        def broken(*args):
            raise RuntimeError("detector bug")

        h.layer.attempt.detector.step = broken
        steps = h.driver.steps
        h.feed(Trace().to(1, pan=10))
        self.assertEqual([e["error"] for e in h.kinds("error")], ["RuntimeError: detector bug"])
        self.assertEqual(h.driver.steps - steps, 31, "every step still went to the policy")
        self.assertTrue(any("stopped after an error" in line for line in h.said))

    def test_the_wrapped_policy_s_errors_reach_the_runner_unchanged(self) -> None:
        h = Harness("act", {})
        h.driver.error = ValueError("inference failed")
        with self.assertRaisesRegex(ValueError, "inference failed"):
            h.layer.step(h.robot, {**pose(), "top": FRAME, "wrist": FRAME})
        h.driver.raised = None
        h.layer.raised = "traceback"
        self.assertEqual(h.driver.raised, "traceback", "the runner's note of it lands on the wrapped driver")


class CandidatesTest(unittest.TestCase):
    def test_candidates_come_from_where_the_supervisor_would_check(self) -> None:
        trace = Trace().to(1, pan=5)
        trace.to(2, pan=20).to(1, gripper=18).to(2, elbow=40).to(1, gripper=40).to(1)
        n = len(trace.poses)
        phases = np.array([0] * 10 + [1] * n + [6] * 10)  # starting, the policy, returning
        states = np.array([[p[k] for k in NAMES] for p in [trace.poses[0]] * 10 + trace.poses + [trace.poses[-1]] * 10])
        rows = candidates.pick("run1", 0, phases, np.zeros(len(phases), bool), states, NAMES, FPS, PROGRAM)
        self.assertEqual([r["moment"] for r in rows], ["after_close", "after_release", "approach", "end"])
        closed = 10 + next(i for i, p in enumerate(trace.poses) if p[GRIPPER] < PROGRAM.gripper.closed_below)
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
            made.append(SupervisedPolicy(driver, PROGRAM, judges({"after_close": 0.95, "stuck": 0.95}),
                                         mode="act", log=recorder.event, frame=lambda: recorder.frames, clock=clock))
            return made[0]

        with tempfile.TemporaryDirectory() as folder:
            rig = Rig(Path(folder) / "run", actions=actions, wrap=wrap)
            rig.start_policy(ticks=1)
            rig.tick_until("paused", limit=40 * FPS)
            rig.reset("failed")
            rig.controller.close()
            events = rig.events("supervisor")
            playbacks = [e for e in events if e["kind"] == "rewind"]
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
