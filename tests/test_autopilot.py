"""`so101 run`'s autopilot (docs/run.md) on the controller's fake arm: it starts, ends and resets attempts as
another operator would, and hands an attempt to a person once they act in it.

The controller steps the autopilot once a tick, after it publishes the status, as it does in `so101 run`.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fakes import control
from test_controller import FPS, Asking, Rig

from so_101.control.autopilot import ASK_GRACE_S, Autopilot
from so_101.control.terminal import KeyReader
from so_101.policy.base import DONE

LIMIT_S = 2.0
RESET_S = 1.0
PHONE = "phone-page"  # the client id a control page made for itself


class Session:
    """A recorded run on the fake arm, with the autopilot over it."""

    def __init__(self, test: unittest.TestCase, episodes: int = 2, ask_at: int | None = None,
                 limit_s: float = LIMIT_S, kind: str | None = None) -> None:
        wrap = (lambda driver, recorder, clock: Asking(driver, ask_at, **({"kind": kind} if kind else {}))) \
            if ask_at else None
        folder = tempfile.TemporaryDirectory()
        test.addCleanup(folder.cleanup)
        self.rig = Rig(Path(folder.name) / "run", control=control(preview_fps=0, attempt_limit_s=limit_s), wrap=wrap)
        test.addCleanup(self.rig.close)
        self.said: list[str] = []
        c = self.rig.controller
        self.pilot = c.autopilot = Autopilot(submit=c.submit, say=self.said.append, episodes=episodes, reset_s=RESET_S)
        self.polling = False  # the phone's page polls before each tick

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            if self.polling:
                self.rig.controller.heartbeat(PHONE)
            self.rig.tick()

    def until(self, test, limit_s: float = 60.0) -> None:
        for _ in range(round(limit_s * FPS)):
            if test():
                return
            self.tick()
        raise AssertionError(f"not reached in {limit_s} s; state {self.rig.status()['state']}, said {self.said[-3:]}")

    def state(self) -> str:
        return self.rig.status()["state"]

    def events(self, type_: str) -> list[dict]:
        return [e for e in self.rig.controller.events(0, 500) if e["type"] == type_]

    def starts(self) -> list[float]:
        return [e["t"] for e in self.events("attempt_start")]

    def commands(self, source: str = "autopilot") -> list[str]:
        return [e["command"] for e in self.events("command") if e["source"] == source]

    def page(self, name: str, **options):
        return self.rig.command(name, source="page", client=PHONE, who="100.64.0.2", **options)


class AutopilotTest(unittest.TestCase):
    def test_it_makes_its_attempts_with_a_reset_between_them_and_exits_at_rest(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.rig.controller.done)
        events = s.rig.controller.events(0, 500)
        self.assertEqual(len(s.starts()), 2)
        ends = [e for e in events if e["type"] == "state" and e.get("state") == "returning"]
        self.assertEqual([e.get("ended_by") for e in ends], ["time_limit", "time_limit"])
        rested = [e["t"] for e in events if e["type"] == "state" and e.get("state") == "resting"]
        self.assertGreaterEqual(s.starts()[1] - rested[0], RESET_S - 1 / FPS, "it waits reset_s between attempts")
        self.assertEqual([e["outcome"] for e in events if e["type"] == "outcome"], [], "every attempt unlabelled")
        commands = s.events("command")
        self.assertTrue(all(e["source"] == "autopilot" and e["accepted"] for e in commands))
        self.assertEqual([e["command"] for e in commands], ["resume", "resume", "quit"])

    def test_an_unanswered_ask_ends_the_attempt_after_the_grace_period(self) -> None:
        s = Session(self, episodes=1, ask_at=5)
        s.until(lambda: s.state() == "paused")
        asked = s.rig.clock()
        s.until(lambda: s.state() == "returning")
        self.assertGreaterEqual(s.rig.clock() - asked, ASK_GRACE_S)
        resets = [e for e in s.events("command") if e["command"] == "reset"]
        self.assertEqual([(e["source"], e.get("outcome")) for e in resets], [("autopilot", None)])
        self.assertEqual(len([line for line in s.said if "asks for a person" in line]), 1, "it says why once")

    def test_a_done_ask_ends_the_attempt_at_once(self) -> None:
        s = Session(self, episodes=1, ask_at=5, kind=DONE)
        s.until(lambda: s.state() == "paused")
        asked = s.rig.clock()
        s.until(lambda: s.state() == "returning", limit_s=1)
        self.assertLess(s.rig.clock() - asked, 0.2, "no grace wait")
        self.assertEqual(s.events("outcome"), [], "unlabelled")
        [paused] = [e for e in s.events("state") if e["state"] == "paused"]
        self.assertEqual(paused["ask"], DONE, "the verdict is in the events")

    def test_a_person_who_acts_takes_the_attempt_over(self) -> None:
        s = Session(self, episodes=2, ask_at=5, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        self.assertTrue(s.rig.command("resume").accepted)  # a person, at the terminal
        s.tick(round((ASK_GRACE_S + 1) * FPS))
        self.assertNotIn("reset", s.commands())
        s.rig.command("pause")
        s.tick(round(LIMIT_S * 2 * FPS))
        self.assertEqual(s.state(), "paused", "the autopilot never resumes an attempt a person took")
        s.rig.reset("failed")
        s.until(lambda: len(s.starts()) == 2, limit_s=10)  # and starts the next once it has ended

    def test_check_pose_a_label_or_taking_control_never_take_the_attempt_over(self) -> None:
        s = Session(self, episodes=1, ask_at=5, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        self.assertTrue(s.rig.command("pose").accepted)
        self.assertTrue(s.rig.command("label", label="stuck").accepted)
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "returning", limit_s=ASK_GRACE_S + 1)
        self.assertEqual(s.commands(), ["resume", "reset"])

    def test_the_outcome_question_keeps_the_grace_reset_off_the_label(self) -> None:
        s = Session(self, episodes=2, ask_at=5, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        keys = KeyReader(s.rig.controller.submit, s.rig.status)
        with redirect_stdout(io.StringIO()):
            keys.on_key("right")  # the terminal asks how it went
            s.tick(round((ASK_GRACE_S + 2) * FPS))
            self.assertEqual(s.state(), "paused", "no grace Reset while a person answers")
            self.assertTrue(s.rig.status()["autopilot"]["waiting"])
            keys.on_key("s")
            s.until(lambda: s.state() == "resting")
        self.assertEqual([e["outcome"] for e in s.events("outcome")], ["succeeded"])
        self.assertNotIn("reset", s.commands())

        # The page's Reset sheet does the same, for as long as it stays open.
        s.until(lambda: s.state() == "paused")
        attempt = s.rig.status()["attempt"]
        self.assertTrue(s.page("answering", for_attempt=attempt).accepted)
        s.tick(round((ASK_GRACE_S + 2) * FPS))
        self.assertEqual(s.state(), "paused")
        self.assertTrue(s.page("answering", for_attempt=attempt, open=False).accepted)  # the sheet is cancelled
        s.until(lambda: s.state() == "returning", limit_s=1)
        self.assertEqual(s.commands(), ["resume", "resume", "reset"])

    def test_a_link_loss_pause_is_said_and_reset_after_the_grace(self) -> None:
        s = Session(self, episodes=2, limit_s=60.0)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "autonomous")
        s.polling = False  # the phone's page is gone
        s.until(lambda: s.state() == "paused")
        lost = s.rig.clock()
        s.until(lambda: s.state() == "returning")
        self.assertGreaterEqual(s.rig.clock() - lost, ASK_GRACE_S)
        self.assertEqual(s.commands(), ["resume", "reset"])
        self.assertEqual(len([line for line in s.said if "link" in line]), 1)
        self.assertEqual(s.events("outcome"), [])
        s.until(lambda: len(s.starts()) == 2, limit_s=20)

    def test_a_start_that_stops_short_is_reset_after_the_grace(self) -> None:
        s = Session(self, episodes=1)
        # The shoulder rises from rest to the start pose, and something stops it on the way.
        s.rig.motors.stops = {"shoulder_lift": s.rig.motors.present()["shoulder_lift"] + 100}
        s.until(lambda: s.state() == "paused")
        s.rig.motors.stops = {}
        stopped = s.rig.clock()
        s.until(lambda: s.state() == "returning")
        self.assertGreaterEqual(s.rig.clock() - stopped, ASK_GRACE_S)
        self.assertEqual(len([line for line in s.said if "stopped short" in line]), 1)
        s.until(lambda: s.rig.controller.done)
        self.assertEqual(s.commands(), ["resume", "reset", "quit"])

    def test_a_persons_pause_on_the_way_to_rest_takes_that_attempt_over(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.state() == "returning")  # the time limit ended attempt 1
        self.assertTrue(s.rig.command("pause").accepted)
        self.assertEqual(s.state(), "paused", "the attempt stays open")
        s.tick(round((ASK_GRACE_S + 2) * FPS))
        self.assertEqual(s.state(), "paused", "it's the person's now")
        s.rig.command("reset")
        s.until(lambda: s.state() == "resting")
        rested = s.rig.clock()
        s.until(lambda: len(s.starts()) == 2, limit_s=10)
        self.assertGreaterEqual(s.rig.clock() - rested, RESET_S, "the reset wait counts from the arrival at rest")
        self.assertEqual(s.commands(), ["resume", "resume"])

    def test_a_fault_on_the_last_attempt_waits_for_a_reset_and_quits_only_at_rest(self) -> None:
        s = Session(self, episodes=1, limit_s=60.0)
        s.until(lambda: s.state() == "autonomous")
        s.rig.motors.failing = True
        s.tick()
        s.rig.motors.failing = False
        s.tick(round(ASK_GRACE_S * FPS))
        self.assertEqual(s.state(), "fault")
        self.assertEqual(s.commands(), ["resume"], "no quit in Fault")
        self.assertFalse(s.rig.status()["confirm_quit"])
        self.assertEqual(len([line for line in s.said if "Fault" in line]), 1, "it says why once")

        # Reset, then a Pause on the way: the arm rests away from rest. The run is over, but not there.
        self.assertTrue(s.rig.command("reset").accepted)
        s.tick(5)
        self.assertTrue(s.rig.command("pause").accepted)
        self.assertEqual(s.state(), "resting")
        self.assertFalse(s.rig.controller.at_rest)
        s.tick(round(ASK_GRACE_S * FPS))
        self.assertEqual(s.commands(), ["resume"], "no quit away from rest")
        self.assertFalse(s.rig.controller.done)
        self.assertEqual(len([line for line in s.said if "at rest" in line]), 1)

        s.rig.command("reset")
        s.until(lambda: s.rig.controller.done, limit_s=20)
        self.assertEqual(s.commands(), ["resume", "quit"])
        self.assertTrue(s.rig.controller.at_rest)

    def test_resting_away_from_rest_starts_nothing_until_a_reset(self) -> None:
        s = Session(self, episodes=2, limit_s=60.0)
        s.until(lambda: s.state() == "autonomous")
        s.rig.motors.failing = True
        s.tick()
        s.rig.motors.failing = False
        s.tick(2 * FPS)
        s.rig.command("reset")
        s.tick(5)
        s.rig.command("pause")
        self.assertEqual(s.state(), "resting")
        s.tick(round(ASK_GRACE_S * FPS))
        self.assertEqual(len(s.starts()), 1, "no attempt starts away from rest")
        s.rig.command("reset")
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        rested = s.rig.clock()
        s.until(lambda: len(s.starts()) == 2, limit_s=10)
        self.assertGreaterEqual(s.rig.clock() - rested, RESET_S)

    def test_a_person_s_quit_stops_it(self) -> None:
        s = Session(self, episodes=3)
        s.until(lambda: s.state() == "autonomous")
        s.rig.command("quit")
        s.until(lambda: s.rig.controller.done)
        self.assertEqual(len(s.starts()), 1)

    def test_a_refused_start_is_said_once_and_waited_out(self) -> None:
        s = Session(self, episodes=1)
        s.rig.missing["top"] = "unplugged"  # Resume needs every device
        s.tick(round(2 * FPS))
        self.assertEqual(len([line for line in s.said if "can't start" in line]), 1)
        self.assertEqual(s.starts(), [])
        del s.rig.missing["top"]
        s.until(lambda: len(s.starts()) == 1, limit_s=10)

    def test_hold_on_the_page_keeps_the_next_attempt_from_starting(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.state() == "autonomous")
        self.assertFalse(s.page("autopilot_hold", hold=True).accepted, "only between attempts")
        s.until(lambda: s.state() == "resting")
        s.tick()  # the status is published before the autopilot hears of the arrival
        status = s.rig.status()["autopilot"]
        self.assertEqual((status["made"], status["of"], status["held"]), (1, 2, False))
        self.assertGreater(status["next_in_s"], 0)
        self.assertTrue(s.rig.status()["allowed"]["autopilot_hold"])
        self.assertTrue(s.page("autopilot_hold", hold=True).accepted)
        s.tick(round(3 * RESET_S * FPS))
        self.assertEqual(len(s.starts()), 1)
        status = s.rig.status()["autopilot"]
        self.assertTrue(status["held"])
        self.assertIsNone(status["next_in_s"])
        self.assertTrue(s.page("autopilot_hold", hold=False).accepted)
        continued = s.rig.clock()
        s.until(lambda: len(s.starts()) == 2, limit_s=10)
        self.assertGreaterEqual(s.rig.clock() - continued, RESET_S - 1 / FPS, "Continue restarts the reset wait")
        json.dumps(s.rig.status(), allow_nan=False)

    def test_a_hold_that_lands_before_the_autopilots_resume_still_holds(self) -> None:
        s = Session(self, episodes=1)
        c = s.rig.controller
        held = c.submit("autopilot_hold", source="terminal", hold=True)
        resumed = c.submit("resume", source="autopilot")  # as if the autopilot had sent it just after
        s.tick()
        self.assertTrue(held.result(timeout=0).accepted)
        self.assertFalse(resumed.result(timeout=0).accepted)
        self.assertEqual(s.state(), "resting")

    def test_after_the_autopilot_fails_hold_says_it_stopped_not_that_there_is_none(self) -> None:
        s = Session(self)

        def broken(facts):
            raise RuntimeError("autopilot bug")

        s.pilot.step = broken
        s.tick()
        result = s.page("autopilot_hold", hold=True)
        self.assertFalse(result.accepted)
        self.assertIn("autopilot stopped after an error", result.reason)
        self.assertNotIn("only so101 run has one", result.reason)

    def test_serve_has_no_autopilot_to_hold(self) -> None:
        rig = Rig()
        self.addCleanup(rig.close)
        self.assertIsNone(rig.status()["autopilot"])
        self.assertNotIn("autopilot_hold", rig.status()["allowed"])
        self.assertFalse(rig.command("autopilot_hold", hold=True).accepted)


if __name__ == "__main__":
    unittest.main()
