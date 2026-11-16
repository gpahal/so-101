"""`so101 run`'s autopilot (docs/run.md) on the controller's fake arm: it starts, ends and resets attempts as
another operator would, and hands an attempt to a person once they act in it.

The autopilot is stepped by hand after each tick, on the rig's fake clock, as its own thread steps it.
"""

import tempfile
import unittest
from pathlib import Path

from fakes import control
from test_controller import FPS, Asking, Rig

from so_101.control.autopilot import ASK_GRACE_S, Autopilot

LIMIT_S = 2.0
RESET_S = 1.0


class Session:
    """A recorded run on the fake arm (the autopilot reads its event log), with the autopilot over it."""

    def __init__(self, test: unittest.TestCase, episodes: int = 2, ask_at: int | None = None,
                 limit_s: float = LIMIT_S) -> None:
        wrap = (lambda driver, recorder, clock: Asking(driver, ask_at)) if ask_at else None
        folder = tempfile.TemporaryDirectory()
        test.addCleanup(folder.cleanup)
        self.rig = Rig(Path(folder.name) / "run", control=control(preview_fps=0, attempt_limit_s=limit_s), wrap=wrap)
        test.addCleanup(self.rig.close)
        self.said: list[str] = []
        c = self.rig.controller
        self.pilot = Autopilot(submit=c.submit, status=c.status, events=c.events, say=self.said.append,
                               clock=self.rig.clock, episodes=episodes, reset_s=RESET_S)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.rig.tick()
            self.pilot.step()

    def until(self, test, limit_s: float = 60.0) -> None:
        for _ in range(round(limit_s * FPS)):
            if test():
                return
            self.tick()
        raise AssertionError(f"not reached in {limit_s} s; state {self.rig.status()['state']}, said {self.said[-3:]}")

    def state(self) -> str:
        return self.rig.status()["state"]

    def starts(self) -> list[float]:
        return [e["t"] for e in self.rig.controller.events(0, 500) if e["type"] == "attempt_start"]


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
        commands = [e for e in events if e["type"] == "command"]
        self.assertTrue(all(e["source"] == "autopilot" for e in commands))
        self.assertEqual([e["command"] for e in commands], ["resume", "resume", "quit"])

    def test_an_unanswered_ask_ends_the_attempt_after_the_grace_period(self) -> None:
        s = Session(self, episodes=1, ask_at=5)
        s.until(lambda: s.state() == "paused")
        asked = s.rig.clock()
        s.until(lambda: s.state() == "returning")
        self.assertGreaterEqual(s.rig.clock() - asked, ASK_GRACE_S)
        resets = [e for e in s.rig.controller.events(0, 500) if e["type"] == "command" and e["command"] == "reset"]
        self.assertEqual([e["source"] for e in resets], ["autopilot"])

    def test_a_person_who_acts_takes_the_attempt_over(self) -> None:
        s = Session(self, episodes=2, ask_at=5, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        self.assertTrue(s.rig.command("resume").accepted)  # a person, at the terminal
        s.tick(round((ASK_GRACE_S + 1) * FPS))
        commands = [e for e in s.rig.controller.events(0, 500) if e["type"] == "command"]
        self.assertNotIn("reset", [e["command"] for e in commands if e["source"] == "autopilot"])
        s.rig.command("pause")
        s.tick(round(LIMIT_S * 2 * FPS))
        self.assertEqual(s.state(), "paused", "the autopilot never resumes an attempt a person took")
        s.rig.reset("failed")
        s.until(lambda: len(s.starts()) == 2, limit_s=10)  # and starts the next once it has ended

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


if __name__ == "__main__":
    unittest.main()
