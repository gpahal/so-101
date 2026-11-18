"""`so101 run`'s autopilot (docs/run.md) on the controller's fake arm: it starts, ends and resets attempts as
another operator would, and hands an attempt to a person once they act in it.

The controller steps the autopilot inside each tick, after people's commands and the link check, as it does in
`so101 run`. A person starts attempt 1 (`Session(start=True)`, r at the terminal), and the autopilot makes the rest,
with the phone's page in control and polling (`Session(watched=True)`): the autopilot moves nothing unwatched.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fakes import control
from test_controller import FPS, Asking, Rig

from so_101.control.autopilot import RELINK_MARGIN_S, Autopilot
from so_101.control.terminal import KeyReader
from so_101.policy.base import DONE

LIMIT_S = 2.0
RESET_S = 1.0
# How long after a page in control watches again the autopilot waits before its next start.
RELINK_S = max(RESET_S, control().link_timeout_s + RELINK_MARGIN_S)
WAIT_S = 30.0  # long enough that any timed Reset the autopilot made would have come
PHONE = "phone-page"
AFTER_FAULT = ("Autopilot waits. The arm was in Fault: the next attempt waits for you. r at the terminal or Resume on "
               "the page starts it, and the autopilot makes the rest.")  # the client id a control page made for itself
# The autopilot's Resume while the phone's page, in control, is out of reach (`Controller._unwatched`).
OUT_OF_REACH = ("The controlling page (100.64.0.2) is out of reach, so the autopilot starts nothing unwatched. Bring the "
                "page back or take control on another, and the next attempt starts the run's reset time "
                "(--reset-time), at least link_timeout_s + 0.5 s, after that; r at the robot's terminal starts it now.")
# The autopilot's Resume with no page in control (`Controller._unwatched`).
NO_PAGE = ("No control page is in control, so the autopilot moves nothing unwatched. Open the page and take control "
           "there, and the next attempt starts the run's reset time (--reset-time), at least link_timeout_s + 0.5 s, after that; r at "
           "the robot's terminal starts it now.")


class Session:
    """A recorded run on the fake arm, with the autopilot over it."""

    def __init__(self, test: unittest.TestCase, episodes: int = 2, ask_at: int | None = None,
                 limit_s: float = LIMIT_S, kind: str | None = None, start: bool = True, watched: bool = True,
                 **settings) -> None:
        """`settings` are more `[control]` settings, as so101.toml's. With `start`, a person starts attempt 1 at the
        terminal on the first tick, as `so101 run` needs. With `watched`, the phone's page takes control first and
        polls before every tick: the autopilot moves nothing unless a page in control watches the arm."""
        wrap = (lambda driver, recorder, clock: Asking(driver, ask_at, **({"kind": kind} if kind else {}))) \
            if ask_at else None
        folder = tempfile.TemporaryDirectory()
        test.addCleanup(folder.cleanup)
        self.told: list[str] = []  # what the controller tells the operator
        self.rig = Rig(Path(folder.name) / "run", control=control(preview_fps=0, attempt_limit_s=limit_s, **settings),
                       wrap=wrap,
                       notify=lambda message, state: self.told.append(message))
        test.addCleanup(self.rig.close)
        self.said: list[str] = []  # what the autopilot says
        c = self.rig.controller
        self.pilot = c.autopilot = Autopilot(command=c.pilot_command, say=self.said.append, episodes=episodes,
                                             reset_s=RESET_S, link_timeout_s=self.rig.controller.link.timeout_s)
        self.polling = False  # the phone's page polls before each tick
        if watched:
            self.polling = True
            test.assertTrue(self.page("take_control").accepted)
        if start:
            self.start()

    def start(self) -> None:
        """r at the robot's terminal, applied on the next tick: a person starts an attempt."""
        self.rig.controller.submit("resume", source="terminal")

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
        commands = [(e["command"], e["source"], e["accepted"]) for e in s.events("command")]
        # The phone's page takes control (the autopilot moves nothing unwatched), attempt 1 is a person's start, and
        # the autopilot makes attempt 2 and quits.
        self.assertEqual(commands, [("take_control", "page", True), ("resume", "terminal", True), ("resume", "autopilot", True),
                                    ("quit", "autopilot", True)])

    def test_the_first_attempt_waits_for_a_person(self) -> None:
        s = Session(self, episodes=2, start=False)
        s.tick(round(10 * FPS))
        self.assertEqual(s.starts(), [], "the autopilot never starts attempt 1")
        self.assertEqual(s.commands(), [])
        waits = ("Autopilot waits. Attempt 1 waits for you: r at the terminal or Resume on the page starts it, and the "
                 "autopilot makes the rest.")
        self.assertEqual([line for line in s.said if "Attempt 1" in line], [waits], "said once")
        status = s.rig.status()["autopilot"]
        self.assertIsNone(status["next_in_s"])
        self.assertIn("Attempt 1 waits for you", status["waiting"])
        s.start()
        s.until(lambda: len(s.starts()) == 2, limit_s=20)
        [first] = [e for e in s.events("command") if e["command"] == "resume" and e["source"] == "terminal"]
        rested = [e["t"] for e in s.events("state") if e.get("state") == "resting"]
        self.assertGreaterEqual(s.starts()[1] - rested[0], RESET_S - 1 / FPS, "attempt 2 after reset_s at rest")
        self.assertEqual(s.commands(), ["resume"])
        self.assertTrue(first["accepted"])

    def test_the_attempt_a_person_starts_is_run_like_any_other(self) -> None:
        # The r that starts attempt 1 doesn't hand it over: the autopilot ends it on a done ask as any other.
        s = Session(self, episodes=1, ask_at=5, kind=DONE)
        s.until(lambda: s.state() == "returning")
        self.assertEqual(s.commands(), ["reset"])

    def test_an_ask_waits_for_a_person(self) -> None:
        s = Session(self, episodes=1, ask_at=5, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "paused", "no Reset: a person decides")
        self.assertEqual(s.commands(), [])
        waits = ("Autopilot waits. Attempt 1 is paused: the autopilot leaves it to a person. r resumes it; the right "
                 "arrow or Reset ends it.")
        self.assertEqual([line for line in s.said if "paused" in line], [waits], "said once")
        s.rig.reset("failed")
        s.until(lambda: s.rig.controller.done, limit_s=10)
        self.assertEqual(s.commands(), ["quit"])

    def test_a_done_ask_ends_the_attempt_at_once(self) -> None:
        s = Session(self, episodes=1, ask_at=5, kind=DONE)
        s.until(lambda: s.state() == "paused")
        asked = s.rig.clock()
        s.until(lambda: s.state() == "returning", limit_s=1)
        self.assertLess(s.rig.clock() - asked, 0.2, "no wait")
        self.assertEqual(s.events("outcome"), [], "unlabelled")
        [paused] = [e for e in s.events("state") if e["state"] == "paused"]
        self.assertEqual(paused["ask"], DONE, "the verdict is in the events")

    def test_a_pause_in_the_tick_that_opens_an_attempt_hands_it_over(self) -> None:
        # Each command carries the attempt open when it arrived: the Pause came after the Resume that opened
        # attempt 1, in the same tick, so it hands attempt 1 over, and the done ask gets no Reset or Quit.
        s = Session(self, episodes=1, ask_at=5, kind=DONE, limit_s=60.0, start=False)
        for name in ("resume", "pause", "resume"):
            s.rig.controller.submit(name, source="terminal")
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(len(s.starts()), 1)
        self.assertEqual(s.commands(), [])

    def test_the_command_that_opens_an_attempt_still_doesnt_hand_it_over(self) -> None:
        # The Resume alone, in the first tick: it arrived with no attempt open, so the autopilot ends attempt 1
        # on its done ask, as for any other.
        s = Session(self, episodes=1, ask_at=5, kind=DONE, limit_s=60.0, start=False)
        s.start()
        s.until(lambda: s.state() == "returning")
        self.assertEqual(s.commands(), ["reset"])

    def test_a_person_who_acts_takes_the_attempt_over(self) -> None:
        s = Session(self, episodes=2, ask_at=5, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        self.assertTrue(s.rig.command("resume").accepted)  # a person, at the terminal
        s.tick(round(WAIT_S * FPS))
        self.assertNotIn("reset", s.commands())
        s.rig.command("pause")
        s.tick(round(LIMIT_S * 2 * FPS))
        self.assertEqual(s.state(), "paused", "the autopilot never resumes an attempt a person took")
        s.rig.reset("failed")
        s.until(lambda: len(s.starts()) == 2, limit_s=10)  # and starts the next once it has ended

    def test_check_pose_a_label_or_taking_control_never_take_the_attempt_over(self) -> None:
        # All three in the tick the autopilot first sees the done ask: it still ends the attempt.
        s = Session(self, episodes=1, ask_at=5, kind=DONE, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        c = s.rig.controller
        c.submit("pose", source="terminal")
        c.submit("label", source="terminal", label="stuck")
        c.submit("take_control", source="page", client=PHONE, who="100.64.0.2")
        s.tick()
        # Past the setup: the phone's take_control (`Session(watched=True)`) and the person's r.
        commands = [(e["command"], e["source"], e["accepted"]) for e in s.events("command")][2:]
        self.assertEqual(commands, [("pose", "terminal", True), ("label", "terminal", True),
                                    ("take_control", "page", True), ("reset", "autopilot", True)],
                         "the person's first, then the autopilot's, in one tick")
        self.assertEqual(s.state(), "returning")

    def test_the_right_arrow_hands_the_attempt_over(self) -> None:
        # The terminal's question tells the controller once, in the tick the autopilot would end the attempt
        # for a done ask: it then never ends it, however long the answer takes.
        s = Session(self, episodes=2, ask_at=5, kind=DONE, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        keys = KeyReader(s.rig.controller.submit, s.rig.status)
        with redirect_stdout(io.StringIO()):
            keys.on_key("right")
            s.tick(round(WAIT_S * FPS))
            self.assertEqual(s.state(), "paused", "no Reset once a person started answering")
            keys.on_key("s")
            s.until(lambda: s.state() == "resting")
        self.assertEqual([e["outcome"] for e in s.events("outcome")], ["succeeded"])
        self.assertEqual(s.commands(), [], "the autopilot sent no Reset")
        self.assertIn(("answering", True), [(e["command"], e["accepted"]) for e in s.events("command")
                                            if e["source"] == "terminal"], "answering is logged")

    def test_the_reset_sheet_hands_the_attempt_over_even_when_cancelled(self) -> None:
        s = Session(self, episodes=2, ask_at=5, kind=DONE, limit_s=60.0)
        s.polling = True
        s.until(lambda: s.state() == "paused")
        attempt = s.rig.status()["attempt"]
        self.assertTrue(s.page("answering", for_attempt=attempt).accepted)  # the sheet opens
        s.tick(round(WAIT_S * FPS))  # and is cancelled: the page sends nothing more
        self.assertEqual(s.state(), "paused")
        self.assertEqual(s.commands(), [], "the autopilot left the attempt to the person")
        self.assertTrue(s.page("reset").accepted)  # the person ends it
        s.until(lambda: s.state() == "resting")
        self.assertEqual(s.commands(), [])

    def test_a_refused_pause_from_the_page_in_control_hands_the_attempt_over(self) -> None:
        s = Session(self, episodes=2, ask_at=5, kind=DONE, limit_s=60.0)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "paused")
        refused = s.page("pause")  # already paused: refused, but a person is at the page and wants it held
        self.assertEqual((refused.accepted, refused.reason), (False, "Already paused."))
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "paused")
        self.assertEqual(s.commands(), [], "no Reset after the person's Pause")

    def test_a_page_not_in_control_hands_nothing_over(self) -> None:
        s = Session(self, episodes=2, ask_at=5, kind=DONE, limit_s=60.0)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "paused")
        other = s.rig.command("pause", source="page", client="old-tab", who="100.64.0.9")
        self.assertFalse(other.accepted)
        self.assertIn("Another page has control", other.reason)
        self.assertEqual(s.state(), "returning", "the done ask still ended the attempt")
        self.assertEqual(s.commands(), ["reset"])
        # Between attempts, that tab's commands don't push the next start back either.
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        rested = s.rig.clock()
        for _ in range(round(RESET_S * FPS) // 3):
            s.rig.command("pose", source="page", client="old-tab", who="100.64.0.9")
            s.tick(2)
        s.until(lambda: len(s.starts()) == 2, limit_s=5)
        self.assertLess(s.starts()[1] - rested, RESET_S + 0.2)

    def test_a_quit_from_a_page_never_stops_it(self) -> None:
        s = Session(self, episodes=3)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: len(s.starts()) == 2)
        refused = s.page("quit")  # always refused from a page
        self.assertFalse(refused.accepted)
        self.assertFalse(s.pilot.done)
        s.until(lambda: len(s.starts()) == 3, limit_s=20)
        self.assertNotIn("Autopilot: stopped, after your Quit.", s.said)

    def test_answering_never_brings_a_lost_link_back(self) -> None:
        # A page out of sight that sends answering (a timer an older page ran, say) must not bring the link
        # back: the next attempt would then start with nobody watching.
        s = Session(self, episodes=3)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: len(s.starts()) == 1)
        attempt = s.rig.status()["attempt"]
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        self.assertTrue(s.page("page_hidden").accepted)
        s.polling = False
        s.tick(round((RESET_S + 2) * FPS))
        self.assertFalse(s.page("answering", for_attempt=attempt).accepted, "attempt 1 has ended")
        self.assertEqual(s.events("link_restored"), [])
        s.tick(round((RESET_S + 1) * FPS))
        self.assertEqual(len(s.starts()), 1, "no start while the page stays hidden")

    def test_answering_is_only_for_the_open_attempt_and_takes_no_open(self) -> None:
        s = Session(self, episodes=1, ask_at=5, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        attempt = s.rig.status()["attempt"]
        self.assertFalse(s.page("answering", for_attempt=attempt + 1).accepted)
        self.assertFalse(s.page("answering", for_attempt=str(attempt)).accepted)
        refused = s.page("answering", for_attempt=attempt, open=False)
        self.assertEqual((refused.accepted, refused.reason), (False, "Answering takes no open."))

    def test_a_link_loss_pause_waits_for_a_person(self) -> None:
        # The arm holds while the page is out of reach, and after it's back: the autopilot never moves the arm
        # for a lost link. A person ends the attempt.
        s = Session(self, episodes=2, limit_s=60.0)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "autonomous")
        s.polling = False  # the phone's page is gone
        s.until(lambda: s.state() == "paused")
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "paused", "no Reset while the page is out of reach")
        self.assertEqual(s.commands(), [])
        handed = ("Autopilot: attempt 1 goes to a person, as the control page's link was lost: the autopilot sends "
                  "nothing more for it. r resumes it; the right arrow or Reset ends it.")
        self.assertEqual([line for line in s.said if "link" in line], [handed], "said once")

        # The page is back: the attempt is still the person's.
        s.polling = True
        s.tick()
        self.assertEqual(len(s.events("link_restored")), 1)
        s.tick(round(WAIT_S * FPS))
        self.assertEqual((s.state(), s.commands()), ("paused", []))

        # A person Resets from the terminal; the next attempt starts reset_s after the arm reaches rest.
        self.assertTrue(s.rig.command("reset").accepted)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        rested = s.rig.clock()
        s.until(lambda: len(s.starts()) == 2, limit_s=RESET_S + 2)
        self.assertGreaterEqual(s.rig.clock() - rested, RESET_S)
        self.assertEqual(s.events("outcome"), [])

    def test_a_link_lost_after_an_ask_gets_no_reset(self) -> None:
        # The council's case: the policy asks, then the phone locks. The arm stays where it is while the page is
        # out of reach, and after it's back.
        s = Session(self, episodes=2, ask_at=5, limit_s=60.0)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "paused")
        self.assertTrue(s.page("page_hidden").accepted)
        s.polling = False
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "paused")
        self.assertEqual(s.commands(), [])
        s.polling = True
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(len(s.events("link_restored")), 1)
        self.assertEqual((s.state(), s.commands()), ("paused", []), "the attempt is the person's for good")

    def test_a_done_ask_with_the_link_lost_gets_no_reset(self) -> None:
        # The phone locks in the tick the autopilot first sees the done ask: no Reset, then or once it's back.
        s = Session(self, episodes=2, ask_at=5, kind=DONE, limit_s=60.0)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "paused")
        s.polling = False
        self.assertTrue(s.page("page_hidden").accepted)
        self.assertEqual(s.state(), "paused")
        s.tick(round(WAIT_S * FPS))
        s.polling = True
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(len(s.events("link_restored")), 1)
        self.assertEqual((s.state(), s.commands()), ("paused", []))

    def test_after_a_link_loss_the_next_attempt_waits_for_the_page(self) -> None:
        s = Session(self, episodes=2, limit_s=60.0)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: s.state() == "autonomous")
        s.polling = False
        s.until(lambda: s.state() == "paused")
        self.assertTrue(s.rig.command("reset").accepted)  # a person at the terminal ends the attempt
        s.tick(round((RESET_S + WAIT_S) * FPS))
        self.assertEqual(len(s.starts()), 1, "the autopilot starts nothing unwatched")
        self.assertEqual((s.state(), s.rig.controller.at_rest), ("resting", True))
        self.assertEqual(s.rig.status()["autopilot"]["waiting"], f"The next attempt can't start ({OUT_OF_REACH}).")

        # The page is back: the next attempt starts RELINK_S later, not at once.
        s.polling = True
        s.tick()
        [restored] = s.events("link_restored")
        s.tick(round(RELINK_S * FPS) - 2)
        self.assertEqual(len(s.starts()), 1, "no start within RELINK_S of the restore")
        s.until(lambda: len(s.starts()) == 2, limit_s=2)
        self.assertGreaterEqual(s.starts()[1] - restored["t"], RELINK_S)

    def test_a_page_hidden_between_attempts_holds_the_next_start_until_it_is_back(self) -> None:
        s = Session(self, episodes=2)
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        s.until(lambda: len(s.starts()) == 1)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)  # the time limit ended attempt 1
        self.assertTrue(s.page("page_hidden").accepted)  # the phone's screen locks, with no attempt open
        s.polling = False
        s.tick(round((RESET_S + WAIT_S) * FPS))
        self.assertEqual(len(s.starts()), 1, "no start while the page is hidden")
        self.assertEqual(s.rig.status()["autopilot"]["waiting"], f"The next attempt can't start ({OUT_OF_REACH}).")
        s.polling = True  # the screen is unlocked
        s.until(lambda: len(s.starts()) == 2, limit_s=RELINK_S + 2)
        [restored] = s.events("link_restored")
        self.assertGreaterEqual(s.starts()[1] - restored["t"], RELINK_S, "the wait counts from the restore")

    # The user chose that the autopilot moves the arm only while a page in control watches it, whatever
    # on_link_loss says. These two replace the tests that a lost link with on_link_loss "continue", or no page
    # ever in control, left the autopilot starting attempts as before.
    def test_with_on_link_loss_continue_a_lost_link_still_stops_the_autopilot(self) -> None:
        s = Session(self, episodes=2, ask_at=round(10 * FPS), kind=DONE, limit_s=60.0, on_link_loss="continue")
        s.until(lambda: s.state() == "autonomous")
        s.polling = False  # the phone's page is gone
        s.until(lambda: len(s.events("link_lost")) == 1, limit_s=10)
        self.assertEqual(s.state(), "autonomous", "the policy keeps running, as continue says")
        s.until(lambda: s.state() == "paused", limit_s=10)  # the done ask
        s.tick(round(WAIT_S * FPS))
        self.assertEqual((s.state(), s.commands()), ("paused", []), "no Reset for an unwatched attempt")
        handed = ("Autopilot: attempt 1 goes to a person, as the control page's link was lost: the autopilot sends "
                  "nothing more for it. r resumes it; the right arrow or Reset ends it.")
        self.assertEqual([line for line in s.said if "goes to a person" in line], [handed], "said once")
        self.assertTrue(s.rig.command("reset").accepted)  # a person ends it at the terminal
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(len(s.starts()), 1, "nothing starts until the page polls again")
        self.assertEqual(s.rig.status()["autopilot"]["waiting"], f"The next attempt can't start ({OUT_OF_REACH}).")
        s.polling = True
        s.until(lambda: len(s.starts()) == 2, limit_s=RELINK_S + 2)
        [restored] = s.events("link_restored")
        self.assertGreaterEqual(s.starts()[1] - restored["t"], RELINK_S)

    def test_with_no_page_in_control_the_autopilot_moves_nothing(self) -> None:
        s = Session(self, watched=False, episodes=2, ask_at=5, kind=DONE, limit_s=60.0)
        s.until(lambda: s.state() == "paused")  # r at the terminal started attempt 1; the policy asked done
        s.tick(round(WAIT_S * FPS))
        self.assertEqual((s.state(), s.commands()), ("paused", []), "no Reset on the done ask")
        handed = ("Autopilot: attempt 1 goes to a person, as no control page is in control: the autopilot sends "
                  "nothing more for it. r resumes it; the right arrow or Reset ends it.")
        self.assertEqual([line for line in s.said if "goes to a person" in line], [handed], "said once")
        self.assertTrue(s.rig.command("reset").accepted)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(len(s.starts()), 1, "no attempt 2 with no page in control")
        self.assertEqual(s.commands(), [])
        self.assertEqual(s.rig.status()["autopilot"]["waiting"], f"The next attempt can't start ({NO_PAGE}).")
        s.polling = True
        self.assertTrue(s.page("take_control").accepted)
        [taken] = s.events("control_taken")
        s.until(lambda: len(s.starts()) == 2, limit_s=RELINK_S + 2)
        self.assertGreaterEqual(s.starts()[1] - taken["t"], RELINK_S - 1 / FPS, "RELINK_S after the page took control")
        self.assertEqual(s.commands(), ["resume"])

    # A page in control can watch again with no poll behind it: a command or a poll that reaches the server after
    # page_hidden, or a take_control that races a hide, hears the page (link.py). Such a link is lost again
    # link_timeout_s later, so the autopilot waits longer than that (RELINK_S) before its next start.
    def assert_a_restore_no_poll_follows_starts_nothing(self, s: Session) -> None:
        """Tick RELINK_S + 2 s with no polls after the link came back: no start, and the link is lost again."""
        s.polling = False
        s.tick(round((RELINK_S + 2) * FPS))
        self.assertEqual(len(s.starts()), 1, "no start on a restore that no poll follows")
        [*_, restored] = s.events("link_restored") or s.events("control_taken")
        self.assertTrue([e for e in s.events("link_lost") if e["t"] > restored["t"]], "the restore timed out")

    def rest_after_attempt_1_then_hide(self, s: Session) -> None:
        """Run to rest after attempt 1, then hide the phone's page: its polls stop, and 3 s pass."""
        s.until(lambda: len(s.starts()) == 1)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest
                and s.rig.status()["autopilot"]["made"] == 1)
        s.polling = False
        self.assertTrue(s.page("page_hidden").accepted)
        s.tick(round(3 * FPS))

    def assert_the_page_back_starts_attempt_2_after_relink_s(self, s: Session) -> None:
        s.polling = True
        s.until(lambda: len(s.starts()) == 2, limit_s=RELINK_S + 2)
        [*_, restored] = s.events("link_restored")
        self.assertGreaterEqual(s.starts()[1] - restored["t"], RELINK_S)

    def test_a_late_command_from_a_hidden_page_starts_nothing(self) -> None:
        s = Session(self, episodes=3)
        self.rest_after_attempt_1_then_hide(s)
        self.assertTrue(s.page("pose").accepted)  # sent before the hide, it reaches the server after
        self.assertEqual(len(s.events("link_restored")), 1, "the late command heard the page")
        self.assert_a_restore_no_poll_follows_starts_nothing(s)
        self.assert_the_page_back_starts_attempt_2_after_relink_s(s)

    def test_a_late_poll_after_page_hidden_starts_nothing(self) -> None:
        s = Session(self, episodes=3)
        self.rest_after_attempt_1_then_hide(s)
        s.rig.controller.heartbeat(PHONE)  # a visible poll, sent before the hide, reaches the server after
        s.tick()
        self.assertEqual(len(s.events("link_restored")), 1, "the late poll heard the page")
        self.assert_a_restore_no_poll_follows_starts_nothing(s)
        self.assert_the_page_back_starts_attempt_2_after_relink_s(s)

    def test_a_take_control_with_no_polls_starts_nothing(self) -> None:
        s = Session(self, watched=False, episodes=2)  # a person's r starts attempt 1, and its time limit ends it
        s.until(lambda: len(s.starts()) == 1)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest
                and s.rig.status()["autopilot"]["made"] == 1)
        self.assertTrue(s.page("take_control").accepted)  # a take_control that raced a hide: no poll follows
        self.assert_a_restore_no_poll_follows_starts_nothing(s)

    def test_a_start_that_stops_short_waits_for_a_person(self) -> None:
        s = Session(self, episodes=1)
        # The shoulder rises from rest to the start pose, and something stops it on the way.
        s.rig.motors.stops = {"shoulder_lift": s.rig.motors.present()["shoulder_lift"] + 100}
        s.until(lambda: s.state() == "paused")
        s.rig.motors.stops = {}
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "paused", "no Reset: it would drive the arm back the way it came")
        self.assertEqual(s.commands(), [])
        self.assertEqual(len([line for line in s.said if "leaves it to a person" in line]), 1)
        self.assertTrue(s.rig.command("reset").accepted)
        s.until(lambda: s.rig.controller.done)
        self.assertEqual(s.commands(), ["quit"])

    def test_a_return_to_rest_that_stops_short_waits_for_a_person(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.state() == "autonomous")
        # Something blocks the elbow's way back up to rest, and stays there.
        s.rig.motors.stops = {"elbow_flex": s.rig.motors.present()["elbow_flex"]}
        s.until(lambda: s.state() == "paused")  # the time limit's return stopped short
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "paused", "it never drives the arm at what blocks it")
        self.assertEqual(s.commands(), [], "no Reset")
        self.assertEqual(len([line for line in s.said if "leaves it to a person" in line]), 1, "said once")
        self.assertIn("leaves it to a person", s.rig.status()["autopilot"]["waiting"])

        # A person clears the way and Resets: the arm reaches rest and the run carries on.
        s.rig.motors.stops = {}
        self.assertTrue(s.rig.command("reset").accepted)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        self.assertIsNone(s.rig.status()["autopilot"]["waiting"])
        s.until(lambda: s.rig.controller.done, limit_s=30)
        self.assertEqual(len(s.starts()), 2)
        self.assertEqual(s.commands(), ["resume", "quit"])

    def test_a_persons_pause_on_the_way_to_rest_takes_that_attempt_over(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.state() == "returning")  # the time limit ended attempt 1
        self.assertTrue(s.rig.command("pause").accepted)
        self.assertEqual(s.state(), "paused", "the attempt stays open")
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "paused", "it's the person's now")
        s.rig.command("reset")
        s.until(lambda: s.state() == "resting")
        rested = s.rig.clock()
        s.until(lambda: len(s.starts()) == 2, limit_s=10)
        self.assertGreaterEqual(s.rig.clock() - rested, RESET_S, "the reset wait counts from the arrival at rest")
        self.assertEqual(s.commands(), ["resume"])

    def test_a_fault_on_the_last_attempt_waits_for_a_reset_and_quits_only_at_rest(self) -> None:
        s = Session(self, episodes=1, limit_s=60.0)
        s.until(lambda: s.state() == "autonomous")
        s.rig.motors.failing = True
        s.tick()
        s.rig.motors.failing = False
        s.tick(round(10 * FPS))
        self.assertEqual(s.state(), "fault")
        self.assertEqual(s.commands(), [], "no quit in Fault")
        self.assertFalse(s.rig.status()["confirm_quit"])
        self.assertEqual(len([line for line in s.said if "Fault" in line]), 1, "it says why once")

        # Reset, then a Pause on the way: the arm rests away from rest. The run is over, but not there.
        self.assertTrue(s.rig.command("reset").accepted)
        s.tick(5)
        self.assertTrue(s.rig.command("pause").accepted)
        self.assertEqual(s.state(), "resting")
        self.assertFalse(s.rig.controller.at_rest)
        s.tick(round(10 * FPS))
        self.assertEqual(s.commands(), [], "no quit away from rest")
        self.assertFalse(s.rig.controller.done)
        self.assertEqual(len([line for line in s.said if "at rest" in line]), 1)

        s.rig.command("reset")
        s.until(lambda: s.rig.controller.done, limit_s=20)
        self.assertEqual(s.commands(), ["quit"])
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
        s.tick(round(10 * FPS))
        self.assertEqual(len(s.starts()), 1, "no attempt starts away from rest")
        s.rig.command("reset")
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        # The user chose this: a fault's Reset brings the arm to rest and doesn't restart the policy. Before, the
        # autopilot started attempt 2 RESET_S after rest; now it waits for a person, as for attempt 1.
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(len(s.starts()), 1, "no attempt starts by itself after a fault")
        self.assertEqual(s.commands(), [])
        self.assertEqual([line for line in s.said if "was in Fault" in line], [AFTER_FAULT], "said once")
        s.start()
        s.until(lambda: len(s.starts()) == 2, limit_s=10)
        self.assertEqual(s.commands(), [], "attempt 2 is the person's start")

    def test_a_fault_reset_in_the_next_tick_still_waits_for_a_person(self) -> None:
        # The fault comes at the end of one tick and the Reset at the start of the next, so the state goes from
        # Fault to Returning before the autopilot steps: it never sees "fault", and the controller tells it.
        # An error in the tick's last step (the loop's rate), after the autopilot: the follower and the policy
        # stay sound, so the Reset is accepted at once and the next attempt could start.
        s = Session(self, episodes=2, limit_s=60.0)
        s.until(lambda: s.state() == "autonomous")
        pace = s.rig.controller.pace
        tick = pace.tick

        def broken():
            pace.tick = tick
            raise RuntimeError("loop bug")

        pace.tick = broken
        s.tick()
        self.assertEqual(s.state(), "fault")
        seen = []
        pilot_step = s.pilot.step
        s.pilot.step = lambda f: (seen.append(f.state), pilot_step(f))
        s.rig.controller.submit("reset", source="terminal")
        s.tick()
        self.assertEqual(seen, ["returning"], "the autopilot never saw Fault")
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest, limit_s=20)
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(len(s.starts()), 1, "no attempt starts by itself after a fault")
        self.assertEqual(s.commands(), [])

    def test_after_a_person_restarts_following_a_fault_the_autopilot_carries_on(self) -> None:
        s = Session(self, episodes=3, limit_s=LIMIT_S)
        s.until(lambda: s.state() == "autonomous")
        s.rig.motors.failing = True
        s.tick()
        s.rig.motors.failing = False
        s.tick(2 * FPS)
        s.rig.command("reset")
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        s.tick(round(5 * FPS))
        self.assertEqual(len(s.starts()), 1)
        s.start()  # the person's r after the fault starts attempt 2
        s.until(lambda: len(s.starts()) == 2, limit_s=10)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest, limit_s=20)  # its time limit ended it
        rested = s.rig.clock()
        s.until(lambda: len(s.starts()) == 3, limit_s=10)
        self.assertGreaterEqual(s.rig.clock() - rested, RESET_S - 1 / FPS, "attempt 3 by itself, RESET_S after rest")
        self.assertEqual(s.commands(), ["resume"])

    def test_a_person_s_quit_stops_it(self) -> None:
        s = Session(self, episodes=3)
        s.until(lambda: s.state() == "autonomous")
        s.rig.command("quit")
        s.until(lambda: s.rig.controller.done)
        self.assertEqual(len(s.starts()), 1)

    def test_a_refused_start_is_said_once_and_waited_out(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: len(s.starts()) == 1)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        s.rig.missing["top"] = "unplugged"  # Resume needs every device
        s.tick(round((RESET_S + 2) * FPS))
        self.assertEqual(len([line for line in s.said if "can't start" in line]), 1)
        self.assertEqual(len(s.starts()), 1)
        del s.rig.missing["top"]
        s.until(lambda: len(s.starts()) == 2, limit_s=10)

    def test_hold_on_the_page_keeps_the_next_attempt_from_starting(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.state() == "autonomous")
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

    def test_a_hold_during_an_attempt_stops_the_next_one_not_this_one(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.state() == "autonomous")
        self.assertTrue(s.rig.status()["allowed"]["autopilot_hold"])
        self.assertTrue(s.page("autopilot_hold", hold=True).accepted)
        self.assertEqual(s.state(), "autonomous", "the attempt runs on")
        self.assertTrue(any("doesn't start until Continue" in line and "runs on" in line for line in s.told), s.told)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest, limit_s=20)
        ends = [e.get("ended_by") for e in s.events("state") if e.get("state") == "returning"]
        self.assertEqual(ends, ["time_limit"], "it ended at its time limit, as without the Hold")
        s.tick(round(3 * RESET_S * FPS))
        self.assertEqual(len(s.starts()), 1, "the next one doesn't start")
        self.assertTrue(s.rig.status()["autopilot"]["held"])
        self.assertTrue(s.page("autopilot_hold", hold=False).accepted)
        s.until(lambda: len(s.starts()) == 2, limit_s=10)

    def test_a_held_autopilot_still_quits_at_rest_after_its_last_attempt(self) -> None:
        s = Session(self, episodes=1)
        s.until(lambda: s.state() == "autonomous")
        self.assertTrue(s.page("autopilot_hold", hold=True).accepted)
        s.until(lambda: s.rig.controller.done, limit_s=20)
        self.assertEqual(s.commands(), ["quit"])
        self.assertTrue(s.rig.controller.at_rest)

    def test_hold_works_with_no_reset_time(self) -> None:
        s = Session(self, episodes=2)
        s.pilot.reset_s = 0.0  # --reset-time 0: the arm rests for a tick or two between attempts
        s.until(lambda: s.state() == "autonomous")
        s.tick()
        self.assertTrue(s.page("autopilot_hold", hold=True).accepted)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest, limit_s=20)
        s.tick(round(2 * FPS))
        self.assertEqual(len(s.starts()), 1)

    # The tick takes people's commands, then steps the autopilot, which carries its command out at once: a
    # person's command queued before the tick in which the autopilot would act is applied first, and the
    # autopilot decides from the facts it made. Each test below makes the autopilot's command due in the
    # next tick, queues a person's, and ticks once.

    def test_a_persons_resume_queued_as_a_done_ask_ends_it_wins(self) -> None:
        s = Session(self, episodes=2, ask_at=5, kind=DONE, limit_s=60.0)
        s.until(lambda: s.state() == "paused")  # the autopilot's Reset is due on the next tick
        s.rig.controller.submit("resume", source="terminal")
        s.tick()
        self.assertEqual(s.state(), "autonomous")
        self.assertEqual(s.commands(), [], "the autopilot sent no Reset")
        s.tick(round(WAIT_S * FPS))
        self.assertEqual(s.state(), "autonomous", "nothing ends the attempt the person resumed")

    def test_a_person_starting_to_answer_as_a_done_ask_ends_it_keeps_the_attempt(self) -> None:
        s = Session(self, episodes=1, ask_at=5, kind=DONE, limit_s=60.0)
        s.until(lambda: s.state() == "paused")
        s.rig.controller.submit("answering", source="terminal", for_attempt=s.rig.status()["attempt"])
        s.tick(round(2 * FPS))
        self.assertEqual(s.state(), "paused", "the attempt waits for the person's answer")
        self.assertEqual(s.commands(), [])
        s.rig.reset("succeeded")
        s.until(lambda: s.state() == "resting")
        self.assertEqual([e["outcome"] for e in s.events("outcome")], ["succeeded"])

    def test_a_hold_queued_before_the_resume_is_due_wins(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: len(s.starts()) == 1)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        s.pilot.ready_at = s.rig.clock() - 1  # the next start is due
        self.assertTrue(s.page("autopilot_hold", hold=True).accepted)  # submitted, then one tick
        s.tick(round(3 * RESET_S * FPS))
        self.assertEqual(len(s.starts()), 1)
        self.assertEqual(s.commands(), [], "the autopilot never sent the Resume")
        self.assertEqual(s.state(), "resting")

    def test_a_persons_resume_as_the_last_quit_is_due_gets_its_attempt(self) -> None:
        s = Session(self, episodes=1)
        s.until(lambda: len(s.starts()) == 1)
        # Attempt 1 ends at its time limit; the status shows the arm at rest a tick before the autopilot's Quit.
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest)
        self.assertFalse(s.rig.controller.done)
        s.rig.controller.submit("resume", source="terminal")
        s.tick()
        self.assertFalse(s.rig.controller.done, "the person's Resume came first")
        self.assertEqual(len(s.starts()), 2)
        s.until(lambda: s.rig.controller.done, limit_s=30)  # it quits once the person's attempt has ended at rest
        self.assertEqual(s.commands(), ["quit"])

    def test_the_autopilot_acts_in_the_tick_it_decides(self) -> None:
        s = Session(self, episodes=2)
        s.until(lambda: s.state() == "resting" and s.rig.controller.at_rest and len(s.starts()) == 1)
        s.tick()
        s.pilot.ready_at = s.rig.clock() - 1
        s.tick()
        self.assertEqual(s.state(), "starting", "its Resume was carried out in the same tick")

    def test_submit_refuses_the_autopilot_source(self) -> None:
        s = Session(self, episodes=1)
        with self.assertRaises(ValueError):
            s.rig.controller.submit("resume", source="autopilot")

    def test_a_refused_command_is_said_once_not_sent_every_tick(self) -> None:
        from so_101.control.autopilot import Facts
        from so_101.control.commands import Result

        sent, said = [], []

        def refuse(name):
            sent.append(name)
            return Result(name, False, "not now")

        pilot = Autopilot(command=refuse, say=said.append, episodes=2, reset_s=0.0, link_timeout_s=3.0)
        paused = dict(state="paused", at_rest=False, attempt=1, attempt_open=True, ask=DONE)
        for i in range(60):
            pilot.step(Facts(now=100.0 + i / FPS, **paused))
        self.assertEqual(sent, ["reset"], "sent once, not every tick")
        self.assertEqual([line for line in said if "refused" in line], ["Autopilot waits. Reset was refused: not now"])
        pilot.step(Facts(now=103.0, people=(("label", None),), **paused))  # a person acts: it may try again
        self.assertEqual(sent, ["reset", "reset"])

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
