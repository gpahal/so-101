"""Keys at the robot's own terminal. The reader only submits commands; the control loop applies them.

It uses LeRobot's TerminalKeyListener, which reads the terminal in cbreak mode with echo off and
restores it on exit. The arrows and Esc mean what they mean in LeRobot's recording and rollout:
the right arrow ends the attempt and asks how it went, the left arrow ends it as one to ignore.
The answer applies only to the attempt the right arrow asked about: the question is dropped once the
attempt has ended, except in Fault, where the answer labels the attempt the fault saved, and the
controller refuses it once another attempt is the current one. With no attempt open (in Fault,
where the attempt was already saved, or resting away from rest) the right arrow resets to rest
without asking. After a time limit ends an attempt, s, f or i answers the controller's outcome prompt. c runs the
checks and prints their report, o the pose against the demos' start pose. Space only pauses, and r
starts or resumes: a terminal sends no key-up, so it can't tell a held Space from a fresh press, and
once the OS's key-repeat delay runs past the control loop's 1 s rule a held Space would start the arm
again. On the page, where key-up is seen, Space still toggles. Esc quits and y confirms exiting where
the arm can't get to rest, by the rules in `states`.
"""

import contextlib
import os
import select
import sys
import termios
from functools import partial

from so_101.control.states import CONFIRM_S

KEYS = {
    "space": "pause",  # pause only: r starts and resumes
    "p": "pause",
    "r": "resume",
    "b": "rewind",  # back
    "t": "teleop",
    "c": "check",  # Run checks
    "o": "pose",  # Check pose
    "esc": "quit",  # LeRobot: stop
    "y": "exit_here",  # after Esc's warning: exit where the arm can't get to rest, motors off
}
HELP_KEYS = {"h", "?"}
# What a refused Space adds, by the state it was pressed in: the states where the page's Space would
# start or resume.
SPACE_REFUSED = {
    "resting": "Space only pauses at the terminal: r starts an attempt.",
    "paused": "Space only pauses at the terminal: r resumes.",
}

# The right arrow (LeRobot: end the episode) asks for one of these, then ends the attempt with it.
# On their own they answer the prompt after a time limit.
OUTCOME_KEYS = {"s": "succeeded", "f": "failed", "i": "ignored"}
ASK = "How did it go? s succeeded, f failed, i ignore this attempt. Any other key keeps it going."

HINTS = {
    "resting": "r starts an attempt, t starts one with the leader, c runs the checks, Esc quits (away "
               "from the rest pose, right arrow returns to rest, or Esc then y exits there: support the arm first).",
    "starting": "Space pauses, right arrow ends the attempt, left arrow ends it as ignored, Esc ends it and quits.",
    "autonomous": "Space pauses, b rewinds, t takes over with the leader, right arrow ends the attempt, left arrow "
                  "ends it as ignored, Esc ends it and quits.",
    "paused": "r resumes the policy, b rewinds, t takes over with the leader, c runs the checks, right arrow ends "
              "the attempt, left arrow ends it as ignored, Esc ends it and quits.",
    "handover": "Space pauses, right arrow ends the attempt, Esc ends it and quits.",
    "teleop": "Space pauses with both arms held, b rewinds, r gives the arm back to the policy, right arrow ends the "
              "attempt, left arrow ends it as ignored, Esc ends it and quits.",
    "rewinding": "b goes back further, Space stops the rewind and holds, right arrow ends the attempt, Esc ends it "
                 "and quits.",
    "returning": "Space pauses and keeps the attempt open, and cancels a quit. Esc quits once at rest.",
    "fault": "c runs the checks, right arrow returns to rest once the follower answers (the attempt was saved at the "
             "fault), Esc then y quits where the arm is: support it first, as the motors turn off.",
}

HELP = (
    "Keys: Space or p pause (Space never resumes here, since a terminal can't see it let go); r start or resume; "
    "b rewind the last moves; t take over with the leader; c run the checks (while resting, paused or after a "
    "fault); o check the pose; h or ? show these keys; right arrow end the attempt and return to rest (then s "
    "succeeded, f failed or i ignored); left arrow end it as ignored; Esc or Ctrl+C end it unlabelled, drive the "
    "arm to rest and quit with the motors off. Where the arm can't get to rest (after a fault, when the move stops "
    f"short, or when it rests away from the rest pose), Esc says so, and y within {CONFIRM_S:g} s quits there once "
    "you support the arm; another Esc only repeats the warning. After a time limit ends an attempt, s, f or i labels "
    "it."
)


def say(message: str, state: str) -> None:
    """Print `message` for the operator. Once the terminal is closed, printing fails (EIO), and the
    message is dropped: the controller calls this inside a tick, and a closed terminal's SIGHUP is
    then quitting the arm to rest, which must not fault."""
    try:
        print(f"{message} {HINTS.get(state, '')}".rstrip(), flush=True)
    except OSError:
        pass


def report(future, note: str | None = None) -> None:
    """Print why a command from the terminal was refused, then `note`, once the loop has applied it."""
    result = future.result()
    if not result.accepted:
        name = result.command.replace("_", " ").capitalize()
        print(f"{name} refused: {result.reason}{f' {note}' if note else ''}", flush=True)
    elif result.report is not None:
        result.report.print("Checks on the open devices:" if result.command == "check" else
                            "The follower against the demos' start pose:")
        print(result.report.summary(), flush=True)


def listener(on_key):
    """LeRobot's terminal key listener, made to stop reading once its terminal reports end of file
    or an I/O error, as when the terminal window closes. LeRobot's own keeps polling a terminal that's
    gone, and its select returns at once every time, so it would spin a CPU core beside the control
    loop."""
    from lerobot.utils.keyboard_input import TerminalKeyListener

    class Listener(TerminalKeyListener):
        def _read_char(self, timeout: float) -> str | None:
            if self._fd is None:
                return None
            try:
                ready, _, _ = select.select([self._fd], [], [], timeout)
                if not ready:
                    return None
                data = os.read(self._fd, 1)
            except (OSError, ValueError):  # EIO from a closed terminal, or a closed descriptor
                data = b""
            if not data:
                self._running = False  # its reading loop ends
                return None
            return data.decode(errors="ignore")

    return Listener(on_key)


class KeyReader:
    """Turns key presses into commands on the controller's queue. It stops reading when its
    terminal goes away."""

    def __init__(self, submit, status) -> None:
        """`submit` queues a command; `status()` is the controller's latest status."""
        self.submit = submit
        self.status = status
        self.listener = listener(self.on_key)
        self.asking: int | None = None  # the attempt the right arrow asked about, until it's answered

    def on_key(self, key: str) -> None:
        status = self.status()
        if self.asking is not None:
            attempt, self.asking = self.asking, None
            # In Fault the attempt is saved and Reset with its outcome labels it. Otherwise, with no attempt
            # open, it ended meanwhile: this key is taken as any other, s, f or i as the answer to a time limit.
            if status["attempt_open"] or status["state"] == "fault":
                if outcome := OUTCOME_KEYS.get(key):
                    self._send("reset", outcome=outcome, for_attempt=attempt)
                    return
                print("Not ended: the attempt goes on.", flush=True)
        if key == "right" and not status["attempt_open"]:
            self._send("reset")  # in Fault the attempt was saved there; at rest none is open: nothing to ask
        elif key == "right":
            self.asking = status["attempt"]
            print(ASK, flush=True)
        elif key == "left":
            self._send("reset", outcome="ignored")
        elif outcome := OUTCOME_KEYS.get(key):
            self._send("outcome", outcome=outcome, for_attempt=status["prompt"])
        elif key in HELP_KEYS:
            print(HELP, flush=True)
        elif key == "space":
            self._send(KEYS[key], note=SPACE_REFUSED.get(status["state"]))
        elif command := KEYS.get(key):
            self._send(command)

    def _send(self, command: str, note: str | None = None, **options) -> None:
        self.submit(command, source="terminal", **options).add_done_callback(partial(report, note=note))

    def start(self) -> None:
        if not sys.stdin.isatty():
            print("No terminal to read keys from: only Ctrl+C works, and it never exits where the arm can't get "
                  "to rest.", flush=True)
        self.listener.start()

    def stop(self) -> None:
        """Stop reading, and restore the terminal's settings if it's still there to take them."""
        with contextlib.suppress(OSError, ValueError, termios.error):
            self.listener.stop()
