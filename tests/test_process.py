"""Process supervision, through real processes and signals: `console.run_lerobot` turning SIGTERM
and SIGHUP into the child's Ctrl+C, `so101-control` quitting on a signal (`quit_on_signals`, in this
process), its cleanups, and `remote.tunnel` restarting ssh when it exits.

Only this virtualenv's python and a fake `ssh` run; nothing here reaches the network.
"""

import contextlib
import io
import json
import logging
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from so_101 import remote

# A child that records each SIGINT on its own line, and exits with code 7 a while after the
# `wanted`-th one, so the test can see whether the wrapper waited for it.
CHILD = """\
import signal, sys, time
from pathlib import Path

log, wanted = Path(sys.argv[1]), int(sys.argv[2])
got = 0

def on_sigint(signum, frame):
    global got
    got += 1
    with log.open("a") as f:
        f.write("SIGINT\\n")

signal.signal(signal.SIGINT, on_sigint)
log.write_text("ready\\n")
deadline = time.monotonic() + 20
while got < wanted and time.monotonic() < deadline:
    time.sleep(0.01)
time.sleep(0.5)
with log.open("a") as f:
    f.write("exiting\\n")
sys.exit(7)
"""

WRAPPER = """\
import sys
from so_101.console import run_lerobot

sys.exit(run_lerobot("python", sys.argv[1:]))
"""


def ready(log: Path) -> bool:
    """Whether a child has written "ready" to `log`. The file exists a moment before the word is in it,
    and a signal handled in that moment appends a line that the word then overwrites."""
    return log.exists() and "ready" in log.read_text()


def wait_for(condition, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.02)


def kill_group(wrapper: subprocess.Popen) -> None:
    """Kill the wrapper and its child, which outlives a wrapper that died."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(wrapper.pid, signal.SIGKILL)
    wrapper.wait()


class RunLerobotSignals(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "child.py").write_text(CHILD)
        (self.dir / "wrapper.py").write_text(WRAPPER)
        self.log = self.dir / "child.log"

    def start(self, wanted: int) -> subprocess.Popen:
        wrapper = subprocess.Popen(
            [sys.executable, str(self.dir / "wrapper.py"), str(self.dir / "child.py"), str(self.log), str(wanted)],
            stdout=subprocess.DEVNULL,
            start_new_session=True,  # keeps any stray signal away from the test runner
        )
        self.addCleanup(kill_group, wrapper)
        wait_for(lambda: ready(self.log), 10, "the child never started")
        return wrapper

    def lines(self) -> list[str]:
        return self.log.read_text().split()

    def send_and_wait(self, wrapper: subprocess.Popen, signals: list[int]) -> int:
        for n, sig in enumerate(signals, 1):
            wrapper.send_signal(sig)
            wait_for(lambda: self.lines().count("SIGINT") >= n, 5, f"signal {n} never reached the child as SIGINT")
            self.assertIsNone(wrapper.poll(), "the wrapper exited before its child")
        return wrapper.wait(timeout=10)

    def test_each_sigterm_or_sighup_reaches_the_child_as_sigint_and_the_wrapper_waits(self) -> None:
        for signals in ([signal.SIGTERM], [signal.SIGTERM, signal.SIGTERM], [signal.SIGHUP]):
            with self.subTest(signals=signals):
                self.log.unlink(missing_ok=True)
                wrapper = self.start(wanted=len(signals))
                code = self.send_and_wait(wrapper, signals)
                self.assertEqual(self.lines(), ["ready", *["SIGINT"] * len(signals), "exiting"])
                self.assertEqual(code, 7)


class QuitOnSignals(unittest.TestCase):
    """Closing the terminal sends SIGHUP to `serve`'s whole process group, and a service manager may
    send SIGTERM to it: the control child gets one directly, and one from `serve`'s wrapper, which
    passes its copy on as SIGINT (`RunLerobotSignals`). Here the process signals itself with each in
    turn. Each reaches the controller as a Quit with source `signal`, which can't confirm an exit
    away from rest (test_controller's QuitTest)."""

    def setUp(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # `quit_on_signals` installs its own
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))

    def test_a_signal_and_the_wrappers_sigint_each_submit_a_quit_from_signal(self) -> None:
        from so_101.control.commands import Result
        from so_101.control.main import quit_on_signals

        quits: list[tuple[str, str]] = []

        def submit(command: str, source: str) -> Future:
            quits.append((command, source))
            future: Future = Future()
            future.set_result(Result(command, True))
            return future

        quit_on_signals(submit)
        with contextlib.redirect_stdout(io.StringIO()):
            for n, signum in enumerate((signal.SIGHUP, signal.SIGINT), 1):
                os.kill(os.getpid(), signum)
                wait_for(lambda: len(quits) >= n, 5, f"signal {n} never submitted a quit")
        self.assertEqual(quits, [("quit", "signal")] * 2)


# A fake ssh: listens on the port its `-L` asks for. Its first run exits after 0.5 s, like a tunnel
# that drops; later runs listen until they're stopped. With a `refuse` file beside it, it exits at
# once with ssh's error code, as when the job refuses the key.
FAKE_SSH = """\
import socket, sys, time
from pathlib import Path

here = Path(__file__).parent
runs = here / "runs"
with runs.open("a") as f:
    f.write("run\\n")
if (here / "refuse").exists():
    sys.exit(255)
port = int(sys.argv[sys.argv.index("-L") + 1].split(":")[1])  # bind_address:port:host:hostport
first = runs.read_text().count("run") == 1
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", port))
server.listen()
server.settimeout(0.05)
deadline = time.monotonic() + (0.5 if first else 60)
while time.monotonic() < deadline:
    try:
        server.accept()[0].close()
    except TimeoutError:
        pass
"""

DELAY_S = 0.5


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


class ControlShutdown(unittest.TestCase):
    """`so101-control`'s whole run on the fake follower, with what reaches beyond this process
    stubbed: config files, devices, signal handlers, the page's server, the policy's loading and the
    key reader. The key reader's start quits
    at rest, and its stop raises, as the first cleanup to run."""

    def setUp(self) -> None:
        from fakes import FakeClock, make_follower
        from test_controller import REST_STEPS

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # `control` installs its own, on this process
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
        self.run_dir = Path(tmp.name) / "test" / "runs" / "20261111-000000"
        self.robot, self.motors = make_follower(FakeClock(), REST_STEPS)
        def disconnect() -> None:  # if the test failed before so101-control disconnected it
            for camera in self.robot.inner.cameras.values():
                camera.is_connected = True
            if self.motors.is_open:
                self.robot.inner.disconnect()

        self.addCleanup(disconnect)
        self.order: list[str] = []  # the cleanups seen, in the order they ran
        close = self.motors.close

        def disconnected() -> None:  # the follower's port closes
            self.order.append("follower disconnected" + (" after run_end" if "run_end" in self.events() else ""))
            close()

        self.motors.close = disconnected

    def events(self) -> list[str]:
        path = self.run_dir / "events.jsonl"
        return [json.loads(line)["type"] for line in path.read_text().splitlines()] if path.exists() else []

    def control(self, before_quit=lambda: None, quit_in: str = "resting", at_exit=lambda: None) -> None:
        """Run `so101-control` until the key reader's quit, after `before_quit()`. Its stop raises.

        The quit comes from the terminal at once, at rest; with another `quit_in`, it comes as a signal
        once the state is `quit_in`. If that never comes within 10 s, or the signal's quit hasn't ended
        the run 2 s later, the terminal quits and confirms, so the run always ends. `at_exit()` runs as the policy stops, just before the
        follower disconnects."""
        from fakes import control, make_engine
        from test_controller import POLICY_ACTIONS, REST_STEPS

        from so_101.control import main
        from so_101.control.policy import EngineDriver
        from so_101.control.server import Bind

        order, robot, test = self.order, self.robot, self
        driver = EngineDriver(make_engine(POLICY_ACTIONS), list(robot.action_features))
        stop_driver = driver.stop

        def stop() -> None:
            stop_driver()
            at_exit()

        driver.stop = stop
        s = SimpleNamespace(control=control(preview_fps=0, rest_pose=REST_STEPS), follower=None, leader=None,
                            camera=lambda name: name, path=Path("so101.toml"))
        task = SimpleNamespace(name="test", cameras=tuple(robot.inner.cameras))
        cfg = SimpleNamespace(task="test", policy=SimpleNamespace(pretrained_path="policy"))

        class Page:
            https = name_url = None
            url = "http://127.0.0.1:8101/"

            def __init__(self, *args, **kwargs) -> None:
                pass

            def start(self) -> None:
                pass

            def stop(self) -> None:
                order.append("page stopped")

        class Keys:
            def __init__(self, submit, status) -> None:
                self.submit, self.status = submit, status
                test.keys_submit = submit  # for `before_quit`

            def start(self) -> None:
                before_quit()
                if quit_in == "resting":
                    self.submit("quit", source="terminal")  # at rest: the run ends on the first tick
                    return

                def quit_later() -> None:
                    deadline = time.monotonic() + 10
                    while self.status().get("state") != quit_in and time.monotonic() < deadline:
                        time.sleep(0.01)
                    if self.status().get("state") == quit_in:
                        self.submit("quit", source="signal")
                        time.sleep(2)  # if that didn't end the run, the confirmed quit below does
                    self.submit("quit", source="terminal")
                    self.submit("exit_here", source="terminal")

                threading.Thread(target=quit_later, daemon=True).start()

            def stop(self) -> None:
                raise RuntimeError("restoring the terminal failed")

        manifest = {"operator": None, "control": {"fps": 30},
                    "policy": {"type": "act", "revision": "main", "inference": "local"}}
        environment = {main.RUN_ENV: json.dumps({"dir": str(self.run_dir), "manifest": manifest})}
        # LeRobot's real init_logging runs, so a call it rejects fails here. It replaces the root
        # logger's handlers and level, which the other tests rely on, so they're put back after.
        root = logging.getLogger()
        self.addCleanup(setattr, root, "handlers", root.handlers[:])
        self.addCleanup(root.setLevel, root.level)
        with mock.patch.dict(os.environ, environment), \
                mock.patch("so_101.settings.load", return_value=s), \
                mock.patch("so_101.tasks.load", return_value={"test": task}), \
                mock.patch("so_101.data.demo_start_pose", return_value=None), \
                mock.patch("so_101.devices.presence", return_value={}), \
                mock.patch.object(main, "check_settings"), \
                mock.patch.object(main, "leader_maker", return_value=mock.Mock(side_effect=OSError("no leader plugged in"))), \
                mock.patch.object(main, "quit_on_signals"), \
                mock.patch.object(main, "interrupt_on_signals"), \
                mock.patch.object(main, "keep_awake", return_value=None), \
                mock.patch("lerobot.rollout.build_rollout_context",
                           return_value=SimpleNamespace(hardware=SimpleNamespace(robot_wrapper=robot))), \
                mock.patch.object(EngineDriver, "from_context", return_value=driver), \
                mock.patch("so_101.control.server.resolve_bind", return_value=Bind("127.0.0.1")), \
                mock.patch("so_101.control.server.ControlServer", Page), \
                mock.patch("so_101.control.terminal.KeyReader", Keys), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(RuntimeError):
            main.control(cfg)

    def test_every_cleanup_runs_when_the_first_raises_and_the_run_closes_before_the_follower_disconnects(self) -> None:
        def unplug() -> None:  # the follower still disconnects while a camera reports itself disconnected
            self.robot.inner.cameras["top"].is_connected = False

        self.control(before_quit=unplug)
        self.assertIn("run_end", self.events())
        self.assertEqual(self.order, ["follower disconnected after run_end", "page stopped"])
        self.assertFalse(self.motors.torqued(), "the motors turned off")
        self.assertFalse(self.motors.is_open)

    def test_a_quit_with_the_follower_silent_leaves_its_motors_on_when_it_answers_again_at_exit(self) -> None:
        def unplug() -> None:  # the follower's USB cable comes out with its power on, then a policy starts
            self.motors.failing = True
            self.keys_submit("resume", source="terminal")

        def answers_again() -> None:  # the cable is back just as serve exits, with the arm raised and holding
            self.motors.failing = False

        self.control(before_quit=unplug, quit_in="fault", at_exit=answers_again)
        self.assertIn("run_end", self.events())
        self.assertTrue(self.motors.torqued(), "its motors were left on: the arm keeps holding")
        self.assertFalse(self.motors.is_open, "its port closed")


class SuspendTest(unittest.TestCase):
    def test_ctrl_z_never_stops_so101_control(self) -> None:
        from so_101.control import main

        # A child of its own: a test process that stops would hang the run, where this one times out.
        code = ("import os, signal, time; from so_101.control import main; main.refuse_suspend(); "
                "os.kill(os.getpid(), signal.SIGTSTP); time.sleep(0.2); print('still running', flush=True)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.splitlines(), [main.SUSPEND_REFUSED, "still running"])


class TunnelRestart(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bin_dir = Path(tmp.name)
        ssh = bin_dir / "ssh"
        ssh.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_SSH))
        ssh.chmod(ssh.stat().st_mode | stat.S_IXUSR)
        self.bin_dir = bin_dir
        self.runs = bin_dir / "runs"
        self.port = free_port()
        for patch in (
            mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}),
            mock.patch.object(remote, "PORT", self.port),
            mock.patch.object(remote, "RESTART_DELAY_S", DELAY_S),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_ssh_is_restarted_after_it_exits_and_stopped_with_the_block(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with remote.tunnel("job"):
                self.assertTrue(port_open(self.port))
                wait_for(lambda: not port_open(self.port), 5, "the first ssh never exited")
                dropped = time.monotonic()
                wait_for(lambda: port_open(self.port), DELAY_S + 5, "the tunnel was never restarted")
                self.assertGreaterEqual(time.monotonic() - dropped, DELAY_S * 0.8, "restarted without waiting")
                self.assertIn("SSH tunnel to the job dropped; reconnecting.", out.getvalue())
            self.assertFalse(port_open(self.port), "ssh still listens after the block")
            time.sleep(DELAY_S * 2)
        self.assertFalse(port_open(self.port), "ssh was started again after the block")
        self.assertEqual(self.runs.read_text().count("run"), 2)
    def test_a_closed_terminal_never_stops_a_restart_and_a_first_open_that_fails_exits_at_once(self) -> None:
        from fakes import DeadStream

        with contextlib.redirect_stdout(DeadStream()):  # the terminal is closed
            with remote.tunnel("job"):
                wait_for(lambda: not port_open(self.port), 5, "the first ssh never exited")
                wait_for(lambda: port_open(self.port), DELAY_S + 5, "the tunnel was never restarted")
            self.assertEqual(self.runs.read_text().count("run"), 2)

            (self.bin_dir / "refuse").touch()
            started = time.monotonic()
            with self.assertRaises(SystemExit) as raised:
                with remote.tunnel("job"):
                    self.fail("the block ran without a tunnel")
            self.assertLess(time.monotonic() - started, 3)
            self.assertIn("The SSH tunnel to the job failed", str(raised.exception))
            time.sleep(DELAY_S * 2)
        self.assertEqual(self.runs.read_text().count("run"), 3, "a failed first open was retried")


if __name__ == "__main__":
    unittest.main()
