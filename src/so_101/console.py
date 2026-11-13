"""Terminal plumbing: run LeRobot commands, and hide harmless startup warnings they all print.

OpenCV and PyAV each bundle FFmpeg's libavdevice. When both load into one process, macOS's
Objective-C runtime prints "objc[...]: Class AVF... is implemented in both ..." for each duplicate
class, straight to file descriptor 2 and before Python can do anything about it. Nothing here uses
those classes, so it's noise. `filter_stderr` puts a pipe on descriptor 2 that drops those lines
and passes everything else through as it arrives; LeRobot commands run as child processes inherit
the pipe, so their copies are dropped too.
"""

import atexit
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

DUPLICATE_CLASS = re.compile(rb"objc\[\d+\]: Class AVF\w+ is implemented in both [^\n]*\n")


def _write_all(fd: int, data: bytes) -> None:
    while data:
        data = data[os.write(fd, data):]


def filter_stderr() -> None:
    """Drop known harmless warnings from this process's and its children's stderr (macOS only)."""
    if sys.platform != "darwin":
        return
    sys.stderr.flush()
    real = os.dup(2)
    read_end, write_end = os.pipe()
    os.dup2(write_end, 2)
    os.close(write_end)

    def pump() -> None:
        # Each duplicate-class warning is one write of a few hundred bytes, under the pipe's
        # atomic-write size, so it always arrives whole in one read.
        with os.fdopen(read_end, "rb", buffering=0) as pipe:
            while chunk := pipe.read(65536):
                _write_all(real, DUPLICATE_CLASS.sub(b"", chunk))

    thread = threading.Thread(target=pump, name="stderr-filter", daemon=True)
    thread.start()

    def restore() -> None:
        sys.stderr.flush()
        os.dup2(real, 2)  # drops this process's last write end, so the pump sees end of file
        thread.join(timeout=2)

    atexit.register(restore)


def run_lerobot(command: str, args: list[str], log_path: Path | None = None) -> int:
    """Run a LeRobot CLI from the same virtualenv and wait for it, returning its exit code.

    Ctrl+C reaches the child too (same process group), so keep waiting while it shuts down cleanly.
    A SIGTERM or SIGHUP sent to this process alone, as `kill` and service managers send it, would
    leave the child running unsupervised, so it's sent on to the child as SIGINT, the Ctrl+C every
    child handles. Nothing is ever escalated to SIGKILL: a `serve` stopped by `kill` still drives
    the arm to rest, and a `record` still finishes its session's files.
    With `log_path`, its output also goes to that file, without the noise `filter_stderr` drops.
    """
    exe = Path(sys.executable).parent / command
    line = f"$ {shlex.join([command, *args])}\n"
    print(line, end="", flush=True)
    with _ForwardAsCtrlC() as forward:
        if log_path is None:
            proc = subprocess.Popen([str(exe), *args])
            forward.attach(proc)
            return _wait(proc)
        with log_path.open("ab", buffering=0) as log:
            log.write(line.encode())
            read_end, write_end = os.pipe()
            proc = subprocess.Popen(
                [str(exe), *args], stdout=write_end, stderr=write_end, env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            forward.attach(proc)
            os.close(write_end)

            def pump() -> None:
                with os.fdopen(read_end, "rb", buffering=0) as pipe:
                    while chunk := pipe.read(65536):
                        clean = DUPLICATE_CLASS.sub(b"", chunk)
                        log.write(clean)
                        _write_all(1, clean)

            thread = threading.Thread(target=pump, name="run-log", daemon=True)
            thread.start()
            code = _wait(proc)
            # A viewer the child started can hold the pipe open after the child exits, so don't wait for it.
            thread.join(timeout=2)
        return code


class _ForwardAsCtrlC:
    """While a child runs, turn each SIGTERM and SIGHUP this process gets into a SIGINT for the child.

    The handlers go in before the child starts, so a signal in between is kept and sent once it has.
    Python runs signal handlers only on the main thread, so elsewhere this does nothing.
    """

    SIGNALS = (signal.SIGTERM, signal.SIGHUP)

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.pending = 0
        self.previous: dict[int, object] = {}

    def __enter__(self) -> "_ForwardAsCtrlC":
        if threading.current_thread() is threading.main_thread():
            for sig in self.SIGNALS:
                self.previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc) -> None:
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def attach(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        while self.pending:
            self.pending -= 1
            proc.send_signal(signal.SIGINT)

    def _handle(self, signum, frame) -> None:
        if self.proc is None:
            self.pending += 1
        else:
            self.proc.send_signal(signal.SIGINT)  # does nothing once the child has been reaped


def _wait(proc: subprocess.Popen) -> int:
    while True:
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            continue
        return 128 - code if code < 0 else code  # killed by a signal: the shell's convention
