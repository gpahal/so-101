"""Terminal plumbing: run LeRobot commands, and hide a harmless macOS warning they all print.

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
import subprocess
import sys
import threading
from pathlib import Path

DUPLICATE_CLASS = re.compile(rb"objc\[\d+\]: Class AVF\w+ is implemented in both [^\n]*\n")


def _write_all(fd: int, data: bytes) -> None:
    while data:
        data = data[os.write(fd, data):]


def filter_stderr() -> None:
    """Drop the duplicate-class warnings from this process's and its children's stderr (macOS only)."""
    if sys.platform != "darwin":
        return
    sys.stderr.flush()
    real = os.dup(2)
    read_end, write_end = os.pipe()
    os.dup2(write_end, 2)
    os.close(write_end)

    def pump() -> None:
        # Each warning is one write of a few hundred bytes, under the pipe's atomic-write size, so
        # it always arrives whole in one read.
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


def run_lerobot(command: str, args: list[str]) -> int:
    """Run a LeRobot CLI from the same virtualenv and wait for it, returning its exit code.

    Ctrl+C reaches the child too (same process group), so keep waiting while it shuts down cleanly.
    """
    exe = Path(sys.executable).parent / command
    print("$", shlex.join([command, *args]), flush=True)
    proc = subprocess.Popen([str(exe), *args])
    while True:
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            continue
        return 128 - code if code < 0 else code  # killed by a signal: the shell's convention
