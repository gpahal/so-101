"""Terminal plumbing: run LeRobot commands, and hide harmless startup warnings they all print.

OpenCV and PyAV each bundle FFmpeg's libavdevice. When both load into one process, macOS's
Objective-C runtime prints "objc[...]: Class AVF... is implemented in both ..." for each duplicate
class, straight to file descriptor 2 and before Python can do anything about it. Nothing here uses
those classes, so it's noise. `filter_stderr` puts a pipe on descriptor 2 that drops those lines
and passes everything else through as it arrives; LeRobot commands run as child processes inherit
the pipe, so their copies are dropped too.

TorchCodec supports FFmpeg 4 to 8. With another FFmpeg, such as Homebrew's 9, every LeRobot import
prints a traceback of over 100 lines before falling back to PyAV, which decodes these videos fine.
The filter replaces that block with the one line that says what happened.
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
TORCHCODEC_START = b"WARNING:lerobot.utils.import_utils:Could not load libtorchcodec."
TORCHCODEC_END = b"[end of libtorchcodec loading traceback].\n"


def _write_all(fd: int, data: bytes) -> None:
    while data:
        data = data[os.write(fd, data):]


class _NoiseFilter:
    """Removes known noise from a stream of stderr chunks, holding back only a possible start of it."""

    def __init__(self) -> None:
        self.pending = b""
        self.in_block = False

    def feed(self, chunk: bytes) -> bytes:
        data = self.pending + chunk
        self.pending = b""
        out = b""
        while data:
            if self.in_block:
                end = data.find(TORCHCODEC_END)
                if end < 0:
                    self.pending = data[-(len(TORCHCODEC_END) - 1):]  # the end marker may be split
                    return DUPLICATE_CLASS.sub(b"", out)
                data = data[end + len(TORCHCODEC_END):]
                self.in_block = False
                continue
            start = data.find(TORCHCODEC_START)
            if start >= 0:
                out += data[:start]
                data = data[start + len(TORCHCODEC_START):]
                self.in_block = True
                continue
            hold = next((n for n in range(min(len(TORCHCODEC_START) - 1, len(data)), 0, -1)
                         if TORCHCODEC_START.startswith(data[-n:])), 0)
            out += data[:len(data) - hold]
            self.pending = data[len(data) - hold:]
            break
        return DUPLICATE_CLASS.sub(b"", out)


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
        noise = _NoiseFilter()
        with os.fdopen(read_end, "rb", buffering=0) as pipe:
            while chunk := pipe.read(65536):
                _write_all(real, noise.feed(chunk))
        if not noise.in_block:
            _write_all(real, noise.pending)

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
    With `log_path`, its output also goes to that file, without the noise `filter_stderr` drops.
    """
    exe = Path(sys.executable).parent / command
    line = f"$ {shlex.join([command, *args])}\n"
    print(line, end="", flush=True)
    if log_path is None:
        proc = subprocess.Popen([str(exe), *args])
        return _wait(proc)
    with log_path.open("ab", buffering=0) as log:
        log.write(line.encode())
        read_end, write_end = os.pipe()
        proc = subprocess.Popen(
            [str(exe), *args], stdout=write_end, stderr=write_end, env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        os.close(write_end)

        def pump() -> None:
            noise = _NoiseFilter()
            with os.fdopen(read_end, "rb", buffering=0) as pipe:
                while chunk := pipe.read(65536):
                    clean = noise.feed(chunk)
                    log.write(clean)
                    _write_all(1, clean)

        thread = threading.Thread(target=pump, name="run-log", daemon=True)
        thread.start()
        code = _wait(proc)
        # A viewer the child started can hold the pipe open after the child exits, so don't wait for it.
        thread.join(timeout=2)
    return code


def _wait(proc: subprocess.Popen) -> int:
    while True:
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            continue
        return 128 - code if code < 0 else code  # killed by a signal: the shell's convention
