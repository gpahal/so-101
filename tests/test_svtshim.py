"""SVT-AV1's macOS teardown crash fix (`so_101.svtshim`, `svtsem.c`): it builds, a restart loads it,
it rewrites exactly SVT-AV1's semaphores, and the checks fail when it's missing.

The crash itself is too rare to show here (about 0.1 to 0.4% of encoder closes), so the tests prove
the fix is in the path instead: the loaded library counts the semaphores it created for SVT-AV1.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np

from so_101 import svtshim
from so_101.check import check_video
from so_101.report import Report

svtshim.ensure()

NEEDED = svtshim.needed()


def clean_env() -> dict[str, str]:
    """This environment without the fix, as a shell started outside so101 has it."""
    return {k: v for k, v in os.environ.items() if k not in (svtshim.INSERT_ENV, svtshim.TRIED_ENV)}


def encode(path: Path, frames: int = 3) -> None:
    out = av.open(str(path), "w")
    stream = out.add_stream("libsvtav1", 30)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for _ in range(frames):
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()


@unittest.skipUnless(NEEDED, "only macOS with an SVT-AV1 older than 4.3.0 needs the fix")
class LoadedTest(unittest.TestCase):
    def test_each_encoder_gets_one_rewritten_semaphore(self):
        import tempfile

        before = svtshim.semaphores_rewritten()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                encode(Path(tmp) / f"{i}.mp4")
            self.assertEqual(svtshim.semaphores_rewritten() - before, 2)
            with av.open(str(Path(tmp) / "0.mp4")) as video:
                self.assertEqual(sum(1 for _ in video.decode(video=0)), 3)

    def test_without_the_fix_a_recording_check_fails_and_require_exits(self):
        report = Report()
        with mock.patch.dict(os.environ, {svtshim.INSERT_ENV: ""}):
            check_video(report, required=True)
        [result] = report.results
        self.assertEqual(result.status, "FAIL")
        self.assertIn("isn't loaded", result.detail)
        self.assertIn("uv run so101", result.fix)
        # A restart was tried (the variable is set in its parent), but the variable didn't arrive.
        env = clean_env() | {svtshim.TRIED_ENV: "1"}
        code = "from so_101 import svtshim; svtshim.ensure(); print(svtshim.loaded()); svtshim.require()"
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "False")
        self.assertEqual(out.returncode, 1)
        self.assertIn("fix isn't loaded in this process", out.stderr)
        self.assertIn("/bin/sh", out.stderr)
        self.assertIn(svtshim.ISSUE, out.stderr)


# The subprocess calls that start this venv's Python, which should inherit the fix: (file, function).
PYTHON_CHILDREN = {
    ("console.py", "run_lerobot"),
    ("review.py", "view"),
    ("dryrun.py", "smoke_train"),
    ("svtshim.py", "ensure"),
}


class OtherProgramsTest(unittest.TestCase):
    def test_every_other_program_is_started_without_the_fix(self):
        """A subprocess call in src/ passes `env=` (svtshim.other_env()) unless it starts this venv's
        Python, so no macOS program is started with the arm64 fix inserted."""
        import ast

        src = Path(svtshim.__file__).parent
        missing = []
        for file in sorted(src.rglob("*.py")):
            if "sim" in file.relative_to(src).parts:
                continue
            tree = ast.parse(file.read_text())
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                for call in ast.walk(function):
                    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                        continue
                    starts = call.func.attr in ("run", "Popen", "check_output", "call") and \
                        isinstance(call.func.value, ast.Name) and call.func.value.id == "subprocess" \
                        or call.func.attr.startswith("exec") and isinstance(call.func.value, ast.Name) \
                        and call.func.value.id == "os"
                    if starts and not any(k.arg == "env" for k in call.keywords) \
                            and (file.name, function.name) not in PYTHON_CHILDREN:
                        missing.append(f"{file.relative_to(src)}:{call.lineno} in {function.name}")
        self.assertEqual(missing, [], "start these with env=svtshim.other_env(), or list them in PYTHON_CHILDREN")
        # other_env() drops only the fix and keeps any other inserted library.
        mine = str(svtshim.path())
        with mock.patch.dict(os.environ, {svtshim.INSERT_ENV: f"/opt/other.dylib:{mine}"}):
            self.assertEqual(svtshim.other_env()[svtshim.INSERT_ENV], "/opt/other.dylib")
        with mock.patch.dict(os.environ, {svtshim.INSERT_ENV: mine}):
            self.assertNotIn(svtshim.INSERT_ENV, svtshim.other_env())
        with mock.patch.dict(os.environ, clean_env(), clear=True):
            self.assertIsNone(svtshim.other_env())
        # /usr/bin/true is arm64e. With System Integrity Protection off (GitHub's macOS runners), it
        # would refuse to start with the arm64 fix inserted, so other_env() must leave it out.
        self.assertNotIn(str(svtshim.path()), (svtshim.other_env() or {}).get(svtshim.INSERT_ENV, ""))
        self.assertEqual(subprocess.run(["/usr/bin/true"], env=svtshim.other_env()).returncode, 0)


if __name__ == "__main__":
    unittest.main()
