"""Keep SVT-AV1's macOS teardown crash from killing a process that encodes video.

LeRobot encodes video with SVT-AV1, which PyAV bundles: version 3.0.0 in av 15.1.0, the newest
LeRobot 0.7.0 allows. When an encoder closes, a race in SVT-AV1 can free one of its semaphores one
count below where it started, and on macOS libdispatch then stops the process with SIGTRAP (exit 133,
`zsh: trace trap`). It happened in about 0.1 to 0.4% of encoder closes here, mostly on short clips.
`so101 serve` encodes inside the control center, so one crash ends the session and loses the open
attempt. Reported as https://github.com/AOMediaCodec/SVT-AV1/issues/2961; SVT-AV1 4.3.0 fixes it.

`svtsem.c` applies that fix at runtime: it changes how SVT-AV1's semaphores are created, as SVT-AV1's
own fix does, and nothing else. macOS loads it only at a process's start, from DYLD_INSERT_LIBRARIES,
so `ensure()` builds it with the Mac's clang into ~/Library/Caches/so101/ and restarts the process
with it loaded. This venv's Python processes inherit the variable, so the LeRobot commands they
start (and their worker processes) load it too. Any other program is started with `other_env()`,
without it: macOS's own programs are arm64e and refuse to start with this arm64 library inserted
when System Integrity Protection is off, as on GitHub's macOS runners; with it on, macOS drops the
variable for them anyway, and for /bin/sh, so a Python started through a shell runs without the fix.
The commands that encode call `ensure()` first, and their checks fail when it isn't loaded
(`so_101.check.check_video`).

Once PyAV bundles SVT-AV1 4.3.0 or later, `needed()` is False, nothing is built or loaded, and this
module and `svtsem.c` can go.
"""

import ctypes
import functools
import hashlib
import os
import platform
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

SOURCE = Path(__file__).with_name("svtsem.c")
FIXED_IN = (4, 3, 0)  # the first SVT-AV1 with commit 049193fc
CACHE_DIR = Path.home() / "Library" / "Caches" / "so101"
INSERT_ENV = "DYLD_INSERT_LIBRARIES"
TRIED_ENV = "SO101_SVT_SHIM_TRIED"  # set before the restart, so a shim that won't load can't loop
ISSUE = "https://github.com/AOMediaCodec/SVT-AV1/issues/2961"


class ShimError(Exception):
    """The fix can't be built."""


def svt_library() -> Path | None:
    """The SVT-AV1 library bundled in PyAV's macOS wheel, if any."""
    spec = find_spec("av")
    for location in (spec.submodule_search_locations or []) if spec else []:
        if found := sorted(Path(location, ".dylibs").glob("libSvtAv1Enc*.dylib")):
            return found[0]
    return None


@functools.cache
def svt_version() -> tuple[int, int, int] | None:
    """SVT-AV1's version, as the library itself reports it."""
    if (library := svt_library()) is None:
        return None
    get_version = ctypes.CDLL(str(library)).svt_av1_get_version
    get_version.restype = ctypes.c_char_p
    if not (match := re.match(r"v?(\d+)\.(\d+)\.(\d+)", get_version().decode())):
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def version_text() -> str:
    version = svt_version()
    return "SVT-AV1 " + (".".join(map(str, version)) if version else "(not found)")


def needed() -> bool:
    """Whether this process could hit the crash: macOS, with an SVT-AV1 older than the fix."""
    return sys.platform == "darwin" and (version := svt_version()) is not None and version < FIXED_IN


def path() -> Path:
    """Where the built fix goes: named by the source and the CPU, so a changed source builds anew."""
    digest = hashlib.sha256(SOURCE.read_bytes() + platform.machine().encode()).hexdigest()[:12]
    return CACHE_DIR / f"svtsem-{digest}.dylib"


def build() -> Path:
    """Build the fix once, with the Mac's clang (Xcode's command line tools), and return its path."""
    target = path()
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
    try:
        result = subprocess.run(
            ["cc", "-dynamiclib", "-O2", "-Wall", "-Werror", "-o", str(partial), str(SOURCE)],
            capture_output=True, text=True, env=other_env(),
        )
    except FileNotFoundError:
        raise ShimError("no C compiler (cc) to build it with") from None
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        lines = (result.stderr or result.stdout).strip().splitlines()
        raise ShimError(f"cc failed: {lines[-1] if lines else f'exit status {result.returncode}'}")
    os.replace(partial, target)  # whole or not at all, if two processes build at once
    return target


def other_env() -> dict[str, str] | None:
    """The environment for a program other than this venv's Python: this one without the fix in
    DYLD_INSERT_LIBRARIES, or None (inherit it) when the fix isn't there. Pass it as `env=` to every
    subprocess that isn't this venv's Python. Inserted into one of macOS's own programs, which are
    arm64e, the arm64 fix stops it from starting when System Integrity Protection is off."""
    inserted = os.environ.get(INSERT_ENV)
    if not inserted:
        return None
    env = dict(os.environ)
    shim = str(path())
    if rest := ":".join(entry for entry in inserted.split(":") if entry and entry != shim):
        env[INSERT_ENV] = rest
    else:
        del env[INSERT_ENV]
    return env


def _handle(library: Path) -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(str(library), mode=os.RTLD_NOLOAD)  # finds it only if already loaded
    except OSError:
        return None


def loaded() -> bool:
    """Whether the fix was loaded at this process's start, the only way its interposing takes effect."""
    library = path()
    return str(library) in os.environ.get(INSERT_ENV, "").split(":") and _handle(library) is not None


def semaphores_rewritten() -> int:
    """How many of SVT-AV1's semaphores the loaded fix has created its way. 0 when it isn't loaded."""
    if (handle := _handle(path())) is None:
        return 0
    count = handle.so101_svt_semaphores
    count.restype = ctypes.c_long
    return count()


def ensure() -> None:
    """Restart this process with the fix loaded, when it's needed and not loaded yet.

    Call it first thing, before any thread starts or stderr is redirected: the restart replaces the
    process with the same interpreter and arguments. A failure to build is left for the checks to
    report, and so is a second restart that still didn't load it.
    """
    if not needed():
        return
    try:
        library = build()
    except ShimError:
        return
    if loaded() or os.environ.get(TRIED_ENV):
        return
    env = dict(os.environ)
    env[INSERT_ENV] = ":".join(filter(None, [str(library), env.get(INSERT_ENV)]))
    env[TRIED_ENV] = "1"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, sys.orig_argv, env)


def problem(in_this_process: bool = True) -> tuple[str, str] | None:
    """Why this process isn't protected, with the fix, or None when it is or needn't be. Without
    `in_this_process`, the fix only has to build."""
    if not needed():
        return None
    try:
        build()
    except ShimError as e:
        return f"{version_text()} can crash at the end of an episode, and its fix can't be built: {e}", \
            "install Xcode's command line tools: xcode-select --install"
    if in_this_process and not loaded():
        return f"{version_text()} can crash at the end of an episode, and its fix isn't loaded in this process", \
            "run so101 directly (`uv run so101 ...`): a macOS system program such as /bin/sh in between " \
            f"drops {INSERT_ENV}"
    return None


def require() -> None:
    """Exit with the reason when this process encodes video unprotected."""
    if found := problem():
        detail, fix = found
        sys.exit(f"{detail} ({ISSUE}). Fix: {fix}.")
