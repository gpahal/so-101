# SVT-AV1's macOS crash when an encoder closes

Status: built (`src/so_101/svtshim.py`, `src/so_101/svtsem.c`). It's a stopgap until PyAV bundles SVT-AV1 4.3.0 or later and LeRobot allows that PyAV. The user-facing part is in the README, under [The video encoder's macOS crash](../README.md#the-video-encoders-macos-crash). Written in November 2026 against LeRobot 0.7.0, PyAV 15.1.0 (SVT-AV1 3.0.0), macOS on an M3 Max and Python 3.13. The measurements were made on that Mac; none was made on the real arm or on CI's macOS VMs.

## The bug

LeRobot encodes each camera's video with SVT-AV1, through PyAV, on one encoder thread per camera. At the end of an episode it sends end of stream, drains the packets and closes the encoder. That close is where SVT-AV1 can crash. LeRobot's teardown is correct: the encoder isn't missing its end of stream, isn't closed on the wrong thread, and isn't left to garbage collection.

- **The race.** SVT-AV1 creates `ref_buffer_available_semaphore` with a count equal to its reference picture list length (`Source/Lib/Globals/enc_handle.c:1415` in 3.0.0).
  - Picture decision waits on it once for each reference picture.
  - The picture manager posts it as references are released.
  - At end of stream, `release_references_eos` posts for the rest, but only for references the picture manager has already recorded. The restoration stage hands the last reference to the picture manager and to packetization at the same time, so packetization can reach end of stream first, and one count is lost.
  - Upstream's own write-up is in [MR !2811](https://gitlab.com/AOMediaCodec/SVT-AV1/-/merge_requests/2811).
- **Why only macOS.** On Apple, `svt_destroy_semaphore` is `dispatch_release`. libdispatch's `_dispatch_semaphore_dispose` traps when a semaphore's value is below its original value ([semaphore.c](https://github.com/apple-oss-distributions/libdispatch/blob/2361ffb78a76f7ee488cd052eb0bc5c767118bf9/src/semaphore.c#L57-L60)). POSIX and Windows semaphores don't check, and the encoder never runs again after end of stream, so the lost count does no harm there.
- **What it looks like.** SIGTRAP and exit code 133, shown as `zsh: trace trap`, with no Python traceback.
  - The crash report (`~/Library/Logs/DiagnosticReports/python3.13-*.ips`) says "BUG IN CLIENT OF LIBDISPATCH: Semaphore object deallocated while in use … Abort Cause 1": exactly one count short.
  - All 24 reports collected here had that signature.
- **How often.** About 0.1 to 0.4% of encoder closes, mostly for very short clips. A full test run closes about 250 encoders, and 2 of 5 runs crashed. Long 640×480 episodes had 0 crashes in 800 closes, which is too few to give a rate.
- **Where it's tracked.**
  - SVT-AV1 [issue #2961](https://github.com/AOMediaCodec/SVT-AV1/issues/2961) (GitLab #2251) came from libavif's Apple-silicon CI in February 2025. It reproduced on an M1 but not on Intel Macs. SVT-AV1 3.0.1 listed it as a known issue.
  - The fix is [commit 049193fc](https://github.com/AOMediaCodec/SVT-AV1/commit/049193fc0e16ba7f1afe8baa51292157b681cb2a), "Fix the Apple teardown SIGTRAP (#2251)". It is +8/−1 in `svt_threads.c`, first released in SVT-AV1 4.3.0 (2026-11-01).
  - We reported it to LeRobot as [huggingface/lerobot#5308](https://github.com/huggingface/lerobot/issues/5308) on 2026-11-11, after finding no earlier report there or in PyAV.

## Why we can't just upgrade

- **PyAV.** No release bundles SVT-AV1 4.3.0 yet. The newest on PyPI, 19.0.2, bundles 4.2.0. PyAV's FFmpeg builds ([pyav-ffmpeg](https://github.com/PyAV-Org/pyav-ffmpeg)) moved to 4.3.0 on 2026-11-02, so the next PyAV release will probably have it.

  | PyAV wheel (macOS arm64) | SVT-AV1 | libaom |
  | --- | --- | --- |
  | 15.1.0 | 3.0.0 | 3.11.0 |
  | 16.1.0 | 3.1.2 | 3.13.1 |
  | 17.0.1 | 3.1.2 | none |
  | 19.0.2 | 4.2.0 | none |

- **LeRobot.** Both 0.7.0 and main pin `av>=15,<16`. Overriding the pin with uv doesn't work past av 17.0:
  - PyAV 17.1 removed `av.option` and `Codec.descriptor`;
  - LeRobot's `datasets/pyav_utils.py` uses both;
  - so `import lerobot.datasets` fails with `AttributeError: module 'av' has no attribute 'option'`.

## The fix

`svtsem.c` replaces `dispatch_semaphore_create` through dyld's `__interpose` section. It's this repo's own code, written from commit 049193fc; SVT-AV1 publishes no such shim.

- **What it changes.** For a creation with a count above 0 whose caller is in `libSvtAv1Enc`, it creates the semaphore at 0 and signals it up to the count, as commit 049193fc does. Every other call passes straight through. Calls made from the shim itself are never interposed, so its inner `dispatch_semaphore_create` is the real one.
- **What it touches.** SVT-AV1 3.0.0 imports only `dispatch_semaphore_create`, `_signal`, `_wait` and `dispatch_release`. Only one of its semaphores starts above 0, so the shim rewrites one semaphore per encoder. The tests check exactly that, through a counter the shim exports.
- **How it's built and loaded** (`svtshim.py`):
  - `build()` compiles it with `cc` into `~/Library/Caches/so101/svtsem-<hash>.dylib`. The hash covers the source and the CPU, so an edited source builds anew. A build takes about 0.1 s.
  - `ensure()` restarts the process with `os.execve(sys.executable, sys.orig_argv, env)`, adding the library to `DYLD_INSERT_LIBRARIES`. It sets `SO101_SVT_SHIM_TRIED` so a library that won't load can't cause a restart loop.
  - `loaded()` asks dyld whether the library is in the process (`RTLD_NOLOAD`) and whether it came in through the variable, since only an inserted library's interposing takes effect.
- **Where it's called.**
  - `cli.main` calls `ensure()` for `record`, `run`, `serve`, `build` and `train`, before the stderr filter starts its thread.
  - `so101-control` and `so101-rollout` call `ensure()` then `require()`. `run_build` calls `require()`.
  - The preflight checks of `record`, its dry run, `run` and `serve` fail unless it's loaded, through `check.check_video`.
  - `tests/fakes.py` calls `ensure()`, so the recorder and build tests run with it.
- **Why it can load at all.** uv's Python (`cpython-3.13.14`) is ad-hoc signed without the hardened runtime (`codesign -dv`: `flags=0x20002(adhoc,linker-signed)`), so dyld honours `DYLD_INSERT_LIBRARIES`. A hardened-runtime Python would ignore it, and the checks would then fail rather than run unprotected.
- **Child processes.** This venv's Python processes inherit the variable, so the LeRobot commands `so101` starts, and their worker processes, load the shim too. Every other program `so101` starts (`git`, `ssh`, `caffeinate`, `tailscale`, `uvc-util`, `cc`) is started with `svtshim.other_env()`, which leaves the shim out. Three cases matter:
  - **SIP off.** macOS's own programs (`git`, `ssh`, `caffeinate`, `cc`, `/bin/sh`) are arm64e. With System Integrity Protection off, as on GitHub's macOS runners, dyld refuses to start one with the arm64 shim inserted ("incompatible architecture (have 'arm64', need 'arm64e')"). `cc` started from a test process with the shim loaded fails this way.
  - **SIP on, the default.** macOS strips `DYLD_*` variables from its own programs anyway, so a Python started through `/bin/sh` runs without the shim, and its checks fail.
  - **The guard.** `tests/test_svtshim.py` checks that every subprocess call in `src/` passes `env=` unless it starts this venv's Python. A program that LeRobot or a library starts from inside a shim-loaded process isn't covered: with SIP off, an arm64e one would fail to start.

### Measured

These runs open, encode, flush and close two encoders at a time, as LeRobot's encoder threads do. They use 64×48 clips of 1 to 3 frames, the size that crashes most.

| Run | Without the shim | With the shim |
| --- | --- | --- |
| 3-frame clips | 3 of 20,000 closes crashed | 0 of 20,000 |
| 1–3-frame clips | 38 of 36,000 | 0 of 36,000 |
| 1–3-frame clips, the shipped `svtsem.c` and the recorder's options (CRF 30, preset 10, `lp=2`) | 134 of 36,000 | 0 of 36,000 |

Every crash without the shim was SIGTRAP with exit code 133. With and without the shim, 20 s of the front camera encoded with the recorder's options gave the same video stream byte for byte (ffmpeg's MD5 of the packets).

## Alternatives considered

In order of how close each comes:

- **Recording with Apple's VideoToolbox H.264.**
  - **What works:** it avoids SVT-AV1 entirely. With quality set through `flags=+qscale` and `global_quality`, it matched AV1 at CRF 30 (VMAF 86.1 against 86.4 front, 82.9 against 82.7 wrist), at about 1.7× the file size, and encoding took 0.8 ms a frame against 2–3 ms.
  - **Why not:** `build` would still encode with SVT-AV1 and could crash, and so could the tests. LeRobot's `vcodec=auto` passes its quality as `q:v`, which PyAV ignores, so `auto` alone gives whatever bitrate VideoToolbox picks and can't be tuned ([huggingface/lerobot#5309](https://github.com/huggingface/lerobot/issues/5309)). It's the fallback if the shim ever can't be used.
- **SVT-AV1's low-delay mode (`pred-struct=1`).**
  - **What works:** it's a config change only. 0 of 36,000 closes crashed, against 38 in the default mode, at 1.16× the size for the same quality.
  - **Why not:** nobody could show from the source why it avoids the race. SVT-AV1 before 4.3.0 also has a known low-delay deadlock ([commit 1d4844d89](https://github.com/AOMediaCodec/SVT-AV1/commit/1d4844d89)), which would leave `finish_episode` waiting 120 s.
- **Encoding in a child process per camera.** It's the most robust: the MP4 is complete when the crash comes, so a child that dies loses nothing. But it means about 200 lines against `LeRobotDataset._build_streaming_encoder`, a private LeRobot interface. LeRobot's own non-streaming path, which already encodes in child processes, adds 2.4 to 4.7 s to each save and fails the save half-written if a worker dies.
- **libaom-av1** (bundled in av 15.1.0). It keeps recordings in AV1 without SVT and kept up in real time. But it gives files about 1.3 to 1.5× the size, PyAV dropped libaom in 17.0, and it segfaulted 12 times in 16,000 closes of tiny clips.
- **Swapping in Homebrew's SVT-AV1 4.3.0.** It is an ABI break: `EbSvtAv1EncConfiguration` grew from 616 to 664 bytes, and fields moved.
- **Rebuilding SVT-AV1 3.0.0 with the patch,** or PyAV against Homebrew's FFmpeg. The first needs cmake, `uv sync` undoes it, and CI needs a build step. The second fails because PyAV 15.1.0 doesn't compile against FFmpeg 9.
- **Never closing the encoder.** Each leaked encoder holds about 100 MB and 47 threads.

## Removing it

Once a PyAV release bundles SVT-AV1 4.3.0 or later and LeRobot allows it:

1. upgrade;
2. confirm `uv run so101 check` says "has its teardown crash fixed";
3. delete `svtshim.py`, `svtsem.c`, `tests/test_svtshim.py`, `cli.VIDEO_COMMANDS`, every `svtshim` import and `env=svtshim.other_env()` argument, and the `ensure()` (`tests/fakes.py` has one too), `require()` and `check_video` calls;
4. update the README section and this note's `Status:`.
