"""`so101 record <task> --dry-run`: rehearse a recording session end to end, then check the result.

The rehearsal runs lerobot-record with the task's real settings into a throwaway dataset under
outputs/dry-run/, checks what it wrote, trains ACT on it for a few steps, and optionally replays an
episode on the follower. Nothing is added to the task's sessions or uploaded.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from so_101 import check
from so_101.console import run_lerobot
from so_101.recording import record_args, task_cameras
from so_101.report import Report
from so_101.settings import Settings
from so_101.tasks import Task
from so_101.validate import validate_dataset

TRAIN_STEPS = 50

# The rehearsal uses the task's own episode_time_s and reset_time_s, so it times what recording will.
SCRIPT = """
Rehearsal: two {episode_s:g}-second episodes with {reset_s:g}-second resets in between, as in tasks.toml.
  1. Episode 1: do the task, then press Right arrow (or n) before the {episode_s:g} s are up.
  2. Reset: put the object back where it started.
  3. Episode 2: start moving, then press Left arrow (or r) after a few seconds to redo it.
  4. Reset, then episode 2 again: do the task and let the {episode_s:g} s run out. Move every joint,
     and open and close the gripper, at some point during the episode.
The keys work in this terminal, or anywhere if your terminal has Accessibility permission.
"""


def rehearsal_checks(report: Report, lengths: list[int], fps: int, episode_s: float) -> None:
    group = "rehearsal"
    full = episode_s * fps
    if len(lengths) > 2:
        report.fail(group, f"{len(lengths)} episodes saved, expected 2: the redo (Left arrow) didn't discard the first try",
                    fix="rerun the dry run, pressing Left arrow during episode 2")
    elif len(lengths) < 2:
        report.warn(group, f"{len(lengths)} episode saved, expected 2: was the session stopped early?",
                    fix="rerun the dry run and follow every step")
    if lengths and lengths[0] < 0.9 * full:
        report.ok(group, f"Right arrow ended episode 1 early ({lengths[0] / fps:.1f} s)")
    elif lengths:
        report.warn(group, f"episode 1 ran the full {episode_s:g} s: Right arrow didn't end it early",
                    fix="keep the terminal focused, or grant your terminal Accessibility permission")
    if len(lengths) >= 2:
        rate = lengths[1] / episode_s
        detail = (f"episode 2 has {lengths[1]} frames in {episode_s:g} s: "
                  f"the recording loop ran at {rate:.1f} of {fps} steps/s")
        if lengths[1] < 0.5 * full:
            report.warn(group, f"episode 2 ended early ({lengths[1] / fps:.1f} s), so the loop rate can't be measured",
                        fix="let episode 2 run the full time")
        elif rate >= 0.97 * fps:
            report.ok(group, detail)
        elif rate >= 0.85 * fps:
            report.warn(group, detail + " (demos play back slightly faster than you moved)",
                        fix="close other apps; record with --no-display")
        else:
            report.fail(group, detail + ": demos would play back noticeably faster than you moved",
                        fix="record with --no-display; close other apps; try --dataset.rgb_encoder.vcodec=auto")


def smoke_train(report: Report, s: Settings, root: Path, repo_id: str, out: Path) -> None:
    """Train ACT for a few steps on the rehearsal data, to prove the dataset loads and trains."""
    exe = Path(sys.executable).parent / "lerobot-train"
    args = [
        str(exe), f"--dataset.repo_id={repo_id}", f"--dataset.root={root}", "--policy.type=act",
        f"--policy.device={s.run.device}", "--policy.push_to_hub=false", f"--steps={TRAIN_STEPS}", "--batch_size=2",
        "--num_workers=0", "--log_freq=10", "--save_checkpoint=false", f"--output_dir={out}",
        "--job_name=dry_run",
    ]
    print(f"\nTraining ACT for {TRAIN_STEPS} steps on the rehearsal data ({s.run.device})...", flush=True)
    started = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True)
    output = result.stdout + result.stderr
    losses = [float(x) for x in re.findall(r"\bloss:([0-9.]+)", output)]
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-15:])
        report.fail("training", f"lerobot-train exited with {result.returncode}:\n{tail}")
    elif len(losses) >= 2 and losses[-1] < losses[0]:
        report.ok("training", f"ACT trained {TRAIN_STEPS} steps in {elapsed:.0f} s; "
                              f"loss fell from {losses[0]:.2f} to {losses[-1]:.2f}")
    else:
        report.warn("training", f"ACT trained {TRAIN_STEPS} steps in {elapsed:.0f} s, but the loss didn't fall ({losses})",
                    fix="look at the rehearsal data with --keep")


def dry_run(
    s: Settings, task: Task, *, display: bool, replay: bool, keep: bool, train: bool, extra: list[str]
) -> None:
    """`extra` arguments go to lerobot-record, as they do for a real recording."""
    out = s.outputs_dir / "dry-run" / time.strftime("%Y%m%d-%H%M%S")
    root, repo_id = out / "dataset", f"local/so101_{task.name}_dryrun"
    out.mkdir(parents=True)

    print("Checking the setup...", flush=True)
    found_cameras = [s.camera(name) for name in task.cameras]
    report, lookups = check.run_checks(
        s, cameras=found_cameras, task=task, loop_fps=task.fps, keyboard=True, snapshot_dir=out / "snapshots",
        video=True,
    )
    report.print("SETUP")

    def finish() -> None:
        (out / "report.json").write_text(json.dumps(report.to_json(), indent=2) + "\n")
        if not keep:
            for path in (root, out / "train"):
                shutil.rmtree(path, ignore_errors=True)
        kept = f" The rehearsal dataset is in {root}." if keep and root.exists() else ""
        print(f"\n{report.summary()} Report: {out / 'report.json'}.{kept}")
        if report.failures:
            sys.exit(1)

    if report.failures:
        finish()
        return

    print(SCRIPT.format(episode_s=task.episode_time_s, reset_s=task.reset_time_s))
    try:
        input("Press Enter to start the rehearsal, Ctrl+C to cancel. ")
    except KeyboardInterrupt:
        print()
        finish()
        return
    cameras = task_cameras(s, task, [lookup.camera for lookup in lookups])
    args = record_args(s, task, cameras, root=root, repo_id=repo_id, episodes=2, display=display)
    data_report = Report()
    code = run_lerobot("lerobot-record", [*args, *extra])
    if code != 0:
        data_report.fail("rehearsal", f"lerobot-record exited with {code}")
    summary = None
    if (root / "meta" / "info.json").is_file():
        summary = validate_dataset(root, repo_id, data_report, group="rehearsal data", require_motion=True)
    else:
        data_report.fail("rehearsal", "no dataset was written")
    if summary is not None:
        rehearsal_checks(data_report, summary.lengths, summary.fps, task.episode_time_s)
    if train and summary is not None and not data_report.failures:
        smoke_train(data_report, s, root, repo_id, out / "train")
    if replay and summary is not None and len(summary.lengths) >= 2 and not data_report.failures:
        print("\nThe follower will now replay episode 2. Put the follower in the pose episode 2 started from, "
              "and keep clear of it.")
        try:
            input("Press Enter to replay, Ctrl+C to skip. ")
        except KeyboardInterrupt:
            print()
            data_report.warn("replay", "skipped")
        else:
            replay_args = [*s.follower.lerobot_args(), f"--dataset.repo_id={repo_id}", f"--dataset.root={root}",
                           "--dataset.episode=1", f"--dataset.fps={summary.fps}"]
            if s.teleop.max_relative_target:
                replay_args.append(f"--robot.max_relative_target={s.teleop.max_relative_target}")
            if run_lerobot("lerobot-replay", replay_args) == 0:
                data_report.ok("replay", "replayed episode 2: it should have matched what you did")
            else:
                data_report.fail("replay", "lerobot-replay failed")
    data_report.print("REHEARSAL")
    report.extend(data_report)
    finish()
