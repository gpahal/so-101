"""Shared by `so101 record` and its dry run: the lerobot-record command line and session records."""

import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from so_101 import data, svtshim
from so_101.report import Report
from so_101.settings import ARMS, Camera, Settings, cameras_arg
from so_101.tasks import Task


def record_args(
    s: Settings,
    task: Task,
    cameras: list[Camera],
    *,
    root: Path,
    repo_id: str,
    episodes: int,
    display: bool,
) -> list[str]:
    """Arguments for lerobot-record. `cameras` are the task's cameras, with their current indices."""
    return [
        *s.follower_args(),
        cameras_arg([cam.lerobot_spec() for cam in cameras]),
        *s.leader.lerobot_args(),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={root}",
        f"--dataset.single_task={task.prompt}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.episode_time_s={task.episode_time_s}",
        f"--dataset.reset_time_s={task.reset_time_s}",
        f"--dataset.fps={task.fps}",
        "--dataset.no_stamp=true",
        "--dataset.push_to_hub=false",
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
        "--dataset.rgb_encoder.preset=10",  # SVT-AV1 has no preset above 10; the default of 12 warns per episode
        f"--display_data={str(display).lower()}",
    ]


def _git(s: Settings, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=s.path.parent, capture_output=True, text=True,
                            env=svtshim.other_env())
    return result.stdout.strip() if result.returncode == 0 else ""


def session_manifest(s: Settings, task: Task, cameras: list[Camera], report: Report, note: str) -> dict:
    """Everything a session was recorded with, so later sessions can be compared with it."""
    calibration = {}
    for name in ARMS:
        path = s.arm(name).calibration_file
        calibration[name] = {
            "file": str(path),
            "sha256": data.file_hash(path),
            "values": json.loads(path.read_text()) if path.is_file() else None,
        }
    return {
        "task": task.name,
        "prompt": task.prompt,
        "note": note,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": None,
        "episodes": 0,
        "recording": {
            "fps": task.fps,
            "episode_time_s": task.episode_time_s,
            "reset_time_s": task.reset_time_s,
            "max_relative_target": s.teleop.max_relative_target,
        },
        "cameras": {
            cam.name: {"camera": cam.name, "index": cam.index, "usb_id": cam.usb_id, "width": cam.width,
                       "height": cam.height, "fps": cam.fps, "rotation": cam.rotation, "controls": cam.controls}
            for cam in cameras
        },
        "calibration": calibration,
        "software": {
            "so101_commit": _git(s, "rev-parse", "HEAD"),
            "so101_uncommitted_changes": bool(_git(s, "status", "--porcelain")),
            "lerobot": version("lerobot"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": s.path.read_text(),
        "checks": report.to_json(),
    }


def policy_commit(checkpoint: Path) -> str | None:
    """The Hub commit a downloaded checkpoint came from, as huggingface_hub records it beside the files."""
    metadata = checkpoint / ".cache" / "huggingface" / "download" / "config.json.metadata"
    return metadata.read_text().splitlines()[0] if metadata.is_file() else None


def run_manifest(
    s: Settings,
    task: Task,
    cameras: list[Camera],
    report: Report,
    *,
    policy: dict,
    image_sizes: dict[str, tuple[int, int]],
) -> dict:
    """Everything a `so101 serve` run ran with: a session's record, plus how the arm was driven and
    `policy`: its type, repo, revision, commit, action steps, and whether it ran here or remotely. For a
    remote run, `so101-control` adds the checkpoint the server admitted it to (`artifact_identity`)."""
    manifest = session_manifest(s, task, cameras, report, note="")
    # run.json is never changed after the start, so it holds no finish time or attempt count: the
    # `run_end` event at the end of events.jsonl has them.
    for key in ("note", "finished", "episodes", "recording"):
        del manifest[key]
    return {
        **manifest,
        "control": {
            "fps": task.fps,
            "max_relative_target": s.teleop.max_relative_target,
            "rest_pose": s.control.rest_pose,
        },
        "policy": {
            **policy,
            "device": None if policy.get("inference") == "remote" else s.run.device,
            "image_sizes": {key: {"width": w, "height": h} for key, (w, h) in image_sizes.items()},
        },
    }
