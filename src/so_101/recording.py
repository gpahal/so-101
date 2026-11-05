"""Shared by `so101 record` and its dry run: the lerobot-record command line and session records."""

import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from so_101 import data
from so_101.report import Report
from so_101.settings import ARMS, Camera, Settings, cameras_arg
from so_101.tasks import Task


def record_args(
    s: Settings,
    task: Task,
    cameras: dict[str, Camera],
    *,
    root: Path,
    repo_id: str,
    episodes: int,
    episode_time_s: float,
    reset_time_s: float,
    display: bool,
) -> list[str]:
    """Arguments for lerobot-record. `cameras` maps the dataset's camera names to cameras with current indices."""
    args = [
        *s.follower.lerobot_args(),
        cameras_arg([cam.lerobot_spec(key=key) for key, cam in cameras.items()]),
        *s.leader.lerobot_args(),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={root}",
        f"--dataset.single_task={task.prompt}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        f"--dataset.fps={task.fps}",
        "--dataset.no_stamp=true",
        "--dataset.push_to_hub=false",
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
        "--dataset.rgb_encoder.preset=10",  # SVT-AV1 has no preset above 10; the default of 12 warns per episode
        f"--display_data={str(display).lower()}",
    ]
    if s.teleop.max_relative_target:
        args.append(f"--robot.max_relative_target={s.teleop.max_relative_target}")
    return args


def task_cameras(s: Settings, task: Task, found: list[Camera]) -> dict[str, Camera]:
    """The task's dataset camera names, mapped to the cameras as found by the checks."""
    by_name = {cam.name: cam for cam in found}
    return {key: by_name.get(ours, s.camera(ours)) for key, ours in task.cameras.items()}


def _git(s: Settings, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=s.path.parent, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def session_manifest(s: Settings, task: Task, cameras: dict[str, Camera], report: Report, note: str) -> dict:
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
            key: {"camera": cam.name, "index": cam.index, "usb_id": cam.usb_id, "width": cam.width,
                  "height": cam.height, "fps": cam.fps, "rotation": cam.rotation, "controls": cam.controls}
            for key, cam in cameras.items()
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
