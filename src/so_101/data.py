"""Where recorded demos live, and the edits applied to them before training.

    <data_dir>/<task>/
      sessions/<YYYYmmdd-HHMMSS>/
        dataset/        one LeRobot dataset per `so101 record` session; never modified
        session.json    what it was recorded with: settings, calibration, cameras, checks
      edits.toml        trims and deletions from `so101 review`
      train/            the training dataset `so101 build` makes from sessions + edits
      build.json        what `train/` was built from, to tell when it is out of date
"""

import hashlib
import json
import shutil
import time
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from so_101.settings import ConfigError, Settings

if TYPE_CHECKING:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    from so_101.tasks import Task


def task_dir(s: Settings, task: "Task") -> Path:
    return s.data_dir / task.name


def edits_path(s: Settings, task: "Task") -> Path:
    return task_dir(s, task) / "edits.toml"


def train_root(s: Settings, task: "Task") -> Path:
    return task_dir(s, task) / "train"


def build_manifest_path(s: Settings, task: "Task") -> Path:
    return task_dir(s, task) / "build.json"


@dataclass(frozen=True)
class Session:
    task: str
    name: str  # YYYYmmdd-HHMMSS
    path: Path

    @property
    def dataset_root(self) -> Path:
        return self.path / "dataset"

    @property
    def manifest_path(self) -> Path:
        return self.path / "session.json"

    @property
    def repo_id(self) -> str:
        # Sessions are never pushed under this name; LeRobot just needs one.
        return f"local/so101_{self.task}_{self.name}"

    @cached_property
    def meta(self) -> "LeRobotDatasetMetadata | None":
        """The dataset's metadata, or None if the session has no usable dataset (empty or interrupted)."""
        if not (self.dataset_root / "meta" / "info.json").is_file():
            return None
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        try:
            meta = LeRobotDatasetMetadata(self.repo_id, root=self.dataset_root)
        except Exception:
            return None
        return meta if meta.total_episodes > 0 and meta.episodes is not None else None

    @property
    def episodes(self) -> int:
        return self.meta.total_episodes if self.meta else 0

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.is_file() else {}

    def write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def sessions(s: Settings, task: "Task") -> list[Session]:
    root = task_dir(s, task) / "sessions"
    if not root.is_dir():
        return []
    return [Session(task.name, p.name, p) for p in sorted(root.iterdir()) if p.is_dir()]


def new_session(s: Settings, task: "Task") -> Session:
    name = time.strftime("%Y%m%d-%H%M%S")
    path = task_dir(s, task) / "sessions" / name
    path.mkdir(parents=True)
    return Session(task.name, name, path)


def remove_session(session: Session) -> None:
    shutil.rmtree(session.path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


# --- Edits ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Edit:
    """What to do with one recorded episode. No keep and no delete means: reviewed, keep all of it."""

    session: str
    episode: int
    keep: tuple[float, float] | None = None  # seconds from the start of the recorded episode
    delete: str | None = None  # reason

    @property
    def key(self) -> tuple[str, int]:
        return (self.session, self.episode)


EDITS_HEADER = """\
# Edits to recorded demos, applied by `uv run so101 build <task>`. `uv run so101 review <task>` writes
# this file; you can also edit it by hand. One [[episode]] per reviewed episode:
#   session, episode   which recording (a folder under sessions/, and its episode number from 0)
#   keep = [start, end]   train on this part only, in seconds from the start of the episode
#   delete = "reason"     leave the episode out of training
#   neither               reviewed: keep all of it
"""


def load_edits(s: Settings, task: "Task") -> dict[tuple[str, int], Edit]:
    path = edits_path(s, task)
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    edits = {}
    for i, entry in enumerate(data.get("episode", [])):
        try:
            keep = entry.get("keep")
            edit = Edit(
                session=str(entry["session"]),
                episode=int(entry["episode"]),
                keep=(float(keep[0]), float(keep[1])) if keep is not None else None,
                delete=str(entry["delete"]) if "delete" in entry else None,
            )
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ConfigError(f"{path}: [[episode]] number {i + 1} is malformed ({e!r}).") from e
        if edit.keep and edit.delete is not None:
            raise ConfigError(f"{path}: session {edit.session} episode {edit.episode} has both keep and delete.")
        if edit.keep and not 0 <= edit.keep[0] < edit.keep[1]:
            raise ConfigError(f"{path}: session {edit.session} episode {edit.episode}: keep needs 0 <= start < end.")
        edits[edit.key] = edit
    return edits


def save_edits(s: Settings, task: "Task", edits: dict[tuple[str, int], Edit]) -> None:
    lines = [EDITS_HEADER]
    for edit in sorted(edits.values(), key=lambda e: e.key):
        lines.append(f'\n[[episode]]\nsession = "{edit.session}"\nepisode = {edit.episode}\n')
        if edit.keep:
            lines.append(f"keep = [{edit.keep[0]:.2f}, {edit.keep[1]:.2f}]\n")
        if edit.delete is not None:
            lines.append(f"delete = {json.dumps(edit.delete)}\n")
    path = edits_path(s, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text("".join(lines))
    tmp.replace(path)


def kept_episodes(s: Settings, task: "Task") -> int:
    """Recorded episodes across all sessions, minus the ones edits.toml deletes."""
    edits = load_edits(s, task)
    return sum(
        1
        for session in sessions(s, task)
        for ep in range(session.episodes)
        if (edit := edits.get((session.name, ep))) is None or edit.delete is None
    )


def demo_start_pose(s: Settings, task: "Task") -> dict[str, float] | None:
    """Each joint's median position at the start of the built demos, if there's a local training set."""
    import numpy as np
    import pandas as pd

    root = train_root(s, task)
    files = sorted(root.glob("data/*/*.parquet"))
    if not files:
        return None
    names = json.loads((root / "meta" / "info.json").read_text())["features"]["observation.state"]["names"]
    frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files)
    starts = np.stack(frames[frames["frame_index"] == 0]["observation.state"])
    return dict(zip(names, np.median(starts, axis=0).round(1).tolist(), strict=True))


# --- Builds ---------------------------------------------------------------------------------------


def build_inputs(s: Settings, task: "Task") -> dict:
    """Everything a build depends on. A build is current when this equals what build.json recorded."""
    return {
        "prompt": task.prompt,
        "sessions": {
            session.name: {"episodes": session.episodes, "frames": session.meta.total_frames if session.meta else 0}
            for session in sessions(s, task)
        },
        "edits_sha256": file_hash(edits_path(s, task)),
    }


def build_is_current(s: Settings, task: "Task") -> bool:
    path = build_manifest_path(s, task)
    if not path.is_file() or not (train_root(s, task) / "meta" / "info.json").is_file():
        return False
    return json.loads(path.read_text()).get("inputs") == build_inputs(s, task)
