"""Where recorded demos and policy runs live, and the edits applied to demos before training.

    <data_dir>/<task>/
      sessions/<YYYYmmdd-HHMMSS>/
        dataset/        one LeRobot dataset per `so101 record` session; never modified
        session.json    what it was recorded with: settings, calibration, cameras, checks
      edits.toml        trims and deletions from `so101 review`
      train/            the training dataset `so101 build` makes from sessions + edits, plus runs with --with
      build.json        what `train/` was built from, to tell when it is out of date
      runs/<YYYYmmdd-HHMMSS>/
        dataset/        one LeRobot dataset per `so101 serve` run, one episode per attempt; never modified
        run.json        what it ran with: settings, calibration, cameras, checks, policy, software.
                        Written once, when the run starts, and never changed
        events.jsonl    every command, state change and attempt end, one JSON object per line. It
                        ends with run_end (finished time, attempts); a run without one crashed
        meta-copy/<n>/meta/  a copy of dataset/meta/ as of the last finished save, of n episodes.
                        A kill during a later save can tear dataset/meta/, which every episode
                        shares; readers fall back to this copy, so it costs only that save's attempt
        labels.toml     review decisions (outcomes, suspect frames); the only file edited later

Sessions and runs are append-only: `so101 review` writes only edits.toml, and `so101 review --runs`
only each run's labels.toml. Runs sit beside the sessions, never among them, so a policy's attempts
reach a build only through `so101 build --with`, which reads the runs and their labels.toml and
rebuilds when they change.
"""

import hashlib
import json
import shutil
import time
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

from so_101.settings import ConfigError, Settings

if TYPE_CHECKING:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    from so_101.tasks import Task


# A run's review decisions, the only file of a run edited after it's recorded.
LABELS_FILE = "labels.toml"
# Beside a run's dataset: the recorder's copy of the dataset's metadata after each finished save.
META_COPY_DIR = "meta-copy"


def task_dir(s: Settings, task: "Task") -> Path:
    return s.data_dir / task.name


def edits_path(s: Settings, task: "Task") -> Path:
    return task_dir(s, task) / "edits.toml"


def train_root(s: Settings, task: "Task") -> Path:
    return task_dir(s, task) / "train"


def build_manifest_path(s: Settings, task: "Task") -> Path:
    return task_dir(s, task) / "build.json"


@dataclass(frozen=True)
class Recording:
    """A folder of sessions/ or runs/: one dataset and the manifest it was recorded with."""

    FOLDER: ClassVar[str]  # the task's folder these sit in
    MANIFEST: ClassVar[str]

    task: str
    name: str  # YYYYmmdd-HHMMSS
    path: Path

    @property
    def dataset_root(self) -> Path:
        return self.path / "dataset"

    @property
    def manifest_path(self) -> Path:
        return self.path / self.MANIFEST

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.is_file() else {}

    def write_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def listed(cls, s: Settings, task: "Task") -> list[Self]:
        """Every one of the task's, oldest first."""
        root = task_dir(s, task) / cls.FOLDER
        if not root.is_dir():
            return []
        return [cls(task.name, p.name, p) for p in sorted(root.iterdir()) if p.is_dir()]


@dataclass(frozen=True)
class Session(Recording):
    FOLDER = "sessions"
    MANIFEST = "session.json"

    @property
    def repo_id(self) -> str:
        # Sessions are never pushed under this name; LeRobot just needs one.
        return f"local/so101_{self.task}_{self.name}"

    @cached_property
    def _loaded(self) -> tuple["LeRobotDatasetMetadata | None", str | None]:
        return _meta(self.repo_id, self.dataset_root)

    @property
    def meta(self) -> "LeRobotDatasetMetadata | None":
        """The dataset's metadata, or None if the session has no usable dataset (empty or interrupted)."""
        return self._loaded[0]

    @property
    def meta_error(self) -> str | None:
        """The error from loading the dataset's metadata, or None if there was none."""
        return self._loaded[1]

    @property
    def episodes(self) -> int:
        return self.meta.total_episodes if self.meta else 0


def _meta(repo_id: str, root: Path) -> tuple["LeRobotDatasetMetadata | None", str | None]:
    """The dataset's metadata, or None without a usable one, and the error from loading it, if any."""
    if not (root / "meta" / "info.json").is_file():
        return None, None
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    try:
        meta = LeRobotDatasetMetadata(repo_id, root=root)
    except Exception as e:
        return None, repr(e)
    return (meta if meta.total_episodes > 0 and meta.episodes is not None else None), None


def sessions(s: Settings, task: "Task") -> list[Session]:
    return Session.listed(s, task)


def new_session(s: Settings, task: "Task") -> Session:
    name = time.strftime("%Y%m%d-%H%M%S")
    path = task_dir(s, task) / Session.FOLDER / name
    path.mkdir(parents=True)
    return Session(task.name, name, path)


@dataclass(frozen=True)
class Run(Recording):
    """One `so101 serve` run: its attempts, as recorded by `control.recorder`."""

    FOLDER = "runs"
    MANIFEST = "run.json"

    @classmethod
    def at(cls, path: Path) -> "Run":
        """The run in the folder `path`, which is <data_dir>/<task>/runs/<name>."""
        path = Path(path)
        return cls(path.parent.parent.name, path.name, path)

    @property
    def events_path(self) -> Path:
        return self.path / "events.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.path / LABELS_FILE

    @cached_property
    def meta(self) -> "LeRobotDatasetMetadata | None":
        """The dataset's metadata, or None if the run saved no attempt or never finalized its dataset.
        When the live metadata can't be loaded, as when a kill during a save tore it, it's the copy
        from the last finished save, still reading the data and video files from the live dataset."""
        if (meta := _meta(self.repo_id, self.dataset_root)[0]) is not None:
            return meta
        if (copy := self.meta_copy_root) is None or (meta := _meta(self.repo_id, copy)[0]) is None:
            return None
        meta.root = self.dataset_root  # only the metadata comes from the copy
        return meta

    def info(self) -> dict | None:
        """The dataset's meta/info.json, or its copy's when the live one can't be read, or None."""
        for root in (self.dataset_root, self.meta_copy_root):
            if root is None:
                continue
            try:
                return json.loads((root / "meta" / "info.json").read_text())
            except (OSError, ValueError):  # missing, or cut short
                continue
        return None

    @property
    def meta_copy_root(self) -> Path | None:
        """The newest copy of the dataset's metadata, as a dataset root with meta/ in it, or None."""
        folder = self.path / META_COPY_DIR
        copies = [p for p in folder.iterdir() if p.name.isdigit()] if folder.is_dir() else []
        return max(copies, key=lambda p: int(p.name)) if copies else None

    def save_meta_copy(self, episodes: int) -> None:
        """Copy the dataset's meta/ folder, just finalized with `episodes` episodes, beside it. The copy
        is built under a temporary name and renamed into place, then older copies, the folders beside it,
        are removed, so a kill at any point leaves at least one whole copy once one save has finished."""
        folder = self.path / META_COPY_DIR
        building = folder / f".building-{episodes}"
        shutil.copytree(self.dataset_root / "meta", building / "meta")
        done = folder / str(episodes)
        building.rename(done)
        for old in folder.iterdir():
            # Only the older copies: a file such as Finder's .DS_Store stays.
            if old != done and old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)

    @property
    def repo_id(self) -> str:
        # Runs are never pushed under this name; LeRobot just needs one.
        return f"local/so101_{self.task}_run_{self.name}"


def runs(s: Settings, task: "Task") -> list[Run]:
    return Run.listed(s, task)


def new_run(s: Settings, task: "Task") -> Run:
    """A run named for now. Its folder isn't made here: `so101-control` makes it, with run.json,
    once it is ready to record, so a start that fails leaves nothing to delete."""
    name = time.strftime("%Y%m%d-%H%M%S")
    return Run(task.name, name, task_dir(s, task) / Run.FOLDER / name)


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


def whole(value) -> int:
    """A whole number as written: 3, not 3.7, "3" or true, which int() would quietly accept."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{value!r} isn't a whole number")
    return value


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
                episode=whole(entry["episode"]),
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
        lines.append(f"\n[[episode]]\nsession = {json.dumps(edit.session, ensure_ascii=False)}\nepisode = {edit.episode}\n")
        if edit.keep:
            lines.append(f"keep = [{edit.keep[0]:.2f}, {edit.keep[1]:.2f}]\n")
        if edit.delete is not None:
            lines.append(f"delete = {json.dumps(edit.delete, ensure_ascii=False)}\n")
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
    """Each joint's median position at the start of the built demos, if there's a local training set.

    Only the demos count: a run's cut added by `build --with` starts mid-task, wherever its Teleop
    or policy segment began. The demos are the build's first episodes, one per build.json entry
    from a session. Without a build.json, as after a kill during a build, there's no start pose."""
    import numpy as np
    import pandas as pd

    root = train_root(s, task)
    files = sorted(root.glob("data/*/*.parquet"))
    manifest = build_manifest_path(s, task)
    if not files or not manifest.is_file():
        return None
    names = json.loads((root / "meta" / "info.json").read_text())["features"]["observation.state"]["names"]
    frames = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files)
    frames = frames[frames["frame_index"] == 0]
    demos = sum("session" in e for e in json.loads(manifest.read_text())["episodes"])
    frames = frames[frames["episode_index"] < demos]
    if frames.empty:
        return None
    starts = np.stack(frames["observation.state"])
    return dict(zip(names, np.median(starts, axis=0).round(1).tolist(), strict=True))


# --- Builds ---------------------------------------------------------------------------------------


def build_inputs(s: Settings, task: "Task", sources: tuple[str, ...] = (), keep_labels: bool = False) -> dict:
    """Everything a build depends on. A build is current when this equals what build.json recorded.

    `sources` are the run sources besides the demos (`so101 build --with`). A demos-only build
    without labels records exactly what it did before runs existed, so its build.json is unchanged.
    """
    inputs = {
        "prompt": task.prompt,
        "sessions": {
            session.name: {"episodes": session.episodes, "frames": session.meta.total_frames if session.meta else 0}
            for session in sessions(s, task)
        },
        "edits_sha256": file_hash(edits_path(s, task)),
    }
    if sources:
        inputs["sources"] = ["demos", *sources]
        inputs["blame_s"] = s.control.blame_s  # it moves the suspect frames, and so the cuts
        inputs["runs"] = {
            run.name: {
                "episodes": run.meta.total_episodes if run.meta else 0,
                "frames": run.meta.total_frames if run.meta else 0,
                "events_bytes": run.events_path.stat().st_size if run.events_path.is_file() else 0,
                "labels_sha256": file_hash(run.labels_path),
            }
            for run in runs(s, task)
        }
    if keep_labels:
        inputs["keep_labels"] = True
    if sources or keep_labels:
        from so_101.control.analysis import RULES_VERSION

        inputs["rules"] = RULES_VERSION  # how runs are cut and labelled: builds under older rules rebuild
    return inputs


def build_is_current(s: Settings, task: "Task", sources: tuple[str, ...] = (), keep_labels: bool = False) -> bool:
    path = build_manifest_path(s, task)
    if not path.is_file() or not (train_root(s, task) / "meta" / "info.json").is_file():
        return False
    return json.loads(path.read_text()).get("inputs") == build_inputs(s, task, sources, keep_labels)
