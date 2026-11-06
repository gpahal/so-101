"""Load `tasks.toml`: policies to run on the follower, with the scene each one expects.

A task with `policy` and `dataset` set is someone else's pretrained policy. A task without them is
your own: demos are recorded under the data directory (see `so_101.data`), uploaded for training to
<your Hub user>/so101_<task>, and trained into <your Hub user>/act_so101_<task>.
"""

import json
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from so_101.settings import ConfigError, config_path


def tasks_path() -> Path:
    return config_path().parent / "tasks.toml"


@cache
def hf_user() -> str:
    from huggingface_hub import get_token, whoami

    if not get_token():
        raise ConfigError("Not logged in to Hugging Face. Run `uv run hf auth login` with a write token first.")
    return whoami()["name"]


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    cameras: dict[str, str]
    use_degrees: bool
    fps: int
    episodes: int
    episode_time_s: float
    reset_time_s: float
    props: list[str]
    setup: list[str]
    policy_repo: str = ""
    dataset_repo: str = ""

    @property
    def own(self) -> bool:
        """True for a task you record and train yourself."""
        return not self.policy_repo

    @property
    def policy(self) -> str:
        return self.policy_repo or self.trained_policy("act")

    def trained_policy(self, policy_type: str) -> str:
        """Hub repo that training this task with `policy_type` pushes to."""
        return f"{hf_user()}/{policy_type}_so101_{self.name}"

    @property
    def dataset(self) -> str:
        """Hub dataset: the one a pretrained policy was trained on, or where your built demos are uploaded."""
        return self.dataset_repo or f"{hf_user()}/so101_{self.name}"

    def local_policy_dir(self, revision: str | None = None) -> Path:
        """Where a checkpoint is downloaded. Each revision gets its own folder, so they can't mix."""
        folder = config_path().parent / "outputs" / "policies" / self.policy
        return folder.with_name(f"{folder.name}@{revision.replace('/', '_')}") if revision else folder

    def download_policy(self, revision: str | None = None) -> Path:
        """Download the checkpoint and make its config loadable by the installed LeRobot.

        Checkpoints saved by newer LeRobot versions can carry config fields this version doesn't
        know, which LeRobot rejects. Fields left unset (null) are dropped; anything else is an error.
        """
        import dataclasses

        from huggingface_hub import snapshot_download
        import lerobot.policies  # noqa: F401  (registers the policy config classes)
        from lerobot.configs.policies import PreTrainedConfig

        # Training checkpoints are several times the size of the final model, and running doesn't use them.
        path = Path(
            snapshot_download(
                self.policy, revision=revision, local_dir=self.local_policy_dir(revision),
                ignore_patterns=["checkpoints/**"],
            )
        )
        config_file = path / "config.json"
        config = json.loads(config_file.read_text())
        known = {f.name for f in dataclasses.fields(PreTrainedConfig.get_choice_class(config["type"]))} | {"type"}
        unknown = {k: v for k, v in config.items() if k not in known}
        if set_fields := {k: v for k, v in unknown.items() if v is not None}:
            raise ConfigError(
                f"{self.policy} sets config fields this LeRobot version doesn't support: {set_fields}. "
                "It needs a newer LeRobot."
            )
        if unknown:
            config_file.write_text(json.dumps({k: v for k, v in config.items() if k in known}, indent=4))
        return path

    @property
    def viewer_url(self) -> str:
        return f"https://huggingface.co/spaces/lerobot/visualize_dataset?path=/{self.dataset}/episode_0"

    def image_sizes(self, revision: str | None = None) -> dict[str, tuple[int, int]]:
        """(width, height) of each camera the checkpoint was trained on, from its config.json."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

        try:
            config_file = hf_hub_download(self.policy, "config.json", revision=revision)
        except RevisionNotFoundError as e:
            raise ConfigError(f"{self.policy} has no revision {revision!r}.") from e
        except EntryNotFoundError as e:
            raise ConfigError(
                f"Revision {revision!r} of {self.policy} has no final model: "
                "training checkpoint tags like 020000 keep theirs under checkpoints/."
            ) from e
        config = json.loads(Path(config_file).read_text())
        prefix = "observation.images."
        return {
            key.removeprefix(prefix): (feature["shape"][2], feature["shape"][1])
            for key, feature in config["input_features"].items()
            if key.startswith(prefix)
        }


def default_name() -> str:
    """The task commands use when none is named: `default` at the top of tasks.toml."""
    data = tomllib.loads(tasks_path().read_text())
    name = data.get("default", "")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"Name a task, or set `default = \"<task>\"` at the top of {tasks_path()}.")
    if not isinstance(data.get(name), dict):
        raise ConfigError(f"`default` in {tasks_path()} is {name!r}, which isn't a task there.")
    return name


def load() -> dict[str, Task]:
    """Every task in tasks.toml: its tables. Top-level keys, such as `default`, aren't tasks."""
    data = {name: t for name, t in tomllib.loads(tasks_path().read_text()).items() if isinstance(t, dict)}
    return {
        name: Task(
            name=name,
            prompt=t["prompt"],
            cameras=t.get("cameras", {"front": "front", "wrist": "wrist"}),
            use_degrees=t.get("use_degrees", True),
            fps=t.get("fps", 30),
            episodes=t.get("episodes", 50),
            episode_time_s=t.get("episode_time_s", 20),
            reset_time_s=t.get("reset_time_s", 10),
            props=t.get("props", []),
            setup=t.get("setup", []),
            policy_repo=t.get("policy", ""),
            dataset_repo=t.get("dataset", ""),
        )
        for name, t in data.items()
    }
