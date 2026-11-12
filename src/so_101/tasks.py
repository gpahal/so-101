"""Load `tasks.toml`: the tasks you record demos for and train a policy on, with the scene each one expects.

A task's demos are recorded under the data directory (see `so_101.data`), in degrees, uploaded for
training to <your Hub user>/so101_<task>, and trained into <your Hub user>/<policy type>_so101_<task>.
"""

import json
import sys
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
    cameras: tuple[str, ...]
    fps: int
    episodes: int
    episode_time_s: float
    reset_time_s: float
    props: list[str]
    setup: list[str]

    def policy_for(self, policy_type: str) -> str:
        """Hub repo that training this task with `policy_type` pushes to."""
        return f"{hf_user()}/{policy_type}_so101_{self.name}"

    @property
    def dataset(self) -> str:
        """Hub dataset your built demos are uploaded to."""
        return f"{hf_user()}/so101_{self.name}"

    # Only ACT runs locally: pi0.5 needs --remote (`commands.policy.policy_task`).
    def local_policy_dir(self, revision: str | None) -> Path:
        """Where a checkpoint is downloaded. Each revision gets its own folder, so they can't mix."""
        folder = config_path().parent / "outputs" / "policies" / self.policy_for("act")
        return folder.with_name(f"{folder.name}@{revision.replace('/', '_')}") if revision else folder

    def download_policy(self, revision: str | None) -> Path:
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
                self.policy_for("act"), revision=revision, local_dir=self.local_policy_dir(revision),
                ignore_patterns=["checkpoints/**"],
            )
        )
        config_file = path / "config.json"
        config = json.loads(config_file.read_text())
        known = {f.name for f in dataclasses.fields(PreTrainedConfig.get_choice_class(config["type"]))} | {"type"}
        unknown = {k: v for k, v in config.items() if k not in known}
        if set_fields := {k: v for k, v in unknown.items() if v is not None}:
            raise ConfigError(
                f"{self.policy_for("act")} sets config fields this LeRobot version doesn't support: {set_fields}. "
                "It needs a newer LeRobot."
            )
        if unknown:
            config_file.write_text(json.dumps({k: v for k, v in config.items() if k in known}, indent=4))
        return path

    def policy_config(self, revision: str | None, policy_type: str) -> dict:
        """The checkpoint's config.json."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

        repo = self.policy_for(policy_type)
        try:
            config_file = hf_hub_download(repo, "config.json", revision=revision)
        except RevisionNotFoundError as e:
            raise ConfigError(f"{repo} has no revision {revision!r}.") from e
        except EntryNotFoundError as e:
            raise ConfigError(
                f"Revision {revision!r} of {repo} has no final model: "
                "training checkpoint tags like 020000 keep theirs under checkpoints/."
            ) from e
        return json.loads(Path(config_file).read_text())


def image_sizes(config: dict) -> dict[str, tuple[int, int]]:
    """(width, height) of each camera a checkpoint was trained on, from its config.json
    (`Task.policy_config`), without its empty slots."""
    prefix = "observation.images."
    return {
        key.removeprefix(prefix): (feature["shape"][2], feature["shape"][1])
        for key, feature in config["input_features"].items()
        if key.startswith(prefix) and not key.startswith(prefix + "empty_camera_")
    }


def read_file() -> dict:
    path = tasks_path()
    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"{path} doesn't exist: tasks.toml is read from the folder of so101.toml, "
                          "which SO101_CONFIG chooses.") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e


def default_name() -> str:
    """The task commands use when none is named: `default` at the top of tasks.toml."""
    data = read_file()
    name = data.get("default", "")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"Name a task, or set `default = \"<task>\"` at the top of {tasks_path()}.")
    if not isinstance(data.get(name), dict):
        raise ConfigError(f"`default` in {tasks_path()} is {name!r}, which isn't a task there.")
    return name


def load() -> dict[str, Task]:
    """Every task in tasks.toml: its tables. Top-level keys, such as `default`, aren't tasks."""
    catalog = {}
    for name, t in read_file().items():
        if not isinstance(t, dict):
            continue
        try:
            catalog[name] = Task(
                name=name,
                prompt=t["prompt"],
                cameras=tuple(t.get("cameras", ["top", "wrist"])),
                fps=t.get("fps", 30),
                episodes=t.get("episodes", 50),
                episode_time_s=t.get("episode_time_s", 20),
                reset_time_s=t.get("reset_time_s", 10),
                props=t.get("props", []),
                setup=t.get("setup", []),
            )
        except (KeyError, ValueError, TypeError) as e:
            problem = f"the setting {e.args[0]!r} is missing" if isinstance(e, KeyError) else e
            raise ConfigError(f"{tasks_path()}: task {name!r}: {problem}") from e
    return catalog


def get_task(name: str | None) -> Task:
    """The named task, or the default from tasks.toml when no name is given."""
    if name is None:
        name = default_name()
        print(f"Task: {name} (the default in tasks.toml)", file=sys.stderr)
    catalog = load()
    if name not in catalog:
        sys.exit(f"Unknown task {name!r}. Tasks: {', '.join(catalog)}. See `uv run so101 tasks`.")
    return catalog[name]
