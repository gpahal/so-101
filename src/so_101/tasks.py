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
        return self.policy_repo or f"{hf_user()}/act_so101_{self.name}"

    @property
    def dataset(self) -> str:
        """Hub dataset: the one a pretrained policy was trained on, or where your built demos are uploaded."""
        return self.dataset_repo or f"{hf_user()}/so101_{self.name}"

    @property
    def local_policy_dir(self) -> Path:
        return config_path().parent / "outputs" / "policies" / self.policy

    def download_policy(self) -> Path:
        """Download the checkpoint and make its config loadable by the installed LeRobot.

        Checkpoints saved by newer LeRobot versions can carry config fields this version doesn't
        know, which LeRobot rejects. Fields left unset (null) are dropped; anything else is an error.
        """
        import dataclasses

        from huggingface_hub import snapshot_download
        import lerobot.policies  # noqa: F401  (registers the policy config classes)
        from lerobot.configs.policies import PreTrainedConfig

        path = Path(snapshot_download(self.policy, local_dir=self.local_policy_dir))
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

    def image_sizes(self) -> dict[str, tuple[int, int]]:
        """(width, height) of each camera the checkpoint was trained on, from its config.json."""
        from huggingface_hub import hf_hub_download

        config = json.loads(Path(hf_hub_download(self.policy, "config.json")).read_text())
        prefix = "observation.images."
        return {
            key.removeprefix(prefix): (feature["shape"][2], feature["shape"][1])
            for key, feature in config["input_features"].items()
            if key.startswith(prefix)
        }


def load() -> dict[str, Task]:
    data = tomllib.loads(tasks_path().read_text())
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
