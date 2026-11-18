"""`so101 train`: which model a training job starts from, checked before anything uploads or bills, and
which build a skills model trains on, recorded where `run --skills` reads it."""

import argparse
import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from so_101 import cli, settings
from so_101.commands.policy import local_checkpoint
from so_101.settings import ConfigError


class BaseTest(unittest.TestCase):
    def test_a_base_replaces_pi05s_default_and_keeps_its_other_flags(self) -> None:
        flags = cli.policy_flags("pi05", cli.parse_base("gpahal/pi05_so101_block-container@100-demos"))
        self.assertEqual(flags[:2], ["--policy.pretrained_path=gpahal/pi05_so101_block-container",
                                     "--policy.pretrained_revision=100-demos"])
        self.assertNotIn("--policy.pretrained_path=lerobot/pi05_base", flags)
        self.assertEqual(flags[2:], [*cli.POLICY_FLAGS["pi05"][1:], *cli.TRAIN_FLAGS])

    def test_a_base_without_a_revision_starts_from_main(self) -> None:
        self.assertEqual(cli.policy_flags("act", cli.parse_base("gpahal/act_so101_block-container")),
                         ["--policy.pretrained_path=gpahal/act_so101_block-container", *cli.TRAIN_FLAGS])

    def test_without_a_base_each_policy_keeps_its_default(self) -> None:
        self.assertEqual(cli.policy_flags("act", None), cli.TRAIN_FLAGS)
        self.assertEqual(cli.policy_flags("pi05", None), [*cli.POLICY_FLAGS["pi05"], *cli.TRAIN_FLAGS])
        self.assertIsNone(cli.default_base("act"))
        self.assertEqual(cli.default_base("pi05"), "lerobot/pi05_base")

    def test_pi05s_flags_train_the_whole_model_with_32_bit_weights(self) -> None:
        # Parsed by LeRobot's own training config, so a renamed or mistyped flag fails here, not in a paid job.
        import draccus
        import lerobot.policies  # noqa: F401  registers the policy types
        import torch
        from lerobot.configs.train import TrainPipelineConfig

        config = draccus.parse(TrainPipelineConfig, args=[
            "--policy.type=pi05", "--dataset.repo_id=x/y", "--policy.push_to_hub=false", *cli.policy_flags("pi05", None)])
        policy = config.policy
        self.assertEqual((policy.train_expert_only, policy.freeze_vision_encoder), (False, False))
        # bfloat16 weights can't hold an update under 1/512 of a weight: only the arithmetic may be bfloat16.
        self.assertEqual((policy.dtype, config.accelerator.mixed_precision), (torch.float32, "bf16"))
        self.assertTrue(policy.gradient_checkpointing)
        self.assertEqual(policy.empty_cameras, 0)  # a blank third camera only costs time
        self.assertTrue(config.dataset.image_transforms.enable)
        self.assertEqual((config.ema.type, config.ema.decay), ("constant", 0.99))
        self.assertEqual((policy.compile_model, policy.compile_mode), (True, "default"))

    def test_every_policy_trains_with_image_augmentation(self) -> None:
        # Parsed by LeRobot's own training config, as each training job parses it.
        import draccus
        import lerobot.policies  # noqa: F401  registers the policy types
        from lerobot.configs.train import TrainPipelineConfig

        for policy in cli.POLICY_FLAGS:
            for base in (None, cli.parse_base(f"gpahal/{policy}_so101_block-container@100-demos")):
                with self.subTest(policy=policy, base=base):
                    config = draccus.parse(TrainPipelineConfig, args=[
                        f"--policy.type={policy}", "--dataset.repo_id=x/y", "--policy.push_to_hub=false", *cli.policy_flags(policy, base)])
                    self.assertTrue(config.dataset.image_transforms.enable)
        # A flag forwarded after them, as `so101 train` passes the rest of its command line, turns it off.
        config = draccus.parse(TrainPipelineConfig, args=[
            "--policy.type=act", "--dataset.repo_id=x/y", "--policy.push_to_hub=false", *cli.policy_flags("act", None),
            "--dataset.image_transforms.enable=false"])
        self.assertFalse(config.dataset.image_transforms.enable)

    def test_a_base_must_be_a_hub_model_name(self) -> None:
        for text in ("act_so101_block-container", "gpahal/", "/act", "a/b/c", "gpahal/act@"):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                cli.parse_base(text)

    def test_a_base_of_another_policy_type_stops_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"type": "act"}))
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(config)) as download:
                cli.check_base("gpahal/act_so101_block-container", "100-demos", "act")
                download.assert_called_once_with("gpahal/act_so101_block-container", "config.json", revision="100-demos")
                with self.assertRaises(SystemExit) as stop:
                    cli.check_base("gpahal/act_so101_block-container", None, "pi05")
        self.assertIn("is a model of type act, not pi05", str(stop.exception))

    def test_a_skills_model_and_a_whole_demo_act_model_never_start_from_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            whole, skills = Path(tmp) / "whole.json", Path(tmp) / "skills.json"
            whole.write_text(json.dumps({"type": "act", "input_features": {"observation.state": {}}}))
            skills.write_text(json.dumps({"type": "act",
                                          "input_features": {"observation.environment_state": {"shape": [3]}}}))
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(whole)):
                cli.check_base("me/act_so101_task", None, "act")
                with self.assertRaises(SystemExit) as stop:
                    cli.check_base("me/act_so101_task", None, "act", skills=3)
                self.assertIn("was trained on the whole demos", str(stop.exception))
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(skills)):
                cli.check_base("me/act_so101_task_skills", None, "act", skills=3)
                with self.assertRaises(SystemExit) as stop:
                    cli.check_base("me/act_so101_task_skills", None, "act")
                self.assertIn("was trained on the skills dataset", str(stop.exception))


    def test_a_skills_base_with_another_number_of_skills_stops_before_training(self) -> None:
        # A base trained on two skills reads 2 values; this task has 3.
        with tempfile.TemporaryDirectory() as tmp:
            two = Path(tmp) / "two.json"
            two.write_text(json.dumps({"type": "act",
                                       "input_features": {"observation.environment_state": {"shape": [2]}}}))
            with mock.patch("huggingface_hub.hf_hub_download", return_value=str(two)):
                cli.check_base("me/act_so101_task_skills", None, "act", skills=2)
                with self.assertRaises(SystemExit) as stop:
                    cli.check_base("me/act_so101_task_skills", None, "act", skills=3)
        self.assertIn("reads a skill one-hot of shape [2], and this task has 3 skills", str(stop.exception))


class LocalTest(unittest.TestCase):
    def test_training_on_this_mac_uploads_nothing_and_saves_where_it_says(self) -> None:
        settings = mock.Mock(outputs_dir=Path("/outputs"), run=mock.Mock(device="mps"))
        task = mock.Mock()
        task.name = "block-container"
        args, output = cli.local_train_args(settings, task, "act", Path("/data/train"))
        self.assertIn("--policy.push_to_hub=false", args, "LeRobot pushes the model by default")
        self.assertIn(f"--output_dir={output}", args)
        self.assertEqual(output.parent, Path("/outputs/train"))
        self.assertTrue(output.name.startswith("act_so101_block-container_"))

    def test_a_local_checkpoint_is_the_last_model_under_the_folder_train_printed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "act_so101_block-container_20261113-120000"
            with self.assertRaises(SystemExit) as stop:
                local_checkpoint(str(output), "act")
            self.assertIn("no model there", str(stop.exception))

            model = output / "checkpoints" / "last" / "pretrained_model"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps({"type": "act"}))
            self.assertEqual(local_checkpoint(str(output), "act"), model.resolve())
            self.assertEqual(local_checkpoint(str(model), "act"), model.resolve())
            with self.assertRaises(SystemExit) as stop:
                local_checkpoint(str(output), "pi05")
            self.assertIn("holds a model of type act, not pi05", str(stop.exception))


class PinTest(unittest.TestCase):
    """`train --skills` records which build the model trains on as the dataset.revision in its
    train_config.json, which `run --skills` and `serve --skills` check (`commands.policy.skills_build`)."""

    COMMIT = "0123456789abcdef0123456789abcdef01234567"

    def train(self, target: str | None, episodes: list[dict] | None = None, stderr: list | None = None,
              recorded: int = 1) -> tuple[list[str], Path, mock.MagicMock]:
        """`so101 train test --skills` with nothing uploaded or run: lerobot-train's arguments, the build's
        build.json and the upload. `episodes` are the build's entries (10 demos of one skill by default);
        `stderr` collects what it warns; `recorded` is the task's `episodes`, the demos it aims for."""
        from test_skills import TASK

        from so_101 import data

        task = dataclasses.replace(TASK, episodes=recorded)

        temp = tempfile.TemporaryDirectory(prefix="so101-pin-")
        self.addCleanup(temp.cleanup)
        # outputs/ beside a so101.toml in the temporary folder: a local run keeps its build there.
        s = self.s = dataclasses.replace(settings.load(), data_dir=Path(temp.name) / "data",
                                         path=Path(temp.name) / "so101.toml")
        manifest = data.build_manifest_path(s, task, data.SKILLS)
        manifest.parent.mkdir(parents=True)
        if episodes is None:
            episodes = [{"session": "s1", "episode": i, "frames": [0, 10], "skill": "pick"} for i in range(10)]
        manifest.write_text(json.dumps({"built": "now", "episodes": episodes}))
        err = mock.MagicMock()
        ns = argparse.Namespace(task="test", policy="act", target=target, base=None, skills=True, keep_labels=False,
                                with_=None)
        with mock.patch.object(cli, "get_task", return_value=task), \
                mock.patch.object(data, "sessions", return_value=[object()]), \
                mock.patch.object(data, "build_is_current", return_value=True), \
                mock.patch("so_101.tasks.hf_user", return_value="me"), \
                mock.patch.object(cli, "refuse_public"), \
                mock.patch.object(cli, "push_dataset", return_value=self.COMMIT) as push, \
                mock.patch.object(cli, "lerobot") as lerobot, \
                mock.patch("sys.stdout"), mock.patch("sys.stderr", err):
            cli.cmd_train(s, ns, [])
        if stderr is not None:
            stderr += [c.args[0] for c in err.write.call_args_list]
        return lerobot.call_args.args[1], manifest, push

    def parsed(self, args: list[str]):
        """The training config LeRobot builds from `args`, as the job parses them."""
        import draccus
        import lerobot.policies  # noqa: F401  registers the policy types
        from lerobot.configs.train import TrainPipelineConfig

        return draccus.parse(TrainPipelineConfig, args=args)

    def test_steps_follow_the_demos_in_the_build_not_the_demos_recorded(self) -> None:
        # Two of three recorded demos made it into the skills build, three skills each, beside a run cut.
        entries = [{"session": "s1", "episode": i, "frames": [0, 10], "skill": skill}
                   for i in (0, 2) for skill in ("pick", "place", "return")]
        entries.append({"run": "r1", "attempt": 1, "episode": 0, "frames": [0, 10], "source": "teleop", "skill": "pick"})
        warned: list[str] = []
        args, _, _ = self.train(None, entries, warned, recorded=3)
        recipe = settings.load().train["act"]
        self.assertIn(f"--steps={recipe.steps(2)}", args)
        self.assertTrue(any("2 of " in line and "demos in the build" in line for line in warned), warned)

    def test_an_empty_build_stops_before_anything_uploads(self) -> None:
        with self.assertRaises(SystemExit) as stop:
            self.train(None, [])
        self.assertIn("holds no demos", str(stop.exception))

    def test_a_hub_job_trains_from_the_commit_it_uploaded_and_records_it(self) -> None:
        args, _, push = self.train(None)
        push.assert_called_once()
        self.assertIn(f"--dataset.revision={self.COMMIT}", args)
        config = self.parsed(args)
        self.assertEqual((config.dataset.repo_id, config.dataset.revision), ("me/so101_test_skills", self.COMMIT))

    def test_a_local_run_records_the_sha256_of_the_build_json_it_trains_on(self) -> None:
        args, manifest, push = self.train("local")
        push.assert_not_called()
        pin = f"so101-build-{hashlib.sha256(manifest.read_bytes()).hexdigest()}"
        self.assertIn(f"--dataset.revision={pin}", args)
        self.assertEqual(self.parsed(args).dataset.revision, pin)

    def test_a_rebuild_keeps_a_local_skills_model_usable(self) -> None:
        # Every build writes build.json anew, with its build time, so after a rebuild (the one `train` runs on
        # a stale build included) only the copy `train` kept keeps the model usable.
        from so_101 import data
        from so_101.commands.policy import skills_build

        args, manifest, _ = self.train("local")
        trained = json.loads(manifest.read_text())
        config = self.parsed(args)
        model = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="so101-model-")))
        (model / "train_config.json").write_text(json.dumps({"dataset": {
            "repo_id": config.dataset.repo_id, "root": str(config.dataset.root), "revision": config.dataset.revision}}))
        manifest.write_text(json.dumps({"built": "later", "episodes": []}))  # a rebuild, as `build --force` writes
        self.assertEqual(skills_build(self.s, model, None, None), trained)
        self.assertEqual(data.trained_build_path(self.s, config.dataset.revision).parent,
                         self.s.outputs_dir / "train" / "builds")

    def test_the_upload_is_tagged_and_its_commit_returned(self) -> None:
        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(private=True)
        api.upload_folder.return_value = mock.Mock(oid=self.COMMIT)
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            self.assertEqual(cli.push_dataset(Path("/data/train"), "me/so101_test_skills"), self.COMMIT)
        # LeRobot loads the dataset at its format tag by default: it marks this very upload.
        api.create_tag.assert_called_once_with("me/so101_test_skills", tag=mock.ANY, revision=self.COMMIT,
                                               repo_type="dataset")


class ForwardedTest(unittest.TestCase):
    def test_a_forwarded_flag_is_read_as_lerobot_reads_it_the_last_one_winning(self) -> None:
        extra = ["--policy.repo_id=a/one", "--steps=10", "--policy.repo_id", "b/two"]
        self.assertEqual(cli.forwarded(extra, "policy.repo_id"), "b/two")
        self.assertIsNone(cli.forwarded(extra, "robot.max_relative_target"))

    def test_a_forwarded_negative_step_cap_stops_the_command_before_it_runs(self) -> None:
        # LeRobot takes the last of a repeated flag, so it would replace so101.toml's checked cap.
        # `cli.main` would filter stderr through a pipe for the rest of the process, hiding later tests' output:
        # this test checks the step cap, not the filter.
        with mock.patch.object(cli, "filter_stderr"), \
                mock.patch.object(cli, "cmd_teleop") as teleop, \
                mock.patch.object(cli.sys, "argv", ["so101", "teleop", "--robot.max_relative_target=-10"]), \
                self.assertRaises(SystemExit) as stop:
            cli.main()
        teleop.assert_not_called()
        self.assertIn("--robot.max_relative_target is '-10'", str(stop.exception))
        self.assertEqual(settings.step_cap("5", "--robot.max_relative_target"), 5.0)


class PrivateTest(unittest.TestCase):
    def test_nothing_is_uploaded_to_a_public_repo(self) -> None:
        from huggingface_hub.errors import RepositoryNotFoundError

        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(private=False)
        with self.assertRaises(SystemExit) as stop:
            cli.refuse_public(api, "gpahal/act_so101_block-container", "model", "train")
        self.assertIn("is public, so nothing was uploaded", str(stop.exception))

        api.repo_info.return_value = mock.Mock(private=True)
        cli.refuse_public(api, "gpahal/act_so101_block-container", "model", "train")
        api.repo_info.side_effect = RepositoryNotFoundError("missing", response=mock.Mock(status_code=404))
        cli.refuse_public(api, "gpahal/act_so101_block-container", "model", "train")  # LeRobot makes it private

    def test_the_built_dataset_never_goes_to_a_public_repo(self) -> None:
        api = mock.Mock()
        api.repo_info.return_value = mock.Mock(private=False)
        with mock.patch("huggingface_hub.HfApi", return_value=api), self.assertRaises(SystemExit):
            cli.push_dataset(Path("/data/train"), "gpahal/so101_block-container")
        api.upload_folder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
