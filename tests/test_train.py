"""`so101 train --base`: which model a training job starts from, checked before anything uploads or bills."""

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
        self.assertEqual(flags[2:], cli.POLICY_FLAGS["pi05"][1:])

    def test_a_base_without_a_revision_starts_from_main(self) -> None:
        self.assertEqual(cli.policy_flags("act", cli.parse_base("gpahal/act_so101_block-container")),
                         ["--policy.pretrained_path=gpahal/act_so101_block-container"])

    def test_without_a_base_each_policy_keeps_its_default(self) -> None:
        self.assertEqual(cli.policy_flags("act", None), [])
        self.assertEqual(cli.policy_flags("pi05", None), cli.POLICY_FLAGS["pi05"])
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


class ForwardedTest(unittest.TestCase):
    def test_a_forwarded_flag_is_read_as_lerobot_reads_it_the_last_one_winning(self) -> None:
        extra = ["--policy.repo_id=a/one", "--steps=10", "--policy.repo_id", "b/two"]
        self.assertEqual(cli.forwarded(extra, "policy.repo_id"), "b/two")
        self.assertIsNone(cli.forwarded(extra, "robot.max_relative_target"))

    def test_a_forwarded_negative_step_cap_stops_the_command_before_it_runs(self) -> None:
        # LeRobot takes the last of a repeated flag, so it would replace so101.toml's checked cap.
        with mock.patch.object(cli, "cmd_teleop") as teleop, \
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
