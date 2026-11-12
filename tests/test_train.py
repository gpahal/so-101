"""`so101 train --base`: which model a training job starts from, checked before anything uploads or bills."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from so_101 import cli
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


if __name__ == "__main__":
    unittest.main()
