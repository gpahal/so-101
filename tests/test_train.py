"""`so101 train --base`: which model a training job starts from, checked before anything uploads or bills."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from so_101 import cli
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


if __name__ == "__main__":
    unittest.main()
