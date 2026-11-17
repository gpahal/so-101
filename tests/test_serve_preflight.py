"""`serve`'s checks of the rest pose and `[control]` settings, on a so101.toml in a temporary folder.

`settings.load()` reads every file below without complaint: only `check.check_control`, which
`serve`'s preflight runs, finds the problems. `serve` is called as the CLI calls it, with the checks
that would open a serial port or ask the Hugging Face Hub stubbed out.
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fakes import MOTORS, calibration

from so_101 import settings
from so_101.check import check_control
from so_101.commands import serve
from so_101.report import Report

ARMS = """
[follower]
id = "follower"

[leader]
id = "leader"
"""
POSE = {motor: 2048 for motor in MOTORS}


def pose_line(pose: dict) -> str:
    return "rest_pose = { " + ", ".join(f"{motor} = {steps}" for motor, steps in pose.items()) + " }"


class ConfigTest(unittest.TestCase):
    """A so101.toml and the follower's calibration file in a temporary folder."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="so101-config-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / "so101.toml"
        calibration_file = self.root / "calibration" / "follower" / "follower.json"
        calibration_file.parent.mkdir(parents=True)
        calibration_file.write_text(json.dumps(calibration()))
        # This file only: no SO101_* override from the shell running the tests.
        env = {key: value for key, value in os.environ.items() if not key.startswith("SO101_")}
        patch = mock.patch.dict(os.environ, {**env, "SO101_CONFIG": str(self.config)}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def fingerprint(self) -> str:
        self.config.write_text(ARMS)
        return settings.load().follower.calibration_fingerprint()

    def write(self, control: str) -> settings.Settings:
        self.config.write_text(ARMS + "\n[control]\n" + control)
        return settings.load()  # never raises on what [control] holds

    def valid_pose(self) -> str:
        return f'{pose_line(POSE)}\nrest_pose_calibration = "{self.fingerprint()}"\n'


class ServePreflightTest(ConfigTest):
    """`so101 serve`'s preflight runs the `[control]` checks as FAILs and exits on one, and the
    control process exits on the same settings."""

    def problems(self, s: settings.Settings) -> list[str]:
        report = Report()
        check_control(report, s, "FAIL")
        return [f"{r.group}: {r.detail}" for r in report.results if r.status == "FAIL"]

    def serve(self, s: settings.Settings, supervisor: str | None = None, task_name: str = "test") -> None:
        """Run `serve` on `task_name`, with `--supervisor` when given, up to where it would download the
        policy, with the checks that open a device or ask the Hub stubbed out. Raises SystemExit when the
        preflight fails, and AssertionError when it passes."""

        def download_policy(revision=None, skills=False, *, folder):
            raise AssertionError("serve passed its preflight")

        task = types.SimpleNamespace(name=task_name, prompt="test", cameras=(), fps=30,
                                     policy_config=lambda revision, policy, skills=False: {"chunk_size": 10, "input_features": {}},
                                     policy_for=lambda policy, skills=False: "x/y",
                                     policy_commit=lambda revision, policy, skills=False: "c" * 40,
                                     download_policy=download_policy)
        ns = argparse.Namespace(task="test", policy="act", remote=False, revision=None, checkpoint=None, action_steps=None,
                                operator=None, supervisor=supervisor)
        out = io.StringIO()
        with mock.patch.object(serve, "policy_task", return_value=task), \
                mock.patch("so_101.check.check_system"), \
                mock.patch("so_101.check.check_arm", return_value=None), \
                contextlib.redirect_stdout(out):
            try:
                serve.cmd_serve(s, ns, [])
            finally:
                self.printed = out.getvalue()

    def test_serve_and_the_control_process_refuse_a_bad_rest_pose_or_control_setting(self) -> None:
        from so_101.control.main import check_settings

        check_settings(self.write(self.valid_pose()))
        with self.assertRaises(SystemExit) as exited:
            check_settings(self.write(self.valid_pose() + 'on_link_loss = "stop"\n'))
        self.assertIn("on_link_loss", str(exited.exception.code))
        with self.assertRaises(SystemExit) as exited:
            check_settings(self.write("rewind_s = 2.5\n"))
        self.assertIn("no rest_pose", str(exited.exception.code))

        cases = {
            "missing": ("", "no rest_pose"),
            "fingerprint from another calibration": (f'{pose_line(POSE)}\nrest_pose_calibration = "0123456789ab"\n',
                                                     "recalibrated"),
            "no fingerprint": (pose_line(POSE) + "\n", "no rest_pose_calibration"),
            "partial pose": ("rest_pose = { gripper = 2048 }\n", "it leaves out shoulder_pan"),
        }
        for case, (control, why) in cases.items():
            with self.subTest(case):
                problems = self.problems(self.write(control))
                self.assertEqual(len(problems), 1)
                self.assertIn(why, problems[0])
                with self.assertRaises(SystemExit) as exited:
                    check_settings(self.write(control))
                self.assertIn(why, str(exited.exception.code))

        with self.assertRaises(SystemExit) as stopped:
            self.serve(self.write(self.valid_pose() + "link_timeout_s = nan\n"))
        self.assertIn("1 FAIL", str(stopped.exception.code))
        [line] = [line for line in self.printed.splitlines() if line.strip().startswith("FAIL")]
        self.assertIn("link_timeout_s", line)

    def test_run_s_duration_0_starts_and_lasts_until_a_person_ends_the_attempt(self) -> None:
        import math

        from so_101.control import main

        s = self.write(self.valid_pose())
        # The real check and `run_settings`, in the order `main.control` runs them.
        ran = main.run_settings(s, {"manifest": {"autopilot": {"duration_s": 0}}}, "test")
        self.assertEqual(ran.control.attempt_limit_s, math.inf)
        self.assertEqual(main.run_settings(s, {"manifest": {"autopilot": {"duration_s": 12.5}}}).control.attempt_limit_s,
                         12.5)
        for duration in (math.inf, math.nan, -1, True, "10"):
            with self.subTest(duration=duration), self.assertRaises(SystemExit) as exited:
                main.run_settings(s, {"manifest": {"autopilot": {"duration_s": duration}}})
            self.assertIn("--duration", str(exited.exception.code))
        with self.assertRaises(SystemExit) as exited:  # the file's own settings are still checked
            main.run_settings(self.write(self.valid_pose() + "link_timeout_s = nan\n"),
                              {"manifest": {"autopilot": {"duration_s": 0}}})
        self.assertIn("link_timeout_s", str(exited.exception.code))

    def test_the_supervisor_runs_only_on_the_task_its_prompt_describes(self) -> None:
        from so_101.control import main
        from so_101.vision.prompt import TASK_NAME

        pose = self.valid_pose()
        for mode in ("shadow", "act"):
            with self.subTest(mode=mode):
                # --supervisor sets the mode the preflight checks, over [supervisor] mode's off.
                with self.assertRaises(SystemExit) as stopped:
                    self.serve(self.write(pose), supervisor=mode)
                self.assertIn("1 FAIL", str(stopped.exception.code))
                [line] = [line for line in self.printed.splitlines() if line.strip().startswith("FAIL")]
                self.assertIn("can't supervise test", line)
                self.assertIn("--supervisor off", self.printed)
                with self.assertRaises(SystemExit):
                    self.serve(self.write(pose + f'\n[supervisor]\nmode = "{mode}"\n'))
                # so101-control re-reads the file and refuses too, with the mode serve passed it.
                with self.assertRaises(SystemExit) as exited:
                    main.run_settings(self.write(pose), {"manifest": {}, "supervisor": mode}, "test")
                self.assertIn("--supervisor off", str(exited.exception.code))
                # `so101 check` warns.
                report = Report()
                check_control(report, self.write(pose + f'\n[supervisor]\nmode = "{mode}"\n'), "WARN", "test")
                [warning] = [r for r in report.warnings if r.group == "supervisor"]
                self.assertIn("can't supervise test", warning.detail)
                self.assertEqual(report.failures, [])
        with self.assertRaisesRegex(AssertionError, "passed its preflight"):  # off for this run: it starts
            self.serve(self.write(pose + '\n[supervisor]\nmode = "act"\n'), supervisor="off")
        with self.assertRaisesRegex(AssertionError, "passed its preflight"):
            self.serve(self.write(pose), supervisor="shadow", task_name=TASK_NAME)
        main.run_settings(self.write(pose), {"manifest": {}, "supervisor": "act"}, TASK_NAME)


if __name__ == "__main__":
    unittest.main()
