"""`so101`'s command line: which arguments each command takes, what `run` and `serve` forward to
LeRobot, what they hand `so101-control`, and how `sim eval` builds `lerobot-eval`'s arguments.

Nothing here opens a device or asks the Hub: the parser is built and parsed, and the commands run
with their checks and child processes stubbed out.
"""

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from test_serve_preflight import ARMS, ConfigTest

from so_101 import cli, settings
from so_101.commands import serve
from so_101.commands import sim as sim_command
from so_101.control.main import RUN_ENV


def parse(*argv: str) -> tuple[argparse.Namespace, list[str]]:
    """Parse `argv` as `so101 ...` does, refusing extra arguments for a command that forwards none."""
    return cli.parse_args(list(argv))


def refused(*argv: str) -> str:
    """The error `so101 ...` stops with on `argv`, from the parser."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err), unittest.TestCase().assertRaises(SystemExit) as stop:
        parse(*argv)
    assert stop.exception.code == 2, stop.exception.code
    return err.getvalue()


class HelpTest(unittest.TestCase):
    def test_every_command_and_action_prints_its_help(self) -> None:
        for argv in (["run"], ["serve"], ["sim"], ["server"],
                     *(["sim", action] for action in sim_command.ACTIONS),
                     *(["server", action] for action in ("start", "stop", "status"))):
            with self.subTest(argv):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stop:
                    parse(*argv, "--help")
                self.assertEqual(stop.exception.code, 0)
                self.assertIn("usage: so101 " + " ".join(argv), out.getvalue())


class SubcommandTest(unittest.TestCase):
    """`sim` and `server` give each action only its own options."""

    def test_each_sim_action_takes_only_its_own_options(self) -> None:
        ns, extra = parse("sim", "build", "block-container", "--refit-cameras")
        self.assertEqual((ns.action, ns.task, ns.refit_cameras, ns.refit_motors, extra),
                         ("build", "block-container", True, False, []))
        ns, _ = parse("sim", "replay", "--episodes", "3", "--limit", "5", "--recording", "x")
        self.assertEqual((ns.episodes, ns.limit, ns.recording), (3, 5.0, "x"))
        ns, _ = parse("sim", "check", "--checkpoint", "dir")
        self.assertEqual((ns.policy, ns.checkpoint, ns.revision), ("act", "dir", None))
        for argv, wrong in ((("sim", "score", "--episodes", "3"), "--episodes"),
                            (("sim", "build", "--policy", "act"), "--policy"),
                            (("sim", "replay", "--refit-cameras"), "--refit-cameras"),
                            (("sim", "aim", "--limit", "5"), "--limit"),
                            (("sim", "physics", "--duration", "5"), "--duration"),
                            (("sim", "check", "--episodes", "3"), "--episodes")):
            with self.subTest(argv):
                self.assertIn(wrong, refused(*argv))

    def test_only_nudge_and_eval_forward_to_lerobot(self) -> None:
        ns, extra = parse("sim", "eval", "--duration", "20", "--eval.use_async_envs=false")
        self.assertEqual((ns.duration, extra), (20.0, ["--eval.use_async_envs=false"]))
        ns, extra = parse("sim", "nudge", "--episodes", "2", "--dataset.video=false")
        self.assertEqual((ns.episodes, extra), (2, ["--dataset.video=false"]))
        for action in ("build", "replay", "score", "physics", "check", "aim"):
            with self.subTest(action):
                self.assertIn(f"so101 sim {action}` passes nothing on to LeRobot",
                              refused("sim", action, "--eval.n_episodes=3"))

    def test_server_stop_and_status_take_no_options(self) -> None:
        ns, _ = parse("server", "start", "block-container", "--policy", "pi05", "--skills", "--timeout", "1h")
        self.assertEqual((ns.action, ns.task, ns.policy, ns.skills, ns.timeout),
                         ("start", "block-container", "pi05", True, "1h"))
        for action in ("stop", "status"):
            ns, extra = parse("server", action)
            self.assertEqual((ns.action, extra), (action, []))
            for wrong in (["--policy", "act"], ["block-container"], ["--skills"]):
                with self.subTest(action=action, wrong=wrong):
                    refused("server", action, *wrong)


class SupervisorFlagTest(unittest.TestCase):
    def test_run_and_serve_take_a_supervisor_mode(self) -> None:
        for command in ("run", "serve"):
            with self.subTest(command):
                self.assertIsNone(parse(command)[0].supervisor)
                for mode in settings.SUPERVISOR_MODES:
                    self.assertEqual(parse(command, "--supervisor", mode)[0].supervisor, mode)
                self.assertIn("invalid choice", refused(command, "--supervisor", "loud"))


class AllowListTest(unittest.TestCase):
    """`run` and `serve` forward only `--inference.*`, `--policy.*`, `--robot.*` and `--rename_map`."""

    def allowed(self, extra: list[str], mode: str = "serve") -> list[str]:
        return serve.lerobot_extra(extra, mode, fps=30)

    def refusal(self, extra: list[str], mode: str = "serve") -> str:
        with self.assertRaises(SystemExit) as stop:
            self.allowed(extra, mode)
        return str(stop.exception.code)

    def test_the_allowed_families_pass_through_unchanged(self) -> None:
        extra = ["--inference.jpeg_quality=50", "--policy.device", "cpu", "--robot.max_relative_target=5",
                 '--rename_map={"observation.images.top": "observation.images.front"}', "--rename_map", "{}"]
        for mode in ("run", "serve"):
            self.assertEqual(self.allowed(extra, mode), extra)
        self.assertEqual(self.allowed([]), [])

    def test_anything_else_is_refused_by_name(self) -> None:
        message = self.refusal(["--display_data=true", "--inference.jpeg_quality=50", "--strategy.type", "dagger"])
        self.assertIn("--display_data", message)
        self.assertIn("--strategy.type", message)
        self.assertNotIn("jpeg_quality", message)
        self.assertIn("--inference.*, --policy.*, --robot.* and --rename_map", message)
        self.assertIn("serve", message)
        self.assertIn("so101 run", self.refusal(["--duration=5"], "run"))
        self.assertIn("stray", self.refusal(["stray"]))
        self.assertIn("--rename_mapping", self.refusal(["--rename_mapping={}"]))
        self.assertIn("--robotics", self.refusal(["--robotics=1"]))

    def test_fps_and_interpolation_get_their_own_reason(self) -> None:
        for extra in (["--fps=15"], ["--fps", "15"]):
            message = self.refusal(extra)
            self.assertIn("tasks.toml", message)
            self.assertIn("30", message)
        message = self.refusal(["--interpolation_multiplier=2"])
        self.assertIn("--interpolation_multiplier", message)
        self.assertIn("once per frame", message)

    def test_serve_refuses_before_its_checks(self) -> None:
        task = types.SimpleNamespace(name="test", prompt="test", fps=30, reset_time_s=10)
        ns = argparse.Namespace(task="test", episodes=None, duration=None, reset_time=None, supervisor=None)
        s = mock.Mock(run=mock.Mock(episodes=1), control=mock.Mock(on_link_loss="pause"))
        with mock.patch.object(serve, "policy_task", return_value=task), \
                mock.patch.object(serve, "policy_setup", side_effect=AssertionError("checked first")):
            for command in (serve.cmd_serve, serve.cmd_run):
                with self.subTest(command.__name__), self.assertRaises(SystemExit) as stop:
                    command(s, ns, ["--display_data=true"])
                self.assertIn("--display_data", str(stop.exception.code))

    def test_run_refuses_a_reset_time_or_duration_that_is_not_finite_seconds_before_its_checks(self) -> None:
        # A nan or inf reset time made the autopilot's next_in_s nan or inf, which the page's strict JSON
        # status can't hold: every /status failed once the arm reached rest.
        task = types.SimpleNamespace(name="test", prompt="test", fps=30, reset_time_s=10)
        s = mock.Mock(run=mock.Mock(episodes=1), control=mock.Mock(on_link_loss="pause"))
        cases = [("reset_time", "--reset-time", value) for value in (float("nan"), float("inf"), -1.0)] + \
                [("duration", "--duration", value) for value in (float("nan"), float("inf"), float("-inf"), -1.0)]
        for key, flag, value in cases:
            ns = argparse.Namespace(task="test", episodes=None, duration=None, reset_time=None, supervisor=None)
            setattr(ns, key, value)
            with self.subTest(flag=flag, value=value), \
                    mock.patch.object(serve, "policy_task", return_value=task), \
                    mock.patch.object(serve, "policy_setup", side_effect=AssertionError("checked first")), \
                    self.assertRaises(SystemExit) as stop:
                serve.cmd_run(s, ns, [])
            self.assertIn(flag, str(stop.exception.code))

    def test_run_refuses_on_link_loss_continue_before_its_checks(self) -> None:
        # With "continue", an attempt the autopilot started would keep driving after the page is lost.
        task = types.SimpleNamespace(name="test", prompt="test", fps=30, reset_time_s=10)
        ns = argparse.Namespace(task="test", episodes=None, duration=None, reset_time=None, supervisor=None)
        s = mock.Mock(run=mock.Mock(episodes=1), control=mock.Mock(on_link_loss="continue"), path="so101.toml")
        with mock.patch.object(serve, "policy_task", return_value=task), \
                mock.patch.object(serve, "policy_setup", side_effect=AssertionError("checked first")), \
                self.assertRaises(SystemExit) as stop:
            serve.cmd_run(s, ns, [])
        self.assertIn("on_link_loss", str(stop.exception.code))


class LaunchTest(ConfigTest):
    """What `run` and `serve` hand `so101-control` and write in run.json's manifest."""

    def launch(self, command: str, s: settings.Settings, skills=(), **flags) -> dict:
        task = types.SimpleNamespace(name="test", prompt="Put it in", fps=30, reset_time_s=10, skills=skills,
                                     policy_for=lambda policy, skills=False: "you/act_so101_test")
        ns = argparse.Namespace(task=None, policy="act", remote=False, revision=None, checkpoint=None,
                                action_steps=None, operator=None, skills=False, episodes=None, duration=None,
                                reset_time=None, supervisor=None)
        for key, value in flags.items():
            setattr(ns, "skills" if key == "skills_flag" else key, value)
        self.checked: list[settings.Settings] = []
        self.ran: list[list[str]] = []

        def policy_setup(s, ns, task):
            self.checked.append(s)
            return types.SimpleNamespace(server=None, checkpoint=None, action_steps=10, arms=("follower", "leader"),
                                         found=[], report=None, sizes={}, args=["--robot.type=so101_follower"],
                                         gripper_sha256="a1b2" if ns.skills else None, commit="c" * 40,
                                         steps_line=lambda s: "", run=lambda command, args, *_: self.ran.append(args))

        get_task = mock.Mock(return_value=task)
        err = io.StringIO()
        with mock.patch("so_101.commands.policy.get_task", get_task), \
                mock.patch.object(serve, "policy_setup", policy_setup), \
                mock.patch.object(serve.data, "new_run", return_value=types.SimpleNamespace(path=self.root / "run")), \
                mock.patch("so_101.recording.run_manifest", side_effect=lambda *args, policy, **kw: {"policy": policy}), \
                mock.patch.object(serve.signal, "signal"), \
                mock.patch.dict(os.environ), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            getattr(serve, f"cmd_{command}")(s, ns, [])
            sent = json.loads(os.environ[RUN_ENV])
        self.get_task_calls = get_task.call_count
        return sent

    def test_run_takes_a_duration_and_reset_time_of_0(self) -> None:
        self.config.write_text(ARMS)
        sent = self.launch("run", settings.load(), duration=0.0, reset_time=0.0)
        self.assertEqual(sent["manifest"]["autopilot"]["duration_s"], 0.0, "0 runs until you end it")
        self.assertEqual(sent["manifest"]["autopilot"]["reset_s"], 0.0)
        self.assertEqual(sent["manifest"]["policy"]["commit"], "c" * 40, "run.json records the commit checked")

    def test_a_skills_run_records_the_skills_and_gripper_thresholds_its_model_was_checked_against(self) -> None:
        # so101-control checks them again after it reloads the files (`control.main.require_run_skills`).
        self.config.write_text(ARMS)
        skill = types.SimpleNamespace(record=lambda: {"name": "pick", "prompt": "Pick it up"})
        for command in ("run", "serve"):
            with self.subTest(command):
                policy = self.launch(command, settings.load(), skills=(skill,), skills_flag=True)["manifest"]["policy"]
                self.assertEqual((policy["skills"], policy["gripper_sha256"]), ([skill.record()], "a1b2"))
                policy = self.launch(command, settings.load())["manifest"]["policy"]
                self.assertEqual((policy["skills"], policy["gripper_sha256"]), (None, None))

    def test_the_supervisor_flag_overrides_the_file_for_the_checks_and_the_control_process(self) -> None:
        self.config.write_text(ARMS + '\n[supervisor]\nmode = "shadow"\n')
        s = settings.load()
        for command in ("run", "serve"):
            with self.subTest(command):
                sent = self.launch(command, s, supervisor="act")
                self.assertEqual(sent["supervisor"], "act")
                [checked] = self.checked
                self.assertEqual(checked.supervisor, dataclasses.replace(s.supervisor, mode="act"),
                                 "serve's preflight checks the mode the run uses")

                sent = self.launch(command, s)
                self.assertNotIn("supervisor", sent, "without the flag so101-control reads so101.toml")
                self.assertIs(self.checked[0], s)

    def test_the_task_is_looked_up_once(self) -> None:
        self.config.write_text(ARMS)
        for command in ("run", "serve"):
            with self.subTest(command):
                self.launch(command, settings.load())
                self.assertEqual(self.get_task_calls, 1)


class SimEvalTest(ConfigTest):
    """`sim eval` builds lerobot-eval's arguments, with no MuJoCo or model needed."""

    def eval(self, *argv: str) -> list[str]:
        ns, extra = parse("sim", "eval", *argv)
        task = types.SimpleNamespace(name="block-container", prompt="Put it in", fps=30, cameras=("top", "wrist"))
        policy = self.root / "model"
        policy.mkdir(exist_ok=True)
        (policy / "config.json").write_text(json.dumps({"chunk_size": 100, "n_action_steps": 100}))
        (self.root / "sims" / "scene.json").parent.mkdir(exist_ok=True)
        (self.root / "sims" / "scene.json").write_text("{}")
        stand_ins = {"so_101.sim": types.ModuleType("so_101.sim"),
                     "so_101.sim.build": types.ModuleType("so_101.sim.build"),
                     "so_101.sim.source": types.SimpleNamespace(ensure_demos=lambda demos, scene: None)}
        run_lerobot = mock.Mock(return_value=0)
        with mock.patch.dict(sys.modules, stand_ins), \
                mock.patch.object(sim_command, "get_task", return_value=task), \
                mock.patch.object(sim_command, "_policy", return_value=policy), \
                mock.patch.object(sim_command.data, "sim_dir", return_value=self.root / "sims"), \
                mock.patch.object(sim_command.data, "dataset_root", return_value=self.root / "demos"), \
                mock.patch.object(sim_command, "run_lerobot", run_lerobot), \
                mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            try:
                sim_command.cmd_sim(settings.load(), ns, extra)
            except SystemExit as stop:
                if stop.code != 0:
                    raise
            else:
                self.fail("sim eval exits with lerobot-eval's status")
        [(command, args), _] = run_lerobot.call_args
        self.assertEqual(command, "lerobot-eval")
        return args

    def env_kwargs(self, args: list[str]) -> dict:
        [kwargs] = [a.removeprefix("--env.make_env_kwargs=") for a in args if a.startswith("--env.make_env_kwargs=")]
        return json.loads(kwargs)

    def test_eval_lasts_the_attempt_limit_or_its_duration(self) -> None:
        self.config.write_text(ARMS + "\n[teleop]\nmax_relative_target = 8\n\n[control]\nattempt_limit_s = 40\n")
        args = self.eval()
        self.assertEqual(self.env_kwargs(args), {"max_relative_target": 8.0, "episode_length": 1200})
        self.assertIn("--eval.n_episodes=10", args)
        self.assertIn("--policy.n_action_steps=100", args)

        args = self.eval("--duration", "20", "--episodes", "4", "--limit", "5", "--action-steps", "30",
                         "--eval.use_async_envs=false")
        self.assertEqual(self.env_kwargs(args), {"max_relative_target": 5.0, "episode_length": 600})
        self.assertIn("--eval.n_episodes=4", args)
        self.assertIn("--policy.n_action_steps=30", args)
        self.assertEqual(args[-1], "--eval.use_async_envs=false")

    def test_eval_refuses_a_duration_that_is_not_a_positive_number(self) -> None:
        self.config.write_text(ARMS)
        for value in ("0", "-5", "nan", "inf"):
            with self.subTest(value), self.assertRaises(SystemExit) as stop:
                self.eval("--duration", value)
            self.assertIn("--duration", str(stop.exception.code))


if __name__ == "__main__":
    unittest.main()
