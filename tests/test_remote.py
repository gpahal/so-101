"""The cloud GPU's server config, client arguments and server matching, checked against LeRobot's own
config classes, without a server: how each policy's action chunks are joined is set in both places,
and the two must agree."""

import argparse
import contextlib
import io
import types
import unittest
from unittest import mock

import draccus

from so_101 import remote
from so_101.commands import policy

SIZES = {"top": (640, 480), "wrist": (640, 480)}


def server(labels: dict, stage: str = "RUNNING"):
    return types.SimpleNamespace(id="job", flavor="l4x1", labels=labels, status=types.SimpleNamespace(stage=stage))


def parse_server(policy_type: str, action_steps: int):
    from lerobot.remote_inference.configs import ServerConfig

    return draccus.decode(ServerConfig, remote.server_config("x/y", policy_type, "main", SIZES, action_steps, 30))


def parse_client(policy_type: str):
    from lerobot.inference.factory import RemoteInferenceConfig

    args = [arg.replace("--inference.", "--", 1) for arg in remote.client_args(policy_type)
            if not arg.startswith("--inference.type=")]
    return draccus.parse(RemoteInferenceConfig, args=args)


class ExecutionTest(unittest.TestCase):
    def test_pi05_server_and_client_both_use_guided_real_time_chunking(self) -> None:
        execution = parse_server("pi05", 50).execution
        self.assertEqual(execution.supported_modes, ["rtc_guided"])
        self.assertTrue(execution.rtc.enabled)
        self.assertEqual(execution.rtc.mode, "guided")
        # Steered over the 1 s the arm can play while a request is out, not LeRobot's default 10 steps.
        self.assertEqual(execution.rtc.execution_horizon, 30)
        self.assertIsNone(execution.n_action_steps)  # LeRobot refuses a fixed slice with RTC
        client = parse_client("pi05")
        self.assertEqual((client.mode, client.chunk_merge), ("rtc_guided", "append"))
        # Asks with 1 s left, all of it steered, and holds up to 2 s for a late chunk before failing.
        self.assertEqual((client.refill_seconds, client.action_starvation_grace_s), (1.0, 2.0))

    def test_the_steered_horizon_never_exceeds_what_the_model_predicts(self) -> None:
        self.assertEqual(parse_server("pi05", 20).execution.rtc.execution_horizon, 20)

    def test_act_keeps_plain_aligned_chunks_of_its_action_steps(self) -> None:
        execution = parse_server("act", 100).execution
        self.assertEqual((execution.supported_modes, execution.n_action_steps), (["chunk"], 100))
        client = parse_client("act")
        self.assertEqual((client.mode, client.chunk_merge), ("chunk", "aligned"))
        self.assertEqual((client.refill_seconds, client.action_starvation_grace_s), (0.5, 1.0))  # LeRobot's

    def test_pi05_on_the_cloud_gpu_takes_no_action_steps(self) -> None:
        with self.assertRaises(SystemExit) as exited:
            remote.refuse_action_steps("pi05", 30)
        self.assertIn("real-time chunking", str(exited.exception.code))
        remote.refuse_action_steps("pi05", None)
        remote.refuse_action_steps("act", 30)


class RequireServerTest(unittest.TestCase):
    def require(self, labels: dict | None, policy_type: str = "pi05", action_steps: int = 50):
        found = None if labels is None else server(labels)
        with mock.patch.object(remote, "active_server", return_value=found):
            return remote.require_server("task", policy_type, "main", action_steps)

    def test_a_server_started_with_real_time_chunking_is_accepted(self) -> None:
        labels = {"so101": remote.SERVER_LABEL, **remote._labels("task", "pi05", "main", 50)}
        self.assertEqual(self.require(labels).id, "job")

    def test_a_server_from_before_real_time_chunking_is_refused_with_the_start_command(self) -> None:
        old = {"task": "task", "policy": "pi05", "revision": "main", "action_steps": "50"}
        with self.assertRaises(SystemExit) as exited:
            self.require(old)
        message = str(exited.exception.code)
        self.assertIn("plain chunks from before real-time chunking", message)
        self.assertIn("uv run so101 server start task --policy pi05 --revision main", message)
        self.assertNotIn("--action-steps", message)

    def test_an_act_server_is_still_named_by_its_action_steps(self) -> None:
        with self.assertRaises(SystemExit) as exited:
            self.require(None, "act", 100)
        self.assertIn("--action-steps 100", str(exited.exception.code))


class StepsLineTest(unittest.TestCase):
    def setup(self, policy_type: str | None, action_steps: int) -> policy.PolicySetup:
        task = types.SimpleNamespace(fps=30)
        job = server(remote._labels("task", policy_type, "main", action_steps)) if policy_type else None
        return policy.PolicySetup(task, action_steps, job, {}, (), [], None, None, [])

    def line(self, policy_type: str | None, action_steps: int) -> str:
        s = types.SimpleNamespace(teleop=types.SimpleNamespace(max_relative_target=10.0))
        return self.setup(policy_type, action_steps).steps_line(s)

    def test_each_way_of_running_says_when_it_predicts_again(self) -> None:
        self.assertEqual(self.line(None, 100), "Predicts every 100 steps (3.3 s), limit 10 per step.")
        self.assertIn("Predicts 100 steps (3.3 s) at a time and asks for the next once the turnaround",
                      self.line("act", 100))
        rtc = self.line("pi05", 50)
        self.assertIn("Predicts 1.7 s at a time with real-time chunking", rtc)
        self.assertNotIn("every", rtc)


class PolicySetupTest(unittest.TestCase):
    def test_run_and_serve_refuse_action_steps_for_pi05_before_asking_for_a_server(self) -> None:
        ns = argparse.Namespace(task="task", policy="pi05", remote=True, revision=None, checkpoint=None,
                                action_steps=30)
        task = types.SimpleNamespace(name="task", policy_config=lambda revision, policy: {"chunk_size": 50})
        with mock.patch.object(remote, "active_server") as active, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                policy.policy_setup(types.SimpleNamespace(), ns, task)
        self.assertIn("takes no --action-steps", str(exited.exception.code))
        active.assert_not_called()


if __name__ == "__main__":
    unittest.main()
