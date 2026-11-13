"""The cloud GPU's server config, client arguments and server matching, checked against LeRobot's own
config classes, without a server: how each policy's action chunks are joined is set in both places,
and the two must agree. The pi0.5 job's script runs against stand-ins for LeRobot's model and server."""

import argparse
import base64
import contextlib
import io
import sys
import types
import unittest
from unittest import mock

import draccus
import torch

from so_101 import remote, settings
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
    def require(self, labels: dict | None, policy_type: str = "pi05", action_steps: int = 50, pin: bool = True):
        found = None if labels is None else server(labels)
        with mock.patch.object(remote, "active_server", return_value=found):
            return remote.require_server("task", policy_type, "main", action_steps, pin)

    def test_a_server_started_with_real_time_chunking_is_accepted(self) -> None:
        labels = {"so101": remote.SERVER_LABEL, **remote._labels("task", "pi05", "main", 50, True)}
        self.assertEqual(self.require(labels).id, "job")

    def test_a_server_started_with_the_other_pin_setting_is_refused_and_says_which_it_runs(self) -> None:
        for started, described in ((True, "unused action columns pinned"), (False, "unused action columns not pinned")):
            with self.subTest(started=started):
                labels = remote._labels("task", "pi05", "main", 50, started)
                self.assertEqual(self.require(labels, pin=started).id, "job")
                with self.assertRaises(SystemExit) as exited:
                    self.require(labels, pin=not started)
                self.assertIn(described, str(exited.exception.code))

    def test_a_pi05_server_from_before_the_pin_is_refused_and_says_so(self) -> None:
        labels = remote._labels("task", "pi05", "main", 50, True)
        del labels[remote.PIN_LABEL]
        with self.assertRaises(SystemExit) as exited:
            self.require(labels)
        self.assertIn("started before the unused action columns setting", str(exited.exception.code))

    def test_an_act_server_has_no_pin_to_match(self) -> None:
        labels = remote._labels("task", "act", "main", 100, True)
        self.assertNotIn(remote.PIN_LABEL, labels)
        self.assertEqual(self.require(labels, "act", 100, pin=False).id, "job")
        self.assertNotIn("unused action columns", remote.describe(server(labels)))

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
        job = server(remote._labels("task", policy_type, "main", action_steps, True)) if policy_type else None
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

    def setup(self, s: types.SimpleNamespace) -> mock.Mock:
        """Run `policy_setup` for pi0.5 on the cloud GPU up to where it asks for the server."""
        ns = argparse.Namespace(task="task", policy="pi05", remote=True, revision=None, checkpoint=None,
                                action_steps=None)
        task = types.SimpleNamespace(name="task", policy_config=lambda revision, policy: {"chunk_size": 50})
        with mock.patch.object(remote, "require_server", side_effect=SystemExit("asked")) as require:
            with self.assertRaises(SystemExit) as exited:
                policy.policy_setup(s, ns, task)
        self.exited = str(exited.exception.code)
        return require

    def test_run_and_serve_ask_for_a_server_with_so101_tomls_pin_setting(self) -> None:
        for pin in (True, False):
            with self.subTest(pin=pin):
                section = settings.Remote(job_target="l4x1", job_timeout="1h", pin_unused_actions=pin)
                require = self.setup(types.SimpleNamespace(path="so101.toml", remote={"pi05": section}))
                require.assert_called_once_with("task", "pi05", "main", 50, pin)

    def test_run_and_serve_stop_without_a_remote_section_for_the_policy(self) -> None:
        require = self.setup(types.SimpleNamespace(path="so101.toml", remote={}))
        self.assertEqual(self.exited, "No [remote.pi05] section in so101.toml.")
        require.assert_not_called()


USED, COLUMNS, STEPS = 3, 8, 4


def model_class() -> type:
    """A stand-in for LeRobot's PI05Pytorch: 8 action columns of which the arm uses 3, and a 4-step
    sampler whose model moves every column by the same amount, the unused ones too."""

    class Model:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(output_features={"action": types.SimpleNamespace(shape=(USED,))})
            self.seen: list[tuple[float, torch.Tensor]] = []

        def sample_actions(self, noise: torch.Tensor) -> torch.Tensor:
            x_t = noise
            for step in range(STEPS):
                time = torch.full((noise.shape[0],), 1 - step / STEPS)
                x_t = x_t - self.denoise_step(prefix_pad_masks=None, past_key_values=None, x_t=x_t, timestep=time) / STEPS
            return x_t

        def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep) -> torch.Tensor:
            self.seen.append((float(timestep.reshape(-1)[0]), x_t.clone()))
            return torch.ones_like(x_t)

    return Model


def run_server_script(policy_type: str, pin: bool) -> tuple[type, type, list[str]]:
    """Run the script `setup_script` writes for the job, with LeRobot's pi0.5 model and server replaced:
    the stand-in model class as the script left it, the policy class, and what the script started."""
    import lerobot.configs  # noqa: F401  the script's real imports, loaded before sys.modules is patched
    import lerobot.scripts  # noqa: F401
    from lerobot.remote_inference import deployment

    setup, command = remote.setup_script(policy_type, {}, pin)
    written = {}
    for line in setup.splitlines():
        _, encoded, _, _, _, _, path = line.split()
        written[path] = base64.b64decode(encoded).decode()
    started: list[str] = []
    package = types.ModuleType("lerobot.policies.pi05")
    modeling = types.ModuleType("lerobot.policies.pi05.modeling_pi05")

    class PI05Policy:
        loaded: list = []  # (path, the precision the config asked for) of each model LeRobot's server loads

        @classmethod
        def from_pretrained(cls, path, *, config=None):
            cls.loaded.append((path, config.dtype))
            return cls()

    modeling.PI05Policy, modeling.PI05Pytorch = PI05Policy, model_class()
    serving = types.ModuleType("lerobot.scripts.lerobot_policy_server")
    serving.main = lambda: started.append("server")
    if command.startswith("python /tmp/serve.py"):
        fakes = {package.__name__: package, modeling.__name__: modeling, serving.__name__: serving}
        downloads: list[tuple] = []
        hub = lambda *args, **kwargs: downloads.append((args, kwargs)) or "/model"  # noqa: E731
        # The script replaces LeRobot's own download function: put ours under it, and LeRobot's back after.
        with mock.patch.dict(sys.modules, fakes), mock.patch.object(deployment, "snapshot_download", hub):
            exec(compile(written["/tmp/serve.py"], "serve.py", "exec"), {"__name__": "__main__"})
            modeling.PI05Policy.resolved = str(deployment.resolve_artifact("me/model", "main"))
            modeling.PI05Policy.downloads = downloads
    return modeling.PI05Pytorch, modeling.PI05Policy, started


class Pi05ServerScriptTest(unittest.TestCase):
    def test_the_script_matches_the_lerobot_methods_it_wraps(self) -> None:
        # The stand-ins above only prove the script against what it assumes. This holds the assumptions
        # to the installed LeRobot; `wait_until_ready` holds them to the job's image.
        import inspect

        from lerobot.policies.pi05 import modeling_pi05

        self.assertEqual(list(inspect.signature(modeling_pi05.PI05Pytorch.denoise_step).parameters),
                         ["self", "prefix_pad_masks", "past_key_values", "x_t", "timestep"])
        self.assertIn("self.denoise_step(", inspect.getsource(modeling_pi05.PI05Pytorch.sample_actions))
        load = inspect.getattr_static(modeling_pi05.PI05Policy, "from_pretrained")
        self.assertIsInstance(load, classmethod)
        self.assertIn("config", inspect.signature(load.__func__).parameters)

    def test_the_first_pinned_step_says_so_once_for_server_start_to_wait_for(self) -> None:
        model = run_server_script("pi05", True)[0]()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            for seed in (1, 2):
                model.sample_actions(torch.randn(1, 5, COLUMNS, generator=torch.Generator().manual_seed(seed)))
        self.assertEqual(printed.getvalue(), f"{remote.PIN_MARKER} (columns {USED} to {COLUMNS - 1})\n")

    def test_with_the_pin_the_model_sees_the_unused_columns_as_in_training(self) -> None:
        model_type, policy_type, started = run_server_script("pi05", True)
        self.assertEqual(started, ["server"])
        self.assertTrue(callable(policy_type.validate_chunk_input_features))  # the empty camera slots, as before
        model = model_type()
        for seed in (1, 2):  # the second chunk must pin to its own noise, not the first chunk's
            with self.subTest(seed=seed):
                model.seen.clear()
                noise = torch.randn(2, 5, COLUMNS, generator=torch.Generator().manual_seed(seed))
                actions = model.sample_actions(noise)
                self.assertEqual(len(model.seen), STEPS)
                for time, x_t in model.seen:
                    # Training showed the model time * noise there, since those columns' target is 0.
                    torch.testing.assert_close(x_t[:, :, USED:], time * noise[:, :, USED:])
                    # The arm's own columns go to the model as the sampler has them.
                    torch.testing.assert_close(x_t[:, :, :USED], noise[:, :, :USED] - (1 - time))
                torch.testing.assert_close(actions[:, :, :USED], noise[:, :, :USED] - 1)

    def test_the_server_downloads_the_model_without_its_training_checkpoints(self) -> None:
        policy_type = run_server_script("pi05", True)[1]
        self.assertEqual(policy_type.resolved, "/model")
        self.assertEqual(policy_type.downloads, [(("me/model",), {"revision": "main", "ignore_patterns": ["checkpoints/**"]})])

    def test_a_model_trained_in_32_bit_is_served_in_bfloat16(self) -> None:
        for pin in (True, False):
            with self.subTest(pin=pin):
                policy_type = run_server_script("pi05", pin)[1]
                config = types.SimpleNamespace(dtype=torch.float32)  # as a 32-bit training run saves it
                self.assertIsInstance(policy_type.from_pretrained("checkpoint", config=config), policy_type)
                self.assertEqual(policy_type.loaded, [("checkpoint", torch.bfloat16)])

    def test_the_pin_follows_a_time_given_per_action_step(self) -> None:
        # Trained real-time chunking gives each step of the chunk its own time.
        model = run_server_script("pi05", True)[0]()
        noise = torch.randn(1, 5, COLUMNS, generator=torch.Generator().manual_seed(3))
        model.denoise_step(prefix_pad_masks=None, past_key_values=None, x_t=noise, timestep=torch.ones(1, 5))
        times = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.5]])
        model.denoise_step(prefix_pad_masks=None, past_key_values=None, x_t=noise - 0.5, timestep=times)
        torch.testing.assert_close(model.seen[-1][1][:, :, USED:], times[:, :, None] * noise[:, :, USED:])

    def test_without_the_pin_the_model_runs_lerobots_sampler_unchanged(self) -> None:
        untouched = model_class()
        model_type, _, started = run_server_script("pi05", False)
        self.assertEqual(started, ["server"])
        self.assertEqual(model_type.sample_actions.__qualname__, untouched.sample_actions.__qualname__)
        self.assertEqual(model_type.denoise_step.__qualname__, untouched.denoise_step.__qualname__)
        model = model_type()
        noise = torch.randn(1, 5, COLUMNS, generator=torch.Generator().manual_seed(4))
        model.sample_actions(noise)
        for time, x_t in model.seen:
            torch.testing.assert_close(x_t, noise - (1 - time))

    def test_act_runs_lerobots_own_server_with_no_script(self) -> None:
        setup, command = remote.setup_script("act", {}, True)
        self.assertEqual(command, "lerobot-policy-server --config_path=/tmp/server.yaml")
        self.assertNotIn("/tmp/serve.py", setup)


class WaitUntilReadyTest(unittest.TestCase):
    LOADED, READY = "All keys loaded successfully!", "Policy server ready"

    def wait(self, lines: list[str], policy_type: str = "pi05", pin: bool = True) -> mock.Mock:
        from huggingface_hub import HfApi

        api = mock.create_autospec(HfApi, instance=True)
        api.fetch_job_logs.return_value = iter(lines)
        with mock.patch("huggingface_hub.HfApi", return_value=api), contextlib.redirect_stdout(io.StringIO()):
            remote.wait_until_ready("job", policy_type, pin)
        return api

    def test_a_pinned_server_is_ready_once_it_has_pinned_a_warm_up_chunk(self) -> None:
        api = self.wait([self.LOADED, f"{remote.PIN_MARKER} (columns 6 to 31)", self.READY])
        api.cancel_job.assert_not_called()

    def test_a_server_that_never_pinned_is_cancelled_rather_than_served_as_pinned(self) -> None:
        from huggingface_hub import HfApi

        api = mock.create_autospec(HfApi, instance=True)
        api.fetch_job_logs.return_value = iter([self.LOADED, self.READY])
        with mock.patch("huggingface_hub.HfApi", return_value=api), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                remote.wait_until_ready("job", "pi05", True)
        self.assertIn("without pinning", str(exited.exception.code))
        api.cancel_job.assert_called_once_with(job_id="job")

    def test_the_pin_is_not_waited_for_when_it_is_off_or_the_policy_has_none(self) -> None:
        self.wait([self.LOADED, self.READY], pin=False).cancel_job.assert_not_called()
        self.wait([self.READY], policy_type="act").cancel_job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
