"""The checks and LeRobot arguments `so101 run` and `so101 serve` share."""

import argparse
import json
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from so_101 import remote, settings, tasks
from so_101.check import preflight
from so_101.console import run_lerobot
from so_101.report import Report
from so_101.settings import ARMS, PortNotFoundError, cameras_arg
from so_101.tasks import get_task


@dataclass(frozen=True)
class PolicySetup:
    """What `run` and `serve` both need to drive the follower with a task's policy, after their
    shared checks: the policy, robot and camera arguments, and the remote server, if any."""

    task: tasks.Task
    action_steps: int
    server: object | None  # the running policy server's job, with --remote
    sizes: dict[str, tuple[int, int]]  # each camera's (width, height), as the checkpoint was trained
    arms: tuple[settings.ArmName, ...]  # the arms the checks covered
    found: list  # the task's cameras, in order, as the checks found them, with their current indices
    report: Report
    checkpoint: Path | None  # the local checkpoint; None with a remote server
    args: list[str]  # the strategy, policy, robot and camera arguments, and the step cap

    def steps_line(self, s: settings.Settings) -> str:
        n, cap = self.action_steps, s.teleop.max_relative_target
        limit = f"limit {cap:g} per step" if cap else "no per-step limit"
        if not self.server:
            return f"Predicts every {n} steps ({n / self.task.fps:.1f} s), {limit}."
        # The remote client asks again once the measured turnaround plus one step is left in its queue,
        # or its refill_seconds if that's more.
        if remote.EXECUTION[self.server.labels["policy"]] == "chunk":
            asks = "asks for the next once the turnaround plus one step is left (at least 0.5 s)"
            return f"Predicts {n} steps ({n / self.task.fps:.1f} s) at a time and {asks}, {limit}."
        return (f"Predicts {n / self.task.fps:.1f} s at a time with real-time chunking: it asks for the next "
                f"once {remote.RTC_REFILL_S:g} s is left (more if the turnaround is longer), and the server "
                f"steers each new prediction to agree with the motion under way, {limit}.")

    def run(self, command: str, args: list[str], log_path: Path | None, summary: Callable[[], None]) -> NoReturn:
        """Run `command` through the SSH tunnel to the remote server, if any, then call `summary`, say
        what still bills, and exit with the command's status: 130 when Ctrl+C lands while the tunnel opens."""
        code = 130
        try:
            with remote.tunnel(self.server.id) if self.server else nullcontext():
                code = run_lerobot(command, args, log_path=log_path)
        except KeyboardInterrupt:
            pass
        finally:
            summary()
            if self.server:
                print(f"The policy server keeps billing until: uv run so101 server stop (job {self.server.id})", flush=True)
        sys.exit(code)


def policy_task(ns: argparse.Namespace) -> tasks.Task:
    """The task `run` or `serve` drives. pi0.5 only runs on the cloud GPU, so it needs --remote. A local
    --checkpoint runs on this Mac, from no Hub revision."""
    task = get_task(ns.task)
    if ns.checkpoint and (ns.remote or ns.revision):
        sys.exit("--checkpoint runs a model saved on this Mac, so it takes neither --remote nor --revision.")
    if ns.policy == "pi05" and not ns.remote:
        sys.exit(f"pi0.5 runs on a cloud GPU. Add --remote, after: uv run so101 server start {task.name} --policy pi05")
    return task


def policy_setup(s: settings.Settings, ns: argparse.Namespace, task: tasks.Task, serving: bool = False) -> PolicySetup:
    """Check the setup and build the policy, robot and camera arguments for `run` and `serve`.

    With --remote, the server must already serve this model: starting one bills until `so101 server
    stop`, so it's never done here. With `serving`, for `serve`, the leader is checked too when it's
    plugged in, and its failures are only warnings, since only Teleop needs it, and the checks also
    fail on a bad rest pose or `[control]` setting."""
    local = local_checkpoint(ns.checkpoint, ns.policy) if ns.checkpoint else None
    config = json.loads((local / "config.json").read_text()) if local else task.policy_config(ns.revision, ns.policy)
    if ns.remote:
        remote.refuse_action_steps(ns.policy, ns.action_steps)
    action_steps = resolve_action_steps(ns.action_steps, config)
    server = None
    if ns.remote:
        if ns.policy not in s.remote:
            sys.exit(f"No [remote.{ns.policy}] section in {s.path}.")
        server = remote.require_server(
            task.name, ns.policy, ns.revision or "main", action_steps, s.remote[ns.policy].pin_unused_actions
        )

    # Capture each camera at the size the checkpoint was trained on.
    sizes = tasks.image_sizes(config)
    if set(sizes) != set(task.cameras):
        model = local or task.policy_for(ns.policy)
        sys.exit(f"{model} expects cameras {sorted(sizes)}, but the task records {sorted(task.cameras)}.")
    arms: tuple[settings.ArmName, ...] = ("follower",)
    if serving:
        try:
            s.leader.resolve_port()
            arms = ARMS
        except PortNotFoundError:
            pass  # the leader is optional: only Teleop needs it
    found, report = preflight(s, "serving the policy" if serving else "running the policy", arms=arms,
                              cameras=[s.camera(name) for name in task.cameras],
                              control="FAIL" if serving else None, teleop_only=("leader",) if serving else (),
                              video=True)
    specs = [cam.lerobot_spec(width=sizes[cam.name][0], height=sizes[cam.name][1]) for cam in found]
    checkpoint = None if server else local or task.download_policy(ns.revision)
    args = [
        f"--strategy.type={'base' if serving else 'episodic'}",
        # A remote server sets the action steps itself, and its client rejects --policy flags.
        *(
            remote.client_args(ns.policy)
            if server
            else [f"--policy.path={checkpoint}", f"--policy.n_action_steps={action_steps}"]
        ),
        *s.follower_args(),
        cameras_arg(specs),
        f"--fps={task.fps}",
        *([] if server else [f"--device={s.run.device}"]),
    ]
    return PolicySetup(task, action_steps, server, sizes, arms, found, report, checkpoint, args)


def local_checkpoint(text: str, policy: str) -> Path:
    """The model folder (`pretrained_model`, with config.json) under `text`: the folder `train
    --target local` printed, which holds the last checkpoint, or a model folder itself."""
    path = Path(text).expanduser()
    for model in (path / "checkpoints" / "last" / "pretrained_model", path):
        if (model / "config.json").is_file():
            break
    else:
        sys.exit(f"--checkpoint {text}: no model there. Pass the folder `train --target local` printed, once it "
                 "has saved a checkpoint, or a pretrained_model folder.")
    found = json.loads((model / "config.json").read_text()).get("type")
    if found != policy:
        sys.exit(f"--checkpoint {text} holds a model of type {found}, not {policy}: pass --policy {found}.")
    return model.resolve()


def resolve_action_steps(requested: int | None, config: dict) -> int:
    """How many actions to carry out per prediction: at least 1, and no more than the model predicts.

    Without a `requested` number (--action-steps), uses the value the model in `config` was trained with.
    """
    steps = requested
    if steps is None:
        steps = config.get("n_action_steps", config.get("chunk_size"))
    if steps is None:
        sys.exit("The model's config has neither n_action_steps nor chunk_size; pass --action-steps.")
    chunk = config.get("chunk_size", steps)
    if steps < 1:
        sys.exit(f"--action-steps must be at least 1, not {steps}.")
    if steps > chunk:
        print(f"This model predicts {chunk} steps at a time, so using {chunk} action steps instead of {steps}.", flush=True)
    return min(steps, chunk)
