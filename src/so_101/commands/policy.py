"""The checks and LeRobot arguments `so101 run` and `so101 serve` share."""

import argparse
import json
import os
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from so_101 import data, remote, rollout, settings, tasks
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
    found: list  # the cameras the checks found, with their current indices
    report: Report
    checkpoint: Path | None  # the local checkpoint; None with a remote server
    args: list[str]  # the strategy, policy, robot and camera arguments, and the step cap

    def steps_line(self, s: settings.Settings) -> str:
        n = self.action_steps
        return f"Predicts every {n} steps ({n / self.task.fps:.1f} s), limit {s.teleop.max_relative_target:g} per step."

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
    """The task `run` or `serve` drives. pi0.5 only runs on the cloud GPU, so it needs --remote."""
    task = get_task(ns.task)
    if ns.policy == "pi05" and not ns.remote:
        sys.exit(f"pi0.5 runs on a cloud GPU. Add --remote, after: uv run so101 server start {task.name} --policy pi05")
    return task


def policy_setup(s: settings.Settings, ns: argparse.Namespace, task: tasks.Task, serving: bool = False) -> PolicySetup:
    """Check the setup and build the policy, robot and camera arguments for `run` and `serve`.

    With --remote, the server must already serve this model: starting one bills until `so101 server
    stop`, so it's never done here. With `serving`, for `serve`, the leader is checked too when it's
    plugged in, and its failures are only warnings, since only Teleop needs it, and the checks also
    fail on a bad rest pose or `[control]` setting."""
    action_steps = resolve_action_steps(s, ns, task.policy_config(ns.revision, ns.policy))
    server = remote.require_server(task.name, ns.policy, ns.revision or "main", action_steps) if ns.remote else None

    # Capture each camera at the size the checkpoint was trained on.
    sizes = task.image_sizes(ns.revision, ns.policy)
    if set(sizes) != set(task.cameras):
        repo = task.policy_for(ns.policy)
        sys.exit(f"{repo} expects cameras {sorted(sizes)}, but the task records {sorted(task.cameras)}.")
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
    by_name = {cam.name: cam for cam in found}
    specs = [by_name[name].lerobot_spec(width=sizes[name][0], height=sizes[name][1]) for name in task.cameras]
    checkpoint = None if server else task.download_policy(ns.revision)
    args = [
        f"--strategy.type={'base' if serving else 'episodic'}",
        # A remote server sets the action steps itself, and its client rejects --policy flags.
        *(
            remote.client_args()
            if server
            else [f"--policy.path={checkpoint}", f"--policy.n_action_steps={action_steps}"]
        ),
        *s.follower.lerobot_args(),
        cameras_arg(specs),
        f"--fps={task.fps}",
        *([] if server else [f"--device={s.run.device}"]),
        *([f"--robot.max_relative_target={s.teleop.max_relative_target}"] if s.teleop.max_relative_target else []),
    ]
    return PolicySetup(task, action_steps, server, sizes, arms, found, report, checkpoint, args)


def describe_pose(pose: dict[str, float]) -> str:
    """Each joint's position, in order, comma-separated."""
    return ", ".join(f"{v:g}" for v in pose.values())


def set_start_pose(s: settings.Settings, task: tasks.Task) -> str | None:
    """Pass the demos' start pose to `so101-rollout`, if there are local demos, and describe it."""
    if start_pose := data.demo_start_pose(s, task):
        os.environ[rollout.START_POSE_ENV] = json.dumps(start_pose)
        return describe_pose(start_pose)
    return None


def resolve_action_steps(s: settings.Settings, ns: argparse.Namespace, config: dict) -> int:
    """How many actions to carry out per prediction: at least 1, and no more than the model predicts.

    Without --action-steps, uses the value the model was trained with.
    """
    steps = ns.action_steps
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
