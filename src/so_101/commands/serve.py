"""`so101 serve` and `so101 run`: check the setup, then run the control center as `so101-control`.

They are one program (docs/run.md). `run` adds an autopilot that starts and ends the attempts, and
passes its settings in run.json's `autopilot`; everything else, the policy stack, the page, the keys,
the safety rules and the recording into data/<task>/runs/, is the same.
"""

import argparse
import dataclasses
import json
import os
import signal
import sys
from functools import partial

from so_101 import data, remote, settings, tasks
from so_101.commands.policy import policy_setup, policy_task

# The LeRobot arguments `run` and `serve` pass on to so101-control, by prefix: how to reach a remote
# server, the policy's own options, the follower's, and LeRobot's renaming of observation keys. Every
# other rollout argument is either set here from so101.toml and tasks.toml, or means nothing to the
# control center, which runs its own loop instead of LeRobot's strategies.
FORWARDED = ("--inference.", "--policy.", "--robot.")
RENAME_MAP = "--rename_map"


def lerobot_extra(extra: list[str], mode: str, fps: int) -> list[str]:
    """`extra` when every argument in it is one `run` and `serve` forward (FORWARDED, or RENAME_MAP), with
    a value either after `=` or as the next argument; otherwise exit naming each one refused."""
    refused, i = [], 0
    while i < len(extra):
        arg = extra[i]
        name = arg.split("=", 1)[0]
        # `--name value`: the next argument is this one's value, unless it's an option itself.
        takes_next = arg.startswith("--") and "=" not in arg and i + 1 < len(extra) and not extra[i + 1].startswith("--")
        i += 2 if takes_next else 1
        if name == "--fps":
            # The control center reads the fps from run.json, and LeRobot's own parts from --fps: one
            # forwarded --fps would leave the recording, the loop and the policy at different rates.
            sys.exit(f"{mode} runs at the task's fps, {fps}: set fps under the task in tasks.toml instead of "
                     "passing --fps.")
        if name == "--interpolation_multiplier":
            sys.exit(f"{mode} steps the arm once per frame at the task's fps, {fps}, so "
                     "--interpolation_multiplier would play each of the policy's actions over that many frames, "
                     "slower than it was trained, instead of adding steps between them. Leave it out.")
        if not (name.startswith(FORWARDED) or name == RENAME_MAP):
            refused.append(name)
    if refused:
        sys.exit(f"so101 {mode} doesn't pass {', '.join(refused)} on to LeRobot: it forwards only --inference.*, "
                 "--policy.*, --robot.* and --rename_map, and sets the rest from so101.toml and tasks.toml. "
                 f"See `uv run so101 {mode} --help` for its own options.")
    return extra


def cmd_serve(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    launch(s, ns, extra, "serve", policy_task(ns))


def cmd_run(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    task = policy_task(ns)
    if ns.duration is not None and ns.duration < 0:
        sys.exit("--duration is seconds of policy time per attempt: 0 or more (0 runs until you end it).")
    autopilot = {
        "episodes": s.run.episodes if ns.episodes is None else ns.episodes,
        "reset_s": task.reset_time_s if ns.reset_time is None else ns.reset_time,
        "duration_s": ns.duration,  # None: [control] attempt_limit_s
    }
    if autopilot["episodes"] < 1:
        sys.exit("--episodes is how many attempts to make: 1 or more.")
    launch(s, ns, extra, "run", task, autopilot)


def launch(s: settings.Settings, ns: argparse.Namespace, extra: list[str], mode: str, task: tasks.Task,
           autopilot: dict | None = None) -> None:
    """Check the setup and start so101-control for `mode` (serve or run) on `task`, with the autopilot's
    settings for run."""
    from so_101.control.main import RUN_ENV
    from so_101.recording import policy_commit, run_manifest

    extra = lerobot_extra(extra, mode, task.fps)
    if ns.supervisor is not None:
        # --supervisor stands in for [supervisor] mode: the preflight checks the mode this run uses.
        s = dataclasses.replace(s, supervisor=dataclasses.replace(s.supervisor, mode=ns.supervisor))
    setup = policy_setup(s, ns, task)
    server = setup.server
    # so101-control reads so101.toml and tasks.toml itself, and works out and prints the demos' start
    # pose. Only what's particular to this run goes to it, in run.json's contents (RUN_ENV).
    policy = {
        "type": ns.policy,
        "repo": None if ns.checkpoint else task.policy_for(ns.policy, ns.skills),
        # A skills model: so101-control's policy stack switches skills (`policy.skills.SkillSequencer`).
        "skills": [skill.record() for skill in task.skills] if ns.skills else None,
        "revision": None if ns.checkpoint else ns.revision or "main",
        "checkpoint": str(setup.checkpoint) if ns.checkpoint else None,  # a model saved on this Mac
        "action_steps": setup.action_steps,
        "inference": "remote" if server else "local",
        "job": server.id if server else None,
        # pi0.5 on the cloud GPU: whether the server pinned the model's unused action columns.
        "pin_unused_actions": {"true": True, "false": False}.get(server.labels.get(remote.PIN_LABEL)) if server else None,
    }
    print(setup.steps_line(s), flush=True)
    if "leader" not in setup.arms:
        print("The leader isn't plugged in: plug it in to take over with Teleop. It connects while the arm rests or is paused.", flush=True)
    if server:
        print(f"The policy runs on the cloud GPU: {remote.describe(server)}.", flush=True)
    # so101-control makes the run's folder and writes this as its run.json once the page's port is
    # bound and the policy has loaded, so a start that fails leaves nothing under data/.
    run = data.new_run(s, task)
    manifest = run_manifest(
        s, task, setup.found, setup.report,
        policy={**policy, "commit": policy_commit(setup.checkpoint) if setup.checkpoint else None},
        image_sizes=setup.sizes,
    )
    manifest["operator"] = ns.operator
    manifest["mode"] = mode
    if autopilot is not None:
        manifest["autopilot"] = autopilot
    # `supervisor` only with the flag: so101-control reads [supervisor] mode from so101.toml otherwise,
    # and records the mode it used in run.json's `supervisor_program`.
    sent = {"dir": str(run.path), "manifest": manifest}
    if ns.supervisor is not None:
        sent["supervisor"] = ns.supervisor
    os.environ[RUN_ENV] = json.dumps(sent)
    print("Not an emergency stop: keep a hand near the follower's power switch.\n", flush=True)
    # Ctrl+Z reaches this process too: stopped, it would leave the terminal to the shell. so101-control
    # inherits the ignore until it sets its own handler (`main.refuse_suspend`). uv keeps running.
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    setup.run("so101-control", [*setup.args, f"--task={task.prompt}", *extra], None, partial(print_serve_summary, run))


def print_serve_summary(run: data.Run) -> None:
    """Say where the run is and how its attempts went, from its events.jsonl (how it ended is its
    `run_end` event)."""
    from so_101.control.analysis import read_events

    if not run.path.is_dir():  # the child stopped before it was ready, as when the port is taken
        print("\nNothing was recorded: so101-control stopped before the run started, so it made no run folder.", flush=True)
        return
    events = read_events(run)  # skips a last line a crash cut short
    saved = sum(bool(e.get("saved")) for e in events if e.get("type") == "attempt_end")
    # An attempt's latest outcome, by its episode: a later one replaces an earlier one, as in `runs`.
    latest = {e.get("episode"): e.get("outcome") for e in events if e.get("type") == "outcome"}
    outcomes = list(latest.values())
    counts = ", ".join(f"{outcomes.count(o)} {o}" for o in ("succeeded", "failed", "ignored"))
    print(f"\nThis run is in {run.path}\n  Attempts saved: {saved} ({counts}, {saved - len(outcomes)} unlabelled)", flush=True)
    if not any(e.get("type") == "run_end" for e in events):
        lost = [e for e in events if e.get("type") == "attempt_end" and not e.get("saved")]
        print("  It has no run_end event: so101-control stopped without closing the run"
              + (f", and {len(lost)} attempt(s) open then weren't saved." if lost else "."), flush=True)
