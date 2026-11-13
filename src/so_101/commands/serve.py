"""`so101 serve`: check the setup, then run the control center as `so101-control`."""

import argparse
import json
import os
import signal
import sys
from functools import partial

from so_101 import data, remote, settings
from so_101.commands.policy import policy_setup, policy_task


def cmd_serve(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    from so_101.control.main import RUN_ENV
    from so_101.recording import policy_commit, run_manifest

    task = policy_task(ns)
    # The control center reads the fps from run.json, and LeRobot's own parts from --fps: one forwarded
    # --fps would leave the recording, the loop and the policy at different rates.
    if any(arg == "--fps" or arg.startswith("--fps=") for arg in extra):
        sys.exit(f"serve runs at the task's fps, {task.fps}: set fps under [{task.name}] in tasks.toml "
                 "instead of passing --fps.")
    setup = policy_setup(s, ns, task, serving=True)
    server = setup.server
    # so101-control reads so101.toml and tasks.toml itself, and works out and prints the demos' start
    # pose. Only what's particular to this run goes to it, in run.json's contents (RUN_ENV).
    policy = {
        "type": ns.policy,
        "repo": None if ns.checkpoint else task.policy_for(ns.policy),
        "revision": None if ns.checkpoint else ns.revision or "main",
        "checkpoint": str(setup.checkpoint) if ns.checkpoint else None,  # a model saved on this Mac
        "action_steps": setup.action_steps,
        "inference": "remote" if server else "local",
        "job": server.id if server else None,
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
    os.environ[RUN_ENV] = json.dumps({"dir": str(run.path), "manifest": manifest})
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
        print("\nNothing was recorded: `serve` stopped before the run started, so it made no run folder.", flush=True)
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
