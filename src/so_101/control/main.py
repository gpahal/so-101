"""`so101-control`: the control center's process, which `so101 serve` runs.

It takes LeRobot's rollout arguments, builds the policy, robot and engine through LeRobot's rollout
context as `lerobot-rollout` does, and then runs the controller instead of a rollout strategy.

It loads so101.toml and tasks.toml itself, as `so101 serve` does, and inherits `serve`'s environment,
so `SO101_CONFIG` and `SO101_DATA_DIR` apply here too. The rest pose, the leader and the `[control]`
limits come from so101.toml, the cameras from the task in tasks.toml, and the demos' start pose from
the task's local training set. Only what's particular to one run comes through one environment
variable, RUN_ENV: the run's folder, which names its task, and the run.json `serve` built, with the
fps, the operator and the policy (whether it runs on the cloud GPU's server, in which job). Before it
connects anything, it checks the rest pose and the `[control]` settings as `serve`'s preflight does,
and works out the start pose, so a bad setting or an unreadable training file stops it first.

The run's folder and its run.json are made here, only once the page's port is bound and the policy
has loaded, so a start that fails makes nothing under data/.

With `--inference.type=remote` LeRobot connects to the cloud GPU's policy server instead of loading
the policy here, and the engine it returns pauses, resumes and resets the same way.
"""

import contextlib
import json
import os
import signal
import sys
import threading
from pathlib import Path

# No `from __future__ import annotations` here: LeRobot's parser reads the config's type from the
# annotation of `cfg` on `main`'s `run`, and a string annotation breaks it.

# JSON: dir (the run's folder, <data_dir>/<task>/runs/<name>, not made yet) and manifest (run.json's
# contents: `control.fps`, `operator`, and `policy` with its type, revision, inference ("local" or
# "remote") and job; a remote run adds the checkpoint its server admitted it to, `artifact_identity`).
# The remote server itself is reached through `--inference.*` arguments; the manifest says what it serves.
RUN_ENV = "SO101_RUN"
# Ctrl+C, `kill` and a closed terminal all quit as Esc does:
# the arm goes to rest before the motors turn off. Without a handler, SIGTERM and SIGHUP would end the
# process at once, and the motors would turn off wherever the arm is. They submit Quit with source
# `signal`, and no signal can confirm exiting where the arm can't get to rest: only y at the terminal
# does (states.py). With no terminal, the arm then holds until its power is switched off. Closing the
# terminal sends SIGHUP to the whole process group, and `so101 serve` passes its own copy on as SIGINT:
# the second Quit is refused while the first is under way.
QUIT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def exit_on_signals() -> None:
    """Until the controller runs, make SIGTERM and SIGHUP raise SystemExit, as Ctrl+C raises
    KeyboardInterrupt, so every cleanup registered so far runs. Left at their default, they would end
    the process with a connected follower's motors still on. `quit_on_signals` replaces this. Main
    thread only."""

    def stop(signum, frame) -> None:
        sys.exit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)


def quit_on_signals(submit) -> None:
    """Make each of QUIT_SIGNALS submit a Quit with source `signal`. Main thread only."""
    from so_101.control import terminal

    def on_signal(signum, frame) -> None:
        submit("quit", source="signal").add_done_callback(terminal.report)

    for signum in QUIT_SIGNALS:
        signal.signal(signum, on_signal)


def interrupt_on_signals() -> None:
    """During cleanup, make each of QUIT_SIGNALS raise KeyboardInterrupt. It ends only the cleanup it
    lands in, and the ExitStack still runs the others, where the signals' default action would end
    them all. One that lands inside turning the motors off cuts that short. Main thread only."""

    def interrupt(signum, frame) -> None:
        raise KeyboardInterrupt

    for signum in QUIT_SIGNALS:
        signal.signal(signum, interrupt)


def check_settings(s) -> None:
    """Exit unless the rest pose and the `[control]` settings pass the checks `so101 serve`'s preflight
    runs (`check.check_control`). `serve` runs them before it starts this process, but this process
    reads so101.toml again, which may have changed since (downloading the policy can take minutes). The
    checks read so101.toml and the follower's calibration file only."""
    from so_101.check import check_control
    from so_101.report import Report

    report = Report()
    check_control(report, s, "FAIL")
    if failures := report.failures:
        problems = "\n".join(f"  {r.group}: {r.detail}" + (f"\n    fix: {r.fix}" if r.fix else "") for r in failures)
        sys.exit(f"so101-control can't start with these settings from {s.path}:\n{problems}")


def leader_maker(arm):
    """Makes a new, unconnected SO-101 leader each call, on the port its board is on now, in degrees
    like the follower, LeRobot's default, so its pose can drive the follower."""
    return lambda: arm.lerobot_device(arm.resolve_port())


def checker(s, task_name: str):
    """Run checks for the controller: `so101 check`'s system checks (disk space and uvc-util), its
    device checks on the buses and cameras the control loop has open, never a second connection to a
    port, and the task's check, which reads only local files. The checks run on the control loop's
    thread, so nothing here touches the network: the Hugging Face login check, which asks the Hub with
    no timeout, is left out, as are the loop timing test (a slow loop is logged as `loop_rate`,
    and the status names it in the previews' note or `warnings`) and the live test, which moves the arm."""

    def run(buses: dict, cameras: dict):
        from so_101 import tasks
        from so_101.check import check_arm_files, check_cameras, check_motors, check_system, check_task, find_cameras
        from so_101.report import Report

        task = tasks.load().get(task_name)  # None if it left tasks.toml while `serve` runs
        report = Report()
        check_system(report, s, [s.camera(name) for name in cameras], keyboard=False, hub=False)
        for name, bus in buses.items():
            check_arm_files(report, s.arm(name))
            check_motors(report, s.arm(name), bus)
        check_cameras(report, find_cameras(report, [s.camera(name) for name in cameras]), cameras)
        if task is not None:
            check_task(report, s, task)
        return report

    return run


def require_serve_env() -> None:
    """Exit unless `so101 serve` set the variable this process reads."""
    if not os.environ.get(RUN_ENV):
        sys.exit(f"so101-control runs only under `so101 serve`, which sets {RUN_ENV}. Run `uv run so101 serve` instead.")


def policy_name(info: dict) -> str:
    """How the status names the policy, such as `pi05@50-demos (cloud GPU)`."""
    name = f"{info['type']}@{info['revision']}"
    return f"{name} (cloud GPU)" if info["inference"] == "remote" else name


def keep_awake() -> str | None:
    """Keep the Mac from idle sleep while this process runs: `caffeinate -i -w <pid>`, which ends
    with it. Returns what to tell the operator, or None off macOS. A closed lid still sleeps a laptop
    without an external display."""
    import shutil
    import subprocess

    from so_101 import svtshim

    if sys.platform != "darwin" or not (caffeinate := shutil.which("caffeinate")):
        return None
    try:
        subprocess.Popen([caffeinate, "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=svtshim.other_env())
    except OSError as error:
        return f"Couldn't keep the Mac awake ({error}): it may sleep during a long run."
    return "The Mac stays awake while so101 serve runs (caffeinate); a closed lid still sleeps it."


def remote_context(cfg):
    """LeRobot's remote rollout context, and a function that connects a new remote engine to the
    same server after the first one fails, which LeRobot's own engine never does.

    The new engine is admitted with the same observation and action contract as the first, which
    LeRobot works out while building the context: it's taken from that first admission. And it must
    find the same checkpoint: the first session's `artifact_identity` becomes `expected_artifact`,
    so a server restarted on another checkpoint is refused."""
    import dataclasses

    from lerobot.remote_inference import RemoteClient, RemoteInferenceEngine
    from lerobot.rollout import build_rollout_context

    admission: dict = {}
    admit = RemoteClient.admit

    def remember(client, **contract):
        admission.update(contract)
        return admit(client, **contract)

    RemoteClient.admit = remember
    try:
        ctx = build_rollout_context(cfg, threading.Event())
    finally:
        RemoteClient.admit = admit
    first = ctx.policy.inference
    pinned = dataclasses.replace(cfg.inference, expected_artifact=first.client.artifact_identity)
    print(f"Connected to the cloud GPU: checkpoint {first.client.artifact_identity}.", flush=True)

    def reconnect():
        client = RemoteClient.connect(pinned)
        try:
            client.admit(**admission)
        except BaseException:
            client.close()
            raise
        return RemoteInferenceEngine(client=client, config=pinned, dataset_features=ctx.data.dataset_features,
                                     rename_map=cfg.rename_map, robot_wrapper=ctx.hardware.robot_wrapper,
                                     task=first.task, shutdown_event=threading.Event())

    return ctx, reconnect


def control(cfg) -> None:
    exit_on_signals()  # first: LeRobot's imports and the policy's loading can take minutes

    import logging

    from lerobot.rollout import build_rollout_context
    from lerobot.utils import init_logging

    from so_101 import data, devices, settings, tasks
    from so_101.control import terminal
    from so_101.control.controller import Controller
    from so_101.control.follower import pose_from_steps
    from so_101.control.policy import EngineDriver
    from so_101.control.recorder import Recorder
    from so_101.control.server import ControlServer, resolve_bind, route_warning
    from so_101.control.watch import DeviceWatch

    init_logging(console_level=logging.WARNING)
    # so101.toml, read once for the whole run.
    s = settings.load()
    check_settings(s)  # before anything is connected or bound
    control_settings = s.control
    run = json.loads(os.environ[RUN_ENV])
    folder, manifest = data.Run.at(Path(run["dir"])), run["manifest"]
    policy, fps = manifest["policy"], manifest["control"]["fps"]
    task = tasks.load()[folder.task]
    # Worked out once, before the robot connects: it reads every training file, and one that can't be
    # read stops this process with nothing connected.
    start_pose = data.demo_start_pose(s, task)
    if start_pose:
        print(f"Each attempt starts at the demos' start pose: {data.describe_pose(start_pose)}", flush=True)
    else:
        print(f"{task.name} has no build to take the demos' start pose from, so each attempt starts from wherever the "
              f"arm is. Make one with: uv run so101 build {task.name}", flush=True)
    # The page's port is taken before the policy loads, so a port in use fails in seconds.
    # With `tailscale serve` set up for the port, the page is HTTPS on this Mac's MagicDNS name, and
    # the server listens on 127.0.0.1 behind it. A Tailscale route that would make it unsafe stops
    # serve here, before anything connects.
    port = control_settings.port
    try:
        bind = resolve_bind(control_settings)
    except ValueError as error:
        sys.exit(str(error))
    # Each cleanup is registered as its resource is made, so every one runs, even when one before it
    # fails, in the reverse order of setup: the key reader stops, the run closes, writing run_end
    # after a quit, the device watch stops, the policy stops, the follower disconnects, its motors
    # turning off, and the page's server, bound first, stops last.
    with contextlib.ExitStack() as cleanup:
        try:
            # Tailscale's routes are checked again every few seconds while serve runs, off the control loop.
            server = ControlServer(bind, port, task=task.name, routes=lambda: route_warning(bind, port))
        except OSError as error:
            sys.exit(f"Can't serve the control page on {bind.address}:{port}: {error}. Is another "
                     "`so101 serve` running? Stop it, or set another port under [control] in so101.toml.")
        cleanup.callback(server.stop)
        for warning in bind.warnings:
            print(warning, flush=True)

        remote = policy["inference"] == "remote"
        print(
            "Connecting to the cloud GPU's policy server and the follower..."
            if remote
            else "Loading the policy and connecting the follower...",
            flush=True,
        )
        if remote:
            ctx, reconnect = remote_context(cfg)
        else:
            ctx, reconnect = build_rollout_context(cfg, threading.Event()), None
        robot = ctx.hardware.robot_wrapper

        def disconnect_follower() -> None:
            """Turn the follower's motors off and close its port, then its cameras. LeRobot's own
            disconnect refuses while any camera reports itself disconnected, which would leave the
            motors on."""
            inner = robot.inner
            try:
                if inner.bus.is_connected:
                    inner.bus.disconnect(inner.config.disable_torque_on_disconnect)
            finally:
                for camera in inner.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()

        cleanup.callback(disconnect_follower)
        driver = EngineDriver.from_context(ctx, reconnect=reconnect)
        cleanup.callback(driver.stop)
        # The robot's camera keys are the so101.toml camera names. The device watch finds those cameras
        # by usb_id, and the arms by serial number.
        arms = [s.follower, s.leader]
        watched = [s.camera(name) for name in task.cameras]
        watch = DeviceWatch(lambda: devices.presence(arms, watched))
        cleanup.callback(watch.stop)
        watch.start()
        if not robot.supports_hold:
            robot.configure_hold()
        recorder = controller = None

        def close_run() -> None:
            if controller is not None:
                controller.close()  # an attempt still open isn't saved; the leader disconnects
            elif recorder is not None:
                recorder.close()

        cleanup.callback(close_run)
        # Now that the page's port is bound and the policy has loaded: a start that failed
        # before here made no folder, so `serve` never has to delete one.
        folder.path.mkdir(parents=True)
        if remote:  # `serve` named only the revision it asked for, and a branch such as main moves
            manifest["policy"]["artifact_identity"] = ctx.policy.inference.client.artifact_identity
        folder.write_manifest(manifest)
        print(f"Recording this run to {folder.path}: one episode per attempt, saved when the arm is back at "
              "rest.", flush=True)
        recorder = Recorder(folder, robot, fps=fps, task=cfg.task)
        controller = Controller(
            robot,
            driver,
            fps=fps,
            rest_pose=pose_from_steps(robot.inner.bus, control_settings.rest_pose),
            control=s.control,
            start_pose=start_pose,
            policy_name=policy_name(policy),
            notify=terminal.say,
            recorder=recorder,
            presence=watch.snapshot,
            # Even when the leader is unplugged now: plugged in later, it connects once the arm rests
            # or is paused. Its port is looked up again by serial number each time it connects.
            leader=leader_maker(s.leader),
            operator=manifest["operator"],
            checks=checker(s, task.name),
        )
        server.controller = controller
        server.start()
        cleanup.callback(interrupt_on_signals)
        quit_on_signals(controller.submit)
        driver.start()
        keys = terminal.KeyReader(controller.submit, controller.status)
        cleanup.callback(keys.stop)  # restores the terminal's settings
        keys.start()
        print(f"\nControl page: {server.url}", flush=True)
        if server.name_url:
            print(f"  or by its MagicDNS name: {server.name_url}", flush=True)
        if server.https:
            print("  HTTPS through Tailscale Serve: the page keeps a phone's screen awake while it has control.",
                  flush=True)
        elif control_settings.bind == "tailscale" and not bind.warnings:
            print(f"  HTTPS is off, so phones may dim and lock. Turn it on once with `tailscale serve --bg "
                  f"--https=443 localhost:{control_settings.port}`, then restart so101 serve.", flush=True)
        if awake := keep_awake():
            print(awake, flush=True)
        print(f"Ready. The arm holds where it is until you start an attempt.\n{terminal.HELP}", flush=True)
        controller.run()
        if controller.status()["state"] != "resting" or not controller.at_rest:
            print("Exiting away from rest: the motors turn off, so the arm goes limp.", flush=True)


def main() -> None:
    require_serve_env()  # first: nothing else here means anything without it
    from so_101 import svtshim

    svtshim.ensure()  # `so101 serve` starts this with the fix loaded
    svtshim.require()  # the recorder encodes each attempt's video with SVT-AV1 in this process
    from lerobot.configs import parser
    from lerobot.rollout import RolloutConfig
    from lerobot.scripts import lerobot_rollout  # noqa: F401  registers the robots and cameras with the parser
    from lerobot.utils import register_third_party_plugins

    register_third_party_plugins()

    @parser.wrap()
    def run(cfg: RolloutConfig) -> None:
        control(cfg)

    run()
    sys.exit(0)
