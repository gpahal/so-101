"""`so101-control`: the control center's process, which `so101 serve` and `so101 run` run.

It takes LeRobot's rollout arguments, builds the policy, robot and engine through LeRobot's rollout
context as `lerobot-rollout` does, and then runs the controller instead of a rollout strategy.

It loads so101.toml and tasks.toml itself, as `so101 serve` and `so101 run` do, and inherits their
environment, so `SO101_CONFIG` and `SO101_DATA_DIR` apply here too. The rest pose, the leader and the `[control]`
limits come from so101.toml, the cameras from the task in tasks.toml, and the demos' start pose from
the task's local training set. Only what's particular to one run comes through one environment
variable, RUN_ENV: the run's folder, which names its task, and the run.json `serve` or `run` built,
with the fps, the operator and the policy (whether it runs on the cloud GPU's server, in which job),
and in `so101 run` the autopilot's settings. Before it connects anything, it checks the rest pose and
the `[control]` settings as their preflight does, and works out the start pose, so a bad setting or an unreadable training file stops it first.

The run's folder and its run.json are made here, only once the page's port is bound and the policy
has loaded, so a start that fails makes nothing under data/.

With `--inference.type=remote` LeRobot connects to the cloud GPU's policy server instead of loading
the policy here, and the engine it returns pauses, resumes and resets the same way.
"""

import contextlib
import dataclasses
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

# No `from __future__ import annotations` here: LeRobot's parser reads the config's type from the
# annotation of `cfg` on `main`'s `run`, and a string annotation breaks it.

# JSON from `so101 serve` or `so101 run`: dir (the run's folder, <data_dir>/<task>/runs/<name>, not made yet),
# supervisor when given (off, shadow or act, over `[supervisor] mode`), and manifest, run.json's contents:
# `control.fps`, `operator`, `mode` (serve or run), `autopilot` in run, and `policy` (type, revision, the Hub
# `commit` its files were read at, inference "local" or "remote", job, and for a skills run `skills` and
# `gripper_sha256`, which `require_run_skills` checks again). The remote server is reached through
# `--inference.*`. This process adds `grasp_thresholds` and `supervisor_program` (the files the perception and
# supervisor read, their versions and `skills.RUN_LABEL_VERSION`; with the supervisor off, the mode alone), and
# for a remote run `policy.artifact_identity`, the checkpoint its server admitted it to.
RUN_ENV = "SO101_RUN"
# Ctrl+C, `kill` and a closed terminal all quit as Esc does:
# the arm goes to rest before the motors turn off. Without a handler, SIGTERM and SIGHUP would end the
# process at once, and the motors would turn off wherever the arm is. They submit Quit with source
# `signal`, and no signal can confirm exiting where the arm can't get to rest: only y at the terminal
# does (states.py). With no terminal, the arm then holds until its power is switched off. Closing the
# terminal sends SIGHUP to the whole process group, and `so101 serve` or `so101 run` passes its own copy on as SIGINT:
# the second Quit is refused while the first is under way.
QUIT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


# Ctrl+Z would stop the loop with the arm powered and holding its last goal, and the keys, the page and
# the policy frozen with it. `so101 serve` and `so101 run` ignore it, and this process says how to quit
# instead. {command} is the one that started it.
SUSPEND_REFUSED = "Ctrl+Z doesn't suspend {command}: the arm would hold with nothing watching it. Esc quits."


def command_name(run: dict) -> str:
    """The command this process runs for, from RUN_ENV's JSON: so101 serve or so101 run."""
    return f"so101 {run.get('manifest', {}).get('mode') or 'serve'}"


def refuse_suspend(command: str = "so101 serve") -> None:
    """Make Ctrl+Z (SIGTSTP) print SUSPEND_REFUSED instead of stopping this process. Main thread only."""

    def refuse(signum, frame) -> None:
        with contextlib.suppress(OSError, ValueError, RuntimeError):  # a closed terminal can't take it
            print(SUSPEND_REFUSED.format(command=command), flush=True)

    signal.signal(signal.SIGTSTP, refuse)


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


def check_settings(s, task: str | None = None) -> None:
    """Exit unless the rest pose, the `[control]` settings and the supervisor's files and mode for `task`
    pass the checks `so101 serve`'s preflight runs (`check.check_control`). `serve` runs them before it
    starts this process, but this process reads so101.toml again, which may have changed since
    (downloading the policy can take minutes). The checks read so101.toml and the follower's calibration
    file only."""
    from so_101.check import check_control
    from so_101.report import Report

    report = Report()
    check_control(report, s, "FAIL", task)
    if failures := report.failures:
        problems = "\n".join(f"  {r.group}: {r.detail}" + (f"\n    fix: {r.fix}" if r.fix else "") for r in failures)
        sys.exit(f"so101-control can't start with these settings from {s.path}:\n{problems}")


def policy_parts(s):
    """The grasp thresholds every run's perception uses, and, when `[supervisor] mode` isn't off, the
    supervisor's program and its judges; with the supervisor off, None for both, and the program isn't
    read. Exits on a bad thresholds file, or a bad program with the supervisor on, before anything is
    connected. API keys come from the environment or .scratch/bench.env beside so101.toml, as
    `so101-failbench` reads them; a judge without its key or SDK is left out, and the start event names
    why."""
    from so_101.grasp import thresholds

    try:
        grasp = thresholds.load(s.perception.thresholds)
    except thresholds.ThresholdsError as error:
        sys.exit(f"so101-control can't load the grasp thresholds its perception reads: {error}")
    if s.supervisor.mode == "off":
        return grasp, None, None
    from so_101.supervisor import program

    try:
        loaded = program.load(s.supervisor.program)
    except program.ProgramError as error:
        sys.exit(f"so101-control can't load the program its supervisor reads: {error}")
    from so_101.vision.env import load_env
    from so_101.supervisor import judges

    load_env(s.path.parent / ".scratch" / "bench.env")
    made = judges.Judges(loaded)
    # The layer tells the operator once which judges are left out, and why, when it starts.
    print(f"Supervisor: {s.supervisor.mode} mode, program v{loaded.version}.", flush=True)
    return grasp, loaded, made


def require_run_skills(policy: dict, task, gripper) -> None:
    """Exit, before anything connects, unless a skills run's skills and `[gripper]` thresholds, as this
    process loaded them, are the ones `serve` or `run` checked the model against and wrote in run.json
    (`policy.skills` and `policy.gripper_sha256`). Either file may change while the policy downloads,
    and the run would then switch skills where its model wasn't trained to, and record skills it didn't
    use. One refusal names every difference. A run without skills checks nothing."""
    from so_101 import skills

    if not policy.get("skills"):
        return
    differ = []
    here = json.loads(json.dumps([skill.record() for skill in task.skills]))
    if here != policy["skills"]:
        names = lambda records: ", ".join(f"{r.get('name')} ({r.get('prompt')!r})" for r in records) or "none"  # noqa: E731
        differ.append(f"[{task.name}.skills] in tasks.toml now has {names(here)}, and the model was checked against "
                      f"{names(policy['skills'])}")
    if skills.gripper_hash(gripper) != policy.get("gripper_sha256"):
        differ.append("the [gripper] thresholds in the grasp thresholds file aren't the ones the model was checked against")
    if differ:
        sys.exit(f"so101-control: {'; '.join(differ)}. They changed after the run's checks: put them back, or start "
                 "the run again so its checks run on the files as they are now.")


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

        task = tasks.load().get(task_name)  # None if it left tasks.toml while the run is on
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
    """Exit unless `so101 serve` or `so101 run` set the variable this process reads."""
    if not os.environ.get(RUN_ENV):
        sys.exit(f"so101-control runs only under `so101 serve` or `so101 run`, which set {RUN_ENV}. Run "
                 "`uv run so101 serve` or `uv run so101 run` instead.")


def run_settings(s, run: dict, task: str | None = None):
    """so101.toml's settings with this run's own over them, checked (`check_settings`, for `task`): `run`'s
    supervisor mode over `[supervisor] mode`, checked with the file's settings, then `so101 run
    --duration` over `[control] attempt_limit_s`, where 0 lasts until a person ends the attempt. The
    duration goes on after the check, which refuses the infinite limit 0 stands for in a file. Exits on a
    value `so101 run` or `so101 serve` shouldn't have passed."""
    from so_101 import settings

    if (mode := run.get("supervisor")) is not None:
        if mode not in settings.SUPERVISOR_MODES:
            sys.exit(f"so101-control: the supervisor mode {mode!r} isn't one of {', '.join(settings.SUPERVISOR_MODES)}.")
        s = dataclasses.replace(s, supervisor=dataclasses.replace(s.supervisor, mode=mode))
    check_settings(s, task)  # before anything is connected or bound
    autopilot = run["manifest"].get("autopilot") or {}
    if (duration := autopilot.get("duration_s")) is not None:
        if (isinstance(duration, bool) or not isinstance(duration, int | float) or not math.isfinite(duration)
                or duration < 0):
            sys.exit(f"so101-control: --duration is {duration!r}: it must be a finite number of seconds of policy "
                     "time per attempt, 0 or more (0 runs until you end it).")
        s = dataclasses.replace(s, control=dataclasses.replace(s.control, attempt_limit_s=duration or math.inf))
    return s


def policy_name(info: dict) -> str:
    """How the status names the policy, such as `pi05@main (cloud GPU)`, or `act (local checkpoint)`."""
    if info.get("checkpoint"):
        return f"{info['type']} (local checkpoint)"
    name = f"{info['type']}@{info['revision']}"
    return f"{name} (cloud GPU)" if info["inference"] == "remote" else name


def keep_awake(command: str = "so101 serve") -> str | None:
    """Keep the Mac from idle sleep while this process runs: `caffeinate -i -w <pid>`, which ends
    with it. Returns what to tell the operator, or None off macOS. A closed lid still sleeps a laptop
    without an external display. `command` is the one this process runs for."""
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
    return f"The Mac stays awake while {command} runs (caffeinate); a closed lid still sleeps it."


def remote_context(cfg):
    """LeRobot's remote rollout context, and a function that connects a new remote engine to the
    same server after the first one fails, which LeRobot's own engine never does.

    A skills run (`policy.skills` in run.json) adds the skill one-hot, observation.environment_state, to
    the features LeRobot works out from the robot while it builds the context (`_skill_features`): the
    server advertises it for a skills model (`remote.server_config`), so the admission, the first engine
    and every reconnected engine carry it, and the engine sends it each tick. The new engine is admitted
    with the server's own capabilities, which the first admission had to equal field for field. And it
    must find the same checkpoint: the first session's `artifact_identity` becomes `expected_artifact`, so
    a server restarted on another checkpoint is refused."""
    import dataclasses

    from lerobot.remote_inference import RemoteClient, RemoteInferenceEngine
    from lerobot.rollout import build_rollout_context

    with _skill_features(_run_skills()):
        ctx = build_rollout_context(cfg, threading.Event())
    first = ctx.policy.inference
    pinned = dataclasses.replace(cfg.inference, expected_artifact=first.client.artifact_identity)
    print(f"Connected to the cloud GPU: checkpoint {first.client.artifact_identity}.", flush=True)

    def reconnect():
        client = RemoteClient.connect(pinned)
        try:
            admit(client, pinned, cfg.fps)
        except BaseException:
            client.close()
            raise
        return RemoteInferenceEngine(client=client, config=pinned, dataset_features=ctx.data.dataset_features,
                                     rename_map=cfg.rename_map, robot_wrapper=ctx.hardware.robot_wrapper,
                                     task=first.task, shutdown_event=threading.Event())

    return ctx, reconnect


def admit(client, config, fps: float) -> None:
    """Admit `client` with the observation and action contract its server advertises, as the first
    session was: LeRobot's context admits the robot's features only when they equal those."""
    caps = client.capabilities
    client.admit(features=caps.features, action_feature=caps.action_feature, semantics=config.semantics,
                 action_interval=1.0 / fps, mode=config.mode)


def _run_skills() -> tuple[str, ...]:
    """The names of a skills run's skills, in order, from run.json's policy.skills; empty otherwise."""
    manifest = json.loads(os.environ.get(RUN_ENV) or "{}").get("manifest") or {}
    return tuple(skill["name"] for skill in (manifest.get("policy") or {}).get("skills") or ())


@contextlib.contextmanager
def _skill_features(names: tuple[str, ...]):
    """While LeRobot builds the rollout context, add the skill one-hot to the features it works out from
    the robot's joints and cameras: its public processor hook would fold extra floats into
    observation.state instead. Nothing changes without skills."""
    if not names:
        yield
        return
    from lerobot.rollout import context

    from so_101.skills import STATE_KEY

    aggregate = context._aggregate_rollout_features

    def with_skill(*args, **kwargs):
        one_hot = {"dtype": "float32", "shape": (len(names),), "names": [f"skill.{n}" for n in names]}
        return {**aggregate(*args, **kwargs), STATE_KEY: one_hot}

    context._aggregate_rollout_features = with_skill
    try:
        yield
    finally:
        context._aggregate_rollout_features = aggregate


def control(cfg) -> None:
    exit_on_signals()  # first: LeRobot's imports and the policy's loading can take minutes
    run = json.loads(os.environ[RUN_ENV])
    command = command_name(run)
    refuse_suspend(command)

    from lerobot.rollout import build_rollout_context
    from lerobot.utils import init_logging

    from so_101 import data, devices, settings, tasks
    from so_101.control import terminal
    from so_101.control.controller import Controller
    from so_101.control.follower import pose_from_steps
    from so_101.skills import RUN_LABEL_VERSION
    from so_101.policy.engine import EngineDriver
    from so_101.control.recorder import Recorder
    from so_101.control.server import ControlServer, resolve_bind, route_warning
    from so_101.control.watch import DeviceWatch

    init_logging(console_level="WARNING")  # a level name: LeRobot calls .upper() on it
    folder, manifest = data.Run.at(Path(run["dir"])), run["manifest"]
    # so101.toml, read once for the whole run, and checked before anything is connected or bound.
    s = run_settings(settings.load(), run, folder.task)
    grasp, program, judges = policy_parts(s)
    # What the perception and the supervisor read, so a run can be matched to the thresholds and
    # program it ran under after either file changes. With the supervisor off no program is read, and
    # supervisor_program records only the mode.
    manifest["grasp_thresholds"] = {"path": str(grasp.path), "gripper_version": grasp.gripper.version,
                                    "rules_version": grasp.rules.version, "run_labels": RUN_LABEL_VERSION}
    manifest["supervisor_program"] = {"mode": s.supervisor.mode}
    if program is not None:
        manifest["supervisor_program"] |= {"path": str(program.path), "version": program.version}
    # `so101 run`: an autopilot starts and ends the attempts (docs/run.md), each lasting its --duration
    # of policy time (`run_settings`).
    autopilot = manifest.get("autopilot")
    control_settings = s.control
    policy, fps = manifest["policy"], manifest["control"]["fps"]
    task = tasks.load()[folder.task]
    require_run_skills(policy, task, grasp.gripper)
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
    # it here, before anything connects.
    port = control_settings.port
    try:
        bind = resolve_bind(control_settings, command)
    except ValueError as error:
        sys.exit(str(error))
    # Each cleanup is registered as its resource is made, so every one runs, even when one before it
    # fails, in the reverse order of setup: the key reader stops, the run closes, writing run_end
    # after a quit, the device watch stops, the policy stops, the follower disconnects, its motors
    # turning off, and the page's server, bound first, stops last.
    with contextlib.ExitStack() as cleanup:
        try:
            # Tailscale's routes are checked again every second while serve or run runs, off the control loop.
            server = ControlServer(bind, port, task=task.name, routes=lambda: route_warning(bind, port, command),
                                   command=command)
        except OSError as error:
            sys.exit(f"Can't serve the control page on {bind.address}:{port}: {error}. Is another "
                     "`so101 serve` or `so101 run` running? Stop it, or set another port under [control] in so101.toml.")
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

        recorder = controller = None

        def disconnect_follower() -> None:
            """Turn the follower's motors off and close its port, then its cameras. LeRobot's own
            disconnect refuses while any camera reports itself disconnected, which would leave the
            motors on. After a quit that found the follower silent, its motors are left as they are
            (`Controller.leaves_motors`): one that answers again by now may hold a raised arm."""
            inner = robot.inner
            leave = controller is not None and controller.leaves_motors
            try:
                if inner.bus.is_connected:
                    inner.bus.disconnect(inner.config.disable_torque_on_disconnect and not leave)
            finally:
                for camera in inner.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()

        cleanup.callback(disconnect_follower)
        driver = EngineDriver.from_context(ctx, reconnect=reconnect)
        policy_layer = None
        # Once the stack is built, stopping it stops every layer (the supervisor's judges) and then
        # the model; before that, only the model is there to stop.
        cleanup.callback(lambda: (policy_layer or driver).stop())
        # The robot's camera keys are the so101.toml camera names. The device watch finds those cameras
        # by usb_id, and the arms by serial number.
        arms = [s.follower, s.leader]
        watched = [s.camera(name) for name in task.cameras]
        watch = DeviceWatch(lambda: devices.presence(arms, watched))
        cleanup.callback(watch.stop)
        watch.start()
        if not robot.supports_hold:
            robot.configure_hold()

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
        # The policy stack over the model (docs/policy-layers.md): the loop steps its top.
        from so_101.policy import stack
        from so_101.policy.base import LayerContext

        policy_layer = stack.build(driver, LayerContext(recorder.event, time.monotonic, grasp, program, fps),
                                   skills=task.skills if policy.get("skills") else (), judges=judges,
                                   mode=s.supervisor.mode, cameras=tuple(robot.inner.cameras))
        cleanup.callback(policy_layer.close)  # before the recorder closes: late verdicts are dropped
        controller = Controller(
            robot,
            policy_layer,
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
            command=command,
        )
        if autopilot:
            from so_101.control.autopilot import Autopilot

            # Stepped by the controller on every tick, after people's commands and the link check.
            controller.autopilot = Autopilot(command=controller.pilot_command,
                                             say=lambda message: terminal.say(message, ""),
                                             episodes=autopilot["episodes"], reset_s=autopilot["reset_s"],
                                             link_timeout_s=s.control.link_timeout_s)
        server.controller = controller
        controller.page_url = server.url
        server.start()
        cleanup.callback(interrupt_on_signals)
        quit_on_signals(controller.submit)
        policy_layer.start()
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
                  f"--https=443 localhost:{control_settings.port}`, then restart {command}.", flush=True)
        if awake := keep_awake(command):
            print(awake, flush=True)
        if autopilot:
            print(f"Ready. The autopilot makes {autopilot['episodes']} attempt(s), {autopilot['reset_s']:g} s apart. "
                  f"It starts and ends attempts only while a control page is in control: open {server.url} and take "
                  "control there. Without one, start each attempt with r and end it with the right arrow yourself. "
                  "Your Pause, Resume, Rewind, Teleop or Reset takes over the attempt it's in, and Hold on the page "
                  f"keeps the next one from starting.\n{terminal.HELP}", flush=True)
        else:
            print(f"Ready. The arm holds where it is until you start an attempt.\n{terminal.HELP}", flush=True)
        controller.run()
        if controller.status()["state"] != "resting" or not controller.at_rest:
            print("Exiting away from rest: the motors turn off, so the arm goes limp.", flush=True)


def main() -> None:
    require_serve_env()  # first: nothing else here means anything without it
    from so_101 import connect, svtshim

    svtshim.ensure()  # `so101 serve` and `so101 run` start this with the fix loaded
    svtshim.require()  # the recorder encodes each attempt's video with SVT-AV1 in this process
    from lerobot.configs import parser
    from lerobot.rollout import RolloutConfig
    from lerobot.scripts import lerobot_rollout  # noqa: F401  registers the robots and cameras with the parser
    from lerobot.utils import register_third_party_plugins

    register_third_party_plugins()
    connect.install()  # the follower connects holding where it is, never at a stale goal

    @parser.wrap()
    def run(cfg: RolloutConfig) -> None:
        control(cfg)

    run()
    sys.exit(0)
