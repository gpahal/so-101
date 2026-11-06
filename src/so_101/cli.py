"""`so101`: run LeRobot commands for the SO-101 arms with ports and ids from `so101.toml`.

Any arguments after the known ones are passed through to the LeRobot command, so
`uv run so101 teleop --teleop_time_s=30` works.
"""

import argparse
import json
import shlex
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from serial.tools import list_ports

from so_101 import data, remote, settings, tasks
from so_101.check import narrow_joints_warning, preflight
from so_101.console import filter_stderr, run_lerobot
from so_101.devices import apply_camera_controls, find_camera, video_devices
from so_101.report import Report
from so_101.settings import ARMS, Arm, ConfigError, PortNotFoundError, cameras_arg


UNTIL_STOPPED_S = 24 * 3600  # an attempt of `--duration 0`: LeRobot needs a length, and Esc or Ctrl+C ends it


def lerobot(command: str, args: list[str]) -> None:
    """Run a LeRobot CLI from the same virtualenv, and exit with its status.

    It runs as a child process rather than replacing this one, so its stderr goes through
    `console.filter_stderr`.
    """
    sys.exit(run_lerobot(command, args))


def arm_status(arm: Arm) -> str:
    try:
        port = arm.resolve_port()
        state = "connected" if port == arm.port else f"connected at {port} (saved port {arm.port} is gone)"
    except PortNotFoundError:
        port, state = arm.port, "not connected"
    calibrated = "yes" if arm.calibration_file.is_file() else "no"
    status = (
        f"{arm.name}\n"
        f"  id:            {arm.id}\n"
        f"  port:          {port} ({state})\n"
        f"  serial number: {arm.serial_number or '-'}\n"
        f"  calibrated:    {calibrated} ({arm.calibration_file})"
    )
    if warning := narrow_joints_warning(arm):
        status += (f"\n  WARNING:       {warning}. Run `uv run so101 calibrate {arm.name}` and move every joint "
                   "through its full range.")
    return status


def cmd_info(s: settings.Settings, _: argparse.Namespace, __: list[str]) -> None:
    print(f"config: {s.path}\ndata:   {s.data_dir}\n")
    for name in ARMS:
        print(arm_status(s.arm(name)), end="\n\n")
    print(f"teleop\n  fps:                 {s.teleop.fps}\n  max_relative_target: {s.teleop.max_relative_target or 'off'}")
    print("\ncameras")
    devices = video_devices()
    for cam in s.cameras.values():
        lookup = find_camera(cam, devices)
        where = lookup.problem or (f"index {lookup.camera.index}" + (f" ({lookup.note})" if lookup.note else ""))
        print(f"  {cam.name}: {where}, {cam.width}x{cam.height} @ {cam.fps} fps, rotation {cam.rotation}")
        if cam.controls:
            controls = ", ".join(f"{k}={v}" for k, v in cam.controls.items())
            print(f"    usb {cam.usb_id}: {controls}")


def cmd_env(s: settings.Settings, _: argparse.Namespace, __: list[str]) -> None:
    for name in ARMS:
        arm = s.arm(name)
        try:
            port = arm.resolve_port()
        except PortNotFoundError:
            port = arm.port
        key = f"SO101_{name.upper()}"
        print(f"export {key}_PORT={shlex.quote(port)}")
        print(f"export {key}_ID={shlex.quote(arm.id)}")
        print(f"export {key}_CALIBRATION_DIR={shlex.quote(str(arm.calibration_dir))}")
    devices = video_devices()
    for cam in s.cameras.values():
        print(f"export SO101_CAMERA_{cam.name.upper()}_INDEX={find_camera(cam, devices).camera.index}")


def usb_ports() -> dict[str, str]:
    return {p.device: p.serial_number or "" for p in list_ports.comports() if "usb" in p.device.lower()}


def cmd_find_port(_: settings.Settings, ns: argparse.Namespace, __: list[str]) -> None:
    before = usb_ports()
    if not before:
        sys.exit("No USB serial ports found. Plug in the controller boards and try again.")
    print("Connected:", ", ".join(before))
    input(f"Unplug the USB cable of the {ns.arm} board, then press Enter. ")
    time.sleep(0.5)
    gone = sorted(set(before) - set(usb_ports()))
    if len(gone) != 1:
        sys.exit(f"Expected exactly one port to disappear, saw {gone or 'none'}. Try again.")
    port = gone[0]
    path = settings.save_port(ns.arm, port, before[port])
    print(f"{ns.arm} port: {port} (serial number {before[port] or 'unknown'})\nSaved to {path}. Plug the cable back in.")


def cmd_scan(s: settings.Settings, ns: argparse.Namespace, __: list[str]) -> None:
    from so_101.scan import scan  # imports LeRobot, which is slow

    for name in [ns.arm] if ns.arm else ARMS:
        arm = s.arm(name)
        try:
            port = arm.resolve_port()
        except PortNotFoundError as e:
            print(f"{name}: {e}")
            continue
        found = scan(port)
        if not found:
            print(f"{name} ({port}): no motors answered at any baud rate. Check servo power, jumpers and cabling.")
        for baudrate, ids in found.items():
            replies = ", ".join(f"{id_}: {reply}" for id_, reply in ids.items())
            print(f"{name} ({port}): {baudrate} baud, {{id: model}} {{{replies}}}")
            if any(reply.error for reply in ids.values()):
                print("  Motors reporting an error are in a protection alarm. Power-cycle the arm to clear it.")


def print_motors(arm: Arm) -> None:
    from so_101.check import TEMP_FAIL_C, TEMP_WARN_C, read_motors

    try:
        port = arm.resolve_port()
        readings = read_motors(arm)
    except PortNotFoundError:
        print(f"{arm.name}: board not found; plug in its USB cable\n")
        return
    except Exception as e:
        print(f"{arm.name} ({port}): {e}\n")
        return
    print(f"{arm.name} ({port})")
    print(f"  {'motor':14} {'temp':>6}  {'volts':>6}  {'load':>5}  {'position':>8}  {'in range':>8}  alarm")
    for r in readings:
        flag = " HOT" if r.temperature_c >= TEMP_FAIL_C else " warm" if r.temperature_c >= TEMP_WARN_C else ""
        in_range = "-" if r.range_pct is None else f"{r.range_pct:.0f}%"
        alarm = ", ".join(r.alarms) or "-"
        print(f"  {r.name:14} {r.temperature_c:>3} °C  {r.voltage_v:>4.1f} V  {r.load_pct:>4.0f}%  {r.position:>8}  "
              f"{in_range:>8}  {alarm}{flag}")
    print()


def cmd_motors(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    arms = [ns.arm] if ns.arm else ARMS
    try:
        while True:
            if ns.watch:
                print("\033[H\033[J", end="")  # clear the screen
                print(time.strftime("%H:%M:%S"), f"every {ns.watch:g} s, Ctrl+C to stop\n")
            for name in arms:
                print_motors(s.arm(name))
            if not ns.watch:
                return
            time.sleep(ns.watch)
    except KeyboardInterrupt:
        print()


def cmd_setup_motors(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    arm = s.arm(ns.arm)
    p = arm.prefix
    lerobot("lerobot-setup-motors", [f"--{p}.type={arm.lerobot_type}", f"--{p}.port={arm.resolve_port()}", *extra])


def cmd_calibrate(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    lerobot("lerobot-calibrate", [*s.arm(ns.arm).lerobot_args(), *extra])


def cmd_check(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    from so_101.check import live_test, run_checks

    if ns.live and ns.json:
        sys.exit("--live asks you to move the arm, so it can't be combined with --json.")
    arms, cameras, loop_fps = ARMS, list(s.cameras.values()), max((c.fps for c in s.cameras.values()), default=30)
    task = None
    if ns.task or ns.task is None and _has_default():
        task = get_task(ns.task)
        cameras = [s.camera(ours) for ours in task.cameras.values()]
        loop_fps = task.fps
        if not task.own:  # someone else's policy: `run` needs the follower and cameras only
            arms, loop_fps = ("follower",), None
    snapshots = s.outputs_dir / "check" / time.strftime("%Y%m%d-%H%M%S")
    report, _ = run_checks(
        s, arms=arms, cameras=cameras, task=task, loop_fps=loop_fps, keyboard=True,
        save_reference=ns.save_reference, snapshot_dir=snapshots,
    )
    live = ns.live and not report.failures and set(arms) == set(ARMS)
    if not ns.json:
        report.print()
    if live:
        before = len(report.results)
        live_test(report, s)
        if not ns.json:
            Report(report.results[before:]).print()
    if ns.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        if snapshots.exists():
            print(f"\nCamera snapshots: {snapshots}")
        if ns.live and not live:
            print("Skipped the live test: it needs both arms and no failures.")
        print(f"\n{report.summary()}")
    if report.failures:
        sys.exit(1)


def cmd_teleop(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    cameras, _ = preflight(s, "teleoperating", cameras=list(s.cameras.values()) if ns.cameras else [])
    args = [*s.follower.lerobot_args(), *s.leader.lerobot_args(), f"--fps={s.teleop.fps}"]
    if s.teleop.max_relative_target:
        args.append(f"--robot.max_relative_target={s.teleop.max_relative_target}")
    if ns.cameras:
        args += [cameras_arg([cam.lerobot_spec() for cam in cameras]), "--display_data=true"]
    lerobot("lerobot-teleoperate", [*args, *extra])


def cmd_cameras(s: settings.Settings, _: argparse.Namespace, extra: list[str]) -> None:
    devices = video_devices()
    if devices is not None:
        print("Cameras connected now, by OpenCV index:")
        for device in devices:
            mine = [cam.name for cam in s.cameras.values() if cam.usb_id and device.matches(cam.usb_id)]
            label = f"  <- {', '.join(mine)} (by usb_id)" if mine else ""
            print(f"  {device.index}: {device.name} [{device.model_id}]{label}")
        print("Commands find cameras with a usb_id by that id, so `index` in so101.toml only matters without one.\n")
    apply_camera_controls(list(s.cameras.values()))
    out = s.outputs_dir / "cameras"
    print(f"Saving a snapshot from every camera to {out}, each named after its index.\n")
    lerobot("lerobot-find-cameras", ["opencv", f"--output-dir={out}", *extra])


def get_task(name: str | None) -> tasks.Task:
    """The named task, or the default from tasks.toml when no name is given."""
    if name is None:
        name = tasks.default_name()
        print(f"Task: {name} (the default in tasks.toml)", file=sys.stderr)
    catalog = tasks.load()
    if name not in catalog:
        sys.exit(f"Unknown task {name!r}. Tasks: {', '.join(catalog)}. See `uv run so101 tasks`.")
    return catalog[name]


def _has_default() -> bool:
    try:
        tasks.default_name()
    except ConfigError:
        return False
    return True


def get_own_task(name: str | None) -> tasks.Task:
    task = get_task(name)
    if not task.own:
        sys.exit(f"{task.name} is someone else's pretrained policy. Add your own task to tasks.toml for this.")
    return task


def cmd_tasks(s: settings.Settings, __: argparse.Namespace, ___: list[str]) -> None:
    for task in tasks.load().values():
        print(f"{task.name}: {task.prompt}")
        if task.own:
            try:
                policy = task.policy
            except ConfigError:
                policy = "<you>/act_so101_" + task.name
            kept = data.kept_episodes(s, task)
            print(f"  yours:   {kept} of {task.episodes} demos kept, in {data.task_dir(s, task)}; trains into {policy}")
        else:
            print(f"  policy:  {task.policy} (ACT, {task.episodes} demos)")
            print(f"  look:    {task.viewer_url}")
        cameras = ", ".join(
            ours if ours == theirs else f"your {ours} as its {theirs}" for theirs, ours in task.cameras.items()
        )
        print(f"  cameras: {cameras}")
        print("  props:")
        for prop in task.props:
            print(f"    - {prop}")
        print("  setup:")
        for step in task.setup:
            print(f"    - {step}")
        if task.own:
            print(f"  steps:   uv run so101 record {task.name}, review, build, train, then run {task.name}")
        print(f"  run:     uv run so101 run {task.name}\n")


def cmd_record(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    from so_101.recording import record_args, session_manifest, task_cameras

    task = get_own_task(ns.task)
    if ns.dry_run:
        from so_101.dryrun import dry_run

        dry_run(s, task, display=not ns.no_display, replay=ns.replay, keep=ns.keep, train=not ns.no_train, extra=extra)
        return

    kept = data.kept_episodes(s, task)
    count = ns.episodes if ns.episodes is not None else task.episodes - kept
    if count <= 0:
        sys.exit(
            f"{task.name} already has {kept} demos (target {task.episodes}). "
            f"Add more with `uv run so101 record {task.name} --episodes 10`, or build and train it."
        )
    found, report = preflight(
        s, "recording", cameras=[s.camera(ours) for ours in task.cameras.values()], task=task,
        loop_fps=task.fps, keyboard=True,
    )
    cameras = task_cameras(s, task, found)
    session = data.new_session(s, task)
    manifest = session_manifest(s, task, cameras, report, ns.note or "")
    session.write_manifest(manifest)

    print(f"{task.name}: {kept} demos kept so far; recording {count} more into session {session.name}.")
    print("Keys: Right arrow or n = finish this episode early, Left arrow or r = redo it, Esc or q = stop.\n")
    args = record_args(
        s, task, cameras, root=session.dataset_root, repo_id=session.repo_id, episodes=count,
        episode_time_s=task.episode_time_s, reset_time_s=task.reset_time_s, display=not ns.no_display,
    )
    code = run_lerobot("lerobot-record", [*args, *extra])

    session = data.Session(session.task, session.name, session.path)  # re-read the dataset just written
    if session.episodes == 0:
        if any((session.dataset_root / "data").rglob("*.parquet")):
            # Frames were written but the episode index wasn't, e.g. LeRobot crashed. Keep it to inspect.
            print(f"\nWARNING: session {session.name} has recorded data but no readable episodes; left in "
                  f"{session.path}. `build` skips it.", file=sys.stderr)
        else:
            data.remove_session(session)
            print(f"\nNo episodes were saved, so session {session.name} was removed.")
    else:
        manifest.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), episodes=session.episodes, exit_code=code)
        session.write_manifest(manifest)
        print(f"\nSaved {session.episodes} episodes in {session.path}.")
        print(f"Next: `uv run so101 review {task.name}` to trim or delete episodes, then `uv run so101 train {task.name}`.")
    if code != 0:
        sys.exit(code)


def cmd_review(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    from so_101.review import review

    review(s, get_own_task(ns.task), ns.episode, ns.all, viewer=not ns.no_view)


def run_build(s: settings.Settings, task: tasks.Task) -> None:
    from so_101.build import build

    print(f"Building the training dataset for {task.name} into {data.train_root(s, task)}...")
    report = build(s, task)
    report.print("Training dataset")
    print(f"{report.summary()}\n")


def cmd_build(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    task = get_own_task(ns.task)
    if data.build_is_current(s, task) and not ns.force:
        print(f"{data.train_root(s, task)} is up to date with the sessions and edits. --force rebuilds it.")
        return
    run_build(s, task)


def cmd_backup(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    from huggingface_hub import HfApi

    task = get_own_task(ns.task)
    folder = data.task_dir(s, task)
    if not data.sessions(s, task):
        sys.exit(f"Nothing to back up: no sessions in {folder}.")
    repo = f"{tasks.hf_user()}/so101_{task.name}_raw"
    print(f"Uploading {folder} (sessions, edits) to the private dataset {repo}. Interrupted uploads resume.")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo, repo_type="dataset", folder_path=folder,
        ignore_patterns=["train/**", "train.tmp/**", "build.json", "**/.DS_Store", ".DS_Store"],
    )
    print(f"Backed up. To restore: uv run hf download {repo} --repo-type dataset --local-dir {folder}")


def push_dataset(root: Path, repo_id: str) -> None:
    """Upload a built dataset, replacing whatever an earlier build left in the repo."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=root, commit_message="so101 build",
        delete_patterns=["data/**", "meta/**", "videos/**"],
    )
    # LeRobot loads the revision tagged with its dataset format version; point it at this upload.
    try:
        api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except RevisionNotFoundError:
        pass
    api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")


# Flags each policy needs beyond its `[train.<policy>]` recipe in so101.toml.
POLICY_FLAGS = {
    "act": [],
    "pi05": [
        "--policy.pretrained_path=lerobot/pi05_base",
        # Train only the action model: on SO-101 this scored 130/200, against 4/110 with LoRA adapters.
        "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true",
        "--policy.gradient_checkpointing=true",
        # 32-bit took 30 s a step on an A100, against 4.6 s for this.
        "--policy.dtype=bfloat16",
        # The model has three camera slots and this kit has two cameras.
        "--policy.empty_cameras=1",
    ],
}
# pi0.5's tokenizer. Its licence is accepted per Hugging Face account, and approval isn't instant.
PI05_TOKENIZER = "google/paligemma-3b-pt-224"


def check_pi05_access() -> None:
    """Stop before uploading or paying if the job couldn't load pi0.5's gated tokenizer."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError

    try:
        HfApi().auth_check(PI05_TOKENIZER)
    except GatedRepoError:
        sys.exit(f"pi0.5 needs its tokenizer: request access at https://huggingface.co/{PI05_TOKENIZER}, "
                 "and train again once it's approved.")


def cmd_train(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    task = get_own_task(ns.task)
    if not data.sessions(s, task):
        sys.exit(f"No recorded demos for {task.name}. Run `uv run so101 record {task.name}` first.")
    if ns.policy not in s.train:
        sys.exit(f"No [train.{ns.policy}] section in {s.path}.")
    recipe = s.train[ns.policy]
    target = ns.target or recipe.job_target
    if target != "local" and target not in recipe.seconds_per_step:
        sys.exit(
            f"No measured {ns.policy} speed on {target}, so no safe timeout. Measured: "
            f"{', '.join(recipe.seconds_per_step)}. Add it to [train.{ns.policy}.seconds_per_step] in {s.path}."
        )
    if ns.policy == "pi05":
        if target == "local":
            sys.exit("pi0.5 can't train locally: it needs an NVIDIA GPU with the `lerobot[pi]` extra.")
        check_pi05_access()
    policy = task.trained_policy(ns.policy)  # needs a Hugging Face login; fail before building
    if not data.build_is_current(s, task):
        run_build(s, task)
    kept = data.kept_episodes(s, task)
    if kept < task.episodes:
        print(f"WARNING: {kept} of {task.episodes} demos; training anyway.", file=sys.stderr)
    steps = recipe.steps(kept)
    root = data.train_root(s, task)
    args = [
        f"--policy.type={ns.policy}",
        f"--policy.repo_id={policy}",
        "--policy.private=true",
        *POLICY_FLAGS[ns.policy],
        *([f"--policy.scheduler_decay_steps={steps}"] if ns.policy == "pi05" else []),
        f"--job_name={ns.policy}_so101_{task.name}",
        f"--steps={steps}",
        f"--batch_size={recipe.batch_size}",
        f"--num_workers={recipe.num_workers}",
        f"--save_freq={recipe.save_freq(steps)}",
        f"--job.target={target}",
    ]
    if target == "local":
        args += [f"--dataset.repo_id=local/so101_{task.name}", f"--dataset.root={root}", f"--policy.device={s.run.device}"]
    else:
        print(f"Uploading the training dataset to the private Hub dataset {task.dataset}...")
        push_dataset(root, task.dataset)
        timeout = recipe.job_timeout(steps, target)
        args += [f"--dataset.repo_id={task.dataset}", f"--job.timeout={timeout}",
                 "--save_checkpoint_to_hub=true",
                 # This Mac's decoder is sent along otherwise. That's PyAV when TorchCodec can't load here,
                 # and PyAV ran out of memory on the job's 192-core host.
                 "--dataset.video_backend=torchcodec"]
        print(
            f"Training {ns.policy} on {kept} demos for {steps:,} steps on Hugging Face Jobs ({target}, "
            f"about {recipe.expected_hours(steps, target):.1f} hours, stops after {timeout})."
        )
        print("Ctrl+C stops watching but leaves the job running; `uv run hf jobs ps` lists jobs.\n")
    lerobot("lerobot-train", [*args, *extra])


def cmd_run(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    task = get_task(ns.task) if ns.policy == "act" else get_own_task(ns.task)
    if ns.policy == "pi05" and not ns.remote:
        sys.exit(f"pi0.5 runs on a cloud GPU. Add --remote, after: uv run so101 server start {task.name} --policy pi05")
    server = remote.require_server(task.name, ns.policy, ns.revision or "main") if ns.remote else None

    # Name each camera after the checkpoint's camera and capture at the size it was trained on.
    sizes = task.image_sizes(ns.revision, ns.policy)
    if set(sizes) != set(task.cameras):
        repo = task.policy_for(ns.policy)
        sys.exit(f"{repo} expects cameras {sorted(sizes)}, but tasks.toml maps {sorted(task.cameras)}.")
    found, _ = preflight(
        s, "running the policy", arms=("follower",), cameras=[s.camera(ours) for ours in task.cameras.values()]
    )
    by_name = {cam.name: cam for cam in found}
    specs = []
    for theirs, ours in task.cameras.items():
        width, height = sizes[theirs]
        specs.append(by_name[ours].lerobot_spec(key=theirs, width=width, height=height))

    duration = s.run.duration_s if ns.duration is None else ns.duration
    episodes = s.run.episodes if ns.episodes is None else ns.episodes
    reset_time = task.reset_time_s if ns.reset_time is None else ns.reset_time
    name = f"{ns.policy}_{task.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = s.outputs_dir / "rollouts" / name
    dataset = run_dir / "dataset"
    log = run_dir / "run.log"
    args = [
        "--strategy.type=episodic",
        *(remote.client_args() if ns.remote else [f"--policy.path={task.download_policy(ns.revision)}"]),
        *s.follower.lerobot_args(),
        cameras_arg(specs),
        f"--robot.use_degrees={str(task.use_degrees).lower()}",
        f"--fps={task.fps}",
        *([] if ns.remote else [f"--device={s.run.device}"]),
        f"--dataset.repo_id=local/rollout_{name}",
        f"--dataset.root={dataset}",
        f"--dataset.single_task={task.prompt}",
        f"--dataset.fps={task.fps}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.episode_time_s={duration or UNTIL_STOPPED_S}",
        f"--dataset.reset_time_s={reset_time}",
        "--dataset.push_to_hub=false",
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
        "--dataset.rgb_encoder.preset=10",
    ]
    if s.run.max_relative_target:
        args.append(f"--robot.max_relative_target={s.run.max_relative_target}")
    if ns.display:
        args.append("--display_data=true")
    run_dir.mkdir(parents=True)
    print(f"\nRecording this run to {run_dir}\n  Video and joint data: {dataset}\n  Log: {log}", flush=True)
    if episodes > 1 and s.run.max_relative_target:
        print(
            "  Between attempts the arm can stop short of its start pose, because max_relative_target limits\n"
            "  LeRobot's 1-second return. Check its pose, or use --episodes 1, when an attempt must start at home.",
            flush=True,
        )
    print("  Keys: right arrow ends an attempt or reset, left arrow redoes it, Esc stops.\n", flush=True)
    code = 130
    try:
        with remote.tunnel(server.id) if server else nullcontext():
            code = run_lerobot("lerobot-rollout", [*args, *extra], log_path=log)
    except KeyboardInterrupt:
        pass
    finally:
        print_rollout_summary(run_dir, dataset, log)
        if server:
            print(f"The policy server keeps billing until: uv run so101 server stop (job {server.id})")
    sys.exit(code)


def print_rollout_summary(run_dir: Path, dataset: Path, log: Path) -> None:
    info = dataset / "meta" / "info.json"
    attempts = json.loads(info.read_text()).get("total_episodes", 0) if info.is_file() else 0
    print(f"\nThis run is in {run_dir}\n  Attempts saved: {attempts}", flush=True)
    for video in sorted((dataset / "videos").glob("*/*/*.mp4")):
        print(f"  Video: {video}", flush=True)
    print(f"  Video and joint data: {dataset}\n  Log: {log}", flush=True)


def cmd_server(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    if ns.action == "stop":
        remote.stop_server()
        return
    if ns.action == "status":
        server = remote.active_server()
        print(f"Running: {remote.describe(server)}." if server else "No policy server is running.")
        return
    task = get_task(ns.task) if ns.policy == "act" else get_own_task(ns.task)
    if ns.policy not in s.remote:
        sys.exit(f"No [remote.{ns.policy}] section in {s.path}.")
    recipe = s.remote[ns.policy]
    remote.start_server(
        repo=task.policy_for(ns.policy),
        task=task.name,
        policy_type=ns.policy,
        revision=ns.revision or "main",
        config=task.policy_config(ns.revision, ns.policy),
        target=ns.target or recipe.job_target,
        timeout=ns.timeout or recipe.job_timeout,
    )


def main() -> None:
    filter_stderr()
    parser = argparse.ArgumentParser(prog="so101", description="Run LeRobot commands for the SO-101 arms using so101.toml.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show config, resolved ports and calibration status").set_defaults(fn=cmd_info)
    sub.add_parser("env", help="print shell exports, e.g. eval \"$(uv run so101 env)\"").set_defaults(fn=cmd_env)

    p = sub.add_parser("find-port", help="detect an arm's board by unplugging it, and save its port")
    p.add_argument("arm", choices=ARMS)
    p.set_defaults(fn=cmd_find_port)

    p = sub.add_parser("scan", help="ping the motors on each bus at every baud rate (read-only)")
    p.add_argument("arm", nargs="?", choices=ARMS, help="default: both")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("motors", help="show each motor's temperature, voltage, load, position and alarms (read-only)")
    p.add_argument("arm", nargs="?", choices=ARMS, help="default: both")
    p.add_argument("--watch", type=float, metavar="SECONDS", help="refresh every SECONDS until Ctrl+C")
    p.set_defaults(fn=cmd_motors)

    p = sub.add_parser("setup-motors", help="set motor ids and baud rates, one motor at a time (writes EEPROM)")
    p.add_argument("arm", choices=ARMS)
    p.set_defaults(fn=cmd_setup_motors)

    p = sub.add_parser("calibrate", help="calibrate an arm")
    p.add_argument("arm", choices=ARMS)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("check", help="check arms, cameras and setup; read-only unless --live")
    p.add_argument("task", nargs="?", help="also check what this task needs (default: the default task, if set)")
    p.add_argument("--live", action="store_true", help="then teleoperate for 10 s to check every joint (moves the arm)")
    p.add_argument("--save-reference", action="store_true", help="save each camera's view as its reference image")
    p.add_argument("--json", action="store_true", help="print the results as JSON")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("teleop", help="drive the follower with the leader")
    p.add_argument("--cameras", action="store_true", help="also stream both cameras to a Rerun viewer")
    p.set_defaults(fn=cmd_teleop)

    sub.add_parser("cameras", help="list cameras and save a snapshot from each, to match indices").set_defaults(
        fn=cmd_cameras
    )
    sub.add_parser("tasks", help="list tasks in tasks.toml, with props and camera setup").set_defaults(fn=cmd_tasks)

    p = sub.add_parser("record", help="record demos of one of your tasks by teleoperating, as a new session")
    p.add_argument("task", nargs="?", help="default: `default` in tasks.toml")
    p.add_argument("--episodes", type=int, help="how many to record (default: up to `episodes` in tasks.toml)")
    p.add_argument("--no-display", action="store_true", help="don't open the Rerun camera viewer")
    p.add_argument("--note", help="a note saved with the session, e.g. 'lamp on, cylinder at spot 3'")
    p.add_argument("--dry-run", action="store_true", help="rehearse: record 2 throwaway episodes and check them")
    p.add_argument("--replay", action="store_true", help="dry run: replay a rehearsal episode on the follower")
    p.add_argument("--keep", action="store_true", help="dry run: keep the rehearsal dataset")
    p.add_argument("--no-train", action="store_true", help="dry run: skip the short training test")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("review", help="step through recorded episodes to trim or delete them (writes edits.toml)")
    p.add_argument("task", nargs="?", help="default: `default` in tasks.toml")
    p.add_argument("--episode", help="only a session (YYYYmmdd-HHMMSS) or one episode (SESSION/N)")
    p.add_argument("--all", action="store_true", help="include episodes already reviewed")
    p.add_argument("--no-view", action="store_true", help="don't open episodes in the Rerun viewer")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("build", help="make the training dataset from all sessions plus edits.toml")
    p.add_argument("task", nargs="?", help="default: `default` in tasks.toml")
    p.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("backup", help="upload a task's sessions and edits to a private Hub dataset")
    p.add_argument("task", nargs="?", help="default: `default` in tasks.toml")
    p.set_defaults(fn=cmd_backup)

    p = sub.add_parser("train", help="build if needed, then train a policy, on Hugging Face Jobs by default")
    p.add_argument("task", nargs="?", help="default: `default` in tasks.toml")
    p.add_argument("--policy", choices=sorted(POLICY_FLAGS), default="act", help="which model to train (default: act)")
    p.add_argument("--target", help="HF Jobs flavor, or `local` for this Mac (default: [train.<policy>] in so101.toml)")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("run", help="run a task's policy on the follower, recording to outputs/rollouts/")
    p.add_argument("task", nargs="?", help="default: `default` in tasks.toml")
    p.add_argument("--duration", type=float, help="seconds per attempt; 0 runs until Esc or Ctrl+C (default: [run] in so101.toml)")
    p.add_argument("--episodes", type=int, help="how many attempts to make (default: [run] in so101.toml)")
    p.add_argument("--reset-time", type=float, help="seconds to reset the scene between attempts (default: the task's reset_time_s)")
    p.add_argument("--display", action="store_true", help="stream cameras and joints to a Rerun viewer")
    p.add_argument("--revision", help="which version of the model: a Hub tag or commit (default: the newest)")
    p.add_argument("--policy", choices=sorted(POLICY_FLAGS), default="act", help="which model to run (default: act)")
    p.add_argument("--remote", action="store_true", help="run the model on the cloud GPU `so101 server start` started")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("server", help="start, stop or check the cloud GPU job that `run --remote` uses")
    p.add_argument("action", choices=["start", "stop", "status"])
    p.add_argument("task", nargs="?", help="for start, default: `default` in tasks.toml")
    p.add_argument("--policy", choices=sorted(POLICY_FLAGS), default="act", help="which model to serve (default: act)")
    p.add_argument("--revision", help="which version of the model: a Hub tag or commit (default: the newest)")
    p.add_argument("--target", help="HF Jobs flavor (default: [remote.<policy>] in so101.toml)")
    p.add_argument("--timeout", help="stop the job after this long, like 30m or 2h (default: [remote.<policy>] in so101.toml)")
    p.set_defaults(fn=cmd_server)

    ns, extra = parser.parse_known_args()
    try:
        ns.fn(settings.load(), ns, extra)
    except (PortNotFoundError, ConfigError) as e:
        sys.exit(str(e))
