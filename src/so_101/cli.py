"""`so101`: run LeRobot commands for the SO-101 arms with ports and ids from `so101.toml`.

On the commands that forward (`setup-motors`, `calibrate`, `teleop`, `cameras`, `record`, `train`,
`run` and `serve`), any arguments after the known ones are passed through to the LeRobot command, so
`uv run so101 teleop --teleop_time_s=30` works. Every other command refuses them.

This file parses the arguments and dispatches. The longer commands, `serve` and `runs`, and what
`run` and `serve` share, are in `so_101.commands`.
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

from serial.tools import list_ports

from so_101 import data, remote, rollout, settings, svtshim, tasks
from so_101.check import narrow_joints_warning, preflight
from so_101.commands.policy import policy_setup, policy_task, resolve_action_steps
from so_101.commands.runs import cmd_runs
from so_101.commands.serve import cmd_serve
from so_101.commands.sim import ACTIONS as SIM_ACTIONS
from so_101.commands.sim import cmd_sim
from so_101.console import filter_stderr, run_lerobot
from so_101.devices import apply_camera_controls, find_camera, video_devices
from so_101.report import Report
from so_101.scan import scan
from so_101.settings import ARMS, Arm, ConfigError, PortNotFoundError, cameras_arg
from so_101.tasks import get_task


UNTIL_STOPPED_S = 24 * 3600  # an attempt of `--duration 0`: LeRobot needs a length, and Esc or Ctrl+C ends it
# A record session longer than this gets a warning: a hard kill loses the whole session (see cmd_record).
LONG_SESSION = 20


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


def usb_ports() -> dict[str, str]:
    return {p.device: p.serial_number or "" for p in list_ports.comports() if "usb" in p.device.lower()}


def cmd_find_port(s: settings.Settings, ns: argparse.Namespace, __: list[str]) -> None:
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
    changed = settings.save_port(ns.arm, port, before[port])
    print(f"{ns.arm} port: {port} (serial number {before[port] or 'unknown'})")
    print(f"{'Saved to' if changed else 'Already saved in'} {s.path}. Plug the cable back in.")


def cmd_scan(s: settings.Settings, ns: argparse.Namespace, __: list[str]) -> None:
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


def cmd_pose(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    from so_101.check import read_pose

    if ns.save_rest:
        # Saved with the pose, so `serve` can tell when the follower was recalibrated since.
        fingerprint = s.follower.calibration_fingerprint()
        if fingerprint is None:
            sys.exit(f"No calibration file for the follower at {s.follower.calibration_file}, so no rest pose was "
                     "saved: the pose is converted with it. Run `uv run so101 calibrate follower` first.")
        try:
            rest = read_pose(s.follower)
        except ConnectionError as e:
            sys.exit(f"Couldn't read the follower's joints, is the arm's power on? {e}")
        path = settings.save_rest_pose(rest, s.path, calibration=fingerprint)
        for joint, steps in rest.items():
            print(f"{joint:14s} {steps:5d} steps")
        print(f"\nSaved this as rest_pose under [control] in {path}, in raw motor steps (0-4095), with "
              f"rest_pose_calibration = \"{fingerprint}\", the follower calibration file's fingerprint.")
        print("The control center's Reset drives the follower here at the end of every attempt. `serve` refuses "
              "to start once the follower is recalibrated, until you save the rest pose again.")
        return
    task = get_task(ns.task)
    try:
        pose = read_pose(s.follower, degrees=True)
    except ConnectionError as e:
        sys.exit(f"Couldn't read the follower's joints, is the arm's power on? {e}")
    start = data.demo_start_pose(s, task)
    if start is None:
        for joint, position in pose.items():
            print(f"{joint:14s} now {position:7.1f}")
        print(f"\n{task.name} has no build in {data.train_root(s, task)} to take the demos' start pose from, so a run "
              "starts from wherever the arm is.")
        print(f"Make one with: uv run so101 build {task.name}")
        return
    off = 0
    for key, demo in start.items():
        joint = key.removesuffix(".pos")
        difference = pose[joint] - demo
        close = abs(difference) <= rollout.TOLERANCE
        off += not close
        print(f"{joint:14s} now {pose[joint]:7.1f}  demos {demo:7.1f}  diff {difference:6.1f}  {'ok' if close else 'OFF'}")
    print(f"\nEvery joint is within {rollout.TOLERANCE:g} of the demos' start pose." if not off else
          f"\n{off} joint(s) more than {rollout.TOLERANCE:g} from the demos' start pose. `so101 run` and `serve` move them there first.")


def cmd_setup_motors(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    lerobot("lerobot-setup-motors", [*s.arm(ns.arm).lerobot_args(), *extra])


def cmd_calibrate(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    lerobot("so101-calibrate", [*s.arm(ns.arm).lerobot_args(), *extra])  # lerobot-calibrate, connecting the follower where it is


def cmd_check(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    from so_101.check import live_test, run_checks

    if ns.live and ns.json:
        sys.exit("--live asks you to move the arm, so it can't be combined with --json.")
    cameras, loop_fps = list(s.cameras.values()), max((c.fps for c in s.cameras.values()), default=30)
    task = None
    if ns.task or ns.task is None and "default" in tasks.read_file():
        task = get_task(ns.task)
        cameras = [s.camera(name) for name in task.cameras]
        loop_fps = task.fps
    snapshots = s.outputs_dir / "check" / time.strftime("%Y%m%d-%H%M%S")
    report, _ = run_checks(
        s, arms=ARMS, cameras=cameras, task=task, loop_fps=loop_fps, keyboard=True,
        snapshot_dir=snapshots, control="WARN", video=False,
    )
    live = ns.live and not report.failures
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
            print("Skipped the live test, because a check failed.")
        print(f"\n{report.summary()}")
    if report.failures:
        sys.exit(1)


def cmd_teleop(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    cameras, _ = preflight(s, "teleoperating", cameras=list(s.cameras.values()) if ns.cameras else [])
    args = [*s.follower_args(), *s.leader.lerobot_args(), f"--fps={s.teleop.fps}"]
    if ns.cameras:
        args += [cameras_arg([cam.lerobot_spec() for cam in cameras]), "--display_data=true"]
    lerobot("so101-teleoperate", [*args, *extra])  # lerobot-teleoperate, connecting the follower where it is


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


def cmd_tasks(s: settings.Settings, __: argparse.Namespace, ___: list[str]) -> None:
    for task in tasks.load().values():
        print(f"{task.name}: {task.prompt}")
        try:
            policy = task.policy_for("act")
        except ConfigError:
            policy = "<you>/act_so101_" + task.name
        kept = data.kept_episodes(s, task)
        print(f"  demos:   {kept} of {task.episodes} kept, in {data.task_dir(s, task)}; trains into {policy}")
        print(f"  cameras: {', '.join(task.cameras)}")
        print("  props:")
        for prop in task.props:
            print(f"    - {prop}")
        print("  setup:")
        for step in task.setup:
            print(f"    - {step}")
        print(f"  steps:   uv run so101 record {task.name}, review, build, train, then run {task.name}")
        print(f"  run:     uv run so101 run {task.name}\n")


def cmd_record(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    from so_101.recording import record_args, session_manifest

    if not ns.dry_run and (flags := [flag for flag, on in (("--replay", ns.replay), ("--keep", ns.keep),
                                                           ("--no-train", ns.no_train)) if on]):
        sys.exit(f"{', '.join(flags)} only apply to a rehearsal. Add --dry-run, or leave them out to record demos.")
    task = get_task(ns.task)
    if ns.dry_run:
        from so_101.dryrun import dry_run

        dry_run(s, task, display=not ns.no_display, replay=ns.replay, keep=ns.keep, train=not ns.no_train, extra=extra)
        return

    if ns.episodes is not None and ns.episodes < 1:
        sys.exit(f"--episodes is {ns.episodes}: name at least 1 episode to record, or leave it out to record "
                 "up to `episodes` in tasks.toml.")
    kept = data.kept_episodes(s, task)
    count = ns.episodes if ns.episodes is not None else task.episodes - kept
    if count <= 0:
        sys.exit(
            f"{task.name} already has {kept} demos (target {task.episodes}). "
            f"Add more with `uv run so101 record {task.name} --episodes 10`, or build and train it."
        )
    cameras, report = preflight(
        s, "recording", cameras=[s.camera(name) for name in task.cameras], task=task,
        loop_fps=task.fps, keyboard=True, video=True,
    )
    session = data.new_session(s, task)
    manifest = session_manifest(s, task, cameras, report, ns.note or "")
    session.write_manifest(manifest)

    print(f"{task.name}: {kept} demos kept so far; recording {count} more into session {session.name}.")
    if count > LONG_SESSION:
        print(f"WARNING: {count} episodes in one session. LeRobot finishes a session's files only when it stops, "
              "so a crash, power cut or kill -9 before then loses every episode in it; Esc, q and Ctrl+C are safe. "
              f"Shorter sessions risk less: `--episodes {LONG_SESSION}`, then record again.")
    print("Keys: Right arrow or n = finish this episode early, Left arrow or r = redo it, Esc or q = stop.\n")
    args = record_args(
        s, task, cameras, root=session.dataset_root, repo_id=session.repo_id, episodes=count, display=not ns.no_display,
    )
    code = run_lerobot("so101-record", [*args, *extra])  # lerobot-record, connecting the follower where it is

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
    if ns.runs:
        from so_101.review_runs import review_runs

        review_runs(s, get_task(ns.task), ns.episode, ns.all, viewer=not ns.no_view)
        return
    from so_101.review import review

    review(s, get_task(ns.task), ns.episode, ns.all, viewer=not ns.no_view)


# Never uploaded: the build is remade from the sessions and edits, Finder's .DS_Store is noise, and
# the rest are half-written files from a save in progress. `*` also matches `/` in these patterns.
BACKUP_IGNORE = ["train/**", "train.tmp/**", "build.json", "build.json.tmp", "**/.DS_Store", ".DS_Store",
                 "*.toml.tmp", "**/meta-copy/.building-*"]


def folder_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and f.name != ".DS_Store")


def backup_summary(s: settings.Settings, task: tasks.Task) -> str:
    """What `backup` uploads, for example: 3 sessions (2.1 GB), 2 runs (310 MB), edits.toml."""

    def size(n: int) -> str:
        return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"

    def count(items: list, noun: str) -> str:
        return f"{len(items)} {noun}{'' if len(items) == 1 else 's'} ({size(sum(folder_bytes(i.path) for i in items))})"

    parts = [count(data.sessions(s, task), "session"), count(data.runs(s, task), "run")]
    if data.edits_path(s, task).is_file():
        parts.append("edits.toml")
    return ", ".join(parts)


def cmd_backup(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    from huggingface_hub import HfApi

    task = get_task(ns.task)
    folder = data.task_dir(s, task)
    if not data.sessions(s, task) and not data.runs(s, task):
        sys.exit(f"Nothing to back up: no sessions or runs in {folder}.")
    repo = f"{tasks.hf_user()}/so101_{task.name}_raw"
    print(f"Uploading {folder}: {backup_summary(s, task)}.")
    print(f"They go to the private dataset {repo}, under the same sessions/ and runs/ paths. Interrupted uploads resume.")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    refuse_public(api, repo, "dataset", "backup")
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, ignore_patterns=BACKUP_IGNORE)
    print(f"Backed up. To restore: uv run hf download {repo} --repo-type dataset --local-dir {folder}")


def forwarded(extra: list[str], name: str) -> str | None:
    """The value a forwarded `--<name>=value` or `--<name> value` gives, the last if repeated, as
    LeRobot's parser takes it, or None."""
    value = None
    for i, arg in enumerate(extra):
        if arg.startswith(f"--{name}="):
            value = arg.split("=", 1)[1]
        elif arg == f"--{name}" and i + 1 < len(extra):
            value = extra[i + 1]
    return value


def refuse_public(api, repo: str, repo_type: str, command: str) -> None:
    """Exit unless the Hub repo `repo` is private or doesn't exist yet. `create_repo(private=True,
    exist_ok=True)` keeps an existing repo as it is, and someone may have made it public."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        private = api.repo_info(repo, repo_type=repo_type).private
    except RepositoryNotFoundError:
        return
    if not private:
        sys.exit(f"{repo} on the Hub is public, so nothing was uploaded. Make it private, or delete it, and run "
                 f"{command} again.")


def push_dataset(root: Path, repo_id: str) -> None:
    """Upload a built dataset, replacing whatever an earlier build left in the repo, which must be private."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    refuse_public(api, repo_id, "dataset", "train")
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


def default_base(policy: str) -> str | None:
    """The Hub model `policy` starts from without `--base`; None trains it from scratch."""
    prefix = "--policy.pretrained_path="
    return next((f.removeprefix(prefix) for f in POLICY_FLAGS[policy] if f.startswith(prefix)), None)


def parse_base(text: str) -> tuple[str, str | None]:
    """`--base REPO[@REVISION]`: a Hub model repo, and the tag, branch or commit to start from (None: main)."""
    repo, at, revision = text.partition("@")
    owner, slash, name = repo.partition("/")
    if not (owner and slash and name) or "/" in name or (at and not revision):
        raise ConfigError(
            f"--base {text!r} isn't a Hub model: give it as user/name, optionally with @ and a tag, branch or "
            "commit, such as gpahal/act_so101_block-container@100-demos."
        )
    return repo, revision or None


def policy_flags(policy: str, base: tuple[str, str | None] | None) -> list[str]:
    """POLICY_FLAGS for `policy`, starting from `base` instead of its default base model when given."""
    if base is None:
        return list(POLICY_FLAGS[policy])
    repo, revision = base
    rest = [f for f in POLICY_FLAGS[policy] if not f.startswith("--policy.pretrained_path=")]
    return [f"--policy.pretrained_path={repo}", *([f"--policy.pretrained_revision={revision}"] if revision else []), *rest]


def check_base(repo: str, revision: str | None, policy: str) -> None:
    """Stop before building, uploading or paying if the job couldn't start from this base model."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError, GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError,
    )

    name = f"{repo}@{revision}" if revision else repo
    try:
        config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    except GatedRepoError:  # before RepositoryNotFoundError, its parent class
        sys.exit(f"--base {name}: request access at https://huggingface.co/{repo}, and train again once it's approved.")
    except RepositoryNotFoundError:
        sys.exit(f"--base {name}: there's no such model on the Hub, or your login can't see it.")
    except RevisionNotFoundError:
        sys.exit(f"--base {name}: {repo} has no revision {revision!r}.")
    except EntryNotFoundError:
        sys.exit(
            f"--base {name} has no config.json at its top level, so it isn't a trained LeRobot model. "
            "Training checkpoint tags like 020000 keep theirs under checkpoints/."
        )
    if config.get("type") != policy:
        sys.exit(f"--base {name} is a model of type {config.get('type')}, not {policy}: train it with --policy {config.get('type')}.")


def check_pi05_access() -> None:
    """Stop before uploading or paying if the job couldn't load pi0.5's gated tokenizer."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError

    try:
        HfApi().auth_check(PI05_TOKENIZER)
    except GatedRepoError:
        sys.exit(f"pi0.5 needs its tokenizer: request access at https://huggingface.co/{PI05_TOKENIZER}, "
                 "and train again once it's approved.")


def run_build(s: settings.Settings, task: tasks.Task, sources: tuple[str, ...] = (), keep_labels: bool = False) -> None:
    from so_101.build import build

    svtshim.require()  # `build` encodes with SVT-AV1 in this process
    also = f", with {', '.join(sources)} from the runs" if sources else ""
    print(f"Building the training dataset for {task.name} into {data.train_root(s, task)}{also}...")
    report = build(s, task, sources, keep_labels)
    report.print("Training dataset")
    print(f"{report.summary()}\n")


def build_sources(ns: argparse.Namespace) -> tuple[str, ...]:
    from so_101.build import parse_sources

    return parse_sources(ns.with_)


def cmd_build(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    task = get_task(ns.task)
    sources = build_sources(ns)
    if data.build_is_current(s, task, sources, ns.keep_labels) and not ns.force:
        inputs = "sessions, edits and runs" if sources else "sessions and edits"
        print(f"{data.train_root(s, task)} is up to date with the {inputs}. --force rebuilds it.")
        return
    run_build(s, task, sources, ns.keep_labels)


def cmd_train(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    task = get_task(ns.task)
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
    base = parse_base(ns.base) if ns.base else None
    if base:
        check_base(*base, ns.policy)
    sources = build_sources(ns)
    policy = task.policy_for(ns.policy)  # needs a Hugging Face login; fail before building
    # A build with --keep-labels has the same demos and cuts, and training ignores its extra columns.
    if not (data.build_is_current(s, task, sources) or data.build_is_current(s, task, sources, keep_labels=True)):
        run_build(s, task, sources)
    kept = data.kept_episodes(s, task)
    if kept < task.episodes:
        print(f"WARNING: {kept} of {task.episodes} demos; training anyway.", file=sys.stderr)
    steps = recipe.steps(kept)
    root = data.train_root(s, task)
    args = [
        f"--policy.type={ns.policy}",
        f"--policy.repo_id={policy}",
        "--policy.private=true",
        *policy_flags(ns.policy, base),
        *([f"--policy.scheduler_decay_steps={steps}"] if ns.policy == "pi05" else []),
        f"--job_name={ns.policy}_so101_{task.name}",
        f"--steps={steps}",
        f"--batch_size={recipe.batch_size}",
        f"--num_workers={recipe.num_workers}",
        f"--save_freq={recipe.save_freq(steps)}",
        f"--job.target={target}",
    ]
    cuts = sum("run" in e for e in json.loads(data.build_manifest_path(s, task).read_text())["episodes"])
    also = f" and {cuts} run episode{'' if cuts == 1 else 's'} ({', '.join(sources)})" if sources else ""
    if base:
        also += f", starting from {ns.base}"
    elif start := default_base(ns.policy):
        also += f", starting from {start}"
    if target == "local":
        local, output = local_train_args(s, task, ns.policy, root)
        args += local
        if (forwarded(extra, "policy.push_to_hub") or "").lower() in ("true", "1", "yes"):
            from huggingface_hub import HfApi

            refuse_public(HfApi(), forwarded(extra, "policy.repo_id") or policy, "model", "train")
        print(f"Training {ns.policy} on {kept} demos{also} for {steps:,} steps on this computer ({s.run.device}). "
              f"Nothing is uploaded: the model is saved in {output}.\n"
              f"To run it: uv run so101 run {task.name} --checkpoint {output}")
    else:
        from huggingface_hub import HfApi

        # The job pushes the model to this repo as it trains, or to a forwarded --policy.repo_id;
        # LeRobot's private=true doesn't change an existing one.
        refuse_public(HfApi(), forwarded(extra, "policy.repo_id") or policy, "model", "train")
        print(f"Uploading the training dataset to the private Hub dataset {task.dataset}...")
        push_dataset(root, task.dataset)
        timeout = recipe.job_timeout(steps, target)
        args += [f"--dataset.repo_id={task.dataset}", f"--job.timeout={timeout}",
                 "--save_checkpoint_to_hub=true",
                 # This Mac's decoder is sent along otherwise. That's PyAV when TorchCodec can't load here,
                 # and PyAV ran out of memory on the job's 192-core host.
                 "--dataset.video_backend=torchcodec"]
        print(
            f"Training {ns.policy} on {kept} demos{also} for {steps:,} steps on Hugging Face Jobs ({target}, "
            f"about {recipe.expected_hours(steps, target):.1f} hours, stops after {timeout})."
        )
        print("Ctrl+C stops watching but leaves the job running; `uv run hf jobs ps` lists jobs.\n")
    lerobot("lerobot-train", [*args, *extra])


def local_train_args(s: settings.Settings, task, policy: str, root: Path) -> tuple[list[str], Path]:
    """lerobot-train's arguments for training on this Mac, and the folder the model is saved in. Nothing
    is uploaded: LeRobot pushes the model to --policy.repo_id when training ends unless told not to."""
    output = s.outputs_dir / "train" / f"{policy}_so101_{task.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    return [f"--dataset.repo_id=local/so101_{task.name}", f"--dataset.root={root}", f"--policy.device={s.run.device}",
            "--policy.push_to_hub=false", f"--output_dir={output}"], output


def cmd_run(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    task = policy_task(ns)
    setup = policy_setup(s, ns, task)

    duration = s.run.duration_s if ns.duration is None else ns.duration
    episodes = s.run.episodes if ns.episodes is None else ns.episodes
    reset_time = task.reset_time_s if ns.reset_time is None else ns.reset_time
    name = f"{ns.policy}_{task.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = s.outputs_dir / "rollouts" / name
    dataset = run_dir / "dataset"
    log = run_dir / "run.log"
    args = [
        *setup.args,
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
    if ns.display:
        args.append("--display_data=true")
    run_dir.mkdir(parents=True)
    print(f"\nRecording this run to {run_dir}\n  Video and joint data: {dataset}\n  Log: {log}", flush=True)
    # so101-rollout drives to the demos' start pose before each attempt, when the task has a build.
    if start_pose := data.demo_start_pose(s, task):
        os.environ[rollout.START_POSE_ENV] = json.dumps(start_pose)
        print(f"  Each attempt starts at the demos' start pose: {data.describe_pose(start_pose)}", flush=True)
    else:
        print("  Each attempt starts at the pose the arm is in now.", flush=True)
    print(f"  {setup.steps_line(s)}", flush=True)
    print("  Keys: right arrow ends an attempt or reset, left arrow redoes it, Esc stops.\n", flush=True)
    setup.run("so101-rollout", [*args, *extra], log, partial(print_rollout_summary, run_dir, dataset, log))


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
        servers = remote.active_servers()
        for server in servers:
            print(f"Running: {remote.describe(server)}.")
        if not servers:
            print("No policy server is running.")
        return
    task = get_task(ns.task)
    if ns.policy not in s.remote:
        sys.exit(f"No [remote.{ns.policy}] section in {s.path}.")
    recipe = s.remote[ns.policy]
    remote.refuse_action_steps(ns.policy, ns.action_steps)
    config = task.policy_config(ns.revision, ns.policy)
    remote.start_server(
        repo=task.policy_for(ns.policy),
        task=task.name,
        policy_type=ns.policy,
        revision=ns.revision or "main",
        image_sizes=tasks.image_sizes(config),
        fps=task.fps,
        action_steps=resolve_action_steps(ns.action_steps, config),
        target=ns.target or recipe.job_target,
        timeout=ns.timeout or recipe.job_timeout,
    )


# The commands that encode video, in this process or a child: they restart with SVT-AV1's crash fix.
VIDEO_COMMANDS = {"record", "run", "serve", "build", "train"}


def add_task(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", nargs="?", help="default: `default` in tasks.toml")


def add_policy_flags(p: argparse.ArgumentParser, verb: str, remote_flag: bool) -> None:
    """Which model a command drives or serves, and how; `verb` goes in --policy's help."""
    p.add_argument("--action-steps", type=int, help="actions to carry out before predicting again (default: the model's trained value; "
                   "not for pi0.5 on the cloud GPU, which uses real-time chunking)")
    p.add_argument("--revision", help="which version of the model: a Hub tag or commit (default: the newest)")
    p.add_argument("--policy", choices=sorted(POLICY_FLAGS), default="act", help=f"which model to {verb} (default: act)")
    if remote_flag:
        p.add_argument("--remote", action="store_true", help="run the model on the cloud GPU `so101 server start` started")
        p.add_argument("--checkpoint", metavar="DIR",
                       help="a model `train --target local` saved on this Mac: the folder it printed (default: the Hub repo)")


def main() -> None:
    command = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
    # `sim nudge` records too; the other `sim` actions encode nothing.
    if command in VIDEO_COMMANDS or sys.argv[1:3] == ["sim", "nudge"]:
        svtshim.ensure()  # first: it restarts this process, which must happen before stderr is filtered
    filter_stderr()
    parser = argparse.ArgumentParser(
        prog="so101", description="Run LeRobot commands for the SO-101 arms using so101.toml.", allow_abbrev=False,
    )
    parser.set_defaults(forwards=False)
    sub = parser.add_subparsers(
        dest="command", required=True, parser_class=partial(argparse.ArgumentParser, allow_abbrev=False),
    )

    sub.add_parser("info", help="show config, resolved ports, camera indices and calibration status").set_defaults(fn=cmd_info)

    p = sub.add_parser("find-port", help="detect an arm's board by unplugging it, and save its port")
    p.add_argument("arm", choices=ARMS)
    p.set_defaults(fn=cmd_find_port)

    p = sub.add_parser("scan", help="ping the motors on each bus at every baud rate (read-only)")
    p.add_argument("arm", nargs="?", choices=ARMS, help="default: both")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("sim", help="build a simulated twin of a task from its demos, and try policies in it")
    p.add_argument("action", choices=SIM_ACTIONS)
    add_task(p)
    p.add_argument("--policy", choices=sorted(POLICY_FLAGS), default="act", help="for check and eval: which model (default: act, the only one that runs on this Mac)")
    p.add_argument("--revision", help="for check and eval: which version of the model: a Hub tag or commit (default: the newest)")
    p.add_argument("--checkpoint", metavar="DIR", help="for check and eval: a model `train --target local` saved on this Mac (default: the Hub repo)")
    p.add_argument("--episodes", type=int, help="for replay: the first so many demos (default: all). For nudge: pushes to record (default 3). For eval: attempts (default 10)")
    p.add_argument("--action-steps", type=int, help="for eval: actions to carry out before predicting again (default: the model's trained value)")
    p.add_argument("--limit", type=float, help="for replay and eval: the per-step movement limit (default: max_relative_target in so101.toml)")
    p.add_argument("--refit-cameras", action="store_true", help="for build: fit the cameras and the joint offsets again, after moving a camera or recalibrating")
    p.add_argument("--refit-motors", action="store_true", help="for build: fit the motors to the real follower again")
    p.set_defaults(fn=cmd_sim, forwards=True)  # eval passes the rest to lerobot-eval, nudge to lerobot-record

    p = sub.add_parser("pose", help="compare the follower's current pose with the demos' start pose, or save it as the rest pose (reads the motors only)")
    add_task(p)
    p.add_argument(
        "--save-rest", action="store_true",
        help="save the follower's current pose as rest_pose in so101.toml instead of comparing; "
        "only reads the motors, so move the arm there by hand first",
    )
    p.set_defaults(fn=cmd_pose)

    p = sub.add_parser("motors", help="show each motor's temperature, voltage, load, position and alarms (read-only)")
    p.add_argument("arm", nargs="?", choices=ARMS, help="default: both")
    p.add_argument("--watch", type=float, metavar="SECONDS", help="refresh every SECONDS until Ctrl+C")
    p.set_defaults(fn=cmd_motors)

    p = sub.add_parser("setup-motors", help="set motor ids and baud rates, one motor at a time (writes EEPROM)")
    p.add_argument("arm", choices=ARMS)
    p.set_defaults(fn=cmd_setup_motors, forwards=True)

    p = sub.add_parser("calibrate", help="calibrate an arm")
    p.add_argument("arm", choices=ARMS)
    p.set_defaults(fn=cmd_calibrate, forwards=True)

    p = sub.add_parser("check", help="check arms, cameras and setup; moves nothing unless --live, and writes only the cameras' UVC controls")
    p.add_argument("task", nargs="?", help="also check what this task needs (default: the default task, if set)")
    p.add_argument("--live", action="store_true", help="then teleoperate for 10 s to check every joint (moves the arm)")
    p.add_argument("--json", action="store_true", help="print the results as JSON")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("teleop", help="drive the follower with the leader")
    p.add_argument("--cameras", action="store_true", help="also stream both cameras to a Rerun viewer")
    p.set_defaults(fn=cmd_teleop, forwards=True)

    sub.add_parser("cameras", help="list cameras and save a snapshot from each, to match indices").set_defaults(
        fn=cmd_cameras, forwards=True
    )
    sub.add_parser("tasks", help="list tasks in tasks.toml, with props, camera setup and demo counts").set_defaults(fn=cmd_tasks)

    p = sub.add_parser("record", help="record demos of one of your tasks by teleoperating, as a new session")
    add_task(p)
    p.add_argument("--episodes", type=int, help="how many to record (default: up to `episodes` in tasks.toml)")
    p.add_argument("--no-display", action="store_true", help="don't open the Rerun camera viewer")
    p.add_argument("--note", help="a note saved with the session, e.g. 'lamp on, block at spot 3'")
    p.add_argument("--dry-run", action="store_true", help="rehearse: record 2 throwaway episodes and check them")
    p.add_argument("--replay", action="store_true", help="dry run: replay a rehearsal episode on the follower")
    p.add_argument("--keep", action="store_true", help="dry run: keep the rehearsal dataset")
    p.add_argument("--no-train", action="store_true", help="dry run: skip the short training test")
    p.set_defaults(fn=cmd_record, forwards=True)

    p = sub.add_parser("review", help="step through recorded episodes to trim or delete them (writes edits.toml); "
                                      "--runs reviews serve runs instead (writes each run's labels.toml)")
    add_task(p)
    p.add_argument("--runs", action="store_true",
                   help="review the interruptions and outcomes of the task's serve runs instead of its demos")
    p.add_argument("--episode", help="only a session (YYYYmmdd-HHMMSS) or one episode (SESSION/N); "
                                     "with --runs, a run (YYYYmmdd-HHMMSS) or one attempt (RUN/N)")
    p.add_argument("--all", action="store_true", help="include episodes, or with --runs attempts, already reviewed")
    p.add_argument("--no-view", action="store_true", help="don't open episodes in the Rerun viewer")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("build", help="make the training dataset from all sessions plus edits.toml")
    add_task(p)
    p.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    p.add_argument("--with", dest="with_", metavar="SOURCES",
                   help="also train on serve runs: corrections, human-attempts, policy-successes, comma-separated "
                        "(default: the demos only)")
    p.add_argument("--keep-labels", action="store_true",
                   help="keep intervention and control.phase, and add control.label and control.attempt_success")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("runs", help="summarise a task's serve runs and export failure-fix pairs to outputs/ (read-only on data/)")
    add_task(p)
    p.set_defaults(fn=cmd_runs)

    p = sub.add_parser("backup", help="upload a task's sessions, edits and runs to a private Hub dataset")
    add_task(p)
    p.set_defaults(fn=cmd_backup)

    p = sub.add_parser("train", help="build if needed, then train a policy, on Hugging Face Jobs by default")
    add_task(p)
    p.add_argument("--policy", choices=sorted(POLICY_FLAGS), default="act", help="which model to train (default: act)")
    p.add_argument("--target", help="HF Jobs flavor, or `local` for this Mac (default: [train.<policy>] in so101.toml)")
    p.add_argument("--with", dest="with_", metavar="SOURCES",
                   help="build with these serve-run sources too, as `so101 build --with` (default: the demos only)")
    p.add_argument("--base", metavar="REPO[@REVISION]",
                   help="start from this Hub model instead of the default (act: none, from scratch; pi05: "
                        f"{default_base('pi05')}), such as a model you trained, pinned with @tag")
    p.set_defaults(fn=cmd_train, forwards=True)

    p = sub.add_parser("run", help="run a task's policy on the follower, recording to outputs/rollouts/")
    add_task(p)
    p.add_argument("--duration", type=float, help="seconds per attempt; 0 runs until Esc or Ctrl+C (default: [run] in so101.toml)")
    p.add_argument("--episodes", type=int, help="how many attempts to make (default: [run] in so101.toml)")
    p.add_argument("--reset-time", type=float, help="seconds to reset the scene between attempts (default: the task's reset_time_s)")
    p.add_argument("--display", action="store_true", help="stream cameras and joints to a Rerun viewer")
    add_policy_flags(p, "run", remote_flag=True)
    p.set_defaults(fn=cmd_run, forwards=True)

    p = sub.add_parser("serve", help="supervise a task's policy from a control page over Tailscale, or the terminal: start, pause, rewind, take over with the leader and end attempts")
    add_task(p)
    add_policy_flags(p, "run", remote_flag=True)
    p.add_argument("--operator", help="your name, recorded on every event your actions cause and shown on the page (default: none)")
    p.set_defaults(fn=cmd_serve, forwards=True)

    p = sub.add_parser("server", help="start, stop or check the cloud GPU job that `run --remote` and `serve --remote` use")
    p.add_argument("action", choices=["start", "stop", "status"])
    p.add_argument("task", nargs="?", help="for start, default: `default` in tasks.toml")
    add_policy_flags(p, "serve", remote_flag=False)
    p.add_argument("--target", help="HF Jobs flavor (default: [remote.<policy>] in so101.toml)")
    p.add_argument("--timeout", help="stop the job after this long, like 30m or 2h (default: [remote.<policy>] in so101.toml)")
    p.set_defaults(fn=cmd_server)

    ns, extra = parser.parse_known_args()
    if extra and not ns.forwards:
        parser.error(f"unrecognized arguments: {' '.join(extra)} (`so101 {ns.command}` passes nothing on to LeRobot)")
    try:
        s = settings.load()
        # LeRobot takes the last of a repeated flag, so a forwarded cap replaces so101.toml's: check it the same way.
        if (cap := forwarded(extra, "robot.max_relative_target")) is not None:
            settings.step_cap(cap, "--robot.max_relative_target")
        ns.fn(s, ns, extra)
    except (PortNotFoundError, ConfigError) as e:
        sys.exit(str(e))
