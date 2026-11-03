"""Health checks for the arms, cameras and setup, run by `so101 check` and before record, teleop and run.

Everything here is read-only: arms are read over their serial bus without LeRobot's `connect()`,
which would start calibrating on a mismatch and switch motor torque on. Only `live_test` moves the
follower, and only when asked for with `so101 check --live`.
"""

import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from serial.tools import list_ports

from so_101 import data
from so_101.devices import CameraLookup, find_cameras, get_control, set_controls, uvc_util
from so_101.report import Report
from so_101.settings import ARMS, Arm, ArmName, Camera, ConfigError, PortNotFoundError, Settings

if TYPE_CHECKING:
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.motors.feetech import FeetechMotorsBus

    from so_101.tasks import Task

STS3215_MODEL = 777
TEMP_WARN_C = 50
TEMP_FAIL_C = 60
VOLTAGE_SPREAD_WARN_V = 1.0
POSITION_SLACK_TICKS = 50  # how far outside its calibrated range a joint may read before it's flagged
CAMERA_SAMPLE_S = 2.5
LOOP_SAMPLE_S = 3.0


def narrow_joints_warning(arm: Arm) -> str:
    """What's wrong when a joint's calibrated range is tiny, or "" if nothing is."""
    joints = arm.narrow_joints()
    if not joints:
        return ""
    return (
        f"{', '.join(joints)} barely moved during calibration, so the {arm.name} can hardly move "
        f"{'it' if len(joints) == 1 else 'them'}"
    )


# --- System ---------------------------------------------------------------------------------------


def check_system(report: Report, s: Settings, cameras: list[Camera], keyboard: bool) -> None:
    from so_101.tasks import hf_user

    try:
        report.ok("hugging face", f"logged in as {hf_user()}")
    except ConfigError:
        report.warn("hugging face", "not logged in: needed to train and back up", fix="uv run hf auth login")
    except Exception as e:  # offline, or the Hub is down
        report.warn("hugging face", f"can't reach the Hub: {e}")

    existing = next(p for p in [s.data_dir, *s.data_dir.parents] if p.exists())
    free_gb = shutil.disk_usage(existing).free / 1e9
    if free_gb < 1:
        report.fail("disk", f"{free_gb:.1f} GB free for {s.data_dir}", fix="free up disk space")
    elif free_gb < 5:
        report.warn("disk", f"{free_gb:.1f} GB free for {s.data_dir}", fix="free up disk space")
    else:
        report.ok("disk", f"{free_gb:.0f} GB free for {s.data_dir}")

    if any(cam.controls for cam in cameras) and uvc_util() is None:
        report.warn("uvc-util", "not installed, so camera controls (exposure, anti-flicker) can't be set",
                    fix="scripts/install-uvc-util.sh")

    if keyboard and sys.platform == "darwin":
        try:
            from ApplicationServices import AXIsProcessTrusted

            trusted = bool(AXIsProcessTrusted())
        except ImportError:
            trusted = None
        if trusted is False:
            report.warn(
                "keyboard",
                "no Accessibility permission: the recording keys only work in this terminal, while it's focused",
                fix="System Settings > Privacy & Security > Accessibility: allow your terminal app",
            )


# --- Arms -----------------------------------------------------------------------------------------


def _open_bus(arm: Arm, port: str) -> "FeetechMotorsBus":
    """The arm's motor bus as LeRobot defines it, with the calibration file loaded, opened read-only."""
    from lerobot.motors.feetech import FeetechMotorsBus

    if arm.name == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        device = SO101Follower(SO101FollowerConfig(port=port, id=arm.id, calibration_dir=arm.calibration_dir))
    else:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        device = SO101Leader(SO101LeaderConfig(port=port, id=arm.id, calibration_dir=arm.calibration_dir))
    # A separate bus, not device.bus: when `device` is garbage-collected its __del__ disconnects its
    # bus if open, which writes torque-off to every motor and closes the port.
    bus = FeetechMotorsBus(port=port, motors=device.bus.motors, calibration=device.calibration)
    bus._connect(handshake=False)  # open the port without LeRobot's checks, which raise on the first problem
    bus.set_baudrate(bus.default_baudrate)
    bus.set_timeout()
    return bus


def close_bus(bus: "FeetechMotorsBus") -> None:
    bus.port_handler.closePort()  # not bus.disconnect(): that writes to the motors to disable torque


def check_arm(report: Report, arm: Arm) -> "FeetechMotorsBus | None":
    """Check one arm. Returns its open bus if the motors answered, for the loop timing test."""
    from so_101.scan import ping

    group = arm.name
    fix = f"uv run so101 calibrate {arm.name}"
    if not arm.calibration_file.is_file():
        report.fail(group, f"not calibrated (no {arm.calibration_file})", fix=fix)
    elif warning := narrow_joints_warning(arm):
        report.fail(group, warning, fix=f"{fix}, and move every joint through its full range")
    try:
        port = arm.resolve_port()
    except PortNotFoundError:
        report.fail(group, "board not found", fix=f"plug in its USB cable and power; `uv run so101 find-port {arm.name}`")
        return None
    serial = next((p.serial_number for p in list_ports.comports() if p.device == port), None)
    if arm.serial_number and serial and serial != arm.serial_number:
        report.fail(
            group,
            f"{port} is another board (serial number {serial}, expected {arm.serial_number})",
            fix="run `uv run so101 find-port follower` and `uv run so101 find-port leader` to save both ports again",
        )
        return None
    try:
        bus = _open_bus(arm, port)
    except Exception as e:
        report.fail(group, f"can't open {port}: {e}", fix="close other programs using the arm, then replug it")
        return None

    try:
        replies = {name: ping(bus, motor.id) for name, motor in bus.motors.items()}
        missing = [f"{name} (id {bus.motors[name].id})" for name, reply in replies.items() if reply is None]
        if missing:
            report.fail(
                group,
                f"no answer from {', '.join(missing)}",
                fix=f"check servo power and the cable after the last motor that answers; `uv run so101 scan {arm.name}`",
            )
            close_bus(bus)
            return None
        wrong = [f"{name} (model {reply.model})" for name, reply in replies.items() if reply.model != STS3215_MODEL]
        if wrong:
            report.fail(group, f"not STS3215 servos: {', '.join(wrong)}",
                        fix="check the motor ids with `uv run so101 scan`")
        alarms = {name: reply.errors() for name, reply in replies.items() if reply.error}
        if alarms:
            listed = ", ".join(f"{name} ({', '.join(errs)})" for name, errs in alarms.items())
            report.fail(group, f"motors in a protection alarm: {listed}",
                        fix="power-cycle the arm; if it returns, look for a jammed joint or a weak power supply")

        volts = {m: v / 10 for m, v in bus.sync_read("Present_Voltage", normalize=False, num_retry=2).items()}
        temps = bus.sync_read("Present_Temperature", normalize=False, num_retry=2)
        moved = "" if port == arm.port else f" (saved port {arm.port} is gone; found by serial number)"
        report.ok(
            group,
            f"{port}{moved}: 6/6 motors, {min(volts.values()):.1f}-{max(volts.values()):.1f} V, "
            f"up to {max(temps.values())} °C",
        )
        if max(volts.values()) - min(volts.values()) > VOLTAGE_SPREAD_WARN_V:
            listed = ", ".join(f"{m} {v:.1f} V" for m, v in volts.items())
            report.warn(group, f"supply voltage differs between motors: {listed}",
                        fix="check the power connections along the servo chain")
        hot = {m: t for m, t in temps.items() if t >= TEMP_WARN_C}
        if hot:
            status = report.fail if max(hot.values()) >= TEMP_FAIL_C else report.warn
            status(group, "hot motors: " + ", ".join(f"{m} {t} °C" for m, t in hot.items()),
                   fix="let the arm cool down; check for a joint straining against something")

        check_calibration(report, arm, bus)
    except Exception as e:
        report.fail(group, f"reading the motors failed: {e}", fix=f"replug the arm; `uv run so101 scan {arm.name}`")
        close_bus(bus)
        return None
    return bus


def check_calibration(report: Report, arm: Arm, bus: "FeetechMotorsBus") -> None:
    """Compare the calibration file with what's stored in the motors. File-only checks are in `check_arm`."""
    group = arm.name
    fix = f"uv run so101 calibrate {arm.name}"
    if not arm.calibration_file.is_file():
        return
    # is_calibrated reads three registers per motor with no retries; one garbled reply shouldn't fail the arm.
    for attempt in range(3):
        try:
            matches = bus.is_calibrated
            break
        except ConnectionError:
            if attempt == 2:
                raise
    if not matches:
        report.fail(
            group,
            "the calibration file doesn't match the calibration stored in the motors "
            "(a motor was replaced, or the arm was calibrated elsewhere)",
            fix=fix,
        )
        return
    positions = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    outside = [
        f"{m} at {p} (range {cal.range_min}-{cal.range_max})"
        for m, p in positions.items()
        if not (cal := bus.calibration[m]).range_min - POSITION_SLACK_TICKS <= p <= cal.range_max + POSITION_SLACK_TICKS
    ]
    if outside:
        report.warn(group, "joints outside their calibrated range: " + ", ".join(outside),
                    fix=f"a horn or gear may have slipped; {fix}")
    else:
        report.ok(group, f"calibration matches the motors ({arm.calibration_file.name})")


# --- Cameras --------------------------------------------------------------------------------------


@dataclass
class CameraSample:
    unique: int = 0
    elapsed: float = 0.0
    frozen: bool = False
    error: str = ""
    profiles: list[np.ndarray] = field(default_factory=list)  # row brightness of new frames
    last: np.ndarray | None = None  # last frame, RGB


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def sample_cameras(cams: dict[str, "OpenCVCamera"], seconds: float) -> dict[str, CameraSample]:
    """Poll all cameras together, counting frames that differ from the one before."""
    samples = {name: CameraSample() for name in cams}
    fingerprints: dict[str, bytes] = {}
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        for name, cam in cams.items():
            sample = samples[name]
            if sample.frozen or sample.error:
                continue
            try:
                frame = cam.read_latest()
            except TimeoutError:
                sample.frozen = True
                continue
            except Exception as e:
                sample.error = str(e)
                continue
            fingerprint = frame[::8, ::8].tobytes()
            if fingerprint != fingerprints.get(name):
                fingerprints[name] = fingerprint
                sample.unique += 1
                sample.last = frame
                if len(sample.profiles) < 20:
                    sample.profiles.append(_gray(frame).mean(axis=1))
        time.sleep(0.003)
    for sample in samples.values():
        sample.elapsed = time.perf_counter() - start
    return samples


def flicker_percent(profiles: list[np.ndarray]) -> float | None:
    """How much the brightness of image rows changes between frames, in % of mean brightness.

    Mains-powered lights make rolling bands that shift from frame to frame; a still scene under
    steady light scores well under 1%.
    """
    if len(profiles) < 3:
        return None
    changes = [
        float(np.sqrt(np.mean((b / max(b.mean(), 1) - a / max(a.mean(), 1)) ** 2)))
        for a, b in zip(profiles, profiles[1:], strict=False)
    ]
    return 100 * float(np.median(changes))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two RGB images shrunk to 32x24 grey: 1 for the same view, about 0 for unrelated."""
    import cv2

    def small(img: np.ndarray) -> np.ndarray:
        g = cv2.resize(_gray(img), (32, 24), interpolation=cv2.INTER_AREA)
        return (g - g.mean()) / (g.std() + 1e-6)

    return float(np.mean(small(a) * small(b)))


def reference_dir(s: Settings) -> Path:
    return s.outputs_dir / "check" / "reference"


def check_cameras(
    report: Report,
    s: Settings,
    cameras: list[Camera],
    save_reference: bool,
    snapshot_dir: Path | None,
    opened: dict[str, "OpenCVCamera"],
) -> list[CameraLookup]:
    """Find, open and sample each camera. Open cameras are added to `opened`, which the caller closes."""
    import cv2
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    lookups = find_cameras(cameras)
    controls_set: set[str] = set()
    for lookup in lookups:
        cam = lookup.camera
        group = f"{cam.name} camera"
        if lookup.problem:
            report.fail(group, lookup.problem, fix="plug it in; `uv run so101 cameras` lists what's connected")
            continue
        # Set controls before opening the camera, as recording does; they're read back once it's open.
        if cam.usb_id and cam.controls:
            if error := set_controls(cam):
                report.warn(group, f"couldn't set controls {cam.controls}: {error}")
            else:
                controls_set.add(cam.name)
        try:
            device = OpenCVCamera(OpenCVCameraConfig(
                index_or_path=cam.index, fps=cam.fps, width=cam.width, height=cam.height,
                rotation=Cv2Rotation(cam.rotation),
            ))
            device.connect()
            opened[cam.name] = device
        except Exception as e:
            report.fail(
                group,
                f"can't open index {cam.index} at {cam.width}x{cam.height}, {cam.fps} fps: {e}",
                fix="allow camera access for your terminal in System Settings > Privacy & Security > Camera; "
                "`uv run so101 cameras` shows which index is which",
            )

    samples = sample_cameras(opened, CAMERA_SAMPLE_S) if opened else {}
    for lookup in lookups:
        cam = lookup.camera
        if cam.name not in samples:
            continue
        group = f"{cam.name} camera"
        if cam.name in controls_set:
            wrong = {k: got for k, v in cam.controls.items() if (got := get_control(cam, k)) is not None and got != v}
            if wrong:
                report.warn(group, f"controls didn't stick once the camera was open: {wrong} (wanted {cam.controls})",
                            fix="replug the camera and run the check again")
        sample = samples[cam.name]
        if sample.error:
            report.fail(group, f"reading frames failed: {sample.error}", fix="replug the camera")
            continue
        if sample.last is None or sample.unique < 2:
            report.fail(group, "the picture is frozen", fix="replug the camera")
            continue
        frame = sample.last
        gray = _gray(frame)
        rate = sample.unique / sample.elapsed
        where = lookup.note or f"index {cam.index}"
        flicker = flicker_percent(sample.profiles)

        brightness, contrast = float(gray.mean()), float(gray.std())
        if contrast < 4:
            report.fail(group, f"the picture is blank (brightness {brightness:.0f}, no detail)",
                        fix="check the lens cap and the light")
        elif brightness < 15:
            report.fail(group, f"the picture is black (brightness {brightness:.0f}/255)",
                        fix="check the lens and the light")
        elif brightness > 245:
            report.fail(group, f"the picture is white (brightness {brightness:.0f}/255)", fix="lower its exposure")
        else:
            flicker_text = "" if flicker is None else f", flicker {flicker:.1f}%"
            report.ok(group, f"{where}: {frame.shape[1]}x{frame.shape[0]}, {rate:.1f} new frames/s, "
                             f"brightness {brightness:.0f}/255{flicker_text}")
            if not 50 <= brightness <= 210:
                report.warn(group, f"{'dark' if brightness < 50 else 'bright'} picture (brightness {brightness:.0f}/255)",
                            fix="adjust the light, or the camera's exposure in so101.toml")

        if rate < 0.9 * cam.fps:
            repeated = max(0.0, 1 - rate / cam.fps)
            report.warn(
                group,
                f"delivers {rate:.1f} new frames/s against the {cam.fps} configured: about {repeated:.0%} of "
                "recorded frames will repeat the one before",
                fix="more light (auto exposure slows cameras down in dim light), or a shorter fixed exposure",
            )
        if flicker is not None and flicker > 2.0:
            report.warn(group, f"rolling bands or flicker ({flicker:.1f}% row brightness change between frames)",
                        fix="set power-line-frequency = 1 (50 Hz) or a fixed exposure under [cameras.<name>.controls]")

        ref_path = reference_dir(s) / f"{cam.name}.png"
        if save_reference:
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ref_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            report.ok(group, f"saved reference image {ref_path}")
        elif ref_path.is_file() and (ref := cv2.imread(str(ref_path))) is not None:
            if ref.shape != frame.shape:
                report.warn(group, f"reference image is {ref.shape[1]}x{ref.shape[0]}, "
                            f"camera is {frame.shape[1]}x{frame.shape[0]}",
                            fix="save a new one with `uv run so101 check --save-reference`")
            else:
                score = similarity(frame, cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
                if score >= 0.7:
                    pass
                elif not lookup.found_by_usb and score < 0.4:
                    report.fail(group, f"doesn't look like its reference image (similarity {score:.2f}): "
                                "probably the wrong camera",
                                fix=f"check which camera is which with `uv run so101 cameras`, then fix {s.path.name}")
                else:
                    report.warn(group, f"looks different from its reference image (similarity {score:.2f}): "
                                "the camera or the scene may have moved",
                                fix="put it back, or save a new reference with `uv run so101 check --save-reference`")
        elif not lookup.found_by_usb:
            report.warn(group, f"can't confirm this is the {cam.name} camera ({lookup.note})",
                        fix="save a reference image with `uv run so101 check --save-reference` once the cameras are placed")

        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot_dir / f"{cam.name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return lookups


# --- Recording loop -------------------------------------------------------------------------------


def check_loop(
    report: Report, buses: dict[ArmName, "FeetechMotorsBus"], cams: dict[str, "OpenCVCamera"], fps: int
) -> None:
    """Time the reads one recording step makes: follower joints, every camera, leader joints."""
    from lerobot.utils.robot_utils import precise_sleep

    period = 1 / fps
    steps, motor_s, camera_s = 0, 0.0, 0.0
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < LOOP_SAMPLE_S:
            t0 = time.perf_counter()
            buses["follower"].sync_read("Present_Position", num_retry=2)
            t1 = time.perf_counter()
            for cam in cams.values():
                cam.read_latest()
            t2 = time.perf_counter()
            buses["leader"].sync_read("Present_Position", num_retry=2)
            t3 = time.perf_counter()
            steps += 1
            motor_s += (t1 - t0) + (t3 - t2)
            camera_s += t2 - t1
            precise_sleep(max(0.0, period - (time.perf_counter() - t0)))  # as the record loop does
    except Exception as e:
        report.fail("loop", f"reading failed after {steps} steps: {e}",
                    fix="check the cables and run the check again; if it repeats, recording would fail the same way")
        return
    rate = steps / (time.perf_counter() - start)
    detail = (f"{rate:.1f} of {fps} steps/s reading both arms and {len(cams)} cameras "
              f"(motors {1000 * motor_s / steps:.1f} ms, cameras {1000 * camera_s / steps:.1f} ms per step)")
    if rate < 0.95 * fps:
        report.warn("loop", detail, fix="close other apps; turn the viewer off with --no-display")
    else:
        report.ok("loop", detail + "; the dry run times the full loop with video encoding")


# --- Task -----------------------------------------------------------------------------------------


def check_task(report: Report, s: Settings, task: "Task") -> None:
    """For your own tasks: will new demos merge with the sessions recorded so far?"""
    if not task.own:
        return
    group = f"task {task.name}"
    done = [session for session in data.sessions(s, task) if session.meta]
    kept = data.kept_episodes(s, task)
    if not done:
        report.ok(group, f"no demos yet; {task.episodes} to record")
        return
    report.ok(group, f"{kept} of {task.episodes} demos kept, in {len(done)} sessions")
    latest = done[-1]
    features = latest.meta.features
    recorded_cams = {k.removeprefix("observation.images."): tuple(f["shape"][:2]) for k, f in features.items()
                     if k.startswith("observation.images.")}
    planned_cams = {key: (s.camera(ours).height, s.camera(ours).width) for key, ours in task.cameras.items()}
    changes = []
    if latest.meta.fps != task.fps:
        changes.append(f"fps {latest.meta.fps} -> {task.fps}")
    if recorded_cams != planned_cams:
        changes.append(f"cameras {recorded_cams} -> {planned_cams} (name: (height, width))")
    if changes:
        report.fail(group, f"new demos wouldn't match session {latest.name}: {'; '.join(changes)}",
                    fix="change the settings back, or record into a new task")
    old = latest.manifest().get("calibration", {})
    changed = [
        name for name in ARMS
        if old.get(name, {}).get("sha256") not in (None, data.file_hash(s.arm(name).calibration_file))
    ]
    if changed:
        report.warn(group, f"{' and '.join(changed)} calibration changed since session {latest.name}",
                    fix="joint values may not line up with earlier demos; review them, or start the task over")


# --- Putting it together --------------------------------------------------------------------------


def run_checks(
    s: Settings,
    *,
    arms: tuple[ArmName, ...] = ARMS,
    cameras: list[Camera] | None = None,
    task: "Task | None" = None,
    loop_fps: int | None = None,
    keyboard: bool = False,
    save_reference: bool = False,
    snapshot_dir: Path | None = None,
) -> tuple[Report, list[CameraLookup]]:
    """Run the checks. `loop_fps` also times a recording step, which needs both arms and the cameras."""
    cameras = list(s.cameras.values()) if cameras is None else cameras
    report = Report()
    check_system(report, s, cameras, keyboard)
    buses: dict[ArmName, FeetechMotorsBus] = {}
    opened: dict[str, OpenCVCamera] = {}
    lookups: list[CameraLookup] = []
    try:
        for name in arms:
            if (bus := check_arm(report, s.arm(name))) is not None:
                buses[name] = bus
        if cameras:
            lookups = check_cameras(report, s, cameras, save_reference, snapshot_dir, opened)
        if task is not None:
            check_task(report, s, task)
        if loop_fps and set(buses) == set(ARMS) and len(opened) == len(cameras) and not report.failures:
            check_loop(report, buses, opened, loop_fps)
    finally:
        for bus in buses.values():
            close_bus(bus)
        for cam in opened.values():
            try:
                cam.disconnect()
            except Exception:
                pass
    return report, lookups


def preflight(s: Settings, title: str, **kwargs) -> tuple[list[Camera], Report]:
    """Run the checks before a command; exit if any fail.

    Returns the cameras with their current indices, and the report.
    """
    print(f"Checking the setup before {title}...", flush=True)
    report, lookups = run_checks(s, **kwargs)
    report.print()
    if report.failures:
        sys.exit(f"\n{report.summary()} Run `uv run so101 check` after fixing them.")
    print(f"{report.summary()}\n", flush=True)
    return [lookup.camera for lookup in lookups], report


# --- Live test (moves the follower) ---------------------------------------------------------------


def live_test(report: Report, s: Settings, seconds: float = 10.0) -> None:
    """Teleoperate for `seconds` and check that every follower joint follows the leader."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    from lerobot.utils.robot_utils import precise_sleep

    from so_101.validate import tracking

    print(
        "\nLive test: the follower will copy the leader for "
        f"{seconds:.0f} seconds.\n"
        "  1. Put both arms in the same rest pose (the follower jumps to the leader's pose).\n"
        "  2. When it starts, move every leader joint through a wide range, and open and close the gripper.\n"
        "  3. Finish back in the rest pose: the follower goes limp when the test ends.\n"
    )
    try:
        input("Press Enter to start, Ctrl+C to skip. ")
    except KeyboardInterrupt:
        print()
        report.warn("live test", "skipped")
        return
    follower = SO101Follower(SO101FollowerConfig(
        port=s.follower.resolve_port(), id=s.follower.id, calibration_dir=s.follower.calibration_dir,
        max_relative_target=s.teleop.max_relative_target or None,
    ))
    leader = SO101Leader(SO101LeaderConfig(
        port=s.leader.resolve_port(), id=s.leader.id, calibration_dir=s.leader.calibration_dir,
    ))
    group = "live test"
    names = list(follower.action_features)
    commanded, measured = [], []
    fps = 30  # the rate LeRobot records at
    try:
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        start = time.perf_counter()
        while (now := time.perf_counter()) - start < seconds:
            action = leader.get_action()
            follower.send_action(action)
            observation = follower.get_observation()
            commanded.append([action[n] for n in names])
            measured.append([observation[n] for n in names])
            remaining = seconds - (now - start)
            print(f"\r  {remaining:4.1f} s left ", end="", flush=True)
            precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - now)))
        rate = len(commanded) / (time.perf_counter() - start)
        print()
    except KeyboardInterrupt:
        print()
        report.warn(group, "stopped with Ctrl+C")
        return
    except Exception as e:
        print()
        report.fail(group, f"teleoperation failed after {len(commanded)} steps: {e}",
                    fix="check both arms' cables and power; run the check again")
        return
    finally:
        for device in (follower, leader):
            if device.is_connected:
                device.disconnect()
    detail = f"{rate:.1f} of {fps} steps/s with the follower moving"
    if rate < 0.95 * fps:
        report.warn(group, detail)
    else:
        report.ok(group, detail)
    tracking(report, group, names, [(np.array(commanded), np.array(measured))], rate, require_motion=True)
