"""Health checks for the arms, cameras and setup, run by `so101 check` and before record, teleop, run and serve.

Nothing here writes to the motors: arms are read over their serial bus without LeRobot's `connect()`,
which would start calibrating on a mismatch and switch motor torque on. The camera check sets each
camera's UVC controls, the only thing written. Only `live_test` moves the follower, and only when
asked for with `so101 check --live`.

The checks open the buses and cameras they need and close them again. A caller that already holds
a device open, such as the control center (`control/main.py`), passes it in instead: the checks use it as it is and leave
it open. Opening a second connection isn't an option for an arm, because two processes on one
serial port corrupt each other's packets.
"""

import shutil
import sys
import time
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from serial.tools import list_ports

from so_101 import data
from so_101.devices import CameraLookup, find_camera, get_control, set_controls, uvc_util, video_devices
from so_101 import settings
from so_101.report import Report
from so_101.settings import ARMS, Arm, ArmName, Camera, ConfigError, PortNotFoundError, Settings

if TYPE_CHECKING:
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.motors import SerialMotorsBus

    from so_101.tasks import Task

STS3215_MODEL = 777
TEMP_WARN_C = 50
TEMP_FAIL_C = 60
VOLTAGE_SPREAD_WARN_V = 1.0
POSITION_SLACK_TICKS = 50  # how far outside its calibrated range a joint may read before it's flagged
CAMERA_SAMPLE_S = 2.5
LOOP_SAMPLE_S = 3.0
# Free disk below which recording is refused, and below which it's warned about. `serve` checks them
# before each attempt too.
DISK_FAIL_GB = 1
DISK_WARN_GB = 5


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


def check_system(report: Report, s: Settings, cameras: list[Camera], keyboard: bool, hub: bool = True) -> None:
    """Hugging Face login (with `hub`), disk space, uvc-util, and with `keyboard` the Accessibility
    permission. The login check asks the Hub, with no timeout."""
    from so_101.tasks import hf_user

    if hub:
        try:
            report.ok("hugging face", f"logged in as {hf_user()}")
        except ConfigError:
            report.warn("hugging face", "not logged in: needed to train and back up", fix="uv run hf auth login")
        except Exception as e:  # offline, or the Hub is down
            report.warn("hugging face", f"can't reach the Hub: {e}")

    existing = next(p for p in [s.data_dir, *s.data_dir.parents] if p.exists())
    free_gb = shutil.disk_usage(existing).free / 1e9
    if free_gb < DISK_FAIL_GB:
        report.fail("disk", f"{free_gb:.1f} GB free for {s.data_dir}", fix="free up disk space")
    elif free_gb < DISK_WARN_GB:
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


def check_video(report: Report, required: bool) -> None:
    """SVT-AV1's macOS teardown crash fix (`so_101.svtshim`). With `required`, this process encodes
    video (or starts one that does), so the fix must be loaded in it; otherwise it only has to build."""
    from so_101 import svtshim

    if not svtshim.needed():
        if sys.platform == "darwin" and svtshim.svt_version():
            report.ok("video encoder", f"{svtshim.version_text()} has its teardown crash fixed")
        return
    if found := svtshim.problem(in_this_process=required):
        report.fail("video encoder", found[0], fix=found[1])
    elif required:
        report.ok("video encoder", f"{svtshim.version_text()}: the teardown crash fix is loaded")
    else:
        report.ok("video encoder", f"{svtshim.version_text()}: its teardown crash fix builds, and the commands "
                  "that record or build load it")


# --- Arms -----------------------------------------------------------------------------------------


def _open_bus(arm: Arm, port: str, degrees: bool = False) -> "SerialMotorsBus":
    """The arm's motor bus as LeRobot defines it, with the calibration file loaded, opened read-only."""
    from lerobot.motors import SerialMotorsBus

    device = arm.lerobot_device(port, use_degrees=degrees)
    # A separate bus, not device.bus: when `device` is garbage-collected its __del__ disconnects its
    # bus if open, which writes torque-off to every motor and closes the port.
    bus = SerialMotorsBus(port=port, motors=device.bus.motors, calibration=device.calibration)
    bus._connect(handshake=False)  # open the port without LeRobot's checks, which raise on the first problem
    return bus


def close_bus(bus: "SerialMotorsBus") -> None:
    bus._close()  # not bus.disconnect(): that writes to the motors to disable torque


@dataclass(frozen=True)
class MotorReading:
    name: str
    temperature_c: int
    voltage_v: float
    load_pct: float  # of maximum torque
    position: int  # raw encoder steps, 0-4095
    range_pct: float | None  # where `position` is within the calibrated range; None if uncalibrated
    alarms: list[str]


def read_pose(arm: Arm, degrees: bool = False) -> dict[str, float] | dict[str, int]:
    """Each joint's position, without writing anything: with `degrees` in the degrees a policy sees,
    otherwise in raw encoder steps (0-4095), without calibration. Opens and closes the arm's bus."""
    bus = _open_bus(arm, arm.resolve_port(), degrees=degrees)
    try:
        pose = bus.sync_read("Present_Position", normalize=degrees, num_retry=2)
    finally:
        close_bus(bus)
    return pose if degrees else {name: int(steps) for name, steps in pose.items()}


def _ping_motors(bus: "SerialMotorsBus") -> tuple[dict, list[str]]:
    """Each motor's ping reply, None if it didn't answer, and "name (id N)" for each that didn't."""
    from so_101.scan import ping

    replies = {name: ping(bus._connected_bus, motor.id) for name, motor in bus.motors.items()}
    return replies, [f"{name} (id {bus.motors[name].id})" for name, reply in replies.items() if reply is None]


def read_motors(arm: Arm) -> list[MotorReading]:
    """Read every motor of an arm without writing anything. Raises if a motor doesn't answer."""
    bus = _open_bus(arm, arm.resolve_port())
    try:
        replies, missing = _ping_motors(bus)
        if missing:
            raise RuntimeError(f"no answer from {', '.join(missing)}: is the arm's power on?")
        temps = bus.sync_read("Present_Temperature", normalize=False, num_retry=2)
        volts = bus.sync_read("Present_Voltage", normalize=False, num_retry=2)
        loads = bus.sync_read("Present_Load", normalize=False, num_retry=2)
        positions = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    finally:
        close_bus(bus)
    readings = []
    for name in bus.motors:
        cal = bus.calibration.get(name)
        span = cal.range_max - cal.range_min if cal else 0
        readings.append(MotorReading(
            name=name,
            temperature_c=int(temps[name]),
            voltage_v=volts[name] / 10,
            # Bit 10 is the direction, which LeRobot decodes as the sign; the rest is tenths of a percent
            load_pct=abs(int(loads[name])) / 10,
            position=int(positions[name]),
            range_pct=100 * (positions[name] - cal.range_min) / span if span else None,
            alarms=replies[name].errors(),
        ))
    return readings


def open_arm(report: Report, arm: Arm) -> "SerialMotorsBus | None":
    """The arm's bus, opened read-only for the caller to close, or None with the reason reported:
    the board isn't found, its port holds another board, or the port won't open."""
    try:
        port = arm.resolve_port()
    except PortNotFoundError:
        report.fail(arm.name, "board not found", fix=f"plug in its USB cable and power; `uv run so101 find-port {arm.name}`")
        return None
    serial = next((p.serial_number for p in list_ports.comports() if p.device == port), None)
    if arm.serial_number and serial and serial != arm.serial_number:
        report.fail(
            arm.name,
            f"{port} is another board (serial number {serial}, expected {arm.serial_number})",
            fix="run `uv run so101 find-port follower` and `uv run so101 find-port leader` to save both ports again",
        )
        return None
    try:
        return _open_bus(arm, port)
    except Exception as e:
        report.fail(arm.name, f"can't open {port}: {e}", fix="close other programs using the arm, then replug it")
        return None


def check_arm_files(report: Report, arm: Arm) -> None:
    """The arm's calibration file: there, and with every joint's range wide enough."""
    fix = f"uv run so101 calibrate {arm.name}"
    if not arm.calibration_file.is_file():
        report.fail(arm.name, f"not calibrated (no {arm.calibration_file})", fix=fix)
    elif warning := narrow_joints_warning(arm):
        report.fail(arm.name, warning, fix=f"{fix}, and move every joint through its full range")


def check_arm(report: Report, arm: Arm) -> "SerialMotorsBus | None":
    """Check one arm, opening its bus. Returns the bus, still open for the caller to close, if the
    motors answered, for the loop timing test; otherwise closes it."""
    check_arm_files(report, arm)
    if (bus := open_arm(report, arm)) is None:
        return None
    if check_motors(report, arm, bus):
        return bus
    close_bus(bus)
    return None


def check_motors(report: Report, arm: Arm, bus: "SerialMotorsBus") -> bool:
    """Check one arm's motors on its open `bus`, which stays open. Returns whether they answered."""
    group = arm.name
    try:
        replies, missing = _ping_motors(bus)
        if missing:
            report.fail(
                group,
                f"no answer from {', '.join(missing)}",
                fix=f"check servo power and the cable after the last motor that answers; `uv run so101 scan {arm.name}`",
            )
            return False
        wrong = [f"{name} (model {reply.model})" for name, reply in replies.items() if reply.model != STS3215_MODEL]
        if wrong:
            report.fail(group, f"not STS3215 servos: {', '.join(wrong)}",
                        fix="check the motor ids with `uv run so101 scan`")
        alarms = {name: reply.errors() for name, reply in replies.items() if reply.error}
        if alarms:
            listed = ", ".join(f"{name} ({', '.join(errs)})" for name, errs in alarms.items())
            fix = "switch the arm's power off and on to clear it; if it returns, look for a jammed joint or a weak supply"
            if "overload" in alarms.get("gripper", []):
                fix = ("switch the arm's power off and on to clear it. A gripper overload means it kept squeezing: "
                       "close the leader's gripper only until the follower holds the object")
            report.fail(group, f"motors in a protection alarm: {listed}", fix=fix)

        volts = {m: v / 10 for m, v in bus.sync_read("Present_Voltage", normalize=False, num_retry=2).items()}
        temps = bus.sync_read("Present_Temperature", normalize=False, num_retry=2)
        moved = "" if bus.port == arm.port else f" (saved port {arm.port} is gone; found by serial number)"
        report.ok(
            group,
            f"{bus.port}{moved}: 6/6 motors, {min(volts.values()):.1f}-{max(volts.values()):.1f} V, "
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
        return False
    return True


def check_calibration(report: Report, arm: Arm, bus: "SerialMotorsBus") -> None:
    """Compare the calibration file with what's stored in the motors. File-only checks are in `check_arm_files`."""
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


def find_cameras(report: Report, cameras: list[Camera]) -> list[CameraLookup]:
    """Where each camera is now, reporting each that isn't found."""
    devices = video_devices()
    lookups = [find_camera(cam, devices) for cam in cameras]
    for lookup in lookups:
        if lookup.problem:
            report.fail(f"{lookup.camera.name} camera", lookup.problem,
                        fix="plug it in; `uv run so101 cameras` lists what's connected")
    return lookups


def open_cameras(report: Report, lookups: list[CameraLookup]) -> tuple[dict[str, "OpenCVCamera"], set[str]]:
    """Set each found camera's controls, then open it, as recording does. Returns the cameras that
    opened, by name, for the caller to close, and the names of those whose controls couldn't be set."""
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    opened, unset = {}, set()
    for cam in (lookup.camera for lookup in lookups if not lookup.problem):
        group = f"{cam.name} camera"
        if cam.usb_id and cam.controls and (error := set_controls(cam)):
            report.warn(group, f"couldn't set controls {cam.controls}: {error}")
            unset.add(cam.name)
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
    return opened, unset


def check_cameras(
    report: Report,
    lookups: list[CameraLookup],
    cams: Mapping[str, "OpenCVCamera"],
    snapshot_dir: Path | None = None,
    record_fps: int | None = None,
    unset: Collection[str] = (),
) -> None:
    """Sample each found camera that's open in `cams`, by name, and read back its controls, except
    for the cameras in `unset`, whose controls couldn't be set and already have a WARN.

    `record_fps` is the rate frames will be recorded at, when below the cameras' configured rate.
    """
    import cv2

    cams = {lookup.camera.name: cams[lookup.camera.name] for lookup in lookups
            if not lookup.problem and lookup.camera.name in cams}
    samples = sample_cameras(cams, CAMERA_SAMPLE_S) if cams else {}
    for lookup in lookups:
        cam = lookup.camera
        if cam.name not in samples:
            continue
        group = f"{cam.name} camera"
        if cam.usb_id and cam.controls and cam.name not in unset:
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

        wanted = min(cam.fps, record_fps or cam.fps)
        if rate < 0.9 * wanted:
            repeated = max(0.0, 1 - rate / wanted)
            # Auto exposure lengthens in dim light and slows a camera; with a fixed exposure this is
            # the camera's own limit (this kit's cameras give 20 fps at 640x480 at any exposure).
            auto = cam.controls.get("auto-exposure-mode", 8) != 1
            report.warn(
                group,
                f"delivers {rate:.1f} new frames/s against the {wanted} recorded: about {repeated:.0%} of "
                "recorded frames will repeat the one before",
                fix=("add light or set a fixed exposure, as auto exposure slows down in dim light; "
                     if auto else "")
                + "if it's the camera's limit, repeated frames are harmless for ACT; setting the task's fps "
                "in tasks.toml to the camera's rate makes them rare",
            )
        if flicker is not None and flicker > 2.0:
            report.warn(group, f"rolling bands or flicker ({flicker:.1f}% row brightness change between frames)",
                        fix="set power-line-frequency = 1 (50 Hz) or a fixed exposure under [cameras.<name>.controls]")

        if not lookup.found_by_usb:
            report.warn(group, f"can't confirm this is the {cam.name} camera ({lookup.note})",
                        fix="`uv run so101 cameras` shows which index is which; set this camera's `usb_id` "
                        "or `index` in so101.toml to match")

        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot_dir / f"{cam.name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# --- Recording loop -------------------------------------------------------------------------------


def check_loop(
    report: Report, buses: dict[ArmName, "SerialMotorsBus"], cams: dict[str, "OpenCVCamera"], fps: int
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
    """Will new demos merge with the sessions recorded so far?"""
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
    planned_cams = {name: (s.camera(name).height, s.camera(name).width) for name in task.cameras}
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


# --- Control center settings ----------------------------------------------------------------------

REST_POSE_FIX = ("move the follower to its rest pose by hand, then run `uv run so101 pose --save-rest` "
                 "(see Saving a rest pose in the README)")


def follower_motors(arm: Arm) -> list[str]:
    """The follower's motors as LeRobot defines them, without opening its port."""
    return list(arm.lerobot_device(arm.port).bus.motors)


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def rest_pose_problem(s: Settings) -> str | None:
    """The first thing wrong with the rest pose `serve` drives to, or None. Without the follower's
    calibration file there's nothing to compare `rest_pose_calibration` with, so it isn't compared."""
    pose, saved, now = s.control.rest_pose, s.control.rest_pose_calibration, s.follower.calibration_fingerprint()
    if pose is None:
        return f"no rest_pose under [control] in {s.path}: Reset drives the arm there at the end of every attempt"
    if not isinstance(pose, dict):
        return f"rest_pose = {pose!r} isn't a table of motor positions"
    motors = follower_motors(s.follower)
    missing = [m for m in motors if m not in pose]
    unknown = [m for m in pose if m not in motors]
    if missing or unknown:
        return ("rest_pose must name exactly the follower's six motors, so the arm is checked at rest in every "
                "joint before the motors turn off"
                + (f"; it leaves out {', '.join(missing)}" if missing else "")
                + (f"; it names unknown {', '.join(unknown)}" if unknown else ""))
    if bad := [f"{m} = {v!r}" for m, v in pose.items()
               if not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v < settings.MOTOR_STEPS)]:
        return (f"rest_pose has positions that aren't raw motor steps from 0 to {settings.MOTOR_STEPS - 1}: "
                + ", ".join(bad))
    if saved is None:
        return ("rest_pose has no rest_pose_calibration, so `serve` can't tell whether the follower was "
                "recalibrated since it was saved, and the pose may no longer fold the arm")
    if now is not None and saved != now:
        return (f"the follower was recalibrated since rest_pose was saved (rest_pose_calibration {saved!r}, "
                f"calibration file now {now!r}), so the saved pose may no longer fold the arm")
    return None


def check_control(report: Report, s: Settings, status: Literal["FAIL", "WARN"] = "FAIL") -> None:
    """The rest pose and the `[control]` settings, which `serve` needs. Reads so101.toml and the
    follower's calibration file only.

    `settings.load()` doesn't check them, so a bad value never stops the commands that fix it. Each
    problem gets `status`: FAIL in `serve`'s preflight, WARN in `so101 check`. `runs`, `build --with`
    and `review --runs` check `blame_s` themselves (`analysis.check_blame_s`)."""
    c = s.control

    def problem(group: str, detail: str, fix: str) -> None:
        report.add(group, status, detail if status == "FAIL" else f"{detail}; `serve` won't start", fix)

    if detail := rest_pose_problem(s):
        problem("rest pose", detail, REST_POSE_FIX)
    elif s.follower.calibration_fingerprint() is None:
        # The follower's own check fails on the missing file; this only says what couldn't be compared.
        report.warn("rest pose", f"there's no calibration file ({s.follower.calibration_file}) to compare "
                    f"rest_pose_calibration {c.rest_pose_calibration!r} with, so `serve` can't tell whether rest_pose "
                    "still folds the arm", fix="put the follower's calibration file back, or calibrate it (`uv run "
                    "so101 calibrate follower`) and save the rest pose again")
    else:
        report.ok("rest pose", "names the follower's six motors, saved with its current calibration")

    rules = settings.CONTROL_SETTINGS
    settings_ok = True

    def bad_value(key: str, detail: str, allowed: str) -> None:
        nonlocal settings_ok
        settings_ok = False
        default = rules[key].default
        default = f'"{default}"' if isinstance(default, str) else f"{default:g}"  # as so101.toml writes it
        problem("control", f"{key} = {getattr(c, key)!r} {detail}",
                f"set {key} under [control] in {s.path} to {allowed}, or delete the line for the default {default}")

    for key, rule in rules.items():
        if (low := rule.minimum) is None:
            continue
        value = getattr(c, key)
        allowed = f"a number {'of at least' if rule.inclusive else 'above'} {low:g}"
        if not _number(value):
            bad_value(key, "isn't a number", allowed)
        elif not math.isfinite(value):
            bad_value(key, "isn't a finite number", allowed)
        elif value < low or value == low and not rule.inclusive:
            bad_value(key, f"is out of range: {rule.why}", allowed)
    if isinstance(c.port, bool) or not isinstance(c.port, int) or not 1 <= c.port <= 65535:
        bad_value("port", "isn't a port number", "a whole number from 1 to 65535")
    for key, rule in rules.items():
        if rule.choices and getattr(c, key) not in rule.choices:
            bad_value(key, "isn't allowed", "one of " + ", ".join(f'"{a}"' for a in rule.choices))
    if settings_ok:
        report.ok("control", "every [control] setting is in its range")


# --- Putting it together --------------------------------------------------------------------------


def run_checks(
    s: Settings,
    *,
    arms: tuple[ArmName, ...] = ARMS,
    cameras: list[Camera] | None = None,
    task: "Task | None" = None,
    loop_fps: int | None = None,
    keyboard: bool = False,
    snapshot_dir: Path | None = None,
    control: Literal["FAIL", "WARN"] | None = None,
    teleop_only: tuple[ArmName, ...] = (),
    video: bool | None = None,
) -> tuple[Report, list[CameraLookup]]:
    """Run the checks. `loop_fps` also times a recording step, which needs both arms and the cameras.
    With `control`, the rest pose and `[control]` settings are checked too, each problem getting that
    status (`check_control`). The arms in `teleop_only` (of `arms`) are needed only for Teleop: each of
    their failures is a WARN that says Teleop won't be available. With `video`, the video encoder's
    crash fix is checked too (`check_video`): True for a command that records. Every device the checks
    open, they close.
    """
    cameras = list(s.cameras.values()) if cameras is None else cameras
    report = Report()
    check_system(report, s, cameras, keyboard)
    if video is not None:
        check_video(report, video)
    if control is not None:
        check_control(report, s, control)
    buses: dict[ArmName, SerialMotorsBus] = {}
    opened: dict[str, OpenCVCamera] = {}
    lookups: list[CameraLookup] = []
    try:
        for name in arms:
            arm_report = Report() if name in teleop_only else report
            if (bus := check_arm(arm_report, s.arm(name))) is not None:
                buses[name] = bus
            if arm_report is not report:
                for r in arm_report.results:
                    if r.status == "FAIL":
                        report.warn(r.group, f"{r.detail}; Teleop won't be available", r.fix)
                    else:
                        report.results.append(r)
        if cameras:
            lookups = find_cameras(report, cameras)
            opened, unset = open_cameras(report, lookups)
            check_cameras(report, lookups, opened, snapshot_dir, loop_fps, unset)
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
    follower = s.follower.lerobot_device(s.follower.resolve_port(),
                                         max_relative_target=s.teleop.max_relative_target or None)
    leader = s.leader.lerobot_device(s.leader.resolve_port())
    group = "live test"
    names = list(follower.action_features)
    commanded, measured = [], []
    fps = 30  # tasks.toml's default rate: the live test runs without a task
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
