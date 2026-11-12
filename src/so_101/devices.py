"""Find cameras by USB id, and set their UVC controls with uvc-util.

OpenCV numbers cameras by position in a list, and the list changes when a camera is plugged in or
out (an iPhone in Continuity Camera mode counts too). On macOS, OpenCV's AVFoundation backend lists
video and muxed devices sorted by `uniqueID` (`cap_avfoundation_mac.mm`); `video_devices` builds
the same list, so a camera's USB vendor and product id give its current OpenCV index.

That list is read once per process: AVFoundation never updates it in a process like `serve`, so the
device watch asks `usb_devices` (ioreg) instead whether each camera is plugged in.
"""

import dataclasses
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from so_101 import svtshim
from so_101.settings import Arm, Camera


def usb_pair(usb_id: str) -> tuple[int, int]:
    """`usb_id` ("vvvv:pppp", hex) as its vendor and product id."""
    vendor, product = (int(part, 16) for part in usb_id.split(":"))
    return vendor, product


def usb_devices() -> set[tuple[int, int]] | None:
    """The vendor and product id of every device on the USB bus now, or None where this lookup isn't
    available (not macOS, or ioreg failed).

    Unlike `video_devices`, it sees a camera plugged in or out while the process runs. AVFoundation's
    list in `serve`'s device watch stayed the same for 28 s with the wrist camera unplugged, even with
    the thread's run loop run before each call; ioreg saw the unplug and the replug, in 17 to 36 ms
    a call.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["ioreg", "-p", "IOUSB", "-l", "-w0"], capture_output=True, text=True, timeout=5,
                             check=True, env=svtshim.other_env()).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = set()
    for device in out.split("+-o ")[1:]:  # each device's properties follow its "+-o" line
        ids = dict(re.findall(r'"(idVendor|idProduct)" = (\d+)', device))
        if len(ids) == 2:
            found.add((int(ids["idVendor"]), int(ids["idProduct"])))
    return found


@dataclass(frozen=True)
class VideoDevice:
    index: int
    name: str
    unique_id: str
    model_id: str

    def matches(self, usb_id: str) -> bool:
        """True if this is the USB camera `usb_id` ("vvvv:pppp", hex)."""
        vendor, product = usb_pair(usb_id)
        # USB cameras have a modelID like "UVC Camera VendorID_13030 ProductID_37409" and a uniqueID
        # ending in the vendor and product ids in hex, e.g. "0x1420000032e69221".
        if re.search(rf"\bVendorID_{vendor} ProductID_{product}\b", self.model_id):
            return True
        return self.unique_id.lower().endswith(f"{vendor:04x}{product:04x}")


def video_devices() -> list[VideoDevice] | None:
    """Cameras in OpenCV index order, or None where this lookup isn't available (not macOS)."""
    if sys.platform != "darwin":
        return None
    try:
        import AVFoundation as AV
        import objc
    except ImportError:
        return None
    # The device watch calls this every 2 s on a thread with no run loop, so nothing drains
    # autoreleased Objective-C objects for it: the pool frees each call's.
    with objc.autorelease_pool():
        devices = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)) + list(
            AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)
        )
        devices.sort(key=lambda d: str(d.uniqueID()))
        return [
            VideoDevice(index=i, name=str(d.localizedName()), unique_id=str(d.uniqueID()), model_id=str(d.modelID()))
            for i, d in enumerate(devices)
        ]


@dataclass(frozen=True)
class CameraLookup:
    camera: Camera  # with `index` set to where the camera is now
    found_by_usb: bool  # the index comes from the USB id, so it is the right device
    problem: str = ""  # why the camera can't be used; empty if it can
    note: str = ""  # how the index was chosen, when that's worth saying


def find_camera(cam: Camera, devices: list[VideoDevice] | None) -> CameraLookup:
    """Work out `cam`'s current OpenCV index from its USB id; fall back to the configured index."""
    if not cam.usb_id:
        return CameraLookup(cam, False, note=f"no usb_id in so101.toml, so using index {cam.index} unchecked")
    if devices is None:
        return CameraLookup(cam, False, note=f"can't list cameras on this system, so using index {cam.index}")
    matches = [d for d in devices if d.matches(cam.usb_id)]
    if not matches:
        present = ", ".join(f"{d.index}: {d.name}" for d in devices) or "none"
        return CameraLookup(cam, False, problem=f"USB camera {cam.usb_id} is not connected (cameras: {present})")
    if len(matches) > 1:
        return CameraLookup(
            cam, False, note=f"{len(matches)} cameras have USB id {cam.usb_id}, so using index {cam.index} unchecked"
        )
    device = matches[0]
    note = "" if device.index == cam.index else f"found at index {device.index} (so101.toml says {cam.index})"
    return CameraLookup(dataclasses.replace(cam, index=device.index), True, note=note)


def arm_problem(arm: Arm) -> str | None:
    """Why `arm`'s board isn't connected, with the fix, or None if it is. Lists USB serial ports
    only, so it never opens the arm's port."""
    from serial.tools import list_ports

    ports = list_ports.comports()
    if arm.serial_number:
        if any(p.serial_number == arm.serial_number for p in ports):
            return None
        board = f"board (serial number {arm.serial_number})"
    elif any(p.device == arm.port for p in ports):
        return None
    else:
        board = f"board ({arm.port})"
    return (f"The {arm.name}'s {board} isn't connected. Plug in its USB cable and power, then restart "
            "so101 serve: the open connection doesn't survive a replug.")


def presence(arms: list[Arm], cameras: list[Camera]) -> dict[str, str | None]:
    """For each arm and camera by name: None when it is connected, or why it isn't, with the fix.

    Arms are found by USB serial number and cameras by `usb_id` on the USB bus (`usb_devices`), or in
    AVFoundation's list where ioreg isn't available. Neither is opened, so this can run on another
    thread while the control loop has them open.
    """
    found: dict[str, str | None] = {arm.name: arm_problem(arm) for arm in arms}
    usb = usb_devices()
    devices = video_devices() if usb is None else None
    for cam in cameras:
        if usb is not None and cam.usb_id:
            problem = "" if usb_pair(cam.usb_id) in usb else f"USB camera {cam.usb_id} is not connected"
        else:
            problem = find_camera(cam, devices).problem
        found[cam.name] = (
            f"The {cam.name} camera is missing: {problem}. Plug it back in, then restart so101 serve: "
            "the open camera doesn't survive a replug."
        ) if problem else None
    return found


def uvc_util() -> str | None:
    return shutil.which("uvc-util") or next(
        (str(p) for p in [Path.home() / ".local" / "bin" / "uvc-util"] if p.is_file()), None
    )


def _uvc_target(cam: Camera) -> list[str]:
    vendor, product = cam.usb_id.split(":")
    return ["-V", f"0x{vendor}:0x{product}"]


def set_controls(cam: Camera) -> str:
    """Set the UVC controls listed in so101.toml on `cam`. Returns an error message, or "" on success."""
    exe = uvc_util()
    if exe is None:
        return "uvc-util is not installed; run scripts/install-uvc-util.sh"
    args = [exe, *_uvc_target(cam)]
    for name, value in cam.controls.items():
        args += ["-s", f"{name}={value}"]
    result = subprocess.run(args, capture_output=True, text=True, env=svtshim.other_env())
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or "error" in output.lower():
        return output or f"uvc-util exited with {result.returncode}"
    return ""


def get_control(cam: Camera, name: str) -> int | None:
    """Read back one UVC control, or None if it can't be read."""
    exe = uvc_util()
    if exe is None:
        return None
    result = subprocess.run([exe, *_uvc_target(cam), "-o", name], capture_output=True, text=True,
                            env=svtshim.other_env())
    numbers = re.findall(r"-?\d+", result.stdout)
    return int(numbers[-1]) if result.returncode == 0 and numbers else None


def apply_camera_controls(cameras: list[Camera]) -> None:
    """Set each camera's UVC controls, printing what happened."""
    for cam in cameras:
        if not (cam.usb_id and cam.controls):
            continue
        settings_text = ", ".join(f"{k}={v}" for k, v in cam.controls.items())
        if error := set_controls(cam):
            print(f"WARNING: could not set {cam.name} camera controls ({settings_text}): {error}", file=sys.stderr)
        else:
            print(f"{cam.name} camera: set {settings_text}")
