"""Find cameras by USB id, and set their UVC controls with uvc-util.

OpenCV numbers cameras by position in a list, and the list changes when a camera is plugged in or
out (an iPhone in Continuity Camera mode counts too). On macOS, OpenCV's AVFoundation backend lists
video and muxed devices sorted by `uniqueID` (`cap_avfoundation_mac.mm`); `video_devices` builds
the same list, so a camera's USB vendor and product id give its current OpenCV index.
"""

import dataclasses
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from so_101.settings import Camera


@dataclass(frozen=True)
class VideoDevice:
    index: int
    name: str
    unique_id: str
    model_id: str

    def matches(self, usb_id: str) -> bool:
        """True if this is the USB camera `usb_id` ("vvvv:pppp", hex)."""
        vendor, product = (int(part, 16) for part in usb_id.split(":"))
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
    except ImportError:
        return None
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
    if cam.index_from_env:
        return CameraLookup(cam, False, note=f"index {cam.index} from SO101_CAMERA_{cam.name.upper()}_INDEX")
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


def find_cameras(cameras: list[Camera]) -> list[CameraLookup]:
    devices = video_devices()
    return [find_camera(cam, devices) for cam in cameras]


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
    result = subprocess.run(args, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or "error" in output.lower():
        return output or f"uvc-util exited with {result.returncode}"
    return ""


def get_control(cam: Camera, name: str) -> int | None:
    """Read back one UVC control, or None if it can't be read."""
    exe = uvc_util()
    if exe is None:
        return None
    result = subprocess.run([exe, *_uvc_target(cam), "-o", name], capture_output=True, text=True)
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
