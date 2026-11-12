"""The device watch's camera lookup, `devices.presence`, on ioreg output as macOS prints it, with ioreg
faked. AVFoundation's list never changes in a running process, so only ioreg sees a camera come and go.
"""

import subprocess
import unittest
from unittest import mock

from so_101 import devices
from so_101.settings import Camera

TOP = Camera("top", index=0, width=640, height=480, fps=30, usb_id="32e6:9221")
WRIST = Camera("wrist", index=1, width=640, height=480, fps=30, usb_id="32e6:9005")


def ioreg(*cameras: tuple[str, int]) -> str:
    """`ioreg -p IOUSB -l -w0` with a hub and the given cameras (name, product id) behind it, trimmed
    from this repo's Mac: properties are listed in no fixed order, and each device's follow its line."""
    lines = [
        "+-o Root  <class IORegistryEntry, id 0x100000100, retain 37>",
        "  +-o AppleT8122USBXHCI@01000000  <class AppleT8122USBXHCI, id 0x1000003eb, registered>",
        "  | +-o USB2.1 Hub@01100000  <class IOUSBHostDevice, id 0x100034034, registered>",
        "  | | {",
        '  | |   "USB Product Name" = "USB2.1 Hub"',
        '  | |   "idVendor" = 1507',
        '  | |   "idProduct" = 1552',
        "  | | }",
    ]
    for name, product in cameras:
        lines += [
            f"  | | +-o {name}@01113000  <class IOUSBHostDevice, id 0x10003b859, registered>",
            "  | |     {",
            f'  | |       "idProduct" = {product}',
            f'  | |       "USB Product Name" = "{name}"',
            '  | |       "idVendor" = 13030',
            "  | |     }",
        ]
    return "\n".join(lines) + "\n"


class PresenceTest(unittest.TestCase):
    def presence(self, out: str | Exception) -> dict[str, str | None]:
        """`presence` for both cameras on macOS, with ioreg printing `out`, or raising it."""
        def run(args, **kwargs):
            self.assertEqual(args[0], "ioreg")
            if isinstance(out, Exception):
                raise out
            return subprocess.CompletedProcess(args, 0, out, "")

        with mock.patch.object(devices.sys, "platform", "darwin"), \
                mock.patch.object(devices.subprocess, "run", run), \
                mock.patch.object(devices, "video_devices", return_value=[]) as avfoundation:
            found = devices.presence([], [TOP, WRIST])
        self.avfoundation_asked = avfoundation.called
        return found

    def test_a_camera_unplugged_and_plugged_back_in_is_seen_each_time(self) -> None:
        both = ioreg(("icspring camera", 0x9221), ("icspring camera", 0x9005))
        self.assertEqual(self.presence(both), {"top": None, "wrist": None})

        unplugged = self.presence(ioreg(("icspring camera", 0x9221)))
        self.assertIsNone(unplugged["top"])
        self.assertIn("The wrist camera is missing: USB camera 32e6:9005 is not connected.", unplugged["wrist"])
        self.assertIn("restart so101 serve", unplugged["wrist"])
        self.assertFalse(self.avfoundation_asked, "AVFoundation's list is stale in a running process")

        self.assertEqual(self.presence(both), {"top": None, "wrist": None})

    def test_without_ioreg_it_falls_back_to_avfoundations_list(self) -> None:
        found = self.presence(FileNotFoundError("ioreg"))
        self.assertTrue(self.avfoundation_asked)
        self.assertIn("The top camera is missing: USB camera 32e6:9221 is not connected", found["top"])


if __name__ == "__main__":
    unittest.main()
