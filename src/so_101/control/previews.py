"""Camera previews for the control page: JPEG snapshots, 320 px wide, from the frames the control loop read.

The loop is the only thread that reads the cameras. At the preview rate it hands over references to
the frames it already has (`put`), which costs a dict assignment and no copy: LeRobot gives each
camera read a new array, and nothing changes it afterwards. The web server's threads do the work:
each frame is resized and JPEG-encoded at most once, by the first request that wants it, and every
page watching that camera gets the same bytes. So previews never open a camera and never cost the
loop more than handing over a reference.

The page asks for one snapshot at a time, at the preview rate, rather than holding a stream open: a
browser allows about 6 connections to a server over plain HTTP, and open streams from a few tabs
would leave the newest none to poll the status or send Pause with.

Previews are the only optional work in the loop. `shedding.Pace` sheds them with `set_fps` when the
loop falls behind: half rate first, then none, and back up once the loop keeps up again.
"""

import threading
import time
from collections.abc import Callable, Iterable

WIDTH = 320  # px: enough to see the scene on a phone, at a quarter of a 640x480 frame's pixels
JPEG_QUALITY = 70
# A frame older than this is no longer served, so the page says "No preview" rather than show a
# picture that looks live. At 5 fps, shed to half, frames come every 0.4 s. A preview_fps so low
# that two frames span more than this keeps a frame for two of its periods instead.
STALE_S = 2.0


class Previews:
    def __init__(self, fps: float, names: Iterable[str], clock: Callable[[], float] = time.monotonic) -> None:
        """`fps` is `preview_fps` from so101.toml; `names` the cameras, as so101.toml names them.
        `clock` is the control loop's: it times how old the frames handed over are."""
        self.clock = clock
        self.target_fps = float(fps)
        self.fps = self.target_fps  # now, after shedding: target_fps, half of it, or 0 (off)
        self.names = tuple(names)
        self._next: float | None = None  # when the loop should next hand frames over
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[int, object]] = {}  # name -> (number, RGB array)
        self._put_at: float | None = None  # when the loop last handed frames over
        self._number = 0
        self._encoded: dict[str, tuple[int, bytes]] = {}  # name -> (number, JPEG)
        self._encoding = {name: threading.Lock() for name in self.names}

    # --- The control loop's side ----------------------------------------------------------------

    def due(self, now: float) -> bool:
        """Whether the loop should hand frames over this tick. Cheap: one comparison."""
        return self.fps > 0 and (self._next is None or now >= self._next)

    def put(self, frames: dict[str, object], now: float) -> None:
        """Keep a reference to each camera's latest frame. A camera absent from `frames` (its read
        failed) loses the frame it had."""
        period = 1 / self.fps
        # On schedule, or from now when the loop fell more than a period behind.
        self._next = now + period if self._next is None or now - self._next > period else self._next + period
        with self._lock:
            self._number += 1
            self._put_at = self.clock()
            self._frames = {name: (self._number, frame) for name, frame in frames.items()}

    def set_fps(self, fps: float) -> None:
        self.fps = fps
        self._next = None

    # --- The web server's side ------------------------------------------------------------------

    def snapshot(self, name: str) -> bytes | None:
        """Camera `name`'s latest frame as a JPEG, or None before the loop has handed one over, when
        the last handover had none from it, or once that handover is STALE_S old (previews shed, or
        the loop stuck)."""
        fps = self.fps  # read once: the control loop may shed previews to 0 between two reads
        limit = max(STALE_S, 2 / fps) if fps > 0 else STALE_S
        with self._lock:
            fresh = self._put_at is not None and self.clock() - self._put_at <= limit
            found = self._frames.get(name) if fresh else None
        return None if found is None else self._jpeg(name, *found)

    def _jpeg(self, name: str, number: int, frame) -> bytes:
        with self._encoding[name]:  # a second page watching the camera reuses the first one's work
            cached = self._encoded.get(name)
            if cached is not None and cached[0] == number:
                return cached[1]
            jpeg = encode(frame)
            if cached is None or number > cached[0]:  # never over a newer frame another request encoded
                self._encoded[name] = (number, jpeg)
            return jpeg


def encode(frame) -> bytes:
    """An RGB frame from LeRobot, as a JPEG WIDTH px wide."""
    import cv2

    height, width = frame.shape[:2]
    small = cv2.resize(frame, (WIDTH, max(1, round(height * WIDTH / width))), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("OpenCV couldn't encode the preview as JPEG.")
    return jpeg.tobytes()
