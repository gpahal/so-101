"""`so101 sim aim`: guide the top camera, and the mat, back to where they were when the demos were recorded.

The sim's top camera is fitted to the demos' frames, and a policy trained on them expects their view. A bump
moves the camera a few pixels and turns it, and the mat, a loose mouse pad, slides on its own.

This compares the live picture with the median of the demos' first frames, window by window, by normalised
cross-correlation. The camera's shift and turn come from windows on things fixed to the room or the table: the
socket and skirting, the floor tiles and the arm's base plate, each searched around a first, coarse reading
of the room's corner at half size. The mat's slide is what its edges moved beyond that. Every window keeps
clear of where the arm (at the start pose or at rest) and the jar appear; take the block off the mat while
aiming.

Needs only OpenCV and the demos' videos, not MuJoCo.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# (x, y, width, height) in the 640 x 480 picture.
FIXED = {
    "socket and skirting": (70, 30, 140, 125),
    "floor tiles": (0, 80, 85, 110),
    "arm's base plate": (115, 365, 85, 110),
}
MAT = {
    "mat, far left edge": (195, 60, 135, 120),
    "mat, far right edge": (470, 100, 170, 170),
    "mat, near edge": (210, 320, 170, 150),
}
# Within these the picture is as close to the demos' view as the demos were to each other (0.1 to 0.4 px).
ALIGNED_PX = 1.0
ALIGNED_DEG = 0.15
MAT_PX = 2.0
OUTLIER_PX = 2.0  # a fixed window this far from the others' fit is covered by something, and left out
SMOOTHING = 15  # readings in the median, half a second at 30 fps
BLUR = 1.0  # pixels
# Where `coarse` looks for features: the room's corner and floor, and the arm's base plate and the table
# round it, all fixed and clear of the arm at the start pose and at rest. Each window then refines its
# reading within SEARCH px.
COARSE = ((0, 0, 240, 175), (100, 350, 120, 130))
COARSE_MATCHES = 12
SEARCH = 12
MAT_SEARCH = 20  # beyond where the camera's move puts the mat's edges
PAD = 80  # the live picture's border, so a search can run a little past its edge
MATCH = 0.6  # the weakest normalised cross-correlation a window's match is used at


def gray(image: np.ndarray) -> np.ndarray:
    """An RGB picture as float grey, lightly blurred against sensor noise."""
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), BLUR)


REFERENCE_DEMOS = 25  # first frames in the demos' view: enough for the median to drop the block wherever it lay


def reference_episodes(recorded: list[tuple[str, int]], deleted: set[tuple[str, int]]) -> list[tuple[str, int]]:
    """Which demos give the demos' view: (session, episode) for up to REFERENCE_DEMOS of the recorded ones,
    spread evenly from the first session to the last, leaving out those review deleted. `recorded` is each
    session's name and episode count, oldest first."""
    kept = [(name, episode) for name, count in recorded for episode in range(count) if (name, episode) not in deleted]
    if len(kept) <= REFERENCE_DEMOS:
        return kept
    return [kept[int(i)] for i in np.linspace(0, len(kept) - 1, REFERENCE_DEMOS).round()]


def reference(frames: list[np.ndarray]) -> np.ndarray:
    """The demos' view: the per-pixel median of their first frames (RGB), which drops the block."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


@dataclass(frozen=True)
class Offset:
    """How the live picture sits against the reference."""

    dx: float  # pixels the camera's picture is moved right of the reference, at its middle
    dy: float  # and down
    degrees: float  # how far it's turned clockwise
    mat_dx: float  # pixels the mat's edges are moved right, beyond what the camera explains
    mat_dy: float  # and down
    windows: dict[str, tuple[float, float, bool]]  # each window's (dx, dy, used)
    found: bool = True  # False when too few fixed windows matched to say; dx, dy and degrees are then `coarse`'s, or 0

    @property
    def camera_aligned(self) -> bool:
        return self.found and bool(np.hypot(self.dx, self.dy) <= ALIGNED_PX and abs(self.degrees) <= ALIGNED_DEG)

    @property
    def mat_aligned(self) -> bool:
        return bool(np.hypot(self.mat_dx, self.mat_dy) <= MAT_PX)

    def _camera_steps(self, decimals: int) -> list[str]:
        steps = []
        if abs(self.dx) > ALIGNED_PX / 2:
            steps.append((abs(self.dx), f"turn it {'right' if self.dx > 0 else 'left'} ({abs(self.dx):.{decimals}f} px)"))
        if abs(self.dy) > ALIGNED_PX / 2:
            steps.append((abs(self.dy), f"tilt it {'down' if self.dy > 0 else 'up'} ({abs(self.dy):.{decimals}f} px)"))
        if abs(self.degrees) > ALIGNED_DEG:
            way = "clockwise" if self.degrees > 0 else "anticlockwise"
            steps.append((abs(self.degrees) * 4, f"roll it {way}, looking out through it ({abs(self.degrees):.{decimals + 1}f} deg)"))
        return [text for _, text in sorted(steps, reverse=True)]

    def advice(self) -> list[str]:
        """What to do, the camera first, its largest correction first. Turning a camera left slides its
        picture right, tilting it up slides the picture down, and rolling it clockwise (looking out
        through it) turns the picture anticlockwise."""
        if not self.found and not (self.dx or self.dy or self.degrees):
            return ["Can't find the demos' view: line the green outlines up by eye (b blends the pictures)."]
        if not self.found:  # only `coarse` matched: far off, so steer by it until the windows match
            return ["Far off, roughly:"] + [f"Camera: {text}" for text in self._camera_steps(0)]
        lines = []
        if self.camera_aligned:
            lines.append("Camera: aligned, leave it here.")
        else:
            lines += [f"Camera: {text}" for text in self._camera_steps(1)]
        if self.mat_aligned:
            lines.append("Mat: where it was.")
        else:
            lines.append(f"Mat: slide it {_picture_way(-self.mat_dx, -self.mat_dy)} in the picture (blue arrows)")
        return lines


def _picture_way(dx: float, dy: float) -> str:
    parts = [(abs(dx), "right" if dx > 0 else "left"), (abs(dy), "down" if dy > 0 else "up")]
    return " and ".join(f"{way} {size:.0f} px" for size, way in sorted(parts, reverse=True) if size >= 0.5) or "nowhere"


def _shift(reference_gray: np.ndarray, live_padded: np.ndarray, window: tuple[int, int, int, int], guess: np.ndarray, search: int) -> tuple[float, float, bool]:
    """How far a window's content moved: where the reference's window, less `search` px each side, best
    matches the live picture (padded by `PAD`) around `guess`, by normalised cross-correlation, to a
    tenth of a pixel. Unlike phase correlation, it weighs each frequency by its strength, so the video's
    compression blocks, which stand still wherever the camera points, don't pull dark windows' matches to
    zero. The last value is False for a match at the search's edge, or a weak one."""
    x, y, w, h = window
    ox, oy = (int(round(v)) for v in guess)
    template = reference_gray[y + search : y + h - search, x + search : x + w - search]
    top, left = PAD + y + oy, PAD + x + ox
    if top < 0 or left < 0 or top + h > live_padded.shape[0] or left + w > live_padded.shape[1]:
        return 0.0, 0.0, False  # the camera's so far off that this window has left the picture
    area = live_padded[top : top + h, left : left + w]
    scores = cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, (px, py) = cv2.minMaxLoc(scores)
    inside = 0 < px < 2 * search and 0 < py < 2 * search
    return (
        ox + px - search + _peak(scores[py, max(px - 1, 0) : px + 2], px),
        oy + py - search + _peak(scores[max(py - 1, 0) : py + 2, px], py),
        bool(inside and best >= MATCH),
    )


def coarse(reference_gray: np.ndarray, live_gray: np.ndarray) -> tuple[float, float, float] | None:
    """A first reading of the picture's shift and turn (radians), to a pixel or two, from ORB features on
    the fixed areas matched with RANSAC, which finds moves of tens of pixels and a few degrees. None when
    too few features agree."""
    mask = np.zeros(reference_gray.shape, np.uint8)
    for x, y, w, h in COARSE:
        mask[y : y + h, x : x + w] = 255
    orb = cv2.ORB_create(1500)
    # The live picture is searched whole: a big move carries the fixed areas out of where they were.
    found = [orb.detectAndCompute(cv2.convertScaleAbs(reference_gray), mask), orb.detectAndCompute(cv2.convertScaleAbs(live_gray), None)]
    if any(descriptors is None for _, descriptors in found):
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(found[0][1], found[1][1])
    if len(matches) < COARSE_MATCHES:
        return None
    before = np.float32([found[0][0][m.queryIdx].pt for m in matches])
    after = np.float32([found[1][0][m.trainIdx].pt for m in matches])
    transform, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if transform is None or inliers.sum() < COARSE_MATCHES:
        return None
    centre = np.array([reference_gray.shape[1], reference_gray.shape[0]]) / 2
    turn = float(np.arctan2(transform[1, 0], transform[0, 0]))
    dx, dy = transform[:, :2] @ centre + transform[:, 2] - centre
    return float(dx), float(dy), turn


def _peak(values: np.ndarray, index: int) -> float:
    """The sub-pixel offset of a peak from a parabola through it and its two neighbours."""
    if index == 0 or len(values) < 3:
        return 0.0
    left, middle, right = values
    bend = left - 2 * middle + right
    return float(np.clip(0.5 * (left - right) / bend, -0.5, 0.5)) if bend < 0 else 0.0


def measure(reference_gray: np.ndarray, live_gray: np.ndarray) -> Offset:
    """The camera's shift and turn against the reference from the fixed windows, leaving out one that
    disagrees with the rest, and the mat's slide beyond it."""
    centre = np.array([reference_gray.shape[1], reference_gray.shape[0]], dtype=np.float64) / 2
    middle = {name: np.array([x + w / 2, y + h / 2]) - centre for name, (x, y, w, h) in {**FIXED, **MAT}.items()}
    live_padded = cv2.copyMakeBorder(live_gray, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    first = coarse(reference_gray, live_gray)
    if first is None:
        return Offset(0.0, 0.0, 0.0, 0.0, 0.0, {n: (0.0, 0.0, False) for n in {**FIXED, **MAT}}, found=False)
    found = {name: _shift(reference_gray, live_padded, FIXED[name], moved(middle[name], *first), SEARCH) for name in FIXED}
    shifts = {name: (dx, dy) for name, (dx, dy, _) in found.items()}
    used = [name for name, (_, _, ok) in found.items() if ok]
    if len(used) < 2:
        windows = {n: (*shifts[n], False) for n in FIXED} | {n: (0.0, 0.0, False) for n in MAT}
        return Offset(first[0], first[1], float(np.degrees(first[2])), 0.0, 0.0, windows, found=False)
    dx, dy, turn = similarity(np.array([middle[n] for n in used]), np.array([shifts[n] for n in used]))
    misses = {n: np.hypot(*(np.array(shifts[n]) - moved(middle[n], dx, dy, turn))) for n in used}
    worst = max(misses, key=misses.get)
    if misses[worst] > OUTLIER_PX and len(used) > 2:
        used.remove(worst)
        dx, dy, turn = similarity(np.array([middle[n] for n in used]), np.array([shifts[n] for n in used]))
    shifts |= {n: _shift(reference_gray, live_padded, MAT[n], moved(middle[n], dx, dy, turn), MAT_SEARCH)[:2] for n in MAT}
    # A straight edge shows only how far it moved across itself: along it, the match is noise.
    normals = np.array([edge_normal(reference_gray, MAT[n]) for n in MAT])
    across = np.array([normal @ (np.array(shifts[n]) - moved(middle[n], dx, dy, turn)) for normal, n in zip(normals, MAT)])
    mat, *_ = np.linalg.lstsq(normals, across, rcond=None)
    windows = {n: (*shifts[n], n in used or n in MAT) for n in shifts}
    return Offset(dx, dy, float(np.degrees(turn)), float(mat[0]), float(mat[1]), windows)


def edge_normal(image: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """The unit direction across the strongest edge in a window: the dominant gradient direction."""
    x, y, w, h = window
    patch = image[y : y + h, x : x + w]
    gx, gy = cv2.Sobel(patch, cv2.CV_32F, 1, 0), cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    values, vectors = np.linalg.eigh(np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]]))
    return vectors[:, np.argmax(values)]


def moved(point: np.ndarray, dx: float, dy: float, turn: float) -> np.ndarray:
    """How far a point (relative to the picture's middle) moves under a shift and a small clockwise turn
    of `turn` radians, in a picture whose y points down."""
    return np.array([dx - turn * point[1], dy + turn * point[0]])


def similarity(points: np.ndarray, shifts: np.ndarray) -> tuple[float, float, float]:
    """The shift and small turn (radians) that best move `points` by `shifts`, by least squares."""
    rows = np.concatenate([[[1.0, 0.0, -y], [0.0, 1.0, x]] for x, y in points])
    (dx, dy, turn), *_ = np.linalg.lstsq(rows, shifts.reshape(-1), rcond=None)
    return float(dx), float(dy), float(turn)


GREEN, ORANGE, BLUE, GREY = (0, 220, 0), (0, 160, 255), (255, 170, 0), (150, 150, 150)  # BGR


def overlay(live: np.ndarray, reference_rgb: np.ndarray, offset: Offset, blend: bool) -> np.ndarray:
    """The live picture (RGB in, BGR out for `cv2.imshow`) with the reference's outlines in green, each
    window and its shift, arrows the way the picture and the mat have to slide, and what to do."""
    picture = cv2.addWeighted(live, 0.5, reference_rgb, 0.5, 0) if blend else live.copy()
    picture = np.ascontiguousarray(picture[:, :, ::-1])
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0), 40, 100) > 0
    picture[edges] = GREEN
    for name, (x, y, w, h) in {**FIXED, **MAT}.items():
        dx, dy, used = offset.windows[name]
        colour = BLUE if name in MAT else GREY if not used else GREEN if np.hypot(dx, dy) <= ALIGNED_PX else ORANGE
        cv2.rectangle(picture, (x, y), (x + w - 1, y + h - 1), colour, 1)
        _text(picture, f"{dx:+.1f},{dy:+.1f}" + ("" if used else " (left out)"), (x + 3, y + h - 5), 0.4, colour)
    middle = np.array([picture.shape[1] // 2, picture.shape[0] // 2])
    if not offset.camera_aligned:
        _arrow(picture, middle, -offset.dx, -offset.dy, ORANGE)
    if not offset.mat_aligned:
        for x, y, w, h in MAT.values():
            _arrow(picture, np.array([x + w // 2, y + h // 2]), -offset.mat_dx, -offset.mat_dy, BLUE)
    colour = GREEN if offset.camera_aligned and offset.mat_aligned else ORANGE
    if offset.found:
        _text(picture, f"camera moved {offset.dx:+.1f} px right, {offset.dy:+.1f} px down, turned {offset.degrees:+.2f} deg clockwise", (10, 22), 0.5, colour)
    for row, line in enumerate(offset.advice()):
        _text(picture, line, (10, 46 + 22 * row), 0.55, colour)
    _text(picture, "green lines: the demos' view   b: blend with it   s: save   q: quit", (10, picture.shape[0] - 10), 0.45, (255, 255, 255))
    return picture


def _arrow(picture: np.ndarray, start: np.ndarray, dx: float, dy: float, colour: tuple[int, int, int]) -> None:
    size = np.hypot(dx, dy)
    if size < 0.5:
        return
    end = start + np.array([dx, dy]) / size * min(max(12.0, 6 * size), 120.0)
    cv2.arrowedLine(picture, tuple(int(v) for v in start), tuple(int(v) for v in end), (0, 0, 0), 5, tipLength=0.3)
    cv2.arrowedLine(picture, tuple(int(v) for v in start), tuple(int(v) for v in end), colour, 2, tipLength=0.3)


def _text(picture: np.ndarray, text: str, origin: tuple[int, int], scale: float, colour: tuple[int, int, int]) -> None:
    cv2.putText(picture, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(picture, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


class Steady:
    """The median of the last few readings, so the numbers hold still while the camera does."""

    def __init__(self, size: int = SMOOTHING):
        self.size, self.recent = size, []

    def add(self, offset: Offset) -> Offset:
        self.recent = [*self.recent[-(self.size - 1) :], offset]
        keys = ("dx", "dy", "degrees", "mat_dx", "mat_dy")
        values = [float(np.median([getattr(o, key) for o in self.recent])) for key in keys]
        return Offset(*values, offset.windows, offset.found and all(o.found for o in self.recent))
