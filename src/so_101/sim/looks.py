"""What things look like in the real cameras' frames: which pixels are the block, the jar and the arm."""

import cv2
import numpy as np

from so_101.sim.camera import HEIGHT, WIDTH


def blob(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int, np.ndarray] | None:
    """Centre, area and bounding box of a mask's largest patch."""
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, 4]))
    if stats[largest, 4] < min_area:
        return None
    return centres[largest], int(stats[largest, 4]), stats[largest, :4]


def red_mask(image: np.ndarray) -> np.ndarray:
    """The block: redder than it is green or blue, even on its dull, shaded sides, and not the table's
    orange or the jar's amber, which have about half as much green as red."""
    a = image.astype(np.int16)
    mask = (a[..., 0] - np.maximum(a[..., 1], a[..., 2]) > 25) & (a[..., 0] > 55) & (a[..., 0] > 2.1 * a[..., 1])
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def red_blob(image: np.ndarray, min_area: int = 60) -> tuple[np.ndarray, int, np.ndarray] | None:
    return blob(red_mask(image), min_area)


def pale(image: np.ndarray) -> np.ndarray:
    """The arm's printed parts: light and grey."""
    a = image.astype(np.int16)
    return (a.min(axis=2) > 105) & ((a.max(axis=2) - a.min(axis=2)) < 40)


def whole(box, margin: int = 2) -> bool:
    """A bounding box that isn't cut off by the picture's edge."""
    return bool(box[0] > margin and box[1] > margin and box[0] + box[2] < WIDTH - margin and box[1] + box[3] < HEIGHT - margin)


def wrist_jar_mask(image: np.ndarray) -> np.ndarray:
    """The jar's tinted wall in a wrist frame: its largest amber patch. The wrist camera sees it far more
    vivid than the top camera does."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    amber = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 90)
    amber = cv2.morphologyEx(amber.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(amber)
    if count < 2:
        return np.zeros(amber.shape, bool)
    return labels == 1 + int(np.argmax(stats[1:, 4]))
