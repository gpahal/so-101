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


# The block's hue is within a few degrees of pure red. The table's wood is orange, at 8 degrees and up
# through cameras that saturate it to (186, 56, 1) against the block's (171, 1, 1): over 120 frames of each
# camera, the hues of pixels redder than 2.1 times their green fell into a block peak at -2 to 2 degrees
# and a wood peak from 8, with the fewest between at 4 to 7.
RED_HUE_DEG = 7.0


def hue_degrees(image: np.ndarray) -> np.ndarray:
    """Each pixel's hue in degrees, from -180 to 180, red at 0."""
    hue = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)[..., 0].astype(np.float32) * (360 / 256)
    return np.where(hue > 180, hue - 360, hue)


def saturation(image: np.ndarray) -> np.ndarray:
    """How far each pixel is from grey, 0 to 1: its spread over its brightest channel."""
    a = image.astype(np.float32)
    return (a.max(axis=2) - a.min(axis=2)) / np.maximum(a.max(axis=2), 1)


def red_mask(image: np.ndarray) -> np.ndarray:
    """The block: redder than it is green or blue, even on its dull, shaded sides, and not the table's
    orange or the jar's amber. With a camera that keeps colours muted those have about half as much green
    as red; with one that saturates them they can have under a third, and only their hue tells them apart."""
    a = image.astype(np.int16)
    mask = (a[..., 0] - np.maximum(a[..., 1], a[..., 2]) > 25) & (a[..., 0] > 55) & (a[..., 0] > 2.1 * a[..., 1])
    mask &= hue_degrees(image) < RED_HUE_DEG
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def red_blob(image: np.ndarray, min_area: int = 60) -> tuple[np.ndarray, int, np.ndarray] | None:
    return blob(red_mask(image), min_area)


def pale(image: np.ndarray) -> np.ndarray:
    """The arm's printed parts, and the wall: light and nearly grey. A warm camera tints both beige, to a
    saturation of up to 0.4, where the wood and the jar's amber are over 0.55."""
    a = image.astype(np.int16)
    return (a.min(axis=2) > 105) & (saturation(image) < 0.45)


def orange(image: np.ndarray) -> np.ndarray:
    """The table's bare wood, and the jar's amber with it: much redder than blue, and strongly coloured,
    which a beige wall or arm isn't."""
    a = image.astype(np.int16)
    return (a[..., 0] - a[..., 2] > 60) & (saturation(image) > 0.6)


def whole(box, margin: int = 2) -> bool:
    """A bounding box that isn't cut off by the picture's edge."""
    return bool(box[0] > margin and box[1] > margin and box[0] + box[2] < WIDTH - margin and box[1] + box[3] < HEIGHT - margin)


def wrist_jar_mask(image: np.ndarray) -> np.ndarray:
    """The jar's tinted wall in a wrist frame: its largest amber patch. The wrist camera sees it far more
    vivid than the top camera does."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hue_degrees(image)
    # Its hue is 24 to 60 degrees through a muted camera and 10 to 20 through one that saturates colours.
    # The wood there is as orange, but much brighter, and the largest patch is the jar's.
    amber = (hue >= 9) & (hue <= 60) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 90) & ~((hsv[..., 2] > 220) & (hsv[..., 1] > 220))
    amber = cv2.morphologyEx(amber.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(amber)
    if count < 2:
        return np.zeros(amber.shape, bool)
    return labels == 1 + int(np.argmax(stats[1:, 4]))
