"""A camera with a wide lens: where it is, how its lens bends the picture, and how to draw through it.

MuJoCo renders straight-line (pinhole) pictures, and both real cameras bend straight lines. So each is
drawn as a wider pinhole picture and then resampled through the lens. The lens is the Kannala-Brandt
model, which covers a plain pinhole (k1 = 1/3, k2 = 2/15) through to a fisheye (k1 = k2 = 0): a ray
`theta` radians off the axis lands `focal * theta * (1 + k1 theta^2 + k2 theta^4)` pixels from the
image's centre.

A pose is the camera's position and rotation vector in MuJoCo's camera axes: x right, y up, looking
down -z.
"""

from dataclasses import dataclass

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
# The widest angle off the axis a pinhole render is asked to cover. Past this the render would need
# several times the real image's pixels, and neither camera sees that far.
MAX_THETA = 1.3


def bend(theta, k1: float, k2: float):
    return theta * (1 + k1 * theta**2 + k2 * theta**4)


def widest(k1: float, k2: float) -> float:
    """The widest angle off the axis the lens still spreads out, up to MAX_THETA. A fitted lens only
    describes the angles its picture holds; past them its formula can fold back on itself."""
    theta = np.linspace(0.0, MAX_THETA, 261)
    folded = np.flatnonzero(1 + 3 * k1 * theta**2 + 5 * k2 * theta**4 < 0.2)
    return float(theta[folded[0] - 1]) if len(folded) else MAX_THETA


def unbend(bent: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """The angle off the axis that lands at `bent`, by Newton's method. No further than `widest`."""
    limit = widest(k1, k2)
    bent = np.minimum(np.asarray(bent, dtype=np.float64), bend(limit, k1, k2))
    theta = np.minimum(bent, limit)
    for _ in range(12):
        slope = 1 + 3 * k1 * theta**2 + 5 * k2 * theta**4
        theta = np.clip(theta - (bend(theta, k1, k2) - bent) / np.maximum(slope, 0.2), 0.0, limit)
    return theta


def rotation(vector) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.asarray(vector, dtype=np.float64))
    return matrix


def rotation_quat(vector) -> np.ndarray:
    import mujoco  # only here, so the lens works without the sim extra

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation(vector).ravel())
    return quat


@dataclass(frozen=True)
class Lens:
    focal: float  # pixels, at 640 x 480
    k1: float
    k2: float
    width: int = WIDTH
    height: int = HEIGHT
    # Where the lens's axis meets the picture, in pixels right of and below its middle. A cheap camera's
    # lens rarely sits square over its sensor.
    centre: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_scene(cls, lens: dict, width: int = WIDTH, height: int = HEIGHT) -> "Lens":
        """The lens saved in a scene, for pictures `width` wide."""
        scale = width / WIDTH
        centre = lens.get("centre", (0.0, 0.0))
        return cls(lens["focal"] * scale, lens["k1"], lens["k2"], width, height, (centre[0] * scale, centre[1] * scale))

    def to_scene(self) -> dict:
        scale = WIDTH / self.width
        return {
            "focal": float(self.focal * scale), "k1": float(self.k1), "k2": float(self.k2),
            "centre": [float(self.centre[0] * scale), float(self.centre[1] * scale)],
        }  # fmt: skip

    @property
    def _axis(self) -> tuple[float, float]:
        """The pixel the lens's axis meets, counting pixels from their centres."""
        return self.width / 2 - 0.5 + self.centre[0], self.height / 2 - 0.5 + self.centre[1]

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixels of points given in the camera's axes."""
        x, y, z = points[..., 0], -points[..., 1], -points[..., 2]
        radius = np.hypot(x, y)
        scale = self.focal * bend(np.arctan2(radius, z), self.k1, self.k2) / np.maximum(radius, 1e-12)
        return np.stack([self._axis[0] + scale * x, self._axis[1] + scale * y], -1)

    def _tangents(self, pixels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """For each pixel, or each of `pixels`, its ray's slopes right and down: where it would land on a
        pinhole image of focal 1 looking the same way."""
        if pixels is None:
            u, v = np.meshgrid(np.arange(self.width) - self._axis[0], np.arange(self.height) - self._axis[1])
        else:
            u, v = pixels[..., 0] - self._axis[0], pixels[..., 1] - self._axis[1]
        radius = np.hypot(u, v)
        scale = np.tan(unbend(radius / self.focal, self.k1, self.k2)) / np.maximum(radius, 1e-12)
        return u * scale, v * scale

    def rays(self, pixels: np.ndarray | None = None) -> np.ndarray:
        """Each pixel's ray, or each of `pixels`', in the camera's axes, with -1 along the view direction."""
        right, down = self._tangents(pixels)
        return np.stack([right, -down, -np.ones_like(right)], -1)

    def render_fovy(self) -> float:
        """The vertical field of view, in degrees, of a pinhole picture of this shape that covers every pixel."""
        right, down = self._tangents()
        reach = max(float(np.abs(down).max()), float(np.abs(right).max()) * self.height / self.width)
        return float(np.degrees(2 * np.arctan(reach * 1.02)))

    def render_map(self, render_width: int, render_height: int) -> tuple[np.ndarray, np.ndarray]:
        """For each pixel, where to read a pinhole render of `render_fovy()` and this size."""
        right, down = self._tangents()
        focal = 0.5 * render_height / np.tan(np.radians(self.render_fovy()) / 2)
        return (
            (render_width / 2 + right * focal - 0.5).astype(np.float32),
            (render_height / 2 + down * focal - 0.5).astype(np.float32),
        )


def to_camera(points: np.ndarray, pose) -> np.ndarray:
    """World points in the axes of a camera at `pose` (position, then rotation vector)."""
    return (np.asarray(points) - np.asarray(pose[:3])) @ rotation(pose[3:6])


def look_through(image: np.ndarray, pose, lens: Lens, points: np.ndarray) -> np.ndarray:
    """What a camera sees at each world point: the colour of the pixel it lands on."""
    pixels = lens.project(to_camera(points, pose)).astype(np.float32)
    return cv2.remap(image, pixels[..., 0], pixels[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
