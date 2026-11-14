"""Work out a task's scene from its demos: the cameras, the arm's joint offsets, the table, block and jar.

Nothing here is measured by hand. The top camera never moves, so the arm's outline and the red block in
its frames, against where the joint angles say they are, give the camera, its lens and each joint's
offset. A demo's first frame shows where its block and jar stood. The wrist camera's place on the
gripper comes from where it sees that block and jar, and its own jaws.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from so_101.sim import arm
from so_101.sim.camera import HEIGHT, WIDTH, Lens, look_through, rotation, rotation_quat, to_camera
from so_101.sim.arm import GRASP_DEPTH, TIPS_ABOVE_MAT, held_block
from so_101.sim.demos import FPS, Demos
from so_101.sim.env import GROUP_BLOCK, BlockContainerEnv, jar_outline, table_points
from so_101.sim.looks import blob, pale, red_blob, red_mask, whole, wrist_jar_mask

# How many ticks older than the joints saved with it a top camera picture is taken to be while the camera
# is being fitted, before `camera_timing` can measure it: what it measured on the first 100 demos. Fitting
# on still frames alone needs no guess, but leaves too few poses: the outline then fitted 5.4 px out, not 3.8.
TOP_DELAY_GUESS = 3
# Degrees a second under which the arm counts as still: the wrist camera's delay isn't known, so its fits
# use only frames where it can't matter.
STILL = 5.0


def calm_frame(state: np.ndarray, index: int, window: int = 12, limit: float = 20.0) -> int | None:
    """The frame near `index` where the arm moves least, or None if it's moving faster than `limit`
    degrees a second all through the window. A camera's picture is a few ticks older than the joints'
    readings saved with it, which only matters while the arm moves: at 20 degrees a second and 3 ticks,
    the jaws are a centimetre from where the picture shows them."""
    speed = np.abs(np.gradient(state[:, :5], axis=0)).max(axis=1) * FPS
    first, last = max(index - window, 0), min(index + window + 1, len(state))
    calmest = first + int(np.argmin(speed[first:last]))
    return calmest if speed[calmest] <= limit else None


def held_demos(demos: Demos) -> dict[int, tuple[int, int]]:
    """Each demo that holds the block, with its grasp and release frames."""
    found = {episode: demos.grasp_and_release(episode) for episode in range(len(demos))}
    return {episode: frames for episode, frames in found.items() if frames is not None}


def split(episodes: list[int]) -> tuple[list[int], list[int]]:
    """Every fourth demo is held back, for `so101 sim score` to check a fit on demos it didn't see."""
    return [e for i, e in enumerate(episodes) if i % 4 != 2], [e for i, e in enumerate(episodes) if i % 4 == 2]


# The top camera and the arm's joint offsets.


def dark_scene(demos: Demos, episodes: list[int]) -> np.ndarray:
    """The top camera's view with the pale arm mostly gone: each pixel's darker moments."""
    stack = []
    for episode in episodes:
        length = len(demos.episode(episode)[0])
        stack += [demos.frame(episode, int(f)) for f in np.linspace(5, length - 5, 5)]
    return np.percentile(np.stack(stack), 15, axis=0).astype(np.uint8)


def arm_region(dark: np.ndarray) -> np.ndarray:
    """Where the pale arm can be told from what's behind it: over the mat, the dark floor and the orange table."""
    a = dark.astype(np.int16)
    region = (a.max(axis=2) < 80) | ((a[..., 0] - a[..., 2]) > 60)
    return cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def arm_mask(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx((pale(image) & region).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


@dataclass
class TopFit:
    """What `fit_top_camera` compares a candidate camera and joint offsets with."""

    outline_states: np.ndarray  # the follower's readings in each outline frame
    outline_distance: list[np.ndarray]  # per frame and pixel, how far the real arm's nearest pixel is
    outline_pixels: list[np.ndarray]  # per frame, a sample of the real arm's pixels
    region: np.ndarray
    block_states: np.ndarray  # the readings that put the jaws round the block in each block sighting
    block_pixels: np.ndarray  # where the top camera saw it
    resting: np.ndarray  # which sightings are of the block on the mat, before its grasp
    block_areas: np.ndarray  # how many pixels the block covered in each resting sighting
    block_corners: np.ndarray  # the block's corners about its centre, as it lay in each resting sighting
    cap: float = 40.0

    def residuals(self, pose, lens: Lens, offsets, gripper_closed: float, deg_per_percent: float, block_height: float) -> dict[str, np.ndarray]:
        """`block_height` is how high a block's centre is when it lies on the mat, in the arm's axes."""
        units = arm.Units(tuple(offsets), gripper_closed, deg_per_percent)
        points = lens.project(to_camera(arm.printed_points(units.to_qpos(self.outline_states)), pose))
        region = self.region.astype(np.float32)
        stray, missed = [], []
        for distance, real, drawn in zip(self.outline_distance, self.outline_pixels, points):
            x, y = drawn[:, 0].astype(np.float32)[None], drawn[:, 1].astype(np.float32)[None]
            judged = cv2.remap(region, x, y, cv2.INTER_LINEAR, borderValue=0)[0]
            stray.append(cv2.remap(distance, x, y, cv2.INTER_LINEAR, borderValue=0)[0] * judged)
            missed.append(np.minimum(cKDTree(drawn).query(real)[0], self.cap))
        block = held_block(units, self.block_states)
        # How big each block on the mat would look: the area its corners enclose in the picture.
        centres = np.c_[block[self.resting, :2], np.full(self.resting.sum(), block_height)]
        outlines = lens.project(to_camera(centres[:, None] + self.block_corners, pose)).astype(np.float32)
        drawn_areas = np.array([cv2.contourArea(cv2.convexHull(outline)) for outline in outlines])
        return {
            "stray": np.concatenate(stray),  # the drawn arm where the real one isn't, in pixels
            "missed": np.concatenate(missed),  # the real arm where the drawn one isn't
            "block": np.clip(lens.project(to_camera(block, pose)) - self.block_pixels, -30, 30).ravel(),
            "height": block[self.resting, 2],  # the grasped block's centre, which sat on the mat
            "size": np.sqrt(drawn_areas / self.block_areas) - 1,  # how much too big the block is drawn
        }


def block_corners(block: dict, yaw: float) -> np.ndarray:
    """The corners of a block lying flat, studs included, about its body's centre and turned by `yaw`."""
    half, low, high = block["width"] / 2, -block["height"] / 2, block["height"] / 2 + block["stud"]["height"]
    reach = block["stud"]["pitch"] / 2 + block["stud"]["radius"]
    corners = [[x, y, z] for x in (-half, half) for y in (-half, half) for z in (low, block["height"] / 2)]
    corners += [[x, y, high] for x in (-reach, reach) for y in (-reach, reach)]
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    return np.asarray(corners) @ turn.T


def top_fit_data(demos: Demos, episodes: list[int], held: dict[int, tuple[int, int]], block: dict, units: arm.Units, seed: int = 0) -> TopFit:
    """`units` only turns each block square to the jaws that later held it, so its offsets can be zero."""
    rng = np.random.default_rng(seed)
    region = arm_region(dark_scene(demos, episodes[:: max(1, len(episodes) // 25)]))
    states, distances, pixels = [], [], []
    for episode in episodes[:: max(1, len(episodes) // 25)]:
        state = demos.episode(episode)[0]
        for moment in (0.03, 0.2, 0.38, 0.52, 0.66, 0.8):
            index = calm_frame(state, int(moment * len(state)))
            if index is None:
                continue
            mask = arm_mask(demos.frame(episode, index), region)
            if mask.sum() < 300:
                continue
            index = max(index - TOP_DELAY_GUESS, 0)  # the joints the picture shows
            ys, xs = np.nonzero(mask)
            pick = rng.choice(len(xs), 120, replace=len(xs) < 120)
            states.append(state[index])
            distances.append(np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3), 40.0))
            pixels.append(np.c_[xs[pick], ys[pick]].astype(np.float64))
    block_states, block_pixels, resting, areas, corners = [], [], [], [], []
    for episode in episodes:
        if episode not in held:
            continue
        hold, let_go = held[episode]
        state = demos.episode(episode)[0]
        carried = {calm_frame(state, int(index), window=6) for index in np.linspace(hold + 20, let_go - 3, 6)} - {None}
        sightings = [(index, hold, True) for index in (0, 5, 10)] + [(index, index - TOP_DELAY_GUESS, False) for index in sorted(carried) if hold + 10 < index < let_go]
        for index, pose_index, on_mat in sightings:
            seen = red_blob(demos.frame(episode, index))
            if seen is not None and whole(seen[2]):
                block_states.append(state[pose_index])
                block_pixels.append(seen[0])
                resting.append(on_mat)
                if on_mat:
                    areas.append(seen[1])
                    corners.append(block_corners(block, jaw_yaw(units, state[hold])))
    return TopFit(np.array(states), distances, pixels, region, np.array(block_states), np.array(block_pixels), np.array(resting), np.array(areas, dtype=np.float64), np.array(corners))


def _affine_camera(points: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """A first guess at a camera from 3D points and their pixels, ignoring perspective: which way it
    faces (its rotation vector), where the points' centre lands, and how many pixels a metre covers there."""
    centre = points.mean(axis=0)
    solved, *_ = np.linalg.lstsq(np.c_[points - centre, np.ones(len(points))], pixels, rcond=None)
    right, down = solved[:3, 0], solved[:3, 1]
    scale = (np.linalg.norm(right) + np.linalg.norm(down)) / 2
    right = right / np.linalg.norm(right)
    down = down - (down @ right) * right
    down /= np.linalg.norm(down)
    to_world = np.column_stack([right, -down, -np.cross(right, down)])  # MuJoCo's camera axes as columns
    vector, _ = cv2.Rodrigues(to_world)
    return vector.ravel(), solved[3], scale


# How far each number moves in one optimiser step: metres, radians, pixels of focal length, lens terms,
# degrees of joint offset, metres of mat height, and pixels of the lens's centre.
TOP_STEPS = np.r_[np.full(3, 0.004), np.full(3, 0.004), 4.0, 0.02, 0.02, np.full(4, 0.25), 0.002, 2.0, 2.0]
LENS_CENTRE_SLACK = 25.0  # pixels the lens's axis can plausibly be off the picture's middle


def offset_slack(limits: dict[str, list[float]]) -> np.ndarray:
    """How far each arm joint's offset can plausibly be from zero, in degrees, shoulder_lift to wrist_roll.

    The follower's zero is the middle of its calibrated range and the model's is the middle of its own,
    so if both ranges ended at the same stops the offset would be zero. It can be half the difference
    of the ranges' widths, plus a degree and a half for how the stops were found. The wrist's roll turns
    freely, so its middle is wherever calibration left it.
    """
    model = arm.arm_model()
    slack = []
    for name in arm.ARM_JOINTS[1:]:
        ours, real = np.degrees(np.ptp(model.joint(name).range)), limits[name][1] - limits[name][0]
        slack.append(10.0 if name == "wrist_roll" else 1.5 + abs(real - ours) / 2)
    return np.array(slack)


def fit_top_camera(data: TopFit, gripper_closed: float, deg_per_percent: float, block: dict, table_height: float | None, limits: dict, report=print) -> dict:
    """Fit the top camera's pose and lens, the arm's joint offsets and the mat's height together.

    The drawn arm has to cover the real arm's pixels in frames spread over the demos, the block has to
    appear where the jaws later closed on it and as big as it really looks, and that grasp has to sit on
    the mat. Each alone leaves something open: outlines barely change when the camera backs off and zooms
    in, the block's sightings are nearly flat, its size says how far the mat is from the camera but not
    from the arm, and only a grasp on the mat ties the arm's reach to it.

    `table_height` is the mat's top in the arm model's axes, if it was measured (the model's base has its
    underside at `arm.base_underside()`); otherwise it's fitted.
    `limits` is each joint's calibrated range, which says how far an offset can plausibly be from zero
    (`offset_slack`). Offsets are pulled towards zero by that measure, but only weakly: on the first 100
    demos the arm's outline wanted 6.5 degrees at the elbow, where the ranges allow under 2, and a pull
    strong enough to refuse that left the outline 5.1 px out instead of 3.8.
    The shoulder's pan has no offset: turning the arm and turning the camera round it look the same.
    """
    units = arm.Units((0.0,) * 5, gripper_closed, deg_per_percent)
    grasped = held_block(units, data.block_states)
    facing, centre_pixel, scale = _affine_camera(grasped, data.block_pixels)
    centre = grasped.mean(axis=0)
    to_world = rotation(facing)
    block_centre = block["height"] / 2  # above the mat, of a block lying on it
    slack = offset_slack(limits)
    # The block's size in the picture only counts when it has to place the mat: 1% then counts as 1.5 pixels.
    size_weight = 0.0 if table_height is not None else 150.0

    def unpack(x):
        return x[:6], Lens(x[6], x[7], x[8], centre=(x[14], x[15])), np.r_[0.0, x[9:13]], x[13]

    def errors(x) -> np.ndarray:
        pose, lens, offsets, height = unpack(x)
        found = data.residuals(pose, lens, offsets, gripper_closed, deg_per_percent, height + block_centre)
        # Each kind of error counts by its mean square, whatever its number of terms.
        return np.concatenate([
            found["stray"] / np.sqrt(len(found["stray"])),
            found["missed"] / np.sqrt(len(found["missed"])),
            found["block"] * (1.5 / np.sqrt(len(found["block"]))),
            (found["height"] - (height + TIPS_ABOVE_MAT + GRASP_DEPTH)) * (2000 / np.sqrt(len(found["height"]))),  # 1 mm counts as 2 pixels
            found["size"] * (size_weight / np.sqrt(len(found["size"]))),
            offsets[1:] / slack * 0.3,
            x[14:16] / LENS_CENTRE_SLACK * 2.0,
        ])  # fmt: skip

    def solve(x, free, evaluations):
        steps = TOP_STEPS[free]

        def moved(z):
            trial = x.copy()
            trial[free] += z * steps
            return errors(trial)

        result = least_squares(moved, np.zeros(len(steps)), diff_step=0.5, max_nfev=evaluations)
        x = x.copy()
        x[free] += result.x * steps
        return x, float(2 * result.cost)

    measured = table_height is not None
    camera_only = np.r_[np.arange(9), [] if measured else [13]].astype(int)
    everything = np.r_[np.arange(13), [] if measured else [13], [14, 15]].astype(int)
    best = None
    # Blocks alone can't tell a near camera with a wide lens from a far one with a long lens, so try a few.
    for distance in (0.3, 0.4, 0.5, 0.65):
        position = centre + to_world[:, 2] * distance  # a camera looks down its -z
        position -= (to_world[:, 0] * (centre_pixel[0] - WIDTH / 2) - to_world[:, 1] * (centre_pixel[1] - HEIGHT / 2)) / scale
        x = np.r_[position, facing, scale * distance, 0.15, 0.0, np.zeros(4), table_height if measured else float(np.median(grasped[data.resting, 2])) - TIPS_ABOVE_MAT - GRASP_DEPTH, 0.0, 0.0]
        x, cost = solve(x, camera_only, 30)
        report(f"  camera {distance:.2f} m from the blocks: mismatch {cost:.2f}")
        if best is None or cost < best[1]:
            best = (x, cost)
    x, cost = best
    for _ in range(6):
        x, improved = solve(x, everything, 40)
        if improved > cost * 0.995:
            break
        cost = improved
    pose, lens, offsets, height = unpack(x)
    found = data.residuals(pose, lens, offsets, gripper_closed, deg_per_percent, height + block_centre)
    return {
        "pose": [float(v) for v in pose],
        "lens": lens.to_scene(),
        "offsets_deg": [round(float(v), 3) for v in offsets],
        "table_height": round(float(height), 4),
        "errors": {
            "outline_px": round(float((found["stray"].mean() + found["missed"].mean()) / 2), 2),
            "block_px": round(float(np.median(np.linalg.norm(found["block"].reshape(-1, 2), axis=1))), 2),
            "grasp_height_mm": [round(float(v * 1000), 1) for v in (found["height"].mean() - height, found["height"].std())],
        },
    }


def on_plane(pose, lens: Lens, pixels: np.ndarray, height: float) -> np.ndarray:
    """Where the rays through pixels of a camera at `pose` meet the level `height`, in the arm's axes."""
    rays = lens.rays(np.atleast_2d(pixels).astype(np.float64)) @ rotation(pose[3:6]).T
    along = (height - pose[2]) / rays[:, 2]
    return np.asarray(pose[:3]) + rays * along[:, None]


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=1))


# The scene the top camera sees: a clean picture of it, and where each demo's block and jar stood.


class ArmView:
    """Draws which pixels the arm covers from a camera, without the rest of the scene."""

    def __init__(self, pose, lens: Lens, units: arm.Units):
        spec = mujoco.MjSpec.from_file(str(arm.arm_xml()))
        spec.worldbody.add_camera(name="view", pos=pose[:3], quat=rotation_quat(pose[3:6]).tolist(), fovy=lens.render_fovy())
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 2 * lens.width, 2 * lens.height
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.units = units
        self.renderer = mujoco.Renderer(self.model, 2 * lens.height, 2 * lens.width)
        self.renderer.enable_segmentation_rendering()
        self.map = lens.render_map(2 * lens.width, 2 * lens.height)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[2] = 1  # the arm's visual meshes
        self.qadr = [self.model.joint(name).qposadr[0] for name in arm.JOINTS]
        self.is_base = self.model.geom_bodyid == self.model.body("base").id

    def shapes(self, state: np.ndarray) -> np.ndarray:
        """Which of the arm's geoms each pixel shows, -1 for none."""
        self.data.qpos[self.qadr] = self.units.to_qpos(state)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="view", scene_option=self.option)
        return cv2.remap(self.renderer.render()[:, :, 0].astype(np.float32), *self.map, cv2.INTER_NEAREST, borderValue=-1).astype(np.int32)

    def moving(self, state: np.ndarray) -> np.ndarray:
        """The pixels covered by the parts of the arm that move."""
        shapes = self.shapes(state)
        return (shapes >= 0) & ~self.is_base[shapes.clip(0)]

    def close(self) -> None:
        self.renderer.close()


def clean_plate(demos: Demos, episodes: list[int], view: ArmView, jar_masks: dict[int, np.ndarray]) -> np.ndarray:
    """The top camera's view with no arm, block or jar: each pixel's median over the frames where
    nothing covers it. Behind the jar, which never leaves, the mat and wall around it are painted in."""
    grow = np.ones((13, 13), np.uint8)
    frames, hidden = [], []
    for episode in episodes:
        state = demos.episode(episode)[0]
        for moment in (0.0, 0.45, 0.7):
            index = int(moment * (len(state) - 1))
            frame = demos.frame(episode, index)
            cover = view.moving(state[index]) | red_mask(frame) | jar_masks[episode]
            frames.append(frame)
            hidden.append(cv2.dilate(cover.astype(np.uint8), grow) > 0)
    frames, hidden = np.stack(frames), np.stack(hidden)
    plate = np.zeros(frames.shape[1:], np.uint8)
    never = np.zeros(frames.shape[1:3], bool)
    for row in range(frames.shape[1]):
        values = np.where(hidden[:, row, :, None], np.nan, frames[:, row].astype(np.float32))
        seen = (~hidden[:, row]).sum(axis=0) >= 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a pixel hidden in every frame: painted in below
            plate[row] = np.nan_to_num(np.nanmedian(values, axis=0)).astype(np.uint8)
        never[row] = ~seen
    return cv2.inpaint(plate, never.astype(np.uint8), 6, cv2.INPAINT_TELEA)


def jar_pixels(image: np.ndarray, near: np.ndarray, reach: float = 85.0) -> np.ndarray:
    """The jar in a top camera frame: its amber, with the dark thread round its neck filled in. `near` is
    a pixel on it, and nothing further than `reach` pixels from there counts, which leaves out the table."""
    a = image.astype(np.int16)
    amber = (a[..., 0] - a[..., 2] > 22) & (a[..., 0] >= a[..., 1]) & (a[..., 1] >= a[..., 2] - 2) & (a[..., 0] > 55) & (a[..., 0] < 2 * a[..., 1])
    v, u = np.indices(amber.shape)
    amber &= np.hypot(u - near[0], v - near[1]) < reach
    amber = cv2.morphologyEx(amber.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(amber)
    if count < 2:
        return np.zeros(amber.shape, bool)
    largest = labels == 1 + int(np.argmax(stats[1:, 4]))
    return cv2.morphologyEx(largest.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def fit_jars(demos: Demos, episodes: list[int], pose, lens: Lens, jar: dict, table_height: float, guess_xy, measured: bool) -> tuple[dict, dict[int, list[float]]]:
    """Where the jar stood in each demo, from its first frame, and the jar's size unless it was measured.

    The jar's amber is everything below the far side of its rim, so its outline there is a cylinder as
    tall as the jar. The size that fits the demos' frames best is found first, then each demo's place.
    """
    near = lens.project(to_camera(np.array([[*guess_xy, table_height + jar["height"] / 2]]), pose))[0]
    real = {episode: jar_pixels(demos.frame(episode, 0), near) for episode in episodes}

    def place(mask, jar, start):
        def missing(xy):
            return 1 - _overlap(jar_outline(pose, lens, xy, jar, table_height), mask)

        best, loss = np.asarray(start, dtype=np.float64), missing(start)
        for step in (0.008, 0.004, 0.002, 0.001, 0.0005):
            improved = True
            while improved:
                improved = False
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    trial = best + np.array([dx, dy]) * step
                    trial_loss = missing(trial)
                    if trial_loss < loss - 1e-6:
                        best, loss, improved = trial, trial_loss, True
        return best, loss

    jar = dict(jar)
    if not measured:
        sample = episodes[:: max(1, len(episodes) // 12)]
        best = None
        for radius in np.arange(0.026, 0.046, 0.002):
            for height in np.arange(0.05, 0.1, 0.005):
                trial = {**jar, "radius": float(radius), "height": float(height)}
                loss = float(np.mean([place(real[episode], trial, guess_xy)[1] for episode in sample]))
                if best is None or loss < best[0]:
                    best = (loss, trial)
        jar = best[1]
    places, losses = {}, []
    for episode in episodes:
        xy, loss = place(real[episode], jar, guess_xy)
        places[episode] = [round(float(v), 4) for v in xy]
        losses.append(loss)
    jar["outline_mismatch"] = round(float(np.mean(losses)), 3)
    return jar, places


def jar_cutout(demos: Demos, episodes: list[int], places: dict[int, list[float]], pose, lens: Lens, jar: dict, table_height: float) -> tuple[np.ndarray, list[float]]:
    """A photo of the jar to paste into the top camera's picture: RGBA at the camera's size, and where
    the jar stood for it. It's the median of the first frames of the demos whose jar stood closest to
    the middle of them all, so it's sharp."""
    spots = np.array([places[episode] for episode in episodes])
    home = np.median(spots, axis=0)
    order = np.argsort(np.linalg.norm(spots - home, axis=1))
    chosen = [episodes[i] for i in order[: max(5, len(episodes) // 10)]]
    home = np.mean([places[episode] for episode in chosen], axis=0)
    photo = np.median(np.stack([demos.frame(episode, 0) for episode in chosen]), axis=0).astype(np.uint8)
    cover = cv2.dilate(jar_outline(pose, lens, home, jar, table_height).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    cover = cv2.GaussianBlur(cover, (0, 0), 1.5)
    return np.dstack([photo, (cover * 255).astype(np.uint8)]), [round(float(v), 4) for v in home]


def table_edges(plate: np.ndarray, pose, lens: Lens, table_height: float) -> dict:
    """The mat's far edge, at the wall, and its left edge, in metres in the arm's axes: where the plate,
    laid flat on the table, stops being dark mat."""
    probe = {"x": [0.0, 0.7], "y": [-0.4, 0.4], "texel": 0.002, "height": table_height}
    flat = look_flat(plate, pose, lens, probe)
    dark = flat.max(axis=2) < 75
    xs = probe["x"][1] - (np.arange(dark.shape[0]) + 0.5) * probe["texel"]
    ys = probe["y"][1] - (np.arange(dark.shape[1]) + 0.5) * probe["texel"]
    # The far edge: the furthest dark texel in each column in front of the arm.
    far = [xs[np.argmax(dark[:, column])] for column in np.flatnonzero((ys > -0.2) & (ys < 0.03)) if dark[:, column].any()]
    # The left edge: the leftmost run of dark texels at least 10 cm long in each row over the far half of the mat.
    left = []
    for row in np.flatnonzero((xs > 0.2) & (xs < 0.36)):
        runs = np.flatnonzero(np.convolve(dark[row], np.ones(50), "valid") == 50)
        if len(runs):
            left.append(ys[runs[0]])
    return {"wall_x": round(float(np.median(far)), 4), "left_y": round(float(np.median(left)), 4)}


def wood_filled(flat: np.ndarray) -> np.ndarray:
    """The table texture for the wrist camera: the mat as the top camera saw it, and everything else the
    table's wood. From the top camera the bare table is only clearly wood near the arm; further away it
    mirrors the white wall, which the wrist camera, looking down, doesn't see."""
    a = flat.astype(np.int16)
    mat = cv2.morphologyEx((a.max(axis=2) < 75).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    wood = (a[..., 0] - a[..., 2] > 60) & ~mat
    filled = flat.copy()
    if wood.sum() > 100:
        filled[~mat & ~wood] = np.median(flat[wood], axis=0).astype(np.uint8)
    return filled


def look_flat(image: np.ndarray, pose, lens: Lens, table: dict) -> np.ndarray:
    """A camera's picture laid flat on the table: the table texture's texels, coloured by what it saw
    there. A texel outside its picture takes the colour of the nearest one inside."""
    points = table_points(table)
    flat = look_through(image, pose, lens, points)
    pixels = lens.project(to_camera(points, pose))
    unseen = (pixels[..., 0] < 0) | (pixels[..., 0] > lens.width - 1) | (pixels[..., 1] < 0) | (pixels[..., 1] > lens.height - 1)
    if unseen.any() and not unseen.all():
        _, nearest = cv2.distanceTransformWithLabels(unseen.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        seen_rows, seen_columns = np.nonzero(~unseen)
        # Labels number the seen texels in row order, from 1.
        flat[unseen] = flat[seen_rows[nearest[unseen] - 1], seen_columns[nearest[unseen] - 1]]
    return flat


# The wrist camera.


def wrist_start() -> tuple[np.ndarray, dict]:
    """Where the arm model mounts its wrist camera, as a first guess, with a lens a little wider than its own."""
    mount = arm.arm_model().camera("wrist_cam")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, mount.quat)
    vector, _ = cv2.Rodrigues(matrix.reshape(3, 3))
    return np.r_[mount.pos, vector.ravel()], {"focal": 330.0, "k1": 0.15, "k2": 0.0}


def wrist_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """In a real wrist frame: the block, and the pale jaws rising from the bottom edge. The jaws are None
    when the pale wall is in view too, since the two can't be told apart."""
    block = red_mask(image)
    light = pale(image).astype(np.uint8)
    if light[: image.shape[0] // 2].mean() > 0.04:
        return block, None
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(light)
    jaws = np.zeros(light.shape, bool)
    for label in range(1, count):
        if stats[label, 1] + stats[label, 3] >= image.shape[0] - 4 and stats[label, 4] > 400:  # touches the bottom
            jaws |= labels == label
    return block, jaws


def block_surface(block: dict, per_face: int = 40, seed: int = 0) -> np.ndarray:
    """Points spread over a block's faces, in its own axes, for drawing its outline without rendering."""
    rng = np.random.default_rng(seed)
    half = np.array([block["width"] / 2, block["width"] / 2, block["height"] / 2])
    points = []
    for axis in range(3):
        for side in (-1, 1):
            face = rng.uniform(-1, 1, (per_face, 3)) * half
            face[:, axis] = side * half[axis]
            points.append(face)
    return np.concatenate(points)


def jar_surface(jar: dict, count: int = 240, seed: int = 0) -> np.ndarray:
    """Points spread over the jar's tinted wall, in its own axes."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, count)
    return np.c_[jar["radius"] * np.cos(angle), jar["radius"] * np.sin(angle), rng.uniform(0, jar["tint_height"], count)]


@dataclass
class WristFit:
    """What `fit_wrist_camera` compares a candidate wrist camera with: frames before grasps, each with
    things as points in the gripper's axes, and where the real camera saw them. A thing is the jaws, the
    block or the jar: its points, how far each pixel is from its real pixels, and a sample of those."""

    frames: list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]
    cap: float = 60.0

    def residuals(self, pose, lens: Lens) -> dict[str, np.ndarray]:
        found: dict[str, list[np.ndarray]] = {}
        for frame in self.frames:
            for name, (points, distance, real) in frame.items():
                local = to_camera(points, pose)
                drawn = lens.project(local)
                # Only what's in front of the camera and in its picture is judged: it sees just the jaws'
                # tips, and often half the jar.
                seen = (local[:, 2] < -0.005) & (drawn[:, 0] > 0) & (drawn[:, 0] < lens.width - 1) & (drawn[:, 1] > 0) & (drawn[:, 1] < lens.height - 1)
                x, y = drawn[:, 0].astype(np.float32)[None], drawn[:, 1].astype(np.float32)[None]
                found.setdefault(f"{name}_stray", []).append(cv2.remap(distance, x, y, cv2.INTER_LINEAR, borderValue=0)[0] * seen)
                found.setdefault(f"{name}_missed", []).append(np.minimum(cKDTree(drawn[seen]).query(real)[0], self.cap) if seen.any() else np.full(len(real), self.cap))
        return {name: np.concatenate(values) for name, values in found.items()}


def wrist_fit_data(demos: Demos, held: dict, episodes: list[int], units: arm.Units, starts: dict[int, dict], scene: dict, seed: int = 0) -> WristFit:
    """`starts` is each demo's entry in the scene's `starts`: its block as [x, y, yaw] and its jar as [x, y]."""
    rng = np.random.default_rng(seed)
    model = arm.arm_model()
    surface = arm.printed_surface()
    jaw_body = model.body("moving_jaw_so101_v1")
    table_height, block, jar = scene["table"]["height"], scene["block"], scene["jar"]
    faces, wall = block_surface(block), jar_surface(jar)
    data = WristFit([])

    def sample(mask, count=100):
        ys, xs = np.nonzero(mask)
        pick = rng.choice(len(xs), count, replace=len(xs) < count)
        return np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3), data.cap), np.c_[xs[pick], ys[pick]].astype(np.float64)

    for episode in episodes:
        state = demos.episode(episode)[0]
        hold = held[episode][0]
        x, y, yaw = starts[episode]["block"]
        flat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        world = {"block": faces @ flat.T + [x, y, table_height + block["height"] / 2], "jar": wall + [*starts[episode]["jar"], table_height]}
        for moment in (0.0, 0.35, 0.6, 0.8, 0.95):
            index = calm_frame(state[: hold - 5], int((hold - 6) * moment), limit=STILL)
            if index is None:
                continue
            image = demos.frame(episode, index, "wrist")
            real_block, real_jaws = wrist_masks(image)
            seen = blob(real_block, 300)
            if seen is None or not whole(seen[2]):
                continue
            qpos = units.to_qpos(state[index])
            rotations, positions = arm.gripper_poses(qpos[None])
            swing = arm._matrix(jaw_body.quat) @ arm._axis_rotations(model.jnt_axis[model.joint("gripper").id], np.array([qpos[5]]))[0]
            frame = {"block": ((world["block"] - positions[0]) @ rotations[0], *sample(real_block))}
            if real_jaws is not None and real_jaws.sum() > 400:
                frame["jaws"] = (np.concatenate([surface["gripper"], surface["moving_jaw_so101_v1"] @ swing.T + jaw_body.pos]), *sample(real_jaws))
            real_jar = wrist_jar_mask(image)
            if real_jar.sum() > 1500:
                frame["jar"] = ((world["jar"] - positions[0]) @ rotations[0], *sample(real_jar))
            data.frames.append(frame)
    return data


def fit_wrist_camera(data: WristFit, report=print) -> dict:
    """The wrist camera's pose on the gripper and its lens: the ones whose drawn jaws, block and jar cover
    the real ones in frames before each grasp. The jaws are fixed to the camera, so they pin its place;
    the block, at a known place on the mat, checks that against the arm's joints; and the jar, out at the
    picture's edge, shows how much the lens bends there."""
    start_pose, start_lens = wrist_start()
    steps = np.r_[np.full(3, 0.002), np.full(3, 0.01), 5.0, 0.02, 0.02]

    def errors(x):
        found = data.residuals(x[:6], Lens(x[6], x[7], x[8]))
        return np.concatenate([values / np.sqrt(max(len(values), 1)) for values in found.values()])

    def solve(x, free):
        def moved(z):
            trial = x.copy()
            trial[free] += z * steps[free]
            return errors(trial)

        low, high = (np.r_[np.full(6, -np.inf), 180.0, -0.3, -0.5] - x) / steps, (np.r_[np.full(6, np.inf), 600.0, 0.6, 0.5] - x) / steps
        result = least_squares(moved, np.zeros(len(free)), bounds=(low[free], high[free]), diff_step=0.5, max_nfev=60)
        x = x.copy()
        x[free] += result.x * steps[free]
        return x, float(2 * result.cost)

    best = None
    for focal in (260.0, 330.0, 420.0):
        for k1 in (0.0, 1 / 3):  # a fisheye, and a lens that keeps straight lines straight
            x, cost = solve(np.r_[start_pose, focal, k1, 0.0], np.arange(8))
            if best is None or cost < best[1]:
                best = (x, cost)
    x, cost = solve(best[0], np.arange(9))
    found = data.residuals(x[:6], Lens(x[6], x[7], x[8]))
    return {
        "pose": [float(v) for v in x[:6]],
        "lens": Lens(x[6], x[7], x[8]).to_scene(),
        "errors": {f"{name}_px": round(float((found[f"{name}_stray"].mean() + found[f"{name}_missed"].mean()) / 2), 2) for name in ("jaws", "block", "jar") if f"{name}_stray" in found},
    }


def blocks_by_wrist(env, demos: Demos, held: dict, layouts: list[dict], reach: float = 0.012, report=print) -> tuple[float, float]:
    """Move each demo's block to where the wrist camera saw it, and return the median move in millimetres
    and the median distance left between the real and drawn block in the wrist camera's frames, in pixels.

    The top camera and the arm's joints place a block a few millimetres apart, and it's the arm that has
    to grasp it. In frames where the arm is still, the block is moved across the mat until the drawn
    block's centre in the wrist camera's picture is where the real one's is. Each layout's `by_wrist` then
    holds that place, and its `seen` the place the top camera gave. No block moves further than `reach`.
    """
    moves, left = [], []
    for layout in layouts:
        episode = layout["episode"]
        state = demos.episode(episode)[0]
        hold = held[episode][0]
        layout["seen"] = list(layout.get("seen", layout["block"]))
        frames = {calm_frame(state[: hold - 5], int((hold - 6) * moment), limit=STILL) for moment in (0.0, 0.3, 0.5, 0.7, 0.85, 0.95)} - {None}
        sightings = []
        for frame in sorted(frames):
            found = red_blob(demos.frame(episode, frame, "wrist"), 300)
            if found is not None and whole(found[2], 6):
                sightings.append((frame, found[0]))
        if not sightings:
            continue
        place = np.array(layout["seen"][:2], dtype=np.float64)

        def centres(xy):
            """Where the drawn block's centre is in each sighting's picture, None where it's cut off."""
            drawn = []
            for frame, _ in sightings:
                env.set_joints(state[frame])
                env.set_block(xy, layout["seen"][2])
                mujoco.mj_forward(env._model, env.data)
                shapes = env.segment("wrist", GROUP_BLOCK)
                rows, columns = np.nonzero(shapes >= 0)
                box = (columns.min(), rows.min(), np.ptp(columns), np.ptp(rows)) if len(rows) > 300 else None
                drawn.append(np.array([columns.mean(), rows.mean()]) if box is not None and whole(box, 6) else None)
            return drawn

        for _ in range(3):
            here = centres(place)
            aside, ahead = centres(place + [0.002, 0.0]), centres(place + [0.0, 0.002])
            rows_a, rows_b = [], []
            for (_, real), now, dx, dy in zip(sightings, here, aside, ahead):
                if now is None or dx is None or dy is None:
                    continue
                rows_a.append(np.column_stack([(dx - now) / 0.002, (dy - now) / 0.002]))  # pixels per metre the block moves
                rows_b.append(real - now)
            if not rows_a:
                break
            step, *_ = np.linalg.lstsq(np.concatenate(rows_a), np.concatenate(rows_b), rcond=None)
            place = place + step
            away = place - layout["seen"][:2]
            if np.linalg.norm(away) > reach:
                place = np.array(layout["seen"][:2]) + away * reach / np.linalg.norm(away)
        left += [float(np.linalg.norm(real - now)) for (_, real), now in zip(sightings, centres(place)) if now is not None]
        moves.append(float(np.linalg.norm(place - layout["seen"][:2])) * 1000)
        layout["by_wrist"] = [round(float(place[0]), 4), round(float(place[1]), 4), layout["seen"][2]]
    return float(np.median(moves)), float(np.median(left))


def replays_into_jar(env, demos: Demos, layouts: list[dict], place: str) -> int:
    """How many demos' real commands, replayed, leave the block in the jar when each block starts at its
    layout's `place`: `seen` by the top camera, or `by_wrist`."""
    count = 0
    for layout in layouts:
        state, action = demos.episode(layout["episode"])
        env.reset(seed=0, options={"joints": state[0], "block": layout.get(place, layout["seen"]), "jar": layout["jar"]})
        for command in action:
            env.apply(command)
        for _ in range(45):  # let it settle
            env.apply(action[-1])
        count += env.block_in_jar()
    return count


def jaw_yaw(units: arm.Units, state: np.ndarray) -> float:
    """The direction the jaws close along, as a turn about the vertical, for one row of follower
    readings. A block they hold lies square to it."""
    qpos = units.to_qpos(state)
    rotations, _ = arm.gripper_poses(qpos[None])
    fixed, moving = arm.jaw_tips()
    model = arm.arm_model()
    body = model.body("moving_jaw_so101_v1")
    swing = arm._matrix(body.quat) @ arm._axis_rotations(model.jnt_axis[model.joint("gripper").id], np.array([qpos[5]]))[0]
    across = rotations[0] @ (body.pos + swing @ moving - fixed)
    return float(np.arctan2(across[1], across[0]) % (np.pi / 2))


def block_yaw(env, image: np.ndarray, xy: np.ndarray, state: np.ndarray, near_yaw: float, within: float = np.radians(5)) -> tuple[list[float], float]:
    """Refine a block's place and find its turn on the mat from a top camera frame: the place and turn
    whose drawn block covers the real one best.

    The turn stays `within` of `near_yaw`, the turn the jaws later held it at: the block is small in the
    top camera's picture, and its outline alone leaves the turn 10 to 20 degrees unsure, while people lay
    a block square to the way they'll grasp it.
    """
    real = red_mask(image)
    env.set_joints(state)

    def cover(x, y, yaw):
        env.set_block([x, y], yaw)
        mujoco.mj_forward(env._model, env.data)
        return _overlap(env.segment("top", GROUP_BLOCK) >= 0, real)

    best = max(((cover(xy[0], xy[1], yaw), xy[0], xy[1], yaw) for yaw in near_yaw + np.linspace(-within, within, 7)), key=lambda found: found[0])
    for step, turn in ((0.003, np.radians(3)), (0.0015, np.radians(1.5)), (0.0007, np.radians(0.7))):
        improved = True
        while improved:
            improved = False
            for dx, dy, dyaw in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, turn), (0, 0, -turn)):
                trial = (best[1] + dx, best[2] + dy, float(np.clip(best[3] + dyaw, near_yaw - within, near_yaw + within)))
                found = cover(*trial)
                if found > best[0] + 1e-4:
                    best, improved = (found, *trial), True
    return [round(float(best[1]), 4), round(float(best[2]), 4), round(float(best[3] % (np.pi / 2)), 4)], float(best[0])


# When the cameras' pictures were taken.


def camera_timing(demos: Demos, held: dict, episodes: list[int], scene: dict) -> dict[str, dict]:
    """For each camera, how many control ticks its picture is behind the joints' readings saved with it,
    and how many new pictures a second it gives.

    The delay is the one that best lines the block up with the joints while the arm moves fast: the held
    block as the top camera sees it carried, and the block on the mat as the wrist camera sees it
    approached. A camera slower than the control loop repeats pictures, which shows as frames that
    barely differ from the one before while the arm is moving.
    """
    units = arm.Units.from_scene(scene)
    layouts = {layout["episode"]: layout for layout in scene["starts"]}
    lenses = {name: Lens.from_scene(scene["cameras"][name]["lens"]) for name in ("top", "wrist")}
    block_height = scene["table"]["height"] + scene["block"]["height"] * 0.6
    seen = {"top": [], "wrist": []}  # episode, frame and pixel of the block, while the arm moves fast
    change = {"top": [], "wrist": []}  # how much each frame differs from the one before
    for episode in episodes:
        state = demos.episode(episode)[0]
        hold, let_go = held[episode]
        speed = np.abs(np.gradient(state[:, :5], axis=0)).max(axis=1) * FPS
        for camera, frames, least_area in (("top", range(hold + 8, let_go - 2), 60), ("wrist", range(20, hold - 10), 150)):
            previous = None
            for frame in frames:
                if speed[frame] < 25:
                    previous = None
                    continue
                image = demos.frame(episode, frame, camera)
                if previous is not None:
                    change[camera].append(float(np.abs(image.astype(np.int16) - previous.astype(np.int16)).mean()))
                previous = image
                found = red_blob(image, least_area)
                if found is not None and whole(found[2]):
                    seen[camera].append((episode, frame, *found[0]))
    timing = {}
    for camera in ("top", "wrist"):
        sightings = np.array(seen[camera])
        misses = {}
        for delay in range(5):
            states = np.array([demos.episode(int(e))[0][max(int(f) - delay, 0)] for e, f in sightings[:, :2]])
            if camera == "top":
                points = to_camera(held_block(units, states), scene["cameras"]["top"]["pose"])
            else:
                rotations, positions = arm.gripper_poses(units.to_qpos(states))
                blocks = np.array([[*layouts[int(e)]["block"][:2], block_height] for e in sightings[:, 0]])
                points = to_camera(np.einsum("nji,nj->ni", rotations, blocks - positions), scene["cameras"]["wrist"]["pose"])
            misses[delay] = float(np.median(np.linalg.norm(lenses[camera].project(points) - sightings[:, 2:4], axis=1)))
        changes = np.array(change[camera])
        repeated = float((changes < 0.35 * np.median(changes)).mean())
        timing[camera] = {"delay_ticks": min(misses, key=misses.get), "fps": round(FPS * (1 - repeated), 1), "miss_px": {str(d): round(m, 1) for d, m in misses.items()}}
    return timing


# The arm's motors.


def starting_motors() -> dict:
    """Stiff and slow, like the follower's servos: each holds within a fraction of a degree but takes
    about a tenth of a second to catch up with a moving goal. All the damping is the motor's own (`kv`)."""
    motors = {name: {"kp": 60.0, "kv": 6.0, "force": 2.94, "friction": 0.05, "armature": 0.028, "damping": 0.0} for name in arm.ARM_JOINTS}
    # LeRobot limits the gripper to half its torque.
    motors["gripper"] = {"kp": 30.0, "kv": 3.0, "force": 1.47, "friction": 0.05, "armature": 0.028, "damping": 0.0}
    return motors


def place_for_frame(env, demos: Demos, held: dict, layout: dict, episode: int, frame: int) -> None:
    """Put the arm, block and jar where they really were at a demo's frame: the block on the mat, in the
    jaws, or in the jar. `layout` is the demo's entry in the scene's `starts`."""
    state = demos.episode(episode)[0]
    env.photo_from(layout)  # as an attempt started from this demo's layout would
    env.set_joints(state[frame])
    env.set_jar(layout["jar"])
    hold, let_go = held.get(episode, (len(state), len(state)))
    if frame <= hold:
        env.set_block(layout["block"][:2], layout["block"][2])
    elif frame <= let_go:
        # Carried: fixed to the gripper as it was when the jaws closed on it, flat on the mat.
        grip_rotation, grip_position = arm.gripper_poses(env.units.to_qpos(state[[hold, frame]]))
        centre = held_block(env.units, state[hold][None])[0]
        centre[2] = env.table_height + env.scene["block"]["height"] / 2
        yaw = layout["block"][2]
        flat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        carried = grip_rotation[1] @ grip_rotation[0].T
        env.set_block_pose(grip_position[1] + carried @ (centre - grip_position[0]), carried @ flat)
    else:
        env.set_block(layout["jar"], layout["block"][2])
        env.data.qpos[env.block_qadr + 2] += env.scene["jar"]["base"]
    mujoco.mj_forward(env._model, env.data)


def pictures_for_frame(env, demos: Demos, held: dict, layout: dict, episode: int, frame: int) -> dict[str, np.ndarray]:
    """What each simulated camera would have handed over at a demo's frame: its picture of the moment
    that many ticks earlier, as the real camera's is. Leaves the scene placed for the frame itself."""
    pictures = {}
    for camera, timing in env.scene["cameras"].items():
        place_for_frame(env, demos, held, layout, episode, max(frame - timing.get("delay_ticks", 0), 0))
        pictures[camera] = env.render_camera(camera)
    place_for_frame(env, demos, held, layout, episode, frame)
    return pictures


def replay_arm(env, demos: Demos, episode: int, layout: dict, block: bool = False) -> np.ndarray:
    """Give the sim a demo's real commands and return its joints' readings, one row per frame. Without
    `block` the block is out of reach, so this is only about how the arm moves."""
    state, action = demos.episode(episode)
    env.reset(seed=0, options={"joints": state[0], "block": layout["block"] if block else [0.1, -0.35, 0.0], "jar": layout["jar"]})
    track = []
    for command in action:
        env.apply(command)
        track.append(env.joint_state())
    return np.asarray(track)


def tracking_error(env, demos: Demos, held: dict, episodes: list[int]) -> np.ndarray:
    """Per joint, how far the simulated arm is from the real one when given the real commands, in the
    follower's units. The block is out of reach, so this is only about how the arm moves."""
    layouts = {layout["episode"]: layout for layout in env.scene["starts"]}
    errors = []
    for episode in episodes:
        state = demos.episode(episode)[0]
        difference = np.abs(replay_arm(env, demos, episode, layouts[episode]) - state)
        # The gripper only counts until it closes on the block, which isn't there.
        difference[held[episode][0] - 15 :, 5] = 0
        errors.append(difference.mean(axis=0))
    return np.mean(errors, axis=0)


# What the motor fit tries for each joint: how stiffly it pulls towards its goal (N m per radian), how fast
# that closes the gap (per second: its `kp / kv`), and its dry friction (N m).
MOTOR_STIFFNESS = (6.0, 12.0, 25.0, 50.0, 100.0)
MOTOR_RATES = (6.0, 10.0, 16.0, 30.0)
MOTOR_FRICTION = (0.05, 0.2, 0.4, 0.6, 0.85)
MOTOR_ARMATURE = (0.01, 0.028, 0.08)


def fit_motors(scene: dict, scene_dir: Path, demos: Demos, held: dict, episodes: list[int], limit: float, report=print) -> dict:
    """Fit each joint's motor to the real follower: how stiffly it pulls, how fast it follows, its dry
    friction and inertia, and the delay before a command acts. Each joint is judged by its own error,
    since its motor moves little else, on the real commands of demos replayed with no block.

    Stiffness, speed and friction are searched together, as a grid. The real servos' gears hold a pose
    by friction, so the shoulder lowers at once but won't lift until its goal is 5 degrees away: a soft
    motor with as much dry friction as the arm's weight. No search that changes one of the three at a
    time gets there from a motor without the friction.
    """
    scene = json.loads(json.dumps(scene))
    save_json(scene_dir / "scene.json", scene)
    env = BlockContainerEnv(scene_dir=str(scene_dir), max_relative_target=limit)
    env.reset(seed=0)
    motors, delay = scene["arm"]["motors"], scene["arm"]["delay_ticks"]

    def error() -> np.ndarray:
        env.set_motors(motors, delay)
        return tracking_error(env, demos, held, episodes)

    def best_of(index: int, choices: list[dict]) -> float:
        """Keep whichever of these settings for a joint's motor gives that joint the least error."""
        motor = motors[arm.JOINTS[index]]
        found = []
        for choice in choices:
            motor.update(choice)
            found.append(error()[index])
        motor.update(choices[int(np.argmin(found))])
        return float(min(found))

    for round_index in range(2):
        errors = []
        for index, name in enumerate(arm.JOINTS):
            motor = motors[name]
            if round_index == 0:
                grid = [{"kp": kp, "kv": round(kp / rate, 3), "friction": friction} for kp in MOTOR_STIFFNESS for rate in MOTOR_RATES for friction in MOTOR_FRICTION]
            else:
                grid = [
                    {"kp": round(motor["kp"] * a, 3), "kv": round(motor["kv"] * b, 3), "friction": round(motor["friction"] * c, 3)}
                    for a in (0.7, 1.0, 1.4) for b in (0.7, 1.0, 1.4) for c in (0.75, 1.0, 1.3)
                ]  # fmt: skip
            if name == "gripper":  # it has to keep squeezing a block it can't close on
                grid = [choice for choice in grid if choice["kp"] >= 20]
            best_of(index, grid)
            errors.append(best_of(index, [{"armature": armature} for armature in MOTOR_ARMATURE]))
        if round_index == 0:
            trials = {}
            for delay in (0, 1, 2):
                trials[delay] = error()[:5].mean()
            delay = min(trials, key=trials.get)
        report(f"  round {round_index + 1}: joint error " + ", ".join(f"{e:.2f}" for e in errors))
    env.close()
    scene["arm"]["motors"], scene["arm"]["delay_ticks"] = motors, delay
    return scene
