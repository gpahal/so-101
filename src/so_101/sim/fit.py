"""Work out a task's scene from its demos: table, objects, background and camera poses.

Nothing here is measured by hand. Joint angles give where the arm grasped and released, the fixed
front camera gives a clean background photo once the moving arm is averaged out, and each camera's
pose comes from lining the simulated arm up with the real one across many frames.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from scipy.optimize import minimize
from torchcodec.decoders import VideoDecoder

ARM_REPO = "lerobot/so101-tabletop-mujoco"
ARM_REVISION = "43a427cc783cd2c09445cf341bea6bb5bd9c22d1"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STILL_BODIES = ("base", "shoulder")
# The real gripper's calibrated range is 129.4 degrees, so 1% is more than the model's 1.1 degrees.
GRIPPER_DEG_PER_PERCENT = 1.294


def arm_xml() -> Path:
    return Path(snapshot_download(ARM_REPO, revision=ARM_REVISION)) / "robotstudio_so101" / "so101.xml"


def joints_to_qpos(state: np.ndarray, gripper_closed: float) -> np.ndarray:
    qpos = np.radians(np.asarray(state, dtype=np.float64))
    qpos[5] = gripper_closed + np.radians(state[5] * GRIPPER_DEG_PER_PERCENT)
    return qpos


@dataclass
class Demos:
    """A LeRobot dataset on disk, with what each episode's joints and front video show."""

    root: Path

    @cached_property
    def frames(self) -> pd.DataFrame:
        files = sorted(self.root.glob("data/*/*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @cached_property
    def episodes(self) -> pd.DataFrame:
        files = sorted(self.root.glob("meta/episodes/*/*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @cached_property
    def _decoders(self) -> dict:
        return {}

    def image(self, episode: int, seconds: float, camera: str = "front") -> np.ndarray:
        row = self.episodes.iloc[episode]
        key = f"videos/observation.images.{camera}"
        path = (
            self.root / "videos" / f"observation.images.{camera}"
            / f"chunk-{int(row[key + '/chunk_index']):03d}" / f"file-{int(row[key + '/file_index']):03d}.mp4"
        )  # fmt: skip
        decoder = self._decoders.setdefault(str(path), VideoDecoder(str(path)))
        at = float(row[key + "/from_timestamp"]) + seconds
        return decoder.get_frames_played_at([at]).data[0].permute(1, 2, 0).numpy()

    def episode(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        rows = self.frames[self.frames["episode_index"] == episode]
        return np.stack(rows["observation.state"]), np.stack(rows["action"])

    def grasp_and_release(self, episode: int) -> tuple[int, int | None]:
        """Frame where the gripper settles on the object, and where it lets go after carrying it.

        A grasp shows as the arm starting to lift with a closed gripper: people close in stages and
        the widest opening is often the release, so the gripper alone doesn't give it away.
        """
        state, action = self.episode(episode)
        opened = int(np.argmax(np.maximum.accumulate(action[:, 5]) > 20))
        carry = None
        for frame in range(opened + 1, len(state)):
            widest = state[opened:frame, 5].max()
            highest_reach = state[opened:frame, 1].max()
            if state[frame, 5] < widest * 0.85 and state[frame, 1] < highest_reach - 8:
                carry = frame
                break
        if carry is None:
            carry = int(np.argmin(state[opened:, 5])) + opened
        widest = opened + int(np.argmax(state[opened:carry, 5]))  # it's still opening before this
        closed = state[widest:carry, 5].min()
        hold = widest + int(np.argmax(state[widest:carry, 5] <= closed + 1))
        let_go = np.flatnonzero(state[carry:, 5] > closed + 1.5)
        return hold, carry + int(let_go[0]) if len(let_go) else None


def background(demos: Demos, episodes: list[int], camera: str = "front") -> np.ndarray:
    """The scene without the arm or object: the median across moments the arm is away from home."""
    stack = []
    for episode in episodes:
        length = len(demos.episode(episode)[0]) / 30
        stack += [demos.image(episode, float(t), camera) for t in np.linspace(0.3, 0.75, 5) * length]
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def distortion_map(width: int, height: int, fy: float, k1: float, render_fy: float) -> tuple[np.ndarray, np.ndarray]:
    """For each output pixel of a lens with radial distortion `k1`, where to read a pinhole render."""
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x, y = (u - width / 2) / fy, (v - height / 2) / fy
    scale = 1 + k1 * (x * x + y * y)
    return (x * scale * render_fy + width / 2).astype(np.float32), (y * scale * render_fy + height / 2).astype(np.float32)


class ArmMasks:
    """Renders which pixels the arm covers from a candidate camera, for lining up with real frames."""

    def __init__(self, width: int = 320, height: int = 240, gripper_closed: float = 0.0):
        spec = mujoco.MjSpec.from_file(str(arm_xml()))
        spec.worldbody.add_camera(name="fit", pos=[0.3, 0, 0.6], fovy=60)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.renderer.enable_segmentation_rendering()
        self.camera = self.model.camera("fit").id
        self.qpos = [self.model.joint(name).qposadr[0] for name in JOINTS]
        self.gripper_closed = gripper_closed
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[2] = 1  # the arm's visual meshes
        # The base and shoulder barely move, so they're part of the background photo, not the real mask.
        still = {self.model.body(name).id for name in STILL_BODIES}
        self.still_geoms = np.array([self.model.geom_bodyid[g] in still for g in range(self.model.ngeom)])

    def render(self, state: np.ndarray, pose: np.ndarray, *, fixed_to_body: str | None = None) -> np.ndarray:
        """`pose` is camera x, y, z, rotation vector, vertical field of view, and radial distortion."""
        self.data.qpos[self.qpos] = joints_to_qpos(state, self.gripper_closed)
        mujoco.mj_forward(self.model, self.data)
        rotation = np.empty(4)
        mujoco.mju_axisAngle2Quat(rotation, pose[3:6] / max(np.linalg.norm(pose[3:6]), 1e-9), np.linalg.norm(pose[3:6]))
        self.model.cam_pos[self.camera] = pose[:3]
        self.model.cam_quat[self.camera] = rotation
        fovy, k1 = pose[6], pose[7]
        fy = 0.5 * self.height / np.tan(np.radians(fovy) / 2)
        wide = max(fovy, min(fovy * (1 + max(k1, 0) * 0.8), 150.0))  # room for the lens's outer pixels
        render_fy = 0.5 * self.height / np.tan(np.radians(wide) / 2)
        self.model.cam_fovy[self.camera] = wide
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        ids = self.renderer.render()[:, :, 0]
        seen = np.zeros(ids.shape, np.uint8)
        visible = ids >= 0
        seen[visible] = 1
        seen[visible & self.still_geoms[ids.clip(0)]] = 2
        map_x, map_y = distortion_map(self.width, self.height, fy, k1, render_fy)
        return cv2.remap(seen, map_x, map_y, cv2.INTER_NEAREST)


def arm_mask(frame: np.ndarray, plate: np.ndarray, width: int, height: int) -> np.ndarray:
    """The pixels that differ from the empty scene, without the blue object."""
    a = frame.astype(np.int16)
    different = (np.abs(a - plate.astype(np.int16)).max(axis=2) > 45).astype(np.uint8)
    blue = (a[..., 2] > a[..., 0] + 12) & (a[..., 2] >= a[..., 1] - 5) & (a[..., 2] > 30)
    different[blue] = 0
    different = cv2.morphologyEx(different, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.resize(different, (width, height), interpolation=cv2.INTER_NEAREST)


def overlap_loss(masks: ArmMasks, pose: np.ndarray, samples: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not 45 <= pose[6] <= 125 or not 0 <= pose[7] <= 0.8:
        return 1.0
    total = 0.0
    for state, real in samples:
        rendered = masks.render(state, pose)
        sim, counted = rendered == 1, rendered != 2  # the still parts count for neither side
        union = np.logical_or(sim, real)[counted].sum()
        total += 1 - np.logical_and(sim, real)[counted].sum() / max(union, 1)
    return total / len(samples)


def look_at_pose(position, target, roll: float, fovy: float, k1: float) -> np.ndarray:
    """A camera pose from where it is and what it points at. MuJoCo cameras look down their -z."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    back = position - target
    back /= np.linalg.norm(back)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    c, s = np.cos(roll), np.sin(roll)
    rotation = np.column_stack([c * right + s * up, -s * right + c * up, back])
    vector, _ = cv2.Rodrigues(rotation)
    return np.r_[position, vector.ravel(), fovy, k1]


def search_camera(masks: ArmMasks, samples: list, *, position, target, roll, tries: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Try random poses around a rough guess, like the camera's place in a photo of the setup."""
    rng = np.random.default_rng(seed)
    best, loss = None, 2.0
    for _ in range(tries):
        pose = look_at_pose(
            np.asarray(position) + rng.uniform(-0.12, 0.12, 3),
            np.asarray(target) + rng.uniform(-0.12, 0.12, 3),
            roll + rng.uniform(-0.6, 0.6),
            rng.uniform(60, 115),
            rng.uniform(0, 0.5),
        )
        trial = overlap_loss(masks, pose, samples)
        if trial < loss:
            best, loss = pose, trial
    return best, loss


def fit_camera(masks: ArmMasks, start: np.ndarray, samples: list, *, steps: int = 3) -> tuple[np.ndarray, float]:
    """Refine a camera pose until the simulated arm covers the same pixels as the real one."""
    best, loss = start.copy(), overlap_loss(masks, start, samples)
    span = np.array([0.03, 0.03, 0.03, 0.05, 0.05, 0.05, 4.0, 0.1])
    for step in range(steps):
        scale = span * 0.5**step
        result = minimize(
            lambda z: overlap_loss(masks, best + z * scale, samples),
            np.zeros(8), method="Powell", options={"maxiter": 6, "xtol": 0.02, "ftol": 1e-3},
        )  # fmt: skip
        if result.fun < loss:
            best, loss = best + result.x * scale, float(result.fun)
    return best, loss


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2))


def blue_blob(image: np.ndarray, min_area: int = 200) -> tuple[np.ndarray, int, np.ndarray] | None:
    """Centre, area and bounding box of the largest blue patch: the object."""
    a = image.astype(np.int16)
    mask = ((a[..., 2] > a[..., 0] + 12) & (a[..., 2] >= a[..., 1] - 5) & (a[..., 2] > 30)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, 4]))
    if stats[largest, 4] < min_area:
        return None
    return centres[largest], int(stats[largest, 4]), stats[largest, :4]


class Kinematics:
    """Where the arm's parts are for a set of joint values, without any physics."""

    def __init__(self, gripper_closed: float):
        self.model = mujoco.MjSpec.from_file(str(arm_xml())).compile()
        self.data = mujoco.MjData(self.model)
        self.qpos = [self.model.joint(name).qposadr[0] for name in JOINTS]
        self.gripper_closed = gripper_closed

    def pose(self, state: np.ndarray, body: str = "gripper") -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qpos] = joints_to_qpos(state, self.gripper_closed)
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.body(body).xpos.copy(), self.data.body(body).xmat.reshape(3, 3).copy()

    def held_object(self, state: np.ndarray, offset: float) -> np.ndarray:
        """Where a held object's centre is: past the jaw tips' midpoint, away from the arm's base."""
        point = self.between_jaws(state)
        outward = point[:2] / np.linalg.norm(point[:2])
        return np.r_[point[:2] + offset * outward, point[2]]

    def between_jaws(self, state: np.ndarray) -> np.ndarray:
        """The point midway between the jaw tips: where a held object's centre is."""
        self.data.qpos[self.qpos] = joints_to_qpos(state, self.gripper_closed)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        tips = [self.data.geom(f"{jaw}_jaw_sph_tip{i}").xpos for jaw in ("fixed", "moving") for i in (1, 2, 3)]
        return np.mean(tips, axis=0)


def fit_wrist_lens(demos: Demos, kinematics: Kinematics, episodes: list[int], grasp_points: dict[int, np.ndarray]) -> dict:
    """Solve the wrist camera's lens and its pose on the gripper.

    Before each grasp the object sits still at the point the gripper later closes on, so every frame that
    shows it pairs a known 3D point in the gripper's frame with the pixel it appears at. Starts from the
    arm model's camera mount, because a free solve finds nonsense from such noisy points.
    """
    world, pixels = [], []
    for episode in episodes:
        hold = demos.grasp_and_release(episode)[0]
        state = demos.episode(episode)[0]
        for frame in range(0, max(hold - 12, 1), 3):
            blob = blue_blob(demos.image(episode, frame / 30, "wrist"))
            if blob is None:
                continue
            centre, _, (x, y, w, h) = blob
            if x <= 2 or y <= 2 or x + w >= 638 or y + h >= 478:  # cut off by the image edge
                continue
            position, rotation = kinematics.pose(state[frame])
            world.append(rotation.T @ (grasp_points[episode] - position))
            pixels.append(centre)
    world, pixels = np.asarray(world, np.float64), np.asarray(pixels, np.float64)

    mount = kinematics.model.camera("wrist_cam")
    start = np.zeros(9)
    mujoco.mju_quat2Mat(start, mount.quat)
    start_rotation = (start.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])).T
    start_vector, _ = cv2.Rodrigues(start_rotation)
    start_translation = -start_rotation @ np.asarray(mount.pos)

    best = None
    for fovy in (50, 60, 70, 80, 90, 100, 110):
        focal = 240 / np.tan(np.radians(fovy) / 2)
        matrix = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1.0]])
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            world, pixels, matrix, None, rvec=start_vector.copy(), tvec=start_translation.copy(),
            useExtrinsicGuess=True, reprojectionError=25, iterationsCount=400, flags=cv2.SOLVEPNP_ITERATIVE,
        )  # fmt: skip
        count = 0 if inliers is None else len(inliers)
        if best is None or count > best[0]:
            best = (count, focal, rvec, tvec)
    _, focal, rvec, tvec = best
    rotation, _ = cv2.Rodrigues(rvec)
    errors = np.linalg.norm(cv2.projectPoints(world, rvec, tvec, np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1.0]]), None)[0].reshape(-1, 2) - pixels, axis=1)
    # OpenCV's camera looks down +z with y down; MuJoCo's looks down -z with y up.
    vector, _ = cv2.Rodrigues(rotation.T @ np.diag([1.0, -1.0, -1.0]))
    return {
        "median_error_px": float(np.median(errors)),
        "points": len(world),
        "matrix": [[focal, 0, 320.0], [0, focal, 240.0], [0, 0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "position": (-rotation.T @ tvec.ravel()).tolist(),
        "rotation": vector.ravel().tolist(),
    }


def table_points(scene: dict, width: int) -> np.ndarray:
    """A texture's texels as points on the table, in the arm's frame."""
    from so_101.sim.env import plane_frame

    origin, frame = plane_frame(scene)
    (cx, cy), (hx, hy) = scene["table"]["centre"], scene["table"]["half_size"]
    height = int(width * hy / hx)
    us, vs = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
    return origin + np.stack([cx + (us * 2 - 1) * hx, cy + (vs * 2 - 1) * hy, np.zeros_like(us)], -1) @ frame.T


def read_through_camera(image: np.ndarray, pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """What a camera at `pose` sees at each 3D point, for a pinhole lens."""
    rotation, _ = cv2.Rodrigues(np.asarray(pose[3:6]))
    local = (points - np.asarray(pose[:3])) @ rotation  # MuJoCo axes: x right, y up, looking down -z
    focal = 240 / np.tan(np.radians(pose[6]) / 2)
    depth = np.maximum(-local[..., 2], 1e-6)
    return cv2.remap(
        image, (local[..., 0] / depth * focal + 320).astype(np.float32), (-local[..., 1] / depth * focal + 240).astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )  # fmt: skip


def is_wood(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] < 22) & (hsv[..., 1] > 120) & (hsv[..., 2] > 110)


def split_sessions(demos: Demos, count: int) -> list[list[int]]:
    """Group episodes recorded with the camera in the same place: the empty corners of the view don't shift."""
    groups, previous = [[0]], None
    for episode in range(count):
        patch = np.float32(cv2.cvtColor(demos.image(episode, 0.03), cv2.COLOR_RGB2GRAY)[300:480, 0:220])
        if previous is not None:
            shift = np.hypot(*cv2.phaseCorrelate(previous, patch)[0])
            if shift > 2.5:
                groups.append([episode])
            else:
                groups[-1].append(episode)
        previous = patch
    return groups


def wrist_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In a real wrist frame: the object, and the pale jaws rising from the bottom edge."""
    a = image.astype(np.int16)
    blue = (a[..., 2] > a[..., 0] + 12) & (a[..., 2] >= a[..., 1] - 5) & (a[..., 2] > 30)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pale = ((hsv[..., 1] < 50) & (hsv[..., 2] > 110)).astype(np.uint8)
    pale[: image.shape[0] * 2 // 5] = 0
    pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pale)
    jaws = np.zeros(pale.shape, bool)
    for label in range(1, count):
        if stats[label, 1] + stats[label, 3] >= image.shape[0] - 4 and stats[label, 4] > 400:  # touches the bottom
            jaws |= labels == label
    return blue, jaws


def wrist_lens_from(pose: np.ndarray) -> dict:
    focal = 240 / np.tan(np.radians(pose[6]) / 2)
    return {
        "matrix": [[focal, 0, 320.0], [0, focal, 240.0], [0, 0, 1.0]],
        "distortion": [float(pose[7]), 0.0, 0.0, 0.0, 0.0],
        "position": pose[:3].tolist(),
        "rotation": pose[3:6].tolist(),
    }


def wrist_outline_loss(env, pose: np.ndarray, samples: list) -> float:
    """How badly the simulated jaws and object cover the real ones, from a candidate wrist camera."""
    if not 45 <= pose[6] <= 125 or not -0.6 <= pose[7] <= 0.1:
        return 2.0
    from so_101.sim.env import rotation_quat

    lens = wrist_lens_from(pose)
    camera = env._model.camera("wrist_cam").id
    env._model.cam_pos[camera] = lens["position"]
    env._model.cam_quat[camera] = rotation_quat(lens["rotation"])
    lens_map = env._lens_map(lens)
    env._show_object("wrist")
    env.renderer.enable_segmentation_rendering()
    total = 0.0
    for qpos, object_xy, real_object, real_jaws in samples:
        env.data.qpos[env.qadr] = qpos
        env.set_object(object_xy)
        mujoco.mj_forward(env._model, env.data)
        env.renderer.update_scene(env.data, camera="wrist_cam", scene_option=env.options["wrist"])
        ids = cv2.remap(env.renderer.render()[:, :, 0].astype(np.int32).astype(np.float32), *lens_map, cv2.INTER_NEAREST).astype(np.int32)
        drawn_object, drawn_jaws = ids == env.object_geoms["wrist"], (ids >= 0) & env.is_jaw[ids.clip(0)]
        for drawn, real, weight in ((drawn_object, real_object, 1.0), (drawn_jaws, real_jaws, 0.7)):
            union = np.logical_or(drawn, real).sum()
            total += weight * (1 - np.logical_and(drawn, real).sum() / max(union, 1))
    env.renderer.disable_segmentation_rendering()
    return total / len(samples)


def fit_wrist_outlines(env, demos: Demos, kinematics: Kinematics, episodes: list[int], grasp: dict, start: np.ndarray, tries: int = 300) -> tuple[np.ndarray, float]:
    """Fit the wrist camera to the outlines of the real jaws and object, so sizes match as well as positions."""
    samples = []
    for episode in episodes[::2]:
        state = demos.episode(episode)[0]
        hold = demos.grasp_and_release(episode)[0]
        object_xy = ((grasp[episode] - env.origin) @ env.frame)[:2]
        for moment in (0.0, 0.5, 0.8):
            index = int(hold * moment)
            real_object, real_jaws = wrist_masks(demos.image(episode, index / 30, "wrist"))
            if real_object.sum() > 300:
                samples.append((env.to_ctrl(state[index]), object_xy.copy(), real_object, real_jaws))
    rng = np.random.default_rng(0)
    best, loss = start.copy(), wrist_outline_loss(env, start, samples)
    for attempt in range(tries):
        scale = 1.0 if attempt < tries // 2 else 0.4
        trial = best + rng.normal(0, 1, 8) * np.array([0.012, 0.012, 0.012, 0.06, 0.06, 0.06, 6.0, 0.08]) * scale
        trial_loss = wrist_outline_loss(env, trial, samples)
        if trial_loss < loss:
            best, loss = trial, trial_loss
    return best, loss


ARM_JOINTS = JOINTS[:5]


def starting_motors() -> dict:
    motors = {name: {"kp": 15.0, "kv": 2.0, "force": 2.94, "friction": 0.0, "armature": 0.028} for name in ARM_JOINTS}
    motors["gripper"] = {"kp": 998.22, "kv": 2.731, "force": 2.94, "friction": 0.0, "armature": 0.028}
    return motors


def tracking_error(scene: dict, scene_dir: Path, demos: Demos, episodes: list[int], limit: float) -> np.ndarray:
    """Per joint, how far the simulated arm ends up from the real one when given the real commands."""
    from so_101.sim.env import CylinderBowlEnv

    (scene_dir / "scene.json").write_text(json.dumps(scene))
    env = CylinderBowlEnv(scene_dir=str(scene_dir), max_relative_target=limit)
    env.reset(seed=0)
    env.set_object([0.0, -0.4])  # out of the way: this is only about how the arm moves
    errors = []
    for episode in episodes:
        state, action = demos.episode(episode)
        hold = demos.grasp_and_release(episode)[0]
        mujoco.mj_resetData(env._model, env.data)
        env._pending.clear()
        env.data.qpos[env.qadr] = env.to_ctrl(state[0])
        env.set_object([0.0, -0.4])
        env.data.ctrl[env.actuators] = env.to_ctrl(state[0])
        mujoco.mj_forward(env._model, env.data)
        track = []
        for command in action:
            env.apply(command)
            track.append(env.joint_state().copy())
        # The gripper only counts until it closes on the object, which isn't there.
        difference = np.abs(np.asarray(track) - state)
        difference[hold - 15 :, 5] = 0
        errors.append(difference.mean(axis=0))
    env.close()
    return np.mean(errors, axis=0)


def fit_motors(scene: dict, scene_dir: Path, demos: Demos, episodes: list[int], limit: float, rounds: int = 3) -> tuple[dict, np.ndarray]:
    """Fit each joint's motor to the real follower: gain, damping, torque, friction, inertia, and delay."""
    scene = json.loads(json.dumps(scene))
    motors = scene["arm"]["motors"]

    def trial(update) -> float:
        update()
        return float(tracking_error(scene, scene_dir, demos, episodes, limit).mean())

    best = trial(lambda: None)
    for delay in (0, 1, 2, 3, 4):
        previous = scene["arm"]["delay_ticks"]
        scene["arm"]["delay_ticks"] = delay
        value = trial(lambda: None)
        if value < best:
            best = value
        else:
            scene["arm"]["delay_ticks"] = previous
    for round_index in range(rounds):
        for name in (*ARM_JOINTS, "gripper"):
            motor = motors[name]
            for field, factors in (("kp", (0.6, 0.8, 1.25, 1.6)), ("kv", (0.6, 0.8, 1.25, 1.6)), ("force", (0.7, 0.85, 1.15, 1.3)), ("armature", (0.5, 0.7, 1.4, 2.0)), ("friction", None)):
                choices = [motor[field] * f for f in factors] if factors else [motor[field] + d for d in (-0.05, -0.02, 0.02, 0.05)]
                for value in choices:
                    if value < 0 or (name == "gripper" and field in ("kp", "force") and value < motors["gripper"][field]):
                        continue  # the gripper has to keep squeezing
                    old = motor[field]
                    motor[field] = round(float(value), 4)
                    result = trial(lambda: None)
                    if result < best - 1e-3:
                        best = result
                    else:
                        motor[field] = old
        print(f"  round {round_index + 1}: mean joint error {best:.2f}", flush=True)
    return scene, tracking_error(scene, scene_dir, demos, episodes, limit)
