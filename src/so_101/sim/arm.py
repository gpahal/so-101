"""The simulated arm: MuJoCo Menagerie's SO-101, and how the real follower's numbers map onto it.

The real follower reports degrees from the middle of each joint's calibrated range, and its gripper as
0 to 100 of its range. The model's zero pose is the arm's drawing, so each joint has an offset between
the two, a few degrees, which `so101 sim build` fits from the cameras (`fit.fit_top_camera`).
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mujoco
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINTS = JOINTS[:5]
JOINT_NAMES = [f"{joint}.pos" for joint in JOINTS]
JAW_BODIES = ("gripper", "moving_jaw_so101_v1")
# MuJoCo Menagerie's SO-101, at a fixed revision so every computer builds the same arm.
ARM_REPO = "lerobot/so101-tabletop-mujoco"
ARM_REVISION = "43a427cc783cd2c09445cf341bea6bb5bd9c22d1"


@cache
def arm_xml() -> Path:
    """The arm model's file, downloaded from the Hub the first time."""
    from huggingface_hub import snapshot_download

    try:
        folder = snapshot_download(ARM_REPO, revision=ARM_REVISION, local_files_only=True)
    except Exception:  # not downloaded yet
        folder = snapshot_download(ARM_REPO, revision=ARM_REVISION)
    return Path(folder) / "robotstudio_so101" / "so101.xml"


@cache
def arm_model() -> mujoco.MjModel:
    """The arm alone, for kinematics. Shared, so never change it."""
    return mujoco.MjSpec.from_file(str(arm_xml())).compile()


@cache
def base_underside() -> float:
    """How far the base's underside is above the arm model's z = 0, in metres: 2.4 mm below it. The real
    base stands on the table, so that's where the table top is in the model's axes."""
    model = arm_model()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    lowest = []
    for geom in range(model.ngeom):
        if model.body(model.geom_bodyid[geom]).name == "base" and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            lowest.append(float((data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T)[:, 2].min()))
    return min(lowest)


@dataclass(frozen=True)
class Units:
    """Converts between the follower's numbers and the model's joint angles, in radians."""

    offsets: tuple[float, ...]  # degrees added to each arm joint's reading to get the model's angle
    gripper_closed: float  # the model's jaw angle, in degrees, where the real gripper reads 0
    gripper_deg_per_percent: float

    @classmethod
    def from_scene(cls, scene: dict) -> "Units":
        arm = scene["arm"]
        return cls(tuple(arm["offsets_deg"]), arm["gripper_closed_deg"], arm["gripper_deg_per_percent"])

    def to_qpos(self, values) -> np.ndarray:
        """Model joint angles for follower readings: one reading, or one per row."""
        values = np.asarray(values, dtype=np.float64)
        qpos = np.radians(values)
        qpos[..., :5] += np.radians(self.offsets)
        qpos[..., 5] = np.radians(self.gripper_closed + values[..., 5] * self.gripper_deg_per_percent)
        return qpos

    def from_qpos(self, qpos) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        values = np.degrees(qpos)
        values[..., :5] -= self.offsets
        values[..., 5] = (np.degrees(qpos[..., 5]) - self.gripper_closed) / self.gripper_deg_per_percent
        return values


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    sin, cos = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + sin * cross + (1 - cos) * (cross @ cross)


def _matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def gripper_poses(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The gripper body's rotation and position for many arm poses at once: `qpos` is one row of model
    joint angles per pose. A fit calls this thousands of times, and MuJoCo does one pose per call."""
    model = arm_model()
    qpos = np.atleast_2d(qpos)
    rotations = np.tile(np.eye(3), (len(qpos), 1, 1))
    positions = np.zeros((len(qpos), 3))
    for index, joint in enumerate(ARM_JOINTS):
        body = model.joint(joint).bodyid[0]
        positions = positions + rotations @ model.body_pos[body]
        rotations = rotations @ _matrix(model.body_quat[body]) @ _axis_rotations(model.jnt_axis[model.joint(joint).id], qpos[:, index])
    return rotations, positions


@cache
def jaw_tips() -> tuple[np.ndarray, np.ndarray]:
    """The fixed jaw's tip and the moving jaw's tip, each in its own body's axes."""
    model = arm_model()
    tips = []
    for jaw in ("fixed", "moving"):
        geoms = [model.geom(f"{jaw}_jaw_sph_tip{i}") for i in (1, 2, 3)]
        tips.append(np.mean([geom.pos for geom in geoms], axis=0))
    return tips[0], tips[1]


def between_jaws_local(jaw_angle: float) -> np.ndarray:
    """The point midway between the jaw tips, in the gripper body's axes, for a model jaw angle in radians."""
    model = arm_model()
    fixed, moving = jaw_tips()
    body = model.body("moving_jaw_so101_v1")
    swing = _matrix(body.quat) @ _axis_rotations(model.jnt_axis[model.joint("gripper").id], np.array([jaw_angle]))[0]
    return (fixed + body.pos + swing @ moving) / 2


MOVING_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


def link_poses(qpos: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every moving body's rotation and position for many poses at once, by body name."""
    model = arm_model()
    qpos = np.atleast_2d(qpos)
    rotations = np.tile(np.eye(3), (len(qpos), 1, 1))
    positions = np.zeros((len(qpos), 3))
    poses = {}
    for index, joint in enumerate(JOINTS):
        body = model.joint(joint).bodyid[0]
        positions = positions + rotations @ model.body_pos[body]
        rotations = rotations @ _matrix(model.body_quat[body]) @ _axis_rotations(model.jnt_axis[model.joint(joint).id], qpos[:, index])
        poses[model.body(body).name] = (rotations, positions)
    return poses


@cache
def printed_surface(per_body: int = 250, seed: int = 0) -> dict[str, np.ndarray]:
    """Points spread over each moving body's printed parts, in the body's axes: the white plastic, not
    the black motors. Projected through a camera they cover the arm's outline without rendering it."""
    model = arm_model()
    rng = np.random.default_rng(seed)
    points = {}
    for name in (*MOVING_BODIES, "camera_mount"):
        body = model.body(name).id
        found, areas = [], []
        for geom in range(model.ngeom):
            material = model.geom_matid[geom]
            if model.geom_bodyid[geom] != body or model.geom_group[geom] != 2 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if material >= 0 and model.material(material).name.startswith("sts3215"):
                continue
            mesh = model.geom_dataid[geom]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            faces = model.mesh_face[model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
            corners = vertices[faces] @ _matrix(model.geom_quat[geom]).T + model.geom_pos[geom]
            found.append(corners)
            areas.append(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2)
        corners, areas = np.concatenate(found), np.concatenate(areas)
        chosen = rng.choice(len(corners), per_body, p=areas / areas.sum())
        a, b = rng.random(per_body), rng.random(per_body)
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        triangle = corners[chosen]
        points[name] = triangle[:, 0] + a[:, None] * (triangle[:, 1] - triangle[:, 0]) + b[:, None] * (triangle[:, 2] - triangle[:, 0])
    # The camera mount is fixed to the gripper.
    mount = model.body("camera_mount")
    points["gripper"] = np.concatenate([points["gripper"], points.pop("camera_mount") @ _matrix(mount.quat).T + mount.pos])
    return points


def printed_points(qpos: np.ndarray) -> np.ndarray:
    """`printed_surface` in the arm's axes for many poses: one row of points per pose."""
    surface = printed_surface()
    return np.concatenate([np.einsum("nij,pj->npi", rotations, surface[name]) + positions[:, None] for name, (rotations, positions) in link_poses(qpos).items()], axis=1)


# Where a held block's centre sits above the midpoint of the jaw tips, towards the gripper, in metres, and
# how far above the mat that midpoint is when the jaws close on a block lying flat. From the wrist
# camera's frames of grasps: the tips reach just past the middle of the block's sides.
GRASP_DEPTH = 0.006
TIPS_ABOVE_MAT = 0.006


def held_block_local(jaw_angles: np.ndarray, depth: float = GRASP_DEPTH) -> np.ndarray:
    """A held block's centre in the gripper's axes: `depth` from the jaw tips' midpoint, towards the gripper."""
    local = np.array([between_jaws_local(angle) for angle in np.atleast_1d(jaw_angles)])
    return local * (1 - depth / np.linalg.norm(local, axis=1, keepdims=True))


def held_block(units: Units, states: np.ndarray, depth: float = GRASP_DEPTH) -> np.ndarray:
    """Where a block held by the jaws is, in the arm's axes, for each row of follower readings."""
    qpos = units.to_qpos(np.atleast_2d(states))
    rotations, positions = gripper_poses(qpos)
    return positions + np.einsum("nij,nj->ni", rotations, held_block_local(qpos[:, 5], depth))
