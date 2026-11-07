"""EnvHub environment: an SO-101 twin of a recorded task, described by the files `so101 sim build` writes.

State and actions use the units of LeRobot's `so101_follower` with degrees: degrees for the arm and
0-100 for the gripper, as absolute targets at 30 fps. Based on LeRobot's `so101-tabletop-mujoco`
(Apache-2.0), with MuJoCo Menagerie's arm.

The front camera shows the real empty scene with the simulated arm and object drawn on top, so most of
its pixels are real. The wrist camera moves, so it renders a table textured from that same photo.
"""

import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_NAMES = [f"{joint}.pos" for joint in JOINTS]
# Which rendering groups things are in. Still parts are in the front camera's real background already.
GROUP_MOVING, GROUP_STILL, GROUP_SCENE = 2, 1, 5  # the arm model keeps collision shapes in 3 and red markers in 4
GROUP_HIDDEN = 3
STILL_BODIES = ("base", "shoulder")
JAW_BODIES = ("gripper", "moving_jaw_so101_v1")
SCENE_DIR_ENV = "SO101_SIM_SCENE"
# MuJoCo Menagerie's SO-101, at a fixed revision so every computer builds the same arm.
ARM_REPO = "lerobot/so101-tabletop-mujoco"
ARM_REVISION = "43a427cc783cd2c09445cf341bea6bb5bd9c22d1"

LEROBOT_SIM = {
    "fps": 30,
    "control_mode": "position",
    "semantics": "so101-degrees-gripper-percent-v1",  # the same as the real arm's policy server
    "state_names": JOINT_NAMES,
    "action_names": JOINT_NAMES,
    "robot_type": "so_follower",
}


def arm_xml() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ARM_REPO, revision=ARM_REVISION)) / "robotstudio_so101" / "so101.xml"


def rotation_quat(vector) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(vector, dtype=np.float64))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return quat


def plane_frame(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    """The table's origin and its orientation in the arm's frame."""
    a, b, c = scene["table"]["plane"]  # z = a x + b y + c
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    x_axis = np.array([1.0, 0.0, a])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    return np.array([0.0, 0.0, c]), np.column_stack([x_axis, y_axis, normal])


def cap_mesh(diameter: float, height: float, rim: float, segments: int = 48) -> tuple[list[float], list[int]]:
    """A short cylinder with a rounded top edge, like the real object: its vertices and faces."""
    radius = diameter / 2
    bend = np.linspace(0, np.pi / 2, 6)
    profile = [(0.0, height)]
    profile += [(radius - rim + rim * np.sin(a), height - rim + rim * np.cos(a)) for a in bend]
    profile += [(radius, 0.0), (0.0, 0.0)]
    vertices, faces = [], []
    for r, z in profile:
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            vertices.append([r * np.cos(angle), r * np.sin(angle), z - height / 2])
    for ring in range(len(profile) - 1):
        for i in range(segments):
            a, b = ring * segments + i, ring * segments + (i + 1) % segments
            c, d = a + segments, b + segments
            faces += [[a, c, b], [b, c, d]]
    return np.asarray(vertices).ravel().tolist(), np.asarray(faces).ravel().tolist()


def build_model(scene: dict, scene_dir: Path, fovy_scale: dict[str, float]) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(arm_xml()))
    origin, frame = plane_frame(scene)
    spec.option.gravity = (-9.81 * frame[:, 2]).tolist()

    arm = scene["arm"]
    for material in spec.materials:
        material.rgba = [*(arm["motor_rgb"] if material.name.startswith("sts3215") else arm["rgb"]), 1.0]
    # Each joint's motor as fitted to how the real follower moved. See `so_101.sim.fit.fit_motors`.
    for actuator in spec.actuators:
        motor = arm["motors"][actuator.name]
        actuator.gainprm[0] = motor["kp"]
        actuator.biasprm[1] = -motor["kp"]
        actuator.biasprm[2] = -motor["kv"]
        actuator.forcerange = [-motor["force"], motor["force"]]
    for joint in spec.joints:
        if joint.name in arm["motors"]:
            joint.frictionloss = arm["motors"][joint.name]["friction"]
            joint.armature = arm["motors"][joint.name]["armature"]

    world = spec.worldbody
    table_quat = np.empty(4)
    mujoco.mju_mat2Quat(table_quat, frame.ravel())
    table = scene["table"]
    centre = origin + frame @ np.array([*table["centre"], -0.01])
    texture = spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str((scene_dir / table["texture"]).resolve()))
    # Ambient and direct light add up to 1, so the real pixels stay as they are outside shadows.
    material = spec.add_material(name="table_mat", reflectance=0.0, specular=0.0, shininess=0.0)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[*table["half_size"], 0.01], pos=centre.tolist(),
        quat=table_quat.tolist(), material="table_mat", group=GROUP_SCENE, friction=[1.0, 0.03, 0.003],
    )  # fmt: skip
    # What's around the table, for the wrist camera: the wall behind it and the floor below it.
    for name, half, local, rgb in (
        ("wall", [0.02, 2.0, 1.0], [table["centre"][0] + table["half_size"][0] + 0.02, 0.0, 0.5], scene["surroundings"]["wall_rgb"]),
        ("floor", [2.0, 2.0, 0.01], [0.0, 0.0, -0.7], scene["surroundings"]["floor_rgb"]),
    ):
        world.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=(origin + frame @ np.array(local)).tolist(),
            quat=table_quat.tolist(), rgba=[*rgb, 1.0], group=GROUP_SCENE, contype=0, conaffinity=0,
        )  # fmt: skip

    # The bowl: a cone of flat segments above a flat base.
    bowl = scene["bowl"]
    body = world.add_body(name="bowl", pos=(origin + frame @ np.array([*bowl["xy"], 0.0])).tolist(), quat=table_quat.tolist())
    thickness, height = 0.003, bowl["height"]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[bowl["base_radius"], thickness, 0], pos=[0, 0, thickness], rgba=[*bowl["rgb"], 1], group=GROUP_SCENE)
    slope = np.arctan2(bowl["rim_radius"] - bowl["base_radius"], height)
    wall_half = 0.5 * np.hypot(bowl["rim_radius"] - bowl["base_radius"], height)
    segments = 32
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        radius = 0.5 * (bowl["rim_radius"] + bowl["base_radius"])
        yaw = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        pitch = np.array([np.cos(slope / 2), 0, np.sin(slope / 2), 0])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, yaw, pitch)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[thickness, np.pi * radius / segments * 1.1, wall_half],
            pos=[radius * np.cos(angle), radius * np.sin(angle), height / 2], quat=quat.tolist(),
            rgba=[*bowl["rgb"], 1], group=GROUP_SCENE,
        )  # fmt: skip

    # The object.
    item = scene["object"]
    free = world.add_body(name="object", pos=(origin + frame @ np.array([*item["spots"][0], item["height"] / 2 + 0.001])).tolist(), quat=table_quat.tolist())
    free.add_freejoint(name="object_joint")
    # A plain cylinder to touch, hidden, and a rounded one to see.
    free.add_geom(
        name="object_contact", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[item["diameter"] / 2, item["height"] / 2, 0],
        mass=item["mass"], group=3, condim=4, friction=[item["friction"], 0.03, 0.003], solref=[0.01, 1.0],
    )  # fmt: skip
    # How the object slides on the table, apart from how the jaws grip it. `so101 sim physics` fits this.
    slide = item.get("table_friction", 1.0)
    spec.add_pair(
        name="object_on_table", geomname1="table", geomname2="object_contact", condim=4,
        friction=[slide, slide, 0.03, 0.003, 0.003], solref=[0.015, 1.0],
    )  # fmt: skip
    # One per camera, because each is drawn at the size that matches that real camera: their fitted lenses
    # disagree a little. The real object's measured size should bring these together.
    shiny = spec.add_material(name="object_mat", specular=0.5, shininess=0.5, reflectance=0.0)
    for camera in ("front", "wrist"):
        scale = item.get(f"{camera}_scale", 1.0)
        vertices, faces = cap_mesh(item["diameter"] * scale, item["height"] * scale, item["diameter"] * scale * 0.12)
        spec.add_mesh(name=f"object_mesh_{camera}", uservert=vertices, userface=faces, smoothnormal=True)
        free.add_geom(
            name=f"object_{camera}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"object_mesh_{camera}", material=shiny.name,
            contype=0, conaffinity=0, mass=0.0001, group=GROUP_HIDDEN,
        )  # fmt: skip

    front = scene["cameras"]["front"]
    world.add_camera(name="front", pos=front["pose"][:3], quat=rotation_quat(front["pose"][3:6]).tolist(), fovy=front["pose"][6])
    wrist = scene["cameras"].get("wrist", {})
    # The room's light. No shadows: MuJoCo's turn whatever they fall on black, which is further from the
    # real wrist camera's soft shadows than having none.
    shadow = scene.get("shadow")
    if shadow:
        # What the light sees, for working out shadows.
        back = -np.asarray(shadow["dir"], dtype=np.float64)
        back /= np.linalg.norm(back)
        right = np.cross([0.0, 1.0, 0.0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.column_stack([right, up, back]).ravel())
        world.add_camera(
            name="light", pos=(np.array([0.2, 0.0, 0.0]) + back * 1.0).tolist(), quat=quat.tolist(),
            orthographic=True, fovy=shadow["view_m"],
        )  # fmt: skip
    light = scene["light"]
    world.add_light(
        pos=light["pos"], dir=light["dir"], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False, diffuse=[light["direct"]] * 3,
        ambient=[light["ambient"]] * 3, specular=[0.3, 0.3, 0.3],
    )  # fmt: skip
    spec.visual.map.znear = 0.002  # the wrist camera is about a centimetre from the jaws
    spec.visual.headlight.ambient = [0.0, 0.0, 0.0]
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    model = spec.compile()
    if wrist.get("lens") is not None:
        camera = model.camera("wrist_cam").id
        model.cam_pos[camera] = wrist["lens"]["position"]
        model.cam_quat[camera] = rotation_quat(wrist["lens"]["rotation"])
    still = {model.body(name).id for name in STILL_BODIES}
    for geom in range(model.ngeom):
        if model.geom_group[geom] == 2 and model.geom_bodyid[geom] in still:
            model.geom_group[geom] = GROUP_STILL
    return model


class CylinderBowlEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_dir: str | None = None,
        spot: int | None = None,
        episode_length: int = 900,
        max_relative_target: float = 10.0,
        settle_success_steps: int = 15,
    ):
        super().__init__()
        self.scene_dir = Path(scene_dir or os.environ[SCENE_DIR_ENV])
        self.scene = json.loads((self.scene_dir / "scene.json").read_text())
        self.task = "so101_cylinder_bowl"
        self.task_description = self.scene["prompt"]
        self.fps = 30
        self.spot = spot
        self._max_episode_steps = episode_length
        self.limit = max_relative_target
        self.settle_success_steps = settle_success_steps
        self.height, self.width = 480, 640
        self.initial_joints = np.asarray(self.scene["start_pose"], dtype=np.float32)
        self._model: mujoco.MjModel | None = None
        image = spaces.Box(0, 255, (self.height, self.width, 3), np.uint8)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-360.0, 360.0, (len(JOINTS),), np.float32),
                "pixels": spaces.Dict({"front": image, "wrist": image}),
            }
        )
        self.action_space = spaces.Box(-360.0, 360.0, (len(JOINTS),), np.float32)
        self._steps = 0
        self._resting = 0
        self._pending: list[np.ndarray] = []

    def _ensure_world(self) -> None:
        """Build on first use, so each process that renders owns its GL context."""
        if self._model is not None:
            return
        self._model = build_model(self.scene, self.scene_dir, {})
        model = self._model
        self.data = mujoco.MjData(model)
        self.substeps = max(1, round(1 / (self.fps * model.opt.timestep)))
        model.opt.timestep = 1 / (self.fps * self.substeps)
        self.renderer = mujoco.Renderer(model, self.height, self.width)
        self.qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
        self.actuators = np.array([model.actuator(name).id for name in JOINTS])
        self.lo, self.hi = model.actuator_ctrlrange[self.actuators].T.copy()
        self.object_qadr = model.joint("object_joint").qposadr[0]
        self.object_body = model.body("object").id
        self.origin, self.frame = plane_frame(self.scene)
        self.background = cv2.cvtColor(cv2.imread(str(self.scene_dir / self.scene["cameras"]["front"]["background"])), cv2.COLOR_BGR2RGB)
        self.options = {"front": mujoco.MjvOption(), "wrist": mujoco.MjvOption()}
        self.options["front"].geomgroup[:] = 0
        self.options["front"].geomgroup[GROUP_MOVING] = 1
        for group in (GROUP_SCENE, GROUP_MOVING, GROUP_STILL):
            self.options["wrist"].geomgroup[group] = 1
        self.options["wrist"].geomgroup[3] = 0
        self.options["wrist"].geomgroup[0] = 0
        self.table_geom = model.geom("table").id
        self.object_geoms = {camera: model.geom(f"object_{camera}").id for camera in ("front", "wrist")}
        jaw_bodies = {model.body(name).id for name in JAW_BODIES}
        self.is_jaw = np.array([model.geom_group[g] == 2 and model.geom_bodyid[g] in jaw_bodies for g in range(model.ngeom)])
        self.printed_materials = [m for m in range(model.nmat) if not model.material(m).name.startswith(("sts3215", "table", "object"))]
        self.maps = {"front": None, "wrist": self._lens_map(self.scene["cameras"]["wrist"].get("lens"))}
        self._prepare_shadows()

    def _prepare_shadows(self) -> None:
        """Soft shadows, drawn by hand: see which parts of the table the arm hides from the light."""
        shadow = self.scene.get("shadow")
        self.shadow = shadow
        if not shadow:
            return
        self.shadow_renderer = mujoco.Renderer(self._model, 256, 256)
        self.shadow_renderer.enable_segmentation_rendering()
        self.shadow_options = mujoco.MjvOption()
        self.shadow_options.geomgroup[:] = 0
        self.shadow_options.geomgroup[GROUP_MOVING] = 1
        self.shadow_options.geomgroup[GROUP_STILL] = 1
        self.shadow_camera = self._model.camera("light").id
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32))
        lens = self.scene["cameras"]["wrist"].get("lens")
        pixels = np.stack([u, v], -1).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(pixels, np.asarray(lens["matrix"]), np.asarray(lens["distortion"])).reshape(self.height, self.width, 2)
        self.wrist_rays = np.dstack([rays[..., 0], -rays[..., 1], -np.ones(rays.shape[:2])])  # in MuJoCo's camera axes

    def _shadow_map(self) -> np.ndarray:
        self.shadow_renderer.update_scene(self.data, camera="light", scene_option=self.shadow_options)
        hidden = (self.shadow_renderer.render()[:, :, 0] >= 0).astype(np.float32)
        return cv2.GaussianBlur(hidden, (0, 0), self.shadow["soft_px"])

    def _table_shadow(self, camera: str, rays: np.ndarray, shadow_map: np.ndarray) -> np.ndarray:
        """How shadowed the table is, per pixel of a camera: 0 is full light."""
        camera_id = self._model.camera(camera).id
        position = self.data.cam_xpos[camera_id]
        rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
        world = rays @ rotation.T
        normal = self.frame[:, 2]
        along = ((self.origin - position) @ normal) / np.minimum(world @ normal, -1e-6)
        points = position + world * along[..., None]
        light_id = self.shadow_camera
        offset = (points - self.data.cam_xpos[light_id]) @ self.data.cam_xmat[light_id].reshape(3, 3)
        half = self.shadow["view_m"] / 2
        map_x = ((offset[..., 0] / half + 1) * 128).astype(np.float32)
        map_y = ((1 - offset[..., 1] / half) * 128).astype(np.float32)
        return cv2.remap(shadow_map, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    @staticmethod
    def _mj_camera(name: str) -> str:
        return "wrist_cam" if name == "wrist" else name

    def _lens_map(self, lens: dict | None):
        """Render through a wide pinhole, then bend it like the real lens: for each real pixel, where to read."""
        if lens is None:
            return None
        camera = self._model.camera("wrist_cam").id
        matrix, distortion = np.asarray(lens["matrix"]), np.asarray(lens["distortion"])
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32))
        pixels = np.stack([u, v], -1).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(pixels, matrix, distortion).reshape(self.height, self.width, 2)
        reach = float(np.abs(rays[..., 1]).max())
        reach = max(reach, float(np.abs(rays[..., 0]).max()) * self.height / self.width)
        render_fy = 0.5 * self.height / (reach * 1.02)
        self._model.cam_fovy[camera] = float(np.degrees(2 * np.arctan(reach * 1.02)))
        return (
            (rays[..., 0] * render_fy + self.width / 2).astype(np.float32),
            (rays[..., 1] * render_fy + self.height / 2).astype(np.float32),
        )

    def to_ctrl(self, values: np.ndarray) -> np.ndarray:
        ctrl = np.radians(np.asarray(values, dtype=np.float64))
        ctrl[5] = self.lo[5] + np.radians(values[5] * self.scene["arm"]["gripper_deg_per_percent"])
        return np.clip(ctrl, self.lo, self.hi)

    def joint_state(self) -> np.ndarray:
        q = self.data.qpos[self.qadr]
        state = np.degrees(q)
        state[5] = np.degrees(q[5] - self.lo[5]) / self.scene["arm"]["gripper_deg_per_percent"]
        return state.astype(np.float32)

    def object_on_table(self) -> np.ndarray:
        """The object's position relative to the table's origin, in the table's frame."""
        return (self.data.xpos[self.object_body] - self.origin) @ self.frame

    def object_in_bowl(self) -> bool:
        """Inside the cone of the bowl's wall, which narrows towards its base."""
        bowl, position = self.scene["bowl"], self.object_on_table()
        height = float(np.clip(position[2] / bowl["height"], 0.0, 1.0))
        inside = bowl["base_radius"] + height * (bowl["rim_radius"] - bowl["base_radius"])
        return bool(np.linalg.norm(position[:2] - np.asarray(bowl["xy"])) < inside and position[2] < bowl["height"])

    def _show_object(self, name: str) -> None:
        for camera, geom in self.object_geoms.items():
            self._model.geom_group[geom] = GROUP_MOVING if camera == name else GROUP_HIDDEN
        self._model.geom_rgba[self.object_geoms[name], :3] = self.scene["object"][f"{name}_rgb"]

    def render_camera(self, name: str) -> np.ndarray:
        return self._render(name)

    def segment(self, name: str) -> np.ndarray:
        """Which geom each pixel of a camera shows, -1 for none."""
        self._show_object(name)
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self._mj_camera(name), scene_option=self.options[name])
        ids = self.renderer.render()[:, :, 0].copy()
        self.renderer.disable_segmentation_rendering()
        if self.maps[name] is not None:
            ids = cv2.remap(ids.astype(np.float32), *self.maps[name], cv2.INTER_NEAREST).astype(np.int32)
        return ids

    def wrist_table_points(self, rays: np.ndarray, height: float = 0.0) -> np.ndarray:
        """Where rays from the wrist camera, in its own axes, meet a plane `height` above the table. In the
        table's frame. Needs `mj_forward` to have placed the camera."""
        camera = self._model.camera("wrist_cam").id
        position = self.data.cam_xpos[camera]
        world = rays @ self.data.cam_xmat[camera].reshape(3, 3).T
        normal = self.frame[:, 2]
        along = ((self.origin + normal * height - position) @ normal) / np.minimum(world @ normal, -1e-6)
        return (position + world * along[..., None] - self.origin) @ self.frame

    def wrist_pixel_on_table(self, pixel, height: float = 0.0) -> np.ndarray:
        """Where a real wrist camera pixel lands on the table, in the table's frame."""
        x, y = np.round(pixel).astype(int).clip(0, [self.width - 1, self.height - 1])
        return self.wrist_table_points(self.wrist_rays[y, x], height)[:2]

    def _render(self, name: str) -> np.ndarray:
        # Each real camera sees the same things in different colours, so they're matched separately.
        self._show_object(name)
        self._model.mat_rgba[self.printed_materials, :3] = self.scene["arm"][f"{name}_rgb"]
        self.renderer.update_scene(self.data, camera=self._mj_camera(name), scene_option=self.options[name])
        image = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self._mj_camera(name), scene_option=self.options[name])
        ids = self.renderer.render()[:, :, 0].copy()
        self.renderer.disable_segmentation_rendering()
        if name == "front":
            drawn = ids >= 0
            soft = cv2.GaussianBlur(drawn.astype(np.float32), (0, 0), 0.8)[..., None]  # no hard rendered edge
            background = self.background.astype(np.float32)
            return (background * (1 - soft) + self._soften(image, name) * soft).astype(np.uint8)
        if self.maps[name] is not None:
            image = cv2.remap(image, *self.maps[name], cv2.INTER_LINEAR)
            ids = cv2.remap(ids.astype(np.float32), *self.maps[name], cv2.INTER_NEAREST).astype(np.int32)
        if self.shadow:
            dark = self._table_shadow("wrist_cam", self.wrist_rays, self._shadow_map())
            dark[ids != self.table_geom] = 0
            image = (image * (1 - dark * self.shadow["wrist_strength"])[..., None]).astype(np.uint8)
        return self._soften(image, name)

    def _soften(self, image: np.ndarray, name: str) -> np.ndarray:
        """How soft the real camera's images are, measured from its frames by `so101 sim build`."""
        sigma = self.scene.get("softness", {}).get(name, 0.8)
        return cv2.GaussianBlur(image, (0, 0), sigma) if sigma > 0 else image

    def _obs(self) -> dict[str, Any]:
        return {"agent_pos": self.joint_state(), "pixels": {name: self.render_camera(name) for name in ("front", "wrist")}}

    def set_object(self, xy) -> None:
        item = self.scene["object"]
        adr = self.object_qadr
        self.data.qpos[adr : adr + 3] = self.origin + self.frame @ np.array([*xy, item["height"] / 2 + 0.001])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.frame.ravel())
        self.data.qpos[adr + 3 : adr + 7] = quat

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_world()
        super().reset(seed=seed)
        options = kwargs.get("options") or {}
        mujoco.mj_resetData(self._model, self.data)
        self.data.qpos[self.qadr] = self.to_ctrl(self.initial_joints)
        self.data.ctrl[self.actuators] = self.data.qpos[self.qadr]
        spots = np.asarray(self.scene["object"]["spots"])
        if "object_xy" in options:
            xy = np.asarray(options["object_xy"])
        elif self.spot is not None:
            xy = spots[self.spot % len(spots)]
        else:
            xy = spots[self.np_random.integers(len(spots))] + self.np_random.uniform(-0.01, 0.01, 2)
        self.set_object(xy)
        mujoco.mj_forward(self._model, self.data)
        self._steps, self._resting = 0, 0
        self._pending.clear()
        return self._obs(), {"is_success": False}

    def apply(self, action: np.ndarray) -> None:
        """Advance one control period, with the same per-step cap the real follower applies."""
        # The bus and the motors take a few ticks to act on a command.
        self._pending.append(np.asarray(action, dtype=np.float64))
        delay = self.scene["arm"]["delay_ticks"]
        action = self._pending[0] if len(self._pending) <= delay else self._pending.pop(0)
        if self.limit:
            present = self.joint_state()
            action = present + np.clip(action - present, -self.limit, self.limit)
        self.data.ctrl[self.actuators] = self.to_ctrl(action)
        mujoco.mj_step(self._model, self.data, nstep=self.substeps)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._ensure_world()
        self.apply(action)
        self._steps += 1
        self._resting = self._resting + 1 if self.object_in_bowl() else 0
        success = self._resting >= self.settle_success_steps
        truncated = self._steps >= self._max_episode_steps
        return self._obs(), float(success), success, truncated, {"is_success": success, "object_in_bowl": self._resting > 0}

    def render(self) -> np.ndarray:
        self._ensure_world()
        return self.render_camera("front")

    def close(self) -> None:
        if self._model is not None:
            self.renderer.close()
            if getattr(self, "shadow", None):
                self.shadow_renderer.close()
            self._model = None


def make_env(n_envs: int = 1, use_async_envs: bool = False, **kwargs: Any) -> dict[str, dict[int, Any]]:
    """EnvHub entry point. `kwargs` override `CylinderBowlEnv`'s, and SO101_SIM_SCENE names the scene."""
    factory = partial(CylinderBowlEnv, **kwargs)
    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return {"so101_cylinder_bowl": {0: vector_cls([factory] * n_envs)}}
