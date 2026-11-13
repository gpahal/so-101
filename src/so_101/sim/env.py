"""EnvHub environment: an SO-101 twin of the block-container task, described by the files `so101 sim build` writes.

State and actions use the units of LeRobot's `so101_follower` with degrees: degrees for the arm and
0-100 for the gripper, as absolute targets at 30 fps. Based on LeRobot's `so101-tabletop-mujoco`
(Apache-2.0), with MuJoCo Menagerie's arm.

The top camera never moves, so its picture is the real empty scene with the simulated arm and block
drawn on top: most of its pixels are real. The jar is real too, a cut-out of it from the demos moved to
where the simulated jar stands, and only drawn as a shape once it has been knocked over. The wrist
camera moves, so its whole view is rendered: the mat and table textured from the top camera's picture,
the wall, the floor, the jar, the block and the jaws.
"""

import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so_101.sim.arm import JAW_BODIES, JOINT_NAMES, JOINTS, Units, arm_xml
from so_101.sim.camera import HEIGHT, WIDTH, Lens, rotation_quat, to_camera

CAMERAS = ("top", "wrist")
# Which rendering group each thing is in, so each camera and each fit can draw just what it needs.
GROUP_SCENE, GROUP_BASE, GROUP_ARM, GROUP_HIDDEN, GROUP_BLOCK, GROUP_JAR = range(6)
SCENE_DIR_ENV = "SO101_SIM_SCENE"
# The built demos' folder. With it, the top camera's arm is drawn from the demos' real frames (`photo`).
DEMOS_DIR_ENV = "SO101_SIM_DEMOS"
TASK = "so101_block_container"
FPS = 30
# Physics steps per control tick: 0.002 s each. The block weighs a few grams and the jaws close hard, and
# at the arm model's own 0.005 s such a grip jitters.
SUBSTEPS = 16
# The pinhole pictures drawn before the lens bends them, as a multiple of the cameras' 640 x 480. Larger,
# so the bent picture's centre, which the lens magnifies, still has a rendered pixel for each real one.
RENDER_SCALE = {"top": 1.5, "wrist": 1.5}
JAR_SEGMENTS = 32
WRIST_CAMERA_PARTS = ("wrist_camera_pcb", "wrist_camera_lens")
GLOW = 0.6
ARM_GLOW = 0.1
# How hard things are to press into. MuJoCo's contacts give way in proportion to the lighter body's weight,
# and with the arm model's settings the jaws sank 2 to 3 mm into each side of the 5 gram block: it then
# stayed wedged when the real gripper lets go by opening 2 mm, and the gripper read 1 to 4 less than the
# real one while holding it. With these the jaws sink 0.4 mm and read within 1 of the real gripper. The
# jaws' settings are the ones that count against the block, because the arm model gives them priority; the
# block's and jar's count against the mat, through their high `solmix`.
HARD_JAWS = dict(solref=[0.0045, 1.0], solimp=[0.99, 0.994, 0.001, 0.5, 2.0])
HARD = dict(solref=[0.0045, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0], solmix=1000.0)

# The jar counts as knocked over, and is drawn as a shape rather than its photo, past this tilt.
JAR_UPRIGHT_COS = float(np.cos(np.radians(12.0)))

LEROBOT_SIM = {
    "fps": FPS,
    "control_mode": "position",
    "semantics": "so101-degrees-gripper-percent-v1",  # the same as the real arm's policy server
    "state_names": JOINT_NAMES,
    "action_names": JOINT_NAMES,
    "robot_type": "so_follower",
}


def _warn(message: str) -> None:
    """MuJoCo's warnings, without the one about depth precision that macOS's OpenGL draws on every depth
    picture: its default handler would also write each to MUJOCO_LOG.TXT in the working folder."""
    if "ARB_clip_control" not in message:
        print(f"MuJoCo: {message}", file=sys.stderr, flush=True)


mujoco.set_mju_user_warning(_warn)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def jar_mesh(jar: dict, bottom: float, top: float, segments: int = 48) -> tuple[list[float], list[int]]:
    """A band of the jar's wall between two heights, with an inside and an outside face, so a camera
    looking in through the open top sees the far side's inner face. Closed across the bottom at height 0."""
    outer, inner = jar["radius"], jar["radius"] - jar["wall"]
    floor = max(bottom, jar["base"])
    profile = [(outer, bottom), (outer, top), (inner, top), (inner, floor)]
    if bottom == 0.0:
        profile = [(0.0, 0.0), *profile, (0.0, floor)]
    vertices, faces = [], []
    for radius, height in profile:
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), height])
    for ring in range(len(profile) - 1):
        for i in range(segments):
            a, b = ring * segments + i, ring * segments + (i + 1) % segments
            c, d = a + segments, b + segments
            faces += [[a, b, c], [b, d, c]]
    return np.asarray(vertices).ravel().tolist(), np.asarray(faces).ravel().tolist()


def table_points(table: dict) -> np.ndarray:
    """The table texture's texels as points on the mat, in the arm's axes. Row 0 is the far left corner."""
    (x0, x1), (y0, y1) = table["x"], table["y"]
    rows, columns = round((x1 - x0) / table["texel"]), round((y1 - y0) / table["texel"])
    xs = x1 - (np.arange(rows) + 0.5) * table["texel"]
    ys = y1 - (np.arange(columns) + 0.5) * table["texel"]
    grid_y, grid_x = np.meshgrid(ys, xs)
    return np.dstack([grid_x, grid_y, np.full_like(grid_x, table["height"])])


def jar_outline(pose, lens: Lens, xy, jar: dict, table_height: float, top: float | None = None) -> np.ndarray:
    """The pixels a standing jar's wall covers from a camera, up to `top` metres (default: all of it)."""
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    circle = np.c_[jar["radius"] * np.cos(angles) + xy[0], jar["radius"] * np.sin(angles) + xy[1]]
    rings = np.concatenate([np.c_[circle, np.full(len(circle), table_height + height)] for height in (0.0, jar["height"] if top is None else top)])
    pixels = lens.project(to_camera(rings, pose))
    mask = np.zeros((lens.height, lens.width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pixels).astype(np.int32)), 1)
    return mask > 0


def build_model(scene: dict, scene_dir: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(arm_xml()))
    spec.option.timestep = 1 / (FPS * SUBSTEPS)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3  # or a held block creeps down between the jaws

    arm = scene["arm"]
    # Each joint's motor as fitted to how the real follower moved. See `so_101.sim.fit.fit_motors`.
    for actuator in spec.actuators:
        motor = arm["motors"][actuator.name]
        actuator.gainprm[0] = motor["kp"]
        actuator.biasprm[1] = -motor["kp"]
        actuator.biasprm[2] = -motor["kv"]
        actuator.forcerange = [-motor["force"], motor["force"]]
    units = Units.from_scene(scene)
    for joint in spec.joints:
        if joint.name in arm["motors"]:
            joint.frictionloss = arm["motors"][joint.name]["friction"]
            joint.armature = arm["motors"][joint.name]["armature"]
            joint.damping = arm["motors"][joint.name]["damping"]
    # The follower's calibrated range for each joint, which LeRobot keeps its goals within, replaces the
    # model's: the real arm folds a few degrees further than the drawing allows.
    limits = np.array([arm["limits"][name] for name in JOINTS], dtype=np.float64)
    low, high = units.to_qpos(limits[:, 0]), units.to_qpos(limits[:, 1])
    for index, name in enumerate(JOINTS):
        joint = next(joint for joint in spec.joints if joint.name == name)
        joint.range = [low[index], high[index]]
        next(actuator for actuator in spec.actuators if actuator.name == name).ctrlrange = [low[index], high[index]]

    world = spec.worldbody
    table, wall, floor = scene["table"], scene["wall"], scene["floor"]
    (x0, x1), (y0, y1) = table["x"], table["y"]
    texture = spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str((scene_dir / table["texture"]).resolve()))
    material = spec.add_material(name="table_mat", reflectance=0.0, specular=0.0, shininess=0.0)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    # A box's texture runs along its own x and y, and the picture's rows run away from the arm: turn the
    # box a quarter turn so its x is the arm's -y.
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(y1 - y0) / 2, (x1 - x0) / 2, 0.02],
        pos=[(x0 + x1) / 2, (y0 + y1) / 2, table["height"] - 0.02], quat=yaw_quat(-np.pi / 2).tolist(), material="table_mat",
        group=GROUP_SCENE, friction=[table["friction"], 0.005, 0.0001], condim=4,
    )  # fmt: skip
    for name, size, pos, rgb in (
        ("wall", [0.02, 3.0, 1.5], [wall["x"] + 0.02, 0.0, 0.5], wall["rgb"]),
        ("floor", [3.0, 3.0, 0.01], [0.0, 0.0, floor["height"] - 0.01], floor["rgb"]),
    ):
        spec.add_material(name=f"{name}_mat", rgba=[*rgb, 1.0])
        world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=f"{name}_mat", group=GROUP_SCENE)
    # The arm, the jaws and the scene are matt. The wall, floor and table show `GLOW` of their colour
    # whatever way they face, so a wall the light only grazes can still be as bright as the real one. The
    # arm shows little of it: the real one's tops are lit and its sides and holes are in shade.
    for item in spec.materials:
        item.specular, item.shininess, item.reflectance = 0.0, 0.0, 0.0
        item.emission = GLOW if item.name in ("table_mat", "wall_mat", "floor_mat") else ARM_GLOW

    # The block: a brick with four hollow studs. Each shape is both seen and touched.
    block = scene["block"]
    half, body_height, stud = block["width"] / 2, block["height"], block["stud"]
    brick = world.add_body(name="block", pos=[0.25, 0.0, table["height"] + body_height / 2 + 0.001])
    brick.add_freejoint(name="block_joint")
    spec.add_material(name="block_mat", specular=0.25, shininess=0.4, reflectance=0.0, emission=ARM_GLOW, rgba=[*block["rgb"], 1.0])
    spec.add_material(name="block_hole_mat", specular=0.0, shininess=0.0, reflectance=0.0, emission=ARM_GLOW, rgba=[*block["hole_rgb"], 1.0])
    solid = dict(group=GROUP_BLOCK, material="block_mat", condim=4, friction=[block["friction"], 0.005, 0.0001], **HARD)
    stud_mass = block["mass"] * 0.03
    brick.add_geom(name="block_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, body_height / 2], mass=block["mass"] - 4 * stud_mass, **solid)
    for index, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        centre = [sx * stud["pitch"] / 2, sy * stud["pitch"] / 2, body_height / 2 + stud["height"] / 2]
        brick.add_geom(name=f"block_stud{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[stud["radius"], stud["height"] / 2, 0], pos=centre, mass=stud_mass, **solid)
        brick.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[stud["radius"] * 0.62, 0.0002, 0], pos=[centre[0], centre[1], body_height / 2 + stud["height"] + 0.0002],
            material="block_hole_mat", group=GROUP_BLOCK, contype=0, conaffinity=0, mass=0,
        )  # fmt: skip

    # The jar: a ring of flat pieces to touch, hidden, and two bands of wall to see, tinted below and clear above.
    jar = scene["jar"]
    can = world.add_body(name="jar", pos=[*jar["home"], table["height"] + 0.0005])
    can.add_freejoint(name="jar_joint")
    touch = dict(group=GROUP_HIDDEN, condim=4, friction=[jar["friction"], 0.005, 0.0001], **HARD)
    piece_mass = jar["mass"] * 0.7 / JAR_SEGMENTS
    can.add_geom(name="jar_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[jar["radius"], jar["base"] / 2, 0], pos=[0, 0, jar["base"] / 2], mass=jar["mass"] * 0.3, **touch)
    middle = jar["radius"] - jar["wall"] / 2
    # The wall starts above the base, so only the base rests on the mat: 32 pieces each touching it made
    # every physics step several times slower.
    wall_half = (jar["height"] - jar["base"]) / 2
    for i in range(JAR_SEGMENTS):
        angle = 2 * np.pi * (i + 0.5) / JAR_SEGMENTS
        can.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[jar["wall"] / 2, np.pi * jar["radius"] / JAR_SEGMENTS * 1.05, wall_half],
            pos=[middle * np.cos(angle), middle * np.sin(angle), jar["base"] + wall_half], quat=yaw_quat(angle).tolist(), mass=piece_mass, **touch,
        )  # fmt: skip
    for name, bottom, top in (("tint", 0.0, jar["tint_height"]), ("clear", jar["tint_height"], jar["height"])):
        vertices, faces = jar_mesh(jar, bottom, top)
        spec.add_mesh(name=f"jar_{name}_mesh", uservert=vertices, userface=faces)
        spec.add_material(name=f"jar_{name}_mat", specular=0.3, shininess=0.6, reflectance=0.0, emission=GLOW, rgba=jar[f"{name}_rgba"])
        can.add_geom(
            name=f"jar_{name}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"jar_{name}_mesh", material=f"jar_{name}_mat",
            contype=0, conaffinity=0, mass=0, group=GROUP_JAR,
        )  # fmt: skip

    top = scene["cameras"]["top"]
    world.add_camera(name="top", pos=top["pose"][:3], quat=rotation_quat(top["pose"][3:6]).tolist())
    light = scene["light"]
    world.add_light(
        # No cast shadows: MuJoCo's leave what they fall on nearly black, and the real ones are soft and grey.
        pos=[0.2, 0.0, 1.5], dir=light["dir"], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False,
        diffuse=[light["direct"]] * 3, ambient=[light["ambient"]] * 3, specular=[0.3, 0.3, 0.3],
    )  # fmt: skip
    spec.visual.map.znear = 0.002  # the wrist camera is about a centimetre from the jaws
    spec.visual.headlight.ambient = [0.0, 0.0, 0.0]
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth = int(WIDTH * max(RENDER_SCALE.values()))
    spec.visual.global_.offheight = int(HEIGHT * max(RENDER_SCALE.values()))

    model = spec.compile()
    wrist = scene["cameras"]["wrist"]
    camera = model.camera("wrist_cam").id
    model.cam_pos[camera] = wrist["pose"][:3]
    model.cam_quat[camera] = rotation_quat(wrist["pose"][3:6])
    # The arm model keeps what's seen in group 2 and what's touched in groups 3 and 4. Here the base, which
    # never moves and so is in the top camera's real picture, gets its own group, and all that's only
    # touched goes in one.
    base = model.body("base").id
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        if name.startswith(("block", "jar", "table", "wall", "floor")) or model.geom_bodyid[geom] in (model.body("block").id, model.body("jar").id):
            continue
        if name in WRIST_CAMERA_PARTS:
            # The wrist camera sits inside its own lens, which would draw an arc across its picture.
            model.geom_group[geom] = GROUP_HIDDEN
        elif model.geom_contype[geom] or model.geom_conaffinity[geom]:
            model.geom_group[geom] = GROUP_HIDDEN
            if model.geom_priority[geom] > 0:  # the jaws, which the arm model lets decide how they touch things
                model.geom_friction[geom, 0] = arm["jaw_friction"]
                model.geom_solref[geom], model.geom_solimp[geom] = HARD_JAWS["solref"], HARD_JAWS["solimp"]
        else:
            model.geom_group[geom] = GROUP_BASE if model.geom_bodyid[geom] == base else GROUP_ARM
    return model


class BlockContainerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        scene_dir: str | None = None,
        demos_dir: str | None = None,
        start: int | None = None,
        jitter: float = 1.0,
        episode_length: int = 900,
        max_relative_target: float = 10.0,
        settle_success_steps: int = 15,
    ):
        """`start` picks one demo's block and jar positions; without it each reset draws one at random.
        `jitter` scales the small random shift added to it, 3 mm and 3 degrees at 1. `demos_dir` is the
        built demos' folder (default: SO101_SIM_DEMOS): with it the top camera's arm is drawn from their
        real frames, and without it from the arm model alone."""
        super().__init__()
        self.scene_dir = Path(scene_dir or os.environ[SCENE_DIR_ENV])
        self.scene = json.loads((self.scene_dir / "scene.json").read_text())
        self.demos_dir = demos_dir or os.environ.get(DEMOS_DIR_ENV)
        self.arm_photos = self.wrist_photos = None
        self.task = TASK
        self.task_description = self.scene["prompt"]
        self.fps = FPS
        self.start = start
        self.jitter = jitter
        self._max_episode_steps = episode_length
        self.limit = max_relative_target
        self.settle_success_steps = settle_success_steps
        self.height, self.width = HEIGHT, WIDTH
        self.units = Units.from_scene(self.scene)
        self.initial_joints = np.asarray(self.scene["start_pose"], dtype=np.float32)
        self._model: mujoco.MjModel | None = None
        image = spaces.Box(0, 255, (self.height, self.width, 3), np.uint8)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-360.0, 360.0, (len(JOINTS),), np.float32),
                "pixels": spaces.Dict({name: image for name in CAMERAS}),
            }
        )
        self.action_space = spaces.Box(-360.0, 360.0, (len(JOINTS),), np.float32)
        self._steps = 0
        self._resting = 0
        self._pending: list[np.ndarray] = []
        self._taken: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in CAMERAS}
        self._photo: dict | None = None
        self._block_start = np.zeros(7)
        self._backdrop: tuple | None = None

    def _ensure_world(self) -> None:
        """Build on first use, so each process that renders owns its GL context."""
        if self._model is not None:
            return
        self._model = build_model(self.scene, self.scene_dir)
        model = self._model
        self.data = mujoco.MjData(model)
        self.render_size = {name: (int(WIDTH * scale), int(HEIGHT * scale)) for name, scale in RENDER_SCALE.items()}
        self.renderers = {name: mujoco.Renderer(model, height, width) for name, (width, height) in self.render_size.items()}
        self._jar_passes: tuple[bytes, np.ndarray] | None = None
        self.qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
        self.actuators = np.array([model.actuator(name).id for name in JOINTS])
        self.lo, self.hi = model.actuator_ctrlrange[self.actuators].T.copy()
        self.block_qadr = model.joint("block_joint").qposadr[0]
        self.jar_qadr = model.joint("jar_joint").qposadr[0]
        self.block_body, self.jar_body = model.body("block").id, model.body("jar").id
        self.table_height = self.scene["table"]["height"]
        jaw_bodies = {model.body(name).id for name in JAW_BODIES}
        self.is_jaw = np.array([model.geom_group[g] == GROUP_ARM and model.geom_bodyid[g] in jaw_bodies for g in range(model.ngeom)])
        self.is_base = model.geom_group == GROUP_BASE
        self.is_block = model.geom_bodyid == self.block_body
        self.is_jar = model.geom_bodyid == self.jar_body
        self.is_arm = model.geom_group == GROUP_ARM
        self.jar_tint_geom = model.geom("jar_tint").id
        self.printed_materials = [m for m in range(model.nmat) if model.material(m).name.endswith("_material") and not model.material(m).name.startswith("sts3215")]
        self.motor_materials = [m for m in range(model.nmat) if model.material(m).name.startswith("sts3215")]
        self.block_material = model.material("block_mat").id
        self.block_hole_material = model.material("block_hole_mat").id
        self.scene_colours = {name: model.material(f"{name}_mat").id for name in ("wall", "floor")}
        self.lenses, self.maps, self.ray_cache = {}, {}, {}
        for name in CAMERAS:
            self.set_lens(name, self.scene["cameras"][name]["lens"])
        self.plate = self._read("plate")
        cutout = cv2.imread(str(self.scene_dir / self.scene["cameras"]["top"]["jar"]["file"]), cv2.IMREAD_UNCHANGED)
        self.jar_photo = cv2.cvtColor(cutout[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        self.jar_cover = cutout[:, :, 3].astype(np.float32) / 255
        # What each camera draws. The top camera's still parts are in its real picture already.
        self.options = {}
        for name, groups in (("top", (GROUP_ARM, GROUP_BLOCK)), ("wrist", (GROUP_SCENE, GROUP_BASE, GROUP_ARM, GROUP_BLOCK, GROUP_JAR))):
            self.options[name] = self._option(*groups)
        self.top_shapes = self._option(GROUP_BASE, GROUP_ARM, GROUP_BLOCK)
        self.jar_only = self._option(GROUP_JAR)
        self.top_with_jar = self._option(GROUP_ARM, GROUP_BLOCK, GROUP_JAR)
        self.arm_only = self._option(GROUP_ARM)
        if self.demos_dir and (Path(self.demos_dir) / "meta" / "info.json").is_file():
            from so_101.sim.demos import Demos

            self.use_photos(Demos(Path(self.demos_dir)))

    def use_photos(self, demos) -> None:
        """Draw the arm in the top camera, and the wrist camera's whole view, from these demos' real frames."""
        from so_101.sim.photo import ArmPhotos, WristPhotos

        self.arm_photos = ArmPhotos(demos, self.scene, self._model)
        self.wrist_photos = WristPhotos(demos, self.scene, self._model)

    @staticmethod
    def _option(*groups: int) -> mujoco.MjvOption:
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        for group in groups:
            option.geomgroup[group] = 1
        return option

    def _read(self, name: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.scene_dir / self.scene["cameras"]["top"][name])), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _mj_camera(name: str) -> str:
        return "wrist_cam" if name == "wrist" else name

    def set_motors(self, motors: dict, delay_ticks: int) -> None:
        """Use these motors from now on, without rebuilding the world. A fit calls it for each set it tries."""
        model = self._model
        for name, motor in motors.items():
            actuator, dof = model.actuator(name).id, model.joint(name).dofadr[0]
            model.actuator_gainprm[actuator, 0] = motor["kp"]
            model.actuator_biasprm[actuator, 1:3] = [-motor["kp"], -motor["kv"]]
            model.actuator_forcerange[actuator] = [-motor["force"], motor["force"]]
            model.dof_frictionloss[dof], model.dof_armature[dof], model.dof_damping[dof] = motor["friction"], motor["armature"], motor["damping"]
        self.scene["arm"]["motors"], self.scene["arm"]["delay_ticks"] = motors, delay_ticks

    def set_lens(self, name: str, lens: dict) -> None:
        """Draw a camera through this lens: set how wide a pinhole picture it needs and where each pixel reads it."""
        self.lenses[name] = Lens.from_scene(lens)
        self._model.cam_fovy[self._model.camera(self._mj_camera(name)).id] = self.lenses[name].render_fovy()
        self.maps[name] = self.lenses[name].render_map(*self.render_size[name])
        self._jar_passes = None
        self.ray_cache.pop(name, None)

    def rays(self, name: str) -> np.ndarray:
        if name not in self.ray_cache:
            self.ray_cache[name] = self.lenses[name].rays()
        return self.ray_cache[name]

    def on_table(self, name: str) -> np.ndarray:
        """Where each pixel of a camera meets the mat's level, in the arm's axes. Needs `mj_forward` to have
        placed the camera."""
        camera = self._model.camera(self._mj_camera(name)).id
        position = self.data.cam_xpos[camera]
        world = self.rays(name) @ self.data.cam_xmat[camera].reshape(3, 3).T
        along = (self.table_height - position[2]) / np.minimum(world[..., 2], -1e-6)
        return position + world * along[..., None]

    def wrist_table_texels(self) -> tuple[np.ndarray, np.ndarray]:
        """For each pixel of the wrist camera, the column and row of the table texture it looks at."""
        table = self.scene["table"]
        points = self.on_table("wrist")
        return (
            ((table["y"][1] - points[..., 1]) / table["texel"] - 0.5).astype(np.float32),
            ((table["x"][1] - points[..., 0]) / table["texel"] - 0.5).astype(np.float32),
        )

    def to_ctrl(self, values: np.ndarray) -> np.ndarray:
        return np.clip(self.units.to_qpos(values), self.lo, self.hi)

    def joint_state(self) -> np.ndarray:
        return self.units.from_qpos(self.data.qpos[self.qadr]).astype(np.float32)

    def set_joints(self, values: np.ndarray) -> None:
        """Put the arm at follower readings, with no physics. Call `mj_forward` afterwards."""
        self.data.qpos[self.qadr] = self.units.to_qpos(values)

    def set_joints_of(self, data: mujoco.MjData, values: np.ndarray) -> None:
        """Put another copy of the world's arm at follower readings, ready to draw."""
        data.qpos[:] = self.data.qpos
        data.qpos[self.qadr] = self.units.to_qpos(values)
        mujoco.mj_kinematics(self._model, data)
        mujoco.mj_camlight(self._model, data)

    def depth_of(self, data: mujoco.MjData, name: str, option: mujoco.MjvOption) -> np.ndarray:
        """How far along a camera's view each pixel's surface is, in metres, far beyond the scene for none."""
        renderer = self.renderers[name]
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=self._mj_camera(name), scene_option=option)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        return cv2.remap(depth, *self.maps[name], cv2.INTER_NEAREST, borderValue=1e9)

    def block_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.block_body].copy(), self.data.xmat[self.block_body].reshape(3, 3).copy()

    def jar_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.jar_body].copy(), self.data.xmat[self.jar_body].reshape(3, 3).copy()

    def jar_upright(self) -> bool:
        return bool(self.jar_pose()[1][2, 2] > JAR_UPRIGHT_COS)

    def block_in_jar(self) -> bool:
        """The block's centre is inside the jar's wall and below its rim, and the jar still stands."""
        jar = self.scene["jar"]
        position, rotation = self.jar_pose()
        local = (self.data.xpos[self.block_body] - position) @ rotation
        return bool(self.jar_upright() and np.hypot(local[0], local[1]) < jar["radius"] - jar["wall"] and 0 < local[2] < jar["height"])

    def set_block(self, xy, yaw: float = 0.0) -> None:
        """Lay the block flat on the mat."""
        adr = self.block_qadr
        self.data.qpos[adr : adr + 3] = [*xy, self.table_height + self.scene["block"]["height"] / 2 + 0.0005]
        self.data.qpos[adr + 3 : adr + 7] = yaw_quat(yaw)
        self.data.qvel[self._model.joint("block_joint").dofadr[0] :][:6] = 0

    def set_block_pose(self, position, rotation) -> None:
        adr = self.block_qadr
        self.data.qpos[adr : adr + 3] = position
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=np.float64).ravel())
        self.data.qpos[adr + 3 : adr + 7] = quat

    def set_jar(self, xy) -> None:
        """Stand the jar on the mat."""
        adr = self.jar_qadr
        self.data.qpos[adr : adr + 3] = [*xy, self.table_height + 0.0005]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._model.joint("jar_joint").dofadr[0] :][:6] = 0

    # Drawing.

    def _pass(self, name: str, option: mujoco.MjvOption, shapes: bool = False, cover: np.ndarray | None = None, data: mujoco.MjData | None = None):
        """One pinhole picture from a camera, bent through its lens: its colours, or with `shapes` which
        geom each pixel shows, -1 for none. With `cover`, which geoms count, it also returns how much of
        each pixel those geoms cover, between 0 and 1: a smooth edge to draw them over a photo with.
        `data` draws another copy of the world than the one being simulated."""
        renderer = self.renderers[name]
        if shapes:
            renderer.enable_segmentation_rendering()
        renderer.update_scene(self.data if data is None else data, camera=self._mj_camera(name), scene_option=option)
        image = renderer.render()
        if not shapes:
            return cv2.remap(image, *self.maps[name], cv2.INTER_LINEAR)
        renderer.disable_segmentation_rendering()
        ids = image[:, :, 0]
        bent = cv2.remap(ids.astype(np.float32), *self.maps[name], cv2.INTER_NEAREST, borderValue=-1).astype(np.int32)
        if cover is None:
            return bent
        counted = ((ids >= 0) & cover[ids.clip(0)]).astype(np.float32)
        return bent, cv2.remap(cv2.GaussianBlur(counted, (0, 0), 0.6), *self.maps[name], cv2.INTER_LINEAR)

    def _jar_from_top(self) -> np.ndarray:
        """The top camera's pixels the jar covers. Kept until the jar moves, which it rarely does."""
        where = self.data.qpos[self.jar_qadr : self.jar_qadr + 7].round(5).tobytes()
        if self._jar_passes is None or self._jar_passes[0] != where:
            self._jar_passes = (where, self._pass("top", self.jar_only, shapes=True) >= 0)
        return self._jar_passes[1]

    def _colour(self, name: str) -> None:
        """Each real camera sees the same things in different colours, so each has its own."""
        colours = self.scene["colours"][name]
        model = self._model
        model.mat_rgba[self.printed_materials, :3] = colours["arm"]
        model.mat_rgba[self.motor_materials, :3] = colours["motor"]
        model.mat_rgba[self.block_material, :3] = colours["block"]
        model.mat_rgba[self.block_hole_material, :3] = colours["block_hole"]
        for part in ("tint", "clear"):
            model.mat_rgba[model.material(f"jar_{part}_mat").id] = colours[f"jar_{part}"]
        for part, material in self.scene_colours.items():
            model.mat_rgba[material, :3] = colours.get(part, self.scene[part]["rgb"])

    def segment(self, name: str, *groups: int) -> np.ndarray:
        """Which geom each pixel of a camera shows, -1 for none. By default, what the camera draws; the
        top camera's also includes the arm's base."""
        option = self._option(*groups) if groups else self.top_shapes if name == "top" else self.options[name]
        return self._pass(name, option, shapes=True)

    def _top_pixel(self, point) -> np.ndarray:
        camera = self._model.camera("top").id
        return self.lenses["top"].project((np.asarray(point) - self.data.cam_xpos[camera]) @ self.data.cam_xmat[camera].reshape(3, 3))

    def jar_shift(self, photo_xy) -> np.ndarray:
        """How far, in the top camera's pixels, the jar stands from where it stood for a photo of it."""
        middle = np.array([0.0, 0.0, self.scene["jar"]["height"] / 2])
        return self._top_pixel(self.jar_pose()[0] + middle) - self._top_pixel(np.array([*photo_xy, self.table_height]) + middle)

    def photo_from(self, layout: dict | None) -> None:
        """Take the block's and jar's pixels from this demo's first real frame from now on, or with None
        go back to drawing them. Leaves the block lying where that demo's lay."""
        if layout is None or self.arm_photos is None:
            self._photo = None
        elif self._photo is None or self._photo["episode"] != layout["episode"]:
            self._photo = self._start_photo(layout)
        if self._photo is not None:
            self.set_block(layout["block"][:2], layout["block"][2])
            mujoco.mj_forward(self._model, self.data)
            self._block_start = self.data.qpos[self.block_qadr : self.block_qadr + 7].copy()
        self._backdrop = None

    def _start_photo(self, layout: dict) -> dict:
        """A demo's first top camera frame, with which of its pixels are its jar and its block. Called
        before the block and jar are placed, since it draws the block where the demo's lay."""
        soft = lambda mask, grow: cv2.GaussianBlur(cv2.dilate(mask.astype(np.uint8), np.ones((grow, grow), np.uint8)).astype(np.float32), (0, 0), 1.2)[..., None]  # noqa: E731
        top = self.scene["cameras"]["top"]
        seen = layout.get("seen", layout["block"])  # where the top camera saw it, which is where its pixels are
        self.set_block(seen[:2], seen[2])
        mujoco.mj_forward(self._model, self.data)
        block = self._pass("top", self._option(GROUP_BLOCK), shapes=True) >= 0
        self.set_block(layout["block"][:2], layout["block"][2])
        mujoco.mj_forward(self._model, self.data)
        return {
            "frame": self.arm_photos.demos.frame(layout["episode"], 0).astype(np.float32),
            "jar": soft(jar_outline(top["pose"], self.lenses["top"], layout["jar"], self.scene["jar"], self.table_height), 7),
            "block": soft(block, 9),  # wide enough for its soft shadow on the mat
            "jar_xy": layout["jar"],
            "block_at": self.data.xpos[self.block_body].copy(),
            "episode": layout["episode"],
        }

    def block_untouched(self) -> bool:
        """The block still lies where it was put at the start, to within a millimetre and half a degree."""
        now = self.data.qpos[self.block_qadr : self.block_qadr + 7]
        return bool(np.linalg.norm(now[:3] - self._block_start[:3]) < 0.001 and abs(now[3:] @ self._block_start[3:]) > 0.99999)

    def backdrop(self) -> np.ndarray:
        """The top camera's real picture with the real jar where the simulated one stands, and with the
        real block too while a demo's first frame supplies them and the block lies untouched. A photo is
        moved by whole pixels, which keeps it as sharp as it was taken."""

        def moved(image, shift):
            return cv2.warpAffine(image, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (self.width, self.height), flags=cv2.INTER_NEAREST)

        if self._photo is None:
            shift = np.round(self.jar_shift(self.scene["cameras"]["top"]["jar"]["xy"]))
            key = (*shift, False)
            if self._backdrop is None or self._backdrop[0] != key:
                cover = moved(self.jar_cover, shift)[..., None]
                self._backdrop = (key, self.plate * (1 - cover) + moved(self.jar_photo, shift) * cover)
            return self._backdrop[1]
        photo, untouched = self._photo, self.block_untouched()
        jar_shift = np.round(self.jar_shift(photo["jar_xy"]))
        block_shift = np.round(self._top_pixel(self.data.xpos[self.block_body]) - self._top_pixel(photo["block_at"])) if untouched else np.zeros(2)
        key = (*jar_shift, *block_shift, untouched)
        if self._backdrop is None or self._backdrop[0] != key:
            cover = moved(photo["jar"], jar_shift)[..., None]
            picture = self.plate * (1 - cover) + moved(photo["frame"], jar_shift) * cover
            if untouched:
                cover = moved(photo["block"], block_shift)[..., None]
                picture = picture * (1 - cover) + moved(photo["frame"], block_shift) * cover
            self._backdrop = (key, picture)
        return self._backdrop[1]

    def render_camera(self, name: str) -> np.ndarray:
        self._colour(name)
        if name == "wrist":
            image = self._soften(self._pass(name, self.options[name]), name)
            if self.wrist_photos is not None:
                image = self.wrist_photos.paint(self, self._pass(name, self.options[name], shapes=True), self.depth_of(self.data, name, self.options[name]), image)
            return image
        standing = self.jar_upright()
        # What the real picture already shows: the arm's base, and the block while a demo's frame supplies it.
        photographed = self.is_base | self.is_block if standing and self._photo is not None and self.block_untouched() else self.is_base
        image = self._pass(name, self.options[name] if standing else self.top_with_jar).astype(np.float32)
        shapes, cover = self._pass(name, self.top_shapes if standing else self.top_with_jar, shapes=True, cover=~photographed)
        drawn = (shapes >= 0) & ~photographed[shapes.clip(0)]
        image = self._soften(image, name)
        if self.arm_photos is not None:
            image, beside = self.arm_photos.paint(self, shapes, self.depth_of(self.data, name, self.top_shapes if standing else self.top_with_jar), image)
            cover = np.maximum(cover, beside)
        if standing:
            if (drawn & self._jar_from_top()).any():
                # Where the jar's wall is in front of the arm or block, the real jar shows instead: nearly
                # all of it where the wall is tinted, part where it's clear. Seen with the jar's wall drawn
                # too: through its open top, the block comes before the far side's wall.
                front = self._pass(name, self.top_with_jar, shapes=True)
                behind = drawn & self.is_jar[front.clip(0)]
                through = np.where(front == self.jar_tint_geom, self.scene["jar"]["tint_see_through"], self.scene["jar"]["clear_see_through"])
                cover = cover * np.where(behind, through, 1.0)
            background = self.backdrop()
        else:
            background = self.plate.astype(np.float32)
        cover = cover[..., None]
        return (background * (1 - cover) + image * cover).clip(0, 255).astype(np.uint8)

    def _soften(self, image: np.ndarray, name: str) -> np.ndarray:
        """How soft the real camera's images are, measured from its frames by `so101 sim build`."""
        sigma = self.scene.get("softness", {}).get(name, 0.8)
        return cv2.GaussianBlur(image, (0, 0), sigma) if sigma > 0 else image

    def _obs(self) -> dict[str, Any]:
        return {"agent_pos": self.joint_state(), "pixels": {name: self._picture(name) for name in CAMERAS}}

    def _picture(self, name: str) -> np.ndarray:
        """What a camera hands over this tick: like the real ones, a picture taken a few ticks ago, and
        the same one again when the camera is slower than the control loop. `so101 sim build` measures
        both from the demos."""
        timing = self.scene["cameras"][name]
        taken = self._taken[name]
        # The camera takes a picture whenever its own clock ticks over: 2 control ticks in 3 at 20 a second.
        fps = min(timing.get("fps", self.fps), self.fps)
        if not taken or int(self._steps * fps / self.fps) != int((self._steps - 1) * fps / self.fps):
            taken.append((self._steps, self.render_camera(name)))
        ready = [picture for tick, picture in taken if tick <= self._steps - timing.get("delay_ticks", 0)]
        del taken[: max(len(ready) - 1, 0)]  # older pictures are never handed over again
        return ready[-1] if ready else taken[0][1]

    # Running.

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """Options: `start` picks a demo's layout, `block` is [x, y, yaw] and `jar` is [x, y], in metres
        and radians in the arm's axes, and `joints` is the arm's pose in the follower's units."""
        self._ensure_world()
        super().reset(seed=seed)
        options = kwargs.get("options") or {}
        mujoco.mj_resetData(self._model, self.data)
        joints = np.asarray(options.get("joints", self.initial_joints), dtype=np.float64)
        self.data.qpos[self.qadr] = self.to_ctrl(joints)
        self.data.ctrl[self.actuators] = self.data.qpos[self.qadr]
        starts = self.scene["starts"]
        chosen = options.get("start", self.start)
        exact = chosen is not None or "block" in options
        layout = starts[(chosen if chosen is not None else int(self.np_random.integers(len(starts)))) % len(starts)]
        block, jar = np.array(options.get("block", layout["block"]), dtype=np.float64), np.array(options.get("jar", layout["jar"]), dtype=np.float64)
        if not exact and self.jitter:
            block += self.np_random.normal(0, 1, 3) * np.array([0.003, 0.003, np.radians(3.0)]) * self.jitter
            jar += self.np_random.normal(0, 0.003, 2) * self.jitter
        # The demo this layout is from, if any: its first real frame then supplies the block's and jar's pixels.
        source = next((entry for entry in starts if np.allclose(entry["block"], options.get("block", layout["block"])) and np.allclose(entry["jar"], options.get("jar", layout["jar"]))), None)
        self.photo_from(source)
        self.set_block(block[:2], float(block[2]))
        self.set_jar(jar)
        mujoco.mj_forward(self._model, self.data)
        self._block_start = self.data.qpos[self.block_qadr : self.block_qadr + 7].copy()
        self._backdrop = None
        self._steps, self._resting = 0, 0
        self._pending.clear()
        self._taken = {name: [] for name in CAMERAS}
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
        mujoco.mj_step(self._model, self.data, nstep=SUBSTEPS)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._ensure_world()
        self.apply(action)
        self._steps += 1
        self._resting = self._resting + 1 if self.block_in_jar() else 0
        success = self._resting >= self.settle_success_steps
        truncated = self._steps >= self._max_episode_steps
        return self._obs(), float(success), success, truncated, {"is_success": success, "block_in_jar": self._resting > 0}

    def render(self) -> np.ndarray:
        self._ensure_world()
        return self.render_camera("top")

    def close(self) -> None:
        if self._model is not None:
            for renderer in self.renderers.values():
                renderer.close()
            self._model = None


def make_env(n_envs: int = 1, use_async_envs: bool = False, **kwargs: Any) -> dict[str, dict[int, Any]]:
    """EnvHub entry point. `kwargs` override `BlockContainerEnv`'s, and SO101_SIM_SCENE names the scene."""
    factory = partial(BlockContainerEnv, **kwargs)
    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return {TASK: {0: vector_cls([factory] * n_envs)}}
