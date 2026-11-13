"""Real pixels for the simulated cameras: the arm in the top camera's picture, taken from the demos.

A rendered arm has the right outline but none of the real one's cables, print lines, holes and shading,
and a policy trained on real frames reads those. The top camera never moves, so how the arm looks
depends only on its joints, and the demos hold tens of thousands of real frames of it. For the pose
being drawn, the nearest of those frames is found, and each of the drawn arm's pixels takes the colour
of the same spot on the same link in that frame. The arm model only has to carry each spot across the
small difference between the two poses, so close to a demo's pose the arm is drawn where and how the
real one appeared, whatever the camera fit's remaining error.
"""

import cv2
import mujoco
import numpy as np
from scipy.spatial import cKDTree

from so_101.sim import arm
from so_101.sim.camera import to_camera
from so_101.sim.demos import Demos
from so_101.sim.looks import red_mask, wrist_jar_mask

# How much a degree of each joint, and a unit of the gripper, counts when looking for the nearest frame:
# the joints nearer the base move more of the arm across the picture.
JOINT_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.7, 0.5, 0.3])
# A spot counts as showing in the real frame if the arm there is within this of it, in metres.
SAME_SURFACE = 0.006
FILL_REACH = 6.0  # pixels
WHOLE_BLOCK = 0.8  # the share of the drawn block's pixels a real frame has to show for any to be used
MARGIN = 12.0  # pixels round the drawn arm that may show more of the real one
REGISTER_REACH = 45.0  # pixels: further off than this, a patch of the right colour is something else
LINKS = arm.MOVING_BODIES


def _read(image: np.ndarray, pixels: np.ndarray, how: int, outside) -> np.ndarray:
    """An image's values at a list of pixels. OpenCV reads at most 32,767 across, so the list is folded
    into rows."""
    wide = 4096
    padded = np.full((-(-len(pixels) // wide) * wide, 2), -1.0, np.float32)
    padded[: len(pixels)] = pixels
    x, y = padded[:, 0].reshape(-1, wide), padded[:, 1].reshape(-1, wide)
    values = cv2.remap(image, x, y, how, borderMode=cv2.BORDER_CONSTANT, borderValue=outside)
    return values.reshape(-1, *values.shape[2:])[: len(pixels)]


def _fill(painted: np.ndarray, wanted: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Give each wanted pixel that wasn't painted the colour of the nearest one that was, if one is within
    a few pixels; otherwise it stays as rendered."""
    missing = wanted & ~done
    if done.any() and missing.any():
        distance, nearest = cv2.distanceTransformWithLabels((~done).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        from_rows, from_columns = np.nonzero(done)  # labels number these in row order, from 1
        fill = missing & (distance < FILL_REACH)
        painted[fill] = painted[from_rows[nearest[fill] - 1], from_columns[nearest[fill] - 1]]
    return painted


class ArmPhotos:
    """The demos' top camera frames, indexed by the arm pose each one shows."""

    def __init__(self, demos: Demos, scene: dict, model: mujoco.MjModel, every: int = 2):
        self.demos = demos
        self.units = arm.Units.from_scene(scene)
        self.delay = scene["cameras"]["top"].get("delay_ticks", 0)
        self.frames, shown = [], []
        # Only the demos the fits used: `so101 sim score` and `check` compare with the others' real frames.
        for episode in scene["episodes"]:
            state = self.demos.episode(episode)[0]
            for frame in range(self.delay, len(state), every):
                self.frames.append((episode, frame))
                shown.append(state[frame - self.delay])  # a picture shows the arm as it was `delay` ticks before
        self.shown = np.array(shown)
        self.tree = cKDTree(self.shown * JOINT_WEIGHTS)
        # Which link each of the model's geoms is part of, -1 for the base, the block and the rest.
        names = {model.body(name).id: index for index, name in enumerate(LINKS)}
        names[model.body("camera_mount").id] = LINKS.index("gripper")  # fixed to the gripper
        self.geom_link = np.array([names.get(int(body), -1) for body in model.geom_bodyid])
        self.scratch = mujoco.MjData(model)
        self._cached: tuple[tuple[int, int], np.ndarray] | None = None
        self.layouts = {layout["episode"]: layout for layout in scene["starts"]}
        self._jars: dict[int, np.ndarray] = {}

    def nearest(self, state: np.ndarray) -> tuple[int, int, np.ndarray]:
        """The demo frame whose arm is closest to these follower readings: its episode, frame and readings."""
        _, index = self.tree.query(np.asarray(state, dtype=np.float64) * JOINT_WEIGHTS)
        return (*self.frames[index], self.shown[index])

    def _frame(self, episode: int, frame: int) -> np.ndarray:
        if self._cached is None or self._cached[0] != (episode, frame):
            self._cached = ((episode, frame), self.demos.frame(episode, frame))
        return self._cached[1]

    def paint(self, env, shapes: np.ndarray, depth: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recolour the arm in a top camera render from the nearest real frame.

        `shapes` and `depth` are which geom each pixel shows and how far along the camera's view it is;
        `image` is the render's colours, which stay wherever the real frame doesn't show that spot:
        behind another link there, or outside its picture.

        Also returns how much of each pixel just outside the drawn arm to take from the real frame: the
        real arm has cables and edges the model doesn't, which ride along with the link beside them.
        """
        links = np.where(shapes >= 0, self.geom_link[shapes.clip(0)], -1)
        beside = np.zeros(shapes.shape, np.float32)
        if not (links >= 0).any():
            return image, beside
        # Each pixel within a few of the arm is treated as part of the nearest link, at that link's depth.
        on_arm = links >= 0
        distance, nearest = cv2.distanceTransformWithLabels((~on_arm).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        arm_rows, arm_columns = np.nonzero(on_arm)  # labels number these in row order, from 1
        near = (distance > 0) & (distance < MARGIN) & (shapes < 0)
        rows, columns = np.nonzero(on_arm | near)
        outside = near[rows, columns]
        from_rows, from_columns = arm_rows[nearest[rows, columns] - 1], arm_columns[nearest[rows, columns] - 1]
        from_rows[~outside], from_columns[~outside] = rows[~outside], columns[~outside]
        link = links[from_rows, from_columns]
        depth = depth.copy()
        depth[rows, columns] = depth[from_rows, from_columns]
        state = env.joint_state().astype(np.float64)
        episode, frame, source_state = self.nearest(state)
        pose = env.scene["cameras"]["top"]["pose"]
        lens = env.lenses["top"]
        camera = env._model.camera("top").id
        # Each pixel's spot in the arm's axes, then on its link, then where that link was in the real frame.
        points = env.data.cam_xpos[camera] + (env.rays("top")[rows, columns] * depth[rows, columns, None]) @ env.data.cam_xmat[camera].reshape(3, 3).T
        here = arm.link_poses(self.units.to_qpos(state))
        there = arm.link_poses(self.units.to_qpos(source_state))
        moved = np.empty_like(points)
        for index, name in enumerate(LINKS):
            chosen = link == index
            local = (points[chosen] - here[name][1][0]) @ here[name][0][0]
            moved[chosen] = local @ there[name][0][0].T + there[name][1][0]
        in_camera = to_camera(moved, pose)
        pixels = lens.project(in_camera).astype(np.float32)
        # Is that spot the nearest thing to the camera in the real frame, or was it behind another link?
        env.set_joints_of(self.scratch, source_state)
        source_depth = env.depth_of(self.scratch, "top", env.arm_only)
        seen = np.abs(_read(source_depth, pixels, cv2.INTER_NEAREST, 1e9) + in_camera[:, 2]) < SAME_SURFACE
        real = self._frame(episode, frame)
        colours = _read(real, pixels, cv2.INTER_LINEAR, 0)
        # A pixel on the drawn arm's edge can land just off the real arm, on the wall or the mat behind it,
        # so only spots whose neighbours also show are taken from the real frame.
        shows = np.zeros(shapes.shape, np.uint8)
        shows[rows[seen & ~outside], columns[seen & ~outside]] = 1
        inside = cv2.erode(shows, np.ones((3, 3), np.uint8)) > 0
        painted = image.copy()
        keep = inside[rows, columns] & ~outside
        painted[rows[keep], columns[keep]] = colours[keep]
        # The rest of the arm, its edges and what another link hid in the real frame, is filled in from beside it.
        painted = _fill(painted, on_arm, inside)
        # Just outside the drawn arm, the real frame is taken where it shows something that isn't the empty
        # scene, the demo's block or its jar: a cable, or the real arm's edge where the model's falls short.
        there_empty = _read(env.plate, pixels[outside], cv2.INTER_LINEAR, 0).astype(np.int16)
        there = colours[outside].astype(np.int16)
        differs = np.abs(there - there_empty).max(axis=1)
        other = _read((red_mask(real) | self._jar_in(env, episode)).astype(np.uint8), pixels[outside], cv2.INTER_NEAREST, 1)
        strength = np.clip((differs - 20.0) / 25.0, 0, 1) * (other == 0) * np.clip(1.5 - distance[rows[outside], columns[outside]] / MARGIN * 1.5, 0, 1)
        painted[rows[outside], columns[outside]] = colours[outside]
        beside[rows[outside], columns[outside]] = strength
        return painted, beside

    def _jar_in(self, env, episode: int) -> np.ndarray:
        """The pixels a demo's jar covers in its top camera frames, with a margin."""
        if episode not in self._jars:
            from so_101.sim.env import jar_outline

            top = env.scene["cameras"]["top"]
            outline = jar_outline(top["pose"], env.lenses["top"], self.layouts[episode]["jar"], env.scene["jar"], env.table_height)
            self._jars[episode] = cv2.dilate(outline.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return self._jars[episode]


# The block's part in a wrist frame decides which real frames can stand in for it.
UNTOUCHED, HELD, RELEASED = 0, 1, 2


class WristPhotos:
    """The demos' wrist camera frames, indexed by where the camera was relative to the block or the jar.

    The wrist camera moves, so nothing in its picture stays put. But each demo's block, jar, mat and
    wall are at known places, and so is the camera, from the joints. So for the view being drawn, the real
    frame taken from the most similar place is found, and every drawn pixel takes the colour of the same
    spot on the same thing in that frame: the mat and wall where they are, the block and jar where that
    demo's stood, the jaws on the gripper.
    """

    def __init__(self, demos: Demos, scene: dict, model: mujoco.MjModel, every: int = 2):
        self.demos = demos
        self.units = arm.Units.from_scene(scene)
        self.delay = scene["cameras"]["wrist"].get("delay_ticks", 0)
        self.table_height = scene["table"]["height"]
        self.block_height, self.jar_base = scene["block"]["height"], scene["jar"]["base"]
        layouts = {layout["episode"]: layout for layout in scene["starts"]}
        camera = model.camera("wrist_cam")
        mount = np.empty(9)
        mujoco.mju_quat2Mat(mount, model.cam_quat[camera.id])
        self.camera_at, self.camera_turn = model.cam_pos[camera.id].copy(), mount.reshape(3, 3)
        self.frames, states, phases, anchors = [], [], [], []
        self.holds = {}
        for episode in scene["episodes"]:
            grasp = demos.grasp_and_release(episode)
            if grasp is None or episode not in layouts:
                continue
            hold, let_go = grasp
            layout = layouts[episode]
            state = demos.episode(episode)[0]
            self.holds[episode] = (layout, state[hold])
            for frame in range(self.delay, len(state), every):
                # Around the grasp and the release the block is on its way from one part to the next.
                phase = UNTOUCHED if frame < hold - 3 else HELD if hold + 5 <= frame < let_go else RELEASED if frame >= let_go + 20 else None
                if phase is None:
                    continue
                self.frames.append((episode, frame))
                states.append(state[frame - self.delay])
                phases.append(phase)
                anchors.append([*layout["block"][:2], self.table_height] if phase == UNTOUCHED else [*layout["jar"], self.table_height])
        self.states, self.phases = np.array(states), np.array(phases)
        keys = self._keys(self.states, np.array(anchors))
        self.trees = {phase: (np.flatnonzero(self.phases == phase), cKDTree(keys[self.phases == phase])) for phase in (UNTOUCHED, HELD, RELEASED) if (self.phases == phase).any()}
        names = {model.body(name).id: index for index, name in enumerate(LINKS)}
        names[model.body("camera_mount").id] = LINKS.index("gripper")
        # What each geom is fixed to: a link of the arm (0 and up), the room (-1), the block (-2) or the jar (-3).
        self.geom_part = np.array([names.get(int(body), -1) for body in model.geom_bodyid])
        self.geom_part[model.geom_bodyid == model.body("block").id] = -2
        self.geom_part[model.geom_bodyid == model.body("jar").id] = -3
        self.geom_part[model.geom_bodyid == model.body("base").id] = -1
        self.scratch = mujoco.MjData(model)
        self._cached: tuple[tuple[int, int], np.ndarray] | None = None

    def _keys(self, states: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """What makes two wrist views alike: where the camera is from the block or jar, which way it
        looks, and how far the jaws are open. A centimetre of the first counts like turning to look at a
        point a centimetre aside 10 cm ahead, and like the jaws 3 of their 100 further open."""
        rotations, positions = arm.gripper_poses(self.units.to_qpos(states))
        camera = positions + rotations @ self.camera_at
        ahead = -(rotations @ self.camera_turn[:, 2])
        return np.c_[camera - anchors, ahead * 0.1, states[:, 5] * 0.003]

    def _frame(self, episode: int, frame: int) -> np.ndarray:
        if self._cached is None or self._cached[0] != (episode, frame):
            self._cached = ((episode, frame), self.demos.frame(episode, frame, "wrist"))
        return self._cached[1]

    def _held_pose(self, layout: dict, hold_state: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where a demo's block was while carried: fixed to the gripper as it lay when the jaws closed on it."""
        rotations, positions = arm.gripper_poses(self.units.to_qpos(np.array([hold_state, state])))
        centre = arm.held_block(self.units, hold_state[None])[0]
        centre[2] = self.table_height + self.block_height / 2
        carried = rotations[1] @ rotations[0].T
        return positions[1] + carried @ (centre - positions[0]), carried @ _yaw(layout["block"][2])

    def paint(self, env, shapes: np.ndarray, depth: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Recolour a wrist camera render from the nearest real frame. `shapes` and `depth` are which
        geom each pixel shows and how far along the camera's view it is; `image` is the render's colours,
        which stay wherever the real frame doesn't show that spot."""
        model = env._model
        camera = model.camera("wrist_cam").id
        state = env.joint_state().astype(np.float64)
        block_at, block_turn = env.block_pose()
        jar_at, jar_turn = env.jar_pose()
        if env.block_untouched():
            phase, anchor = UNTOUCHED, block_at
        elif env.block_in_jar():
            phase, anchor = RELEASED, jar_at
        elif block_at[2] > env.table_height + self.block_height:  # off the mat: carried
            phase, anchor = HELD, jar_at
        else:  # knocked or dropped somewhere on the mat
            phase, anchor = UNTOUCHED, block_at
        if phase not in self.trees:  # no demo frame shows the block in that part: leave the render as it is
            return image
        anchor = np.array([anchor[0], anchor[1], self.table_height])
        indices, tree = self.trees[phase]
        index = indices[tree.query(self._keys(state[None], anchor[None])[0])[1]]
        episode, frame = self.frames[index]
        source_state = self.states[index]
        layout, hold_state = self.holds[episode]

        # The real frame's world: its arm, and its block and jar where that demo's were.
        scratch = self.scratch
        env.set_joints_of(scratch, source_state)
        if self.phases[index] == UNTOUCHED:
            source_block = (np.array([*layout["block"][:2], self.table_height + self.block_height / 2]), _yaw(layout["block"][2]))
        elif self.phases[index] == HELD:
            source_block = self._held_pose(layout, hold_state, source_state)
        else:
            source_block = (np.array([*layout["jar"], self.table_height + self.jar_base + self.block_height / 2]), _yaw(layout["block"][2]))
        source_jar = (np.array([*layout["jar"], self.table_height]), np.eye(3))
        for address, (position, turn) in ((env.block_qadr, source_block), (env.jar_qadr, source_jar)):
            scratch.qpos[address : address + 3] = position
            mujoco.mju_mat2Quat(scratch.qpos[address + 3 : address + 7], np.ascontiguousarray(turn).ravel())
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_camlight(model, scratch)

        parts = np.where(shapes >= 0, self.geom_part[shapes.clip(0)], -9)
        rows, columns = np.nonzero(parts > -9)
        part = parts[rows, columns]
        points = env.data.cam_xpos[camera] + (env.rays("wrist")[rows, columns] * depth[rows, columns, None]) @ env.data.cam_xmat[camera].reshape(3, 3).T
        here = arm.link_poses(self.units.to_qpos(state))
        there = arm.link_poses(self.units.to_qpos(source_state))
        moves = {index: ((here[name][0][0], here[name][1][0]), (there[name][0][0], there[name][1][0])) for index, name in enumerate(LINKS)}
        moves[-2] = ((block_turn, block_at), (source_block[1], source_block[0]))
        moves[-3] = ((jar_turn, jar_at), (source_jar[1], source_jar[0]))
        moved = points.copy()  # the room stays where it is
        for index, ((turn, at), (source_turn, source_at)) in moves.items():
            chosen = part == index
            if chosen.any():
                moved[chosen] = ((points[chosen] - at) @ turn) @ source_turn.T + source_at
        in_camera = (moved - scratch.cam_xpos[camera]) @ scratch.cam_xmat[camera].reshape(3, 3)
        pixels = env.lenses["wrist"].project(in_camera).astype(np.float32)
        source_depth = env.depth_of(scratch, "wrist", env.options["wrist"])
        # Further off, the same error in where a thing stands is a smaller share of its distance.
        seen = np.abs(_read(source_depth, pixels, cv2.INTER_NEAREST, 1e9) + in_camera[:, 2]) < np.maximum(SAME_SURFACE, 0.05 * -in_camera[:, 2])
        real = self._frame(episode, frame)

        # The demo's block and jar are never quite where the arm model and the camera fit put them in its
        # frame. Each is found in the frame by its colour, and read from where it really is.
        source_parts = self.geom_part[env._pass("wrist", env.options["wrist"], shapes=True, data=scratch).clip(0)]
        read_at = pixels.copy()
        red = red_mask(real)
        for index, found, least in ((-2, red, 150), (-3, wrist_jar_mask(real), 800)):
            expected = source_parts == index
            if expected.sum() >= least and found.sum() >= least:
                shift = np.array(np.nonzero(found)).mean(axis=1)[::-1] - np.array(np.nonzero(expected)).mean(axis=1)[::-1]
                if np.linalg.norm(shift) < REGISTER_REACH:
                    read_at[part == index] += shift.astype(np.float32)
        # The block is red and nothing else is: a spot on the mat that reads red is the real block where the
        # model didn't expect it, and a spot on the block that doesn't is the mat. And if the real frame
        # saw the block from another side, only some of its spots show there, and a block patched together
        # from real and rendered pieces looks like neither: it's then left as rendered, whole.
        reads_red = _read(red.astype(np.uint8), read_at, cv2.INTER_NEAREST, 0) > 0
        block = part == -2
        seen &= np.where(block, reads_red, ~reads_red)
        if block.any() and seen[block].mean() < WHOLE_BLOCK:
            seen[block] = False
        # The jaws are fixed to the camera, so they're read from the very pixels they're drawn at, with the
        # opening of the nearest frame, which is rarely more than a pixel different.
        jaws = (part == LINKS.index("gripper")) | (part == LINKS.index("moving_jaw_so101_v1"))
        own = np.c_[columns, rows].astype(np.float32)
        read_at[jaws] = own[jaws]
        seen |= jaws
        # Round the jaws, what the drawn jaws don't cover of the real ones would otherwise be left as rendered mat.
        near_jaws = cv2.dilate((np.isin(parts, (LINKS.index("gripper"), LINKS.index("moving_jaw_so101_v1")))).astype(np.uint8), np.ones((13, 13), np.uint8))[rows, columns] > 0
        beside = near_jaws & ~seen
        read_at[beside] = own[beside]
        seen |= beside

        colours = _read(real, read_at, cv2.INTER_LINEAR, 0)
        painted = image.copy()
        painted[rows[seen], columns[seen]] = colours[seen]
        done = np.zeros(shapes.shape, bool)
        done[rows[seen], columns[seen]] = True
        return _fill(painted, parts > -9, done)


def _yaw(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
