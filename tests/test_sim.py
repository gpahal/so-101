"""The simulated twin (`so101 sim`): its lens, the arm's kinematics and the environment's physics.

The lens needs nothing extra. The rest needs the sim extra (`uv sync --extra sim`) and the arm model,
which `so_101.sim.arm.arm_xml` downloads from the Hub once, so those tests skip without either: CI
installs neither. They build a small scene of their own, so they need no demos.
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from so_101.sim.camera import Lens, bend, to_camera, unbend, widest


def _arm_model_missing() -> str | None:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return "needs the sim extra: uv sync --extra sim"
    from huggingface_hub import snapshot_download

    from so_101.sim.arm import ARM_REPO, ARM_REVISION

    try:
        snapshot_download(ARM_REPO, revision=ARM_REVISION, local_files_only=True)
    except Exception:
        return "needs the arm model, which any `so101 sim` command downloads"
    return None


SKIP = _arm_model_missing()


class LensTest(unittest.TestCase):
    def test_unbend_inverts_bend(self) -> None:
        angles = np.linspace(0, 1.2, 25)
        for k1, k2 in ((0.0, 0.0), (1 / 3, 2 / 15), (0.15, -0.05)):
            np.testing.assert_allclose(unbend(bend(angles, k1, k2), k1, k2), angles, atol=1e-6)

    def test_unbend_stops_where_the_lens_formula_folds_back(self) -> None:
        # This lens bends rays back towards the axis past about 1 radian, so two angles land on one pixel.
        k1, k2 = 0.15, -0.25
        limit = widest(k1, k2)
        self.assertLess(limit, 1.1)
        angles = np.linspace(0, limit, 20)
        np.testing.assert_allclose(unbend(bend(angles, k1, k2), k1, k2), angles, atol=1e-6)
        self.assertAlmostEqual(float(unbend(np.array([5.0]), k1, k2)[0]), limit, places=6)

    def test_a_pixels_ray_projects_back_onto_the_pixel(self) -> None:
        lens = Lens(focal=400.0, k1=0.1, k2=-0.05, centre=(12.0, -7.0))
        rays = lens.rays()
        pixels = lens.project(rays * 0.3)
        v, u = np.indices((lens.height, lens.width))
        np.testing.assert_allclose(pixels[..., 0], u, atol=1e-3)
        np.testing.assert_allclose(pixels[..., 1], v, atol=1e-3)

    def test_the_render_map_reads_each_pixels_ray_from_the_pinhole_picture(self) -> None:
        lens = Lens(focal=300.0, k1=0.2, k2=0.0, centre=(-9.0, 14.0))
        width, height = 1280, 960
        map_x, map_y = lens.render_map(width, height)
        focal = 0.5 * height / np.tan(np.radians(lens.render_fovy()) / 2)
        rays = lens.rays()
        np.testing.assert_allclose(map_x, width / 2 + rays[..., 0] * focal - 0.5, atol=1e-2)
        np.testing.assert_allclose(map_y, height / 2 - rays[..., 1] * focal - 0.5, atol=1e-2)
        # Every pixel reads from inside the pinhole picture.
        self.assertGreaterEqual(map_x.min(), 0)
        self.assertLessEqual(map_x.max(), width - 1)
        self.assertGreaterEqual(map_y.min(), 0)
        self.assertLessEqual(map_y.max(), height - 1)

    def test_a_point_straight_ahead_lands_in_the_middle(self) -> None:
        lens = Lens(focal=400.0, k1=0.1, k2=0.0)
        pose = [0.1, -0.2, 0.5, 0.0, 0.0, 0.0]  # looking straight down, x to the right
        pixel = lens.project(to_camera(np.array([[0.1, -0.2, 0.0]]), pose))[0]
        np.testing.assert_allclose(pixel, [319.5, 239.5], atol=1e-6)
        right = lens.project(to_camera(np.array([[0.15, -0.2, 0.0]]), pose))[0]
        self.assertGreater(right[0], pixel[0])
        self.assertAlmostEqual(right[1], pixel[1], places=6)
        # A lens whose axis is off the picture's middle shifts everything by that much.
        shifted = Lens(focal=400.0, k1=0.1, k2=0.0, centre=(10.0, -4.0))
        np.testing.assert_allclose(shifted.project(to_camera(np.array([[0.1, -0.2, 0.0]]), pose))[0], [329.5, 235.5], atol=1e-6)
        saved = Lens.from_scene(shifted.to_scene())
        self.assertEqual((saved.focal, saved.k1, saved.k2, saved.centre), (400.0, 0.1, 0.0, (10.0, -4.0)))


def scene_files(folder: Path) -> dict:
    """A small scene with every file the environment reads, built without any demos."""
    from so_101.sim import fit
    from so_101.sim.env import table_points

    table = {"x": [-0.1, 0.42], "y": [-0.3, 0.12], "texel": 0.004, "height": 0.003, "texture": "table.png", "friction": 0.6}
    texture = np.full((*table_points(table).shape[:2], 3), 40, np.uint8)
    cv2.imwrite(str(folder / "table.png"), texture)
    cv2.imwrite(str(folder / "top_background.png"), np.full((480, 640, 3), 90, np.uint8))
    cutout = np.zeros((480, 640, 4), np.uint8)
    cutout[100:160, 300:360] = (60, 150, 200, 255)
    cv2.imwrite(str(folder / "top_jar.png"), cutout)
    wrist_pose, wrist_lens = fit.wrist_start()
    colours = {
        "arm": [0.8, 0.8, 0.8], "motor": [0.1, 0.1, 0.1], "block": [0.8, 0.1, 0.1], "block_hole": [0.4, 0.05, 0.05],
        "jar_tint": [0.7, 0.5, 0.2, 0.9], "jar_clear": [0.8, 0.8, 0.8, 0.3], "wall": [0.8, 0.8, 0.8], "floor": [0.2, 0.2, 0.2],
    }  # fmt: skip
    scene = {
        "prompt": "Pick up the red block and put it in the cylindrical container",
        "start_pose": [0.0, -103.0, 94.0, 60.5, -2.0, 1.3],
        "arm": {
            "offsets_deg": [0.0, -1.7, -5.5, -0.7, -0.2], "gripper_closed_deg": -11.7, "gripper_deg_per_percent": 1.314,
            "motors": fit.starting_motors(), "delay_ticks": 0, "jaw_friction": 1.0,
            "limits": {
                "shoulder_pan": [-111.5, 111.5], "shoulder_lift": [-103.9, 103.9], "elbow_flex": [-96.6, 96.6],
                "wrist_flex": [-95.3, 95.3], "wrist_roll": [-180.0, 180.0], "gripper": [0.0, 100.0],
            },
        },
        "table": table,
        "wall": {"x": 0.42, "rgb": [0.8, 0.8, 0.8]},
        "floor": {"height": -0.74, "rgb": [0.2, 0.2, 0.2]},
        "block": {
            "width": 0.0318, "height": 0.0192, "mass": 0.0055, "friction": 0.3,
            "stud": {"radius": 0.0047, "height": 0.0045, "pitch": 0.016}, "rgb": [0.8, 0.1, 0.1], "hole_rgb": [0.4, 0.05, 0.05],
        },
        "jar": {
            "home": [0.37, -0.01], "radius": 0.032, "height": 0.063, "tint_height": 0.05, "wall": 0.002, "base": 0.003, "mass": 0.03,
            "friction": 0.4, "tint_rgba": [0.7, 0.5, 0.2, 0.9], "clear_rgba": [0.8, 0.8, 0.8, 0.3], "tint_see_through": 0.1, "clear_see_through": 0.6,
        },
        "cameras": {
            "top": {
                "pose": [0.12, -0.16, 0.36, 0.45, -0.16, -0.91], "lens": {"focal": 440.0, "k1": 0.15, "k2": -0.25},
                "plate": "top_background.png", "jar": {"file": "top_jar.png", "xy": [0.37, -0.01]},
            },
            "wrist": {"pose": wrist_pose.tolist(), "lens": wrist_lens},
        },
        "light": {"dir": [0.15, 0.1, -1.0], "ambient": 0.6, "direct": 0.5},
        "colours": {"top": colours, "wrist": colours},
        "softness": {"top": 0.8, "wrist": 0.8},
        "starts": [{"episode": 0, "block": [0.27, 0.0, 0.1], "jar": [0.37, -0.01], "grasp": [0.27, 0.0]}],
        "episodes": [0],
    }  # fmt: skip
    (folder / "scene.json").write_text(json.dumps(scene))
    return scene


@unittest.skipIf(SKIP, SKIP)
class ArmTest(unittest.TestCase):
    def test_batched_kinematics_match_mujoco(self) -> None:
        import mujoco

        from so_101.sim import arm

        model = arm.arm_model()
        data = mujoco.MjData(model)
        qpos = np.random.default_rng(0).uniform(-1, 1, (20, 6))
        poses = arm.link_poses(qpos)
        rotations, positions = arm.gripper_poses(qpos)
        for row in range(len(qpos)):
            data.qpos[:] = qpos[row]
            mujoco.mj_kinematics(model, data)
            for body in arm.MOVING_BODIES:
                np.testing.assert_allclose(poses[body][1][row], data.body(body).xpos, atol=1e-9)
                np.testing.assert_allclose(poses[body][0][row], data.body(body).xmat.reshape(3, 3), atol=1e-9)
            np.testing.assert_allclose(positions[row], data.body("gripper").xpos, atol=1e-9)
            np.testing.assert_allclose(rotations[row], data.body("gripper").xmat.reshape(3, 3), atol=1e-9)
            tips = np.mean([data.geom(f"{jaw}_jaw_sph_tip{i}").xpos for jaw in ("fixed", "moving") for i in (1, 2, 3)], axis=0)
            np.testing.assert_allclose(positions[row] + rotations[row] @ arm.between_jaws_local(qpos[row, 5]), tips, atol=1e-9)

    def test_base_stands_just_below_the_models_origin(self) -> None:
        from so_101.sim import arm

        self.assertAlmostEqual(arm.base_underside(), -0.0024, places=4)

    def test_units_convert_both_ways(self) -> None:
        from so_101.sim.arm import Units

        units = Units((0.0, -1.7, -5.5, -0.7, -0.2), -11.7, 1.314)
        readings = np.array([[12.0, 23.0, 2.0, 67.0, -17.5, 19.0], [0.0, -103.0, 94.0, 60.5, -2.0, 1.3]])
        np.testing.assert_allclose(units.from_qpos(units.to_qpos(readings)), readings, atol=1e-9)
        self.assertAlmostEqual(np.degrees(units.to_qpos(readings[0])[1]), 23.0 - 1.7)
        self.assertAlmostEqual(np.degrees(units.to_qpos(readings[0])[5]), -11.7 + 19.0 * 1.314)


@unittest.skipIf(SKIP, SKIP)
class EnvTest(unittest.TestCase):
    def setUp(self) -> None:
        from so_101.sim.env import BlockContainerEnv

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.scene = scene_files(Path(folder.name))
        self.env = BlockContainerEnv(scene_dir=folder.name)
        self.addCleanup(self.env.close)

    def test_reset_puts_the_arm_block_and_jar_where_asked(self) -> None:
        observation, info = self.env.reset(seed=0, options={"start": 0})
        np.testing.assert_allclose(observation["agent_pos"], self.scene["start_pose"], atol=1e-3)
        np.testing.assert_allclose(self.env.block_pose()[0][:2], [0.27, 0.0], atol=1e-6)
        np.testing.assert_allclose(self.env.jar_pose()[0][:2], [0.37, -0.01], atol=1e-6)
        self.assertFalse(info["is_success"])
        self.assertEqual({name: image.shape for name, image in observation["pixels"].items()}, {"top": (480, 640, 3), "wrist": (480, 640, 3)})
        self.assertEqual(observation["pixels"]["top"].dtype, np.uint8)

    def test_a_random_start_is_near_a_demos_and_the_same_for_the_same_seed(self) -> None:
        self.env.reset(seed=3)
        first = self.env.block_pose()[0][:2].copy()
        self.assertLess(np.linalg.norm(first - [0.27, 0.0]), 0.02)
        self.assertGreater(np.linalg.norm(first - [0.27, 0.0]), 0)
        self.env.reset(seed=3)
        np.testing.assert_allclose(self.env.block_pose()[0][:2], first)

    def test_the_block_and_jar_rest_on_the_mat_while_the_arm_holds_still(self) -> None:
        self.env.reset(seed=0, options={"start": 0})
        for _ in range(60):
            _, _, terminated, _, info = self.env.step(np.asarray(self.scene["start_pose"]))
        self.assertFalse(terminated)
        self.assertFalse(info["block_in_jar"])
        block, turn = self.env.block_pose()
        self.assertAlmostEqual(block[2], self.env.table_height + self.scene["block"]["height"] / 2, delta=0.001)
        np.testing.assert_allclose(block[:2], [0.27, 0.0], atol=0.001)
        self.assertGreater(turn[2, 2], 0.999)
        self.assertTrue(self.env.jar_upright())
        np.testing.assert_allclose(self.env.jar_pose()[0][:2], [0.37, -0.01], atol=0.001)

    def test_a_block_dropped_into_the_jar_succeeds_once_it_has_settled(self) -> None:
        self.env.reset(seed=0, options={"start": 0})
        self.env.set_block_pose([0.37, -0.01, self.env.table_height + 0.1], np.eye(3))
        hold = np.asarray(self.scene["start_pose"])
        ended = None
        for step in range(90):
            _, reward, terminated, _, info = self.env.step(hold)
            if terminated:
                ended = step
                break
        self.assertIsNotNone(ended, "the block never settled in the jar")
        self.assertGreaterEqual(ended, self.env.settle_success_steps - 1)
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["is_success"])
        self.assertTrue(self.env.jar_upright())

    def test_a_block_under_a_jaw_is_moved_clear_of_where_the_jaws_came_down(self) -> None:
        import mujoco

        from so_101.sim import fit

        env = self.env
        env.reset(seed=0, options={"start": 0})
        # Lower the open jaws until the fixed jaw's tip is within the block's height above the mat.
        tip = env._model.geom("fixed_jaw_sph_tip1").id
        for lift in np.arange(0.0, 60.0, 0.5):
            reading = np.array([0.0, lift, 5.0, 66.0, 0.0, 35.0])
            env.set_joints(reading)
            mujoco.mj_forward(env._model, env.data)
            if env.data.geom_xpos[tip][2] < env.table_height + 0.008:
                break
        # The block's edge 6 mm under that tip, as when the sim starts it a few millimetres off.
        outward = env.data.geom_xpos[tip][:2] / np.linalg.norm(env.data.geom_xpos[tip][:2])
        under = [float(v) for v in env.data.geom_xpos[tip][:2] + outward * 0.010]
        env.set_block(under, float(np.arctan2(outward[1], outward[0])))
        mujoco.mj_forward(env._model, env.data)
        self.assertGreater(env.arm_in_block(), 0.002)

        # A recording that holds that pose with its jaws open, then closes them and holds the block.
        grip = np.r_[np.full(60, 35.0), np.linspace(35.0, 18.0, 10), np.full(30, 18.0)]
        state = np.tile(reading, (len(grip), 1))
        state[:, 5] = grip

        class Recording:
            def episode(self, episode):
                return state, state

        self.assertEqual(fit.closing_starts(state, 72), 60)
        yaw = float(np.arctan2(outward[1], outward[0]))
        layouts = [{"episode": 0, "seen": [*under, yaw]}, {"episode": 1, "seen": [under[0], under[1] + 0.1, yaw]}]
        moved, by = fit.clear_of_jaws(env, Recording(), {0: (72, 99), 1: (72, 99)}, layouts, "seen")
        self.assertEqual(moved, 1)
        self.assertGreater(by, 1.0)
        self.assertLess(by, 14.5)
        self.assertEqual(layouts[1]["seen_clear"], layouts[1]["seen"])  # well away from the jaws: left alone
        env.set_joints(reading)
        env.set_block(layouts[0]["seen_clear"][:2], yaw)
        mujoco.mj_forward(env._model, env.data)
        self.assertLessEqual(env.arm_in_block(), 0.0007)

    def test_a_block_beside_the_jar_is_not_in_it(self) -> None:
        self.env.reset(seed=0, options={"start": 0, "block": [0.37, -0.09, 0.0]})
        for _ in range(30):
            _, _, terminated, _, info = self.env.step(np.asarray(self.scene["start_pose"]))
        self.assertFalse(terminated)
        self.assertFalse(info["block_in_jar"])

    def test_each_step_moves_a_joint_no_further_than_the_limit(self) -> None:
        self.env.reset(seed=0, options={"start": 0})
        target = np.asarray(self.scene["start_pose"], dtype=np.float64)
        target[0] += 60.0
        before = self.env.joint_state()[0]
        self.env.step(target)
        self.assertLessEqual(self.env.joint_state()[0] - before, self.env.limit + 0.5)
        for _ in range(60):
            self.env.step(target)
        self.assertAlmostEqual(float(self.env.joint_state()[0]), target[0], delta=2.0)

    def test_a_camera_hands_over_old_pictures_and_repeats_them_when_it_is_slow(self) -> None:
        # A camera 2 ticks behind that takes 20 pictures a second, in a loop of 30.
        self.env.reset(seed=0, options={"start": 0})
        self.env.scene["cameras"]["top"] |= {"delay_ticks": 2, "fps": 20}
        self.env.render_camera = lambda name: np.full((2, 2, 3), self.env._steps, np.uint8)  # a picture of the tick it's taken on
        self.env.reset(seed=0, options={"start": 0})
        hold = np.asarray(self.scene["start_pose"])
        handed = [int(self.env.step(hold)[0]["pixels"]["top"][0, 0, 0]) for _ in range(12)]
        # It takes one on ticks 0, 2, 3, 5, 6, 8, 9: none on 1, 4, 7, 10. Each is handed over 2 ticks later.
        self.assertEqual(handed, [0, 0, 0, 2, 3, 3, 5, 6, 6, 8, 9, 9])
        current = [int(self.env.step(hold)[0]["pixels"]["wrist"][0, 0, 0]) for _ in range(3)]
        self.assertEqual(current, [13, 14, 15])  # no delay and 30 a second: this tick's

    def test_the_top_camera_draws_the_block_where_its_lens_puts_it(self) -> None:
        from so_101.sim.env import GROUP_BLOCK

        self.env.reset(seed=0, options={"start": 0})
        shapes = self.env.segment("top", GROUP_BLOCK)
        v, u = np.nonzero(shapes >= 0)
        self.assertGreater(len(u), 300)
        top = self.scene["cameras"]["top"]
        centre = np.array([[0.27, 0.0, self.env.table_height + self.scene["block"]["height"] / 2]])
        expected = Lens.from_scene(top["lens"]).project(to_camera(centre, top["pose"]))[0]
        np.testing.assert_allclose([u.mean(), v.mean()], expected, atol=3.0)
        # The block is drawn over the real background, which is unchanged away from it and the arm.
        picture = self.env.render_camera("top")
        np.testing.assert_array_equal(picture[470, 630], [90, 90, 90])
        self.assertGreater(int(picture[int(v.mean()), int(u.mean()), 0]), int(picture[int(v.mean()), int(u.mean()), 1]) + 40)


class OneDemo:
    """A stand-in for the built demos: one episode that reaches out, closes on a block and holds it, whose
    cameras show one flat colour each."""

    TOP, WRIST = (0, 255, 0), (0, 0, 255)

    def __init__(self, start_pose: list[float]) -> None:
        reach = np.array([5.0, 20.0, 5.0, 65.0, -2.0, 30.0])
        state = np.linspace(np.asarray(start_pose, dtype=np.float64), reach, 30)
        held = np.tile(np.r_[reach[:5], 19.0], (40, 1))
        self.state = np.concatenate([state, held])
        self.action = self.state.copy()
        self.action[30:, 5] = 14.0  # told to close further than the block lets it

    def episode(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        return self.state, self.action

    def grasp_and_release(self, episode: int) -> tuple[int, int]:
        return 33, 70

    def frame(self, episode: int, index: int, camera: str = "top") -> np.ndarray:
        return np.full((480, 640, 3), self.TOP if camera == "top" else self.WRIST, np.uint8)


@unittest.skipIf(SKIP, SKIP)
class PhotoTest(unittest.TestCase):
    """Drawing from the demos' real frames. The stand-in demo's frames are one colour, so a pixel taken
    from a real frame has exactly that colour, and a rendered one doesn't."""

    def setUp(self) -> None:
        from so_101.sim.env import BlockContainerEnv

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.scene = scene_files(Path(folder.name))
        self.demo = OneDemo(self.scene["start_pose"])
        self.env = BlockContainerEnv(scene_dir=folder.name)
        self.addCleanup(self.env.close)
        self.env.reset(seed=0, options={"start": 0})
        self.env.use_photos(self.demo)

    def place(self, frame: int) -> None:
        import mujoco

        self.env.reset(seed=0, options={"start": 0, "joints": self.demo.state[frame]})
        mujoco.mj_forward(self.env._model, self.env.data)

    def test_the_top_cameras_arm_is_drawn_from_the_nearest_real_frame(self) -> None:
        self.place(20)
        picture = self.env.render_camera("top")
        shapes = self.env.segment("top")
        arm_pixels = cv2.erode(((shapes >= 0) & self.env.is_arm[shapes.clip(0)]).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertGreater(arm_pixels.sum(), 5000)
        from_frame = (picture[arm_pixels] == OneDemo.TOP).all(axis=1)
        self.assertGreater(from_frame.mean(), 0.97)
        # Away from the arm, the picture is still the real background.
        np.testing.assert_array_equal(picture[470, 630], [90, 90, 90])

    def test_the_block_and_jar_come_from_the_demos_first_frame_until_the_block_moves(self) -> None:
        self.place(0)
        top = self.scene["cameras"]["top"]
        lens = Lens.from_scene(top["lens"])
        block = lens.project(to_camera(np.array([[0.27, 0.0, 0.013]]), top["pose"]))[0].astype(int)
        jar = lens.project(to_camera(np.array([[0.37, -0.01, 0.03]]), top["pose"]))[0].astype(int)
        picture = self.env.render_camera("top")
        np.testing.assert_array_equal(picture[block[1], block[0]], OneDemo.TOP)
        np.testing.assert_array_equal(picture[jar[1], jar[0]], OneDemo.TOP)
        self.assertTrue(self.env.block_untouched())
        # Once the block has moved it's drawn, red, where it now lies, and the photo of it is gone.
        self.env.set_block([0.2, -0.1], 0.0)
        import mujoco

        mujoco.mj_forward(self.env._model, self.env.data)
        self.assertFalse(self.env.block_untouched())
        picture = self.env.render_camera("top")
        np.testing.assert_array_equal(picture[block[1], block[0]], [90, 90, 90])
        moved = lens.project(to_camera(np.array([[0.2, -0.1, 0.013]]), top["pose"]))[0].astype(int)
        self.assertGreater(int(picture[moved[1], moved[0], 0]), int(picture[moved[1], moved[0], 1]) + 40)
        np.testing.assert_array_equal(picture[jar[1], jar[0]], OneDemo.TOP)

    def test_the_wrist_cameras_view_is_drawn_from_the_nearest_real_frame(self) -> None:
        self.place(20)
        picture = self.env.render_camera("wrist")
        from_frame = (picture == OneDemo.WRIST).all(axis=2)
        self.assertGreater(from_frame.mean(), 0.9)

    def test_without_demos_everything_is_rendered(self) -> None:
        self.env.arm_photos = self.env.wrist_photos = None
        self.place(20)
        for camera, colour in (("top", OneDemo.TOP), ("wrist", OneDemo.WRIST)):
            self.assertFalse((self.env.render_camera(camera) == colour).all(axis=2).any())


if __name__ == "__main__":
    unittest.main()
