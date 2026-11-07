"""Build a task's simulated scene from its demos, and check it. `so101 sim build`, `replay` and `check`."""

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from so_101.sim import env as sim_env
from so_101.sim import fit

GRIPPER_CLOSED = np.radians(-10.0)
# From the setup photos: the camera is on a spring arm near the back wall, to the arm's right, high up,
# looking across the table, with the image upside down. The search only needs to start near this.
FRONT_CAMERA_GUESS = {"position": (0.24, -0.14, 0.30), "target": (0.2, 0.10, 0.0), "rolls": (np.pi, np.pi * 0.75, np.pi * 1.25)}
# What the demos can't show. Override any of these in [sim] in so101.toml.
DEFAULTS = {
    # What let replays of the real demos work best. Measure the real object and set these to be exact.
    "object_diameter": 0.032,
    "object_height": 0.032,
    "grasp_offset": 0.012,  # how far past the jaw tips' midpoint, away from the base, the object's centre is
    "object_mass": 0.02,
    "object_friction": 1.0,  # against the jaws
    "table_friction": 1.0,  # against the table: `so101 sim physics` fits this from real pushes
    "fingertip_clearance": 0.012,  # how far the fingertips stop above the table at a grasp
    "bowl_rim_radius": 0.075,
    "bowl_base_radius": 0.032,
    "bowl_height": 0.07,
}


def build(
    demos_root: Path, out: Path, prompt: str, overrides: dict, teleop_limit: float, device: str,
    refit_cameras: bool = False, refit_motors: bool = False,
) -> dict:
    """Write `scene.json`, the background and the table texture to `out`, and return the scene."""
    out.mkdir(parents=True, exist_ok=True)
    options = {**DEFAULTS, **overrides}
    previous_path = out / "scene.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}

    demos = fit.Demos(demos_root)
    count = len(demos.episodes)
    sessions = fit.split_sessions(demos, count)
    episodes = max(sessions, key=len)
    print(f"{count} demos in {len(sessions)} camera positions {[len(s) for s in sessions]}; using the largest, episodes {episodes[0]}-{episodes[-1]}.", flush=True)
    fit_on, held_back = fit.split(episodes)  # `so101 sim score` checks the fit on demos it didn't see

    kinematics = fit.Kinematics(GRIPPER_CLOSED)
    grasp, release = {}, []
    for episode in range(count):
        state = demos.episode(episode)[0]
        hold, let_go = demos.grasp_and_release(episode)
        grasp[episode] = kinematics.held_object(state[hold], options["grasp_offset"])
        if let_go is not None:
            release.append(kinematics.between_jaws(state[let_go]))
    heights = np.array([grasp[e][2] for e in range(count)])
    points = np.array([grasp[e] for e in range(count) if abs(grasp[e][2] - np.median(heights)) < 0.02])  # a bad reading is far off the table
    design = np.c_[points[:, :2], np.ones(len(points))]
    plane, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    tilt = float(np.degrees(np.arctan(np.linalg.norm(plane[:2]))))
    print(f"Table: {tilt:.1f} degrees from the arm's base plane, grasps fit it within {np.abs(design @ plane - points[:, 2]).mean() * 1000:.1f} mm.", flush=True)
    _, _, spots = cv2.kmeans(points[:, :2].astype(np.float32), min(5, len(points)), None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4), 8, cv2.KMEANS_PP_CENTERS)

    print("Making the background from the demos...", flush=True)
    background = fit.background(demos, episodes)
    cv2.imwrite(str(out / "front_background.png"), background[:, :, ::-1])

    pose = previous.get("cameras", {}).get("front", {}).get("pose")
    if pose is None or refit_cameras:
        print("Fitting the front camera, which takes a few minutes...", flush=True)
        def samples(width, height, every):
            found = []
            for episode in fit_on[::every]:
                state = demos.episode(episode)[0]
                for moment in (0.25, 0.45, 0.7):
                    index = int(moment * len(state))
                    found.append((state[index], fit.arm_mask(demos.image(episode, index / 30), background, width, height)))
            return found
        coarse, coarse_samples = fit.ArmMasks(160, 120, GRIPPER_CLOSED), samples(160, 120, 4)
        best, loss = None, 2.0
        for roll in FRONT_CAMERA_GUESS["rolls"]:
            trial, trial_loss = fit.search_camera(coarse, coarse_samples, position=FRONT_CAMERA_GUESS["position"], target=FRONT_CAMERA_GUESS["target"], roll=roll, tries=500, seed=int(roll * 100))
            if trial_loss < loss:
                best, loss = trial, trial_loss
        pose, loss = fit.fit_camera(fit.ArmMasks(320, 240, GRIPPER_CLOSED), best, samples(320, 240, 2), steps=4)
        pose = pose.tolist()
        print(f"Front camera: {loss:.2f} mismatch with the real arm, {pose[6]:.0f} degree view.", flush=True)

    lens = previous.get("cameras", {}).get("wrist", {}).get("lens")
    if lens is None or refit_cameras:
        print("Starting the wrist camera from the object's position in its frames...", flush=True)
        lens = fit.fit_wrist_lens(demos, kinematics, fit_on, grasp)
        print(f"Wrist camera start: {lens['median_error_px']:.0f} px median error over {lens['points']} sightings.", flush=True)

    let_go = np.median(np.array(release), axis=0)
    scene = {
        "arm_xml": str(fit.arm_xml()),
        "prompt": prompt,
        "start_pose": np.median(np.stack([demos.episode(e)[0][0] for e in range(count)]), axis=0).round(2).tolist(),
        "arm": {
            "rgb": [0.95, 0.88, 0.76], "motor_rgb": [0.08, 0.08, 0.08],
            "front_rgb": [0.95, 0.88, 0.76], "wrist_rgb": [0.75, 0.74, 0.73],
            "motors": previous.get("arm", {}).get("motors") or fit.starting_motors(),
            "delay_ticks": previous.get("arm", {}).get("delay_ticks", 0),
            "gripper_deg_per_percent": fit.GRIPPER_DEG_PER_PERCENT,
        },
        "table": {
            "plane": [float(plane[0]), float(plane[1]), float(plane[2] - options["fingertip_clearance"])],
            "centre": [0.20, 0.0], "half_size": [0.24, 0.45], "texture": "table.png",
        },
        "surroundings": {"wall_rgb": [0.93, 0.86, 0.76], "floor_rgb": [0.72, 0.70, 0.76]},
        "bowl": {
            "xy": let_go[:2].round(3).tolist(), "rim_radius": options["bowl_rim_radius"], "base_radius": options["bowl_base_radius"],
            "height": options["bowl_height"], "rgb": [0.25, 0.62, 0.42],
        },
        "object": {
            "spots": spots.round(4).tolist(), "grasp_offset": options["grasp_offset"], "diameter": options["object_diameter"], "height": options["object_height"], "mass": options["object_mass"],
            "friction": options["object_friction"], "table_friction": overrides.get("table_friction", previous.get("object", {}).get("table_friction", DEFAULTS["table_friction"])), "wrist_rgb": [0.12, 0.27, 0.62], "front_rgb": [0.15, 0.32, 0.42],
        },
        "cameras": {
            "front": {"pose": pose, "background": "front_background.png"},
            "wrist": {"lens": lens},
        },
        "light": {"pos": [-0.25, 0.05, 1.2], "dir": [0.22, -0.04, -1.0], "ambient": 0.5, "direct": 0.5},
        # Fitted to real wrist frames. Shadows only darken the wrist camera's view: the front camera's
        # background is a real photo and comes out worse with more added.
        "shadow": {"dir": [0.3, 0.4, -1.0], "view_m": 1.0, "soft_px": 10, "wrist_strength": 0.45},
        "episodes": episodes,
        "held_back": held_back,
    }

    # The table for the wrist camera: the front camera's real table, recoloured to how the wrist camera
    # sees wood, with everything that isn't wood filled in. Its grain and knots stay where they really are.
    texture_points = fit.table_points(scene, 1024)
    texture = fit.read_through_camera(background, np.asarray(pose), texture_points)
    wood = cv2.erode(fit.is_wood(texture).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    # Bright percentiles, because the wrist camera often looks through the arm's own shadow.
    wrist_wood = np.concatenate([demos.image(e, 0.03, "wrist")[40:300, 200:440].reshape(-1, 3) for e in fit_on[::3]])
    gain = np.percentile(wrist_wood, 70, axis=0) / np.percentile(texture[wood], 70, axis=0)
    filled = texture.copy()
    filled[~wood] = np.median(texture[wood], axis=0)
    filled = cv2.GaussianBlur(filled, (0, 0), 12) * (~wood[..., None]) + texture * wood[..., None]
    recoloured = np.clip(filled * gain, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "table.png"), recoloured[:, :, ::-1])

    previous_path.write_text(json.dumps(scene, indent=1))
    if "outline_loss" not in lens or refit_cameras:
        print("Fitting the wrist camera to the real jaws and object, which takes a few minutes...", flush=True)
        env = sim_env.CylinderBowlEnv(scene_dir=str(out))
        env.reset(seed=0)
        focal = lens["matrix"][1][1]
        start = np.r_[lens["position"], lens["rotation"], np.degrees(2 * np.arctan(240 / focal)), lens["distortion"][0]]
        pose, loss = fit.fit_wrist_outlines(env, demos, kinematics, fit_on, grasp, start)
        env.close()
        lens = {**fit.wrist_lens_from(pose), "outline_loss": loss}
        scene["cameras"]["wrist"]["lens"] = lens
        print(f"Wrist camera: {pose[6]:.0f} degree view, distortion {pose[7]:.2f}, outline mismatch {loss:.2f} (0 is perfect).", flush=True)
        previous_path.write_text(json.dumps(scene, indent=1))
    motors = previous.get("arm", {}).get("motors")
    if motors is None or refit_motors:
        print("Fitting each joint's motor to how the real follower moved...", flush=True)
        scene["arm"]["motors"], scene["arm"]["delay_ticks"] = fit.starting_motors(), 0
        fitted, error = fit.fit_motors(scene, out, demos, list(range(1, count, max(1, count // 6))), teleop_limit)
        scene = fitted
        held_out = fit.tracking_error(scene, out, demos, list(range(2, count, max(1, count // 6))), teleop_limit)
        print("Joint error on demos it wasn't fitted to: " + ", ".join(f"{e:.1f}" for e in held_out), flush=True)
    scene = _match_sizes(scene, out, demos, kinematics, fit_on, measured="object_diameter" in overrides)
    scene = _match_colours(scene, out, demos, kinematics, fit_on)
    scene = _match_softness(scene, out, demos, kinematics, fit_on)
    previous_path.write_text(json.dumps(scene, indent=1))
    return scene


def _place(env: sim_env.CylinderBowlEnv, kinematics: fit.Kinematics, state: np.ndarray, point: np.ndarray) -> None:
    env.data.qpos[env.qadr] = env.to_ctrl(state)
    env.set_object(((point - env.origin) @ env.frame)[:2])
    mujoco.mj_forward(env._model, env.data)


def _match_sizes(scene, out, demos, kinematics, episodes, measured: bool) -> dict:
    """Draw the object as big as it really looks in each camera. Doesn't change what the jaws grip.

    A measured size needs no adjusting. If it still looks wrong in `so101 sim score`, a camera is off.
    """
    if measured:
        scene["object"]["front_scale"] = scene["object"]["wrist_scale"] = 1.0
        return scene
    # The real object's front view is too dark to measure reliably, so that size is judged by eye.
    scene["object"]["front_scale"] = 0.9
    for _ in range(3):
        (out / "scene.json").write_text(json.dumps(scene, indent=1))
        env = sim_env.CylinderBowlEnv(scene_dir=str(out))
        env.reset(seed=0)
        ratios = {"front": [], "wrist": []}
        for episode in episodes[::2]:
            state = demos.episode(episode)[0]
            hold = demos.grasp_and_release(episode)[0]
            for moment in (0.0, 0.55):
                index = int(hold * moment)
                _place(env, kinematics, state[index], kinematics.held_object(state[hold], scene["object"]["grasp_offset"]))
                for camera in ("wrist",):
                    real, drawn = fit.blue_blob(demos.image(episode, index / 30, camera)), fit.blue_blob(env.render_camera(camera))
                    uncut = lambda box: box[0] > 2 and box[1] > 2 and box[0] + box[2] < 638 and box[1] + box[3] < 478
                    if real and drawn and uncut(real[2]) and uncut(drawn[2]):
                        # By its outline: the real object's dark side doesn't all count as blue, so area reads small.
                        ratios[camera].append(max(real[2][2:]) / max(drawn[2][2:]))
        env.close()
        for camera, found in ratios.items():
            if found:
                scene["object"][f"{camera}_scale"] = round(float(np.clip(scene["object"].get(f"{camera}_scale", 1.0) * np.median(found), 0.4, 1.5)), 3)
    print(f"Object drawn at {scene['object']['front_scale']:.2f}x in the front camera and {scene['object']['wrist_scale']:.2f}x in the wrist camera. Measuring the real one will show which is right.", flush=True)
    return scene


def _bright_blue(image: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    patch = image[y : y + h, x : x + w].reshape(-1, 3)
    return patch[patch[:, 2] > patch[:, 0] + 12]


def _match_colours(scene, out, demos, kinematics, episodes) -> dict:
    """Adjust colours until the render matches real frames: the object, arm and table, per camera."""
    background = cv2.cvtColor(cv2.imread(str(out / "front_background.png")), cv2.COLOR_BGR2RGB)
    table_file = out / scene["table"]["texture"]
    for _ in range(4):
        (out / "scene.json").write_text(json.dumps(scene, indent=1))
        env = sim_env.CylinderBowlEnv(scene_dir=str(out))
        env.reset(seed=0)
        gains = {"front_object": [], "wrist_object": [], "front_arm": [], "wrist_arm": [], "wood": []}
        for episode in episodes[::2]:
            state = demos.episode(episode)[0]
            hold = demos.grasp_and_release(episode)[0]
            _place(env, kinematics, state[0], kinematics.held_object(state[hold], scene["object"]["grasp_offset"]))
            for camera in ("front", "wrist"):
                real_image, drawn_image = demos.image(episode, 0.03, camera), env.render_camera(camera)
                real, drawn = fit.blue_blob(real_image, 150), fit.blue_blob(drawn_image, 150)
                if real and drawn:
                    a, b = _bright_blue(real_image, real[2]), _bright_blue(drawn_image, drawn[2])
                    if len(a) and len(b):  # the top of the object: its brightest part
                        gains[f"{camera}_object"].append(np.percentile(a, 75, axis=0) / np.maximum(np.percentile(b, 75, axis=0), 1))
            real_wrist, drawn_wrist = demos.image(episode, 0.03, "wrist"), env.render_camera("wrist")
            real_jaws, drawn_jaws = fit.wrist_masks(real_wrist)[1], fit.wrist_masks(drawn_wrist)[1]
            if real_jaws.sum() > 500 and drawn_jaws.sum() > 500:
                gains["wrist_arm"].append(np.median(real_wrist[real_jaws], axis=0) / np.maximum(np.median(drawn_wrist[drawn_jaws], axis=0), 1))
            for image, store in ((real_wrist, "real"), (drawn_wrist, "drawn")):
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                wood = (hsv[..., 0] < 25) & (hsv[..., 1] > 40) & (hsv[..., 2] > 60)
                wood[image.shape[0] * 3 // 5 :] = False
                if store == "real":
                    real_wood = np.percentile(image[wood], 70, axis=0) if wood.sum() > 500 else None
                elif real_wood is not None and wood.sum() > 500:
                    gains["wood"].append(real_wood / np.maximum(np.percentile(image[wood], 70, axis=0), 1))
            index = int(hold * 0.6)
            _place(env, kinematics, state[index], kinematics.held_object(state[hold], scene["object"]["grasp_offset"]))
            real_image, drawn_image = demos.image(episode, index / 30), env.render_camera("front")
            real_mask = fit.arm_mask(real_image, background, 640, 480) > 0
            drawn_mask = np.abs(drawn_image.astype(np.int16) - background.astype(np.int16)).max(axis=2) > 8
            both = real_mask & drawn_mask & (real_image.min(axis=2) > 90) & (drawn_image.min(axis=2) > 90)
            if both.sum() > 200:
                gains["front_arm"].append(np.median(real_image[both], axis=0) / np.maximum(np.median(drawn_image[both], axis=0), 1))
        env.close()
        for camera in ("front", "wrist"):
            if gains[f"{camera}_object"]:
                scene["object"][f"{camera}_rgb"] = (np.asarray(scene["object"][f"{camera}_rgb"]) * np.median(gains[f"{camera}_object"], axis=0)).clip(0, 1).round(3).tolist()
            if gains[f"{camera}_arm"]:
                scene["arm"][f"{camera}_rgb"] = (np.asarray(scene["arm"][f"{camera}_rgb"]) * np.median(gains[f"{camera}_arm"], axis=0)).clip(0, 1).round(3).tolist()
        if gains["wood"]:
            texture = cv2.imread(str(table_file)).astype(np.float32)
            texture *= np.median(gains["wood"], axis=0)[::-1]  # the file is BGR
            cv2.imwrite(str(table_file), texture.clip(0, 255).astype(np.uint8))
    print(f"Colours from real frames: wrist object {scene['object']['wrist_rgb']}, wrist arm {scene['arm']['wrist_rgb']}.", flush=True)
    return scene


def _sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """Detail around the masked edges, or NaN if there's too little to measure."""
    if mask.sum() < 200:
        return float("nan")
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_32F)[mask].var())


def _match_softness(scene, out, demos, kinematics, episodes) -> dict:
    """Blur each camera's rendered parts as much as the real camera's picture is blurred. The front camera is
    judged around the arm, its only rendered part. The wrist camera is judged over its whole picture: its
    rendered jaws don't line up with the real ones closely enough to compare just their edges."""
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    env = sim_env.CylinderBowlEnv(scene_dir=str(out))
    env.reset(seed=0)
    grow = np.ones((9, 9), np.uint8)
    moments = []
    for episode in episodes[::2]:
        state = demos.episode(episode)[0]
        hold, let_go = demos.grasp_and_release(episode)
        moments += [(episode, state, int(moment * hold), hold, let_go) for moment in (0.2, 0.6)]
    real = {"front": [], "wrist": []}
    for episode, _, frame, _, _ in moments:
        front = demos.image(episode, frame / 30)
        real["front"].append(_sharpness(front, cv2.dilate(fit.arm_mask(front, env.background, 640, 480), grow) > 0))
        wrist = demos.image(episode, frame / 30, "wrist")
        real["wrist"].append(_sharpness(wrist, np.ones(wrist.shape[:2], bool)))
    softness = {}
    for camera in fit.CAMERAS:
        trials = {}
        for sigma in (0.0, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0):
            env.scene["softness"] = {camera: sigma}
            found = []
            for episode, state, frame, hold, let_go in moments:
                fit.place_for_frame(env, kinematics, state, frame, hold, let_go, scene["object"]["grasp_offset"])
                ids = env.segment(camera)
                region = cv2.dilate((ids >= 0).astype(np.uint8), grow) > 0 if camera == "front" else np.ones(ids.shape, bool)
                found.append(_sharpness(env.render_camera(camera), region))
            trials[sigma] = abs(np.nanmedian(found) - np.nanmedian(real[camera]))
        softness[camera] = min(trials, key=trials.get)
    env.close()
    scene["softness"] = softness
    print(f"Softness from real frames: front {softness['front']}, wrist {softness['wrist']} pixels.", flush=True)
    return scene


def replay(demos_root: Path, scene_dir: Path, episodes: list[int] | None, limit: float, video: Path | None) -> dict:
    """Replay real demos' commands in the sim. A sim that can't redo them can't judge a policy."""
    demos = fit.Demos(demos_root)
    kinematics = fit.Kinematics(GRIPPER_CLOSED)
    env = sim_env.CylinderBowlEnv(scene_dir=str(scene_dir), max_relative_target=limit)
    env.reset(seed=0)
    chosen = episodes if episodes is not None else list(range(len(demos.episodes)))
    results = []
    writer = None
    for index, episode in enumerate(chosen):
        state, action = demos.episode(episode)
        hold = demos.grasp_and_release(episode)[0]
        env.reset(seed=0)
        _place(env, kinematics, state[0], kinematics.held_object(state[hold], env.scene["object"]["grasp_offset"]))
        env._pending.clear()
        env.data.ctrl[env.actuators] = env.to_ctrl(state[0])
        track, height = [], []
        for frame, command in enumerate(action):
            env.apply(command)
            track.append(env.joint_state().copy())
            height.append(env.object_on_table()[2])
            if video is not None and index == 0 and frame % 2 == 0:
                sim_front, sim_wrist = env.render_camera("front"), env.render_camera("wrist")
                real_front, real_wrist = demos.image(episode, frame / 30), demos.image(episode, frame / 30, "wrist")
                row = np.vstack([np.hstack([real_front, sim_front]), np.hstack([real_wrist, sim_wrist])])
                row = cv2.resize(row, (960, 720))
                if writer is None:
                    video.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (960, 720))
                writer.write(row[:, :, ::-1])
        for _ in range(45):  # let it settle
            env.apply(action[-1])
        track = np.asarray(track)
        results.append({
            "episode": episode,
            "error": np.abs(track - state).mean(axis=0),
            "lifted": float(max(height) - height[0]) > 0.03,
            "in_bowl": env.object_in_bowl(),
        })
    if writer is not None:
        writer.release()
    env.close()
    return {
        "results": results,
        "lifted": sum(r["lifted"] for r in results),
        "in_bowl": sum(r["in_bowl"] for r in results),
        "error": np.mean([r["error"] for r in results], axis=0),
    }


def check(demos_root: Path, scene_dir: Path, policy_dir: Path, device: str, frames: int = 80) -> dict:
    """Does the model act the same on rendered images as on real ones?

    The model predicts actions from the same moment in real demos twice: once from the real camera images,
    once from sim images with the arm and object in the same place. If sim images change its predictions
    much more than the real ones miss the demo, the sim can't judge this model.
    """
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(policy_dir)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(policy_dir, config=config).to(device).eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=policy_dir, preprocessor_overrides={"device_processor": {"device": device}}
    )
    demos = fit.Demos(demos_root)
    kinematics = fit.Kinematics(GRIPPER_CLOSED)
    env = sim_env.CylinderBowlEnv(scene_dir=str(scene_dir))
    env.reset(seed=0)
    prompt = env.scene["prompt"]
    episodes = env.scene.get("held_back") or env.scene["episodes"]
    rng = np.random.default_rng(0)

    def predict(front: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        def image(pixels):
            return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()[None].contiguous() / 255

        batch = {
            "observation.images.front": image(front),
            "observation.images.wrist": image(wrist),
            "observation.state": torch.from_numpy(state.copy()).float()[None],
            "task": [prompt],
        }
        with torch.inference_mode():
            return post(policy.predict_action_chunk(pre(batch)))[0].cpu().float().numpy()

    rows = []
    for _ in range(frames):
        episode = int(rng.choice(episodes))
        state, action = demos.episode(episode)
        hold = demos.grasp_and_release(episode)[0]
        let_go = demos.grasp_and_release(episode)[1]
        index = int(rng.integers(0, len(state) - 1))  # any moment of the demo
        fit.place_for_frame(env, kinematics, state, index, hold, let_go, env.scene["object"]["grasp_offset"])
        chunk_len = min(50, len(action) - index)
        real = predict(demos.image(episode, index / 30), demos.image(episode, index / 30, "wrist"), state[index])
        sim_front_only = predict(env.render_camera("front"), demos.image(episode, index / 30, "wrist"), state[index])
        sim_wrist_only = predict(demos.image(episode, index / 30), env.render_camera("wrist"), state[index])
        sim = predict(env.render_camera("front"), env.render_camera("wrist"), state[index])
        rows.append((
            np.abs(real[:chunk_len] - action[index : index + chunk_len]).mean(axis=0),
            np.abs(sim[:chunk_len] - action[index : index + chunk_len]).mean(axis=0),
            np.abs(sim_front_only[:chunk_len] - action[index : index + chunk_len]).mean(axis=0),
            np.abs(sim_wrist_only[:chunk_len] - action[index : index + chunk_len]).mean(axis=0),
            np.abs(sim[:chunk_len] - real[:chunk_len]).mean(axis=0),
        ))
    env.close()
    rows = np.asarray(rows).mean(axis=0)
    return {"real": rows[0], "sim": rows[1], "sim_front": rows[2], "sim_wrist": rows[3], "sim_vs_real": rows[4]}
