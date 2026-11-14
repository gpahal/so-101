"""Build a task's simulated scene from its demos, and check it. `so101 sim build`, `replay` and `check`."""

import json
from pathlib import Path

import cv2
import numpy as np

from so_101.sim import arm, fit
from so_101.sim import env as sim_env
from so_101.sim.camera import Lens

# What the demos can't show. Override any of these in [sim] in so101.toml.
DEFAULTS = {
    # A LEGO DUPLO 2x2 brick: 31.8 mm square and 19.2 mm tall, with four hollow studs 9.4 mm across and
    # 4.5 mm tall, 16 mm apart. Its weight is a typical one's.
    "block_width": 0.0318,
    "block_height": 0.0192,
    "stud_diameter": 0.0094,
    "stud_height": 0.0045,
    "stud_pitch": 0.016,
    "block_mass": 0.0055,
    "block_friction": 0.3,  # plastic on plastic; against the jaws and the mat, theirs counts if it's higher
    # The printed jaws against the block. Printed plastic on a smooth brick is around 0.3 to 0.5, and the
    # arm model's own 1.0 kept hold of blocks the real jaws let go. Replaying the first 100 demos on two
    # builds put 79 and 82 of 96 blocks in the jar at 1.0, 85 at 0.6, 84 at 0.5, 87 and 90 at 0.45, 83 and 89
    # at 0.4, 88 and 89 at 0.35, and 86 and 87 at 0.3.
    "jaw_friction": 0.35,
    "table_friction": 0.6,  # the mat, against the block and the jar: `so101 sim physics` fits this from real pushes
    # The mat's top above the surface the arm's base stands on: a thin mat on the same table. The joint
    # offsets are fitted so that every demo's grasp lands on it. Set to "fit", it's fitted with the top
    # camera instead, from how big the block looks; on these demos that fit matched the arm's outline
    # worse (6.0 px against 3.8), so it isn't the default.
    "mat_height": 0.003,
    # The jar. Its diameter and tinted band are fitted to the top camera's frames unless jar_diameter is set.
    "jar_diameter": None,
    "jar_height": None,
    "jar_tint_height": None,
    "jar_rim_height": 0.013,  # the clear neck above the tinted band, when jar_height isn't set
    "jar_wall": 0.002,
    "jar_base": 0.003,
    "jar_mass": 0.03,
    "jar_friction": 0.4,
    # The strip of bare table left of the mat, which the wrist camera sees past the mat's edge. The top
    # camera can't measure it: from there the bare table mirrors the wall.
    "table_past_mat": 0.025,
    "floor_below": 0.74,  # the floor below the table top: a desk's height
}
# Where a demo's block can start, by the key its layout keeps each place under, best guess first on a tie.
BLOCK_PLACES = ("seen_clear", "by_wrist_clear", "seen", "by_wrist")
BLOCK_PLACE_NAMES = {
    "seen": "the top camera's places", "by_wrist": "the wrist camera's", "seen_clear": "the top camera's moved clear of the jaws",
    "by_wrist_clear": "the wrist camera's moved clear",
}  # fmt: skip
TABLE = {"x": [-0.12, 0.42], "y": [-0.45, 0.1], "texel": 0.001}  # replaced by the edges the top camera sees


def joint_limits(calibration_file: Path) -> dict[str, list[float]]:
    """Each joint's calibrated range in the follower's units: degrees either side of its middle for the
    arm, and 0 to 100 for the gripper. LeRobot keeps every goal within it."""
    calibration = json.loads(calibration_file.read_text())
    limits = {}
    for name in arm.ARM_JOINTS:
        half = (calibration[name]["range_max"] - calibration[name]["range_min"]) / 2 * 360 / 4095
        limits[name] = [round(-half, 2), round(half, 2)]
    limits["gripper"] = [0.0, 100.0]
    return limits


def gripper_units(calibration_file: Path) -> float:
    """Degrees of jaw per 1% of the follower's gripper reading: its calibrated range over 100."""
    gripper = json.loads(calibration_file.read_text())["gripper"]
    return (gripper["range_max"] - gripper["range_min"]) * 360 / 4095 / 100


def build(
    demos_root: Path, out: Path, prompt: str, overrides: dict, teleop_limit: float, repo: str, calibration_file: Path,
    refit_cameras: bool = False, refit_motors: bool = False, report=print,
) -> dict:  # fmt: skip
    """Write `scene.json`, the top camera's background and jar and the table texture to `out`, and return the scene.

    Fits already in `out` are kept, so rebuilding from the same demos gives the same files. `repo` is where
    the demos are on the Hub, recorded with their revision so the sim can be rebuilt elsewhere.
    """
    from so_101.sim import source

    if not calibration_file.is_file():
        raise ValueError(f"No follower calibration at {calibration_file}: the sim takes each joint's range from it.")
    out.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(overrides) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"[sim] in so101.toml has settings the sim doesn't know: {', '.join(unknown)}. It takes: {', '.join(DEFAULTS)}.")
    options = {**DEFAULTS, **overrides}
    scene_path = out / "scene.json"
    previous = json.loads(scene_path.read_text()) if scene_path.is_file() else {}
    kept = previous.get("fits", {})

    demos = fit.Demos(demos_root)
    held = fit.held_demos(demos)
    episodes = sorted(held)
    fit_on, held_back = fit.split(episodes)  # `so101 sim score` checks the fits on demos they didn't see
    report(f"{len(demos)} demos, {len(episodes)} of them hold the block: fitting on {len(fit_on)} and holding {len(held_back)} back.")
    start_pose = np.median(np.stack([demos.episode(e)[0][0] for e in episodes]), axis=0)
    deg_per_percent = gripper_units(calibration_file)
    # The follower's gripper reads its resting value against its end stop, which is the model's lower limit.
    gripper_closed = float(np.degrees(arm.arm_model().joint("gripper").range[0]) - start_pose[5] * deg_per_percent)
    block = {
        "width": options["block_width"], "height": options["block_height"], "mass": options["block_mass"], "friction": options["block_friction"],
        "stud": {"radius": options["stud_diameter"] / 2, "height": options["stud_height"], "pitch": options["stud_pitch"]},
        "rgb": [0.75, 0.1, 0.1], "hole_rgb": [0.45, 0.05, 0.05],
    }  # fmt: skip

    if options["mat_height"] != "fit" and not isinstance(options["mat_height"], (int, float)):
        raise ValueError(f'mat_height in [sim] in so101.toml is {options["mat_height"]!r}: give it in metres, or "fit".')
    # In the arm model's axes: its base's underside, which stands on the table, is 2.4 mm below its z = 0.
    mat_height = None if options["mat_height"] == "fit" else round(float(options["mat_height"]) + arm.base_underside(), 4)
    top = kept.get("top")
    if top is None or refit_cameras or (mat_height is not None and top.get("table_height") != mat_height):
        report("Fitting the top camera, the arm's joint offsets and the mat's height, which takes a few minutes...")
        unset = arm.Units((0.0,) * 5, gripper_closed, deg_per_percent)
        top = fit.fit_top_camera(fit.top_fit_data(demos, fit_on, held, block, unset), gripper_closed, deg_per_percent, block, mat_height, joint_limits(calibration_file), report)
        kept.pop("wrist", None)
    table_height = top["table_height"]
    errors = top["errors"]
    report(
        f"Top camera: the drawn arm's outline is {errors['outline_px']:.1f} px from the real one, and the block appears "
        f"{errors['block_px']:.1f} px from where the jaws closed on it. The mat's top is {(table_height - arm.base_underside()) * 1000:+.0f} mm from the "
        f"surface the arm's base stands on{'' if mat_height is not None else ' (fitted)'}. Joint offsets (degrees): "
        + ", ".join(f"{name} {offset:+.1f}" for name, offset in zip(arm.ARM_JOINTS, top["offsets_deg"]))
    )
    units = arm.Units(tuple(top["offsets_deg"]), gripper_closed, deg_per_percent)
    top_lens = Lens.from_scene(top["lens"])

    # Where each demo's jar stood, and the jar's size.
    grasps = {e: fit.held_block(units, demos.episode(e)[0][held[e][0]][None])[0] for e in episodes}
    releases = np.array([fit.held_block(units, demos.episode(e)[0][held[e][1]][None])[0] for e in episodes])
    measured = options["jar_diameter"] is not None
    jar = {"radius": (options["jar_diameter"] or 0.07) / 2, "height": options["jar_tint_height"] or 0.05}
    jar, jar_places = fit.fit_jars(demos, episodes, top["pose"], top_lens, jar, table_height, np.median(releases[:, :2], axis=0), measured)
    # The amber the top camera sees ends at the tinted band's top, so that's the height the fit found.
    tint_height = options["jar_tint_height"] or jar["height"]
    jar = {
        "home": None, "radius": round(jar["radius"], 4), "height": round(options["jar_height"] or tint_height + options["jar_rim_height"], 4),
        "tint_height": round(tint_height, 4), "wall": options["jar_wall"], "base": options["jar_base"], "mass": options["jar_mass"],
        "friction": options["jar_friction"], "outline_mismatch": jar["outline_mismatch"],
        "tint_rgba": [0.62, 0.47, 0.25, 0.985], "clear_rgba": [0.8, 0.8, 0.8, 0.3],
        # How much of what's behind the jar's wall the top camera's real photo of the jar hides.
        "tint_see_through": 0.12, "clear_see_through": 0.6,
    }  # fmt: skip
    spread = np.array(list(jar_places.values())).std(axis=0) * 1000
    report(f"Jar: {jar['radius'] * 2000:.0f} mm across, tinted to {jar['tint_height'] * 1000:.0f} mm, and it stood within {spread[0]:.0f} x {spread[1]:.0f} mm (1 SD) over the demos.")

    report("Making the top camera's empty background and the jar's photo...")
    whole_jar = {e: fit.jar_outline(top["pose"], top_lens, jar_places[e], jar, table_height) for e in episodes}
    view = fit.ArmView(top["pose"], top_lens, units)
    plate = fit.clean_plate(demos, fit_on[:: max(1, len(fit_on) // 40)], view, whole_jar)
    view.close()
    cv2.imwrite(str(out / "top_background.png"), plate[:, :, ::-1])
    cutout, home = fit.jar_cutout(demos, episodes, jar_places, top["pose"], top_lens, jar, table_height)
    cv2.imwrite(str(out / "top_jar.png"), cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))
    jar["home"] = home

    edges = fit.table_edges(plate, top["pose"], top_lens, table_height)
    table = {
        "x": [TABLE["x"][0], edges["wall_x"]], "y": [TABLE["y"][0], round(edges["left_y"] + options["table_past_mat"], 4)], "texel": TABLE["texel"],
        "height": table_height, "texture": "table.png", "friction": overrides.get("table_friction", previous.get("table", {}).get("friction", DEFAULTS["table_friction"])),
        "friction_from": None if "table_friction" in overrides else previous.get("table", {}).get("friction_from"),
    }  # fmt: skip
    report(
        f"Mat: the wall is {edges['wall_x'] * 100:.1f} cm in front of the arm's base, and the mat's left edge "
        f"{edges['left_y'] * 100:.1f} cm to its left."
    )
    flat = fit.wood_filled(fit.look_flat(plate, top["pose"], top_lens, table))
    cv2.imwrite(str(out / "table_top.png"), flat[:, :, ::-1])
    cv2.imwrite(str(out / "table.png"), flat[:, :, ::-1])
    wall_rgb = (np.median(plate[20:60, 420:620].reshape(-1, 3), axis=0) / 255).round(3).tolist()

    colours = {
        camera: {
            "arm": [0.72, 0.72, 0.73], "motor": [0.08, 0.08, 0.08], "block": [0.75, 0.1, 0.1], "block_hole": [0.45, 0.05, 0.05],
            "jar_tint": jar["tint_rgba"], "jar_clear": jar["clear_rgba"], "wall": wall_rgb, "floor": [0.25, 0.25, 0.28],
        }
        for camera in sim_env.CAMERAS
    }  # fmt: skip
    wrist_pose, wrist_lens = fit.wrist_start()
    # The wrist camera is fitted to the jar's outline among others, so a jar of another size needs it again.
    same_jar = all(previous.get("jar", {}).get(key) == jar[key] for key in ("radius", "height", "tint_height"))
    wrist = (None if refit_cameras or not same_jar else kept.get("wrist")) or {"pose": wrist_pose.tolist(), "lens": wrist_lens}
    scene = {
        "prompt": prompt,
        "demos": source.describe(demos_root, repo),
        "start_pose": start_pose.round(2).tolist(),
        "arm": {
            "offsets_deg": top["offsets_deg"], "gripper_closed_deg": round(gripper_closed, 3), "gripper_deg_per_percent": round(deg_per_percent, 4),
            "limits": joint_limits(calibration_file),
            "motors": previous.get("arm", {}).get("motors") or fit.starting_motors(),
            "delay_ticks": previous.get("arm", {}).get("delay_ticks", 0), "jaw_friction": options["jaw_friction"],
        },
        "table": table,
        "wall": {"x": edges["wall_x"], "rgb": wall_rgb},
        "floor": {"height": round(table_height - options["floor_below"], 3), "rgb": [0.25, 0.25, 0.28]},
        "block": block,
        "jar": jar,
        "cameras": {
            "top": {"pose": top["pose"], "lens": top["lens"], "plate": "top_background.png", "jar": {"file": "top_jar.png", "xy": home}},
            "wrist": {"pose": wrist["pose"], "lens": wrist["lens"]},
        },
        "light": {"dir": [0.15, 0.1, -1.0], "ambient": 0.6, "direct": 0.5},
        "colours": colours,
        "softness": previous.get("softness", {"top": 0.8, "wrist": 0.8}),
        "starts": [{"episode": e, "block": [*np.round(grasps[e][:2], 4).tolist(), round(fit.jaw_yaw(units, demos.episode(e)[0][held[e][0]]), 4)], "jar": jar_places[e], "grasp": np.round(grasps[e][:2], 4).tolist()} for e in episodes],
        "episodes": fit_on,
        "held_back": held_back,
        "fits": {"top": top},
    }  # fmt: skip
    fit.save_json(scene_path, scene)

    report("Finding where each demo's block lay from the top camera's first frame...")
    env = sim_env.BlockContainerEnv(scene_dir=str(out))
    env.reset(seed=0)
    seen, apart = [], []
    for layout in scene["starts"]:
        episode = layout["episode"]
        first = demos.frame(episode, 0)
        blob = fit.red_blob(first, 40)
        if blob is None:  # hidden in the first frame: keep it where the jaws closed on it
            continue
        guess = fit.on_plane(top["pose"], top_lens, blob[0], table_height + options["block_height"] * 0.6)[0, :2]
        state = demos.episode(episode)[0]
        layout["block"], cover = fit.block_yaw(env, first, guess, state[0], fit.jaw_yaw(units, state[held[episode][0]]))
        seen.append(cover)
        apart.append(np.linalg.norm(np.array(layout["block"][:2]) - layout["grasp"]) * 1000)
    env.close()
    scene["fits"]["block"] = {"cover": round(float(np.mean(seen)), 3), "from_grasp_mm": round(float(np.median(apart)), 1)}
    report(f"Blocks: the drawn block covers {np.mean(seen):.0%} of the real one, and lay {np.median(apart):.1f} mm (median) from where the jaws later closed.")
    fit.save_json(scene_path, scene)

    if "errors" not in wrist or refit_cameras:
        report("Fitting the wrist camera to the real jaws and block...")
        starts = {layout["episode"]: layout for layout in scene["starts"]}
        wrist = fit.fit_wrist_camera(fit.wrist_fit_data(demos, held, fit_on[::2], units, starts, scene), report)
    report("Wrist camera: its drawn outlines are " + ", ".join(f"{px:.1f} px from the real {name[:-3]}" for name, px in wrist["errors"].items()) + ".")
    scene["cameras"]["wrist"] = {"pose": wrist["pose"], "lens": wrist["lens"]}
    scene["fits"]["wrist"] = wrist
    fit.save_json(scene_path, scene)

    env = sim_env.BlockContainerEnv(scene_dir=str(out))
    env.reset(seed=0)
    moved, left = fit.blocks_by_wrist(env, demos, held, scene["starts"])
    cleared = {place: fit.clear_of_jaws(env, demos, held, scene["starts"], place) for place in ("seen", "by_wrist")}
    env.close()
    scene["fits"]["block"] |= {"wrist_move_mm": round(moved, 1), "wrist_px": round(left, 1), "moved_clear": {place: [count, round(by, 1)] for place, (count, by) in cleared.items()}}
    fit.save_json(scene_path, scene)

    timing = kept.get("timing")
    if timing is None or refit_cameras:
        report("Measuring how far behind the joints each camera's picture is...")
        timing = fit.camera_timing(demos, held, fit_on[::4], scene)
    for camera in sim_env.CAMERAS:
        scene["cameras"][camera] |= {"delay_ticks": timing[camera]["delay_ticks"], "fps": timing[camera]["fps"]}
    scene["fits"]["timing"] = timing
    report("Cameras: " + ", ".join(f"the {camera} one's picture is {timing[camera]['delay_ticks']} ticks old and it gives {timing[camera]['fps']:g} a second" for camera in sim_env.CAMERAS) + ".")
    fit.save_json(scene_path, scene)

    if not previous.get("arm", {}).get("motors") or refit_motors:
        report("Fitting each joint's motor to how the real follower moved, which takes about 15 minutes...")
        scene["arm"]["motors"], scene["arm"]["delay_ticks"] = fit.starting_motors(), 0
        scene = fit.fit_motors(scene, out, demos, held, fit_on[:: max(1, len(fit_on) // 5)], teleop_limit, report=report)
        scene = _match_grip(scene, out, demos, held, fit_on[:: max(1, len(fit_on) // 8)], teleop_limit, report)
    fit.save_json(scene_path, scene)
    env = sim_env.BlockContainerEnv(scene_dir=str(out), max_relative_target=teleop_limit)
    env.reset(seed=0)
    error = fit.tracking_error(env, demos, held, held_back[:: max(1, len(held_back) // 6)])
    # Where does each demo's block start? The top camera and the wrist camera place it a few millimetres
    # apart, and either place can lie under where a real jaw came down. The arm has to grasp it, so it's
    # whichever of the four lets most of the real demos' commands put it in the jar; a tie goes to a
    # place clear of the jaws, since the real block was.
    fitted = [layout for layout in scene["starts"] if layout["episode"] in fit_on]
    into_jar = {place: fit.replays_into_jar(env, demos, fitted, place) for place in BLOCK_PLACES}
    env.close()
    chosen = max(BLOCK_PLACES, key=into_jar.get)
    for layout in scene["starts"]:
        layout["block"] = layout.get(chosen, layout["seen"])
    scene["fits"]["block"] |= {"placed": chosen, "replays_into_jar": into_jar}
    seen_clear, wrist_clear = scene["fits"]["block"]["moved_clear"]["seen"], scene["fits"]["block"]["moved_clear"]["by_wrist"]
    report(
        f"Blocks: the wrist camera saw them {scene['fits']['block']['wrist_move_mm']:.1f} mm (median) from where the top camera did. "
        f"{seen_clear[0]} of the top camera's places and {wrist_clear[0]} of the wrist camera's lay under where a real jaw came down, and "
        f"were moved clear by {seen_clear[1]:.0f} and {wrist_clear[1]:.0f} mm (median). Replaying {len(fit_on)} demos puts "
        + ", ".join(f"{into_jar[place]} in the jar from {BLOCK_PLACE_NAMES[place]}" for place in BLOCK_PLACES)
        + f", so each block starts at {BLOCK_PLACE_NAMES[chosen]}."
    )
    scene["fits"]["joint_error"] = [round(float(e), 2) for e in error]
    report("Joint error on demos the motors weren't fitted to: " + ", ".join(f"{e:.1f}" for e in error))
    scene = _match_colours(scene, out, demos, held, fit_on, report)
    scene = _match_softness(scene, out, demos, held, fit_on, report)
    fit.save_json(scene_path, scene)
    return scene


def _match_grip(scene: dict, out: Path, demos: fit.Demos, held: dict, episodes: list[int], limit: float, report) -> dict:
    """Shift the gripper's zero until the simulated jaws read what the real ones did while holding the block.

    The jaws stop on the block, so what they read then says how the reading maps to their opening, more
    closely than the end stop does.
    """
    layouts = {layout["episode"]: layout for layout in scene["starts"]}
    for _ in range(2):
        fit.save_json(out / "scene.json", scene)
        env = sim_env.BlockContainerEnv(scene_dir=str(out), max_relative_target=limit)
        gaps = []
        for episode in episodes:
            hold, let_go = held[episode]
            track = fit.replay_arm(env, demos, episode, layouts[episode], block=True)
            real = demos.episode(episode)[0]
            middle = slice(hold + 10, max(let_go - 10, hold + 11))
            if track[middle, 5].min() > 8:  # it held the block
                gaps.append(float(np.median(track[middle, 5]) - np.median(real[middle, 5])))
        env.close()
        if not gaps:
            report("  no replayed demo held the block, so the gripper's zero stays at its end stop.")
            return scene
        shift = float(np.median(gaps))
        scene["arm"]["gripper_closed_deg"] = round(scene["arm"]["gripper_closed_deg"] + shift * scene["arm"]["gripper_deg_per_percent"], 3)
    report(f"Gripper: holding the block reads within {abs(shift):.1f} of the real {np.median([np.median(demos.episode(e)[0][held[e][0] + 10 : held[e][1] - 10, 5]) for e in episodes]):.1f}.")
    return scene


def _regions(env: sim_env.BlockContainerEnv, camera: str, shapes: np.ndarray) -> dict[str, np.ndarray]:
    """Which pixels of a camera show each thing whose colour is matched, a little in from its edges."""
    model = env._model
    seen = shapes.clip(0)
    printed = np.isin(model.geom_matid, env.printed_materials)
    regions = {"arm": env.is_arm[seen] & printed[seen], "block": env.is_block[seen]}
    if camera == "wrist":
        regions |= {name: seen == model.geom(name).id for name in ("wall", "floor", "table")}
        regions["jar_tint"] = seen == env.jar_tint_geom
    shrink = np.ones((7, 7), np.uint8)
    return {name: cv2.erode(((shapes >= 0) & mask).astype(np.uint8), shrink) > 0 for name, mask in regions.items()}


def _real_look(picture: np.ndarray) -> dict[str, np.ndarray]:
    """Which pixels of a real frame could be each thing, by colour. The drawn and real things never line
    up exactly, and a region's colour is matched only where the real frame agrees it's that thing."""
    a = picture.astype(np.int16)
    hsv = cv2.cvtColor(picture, cv2.COLOR_RGB2HSV)
    dark = a.max(axis=2) < 70
    return {
        "block": fit.red_mask(picture),
        "arm": fit.pale(picture),
        "wall": fit.pale(picture),
        "floor": a.max(axis=2) < 140,
        "jar_tint": (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 80),
        "mat": dark,
        "rest": (a[..., 0] - a[..., 2] > 50) & ~dark,  # the table's wood
    }


def _match_colours(scene: dict, out: Path, demos: fit.Demos, held: dict, episodes: list[int], report) -> dict:
    """Adjust colours until the render matches real frames: the arm, block, jar, wall, floor and table, per camera."""
    layouts = {layout["episode"]: layout for layout in scene["starts"]}
    top_table = cv2.imread(str(out / "table_top.png")).astype(np.float32)[:, :, ::-1]
    is_mat = top_table.max(axis=2) < 75
    table_gain = {"mat": np.ones(3), "rest": np.ones(3)}
    moments = [(episode, int(held[episode][0] * moment)) for episode in episodes[:: max(1, len(episodes) // 12)] for moment in (0.0, 0.5, 0.85)]
    real = {(episode, frame, camera): demos.frame(episode, frame, camera) for episode, frame in moments for camera in sim_env.CAMERAS}
    looks = {key: _real_look(picture) for key, picture in real.items()}
    for _ in range(4):
        recoloured = np.where(is_mat[..., None], top_table * table_gain["mat"], top_table * table_gain["rest"])
        cv2.imwrite(str(out / scene["table"]["texture"]), recoloured.clip(0, 255).astype(np.uint8)[:, :, ::-1])
        fit.save_json(out / "scene.json", scene)
        env = sim_env.BlockContainerEnv(scene_dir=str(out))
        env.reset(seed=0)
        gains = {camera: {} for camera in sim_env.CAMERAS}
        for episode, frame in moments:
            fit.place_for_frame(env, demos, held, layouts[episode], episode, frame)
            for camera in sim_env.CAMERAS:
                picture, drawn, look = real[episode, frame, camera], env.render_camera(camera), looks[episode, frame, camera]
                regions = _regions(env, camera, env.segment(camera))
                if "table" in regions:
                    # The mat and the wood are matched apart: the wrist camera crushes the mat to black.
                    texels, table = env.wrist_table_texels(), regions.pop("table")
                    for part, chosen in (("mat", is_mat), ("rest", ~is_mat)):
                        regions[part] = table & cv2.remap(chosen.astype(np.uint8), *texels, cv2.INTER_NEAREST).astype(bool)
                for name, mask in regions.items():
                    mask = mask & look[name]
                    if mask.sum() > 300:
                        gains[camera].setdefault(name, []).append((np.median(picture[mask], axis=0) + 2) / (np.median(drawn[mask], axis=0) + 2))
        env.close()
        for camera in sim_env.CAMERAS:
            for name, found in gains[camera].items():
                gain = np.median(found, axis=0)
                if name in table_gain:
                    table_gain[name] = table_gain[name] * gain
                else:
                    # A jar colour's fourth number is how solid it is, which stays.
                    old = scene["colours"][camera][name]
                    scene["colours"][camera][name] = [*(np.asarray(old[:3]) * gain).clip(0, 1).round(3).tolist(), *old[3:]]
            scene["colours"][camera]["block_hole"] = (np.asarray(scene["colours"][camera]["block"]) * 0.55).round(3).tolist()
    report(f"Colours from real frames: top arm {scene['colours']['top']['arm']}, wrist arm {scene['colours']['wrist']['arm']}, wrist block {scene['colours']['wrist']['block']}.")
    return scene


def _sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """Detail around the masked edges, or NaN if there's too little to measure."""
    if mask.sum() < 200:
        return float("nan")
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_32F)[mask].var())


def _match_softness(scene: dict, out: Path, demos: fit.Demos, held: dict, episodes: list[int], report) -> dict:
    """Blur each camera's rendered parts as much as the real camera's picture is blurred. The top camera is
    judged around the arm and block, its only rendered parts. The wrist camera is judged over its whole
    picture: its rendered jaws don't line up with the real ones closely enough to compare just their edges."""
    fit.save_json(out / "scene.json", scene)
    layouts = {layout["episode"]: layout for layout in scene["starts"]}
    env = sim_env.BlockContainerEnv(scene_dir=str(out))
    env.reset(seed=0)
    grow = np.ones((9, 9), np.uint8)
    moments = [(episode, int(moment * held[episode][0])) for episode in episodes[:: max(1, len(episodes) // 10)] for moment in (0.3, 0.7)]
    real, regions = {camera: [] for camera in sim_env.CAMERAS}, {}
    for episode, frame in moments:
        fit.place_for_frame(env, demos, held, layouts[episode], episode, frame)
        drawn = env.segment("top")
        regions[episode, frame] = cv2.dilate(((drawn >= 0) & ~env.is_base[drawn.clip(0)]).astype(np.uint8), grow) > 0
        real["top"].append(_sharpness(demos.frame(episode, frame), regions[episode, frame]))
        wrist = demos.frame(episode, frame, "wrist")
        real["wrist"].append(_sharpness(wrist, np.ones(wrist.shape[:2], bool)))
    softness = {}
    for camera in sim_env.CAMERAS:
        trials = {}
        for sigma in (0.0, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0):
            env.scene["softness"] = {camera: sigma}
            found = []
            for episode, frame in moments:
                fit.place_for_frame(env, demos, held, layouts[episode], episode, frame)
                picture = env.render_camera(camera)
                found.append(_sharpness(picture, regions[episode, frame] if camera == "top" else np.ones(picture.shape[:2], bool)))
            trials[sigma] = abs(np.nanmedian(found) - np.nanmedian(real[camera]))
        softness[camera] = min(trials, key=trials.get)
    env.close()
    scene["softness"] = softness
    report(f"Softness from real frames: top {softness['top']}, wrist {softness['wrist']} pixels.")
    return scene


def replay(demos_root: Path, scene_dir: Path, episodes: list[int] | None, limit: float, video: Path | None) -> dict:
    """Replay real demos' commands in the sim. A sim that can't redo them can't judge a policy.

    Each demo's block starts where the top camera saw it, as in `so101 sim eval`. Nothing corrects the
    arm along the way as a policy would, so a few millimetres' error in the block's place or the arm's
    path can cost a grasp: expect most demos to succeed, not all.
    """
    demos = fit.Demos(demos_root)
    held = fit.held_demos(demos)
    env = sim_env.BlockContainerEnv(scene_dir=str(scene_dir), max_relative_target=limit)
    layouts = {layout["episode"]: layout for layout in env.scene["starts"]}
    chosen = [e for e in (episodes if episodes is not None else sorted(layouts)) if e in layouts]
    results = []
    writer = None
    for index, episode in enumerate(chosen):
        state, action = demos.episode(episode)
        layout = layouts[episode]
        env.reset(seed=0, options={"joints": state[0], "block": layout["block"], "jar": layout["jar"]})
        start_height = env.block_pose()[0][2]
        track, height = [], []
        for frame, command in enumerate(action):
            env.apply(command)
            track.append(env.joint_state().copy())
            height.append(env.block_pose()[0][2])
            if video is not None and index == 0 and frame % 2 == 0:
                rows = [np.hstack([demos.frame(episode, frame, camera), env.render_camera(camera)]) for camera in sim_env.CAMERAS]
                row = cv2.resize(np.vstack(rows), (960, 720))
                if writer is None:
                    video.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (960, 720))
                writer.write(row[:, :, ::-1])
        for _ in range(45):  # let it settle
            env.apply(action[-1])
        results.append({
            "episode": episode,
            "error": np.abs(np.asarray(track) - state).mean(axis=0),
            "lifted": float(max(height) - start_height) > 0.03,
            "in_jar": env.block_in_jar(),
        })  # fmt: skip
    if writer is not None:
        writer.release()
    env.close()
    return {
        "results": results,
        "lifted": sum(r["lifted"] for r in results),
        "in_jar": sum(r["in_jar"] for r in results),
        "error": np.mean([r["error"] for r in results], axis=0),
    }


def replay_recording(recording: Path, scene_dir: Path, limit: float) -> dict:
    """Replay another recording's real commands in the sim: a run a policy drove on the arm, say, which
    no fit ever saw. Real commands that worked on the arm should work here, whoever gave them.

    Each attempt's jar is placed from its first top camera frame and its block from the frame 3 s in, as
    a build places a demo's: a run's first frames can still show a hand putting the block down. The block
    is then moved clear of where that attempt's own jaws came down. All this needs the top camera where
    it was for the demos. Attempts that never held the block on the arm are replayed too but counted
    apart: there's nothing they should lift.
    """
    run = fit.Demos(recording)
    held = fit.held_demos(run)
    env = sim_env.BlockContainerEnv(scene_dir=str(scene_dir), max_relative_target=limit)
    env.reset(seed=0)
    scene, units = env.scene, arm.Units.from_scene(env.scene)
    top, table_height = scene["cameras"]["top"], scene["table"]["height"]
    lens = Lens.from_scene(top["lens"])
    episodes = list(range(len(run)))
    jar = {"radius": scene["jar"]["radius"], "height": scene["jar"]["tint_height"]}
    _, jars = fit.fit_jars(run, episodes, top["pose"], lens, jar, table_height, scene["jar"]["home"], True)
    layouts = []
    delay = top.get("delay_ticks", 0)
    for episode in episodes:
        state = run.episode(episode)[0]
        frame = min(90, len(state) - 1)
        early = run.frame(episode, frame)
        blob = fit.red_blob(early, 40)
        if blob is None:  # the block is hidden then, so there's no saying where it lay
            continue
        guess = fit.on_plane(top["pose"], lens, blob[0], table_height + scene["block"]["height"] * 0.6)[0, :2]
        turn = fit.jaw_yaw(units, state[held[episode][0]]) if episode in held else 0.0
        seen, _ = fit.block_yaw(env, early, guess, state[max(frame - delay, 0)], turn)
        layouts.append({"episode": episode, "seen": seen, "jar": jars[episode]})
    fit.clear_of_jaws(env, run, held, layouts, "seen")
    results = []
    for layout in layouts:
        state, action = run.episode(layout["episode"])
        env.reset(seed=0, options={"joints": state[0], "block": layout["seen_clear"], "jar": layout["jar"]})
        start = highest = env.block_pose()[0][2]
        for command in action:
            env.apply(command)
            highest = max(highest, env.block_pose()[0][2])
        for _ in range(45):  # let it settle
            env.apply(action[-1])
        results.append({
            "episode": layout["episode"], "held": layout["episode"] in held, "lifted": bool(highest - start > 0.03), "in_jar": env.block_in_jar(),
            "moved_mm": float(np.linalg.norm(np.subtract(layout["seen_clear"][:2], layout["seen"][:2])) * 1000),
        })  # fmt: skip
    env.close()
    return {"attempts": len(run), "results": results}


def check(demos_root: Path, scene_dir: Path, policy_dir: Path, device: str, frames: int = 80) -> dict:
    """Does the model act the same on rendered images as on real ones?

    The model predicts actions from the same moment in real demos twice: once from the real camera images,
    once from sim images with the arm, block and jar in the same place. If sim images change its
    predictions much more than the real ones miss the demo, the sim can't judge this model.
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(policy_dir)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(policy_dir, config=config).to(device).eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=policy_dir, preprocessor_overrides={"device_processor": {"device": device}}
    )
    demos = fit.Demos(demos_root)
    held = fit.held_demos(demos)
    env = sim_env.BlockContainerEnv(scene_dir=str(scene_dir))
    env.reset(seed=0)
    prompt = env.scene["prompt"]
    layouts = {layout["episode"]: layout for layout in env.scene["starts"]}
    episodes = [e for e in (env.scene.get("held_back") or env.scene["episodes"]) if e in layouts]
    rng = np.random.default_rng(0)

    def predict(top: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        def image(pixels):
            return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()[None].contiguous() / 255

        batch = {
            "observation.images.top": image(top),
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
        index = int(rng.integers(0, len(state) - 1))  # any moment of the demo
        real_top, real_wrist = demos.frame(episode, index), demos.frame(episode, index, "wrist")
        drawn = fit.pictures_for_frame(env, demos, held, layouts[episode], episode, index)
        sim_top, sim_wrist = drawn["top"], drawn["wrist"]
        chunk_len = min(50, len(action) - index)
        wanted = action[index : index + chunk_len]
        real = predict(real_top, real_wrist, state[index])
        sim = predict(sim_top, sim_wrist, state[index])
        rows.append((
            np.abs(real[:chunk_len] - wanted).mean(axis=0),
            np.abs(sim[:chunk_len] - wanted).mean(axis=0),
            np.abs(predict(sim_top, real_wrist, state[index])[:chunk_len] - wanted).mean(axis=0),
            np.abs(predict(real_top, sim_wrist, state[index])[:chunk_len] - wanted).mean(axis=0),
            np.abs(sim[:chunk_len] - real[:chunk_len]).mean(axis=0),
        ))  # fmt: skip
    env.close()
    rows = np.asarray(rows).mean(axis=0)
    return {"real": rows[0], "sim": rows[1], "sim_top": rows[2], "sim_wrist": rows[3], "sim_vs_real": rows[4]}
