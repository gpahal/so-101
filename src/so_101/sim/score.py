"""How real the sim is, with no model involved. `so101 sim score`.

Every number compares the sim with what the real cameras and joints recorded, on demos held back from
fitting where the fit could have seen them. A policy never enters into it, so the result means the same
for any model. Real frames of the same arm pose from another demo give the scale: that's how much two
real moments differ, so the sim can't be expected to get closer than that.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from so_101.sim import fit
from so_101.sim import build as sim_build
from so_101.sim.env import CylinderBowlEnv


class ImageDistance:
    """Compares images by ImageNet ViT features, a network neither ACT nor pi0.5 uses."""

    def __init__(self, device: str):
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        self.device = device
        self.net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        self.tokens = None
        self.net.encoder.register_forward_hook(lambda _, __, output: setattr(self, "tokens", output))

    def features(self, image: np.ndarray) -> torch.Tensor:
        small = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(small).permute(2, 0, 1).float()[None].contiguous().to(self.device) / 255
        with torch.inference_mode():
            self.net((tensor - self.mean) / self.std)
        return torch.nn.functional.normalize(self.tokens[0, 1:], dim=-1)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        """0 is identical. Averaged over the image's patches, so it notices where things are."""
        return float(1 - (self.features(a) * self.features(b)).sum(dim=-1).mean())


def _pixels(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def _closest_moment(demos: fit.Demos, state: np.ndarray, others: list[int]) -> tuple[int, int]:
    best, where = np.inf, (others[0], 0)
    for other in others:
        track = demos.episode(other)[0]
        distance = np.abs(track - state).sum(axis=1)
        frame = int(np.argmin(distance))
        if distance[frame] < best:
            best, where = distance[frame], (other, frame)
    return where


def images_and_geometry(demos_root: Path, scene_dir: Path, device: str, sheet: Path | None, frames: int = 60) -> dict:
    demos = fit.Demos(demos_root)
    kinematics = fit.Kinematics(sim_build.GRIPPER_CLOSED)
    env = CylinderBowlEnv(scene_dir=str(scene_dir))
    env.reset(seed=0)
    distance = ImageDistance(device)
    episodes = env.scene.get("held_back") or env.scene["episodes"]
    offset = env.scene["object"]["grasp_offset"]
    rng = np.random.default_rng(0)
    rows = {
        camera: {f"{measure}{kind}": [] for measure in ("pixels", "features", "sharpness") for kind in ("", "_reference")}
        for camera in fit.CAMERAS
    }
    arm, jaws, object_px, object_mm, object_size, shown = [], [], [], [], [], []
    for sample in range(frames):
        episode = int(rng.choice(episodes))
        state = demos.episode(episode)[0]
        hold, let_go = demos.grasp_and_release(episode)
        frame = int(rng.integers(0, len(state)))
        fit.place_for_frame(env, kinematics, state, frame, hold, let_go, offset)
        other, other_frame = _closest_moment(demos, state[frame], [e for e in episodes if e != episode])
        pair = {}
        for camera in fit.CAMERAS:
            real = demos.image(episode, frame / 30, camera)
            final = env.render_camera(camera)
            reference = demos.image(other, other_frame / 30, camera)
            row = rows[camera]
            for name, image in (("", final), ("_reference", reference)):
                row["pixels" + name].append(_pixels(real, image))
                row["features" + name].append(distance(real, image))
                # A blurry image can score well on the two above, since a smeared guess is never far off.
                row["sharpness" + name].append(_sharpness(image) / max(_sharpness(real), 1e-6))
            pair[camera] = (real, final)
            ids = env.segment(camera)
            if camera == "front":
                drawn = (ids >= 0) & (ids != env.object_geoms["front"])
                arm.append(_overlap(fit.arm_mask(real, env.background, 640, 480) > 0, drawn))
            else:
                real_object, real_jaws = fit.wrist_masks(real)
                if real_jaws.sum() > 400:
                    jaws.append(_overlap(real_jaws, (ids >= 0) & env.is_jaw[ids.clip(0)]))
        if sheet is not None and sample < 8:
            shown.append(np.hstack([*pair["front"], *pair["wrist"]]))
        # The object while it sits still on the table, before the gripper closes in.
        for moment in (0.0, 0.3, 0.6):
            index = int(moment * max(hold - 15, 1))
            fit.place_for_frame(env, kinematics, state, index, hold, let_go, offset)
            real = demos.image(episode, index / 30, "wrist")
            found, drawn = fit.blue_blob(real), fit.blue_blob(env.render_camera("wrist"))
            whole = lambda box: box[0] > 2 and box[1] > 2 and box[0] + box[2] < 638 and box[1] + box[3] < 478
            if found and whole(found[2]):
                where = env.wrist_pixel_on_table(found[0], env.scene["object"]["height"])
                object_mm.append(float(np.linalg.norm(where - env.object_on_table()[:2]) * 1000))
                if drawn and whole(drawn[2]):
                    object_px.append(float(np.linalg.norm(found[0] - drawn[0])))
                    object_size.append(max(found[2][2:]) / max(drawn[2][2:]))
    env.close()
    if sheet is not None:
        sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet), cv2.resize(np.vstack(shown), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "images": {camera: {name: float(np.mean(values)) for name, values in row.items()} for camera, row in rows.items()},
        "geometry": {
            "front_arm_overlap": float(np.mean(arm)),
            "wrist_jaw_overlap": float(np.mean(jaws)),
            "wrist_object_offset_px": float(np.median(object_px)),
            "wrist_object_size_ratio": float(np.median(object_size)),
            "wrist_object_table_error_mm": float(np.median(object_mm)),
        },
    }


def physics(demos_root: Path, scene_dir: Path, limit: float) -> dict:
    result = sim_build.replay(demos_root, scene_dir, None, limit, None)
    count = len(result["results"])
    return {"demos": count, "lifted": result["lifted"], "in_bowl": result["in_bowl"], "joint_error": [round(float(e), 2) for e in result["error"]]}


def show(result: dict) -> None:
    print("\nImages, on demos held back from fitting. Lower is closer to the real frame, and sharpness should be near 1.")
    print(f"{'':8s}{'':12s}{'sim':>8s}{'real, same pose':>17s}")
    for camera, row in result["images"].items():
        for label, key in (("pixels", "pixels"), ("features", "features"), ("sharpness", "sharpness")):
            digits = 1 if key == "pixels" else 2 if key == "sharpness" else 3
            print(f"{camera:8s}{label:12s}{row[key]:8.{digits}f}{row[key + '_reference']:17.{digits}f}")
    geometry = result["geometry"]
    print("\nGeometry, the sim's shapes against the real ones, on held-back demos.")
    print(f"Front camera: arm overlap {geometry['front_arm_overlap']:.2f} (1 is perfect, shadows and dark parts keep it below that)")
    print(f"Wrist camera: jaw overlap {geometry['wrist_jaw_overlap']:.2f}")
    print(f"Wrist camera: object {geometry['wrist_object_offset_px']:.0f} px from where it's drawn, {geometry['wrist_object_size_ratio']:.2f}x the drawn size")
    print(f"Wrist camera: object located on the table within {geometry['wrist_object_table_error_mm']:.0f} mm of where the gripper later closed on it")
    physics_result = result["physics"]
    print("\nPhysics, replaying every real demo's commands.")
    print(f"{physics_result['lifted']}/{physics_result['demos']} lifted the object, {physics_result['in_bowl']}/{physics_result['demos']} reached the bowl. Joint error: " + ", ".join(f"{e:.1f}" for e in physics_result["joint_error"]))
    nudges = result.get("nudges")
    if nudges:
        print(f"Nudges: the object ends up {nudges['error_mm']:.0f} mm from where it really stopped, after really moving {nudges['moved_mm']:.0f} mm.")
