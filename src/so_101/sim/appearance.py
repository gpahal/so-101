"""Teach the sim what its cameras really look like, from pairs of real and rendered frames.

Rendering can't reproduce wood grain, reflections, cables, or how a webcam handles exposure. The demos
give thousands of moments seen both ways, so a small network learns the difference and the sim applies
it as a last rendering step. It only corrects appearance: geometry and physics come from the sim. It
learns from real camera frames alone, never from a policy, and some demos are held back so `so101 sim
score` is measured on frames it never saw.
"""

import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTH, HEIGHT = 320, 240
CAMERAS = ("front", "wrist")
JOINT_SCALE = torch.tensor([90.0, 90.0, 90.0, 90.0, 90.0, 50.0])


class Translator(nn.Module):
    """A small U-Net that predicts a correction to add to a rendered image. Joint angles tell it the arm's
    pose, which a camera can't see directly but which decides where shadows and reflections fall."""

    def __init__(self, width: int = 16):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())

        self.down = nn.ModuleList([block(5, width), block(width, width * 2), block(width * 2, width * 4), block(width * 4, width * 8)])
        self.film = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, width * 16))
        self.up = nn.ModuleList([block(width * 12, width * 4), block(width * 6, width * 2), block(width * 3, width)])
        self.out = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device), torch.linspace(-1, 1, w, device=image.device), indexing="ij")
        x = torch.cat([image, xs.expand(b, 1, h, w), ys.expand(b, 1, h, w)], dim=1)
        skips = []
        for index, layer in enumerate(self.down):
            x = layer(x)
            if index < len(self.down) - 1:
                skips.append(x)
                x = F.avg_pool2d(x, 2)
        scale, shift = self.film(joints / JOINT_SCALE.to(image.device)).chunk(2, dim=1)
        x = x * (1 + scale[..., None, None]) + shift[..., None, None]
        for layer, skip in zip(self.up, reversed(skips)):
            x = layer(torch.cat([F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False), skip], dim=1))
        return self.out(x)


def apply(model: Translator, image: np.ndarray, joints: np.ndarray, device: str) -> np.ndarray:
    """Correct a full-size rendered image. The correction is computed small, so fine detail stays as rendered."""
    with torch.inference_mode():
        small = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(small).permute(2, 0, 1).float()[None].to(device) / 255
        correction = model(tensor, torch.from_numpy(np.asarray(joints, dtype=np.float32))[None].to(device))[0]
        correction = F.interpolate(correction[None], size=image.shape[:2], mode="bilinear", align_corners=False)[0]
        correction = correction.permute(1, 2, 0).float().cpu().numpy()
    return np.clip(image.astype(np.float32) + correction * 255, 0, 255).astype(np.uint8)


def load(scene_dir: Path, device: str) -> dict[str, Translator]:
    models = {}
    for camera in CAMERAS:
        model = Translator()
        model.load_state_dict(torch.load(scene_dir / f"appearance_{camera}.pt", map_location="cpu"))
        models[camera] = model.to(device).eval()
    return models


def place_for_frame(env, kinematics, state: np.ndarray, frame: int, hold: int, let_go: int | None, offset: float) -> None:
    """Put the arm and object where they really were at a demo's frame: on the table, held, or in the bowl."""
    point = kinematics.held_object(state[hold], offset)
    if frame <= hold:
        env.data.qpos[env.qadr] = env.to_ctrl(state[frame])
        env.set_object(((point - env.origin) @ env.frame)[:2])
    elif let_go is None or frame <= let_go:
        grip_position, grip_rotation = kinematics.pose(state[hold])
        table_point = env.origin + env.frame @ np.r_[(point - env.origin) @ env.frame[:, :2], env.scene["object"]["height"] / 2]
        local = grip_rotation.T @ (table_point - grip_position)
        position, rotation = kinematics.pose(state[frame])
        env.data.qpos[env.qadr] = env.to_ctrl(state[frame])
        adr = env.object_qadr
        env.data.qpos[adr : adr + 3] = position + rotation @ local
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (rotation @ grip_rotation.T @ env.frame).ravel())
        env.data.qpos[adr + 3 : adr + 7] = quat
    else:
        env.data.qpos[env.qadr] = env.to_ctrl(state[frame])
        adr = env.object_qadr
        env.data.qpos[adr : adr + 3] = env.origin + env.frame @ np.r_[env.scene["bowl"]["xy"], env.scene["object"]["height"] / 2 + 0.008]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, env.frame.ravel())
        env.data.qpos[adr + 3 : adr + 7] = quat
    mujoco.mj_forward(env._model, env.data)


def split(episodes: list[int]) -> tuple[list[int], list[int]]:
    """Every fourth demo is held back."""
    return [e for i, e in enumerate(episodes) if i % 4 != 2], [e for i, e in enumerate(episodes) if i % 4 == 2]


def render_pairs(demos, kinematics, scene_dir: Path, episodes: list[int], every: int = 3):
    """Each sampled moment of the demos, as the real cameras saw it and as the sim renders it."""
    from so_101.sim.env import CylinderBowlEnv

    env = CylinderBowlEnv(scene_dir=str(scene_dir), use_appearance=False)
    env.reset(seed=0)
    offset = env.scene["object"]["grasp_offset"]
    pairs = {camera: {"sim": [], "real": []} for camera in CAMERAS}
    joints = []
    for episode in episodes:
        state = demos.episode(episode)[0]
        hold, let_go = demos.grasp_and_release(episode)
        for frame in range(0, len(state), every):
            place_for_frame(env, kinematics, state, frame, hold, let_go, offset)
            for camera in CAMERAS:
                pairs[camera]["sim"].append(cv2.resize(env.render_camera(camera), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
                pairs[camera]["real"].append(cv2.resize(demos.image(episode, frame / 30, camera), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            joints.append(state[frame])
    env.close()
    return {c: (np.stack(p["sim"]), np.stack(p["real"])) for c, p in pairs.items()}, np.asarray(joints, dtype=np.float32)


class Features(nn.Module):
    """Early VGG-16 layers, a general measure of how alike two images look. Neither ACT nor pi0.5 uses this
    network, so what it teaches the sim isn't shaped by what either model happens to look at."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        layers = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.stages = nn.ModuleList([layers[:4], layers[4:9], layers[9:16]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        found, x = [], (image - self.mean) / self.std
        for stage in self.stages:
            x = stage(x)
            found.append(x)
        return found


def train(scene_dir: Path, sim: np.ndarray, real: np.ndarray, joints: np.ndarray, camera: str, device: str, steps: int, batch: int = 12) -> None:
    model = Translator().to(device)
    features = Features().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    joint_tensor = torch.from_numpy(joints)
    started, average = time.time(), None
    for step in range(steps):
        picks = rng.integers(0, len(sim), batch)
        source = torch.from_numpy(sim[picks]).permute(0, 3, 1, 2).float().to(device) / 255
        target = torch.from_numpy(real[picks]).permute(0, 3, 1, 2).float().to(device) / 255
        output = (source + model(source, joint_tensor[picks].to(device))).clamp(0, 1)
        loss = F.l1_loss(output, target)
        for a, b in zip(features(output), features(target)):
            loss = loss + 0.1 * F.l1_loss(a, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        average = loss.item() if average is None else 0.98 * average + 0.02 * loss.item()
        if (step + 1) % 1000 == 0:
            print(f"  {camera}: step {step + 1}/{steps}, loss {average:.4f}, {time.time() - started:.0f} s", flush=True)
    torch.save(model.state_dict(), scene_dir / f"appearance_{camera}.pt")


def fit_appearance(demos, kinematics, scene_dir: Path, device: str, steps: int) -> None:
    scene = json.loads((scene_dir / "scene.json").read_text())
    fit_on, held_back = split(scene["episodes"])
    print(f"Rendering pairs from {len(fit_on)} demos, holding back {held_back}...", flush=True)
    pairs, joints = render_pairs(demos, kinematics, scene_dir, fit_on)
    print(f"{len(joints)} moments.", flush=True)
    for camera in CAMERAS:
        train(scene_dir, *pairs[camera], joints, camera, device, steps)
    scene["appearance"] = True
    scene["held_back"] = held_back
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))
