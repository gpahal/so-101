"""Fit how the block slides on the mat, from real recordings of it being pushed. `so101 sim physics`.

A nudge recording starts and ends with the arm clear of the block in the top camera's view, so the
block's real start and end positions can be read from its first and last frames. Replaying its commands
in the sim at different friction values shows which one moves the block as far as it really moved. No
policy is involved.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so_101.sim import fit
from so_101.sim.camera import Lens
from so_101.sim.env import BlockContainerEnv

FRICTIONS = (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3)
# Where a push goes depends on exactly where the jaws meet the block, and the reading of its start isn't
# that exact. So each push is tried from around its reading, in metres, and those results are averaged.
START_SPREAD = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) * 0.003
MIN_MOVE = 0.01  # a push moved it at least this far, in metres, or it says little about sliding


@dataclass
class Push:
    source: str
    episode: int
    state: np.ndarray
    action: np.ndarray
    start: np.ndarray  # x, y and turn on the mat
    end: np.ndarray

    @property
    def moved(self) -> float:
        return float(np.linalg.norm(self.end[:2] - self.start[:2]))


def locate(env: BlockContainerEnv, demos: fit.Demos, episode: int, state: np.ndarray, frames) -> np.ndarray | None:
    """Where the block lies on the mat, from the top camera at these frames. None if it's never fully in view."""
    top = env.scene["cameras"]["top"]
    height = env.table_height + env.scene["block"]["height"] * 0.6
    found = []
    for frame in frames:
        image = demos.frame(episode, frame)
        seen = fit.red_blob(image, 60)
        if seen is None or not fit.whole(seen[2]):
            continue
        guess = fit.on_plane(top["pose"], Lens.from_scene(top["lens"]), seen[0], height)[0, :2]
        # A pushed block can end at any turn, so try each eighth of its quarter turn.
        place, cover = max((fit.block_yaw(env, image, guess, state[frame], near, np.radians(6)) for near in np.radians(np.arange(0, 90, 11.25))), key=lambda found: found[1])
        if cover > 0.5:
            found.append(place)
    return np.median(found, axis=0) if found else None


def read_pushes(roots: list[Path], scene_dir: Path) -> tuple[list[Push], list[str]]:
    """The pushes in these recordings, and why any were left out."""
    env = BlockContainerEnv(scene_dir=str(scene_dir))
    env.reset(seed=0)
    pushes, skipped = [], []
    for root in roots:
        demos = fit.Demos(root)
        for episode in range(len(demos)):
            state, action = demos.episode(episode)
            name = f"{root.parent.name} episode {episode}"
            ends = (range(0, min(10, len(state)), 3), range(max(len(state) - 10, 0), len(state), 3))
            start, end = (locate(env, demos, episode, state, frames) for frames in ends)
            if start is None or end is None:
                skipped.append(f"{name}: the block isn't clear in the top camera's view at the {'start' if start is None else 'end'}")
                continue
            push = Push(name, episode, state, action, start, end)
            if push.moved < MIN_MOVE:
                skipped.append(f"{name}: the block only moved {push.moved * 1000:.0f} mm")
                continue
            pushes.append(push)
    env.close()
    return pushes, skipped


def replay_push(env: BlockContainerEnv, push: Push, start: np.ndarray) -> np.ndarray:
    """How far the block moves across the mat when the sim gets the real commands."""
    env.reset(seed=0, options={"joints": push.state[0], "block": start, "jar": env.scene["jar"]["home"]})
    for command in push.action:
        env.apply(command)
    for _ in range(30):  # let it settle
        env.apply(push.action[-1])
    return env.block_pose()[0][:2] - start[:2]


def errors_at(scene: dict, scene_dir: Path, pushes: list[Push], friction: float, limit: float) -> np.ndarray:
    """For each push, how far the sim's average move is from the real one."""
    scene = {**scene, "table": {**scene["table"], "friction": friction}}
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))
    env = BlockContainerEnv(scene_dir=str(scene_dir), max_relative_target=limit)
    found = []
    for push in pushes:
        moves = [replay_push(env, push, push.start + np.r_[shift, 0.0]) for shift in START_SPREAD]
        found.append(float(np.linalg.norm(np.mean(moves, axis=0) - (push.end[:2] - push.start[:2]))))
    env.close()
    return np.asarray(found)


def fit_friction(scene_dir: Path, pushes: list[Push], limit: float, came_from: dict | None = None) -> dict:
    """Try each friction value and keep the best. Restores `scene.json` if anything goes wrong."""
    path = scene_dir / "scene.json"
    original = path.read_text()
    scene = json.loads(original)
    try:
        table = {friction: errors_at(scene, scene_dir, pushes, friction, limit) for friction in FRICTIONS}
    finally:
        path.write_text(original)
    best = min(table, key=lambda friction: table[friction].mean())
    scene["table"]["friction"] = best
    scene["table"]["friction_from"] = came_from
    path.write_text(json.dumps(scene, indent=1))
    return {
        "friction": best,
        "error_mm": float(table[best].mean() * 1000),
        "moved_mm": float(np.mean([push.moved for push in pushes]) * 1000),
        "pushes": len(pushes),
        "by_friction": {friction: float(errors.mean() * 1000) for friction, errors in table.items()},
    }


def nudge_roots(task_dir: Path) -> list[Path]:
    return sorted(p for p in (task_dir / "nudges").glob("*/dataset") if (p / "meta" / "info.json").is_file())
