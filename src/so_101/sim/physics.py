"""Fit how the object slides on the table, from real recordings of it being pushed. `so101 sim physics`.

A nudge recording starts and ends with the arm at rest and the object in the wrist camera's view, so the
real start and end positions can be read from the video. Replaying its commands in the sim at different
friction values shows which one moves the object as far as it really moved. No policy is involved.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from so_101.sim import fit
from so_101.sim.env import CylinderBowlEnv

FRICTIONS = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3)
# Where a push goes depends on exactly where the jaws meet the object, and the reading of its start isn't
# that exact. So each push is tried from around its reading, in metres, and those results are averaged.
START_SPREAD = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) * 0.006
MIN_MOVE = 0.01  # a push moved it at least this far, in metres, or it says little about sliding


@dataclass
class Push:
    source: str
    episode: int
    state: np.ndarray
    action: np.ndarray
    start: np.ndarray
    end: np.ndarray

    @property
    def moved(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


def locate(env: CylinderBowlEnv, demos: fit.Demos, episode: int, state: np.ndarray, frames) -> np.ndarray | None:
    """Where the object sits on the table, from the wrist camera at these frames. None if it's never fully in view.

    A blob's centre isn't quite the object's, so draw the object there and move it until the two blobs agree.
    """
    whole = lambda box: box[0] > 2 and box[1] > 2 and box[0] + box[2] < 638 and box[1] + box[3] < 478
    height, found = env.scene["object"]["height"], []
    for frame in frames:
        env.data.qpos[env.qadr] = env.to_ctrl(state[frame])
        env.set_object([0.0, -0.4])
        mujoco.mj_forward(env._model, env.data)
        blob = fit.blue_blob(demos.image(episode, frame / 30, "wrist"))
        if blob is None or not whole(blob[2]):
            continue
        real = env.wrist_pixel_on_table(blob[0], height)
        where = real.copy()
        for _ in range(4):
            env.set_object(where)
            mujoco.mj_forward(env._model, env.data)
            drawn = fit.blue_blob(env.render_camera("wrist"))
            if drawn is None:
                break
            where = where + real - env.wrist_pixel_on_table(drawn[0], height)
        found.append(where)
    return np.median(found, axis=0) if found else None


def read_pushes(roots: list[Path], scene_dir: Path) -> tuple[list[Push], list[str]]:
    """The pushes in these recordings, and why any were left out."""
    env = CylinderBowlEnv(scene_dir=str(scene_dir))
    env.reset(seed=0)
    pushes, skipped = [], []
    for root in roots:
        demos = fit.Demos(root)
        for episode in range(len(demos.episodes)):
            state, action = demos.episode(episode)
            name = f"{root.parent.name} episode {episode}"
            ends = (range(0, min(10, len(state)), 3), range(max(len(state) - 10, 0), len(state), 3))
            start, end = (locate(env, demos, episode, state, frames) for frames in ends)
            if start is None or end is None:
                skipped.append(f"{name}: the object isn't in the wrist camera's view at the {'start' if start is None else 'end'}")
                continue
            push = Push(name, episode, state, action, start, end)
            if push.moved < MIN_MOVE:
                skipped.append(f"{name}: the object only moved {push.moved * 1000:.0f} mm")
                continue
            pushes.append(push)
    env.close()
    return pushes, skipped


def replay_push(env: CylinderBowlEnv, push: Push, start: np.ndarray) -> np.ndarray:
    """How much the object moves, in the table's frame, when the sim gets the real commands."""
    env.reset(seed=0, options={"object_xy": start})
    env._pending.clear()
    env.data.qpos[env.qadr] = env.to_ctrl(push.state[0])
    env.data.ctrl[env.actuators] = env.to_ctrl(push.state[0])
    mujoco.mj_forward(env._model, env.data)
    for command in push.action:
        env.apply(command)
    for _ in range(30):  # let it settle
        env.apply(push.action[-1])
    return env.object_on_table()[:2] - start


def errors_at(scene: dict, scene_dir: Path, pushes: list[Push], friction: float, limit: float) -> np.ndarray:
    """For each push, how far the sim's average move is from the real one."""
    scene = {**scene, "object": {**scene["object"], "table_friction": friction}}
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))
    env = CylinderBowlEnv(scene_dir=str(scene_dir), max_relative_target=limit)
    found = []
    for push in pushes:
        moves = [replay_push(env, push, push.start + shift) for shift in START_SPREAD]
        found.append(float(np.linalg.norm(np.mean(moves, axis=0) - (push.end - push.start))))
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
    scene["object"]["table_friction"] = best
    scene["object"]["table_friction_from"] = came_from
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
