"""Candidate samples from `so101 serve` runs, for labelling into a new labels file.

The grasp detector (`grasp.detector`) replays each saved attempt's policy frames, as the perception saw
them live, and each moment it would have checked becomes a candidate row: `after_close` and
`after_release` at the program's delays, `stuck` where a rule fired, `approach` halfway from the policy's
first frame to its first close (3 s in without one), and `end` at the episode's last frame. Rows have the
same columns as `labels.csv`, with `label` and `category` empty: fill them in, then build with
`so101-failbench build --labels <file>`. `src` is the run's folder name under the runs folder.
"""

import csv
from pathlib import Path

import numpy as np

AUTONOMOUS = 1  # control.phase, as `control.recorder.Phase`
COLUMNS = ("id", "src", "ep", "t", "moment", "label", "category", "gripper", "note")
HEADER = """# Candidate samples from so101 serve runs, picked by the grasp detector (so101-failbench candidates).
# Fill in label (ok or wrong) and category (a key of vision/prompt.py's CATEGORIES, none for ok) from the
# frames, delete the rows you can't decide, and say in note how each label was reached.
"""


def pick(run: str, episode: int, phases: np.ndarray, waiting: np.ndarray, states: np.ndarray, names: list[str],
         fps: float, program, thresholds) -> list[dict]:
    """The candidate rows of one attempt. `phases`, `waiting` (no policy action) and `states` (frames x joints,
    named `names`) are the episode's columns in frame order. `program` gives the supervisor's check
    delays, `thresholds` the detector's."""
    from so_101.grasp.detector import Detector

    detector = Detector(thresholds)
    gripper = names.index("gripper.pos")
    previous, frames, rows = None, [], []
    found: list[tuple[int, str, str]] = []  # (frame, moment, note)
    first_close = None
    for i, phase in enumerate(phases):
        if phase != AUTONOMOUS:
            previous = phase
            continue
        if previous is not None and previous != AUTONOMOUS:
            detector.resumed(True, dict(zip(names, map(float, states[i]), strict=True)))
        t = len(frames) / fps  # the policy's clock: its frames so far
        frames.append((t, i))
        previous = phase
        pose = dict(zip(names, map(float, states[i]), strict=True))
        for s in detector.step(t, pose, not waiting[i]):
            if s.kind == "close":
                first_close = first_close if first_close is not None else (t, i)
                found.append((s.t + program.after_close_s, "after_close", f"close {s.detail['close']}"))
            elif s.kind == "release":
                found.append((s.t + program.after_release_s, "after_release", "release"))
            elif s.kind == "rule":
                found.append((s.t, "stuck", f"rule {s.name}"))
    if not frames:
        return []
    start_t = frames[0][0]
    approach_t = (start_t + first_close[0]) / 2 if first_close else start_t + 3.0
    found.append((approach_t, "approach", "halfway to the first close" if first_close else "3 s in, no close"))
    times = np.array([f[0] for f in frames])
    picked = []
    for when, moment, note in found:
        k = int(np.searchsorted(times, when - 1e-9))
        if k >= len(frames):
            continue  # the policy stopped driving before then
        picked.append((frames[k][1], moment, note))
    picked.append((len(phases) - 1, "end", "last frame"))
    seen = set()
    for frame, moment, note in picked:
        key = f"{run}-e{episode}-t{frame}"
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": key, "src": run, "ep": episode, "t": frame, "moment": moment, "label": "", "category": "",
                     "gripper": round(float(states[frame][gripper]), 1), "note": note})
    return rows


def from_runs(runs: Path, program, thresholds) -> list[dict]:
    """Every saved attempt's candidates, from each run folder under `runs`."""
    import json

    from so_101.control.analysis import read_data

    rows = []
    for folder in sorted(p for p in Path(runs).iterdir() if (p / "dataset" / "meta" / "info.json").is_file()):
        info = json.loads((folder / "dataset" / "meta" / "info.json").read_text())
        names = list(info["features"]["observation.state"]["names"])
        table, _ = read_data(folder / "dataset", ["episode_index", "frame_index", "observation.state", "control.phase",
                                                  "control.policy_action"])
        if table is None:
            continue
        for episode, ep in table.groupby("episode_index"):
            ep = ep.sort_values("frame_index")
            phases = np.array([int(np.ravel(p)[0]) for p in ep["control.phase"]])
            waiting = np.array([bool(np.isnan(np.asarray(v, dtype=float)).all()) for v in ep["control.policy_action"]])
            states = np.stack([np.asarray(v, dtype=float) for v in ep["observation.state"]])
            rows += pick(folder.name, int(episode), phases, waiting, states, names, float(info["fps"]), program,
                         thresholds)
    return rows


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(HEADER)
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
