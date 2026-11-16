"""Candidate samples from `so101 serve` and `so101 run` runs, for labelling into a labels file.

The grasp detector (`grasp.detector`) replays each saved attempt as the perception saw it live: only the
Autonomous frames the model drove (`control.driver` POLICY or WAITING), on the model's own clock, with
`resumed` called where the model drives again after anything else held or moved the arm (a pause, a
person's Teleop or Rewind, a layer's playback or hold). Each moment the supervisor would have checked
becomes a candidate row: `after_close` and `after_release` at the program's delays, `stuck` where a rule
fired, `approach` halfway from the policy's first frame to its first close (3 s in without one), and `end`
at the episode's last frame. A run recorded before the `control.driver` column counts every Autonomous
frame as the model's.

Rows have the columns `dataset.read_labels` reads, with `label` and `category` empty. `write` merges into
the file it writes: a row whose id is already there keeps its label, category and note, and a labelled row
the rules no longer pick stays, so running it again after more runs never loses a label. `src` is the
run's folder name under the runs folder.
"""

import csv
from pathlib import Path

import numpy as np

from so_101.settings import replace_file

AUTONOMOUS = 1  # control.phase, as `control.recorder.Phase`
MOVING_PHASES = (0, 4, 5, 6)  # STARTING, TELEOP, REWINDING and RETURNING: the arm moved
POLICY, WAITING, PLAYBACK = 1, 2, 3  # control.driver, as `policy.base`
COLUMNS = ("id", "src", "ep", "t", "moment", "label", "category", "gripper", "note")
KEPT = ("label", "category", "note")  # what a person fills in, kept by id when the file is written again
HEADER = """# Candidate samples from so101 serve and run runs, picked by the grasp detector (so101-failbench candidates).
# Fill in label (ok or wrong) and category (a key of vision/prompt.py's CATEGORIES, none for ok) from the
# frames, and say in note how each label was reached. Leave label empty where you can't decide: build
# leaves those rows out. Running candidates again keeps every label, category and note, by id.
"""


def pick(run: str, episode: int, phases: np.ndarray, waiting: np.ndarray, states: np.ndarray, names: list[str],
         fps: float, program, thresholds, drivers: np.ndarray | None = None) -> list[dict]:
    """The candidate rows of one attempt. `phases`, `waiting` (no policy action), `states` (frames x joints,
    named `names`) and `drivers` (`control.driver`, None for a run recorded before it) are the episode's
    columns in frame order. `program` gives the supervisor's check delays, `thresholds` the detector's."""
    from so_101.grasp.detector import Detector

    detector = Detector(thresholds)
    gripper = names.index("gripper.pos")
    frames, rows = [], []  # (the model's seconds, frame) of each frame it drove
    found: list[tuple[float, str, str]] = []  # (the model's seconds, moment, note)
    first_close = None
    t = 0.0
    skipped = moved = False  # since the model last drove: frames it didn't, and whether the arm moved in them
    for i, phase in enumerate(phases):
        driver = None if drivers is None else int(drivers[i])
        if phase != AUTONOMOUS or driver not in (None, POLICY, WAITING):
            skipped = True
            moved = moved or phase in MOVING_PHASES or driver == PLAYBACK
            continue
        pose = dict(zip(names, map(float, states[i]), strict=True))
        if skipped:
            detector.resumed(moved, pose)
        if frames and not skipped:
            t += 1 / fps  # the perception's clock: a gap in the model's driving counts nothing
        skipped = moved = False
        frames.append((t, i))
        drove = driver == POLICY if driver is not None else not waiting[i]
        for s in detector.step(t, pose, drove):
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
    from so_101.control.analysis import read_data
    from so_101.data import Run

    rows = []
    for folder in sorted(p for p in Path(runs).iterdir() if (p / "dataset").is_dir()):
        run = Run.at(folder)
        if (info := run.info()) is None:
            continue
        names = list(info["features"]["observation.state"]["names"])
        table, _ = read_data(run.dataset_root, ["episode_index", "frame_index", "observation.state", "control.phase",
                                                "control.policy_action"], optional=("control.driver",))
        if table is None:
            continue
        for episode, ep in table.groupby("episode_index"):
            ep = ep.sort_values("frame_index")
            phases = np.array([int(np.ravel(p)[0]) for p in ep["control.phase"]])
            waiting = np.array([bool(np.isnan(np.asarray(v, dtype=float)).all()) for v in ep["control.policy_action"]])
            states = np.stack([np.asarray(v, dtype=float) for v in ep["observation.state"]])
            drivers = (np.array([int(np.ravel(d)[0]) for d in ep["control.driver"]])
                       if "control.driver" in ep and not ep["control.driver"].isna().any() else None)
            rows += pick(folder.name, int(episode), phases, waiting, states, names, float(info["fps"]), program,
                         thresholds, drivers=drivers)
    return rows


def _existing(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def write(rows: list[dict], path: Path) -> int:
    """Write `rows` to `path`, merged by id with what's there: an existing row keeps its label, category
    and note, and a labelled row not in `rows` stays, after them. Returns how many labelled rows were kept."""
    import io

    old = {row["id"]: row for row in _existing(path)}
    merged, kept = [], 0
    for row in rows:
        before = old.pop(row["id"], None)
        if before is not None and (before.get("label") or "").strip():
            row = {**row, **{k: before.get(k) or "" for k in KEPT}}
            kept += 1
        merged.append(row)
    for row in old.values():
        if (row.get("label") or "").strip():
            merged.append({k: row.get(k) or "" for k in COLUMNS})
            kept += 1
    out = io.StringIO()
    out.write(HEADER)
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, out.getvalue())
    return kept
