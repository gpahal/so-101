"""Check that a recorded or built dataset is complete, in sync and trainable."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from so_101.dataset_io import column, episode_frames, episode_table
from so_101.report import Report

# A joint counts as moved when its commanded position spans at least this much (degrees; the
# gripper is 0-100).
MOVED_SPAN = {"gripper": 20.0}
MOVED_SPAN_DEFAULT = 15.0
MAX_LAG_S = 0.5


@dataclass(frozen=True)
class DatasetSummary:
    lengths: list[int]  # frames per episode
    fps: int


def tracking(
    report: Report,
    group: str,
    names: list[str],
    episodes: list[tuple[np.ndarray, np.ndarray]],
    fps: float,
    require_motion: bool,
    strict: bool = True,
) -> None:
    """Check that the follower followed the leader.

    `episodes` holds (commanded, measured) arrays of shape (frames, joints): the leader's positions
    and the follower's positions read at the same steps. With `strict` off, a joint that clearly
    didn't follow is a warning rather than a failure: the check is a heuristic (a gripper closing on
    an object stops short of the leader, for one).
    """
    episodes = [(cmd, meas) for cmd, meas in episodes if len(cmd) > 2]
    if not episodes:
        report.warn(group, "too little data to check how the follower follows the leader")
        return
    max_lag = max(0, min(int(MAX_LAG_S * fps), min(len(c) for c, _ in episodes) // 4))
    still, problems, lags, errors = [], [], [], []
    for j, raw_name in enumerate(names):
        name = raw_name.removesuffix(".pos")
        cmd_span = max(float(np.ptp(cmd[:, j])) for cmd, _ in episodes)
        meas_span = max(float(np.ptp(meas[:, j])) for _, meas in episodes)
        if cmd_span < MOVED_SPAN.get(name, MOVED_SPAN_DEFAULT):
            still.append(name)
            continue
        # The follower reaches a commanded position a few steps later; find that delay.
        err_by_lag = []
        for lag in range(max_lag + 1):
            diffs = np.concatenate([np.abs(meas[lag:, j] - cmd[: len(cmd) - lag, j]) for cmd, meas in episodes])
            err_by_lag.append(float(np.median(diffs)))
        lag = int(np.argmin(err_by_lag))
        err = err_by_lag[lag]
        ratio = meas_span / cmd_span
        lags.append(lag)
        errors.append(err)
        unit = "" if name == "gripper" else "°"
        if name == "gripper" and ratio < 0.8:
            # Falling short only on the closing side means the jaws stopped on something, usually the
            # object. Squeezing the leader further makes the follower's gripper stall against it.
            cmd_lo = min(float(cmd[:, j].min()) for cmd, _ in episodes)
            cmd_hi = max(float(cmd[:, j].max()) for cmd, _ in episodes)
            meas_lo = min(float(meas[:, j].min()) for _, meas in episodes)
            meas_hi = max(float(meas[:, j].max()) for _, meas in episodes)
            if meas_lo - cmd_lo > 2 * max(0.0, cmd_hi - meas_hi):
                report.warn(
                    group,
                    f"gripper: the follower's jaws stopped at {meas_lo:.0f} while the leader closed to "
                    f"{cmd_lo:.0f}, usually because they were holding the object",
                    fix="close the leader's gripper only until the follower holds the object: squeezing "
                    "further stalls the follower's gripper and trips its overload alarm (blinking red). If it "
                    "stops short with nothing in its jaws, recalibrate the follower",
                )
                problems.append(name)
                continue
        if ratio < 0.5:
            (report.fail if strict else report.warn)(
                group,
                f"{name}: the follower covered only {ratio:.0%} of the leader's motion "
                f"({meas_span:.0f}{unit} of {cmd_span:.0f}{unit})",
                fix="recalibrate the follower (`uv run so101 calibrate follower`), moving every joint fully; "
                "check the joint isn't jammed",
            )
            problems.append(name)
        elif ratio < 0.8 or err > 8:
            report.warn(
                group,
                f"{name}: the follower lags or falls short (covered {ratio:.0%} of the leader's motion, "
                f"median error {err:.1f}{unit})",
                fix="check the joint for friction or a loose horn; compare calibrations of both arms",
            )
            problems.append(name)
    if lags and len(problems) < len(lags):
        report.ok(
            group,
            f"follower follows the leader ({len(lags) - len(problems)} joints; delay about "
            f"{1000 * float(np.median(lags)) / fps:.0f} ms, median error {float(np.median(errors)):.1f})",
        )
    if still and require_motion:
        report.warn(
            group,
            f"didn't move, so not checked: {', '.join(still)}",
            fix="move every joint, and open and close the gripper",
        )


def idle_trim(action: np.ndarray, fps: float) -> tuple[float, float] | None:
    """Suggest [start, end] seconds that cut idle time at the start and end of an episode.

    Keeps 0.2 s before the first motion and 0.7 s after the last, so the policy still learns to
    start promptly and to settle at the end. Returns None if nothing moved.
    """
    if len(action) < 3:
        return None
    speed = np.max(np.abs(np.diff(action, axis=0)), axis=1) * fps  # fastest joint, units per second
    moving = speed > 5.0
    # Ignore single-step blips (encoder noise): require 3 moving steps in a row.
    sustained = np.convolve(moving.astype(int), np.ones(3, dtype=int), mode="same") >= 3
    idx = np.flatnonzero(sustained)
    if idx.size == 0:
        return None
    duration = len(action) / fps
    start = max(0.0, float(idx[0] - 1) / fps - 0.2)
    end = min(duration, float(idx[-1] + 2) / fps + 0.7)
    return round(start, 2), round(end, 2)


def validate_dataset(
    root: Path,
    repo_id: str,
    report: Report,
    *,
    group: str = "dataset",
    require_motion: bool = False,
    strict_tracking: bool = True,
) -> DatasetSummary | None:
    """Check episodes, timestamps, videos, stats and follower tracking. Adds results to `report`."""
    try:
        meta = LeRobotDatasetMetadata(repo_id, root=root)
    except Exception as e:
        report.fail(group, f"can't open the dataset at {root}: {e}")
        return None
    if meta.total_episodes == 0 or meta.episodes is None:
        report.fail(group, "the dataset has no episodes")
        return None
    fps = meta.fps
    lengths = [int(meta.episodes[ep]["length"]) for ep in range(meta.total_episodes)]
    report.ok(group, f"{meta.total_episodes} episodes, {sum(lengths)} frames, {sum(lengths) / fps:.1f} s at {fps} fps")

    # Joint data: one row per frame, timestamps at exactly 1/fps.
    names = list(meta.features["action"]["names"])
    pairs, bad_rows = [], []
    for ep in range(meta.total_episodes):
        table = episode_table(meta, ep)
        frame_index = table["frame_index"].to_numpy()
        timestamps = table["timestamp"].to_numpy(dtype=np.float64)
        if (
            len(table) != lengths[ep]
            or not np.array_equal(frame_index, np.arange(lengths[ep]))
            or not np.allclose(timestamps, frame_index / fps, atol=1e-4)
        ):
            bad_rows.append(ep)
            continue
        pairs.append((column(table, "action"), column(table, "observation.state")))
    if bad_rows:
        report.fail(group, f"episodes with missing rows or uneven timestamps: {bad_rows}")
    else:
        report.ok(group, "joint data: every frame present, timestamps evenly spaced")

    # Video: every frame decodes, at the configured size, with a sane picture.
    for key in meta.video_keys:
        height, width = meta.features[key]["shape"][:2]
        camera = f"{key.removeprefix('observation.images.')} video"
        brightness, problems = [], []
        for ep in range(meta.total_episodes):
            try:  # episode_frames yields every frame of the episode's length, or raises
                for i, frame in episode_frames(meta, ep, key):
                    if frame.shape[:2] != (height, width):
                        problems.append(f"episode {ep} is {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}")
                        break
                    if i % 10 == 0:
                        brightness.append(float(frame.mean()))
            except Exception as e:
                problems.append(str(e))
        if problems:
            report.fail(camera, "; ".join(problems[:3]))
            continue
        mean = float(np.mean(brightness))
        if mean < 25:
            report.fail(camera, f"mostly black (mean brightness {mean:.0f}/255)", fix="check the lens and the light")
        elif mean > 235:
            report.fail(camera, f"mostly white (mean brightness {mean:.0f}/255)", fix="lower the exposure")
        else:
            report.ok(camera, f"every frame decodes at {width}x{height}, mean brightness {mean:.0f}/255")

    if meta.stats is None or not {"action", "observation.state"} <= set(meta.stats):
        report.fail(group, "normalization stats are missing (meta/stats.json)")

    tracking(report, "follower", names, pairs, fps, require_motion, strict=strict_tracking)
    return DatasetSummary(lengths, fps)
