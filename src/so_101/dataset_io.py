"""Read episodes of a local LeRobot v3 dataset: joint arrays from parquet, camera frames from video."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def episode_table(meta: LeRobotDatasetMetadata, ep: int) -> pd.DataFrame:
    """The episode's rows (timestamp, frame_index, observation.state, action, ...), in frame order."""
    df = pd.read_parquet(meta.root / meta.get_data_file_path(ep))
    return df[df["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)


def column(table: pd.DataFrame, name: str) -> np.ndarray:
    """A vector column as a (frames, values) float32 array."""
    return np.stack(table[name].to_numpy()).astype(np.float32)


def episode_frames(
    meta: LeRobotDatasetMetadata, ep: int, video_key: str, start: int = 0, stop: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame index within the episode, RGB HxWx3 uint8 frame) for frames start..stop-1.

    Episodes share video files, each episode starting at its own `from_timestamp`. Frame i of the
    episode is the one shown at from_timestamp + i / fps.
    """
    import av

    av.logging.set_level(av.logging.ERROR)  # hide FFmpeg's per-frame colour-conversion warnings
    length = int(meta.episodes[ep]["length"])
    stop = length if stop is None else min(stop, length)
    if start >= stop:
        return
    from_ts = float(meta.episodes[ep][f"videos/{video_key}/from_timestamp"])
    fps = meta.fps
    with av.open(str(meta.root / meta.get_video_file_path(ep, video_key))) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek a second early so decoding starts from a keyframe before the first wanted frame.
        target = max(from_ts + start / fps - 1.0, 0.0)
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        expected = start
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            i = round((float(frame.pts * stream.time_base) - from_ts) * fps)
            if i < expected:
                continue
            if i != expected:
                raise ValueError(
                    f"{video_key}: episode {ep} is missing video frame {expected} (next frame found is {i})."
                )
            yield i, frame.to_ndarray(format="rgb24")
            expected += 1
            if expected == stop:
                return
    if expected < stop:
        raise ValueError(f"{video_key}: episode {ep} video ends after {expected} of {stop} frames.")
