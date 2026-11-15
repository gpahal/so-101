"""A task's built demos on disk: each episode's joints and the frames its cameras recorded."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

FPS = 30


@dataclass
class Demos:
    """A LeRobot dataset on disk, with what each episode's joints and videos show."""

    root: Path

    @cached_property
    def frames(self) -> pd.DataFrame:
        files = sorted(self.root.glob("data/*/*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @cached_property
    def episodes(self) -> pd.DataFrame:
        files = sorted(self.root.glob("meta/episodes/*/*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @cached_property
    def _decoders(self) -> dict:
        return {}

    @cached_property
    def _tracks(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return {
            int(episode): (np.stack(rows["observation.state"]).astype(np.float64), np.stack(rows["action"]).astype(np.float64))
            for episode, rows in self.frames.groupby("episode_index")
        }

    def __len__(self) -> int:
        return len(self.episodes)

    def image(self, episode: int, seconds: float, camera: str = "top") -> np.ndarray:
        row = self.episodes.iloc[episode]
        key = f"videos/observation.images.{camera}"
        path = (
            self.root / "videos" / f"observation.images.{camera}"
            / f"chunk-{int(row[key + '/chunk_index']):03d}" / f"file-{int(row[key + '/file_index']):03d}.mp4"
        )  # fmt: skip
        decoder = self._decoders.setdefault(str(path), VideoDecoder(str(path)))
        at = float(row[key + "/from_timestamp"]) + seconds
        return decoder.get_frames_played_at([at]).data[0].permute(1, 2, 0).numpy()

    def frame(self, episode: int, index: int, camera: str = "top") -> np.ndarray:
        """A frame by its number. Half a frame in, because a time on the boundary can give the frame before,
        which for a demo's first frame is the end of the previous demo."""
        return self.image(episode, (index + 0.5) / FPS, camera)

    def episode(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        """The follower's readings and the commands sent to it, one row per frame."""
        return self._tracks[episode]

    def grasp_and_release(self, episode: int) -> tuple[int, int] | None:
        """The frame the gripper has settled on the block, and the frame it starts to let go. None if the
        demo holds nothing.

        A hold shows as squeezing: the gripper is told to close further than it does, and stays put.
        """
        state, action = self.episode(episode)
        grip, command = state[:, 5], action[:, 5]
        squeezing = (command < grip - 1.5) & (grip > 5) & (np.abs(np.gradient(grip)) < 0.6)
        best, start, gap = (0, 0), None, 0
        for index, held in enumerate(np.r_[squeezing, np.zeros(6, bool)]):
            if held:
                start, gap = index if start is None else start, 0
            elif start is not None:
                gap += 1
                if gap > 4:  # a few frames of slack don't end a hold
                    end = index - gap + 1
                    if end - start > best[1] - best[0]:
                        best = (start, end)
                    start = None
        if best[1] - best[0] < 15:
            return None
        return best[0] + 3, best[1]
