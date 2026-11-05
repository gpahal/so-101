"""`so101 build`: turn recorded sessions plus edits.toml into one training dataset.

Raw sessions are never changed. Each kept episode is read back (joint rows and decoded video
frames), cut to its `keep` range, and written into a fresh dataset with LeRobot's own writer, so
frame indices, timestamps, video files and normalization stats are all consistent. Decoding and
re-encoding loses a little image quality, so the build encodes at a higher quality (lower CRF)
than recording does.
"""

import contextlib
import json
import shutil
import time
from dataclasses import dataclass

from so_101 import data
from so_101.dataset_io import column, episode_frames, episode_table
from so_101.report import Report
from so_101.settings import ConfigError, Settings
from so_101.tasks import Task
from so_101.validate import validate_dataset

BUILD_CRF = 23
# SVT-AV1 has no preset above 10 here: LeRobot's default of 12 runs as 10, with a warning per episode.
BUILD_PRESET = 10
MIN_EPISODE_S = 0.5


@dataclass(frozen=True)
class Plan:
    session: data.Session
    episode: int
    start: int  # first frame kept
    stop: int  # one past the last frame kept


def plan(s: Settings, task: Task) -> tuple[list[Plan], list[str]]:
    """Which frames of which episodes go into the build, and notes on what was left out."""
    edits = data.load_edits(s, task)
    plans, notes, recorded = [], [], set()
    reference = None
    for session in data.sessions(s, task):
        meta = session.meta
        if meta is None:
            notes.append(f"session {session.name}: no usable dataset (empty or interrupted), skipped")
            continue
        shape = (meta.fps, meta.robot_type, {k: (f["dtype"], tuple(f["shape"])) for k, f in meta.features.items()})
        if reference is None:
            reference = (session.name, shape)
        elif shape != reference[1]:
            raise ConfigError(
                f"Session {session.name} was recorded with different settings (fps, cameras or joints) from "
                f"session {reference[0]}, so they can't go into one dataset. Move one of them out of "
                f"{session.path.parent}."
            )
        for ep in range(meta.total_episodes):
            recorded.add((session.name, ep))
            length = int(meta.episodes[ep]["length"])
            edit = edits.get((session.name, ep))
            if edit and edit.delete is not None:
                notes.append(f"session {session.name} episode {ep}: deleted ({edit.delete or 'no reason given'})")
                continue
            start, stop = 0, length
            if edit and edit.keep:
                start = min(length, round(edit.keep[0] * meta.fps))
                stop = min(length, round(edit.keep[1] * meta.fps))
            if stop - start < MIN_EPISODE_S * meta.fps:
                raise ConfigError(
                    f"Session {session.name} episode {ep}: keeping frames {start}-{stop} of {length} leaves "
                    f"less than {MIN_EPISODE_S} s. Fix it in {data.edits_path(s, task)}."
                )
            plans.append(Plan(session, ep, start, stop))
    unknown = sorted(f"{session}/{episode}" for session, episode in set(edits) - recorded)
    if unknown:
        notes.append(f"edits.toml mentions episodes that don't exist: {', '.join(unknown)}")
    return plans, notes


def build(s: Settings, task: Task) -> Report:
    """Build data/<task>/train from the sessions and edits. Returns the validation report."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import DEFAULT_FEATURES

    plans, notes = plan(s, task)
    for note in notes:
        print(f"  {note}")
    if not plans:
        raise ConfigError(f"Nothing to build for {task.name}: record some demos first (`uv run so101 record {task.name}`).")

    source = plans[0].session.meta
    features = {
        key: {"dtype": f["dtype"], "shape": tuple(f["shape"]), "names": f.get("names")}
        for key, f in source.features.items()
        if key not in DEFAULT_FEATURES
    }
    root = data.train_root(s, task)
    tmp = root.with_name("train.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    out = LeRobotDataset.create(
        repo_id=task.dataset_repo or f"local/so101_{task.name}",
        fps=source.fps,
        features=features,
        root=tmp,
        robot_type=source.robot_type,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(crf=BUILD_CRF, preset=BUILD_PRESET),
        # Streaming drops frames when its encoder falls behind, which suits a live recording but not
        # this: nothing is waiting on a build, so write every frame and encode each episode after.
        streaming_encoding=False,
        image_writer_threads=8,
    )

    video_keys = source.video_keys
    started = time.perf_counter()
    total = sum(p.stop - p.start for p in plans)
    done = 0
    mapping = []
    try:
        for n, p in enumerate(plans):
            meta = p.session.meta
            table = episode_table(meta, p.episode)
            state, action = column(table, "observation.state"), column(table, "action")
            streams = [episode_frames(meta, p.episode, key, p.start, p.stop) for key in video_keys]
            try:
                for i in range(p.start, p.stop):
                    frame = {"observation.state": state[i], "action": action[i], "task": task.prompt}
                    for key, stream in zip(video_keys, streams, strict=True):
                        index, image = next(stream)
                        assert index == i, f"{key}: got frame {index}, expected {i}"
                        frame[key] = image
                    out.add_frame(frame)
            except (ValueError, StopIteration, AssertionError) as e:
                raise ConfigError(
                    f"Session {p.session.name} episode {p.episode} is damaged ({e!r}). Delete it with "
                    f"`uv run so101 review {task.name} --episode {p.session.name}/{p.episode}` and build again."
                ) from e
            out.save_episode()
            done += p.stop - p.start
            mapping.append({"session": p.session.name, "episode": p.episode, "frames": [p.start, p.stop]})
            elapsed = time.perf_counter() - started
            print(
                f"\r  episode {n + 1}/{len(plans)}: {done}/{total} frames, {done / max(elapsed, 1e-6):.0f} frames/s ",
                end="",
                flush=True,
            )
        print()
        out.finalize()
    except BaseException:
        with contextlib.suppress(Exception):
            out.finalize()  # stop the video encoder threads before deleting their output
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    report = Report()
    # Tracking only warns here: the demos are already recorded, and the dry run is where it should block.
    validate_dataset(tmp, f"local/so101_{task.name}", report, group="train dataset", strict_tracking=False)
    if report.failures:
        report.print("Validation of the new build")
        raise ConfigError(f"The build failed validation; it was left at {tmp} for inspection. The old build is unchanged.")
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": data.build_inputs(s, task),
        "episodes": mapping,
        "frames": total,
        "notes": notes,
    }
    data.build_manifest_path(s, task).write_text(json.dumps(manifest, indent=2) + "\n")
    return report
