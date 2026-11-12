"""Record a `so101 serve` run: one LeRobot dataset episode per attempt, and an event log.

The controller owns the recorder and is its only caller. Every tick of an open attempt adds one
frame, whatever the arm is doing, so timestamps stay evenly spaced. The episode is saved when the
arm arrives at rest, or where the arm holds: at once after a fault, or where a quit's drive to rest
stopped short. Each attempt's data file is finished (finalized) on disk before the attempt counts as
saved, its `attempt_end` is written or the next one starts, and events are written as they happen,
one JSON object per line, flushed per line. After each finished save, a copy of the dataset's shared
metadata goes beside it (`data.Run.save_meta_copy`), which readers fall back to when a kill during a
later save tears the live one. So a crash, even a SIGKILL or a power cut, loses only the open
attempt's frames (or the one being saved), and none of its events. A run that ended cleanly ends
with `run_end`; one without it crashed. The run's `run.json` isn't the recorder's: it is never edited.

Each event has `id` (numbered from 1: its line in the file), `type`, `t` (seconds since the run
started, on the control loop's clock) and `wall` (Unix time). During an attempt it also has
`attempt` (numbered from 1) and `frame`: the index, in that attempt's episode, of the next frame to
be recorded. A `state` event's `frame` is the first frame recorded in the new state, so events line
up with the `control.phase` column. No other field can replace these.

    attempt_start  attempt
    command        command, accepted, reason (None when accepted), its options (outcome, label, for_attempt),
                   source (terminal, page or signal), who (a page's caller, for page commands
                   only: its Tailscale login behind HTTPS, "this Mac", or its IP address), via
                   ("toggle" when the page's Space became this pause or resume) and operator
    state          state, previous, cause (the command or event that caused it), ended_by (when
                   returning), note
    fault          error (what failed, and the fix), device (follower or a camera's name, when one is to
                   blame) and traceback (when there is one). The arm holds where it is, and an open
                   attempt is saved there, ended_by fault
    handover       leader and follower (the poses when the leader was freed), farthest (the joint
                   farthest apart) and gap (how far, in the arm's units)
    rewind         logged when a rewind ends. start_frame and end_frame: the frames from start_frame up
                   to, not including, end_frame had their actions undone, and the arm went back to the
                   pose measured in start_frame. Ranges of rewinds in a row are adjacent, not
                   overlapping. seconds (of motion undone), cause (arrived, stopped_short, pause,
                   reset, quit or fault), farthest and gap (the joint farthest from start_frame's pose
                   at the end, and how far)
    attempt_end    ended_by, saved, episode (None when not saved), frames, seconds (per phase), for
                   a time limit, limit (the setting that fired, such as attempt_limit_s) and limit_s,
                   and when the video encoder fell behind and dropped frames of a saved attempt,
                   dropped_frames (per camera, such as {"top": 3}): its video and its other columns
                   don't line up, so builds leave it out
    outcome        outcome (succeeded, failed or ignored), episode and operator. An outcome given after
                   its attempt closed (at rest after a time limit, or with the Reset after a fault)
                   carries the attempt it labels but no frame.
    label          reason (why the operator stepped in, such as missed_grasp) and operator
    control_taken  who (the caller of the page that took control), previous (who had it, or None),
                   implicit (true when a page's first command took it, not Take control) and operator
    link_lost      who, seconds (since the controlling page's last poll), on_link_loss and driving (whether
                   the policy drove, or the arm moved to the start pose to start it)
    page_hidden    who, on_link_loss and driving: the controlling page went out of sight, which loses
                   the link at once
    link_restored  who and seconds (how long the link was lost)
    check, pose    Run checks and Check pose: summary, fails, warns, problems (each result that
                   didn't pass: group, status and detail) and operator
    loop_rate      hz (measured over a second), target_hz (the task's fps) and slow (below 95% of it).
                   Logged when the loop falls behind and when it keeps up again
    recording_failed  error: saving or finishing an attempt's file failed (its attempt_end says saved
                   false), or the dataset couldn't be reopened after a saved attempt. Either way no new
                   attempt can be recorded in this run
    run_end        the last event of a run that ended with a quit: finished (local time,
                   YYYY-mm-ddTHH:MM:SS), attempts (opened) and episodes (saved)

`operator` (the name given with `so101 serve --operator`, or None) is on the events a person's
action causes: command, label, outcome, control_taken, check and pose. Events the robot causes on
its own, such as fault, link_lost and loop_rate, don't have it.

The last `KEEP_EVENTS` events stay in memory too, so the control page can read them with `since`
without reading the file.

Writing an event is best-effort: a write that fails, as on a full or unplugged data disk, prints one
warning on stderr per event type, and the page still sees the event. Python keeps the lines it
couldn't write in memory and writes them, in order, with the next flush that succeeds, so on a disk
that fills and then frees they land late; any still unwritten when the run closes are lost. A log
line never stops the control loop, which must keep holding the arm, and neither does a warning that
can't be printed because the terminal is gone.
"""

import json
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from enum import IntEnum

import numpy as np

from so_101.data import Run


class Phase(IntEnum):
    """`control.phase`: what the arm was doing in a frame. Named like the controller's states."""

    STARTING = 0
    AUTONOMOUS = 1
    PAUSED = 2
    HANDOVER = 3
    TELEOP = 4
    REWINDING = 5
    RETURNING = 6


OUTCOMES = ("succeeded", "failed", "ignored")
KEEP_EVENTS = 500  # recent events kept in memory for the control page


def jsonable(record: dict) -> dict:
    """`record` as JSON would give it back: a value JSON can't hold becomes its `str`, so the event
    log and the page's /events can always write it."""
    return json.loads(json.dumps(record, default=str))


def warn(message: str) -> None:
    """Print `message` on stderr. Once the terminal is closed, printing fails (EIO), and the warning is
    dropped: the control loop writes events inside a tick, which must not fail for it."""
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


class EventTail:
    """Numbers events from 1 and keeps the most recent ones in memory. The control loop appends;
    the web server's threads read with `since`."""

    def __init__(self) -> None:
        self.last_id = 0
        self._recent: deque[dict] = deque(maxlen=KEEP_EVENTS)
        self._lock = threading.Lock()

    def number(self, record: dict) -> dict:
        with self._lock:
            self.last_id += 1
            record = {"id": self.last_id, **record, "id": self.last_id}  # first, and never a caller's
            self._recent.append(record)
        return record

    def since(self, after: int, limit: int) -> list[dict]:
        """The events numbered above `after`, oldest first: at most the newest `limit` of them."""
        with self._lock:
            events = [record for record in self._recent if record["id"] > after]
        return events[-limit:]


def features(robot) -> dict[str, dict]:
    """The run dataset's columns: the robot's own, plus control columns that don't start with
    `observation.`, since LeRobot makes every `observation.*` column a policy input."""
    from lerobot.utils.feature_utils import hw_to_dataset_features

    joints = list(robot.action_features)
    return {
        **hw_to_dataset_features(robot.observation_features, "observation", True),
        **hw_to_dataset_features(robot.action_features, "action", True),
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        "control.phase": {"dtype": "int8", "shape": (1,), "names": None},
        "control.policy_action": {"dtype": "float32", "shape": (len(joints),), "names": joints},
        "control.leader": {"dtype": "float32", "shape": (len(joints),), "names": joints},
    }


class Recorder:
    def __init__(
        self,
        folder: Run,
        robot,
        *,
        fps: int,
        task: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create `dataset/` and `events.jsonl` in the run's `folder`. `robot` gives the columns; `task` is
        the prompt every frame is labelled with."""
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.folder = folder
        self.run_dir = folder.path
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.task = task
        self.clock = clock
        self.started = clock()
        self.joints = list(robot.action_features)
        self._writing = {
            "streaming_encoding": True,
            "encoder_threads": 2,
            "rgb_encoder": RGBEncoderConfig(preset=10),  # SVT-AV1 has no preset above 10; the default of 12 warns
        }
        self.dataset = LeRobotDataset.create(
            folder.repo_id, fps, features(robot), root=folder.dataset_root, robot_type=robot.robot_type,
            **self._writing,
        )
        self._events = folder.events_path.open("a", encoding="utf-8")
        self._unwritten: set[str] = set()  # the event types whose write has failed, warned about once each
        self.tail = EventTail()
        self.attempt: int | None = None
        self.frames = 0
        self._phases: Counter[Phase] = Counter()
        self.saved = 0
        self.failure: str | None = None  # why no new attempt can be recorded, once finishing the files failed
        self.dropped: dict[str, int] = {}  # the video frames, per camera, the attempt saved last dropped

    # --- Events ------------------------------------------------------------------------------------

    def event(self, type_: str, /, **fields) -> None:
        """Append one event and flush it to disk. The event's own fields (id, type, t, wall, and
        attempt and frame during an attempt) win over a field of the same name in `fields`. A write
        that fails warns once per event type, and never raises (see the module's docstring)."""
        own = {"type": type_, "t": round(self.clock() - self.started, 3), "wall": round(time.time(), 3)}
        if self.attempt is not None:
            own |= {"attempt": self.attempt, "frame": self.frames}
        try:
            record = self.tail.number(jsonable({**own, **fields, **own}))  # own fields first, and winning
            self._events.write(json.dumps(record) + "\n")
            self._events.flush()
        except (OSError, TypeError, ValueError, RecursionError) as error:  # ValueError: the file is already closed
            if type_ not in self._unwritten:
                self._unwritten.add(type_)
                warn(f"Couldn't write a {type_} event to {self.folder.events_path}: {error}. The arm and "
                     f"the control loop carry on; {type_} events are written once the disk takes them again, "
                     "and lost if it doesn't before so101 serve exits. Is the disk full or unplugged?")

    # --- Attempts ----------------------------------------------------------------------------------

    def start_attempt(self, attempt: int) -> None:
        self.discard_attempt("superseded")  # a fault's own discard may have failed: its frames can't join this one
        self.attempt = attempt
        self.frames = 0
        self._phases.clear()
        self.event("attempt_start")

    def add_frame(
        self,
        observation: dict,
        action: dict[str, float] | None,
        phase: Phase,
        *,
        intervention: bool = False,
        policy_action: dict[str, float] | None = None,
        leader: dict[str, float] | None = None,
    ) -> None:
        """One frame of the open attempt. `action` is the goal sent this tick, after the step cap.
        `intervention` is set in Teleop once the policy has driven in the attempt (an action of its
        reached the arm): a human stepping in on the policy. `policy_action` is the policy's raw action
        on a tick one reached the arm, and `leader` the leader's pose, read every tick it's connected.
        Either is NaN when missing: an Autonomous tick the policy sent none, as while waiting for a
        cloud GPU, or a tick the leader wasn't read."""
        from lerobot.utils.feature_utils import build_dataset_frame

        if self.attempt is None:
            return
        features = self.dataset.meta.features
        frame = {
            **build_dataset_frame(features, observation, prefix="observation"),
            "action": self._joints(action),
            "intervention": np.array([intervention], dtype=bool),
            "control.phase": np.array([phase], dtype=np.int8),
            "control.policy_action": self._joints(policy_action),
            "control.leader": self._joints(leader),
            "task": self.task,
        }
        self.dataset.add_frame(frame)
        self.frames += 1
        self._phases[phase] += 1

    def _joints(self, values: dict[str, float] | None) -> np.ndarray:
        if values is None:
            return np.full(len(self.joints), np.nan, dtype=np.float32)
        return np.array([values[joint] for joint in self.joints], dtype=np.float32)

    def save_attempt(self, ended_by: str | None, outcome: str | None = None, *, operator: str | None = None,
                     **details) -> int | None:
        """Save the open attempt as the next episode. Returns its episode index. `details` go into
        the `attempt_end` event; `operator`, who gave the outcome, into the `outcome` event.

        The attempt counts as saved, and `attempt_end` is written, only once its data file is finished
        (`finalize`): LeRobot writes a data file's footer and the episode index only then, so a run
        killed outright (SIGKILL, power loss) before it would lose every attempt saved so far, and an
        unfinished file can't be read. A save or finalize that fails, as on a full or unplugged disk,
        loses the attempt and sets `failure`: the dataset may be left half-written, so no new attempt
        is recorded into it. It returns None and doesn't raise: the arm is at rest or holds either
        way. Reopening the dataset for the next attempt can fail too (`_reopen`), which keeps this
        attempt saved. An attempt with no frames (a fault on the tick it opened) is discarded, and
        None returned."""
        if self.attempt is None:
            return None
        if self.frames == 0:
            self.discard_attempt(ended_by, **details)
            return None
        episode = self.saved
        try:
            self.dropped = self._dropped_frames()  # before the save, which ends the encoder's episode
            self.dataset.save_episode()
            self.dataset.finalize()
        except Exception as error:
            attempt = self.attempt
            self._end(ended_by, None, **details)
            self.attempt = None
            self.failure = (f"Saving attempt {attempt} failed ({type(error).__name__}: {error}), so it isn't in the "
                            "run's dataset and no new attempt can be recorded. Check the disk the run is recorded "
                            "on, then restart so101 serve to start a new run.")
            self.event("recording_failed", error=self.failure)
            return None
        self.saved += 1
        self._end(ended_by, episode, **details, **({"dropped_frames": self.dropped} if self.dropped else {}))
        if outcome is not None:
            self.event("outcome", outcome=outcome, episode=episode, operator=operator)
        self.attempt = None
        self._copy_meta()
        self._reopen()
        return episode

    def _copy_meta(self) -> None:
        """Copy the finished dataset's shared metadata beside it (`data.Run.save_meta_copy`), which a
        kill during the next save can tear: readers fall back to the copy. A copy that fails warns and
        leaves the last one in place; the attempt is saved either way."""
        try:
            self.folder.save_meta_copy(self.saved)
        except Exception as error:
            warn(f"Couldn't copy the run's dataset metadata after episode {self.saved - 1} ({type(error).__name__}: "
                 f"{error}). The episode is saved; a kill during the next save could cost the run's earlier "
                 "attempts until a later copy succeeds.")

    def _dropped_frames(self) -> dict[str, int]:
        """The video frames LeRobot's streaming encoder dropped in the open attempt, per camera, as when
        it fell behind and its queue was full. LeRobot keeps the counts in the encoder's private
        `_dropped_frames` and only logs them. test_controller's
        `test_frames_the_video_encoder_dropped_are_on_the_saved_attempts_attempt_end` fails if they move."""
        encoder = getattr(getattr(self.dataset, "writer", None), "_streaming_encoder", None)
        counts = getattr(encoder, "_dropped_frames", None) or {}
        return {key.removeprefix("observation.images."): int(n) for key, n in counts.items() if n}

    def _reopen(self) -> None:
        """Reopen the finished dataset for the next attempt. A failure keeps the episode saved, and
        refuses new attempts."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        try:
            self.dataset = LeRobotDataset.resume(self.folder.repo_id, root=self.folder.dataset_root, **self._writing)
        except Exception as error:  # the episode was saved; only the next attempt can't be recorded
            self.failure = (f"The run's dataset couldn't be reopened after episode {self.saved - 1} "
                            f"({type(error).__name__}: {error}), so no new attempt can be recorded. Restart "
                            "so101 serve to start a new run.")
            self.event("recording_failed", error=self.failure)

    def discard_attempt(self, ended_by: str | None, **details) -> None:
        """Drop the open attempt's frames, as when the control loop stops with an attempt open."""
        if self.attempt is None:
            return
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        self._end(ended_by, None, **details)
        self.attempt = None

    def _end(self, ended_by: str | None, episode: int | None, **details) -> None:
        seconds = {phase.name.lower(): round(n / self.fps, 2) for phase, n in sorted(self._phases.items())}
        self.event("attempt_end", ended_by=ended_by, saved=episode is not None, episode=episode,
                   frames=self.frames, seconds=seconds, **details)

    def close(self, run_end: dict | None = None) -> None:
        """Finish the dataset's files. Call once, at exit. After a failed save it tries again, which
        may fail the same way. With `run_end` (the run ended cleanly), the event log ends with a
        `run_end` event: its fields, plus when the run finished and how many episodes it saved,
        written once the dataset is finished. A second call does nothing. Closing the event log never
        raises: on a disk still full, the lines it couldn't write are lost, with a warning."""
        if self._events.closed:
            return
        try:
            if self.attempt is not None:
                self.discard_attempt("crash")  # a quit always closes its attempt first
            self.dataset.finalize()
            if run_end is not None:
                self.event("run_end", **run_end, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), episodes=self.saved)
        finally:
            try:
                self._events.close()  # closed even when its last flush fails
            except (OSError, ValueError) as error:
                warn(f"Couldn't finish writing {self.folder.events_path}: {error}. The events it couldn't "
                     "write before so101 serve exited are lost; the page showed them.")

