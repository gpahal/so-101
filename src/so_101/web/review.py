"""The reviewer: the web page `so101 review` (demos) and `so101 review --runs` (`serve` and `run`
attempts) open, each showing only its kind, with every decision taken on it.

It runs on the shared page server (`web/server.py`), so it has the control page's never-public
checks and bind rule, `[control] bind`, on its own port, `[control] review_port`, so a review can
run while `serve` does. It reads the recordings and writes only what the terminal prompts write:
each decision goes through `review.save` (edits.toml) or `review_runs.apply` (the run's
labels.toml), so both ways write identical files.

    GET  /                 the page (review.html beside this file)
    GET  /items            {"kind": "demos" or "runs", "task", "warning", "notes", "items": [...]}:
                           what this review covers, oldest first, each {"id", "title", "seconds",
                           "now" (its decision so far, in words), "reviewed"}. An id is SESSION/N or
                           RUN/N, as `--episode` takes them. "notes" are the attempts left out
                           because their episode isn't in the run's episode index, as after a kill
                           during its save
    GET  /item/<id>        everything the page draws for one item: "fps", "frames", "seconds",
                           "cameras", "joints" ({"names", "values"}, one list per joint), "bands"
                           (spans of each `control.skill` or `control.driver` column a recording has,
                           each span with its "value" and "label", the skill's or driver's name),
                           and for a demo "suggestion" ({"keep", "delete", "hint"}), "edit" and
                           "min_seconds"; for an attempt "phases", "interruptions", "events" (each
                           with "type", "frame" and "text", and "start_frame" and "end_frame" when
                           it marks a range), "outcome", "recorded_outcome" and "labelled"
    GET  /frames/<id>/<camera>?from=A&to=B
                           frames A up to B (stop not included, at most FRAMES_PER_REQUEST) of the
                           camera, as JPEGs up to WIDTH px wide, packed one after another, each
                           behind its length as 4 bytes, big-endian, in an application/octet-stream
                           body. Only the frames asked for are decoded, and the server keeps the
                           last KEEP_BYTES of JPEGs (`FrameCache`). When the video stops decoding
                           partway, as one cut short does, the frames before that are sent; a
                           request that decodes none is a 500 that says why
    POST /decision         for a demo {"item", "edit": "accept" | "keep" | "trim" | "delete",
                           "keep"?: [start_s, end_s], "reason"?: ...}; for an attempt {"item",
                           "outcome"} or {"item", "interruption": its frame, "suspect": [start,
                           stop] or [] for none} or {"item", "interruption", "auto": true}. Each
                           names its item, so a tab may decide on its item whichever another tab
                           opened since. Refused with 404 for an item the review doesn't cover, and
                           with 400 for anything else it doesn't take. Replies {"saved", "item",
                           "entry"}: what was written, in words, the item and its list entry as they
                           are now

A request that finds edits.toml or a labels.toml it can't read, as one changed by hand mid-review, is
refused with 409 and the file's error; one whose recording can't be read, a 500 that says why. Errors
close the connection, as every page server's do (`web.server.Handler._send`).
"""

import contextlib
import math
import re
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from http import HTTPStatus
from importlib import resources
from urllib.parse import SplitResult, parse_qs, unquote

import numpy as np

from so_101 import data, review, review_runs, settings
from so_101.build import MIN_EPISODE_S
from so_101.control import analysis
from so_101.control.recorder import Phase
from so_101.policy.base import DRIVERS
from so_101.dataset_io import column, episode_frames
from so_101.settings import ConfigError, Settings
from so_101.tasks import Task
from so_101.web import server as web

SITE = web.Site(page="review page", command="so101 review", posts="Decisions", guards="your recordings",
                remote="other device", thread="review-page")
WIDTH = 480  # px: enough to judge a grasp, small enough that a 15 s episode of two cameras is about 25 MB
JPEG_QUALITY = 75
FRAMES_PER_REQUEST = 150  # 5 s at 30 fps, so the page shows the start while the rest loads
# JPEGs the server keeps, for a reload or a second tab: about three 15 s episodes of two cameras.
KEEP_BYTES = 80_000_000
# The slowest link a frames body is given time for: 1 Mbit/s, so a 150-frame chunk of about 4 MB
# may take 32 s to send, where any other reply has the connection's 10 s (`web.server.Handler.timeout`).
SLOWEST_LINK_BYTES_PER_S = 125_000
# Columns a recording may have that split it into spans worth a band of their own on the timeline.
BAND_COLUMNS = ("control.skill", "control.driver")
# Events the timeline draws in other ways (`state` as the phases, the attempt's own start and end), or
# that say nothing about the arm's motion. Every other type is drawn, so a new one needs no change here.
HIDDEN_EVENTS = {"state", "command", "attempt_start", "attempt_end", "loop_rate", "control_taken", "link_restored",
                 "check", "pose"}
EVENT_SKIP = {"id", "type", "t", "wall", "attempt", "frame", "operator"}  # fields an event's text leaves out


class Refused(ValueError):
    """A decision the page can't make: the message says why."""


def jpeg(frame: np.ndarray) -> bytes:
    """An RGB frame as a JPEG at most WIDTH px wide."""
    import cv2

    height, width = frame.shape[:2]
    if width > WIDTH:
        frame = cv2.resize(frame, (WIDTH, max(1, round(height * WIDTH / width))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("OpenCV couldn't encode the frame as JPEG.")
    return encoded.tobytes()


def pack(jpegs: list[bytes]) -> bytes:
    """JPEGs one after another, each behind its length as 4 bytes, big-endian."""
    return b"".join(len(j).to_bytes(4, "big") + j for j in jpegs)


def unpack(body: bytes) -> list[bytes]:
    """The JPEGs `pack` packed."""
    out, at = [], 0
    while at < len(body):
        n = int.from_bytes(body[at : at + 4], "big")
        out.append(body[at + 4 : at + 4 + n])
        at += 4 + n
    return out


def send_timeout(length: int) -> float:
    """Seconds a body of `length` bytes may take to send: at SLOWEST_LINK_BYTES_PER_S, and never less
    than any reply has."""
    return max(float(web.Handler.timeout), length / SLOWEST_LINK_BYTES_PER_S)


Decoded = tuple[list[bytes], str | None]  # the JPEGs that decoded, and why the rest didn't, or None


class FrameCache:
    """JPEGs by key, decoded once, the newest up to `keep_bytes` of them. Two requests for the same key
    wait for one decode. A decode that failed partway isn't kept, so asking again decodes again."""

    def __init__(self, keep_bytes: int) -> None:
        self.keep_bytes = keep_bytes
        self._done: OrderedDict[tuple, list[bytes]] = OrderedDict()
        self._lock = threading.Lock()
        self._decoding: dict[tuple, threading.Lock] = {}

    def get(self, key: tuple, decode: Callable[[], Decoded]) -> Decoded:
        with self._lock:
            if key in self._done:
                self._done.move_to_end(key)
                return self._done[key], None
            one = self._decoding.setdefault(key, threading.Lock())
        with one:
            with self._lock:
                if key in self._done:
                    return self._done[key], None
            try:
                frames, error = decode()
            finally:
                with self._lock:
                    self._decoding.pop(key, None)
            if error is None:
                with self._lock:
                    self._done[key] = frames
                    while len(self._done) > 1 and self._bytes() > self.keep_bytes:
                        self._done.popitem(last=False)
        return frames, error

    def _bytes(self) -> int:
        return sum(len(j) for jpegs in self._done.values() for j in jpegs)


def _joints(meta, table) -> dict:
    names = meta.features["observation.state"]["names"]
    values = np.round(column(table, "observation.state").T.astype(float), 2)
    return {"names": list(names), "values": [[v if math.isfinite(v) else None for v in row] for row in values.tolist()]}


NO_SKILL = -1  # control.skill without skills


def _bands(table, skills: list[str] = ()) -> list[dict]:
    """The spans of equal value of each BAND_COLUMNS column the table has, each labelled with the
    name of the driver (`policy.base.DRIVERS`) or of the skill (`skills`, run.json's policy.skills, by
    index). A skill column that is NO_SKILL throughout, as every run without skills records, has no
    band."""
    bands = []
    for name in BAND_COLUMNS:
        if name not in table.columns:
            continue
        values = [_scalar(v) for v in table[name]]
        if name == "control.skill" and all(v == NO_SKILL for v in values):
            continue
        names = DRIVERS if name == "control.driver" else {NO_SKILL: "no skill", **dict(enumerate(skills))}
        spans, start = [], 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                value = values[start]
                spans.append({"value": value, "label": str(names.get(value, value)), "start": start, "stop": i})
                start = i
        bands.append({"name": name.removeprefix("control."), "spans": spans})
    return bands


def _skill_names(run: data.Run) -> list[str]:
    """The skills a run's control.skill indexes, from its run.json's policy.skills."""
    try:
        recorded = (run.manifest().get("policy") or {}).get("skills") or []
    except (OSError, ValueError, AttributeError):  # run.json unreadable: the bands show the indices
        return []
    return [str(r.get("name", n)) if isinstance(r, dict) else str(r) for n, r in enumerate(recorded)]


def notes_for(notes: list[str], number: int) -> list[str]:
    """The notes that name attempt `number`, as a whole number: attempt 1's, never attempt 12's."""
    return [note for note in notes if re.search(rf"\battempt {number}\b", note)]


def _scalar(value):
    """A cell as JSON holds it: a vector's first value, a number or a string."""
    if isinstance(value, np.ndarray | list | tuple):
        value = np.ravel(value)[0] if len(np.ravel(value)) else None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if isinstance(value, int | float | str | bool) or value is None else str(value)


def _event(e: dict) -> dict:
    """An event for the timeline: its type and frame, the range it marks, and its fields in words."""
    parts = []
    for key, value in e.items():
        if key in EVENT_SKIP or value is None:
            continue
        text = value if isinstance(value, str) else repr(value) if not isinstance(value, dict | list) else str(value)
        if len(text) <= 60:
            parts.append(f"{key} {text}")
    out = {"type": e["type"], "frame": e["frame"], "text": ", ".join(parts)[:200]}
    if isinstance(e.get("start_frame"), int) and isinstance(e.get("end_frame"), int):
        out |= {"start_frame": e["start_frame"], "end_frame": e["end_frame"]}
    return out


def _shown(e: dict, number: int) -> bool:
    if e.get("attempt") != number or not isinstance(e.get("frame"), int) or isinstance(e.get("frame"), bool):
        return False
    if e["type"] in HIDDEN_EVENTS:
        return False
    return e["type"] != "supervisor" or e.get("kind") == "decision"


def _cameras(meta) -> list[str]:
    return [key.removeprefix("observation.images.") for key in meta.video_keys]


class Demos:
    """`so101 review`: the demos to review, and their decisions, written by `review.save`."""

    kind = "demos"
    notes: tuple[str, ...] = ()  # a demo is never left out

    def __init__(self, s: Settings, task: Task, only: str | None, everything: bool) -> None:
        self.s, self.task = s, task
        self.demos = {demo.id: demo for demo in review.demos(s, task, only, everything)}
        self._suggestions: dict[str, review.Suggestion] = {}

    def __bool__(self) -> bool:
        return bool(self.demos)

    def empty(self, only: str | None, everything: bool) -> str:
        return review.nothing_left(self.task, only, everything)

    def video(self, id: str) -> tuple:
        demo = self.demos[id]
        return demo.session.meta, demo.episode

    def suggestion(self, id: str, table=None) -> review.Suggestion:
        if id not in self._suggestions:
            demo = self.demos[id]
            table = review.read(demo) if table is None else table
            self._suggestions[id] = review.suggest(demo, column(table, "action"), demo.session.meta.fps)
        return self._suggestions[id]

    def entries(self) -> list[dict]:
        edits = data.load_edits(self.s, self.task)
        return [self._entry(demo, edits) for demo in self.demos.values()]

    def entry(self, id: str) -> dict:
        return self._entry(self.demos[id], data.load_edits(self.s, self.task))

    def _entry(self, demo: review.Demo, edits: dict) -> dict:
        meta = demo.session.meta
        frames = int(meta.episodes[demo.episode]["length"])
        return {"id": demo.id, "title": f"session {demo.session.name}, episode {demo.episode}",
                "seconds": round(frames / meta.fps, 2), "now": review.describe(edits.get(demo.key)),
                "reviewed": demo.key in edits}

    def detail(self, id: str) -> dict:
        demo = self.demos[id]
        meta = demo.session.meta
        table = review.read(demo)
        suggestion = self.suggestion(id, table)
        edit = data.load_edits(self.s, self.task).get(demo.key)
        return {
            **self._entry(demo, data.load_edits(self.s, self.task)), "kind": self.kind,
            "fps": meta.fps, "frames": len(table), "cameras": _cameras(meta), "joints": _joints(meta, table),
            "bands": _bands(table), "notes": [], "min_seconds": MIN_EPISODE_S,
            "suggestion": {"keep": suggestion.edit.keep, "delete": suggestion.edit.delete, "hint": suggestion.hint},
            "edit": None if edit is None else {"keep": edit.keep, "delete": edit.delete},
        }

    def decide(self, id: str, body: dict) -> str:
        """Write the decision `body` asks for, and return it in words."""
        suggestion = self.suggestion(id)
        choice = body.get("edit")
        allowed = {"accept": set(), "keep": set(), "trim": {"keep"}, "delete": {"reason"}}
        if not isinstance(choice, str) or choice not in allowed or not set(body) - {"item", "edit"} <= allowed[choice]:
            raise Refused('A demo takes {"edit": "accept"}, {"edit": "keep"}, {"edit": "trim", "keep": [START, END]} '
                          'or {"edit": "delete", "reason": ...}.')
        if choice == "accept":
            edit = suggestion.edit
        elif choice == "keep":
            edit = review.keep_all(suggestion)
        elif choice == "delete":
            reason = body.get("reason", "")
            if not isinstance(reason, str):
                raise Refused("A reason is text.")
            edit = review.delete(suggestion, reason)
        else:
            keep = body.get("keep")
            if not (isinstance(keep, list) and len(keep) == 2 and all(_number(v) for v in keep)):
                raise Refused("Keep takes [START, END], in seconds.")
            edit = review.keep_range(suggestion, float(keep[0]), float(keep[1]))
            if isinstance(edit, str):
                raise Refused(edit)
        review.save(self.s, self.task, edit)
        return review.describe(edit)


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class Runs:
    """`so101 review --runs`: the attempts to review, and their decisions, written by `review_runs.apply`."""

    kind = "runs"

    def __init__(self, s: Settings, task: Task, only: str | None, everything: bool) -> None:
        """Prints each run's notes, as `review_runs.collect` does, and each attempt left out."""
        self.s, self.task = s, task
        self.todo = review_runs.collect(s, task, only, everything)
        self.notes: list[str] = []  # the attempts left out, and why
        self.attempts = {}
        for run, number in self.todo.attempts:
            if (why := self._unreadable(run, number)) is not None:
                self.notes.append(why)
                print(f"note: {why}")
                continue
            self.attempts[f"{run.name}/{number}"] = (run, number)

    def _unreadable(self, run: data.Run, number: int) -> str | None:
        """Why the attempt's frames can't be shown, or None. A kill after its data file and
        attempt_end were written, before its episode reached the metadata, leaves an episode the
        analysis finds and the episode index doesn't list, so no video can be found for it."""
        _, attempt = review_runs.find(self.todo.records[run.name], analysis.load_labels(run), self.todo.blame_s,
                                      number)
        meta = run.meta
        indexed = set() if meta is None else {int(e) for e in meta.episodes["episode_index"]}
        if attempt.episode in indexed:
            return None
        return (f"run {run.name} attempt {number}: episode {attempt.episode} isn't in the dataset's episode index, as "
                "after a kill during its save, so it's left out")

    def __bool__(self) -> bool:
        return bool(self.attempts)

    def empty(self, only: str | None, everything: bool) -> str:
        return review_runs.nothing_left(self.s, self.task, only, everything)

    def _find(self, id: str) -> tuple[data.Run, analysis.Labels, analysis.RunAnalysis, analysis.Attempt]:
        run, number = self.attempts[id]
        labels = analysis.load_labels(run)
        a, attempt = review_runs.find(self.todo.records[run.name], labels, self.todo.blame_s, number)
        return run, labels, a, attempt

    def video(self, id: str) -> tuple:
        run, _, _, attempt = self._find(id)
        return run.meta, attempt.episode

    def entries(self) -> list[dict]:
        return [self.entry(id) for id in self.attempts]

    def entry(self, id: str) -> dict:
        run, labels, a, attempt = self._find(id)
        return {"id": id, "title": f"run {run.name}, attempt {attempt.number}",
                "seconds": round(attempt.frames / a.fps, 2), "now": review_runs.describe_outcome(attempt),
                "reviewed": review_runs.reviewed(attempt, labels)}

    def detail(self, id: str) -> dict:
        from so_101.dataset_io import episode_table

        run, labels, a, attempt = self._find(id)
        meta = run.meta
        table = episode_table(meta, attempt.episode)
        events = self.todo.records[run.name].events
        return {
            **self.entry(id), "kind": self.kind, "fps": a.fps, "frames": attempt.frames, "cameras": _cameras(meta),
            "joints": _joints(meta, table), "bands": _bands(table, _skill_names(run)),
            "notes": notes_for(a.notes, attempt.number),
            "ended_by": attempt.ended_by,
            "phases": [{"phase": Phase(s.phase).name.lower(), "start": s.start, "stop": s.stop}
                       for s in attempt.segments],
            "interruptions": [
                {"frame": i.frame, "start": i.segment.start, "command": i.command, "reason": i.reason, "then": i.then,
                 "suspect": i.suspect, "source": i.suspect_source,
                 "text": review_runs.describe_suspect(i, a.fps)}
                for i in attempt.interruptions
            ],
            "events": [_event(e) for e in events if _shown(e, attempt.number)],
            "outcome": attempt.outcome or analysis.UNLABELLED,
            "recorded_outcome": attempt.recorded_outcome or analysis.UNLABELLED,
            "labelled": attempt.number in labels.outcomes,
            "outcomes": list(review_runs.OUTCOME_CHOICES),
        }

    def decide(self, id: str, body: dict) -> str:
        """Write the decision `body` asks for, and return it in words."""
        run, _, a, attempt = self._find(id)
        keys = set(body) - {"item"}
        if keys == {"outcome"}:
            if body["outcome"] not in review_runs.OUTCOME_CHOICES:
                raise Refused(f"An outcome is one of {', '.join(review_runs.OUTCOME_CHOICES)}.")
            review_runs.apply(run, body["outcome"], attempt, None)
            _, _, a, attempt = self._find(id)
            return f"outcome {review_runs.describe_outcome(attempt)}"
        if keys not in ({"interruption", "suspect"}, {"interruption", "auto"}):
            raise Refused('An attempt takes {"outcome": ...}, {"interruption": FRAME, "suspect": [START, STOP] or []} '
                          'or {"interruption": FRAME, "auto": true}.')
        i = next((i for i in attempt.interruptions if _whole(body["interruption"]) and i.frame == body["interruption"]),
                 None)
        if i is None:
            raise Refused(f"Attempt {attempt.number} has no interruption at frame {body['interruption']!r}.")
        if "auto" in keys:
            if body["auto"] is not True:
                raise Refused('Send "auto": true to go back to the automatic suspect frames.')
            decision = review_runs.Suspect(None, auto=True)
        elif body["suspect"] == []:
            decision = review_runs.Suspect(None)
        elif isinstance(body["suspect"], list) and len(body["suspect"]) == 2 and all(map(_whole, body["suspect"])):
            decision = review_runs.suspect_range(*body["suspect"], attempt.frames)
            if isinstance(decision, str):
                raise Refused(decision.removeprefix("!"))
        else:
            raise Refused("Suspect takes [START, STOP], whole frames with STOP not included, or [] for none.")
        review_runs.apply(run, decision, attempt, i)
        _, _, a, attempt = self._find(id)
        changed = next(x for x in attempt.interruptions if x.frame == i.frame)
        return f"suspect {review_runs.describe_suspect(changed, a.fps)}"


def page() -> bytes:
    return resources.files("so_101.web").joinpath("review.html").read_bytes()


class ReviewServer(web.PageServer):
    def __init__(self, bind: web.Bind, port: int, *, source: Demos | Runs, task: str,
                 routes: Callable[[], str | None]) -> None:
        """Binds now, so a port in use fails at once. `source` is what's reviewed."""
        super().__init__(bind, port, Handler, site=SITE, routes=routes)
        self.source = source
        self.task = task
        self.frames = FrameCache(KEEP_BYTES)
        self.deciding = threading.Lock()  # one decision at a time, as each rewrites a whole file

    def decoded(self, id: str, camera: str, start: int, stop: int) -> Decoded:
        """Frames `start` up to `stop` of the item's camera, as JPEGs: those that decode, and why the
        rest didn't (a video cut short or missing a frame), or None."""
        import av

        def decode() -> Decoded:
            meta, episode = self.source.video(id)
            jpegs = []
            try:
                for _, frame in episode_frames(meta, episode, f"observation.images.{camera}", start, stop):
                    jpegs.append(jpeg(frame))
            # A video file missing, cut short or unreadable. PyAV's EOFError and UndefinedError are
            # neither an OSError nor a ValueError.
            except (OSError, ValueError, av.error.FFmpegError) as error:
                return jpegs, str(error) or type(error).__name__
            return jpegs, None

        return self.frames.get((id, camera, start, stop), decode)


class Handler(web.Handler):
    server: ReviewServer
    POSTS = ("/decision",)

    def get(self, url: SplitResult) -> None:
        path = unquote(url.path)
        if path == "/":
            self._send(HTTPStatus.OK, page(), "text/html; charset=utf-8")
        elif path == "/items":
            self._answer(lambda: (HTTPStatus.OK, self._items()))
        elif path.startswith("/item/"):
            id = path.removeprefix("/item/")
            self._answer(lambda: (HTTPStatus.OK, self.server.source.detail(id)) if id in self._ids() else self._no(id))
        elif path.startswith("/frames/"):
            self._frames(path.removeprefix("/frames/"), parse_qs(url.query))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"No {url.path} here."})

    def _items(self) -> dict:
        source = self.server.source
        return {"kind": source.kind, "task": self.server.task, "warning": self.server.warning,
                "notes": list(source.notes), "items": source.entries()}

    def _ids(self):
        source = self.server.source
        return source.demos if isinstance(source, Demos) else source.attempts

    @staticmethod
    def _no(id) -> tuple[HTTPStatus, dict]:
        return HTTPStatus.NOT_FOUND, {"error": f"No item {id!r} in this review."}

    @staticmethod
    def _failure(error: Exception) -> tuple[HTTPStatus, dict]:
        """The reply for an error a request met: 409 for an edits.toml or labels.toml that can't be
        read, which a person fixes, and 500 for a recording that can't."""
        if isinstance(error, ConfigError):
            return HTTPStatus.CONFLICT, {"error": str(error)}
        why = repr(error) if isinstance(error, LookupError) else str(error)  # a KeyError's str is only its key
        return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Can't read the recording: {why}"}

    def _answer(self, make: Callable[[], tuple[HTTPStatus, dict]]) -> None:
        """Reply with what `make` returns, or with the error that stopped it (`_failure`)."""
        try:
            status, body = make()
        except (ConfigError, OSError, LookupError, ValueError) as error:  # ValueError: a Parquet file it can't read
            status, body = self._failure(error)
        self._json(status, body)

    def _frames(self, rest: str, query: dict) -> None:
        id, _, camera = rest.rpartition("/")
        if id not in self._ids():
            self._json(*self._no(id))
            return
        try:
            result = self._chunk(id, camera, query)
        except (ConfigError, OSError, LookupError, ValueError) as error:
            result = self._failure(error)
        if isinstance(result, bytes):
            self._send(HTTPStatus.OK, result, "application/octet-stream", send_s=send_timeout(len(result)))
        else:
            self._json(*result)

    def _chunk(self, id: str, camera: str, query: dict) -> bytes | tuple[HTTPStatus, dict]:
        """The packed JPEGs a /frames request asks for, or the error reply."""
        meta, episode = self.server.source.video(id)
        if camera not in _cameras(meta):
            return HTTPStatus.NOT_FOUND, {"error": f"No camera {camera!r}: only {', '.join(_cameras(meta))}."}
        length = int(meta.episodes[episode]["length"])
        try:
            start = int(query.get("from", ["0"])[0])
            stop = min(int(query.get("to", [str(length)])[0]), length, start + FRAMES_PER_REQUEST)
            if not 0 <= start < stop:
                raise ValueError("empty")
        except ValueError:
            return HTTPStatus.BAD_REQUEST, {"error": f"from and to are frames, 0 <= from < to, of {length}."}
        jpegs, error = self.server.decoded(id, camera, start, stop)
        if not jpegs:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": f"Can't read {camera}'s video from frame {start}: {error or 'no frames decoded'}"}
        return pack(jpegs)

    def post(self, url: SplitResult) -> None:
        try:
            body = self._read_json()
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": 'Send a JSON object, like {"item": "...", "edit": "accept"}.'})
            return
        id = body.get("item")
        if not isinstance(id, str) or id not in self._ids():
            self._json(*self._no(id))
            return
        source = self.server.source

        def decide() -> tuple[HTTPStatus, dict]:
            try:
                with self.server.deciding:
                    saved = source.decide(id, body)
            except Refused as refused:
                return HTTPStatus.BAD_REQUEST, {"error": str(refused)}
            with contextlib.suppress(OSError, ValueError):
                print(f"  {id}: saved {saved}", flush=True)
            return HTTPStatus.OK, {"saved": saved, "item": source.detail(id), "entry": source.entry(id)}

        self._answer(decide)


def _port_problem(s: Settings) -> str | None:
    """What's wrong with `[control] review_port`, or None. `check` reports it too (`check.check_control`)."""
    port = s.control.review_port
    if not _whole(port) or not 1 <= port <= 65535:
        return (f"[control] review_port in {s.path} is {port!r}: set it to a whole number from 1 to 65535, or delete "
                f"the line for the default {settings.CONTROL_SETTINGS['review_port'].default}.")
    return None


def open_page(s: Settings, task: Task, source: Demos | Runs, only: str | None, everything: bool) -> None:
    """Serve the reviewer for `source` until Ctrl+C, printing its address and each decision."""
    if not source:
        print(source.empty(only, everything))
        return
    if problem := _port_problem(s):
        sys.exit(problem)
    port = s.control.review_port
    try:
        bind = web.resolve_bind(s.control.bind, port, SITE)
    except ValueError as error:
        sys.exit(str(error))
    try:
        server = ReviewServer(bind, port, source=source, task=task.name,
                              routes=lambda: web.route_warning(bind, port, SITE))
    except OSError as error:
        sys.exit(f"Can't serve the review page on {bind.address}:{port}: {error}. Is another `so101 review` running? "
                 "Stop it, or set another review_port under [control] in so101.toml.")
    try:
        for warning in bind.warnings:
            print(warning, flush=True)
        server.start()
        what = "demos" if source.kind == "demos" else "attempts"
        count = len(source.entries())
        print(f"Reviewing {count} {what if count != 1 else what[:-1]} of {task.name}. Decisions are saved to "
              + (f"{data.edits_path(s, task)}" if source.kind == "demos" else f"each run's {analysis.LABELS_FILE}")
              + " as you make them; the recordings are never changed.")
        print(f"\nReview page: {server.url}", flush=True)
        if server.name_url:
            print(f"  or by its MagicDNS name: {server.name_url}", flush=True)
        print("Ctrl+C stops.", flush=True)
        threading.Event().wait()
    except KeyboardInterrupt:
        print()
    finally:
        server.stop()
    if source.kind == "demos":
        print(f"Stopped. Run `uv run so101 build {task.name}` to apply the edits.")
    else:
        print(f"Stopped. `uv run so101 runs {task.name}` shows the runs with these labels.")
