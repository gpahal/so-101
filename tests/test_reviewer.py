"""The reviewer (`so101 review` and `review --runs` on a web page) over real HTTP: a `ReviewServer` on
127.0.0.1, port 0, over a demo session and a `serve` run recorded in a temporary data folder.

The demo is a small LeRobot dataset whose cameras show a different grey in each frame, so a frame's
picture says which frame and camera it is. The run is recorded through the controller harness in
test_controller, so it has the events and columns a real `so101 serve` run has. Each decision is
made twice, on the page over one copy of the recordings and at the terminal prompts over another,
and the two copies' edits.toml or labels.toml must match byte for byte.
"""

import contextlib
import dataclasses
import http.client
import io
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fakes import FakeClock, make_follower
from test_build import TASK
from test_controller import FPS, GRIPPER_ACTIONS, REST_STEPS, Rig
from test_serve_preflight import ARMS, ConfigTest

from so_101 import data, settings
from so_101.check import check_control
from so_101.control import analysis
from so_101.control.recorder import features
from so_101.policy.base import DRIVERS
from so_101.report import Report
from so_101.settings import ConfigError
from so_101.review import review
from so_101.review_runs import review_runs
from so_101.web import review as reviewer
from so_101.web import server as web

SESSION = "20260101-100000"
RUN = "20260101-110000"
DEMO = f"{SESSION}/0"
DEMO_FRAMES = 90


def grey(camera: str, i: int) -> int:
    """The grey camera `camera` shows in demo frame `i`: 18 levels 12 apart, the wrist's the top's reversed."""
    level = 20 + 12 * (i % 18)
    return level if camera == "top" else 244 - level


def make_demo(root: Path) -> None:
    """One demo of DEMO_FRAMES frames: still for a second, moving for a second, still for a second."""
    from lerobot.datasets import LeRobotDataset

    robot, _ = make_follower(FakeClock(), REST_STEPS)
    try:
        demo = {k: f for k, f in features(robot).items() if k.startswith(("observation.", "action"))}
        dataset = LeRobotDataset.create("local/demo", FPS, demo, root=root, robot_type=robot.robot_type,
                                        streaming_encoding=False)
    finally:
        robot.inner.disconnect()
    for i in range(DEMO_FRAMES):
        state = np.full(6, 2.0 * min(max(i - FPS, 0), FPS), dtype=np.float32)
        dataset.add_frame({"observation.state": state, "action": state, "task": "test",
                           **{f"observation.images.{camera}": np.full((48, 64, 3), grey(camera, i), dtype=np.uint8)
                              for camera in ("top", "wrist")}})
    dataset.save_episode()
    dataset.finalize()


class Page:
    """A started ReviewServer over `source`, and requests to it."""

    def __init__(self, test: unittest.TestCase, source) -> None:
        self.server = reviewer.ReviewServer(web.Bind("127.0.0.1"), 0, source=source, task=TASK.name,
                                            routes=lambda: None)
        self.server.start()
        test.addCleanup(self.server.stop)
        self.port = self.server.server_address[1]

    def request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None,
                host: str | None = None) -> tuple[int, dict | bytes]:
        """Send one request, a POST as JSON. Returns the status code and the body, parsed when it's JSON."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        # The server prints each decision it saves, before it replies.
        with contextlib.redirect_stdout(io.StringIO()):
            return self._send(connection, method, path, body, headers, host)

    def _send(self, connection, method, path, body, headers, host) -> tuple[int, dict | bytes]:
        try:
            sent = {"Host": host or f"127.0.0.1:{self.port}", **(headers or {})}
            raw = None
            if method == "POST":
                raw = json.dumps(body or {}).encode()
                sent.setdefault("Content-Type", "application/json")
            connection.request(method, path, body=raw, headers=sent)
            response = connection.getresponse()
            got = response.read()
            kind = response.getheader("Content-Type") or ""
            return response.status, json.loads(got) if kind == "application/json" else got
        finally:
            connection.close()

    def get(self, path: str):
        code, body = self.request("GET", path)
        assert code == 200, (path, code, body)
        return body

    def decide(self, item: str, **decision) -> dict:
        """Open `item` and send `decision` for it, as the page does. Returns the reply."""
        self.get(f"/item/{item}")
        code, body = self.request("POST", "/decision", {"item": item, **decision})
        assert code == 200, (decision, code, body)
        return body


class ReviewerTest(unittest.TestCase):
    """One demo session and one run, recorded once, copied for the page (`self.page_s`) and for the
    terminal (`self.terminal_s`)."""

    @classmethod
    def setUpClass(cls) -> None:
        temp = tempfile.TemporaryDirectory(prefix="so101-review-")
        cls.addClassCleanup(temp.cleanup)
        cls.recorded = Path(temp.name) / "recorded"
        task_dir = cls.recorded / TASK.name
        make_demo(task_dir / "sessions" / SESSION / "dataset")
        rig = Rig(task_dir / "runs" / RUN, actions=GRIPPER_ACTIONS)
        try:
            # 1: the policy, then Reset with an outcome: labelled live.
            rig.policy(1)
            rig.reset("succeeded")
            # 2: the policy, paused with a reason, then Reset without an outcome.
            rig.policy(2)
            rig.command("pause")
            rig.tick(5)
            rig.command("label", label="missed_grasp")
            rig.tick(5)
            rig.command("reset")
            rig.tick_until("resting")
            rig.controller.close()
        finally:
            rig.close()

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="so101-review-")
        self.addCleanup(temp.cleanup)
        loaded = settings.load()
        self.page_s, self.terminal_s = (
            dataclasses.replace(loaded, data_dir=Path(temp.name) / name,
                                control=dataclasses.replace(loaded.control, blame_s=2.5))
            for name in ("page", "terminal"))
        for s in (self.page_s, self.terminal_s):
            shutil.copytree(self.recorded, s.data_dir)

    def quietly(self, fn, *args, answers: list[str] = ()):
        """Call `fn`, with `answers` typed at its prompts, printing nothing. Returns what it printed."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), mock.patch("builtins.input", side_effect=list(answers)):
            fn(*args)
        return printed.getvalue()

    def demos(self, everything: bool = False) -> reviewer.Demos:
        return reviewer.Demos(self.page_s, TASK, None, everything)

    def runs(self, everything: bool = False) -> reviewer.Runs:
        with contextlib.redirect_stdout(io.StringIO()):
            return reviewer.Runs(self.page_s, TASK, None, everything)

    # --- Demos ------------------------------------------------------------------------------------------

    def test_a_demo_is_listed_opened_and_its_frames_sent_for_the_range_and_camera_asked(self) -> None:
        page = Page(self, self.demos())
        listing = page.get("/items")
        self.assertEqual(listing["kind"], "demos")
        self.assertEqual([(i["id"], i["reviewed"], i["now"]) for i in listing["items"]], [(DEMO, False, "not reviewed")])
        self.assertEqual(listing["items"][0]["seconds"], DEMO_FRAMES / FPS)

        item = page.get(f"/item/{DEMO}")
        self.assertEqual((item["frames"], item["fps"], item["cameras"]), (DEMO_FRAMES, FPS, ["top", "wrist"]))
        self.assertEqual(len(item["joints"]["names"]), 6)
        self.assertEqual([len(v) for v in item["joints"]["values"]], [DEMO_FRAMES] * 6)
        self.assertEqual(item["joints"]["values"][0][-1], 2.0 * FPS)
        # Still for the first and last second: the suggestion cuts both, as the terminal's does.
        start, end = item["suggestion"]["keep"]
        self.assertTrue(0.5 < start < 1.0 and 2.0 < end < 3.0, item["suggestion"])
        self.assertIn("keep", item["suggestion"]["hint"])
        self.assertIsNone(item["edit"])
        self.assertEqual(item["bands"], [], "a demo has no skill or driver column")

        for camera in ("top", "wrist"):
            jpegs = reviewer.unpack(page.get(f"/frames/{DEMO}/{camera}?from=40&to=52"))
            self.assertEqual(len(jpegs), 12)
            for i, encoded in enumerate(jpegs, start=40):
                import cv2

                picture = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(picture.shape, (48, 64, 3), "never wider than the recording")
                self.assertLess(abs(float(picture.mean()) - grey(camera, i)), 5, f"{camera} frame {i}")
        # Past the end is cut at the end; nothing at all, or a camera it doesn't have, is refused.
        self.assertEqual(len(reviewer.unpack(page.get(f"/frames/{DEMO}/top?from=85&to=200"))), 5)
        self.assertEqual(page.request("GET", f"/frames/{DEMO}/top?from=90&to=95")[0], 400)
        self.assertEqual(page.request("GET", f"/frames/{DEMO}/side?from=0&to=5")[0], 404)
        self.assertEqual(page.request("GET", f"/frames/{SESSION}/7/top?from=0&to=5")[0], 404)
        self.assertEqual(page.request("GET", f"/item/{SESSION}/7")[0], 404)

    def test_each_demo_decision_writes_what_the_terminal_writes(self) -> None:
        page = Page(self, self.demos(everything=True))
        rows = [
            ("Enter", {"edit": "accept"}, [""]),
            ("k", {"edit": "keep"}, ["k"]),
            ("t", {"edit": "trim", "keep": [0.5, 2.25]}, ["t 0.5 2.25"]),
            ("t past the end", {"edit": "trim", "keep": [1, 99]}, ["t 1 99"]),
            ("d", {"edit": "delete", "reason": "dropped the cube"}, ["d dropped the cube"]),
            # The terminal splits its answer into words, so the page's reason is collapsed the same way.
            ("d with odd spaces", {"edit": "delete", "reason": "  dropped \t the cube\n"}, ["d dropped the cube"]),
        ]
        for case, decision, answers in rows:
            with self.subTest(case):
                reply = page.decide(DEMO, **decision)
                self.quietly(review, self.terminal_s, TASK, DEMO, False, answers=answers)
                written = data.edits_path(self.page_s, TASK).read_text()
                self.assertEqual(written, data.edits_path(self.terminal_s, TASK).read_text())
                self.assertEqual(reply["entry"]["now"], reply["saved"])
                self.assertTrue(reply["entry"]["reviewed"])
        self.assertEqual(data.load_edits(self.page_s, TASK)[(SESSION, 0)].delete, "dropped the cube")

        # What the terminal refuses, the page refuses, and nothing is written.
        before = data.edits_path(self.page_s, TASK).read_text()
        for decision in ({"edit": "trim", "keep": [1.0, 1.2]}, {"edit": "trim", "keep": [2, 1]},
                         {"edit": "trim", "keep": [1, True]}, {"edit": "trim"}, {"edit": "keep", "reason": "x"},
                         {"edit": "maybe"}, {"edit": "delete", "reason": 3}):
            with self.subTest(refused=decision):
                code, body = page.request("POST", "/decision", {"item": DEMO, **decision})
                self.assertEqual(code, 400, body)
        self.assertEqual(data.edits_path(self.page_s, TASK).read_text(), before)
        self.assertEqual(self.demos().demos, {}, "reviewed demos aren't listed again without --all")

    # --- Runs -------------------------------------------------------------------------------------------

    def test_an_attempt_labelled_live_is_left_out_unless_all_are_asked_for(self) -> None:
        self.assertEqual(list(self.runs().attempts), [f"{RUN}/2"])
        self.assertEqual(list(self.runs(everything=True).attempts), [f"{RUN}/1", f"{RUN}/2"])
        # The terminal goes by the same rule: only attempt 2 is asked about.
        printed = self.quietly(review_runs, self.terminal_s, TASK, None, False, answers=["q"])
        self.assertIn("Reviewing 1 attempt", printed)
        self.assertIn(f"run {RUN}, attempt 2", printed)
        self.assertNotIn("attempt 1 ", printed)

        page = Page(self, self.runs(everything=True))
        entries = {i["id"]: i for i in page.get("/items")["items"]}
        self.assertTrue(entries[f"{RUN}/1"]["reviewed"])
        self.assertEqual(entries[f"{RUN}/1"]["now"], "succeeded")
        self.assertFalse(entries[f"{RUN}/2"]["reviewed"])

    def test_an_attempt_is_opened_with_its_phases_interruption_and_events(self) -> None:
        page = Page(self, self.runs())
        self.assertEqual(page.get("/items")["kind"], "runs")
        item = page.get(f"/item/{RUN}/2")
        phases = {p["phase"] for p in item["phases"]}
        self.assertTrue({"autonomous", "paused", "returning"} <= phases, phases)
        self.assertEqual(sum(p["stop"] - p["start"] for p in item["phases"]), item["frames"])
        [interruption] = item["interruptions"]
        self.assertEqual((interruption["command"], interruption["reason"]), ("pause", "missed_grasp"))
        paused = next(p for p in item["phases"] if p["phase"] == "paused")
        self.assertEqual(interruption["frame"], paused["start"])
        self.assertEqual(interruption["source"], "blame")
        # The last blame_s (2.5 s) of the policy's frames, or all of them when it drove for less.
        driving = next(p for p in item["phases"] if p["phase"] == "autonomous" and p["stop"] == interruption["frame"])
        blamed = max(driving["start"], interruption["frame"] - round(2.5 * FPS))
        self.assertEqual(interruption["suspect"], [blamed, interruption["frame"]])
        labels = [e for e in item["events"] if e["type"] == "label"]
        self.assertEqual(len(labels), 1)
        self.assertIn("missed_grasp", labels[0]["text"])
        self.assertFalse({"state", "command"} & {e["type"] for e in item["events"]}, "drawn as phases, or noise")
        self.assertEqual((item["outcome"], item["recorded_outcome"], item["labelled"]), ("unlabelled", "unlabelled", False))
        self.assertEqual(len(item["joints"]["values"][0]), item["frames"])
        jpegs = reviewer.unpack(page.get(f"/frames/{RUN}/2/wrist?from=3&to=9"))
        self.assertEqual(len(jpegs), 6)

    def test_each_attempt_decision_writes_what_the_terminal_writes(self) -> None:
        page = Page(self, self.runs())
        item = f"{RUN}/2"
        frame = page.get(f"/item/{item}")["interruptions"][0]["frame"]
        rows = [
            ("f", {"interruption": frame, "suspect": [10, 20]}, ["f 10 20", "n"]),
            ("f in seconds", {"interruption": frame, "suspect": [15, 45]}, ["f 0.5s 1.5s", "n"]),
            ("c", {"interruption": frame, "suspect": []}, ["c", "n"]),
            ("a", {"interruption": frame, "auto": True}, ["a", "n"]),
            ("o f", {"outcome": "failed"}, ["", "o f"]),
            ("o i", {"outcome": "ignored"}, ["", "o i"]),
            ("Enter confirms", {"outcome": "ignored"}, ["", ""]),
            ("o u", {"outcome": "unlabelled"}, ["", "o u"]),
        ]
        for case, decision, answers in rows:
            with self.subTest(case):
                reply = page.decide(item, **decision)
                self.quietly(review_runs, self.terminal_s, TASK, item, False, answers=answers)
                run = data.Run.at(self.page_s.data_dir / TASK.name / "runs" / RUN)
                written = run.labels_path.read_text()
                other = data.Run.at(self.terminal_s.data_dir / TASK.name / "runs" / RUN).labels_path.read_text()
                self.assertEqual(written, other)
                self.assertIn(reply["saved"].split()[0], ("suspect", "outcome"))
        labels = analysis.load_labels(run)
        self.assertEqual(labels.outcomes, {2: None})
        self.assertEqual(labels.suspects, {})
        self.assertTrue(reply["entry"]["reviewed"], "an outcome in labels.toml, even unlabelled, is a review")

        before = run.labels_path.read_text()
        for decision in ({"interruption": frame, "suspect": [20, 10]}, {"interruption": frame, "suspect": [0, 10**6]},
                         {"interruption": frame + 1, "suspect": []}, {"interruption": frame, "auto": False},
                         {"interruption": frame, "suspect": [1.5, 3]}, {"interruption": True, "suspect": []},
                         {"outcome": "maybe"}, {"outcome": "failed", "interruption": frame}, {}):
            with self.subTest(refused=decision):
                code, body = page.request("POST", "/decision", {"item": item, **decision})
                self.assertEqual(code, 400, body)
        self.assertEqual(run.labels_path.read_text(), before)

    def test_an_attempt_has_named_driver_bands_and_no_skill_band_without_skills(self) -> None:
        item = Page(self, self.runs()).get(f"/item/{RUN}/2")
        bands = {band["name"]: band for band in item["bands"]}
        self.assertEqual(set(bands), {"driver"}, "a run without skills records -1 on every frame: no band")
        names = {span["label"] for span in bands["driver"]["spans"]}
        self.assertTrue({"policy", "other"} <= names <= set(DRIVERS.values()), names)
        self.assertEqual(sum(s["stop"] - s["start"] for s in bands["driver"]["spans"]), item["frames"])

    # --- What the page's script sends -------------------------------------------------------------------

    def test_every_body_the_pages_script_sends_is_taken(self) -> None:
        """Each decision the script in review.html sends, as `decide` sends it: {item, ...body}. The
        snippet is the call in the script, so a change to one fails here until the other matches."""
        script = reviewer.page().decode()
        self.assertIn("body: JSON.stringify({ item: item.id, ...body })", script)
        demos = Page(self, self.demos(everything=True))
        rows = [
            ('demoDecide(touched ? { edit: "trim", keep } : { edit: "accept" })', {"edit": "accept"}),
            ('demoDecide(touched ? { edit: "trim", keep } : { edit: "accept" })', {"edit": "trim", "keep": [0.5, 2.25]}),
            ('demoDecide({ edit: "keep" })', {"edit": "keep"}),
            ('demoDecide({ edit: "delete", reason: $("reason").value })', {"edit": "delete", "reason": ""}),
            ('demoDecide({ edit: "delete", reason: $("reason").value })', {"edit": "delete", "reason": "dropped it"}),
        ]
        for snippet, body in rows:
            with self.subTest(body=body):
                self.assertIn(snippet, script)
                demos.decide(DEMO, **body)

        runs = Page(self, self.runs())
        item = f"{RUN}/2"
        frame = runs.get(f"/item/{item}")["interruptions"][0]["frame"]
        self.assertIn("decide({ interruption: i.frame, ...body })", script)
        rows = [
            ("suspect({ suspect: pending })", {"interruption": frame, "suspect": [10, 20]}),
            ("suspect({ suspect: [] })", {"interruption": frame, "suspect": []}),
            ("suspect({ auto: true })", {"interruption": frame, "auto": True}),
            ("decide({ outcome: o })", {"outcome": "failed"}),
        ]
        for snippet, body in rows:
            with self.subTest(body=body):
                self.assertIn(snippet, script)
                runs.decide(item, **body)

    def test_a_decision_names_its_item_whichever_was_opened_last(self) -> None:
        page = Page(self, self.runs(everything=True))
        page.get(f"/item/{RUN}/1")  # another tab opened attempt 1 since
        code, body = page.request("POST", "/decision", {"item": f"{RUN}/2", "outcome": "failed"})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["entry"]["id"], f"{RUN}/2")
        run = data.Run.at(self.page_s.data_dir / TASK.name / "runs" / RUN)
        self.assertEqual(analysis.load_labels(run).outcomes, {2: "failed"})
        code, body = page.request("POST", "/decision", {"item": f"{RUN}/9", "outcome": "failed"})
        self.assertEqual(code, 404, body)
        self.assertIn("No item", body["error"])

    def test_a_body_of_the_wrong_shape_is_refused_with_400(self) -> None:
        demos = Page(self, self.demos())
        for body in ({"edit": []}, {"edit": {}}, {"edit": None}, {"edit": 3}):
            with self.subTest(body=body):
                code, reply = demos.request("POST", "/decision", {"item": DEMO, **body})
                self.assertEqual(code, 400, reply)
        runs = Page(self, self.runs())
        for body in ({"outcome": []}, {"outcome": {}}, {"interruption": [], "suspect": []},
                     {"interruption": {}, "auto": True}, {"interruption": 1, "suspect": {}}):
            with self.subTest(body=body):
                code, reply = runs.request("POST", "/decision", {"item": f"{RUN}/2", **body})
                self.assertEqual(code, 400, reply)
        self.assertEqual(demos.request("POST", "/decision", {"item": [], "edit": "keep"})[0], 404, "an item is its id")

    # --- Files that change or break under a review --------------------------------------------------------

    def test_a_labels_toml_broken_mid_review_answers_each_request_with_the_error(self) -> None:
        page = Page(self, self.runs())
        item = f"{RUN}/2"
        page.get(f"/item/{item}")
        run = data.Run.at(self.page_s.data_dir / TASK.name / "runs" / RUN)
        run.labels_path.write_text("outcomes = [\n")  # cut short, as by an editor's half-written save
        for method, path, body in (("GET", "/items", None), ("GET", f"/item/{item}", None),
                                   ("GET", f"/frames/{item}/top?from=0&to=5", None),
                                   ("POST", "/decision", {"item": item, "outcome": "failed"})):
            with self.subTest(path=path, method=method):
                code, reply = page.request(method, path, body)
                self.assertEqual(code, 409, reply)
                self.assertIn(str(run.labels_path), reply["error"])
        self.assertEqual(run.labels_path.read_text(), "outcomes = [\n", "nothing was written over it")
        run.labels_path.unlink()
        self.assertEqual(page.decide(item, outcome="failed")["entry"]["now"], "failed (labels.toml; recorded as unlabelled)")

    def test_an_edits_toml_broken_mid_review_answers_with_the_error(self) -> None:
        page = Page(self, self.demos())
        page.get(f"/item/{DEMO}")
        path = data.edits_path(self.page_s, TASK)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[[episode]\n")
        for method, path_, body in (("GET", "/items", None), ("GET", f"/item/{DEMO}", None),
                                    ("POST", "/decision", {"item": DEMO, "edit": "keep"})):
            with self.subTest(path=path_, method=method):
                code, reply = page.request(method, path_, body)
                self.assertEqual(code, 409, reply)
                self.assertIn(str(path), reply["error"])

    def kill_during_second_save(self) -> data.Run:
        """The page's copy of the run as a kill leaves it after attempt 2's data file and attempt_end
        were written, but before its episode reached the metadata: the live meta/info.json torn, and
        the copy of the metadata from the last finished save, attempt 1's, with one episode."""
        run = data.Run.at(self.page_s.data_dir / TASK.name / "runs" / RUN)
        copies = run.path / data.META_COPY_DIR
        (copies / "2").rename(copies / "1")
        meta = copies / "1" / "meta"
        (meta / "episodes" / "chunk-000" / "file-001.parquet").unlink()
        info = json.loads((meta / "info.json").read_text())
        info["total_episodes"] = 1
        (meta / "info.json").write_text(json.dumps(info))
        (run.dataset_root / "meta" / "info.json").write_text('{"codebase_version": ')
        return run

    def test_an_attempt_a_kill_left_out_of_the_episode_index_is_left_out_and_said(self) -> None:
        self.kill_during_second_save()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            runs = reviewer.Runs(self.page_s, TASK, None, True)
        self.assertEqual(list(runs.attempts), [f"{RUN}/1"])
        self.assertIn("attempt 2", printed.getvalue())
        self.assertIn("episode index", printed.getvalue())
        page = Page(self, runs)
        listing = page.get("/items")
        self.assertEqual([i["id"] for i in listing["items"]], [f"{RUN}/1"])
        [note] = listing["notes"]
        self.assertIn(f"run {RUN} attempt 2", note)
        self.assertEqual(page.request("GET", f"/item/{RUN}/2")[0], 404)
        # Attempt 1 is whole, its frames read through the copy's metadata.
        item = page.get(f"/item/{RUN}/1")
        self.assertEqual(len(reviewer.unpack(page.get(f"/frames/{RUN}/1/top?from=0&to={item['frames']}"))), item["frames"])

    # --- Frames -----------------------------------------------------------------------------------------

    def test_frames_are_decoded_a_chunk_at_a_time_as_asked(self) -> None:
        calls = []
        real = reviewer.episode_frames

        def recording(meta, episode, key, start=0, stop=None):
            calls.append((key.removeprefix("observation.images."), start, stop))
            return real(meta, episode, key, start, stop)

        page = Page(self, self.demos())
        with mock.patch.object(reviewer, "episode_frames", recording):
            self.assertEqual(len(reviewer.unpack(page.get(f"/frames/{DEMO}/top?from=40&to=52"))), 12)
            self.assertEqual(len(reviewer.unpack(page.get(f"/frames/{DEMO}/top?from=40&to=52"))), 12)
            self.assertEqual(len(reviewer.unpack(page.get(f"/frames/{DEMO}/wrist?from=0&to=500"))), DEMO_FRAMES)
        self.assertEqual(calls, [("top", 40, 52), ("wrist", 0, DEMO_FRAMES)], "decoded once, and only what was asked")

    def test_the_frames_that_decode_are_sent_when_the_rest_of_the_video_doesnt(self) -> None:
        real = reviewer.episode_frames

        def short(meta, episode, key, start=0, stop=None):
            for i, frame in real(meta, episode, key, start, stop):
                if i == 45:
                    raise ValueError(f"{key}: episode {episode} video ends after 45 of {stop} frames.")
                yield i, frame

        page = Page(self, self.demos())
        with mock.patch.object(reviewer, "episode_frames", short):
            jpegs = reviewer.unpack(page.get(f"/frames/{DEMO}/top?from=40&to=52"))
            self.assertEqual(len(jpegs), 5, "frames 40 to 44, the ones before the video ends")
            code, body = page.request("GET", f"/frames/{DEMO}/top?from=45&to=52")
        self.assertEqual(code, 500)
        self.assertIn("ends after 45", body["error"])
        self.assertIn("top", body["error"])
        # Nothing that failed was kept: once the video reads, the same frames are sent.
        self.assertEqual(len(reviewer.unpack(page.get(f"/frames/{DEMO}/top?from=45&to=52"))), 7)

    def test_a_video_pyav_can_t_read_or_a_recording_that_won_t_parse_is_a_500_that_says_why(self) -> None:
        import av

        page = Page(self, self.demos())
        for error in (av.error.EOFError(-541478725, "End of file"), av.error.UndefinedError(-1, "Unknown error")):
            def broken(meta, episode, key, start=0, stop=None, error=error):
                raise error
                yield

            with self.subTest(error=type(error).__name__), mock.patch.object(reviewer, "episode_frames", broken):
                code, body = page.request("GET", f"/frames/{DEMO}/top?from=0&to=5")
                self.assertEqual(code, 500)
                self.assertIn(str(error), body["error"])
        with mock.patch.object(page.server.source, "detail", side_effect=ValueError("Parquet magic bytes not found")):
            code, body = page.request("GET", f"/item/{DEMO}")
        self.assertEqual(code, 500)
        self.assertIn("Parquet magic bytes not found", body["error"])
        with mock.patch.object(page.server.source, "video", side_effect=av.error.UndefinedError(-1, "Unknown error")), \
                contextlib.redirect_stderr(io.StringIO()):
            code, body = page.request("GET", f"/frames/{DEMO}/wrist?from=0&to=5")
        self.assertEqual(code, 500, "the page server's own fallback, for an error no route caught")
        self.assertIn("UndefinedError", body["error"])

    def test_a_frames_body_may_take_longer_to_send_than_a_silent_connection_may_last(self) -> None:
        page = Page(self, self.demos())
        jpegs = [bytes([i % 256]) * 60_000 for i in range(reviewer.FRAMES_PER_REQUEST)]  # 9 MB: more than the loopback's socket buffers hold
        with mock.patch.object(web.Handler, "timeout", 0.3), \
                mock.patch.object(page.server, "decoded", return_value=(jpegs, None)):
            connection = http.client.HTTPConnection("127.0.0.1", page.port, timeout=30)
            self.addCleanup(connection.close)
            connection.request("GET", f"/frames/{DEMO}/top?from=0&to=90", headers={"Host": f"127.0.0.1:{page.port}"})
            response = connection.getresponse()
            time.sleep(1.0)  # a phone on a slow link: longer than the connection's 0.3 s timeout
            got = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(reviewer.unpack(got), jpegs)
        self.assertEqual(reviewer.send_timeout(4_000_000), 32.0, "1 Mbit/s for a 4 MB chunk")
        self.assertEqual(reviewer.send_timeout(10), web.Handler.timeout)

    def test_notes_are_matched_to_an_attempt_by_its_whole_number(self) -> None:
        notes = ["attempt 1's episode 0 isn't in the dataset", "attempt 12's episode 5 isn't in the dataset",
                 "attempt 21 dropped 3 frames", "skipped data/chunk-000/file-001.parquet"]
        self.assertEqual(reviewer.notes_for(notes, 1), [notes[0]])
        self.assertEqual(reviewer.notes_for(notes, 12), [notes[1]])
        self.assertEqual(reviewer.notes_for(notes, 2), [])

    # --- Never public -----------------------------------------------------------------------------------

    def test_the_review_page_answers_only_its_own_page(self) -> None:
        page = Page(self, self.demos())
        # DNS rebinding: another name for this address.
        code, body = page.request("GET", "/items", host=f"evil.example:{page.port}")
        self.assertEqual(code, 403)
        self.assertIn("only to its own addresses", body["error"])
        self.assertIn("so101 review", body["error"])
        self.assertEqual(page.request("GET", "/", host=f"localhost:{page.port}")[0], 200)
        # Through Tailscale Serve or Funnel, while it listens on 127.0.0.1 without HTTPS.
        for headers in ({"X-Forwarded-For": "100.64.0.2"}, {"Tailscale-Funnel-Request": "?1"}):
            self.assertEqual(page.request("GET", "/items", headers=headers)[0], 403)
        # A decision from another site, or one another site could send without asking, writes nothing.
        page.get(f"/item/{DEMO}")
        for headers in ({"Origin": "http://evil.example"}, {"Content-Type": "text/plain"},
                        {"Content-Type": "application/x-www-form-urlencoded"}):
            code, body = page.request("POST", "/decision", {"item": DEMO, "edit": "keep"}, headers=headers)
            self.assertEqual(code, 403, headers)
        self.assertIn("only from the review page", page.request(
            "POST", "/decision", {"item": DEMO, "edit": "keep"}, headers={"Origin": "http://evil.example"})[1]["error"])
        # A decision for an item this review doesn't cover.
        code, body = page.request("POST", "/decision", {"item": f"{SESSION}/1", "edit": "keep"})
        self.assertEqual(code, 404)
        self.assertFalse(data.edits_path(self.page_s, TASK).exists(), "nothing was written")
        # A GET with a body, or a POST elsewhere.
        self.assertEqual(page.request("GET", "/items", headers={"Content-Length": "5"})[0], 400)
        self.assertEqual(page.request("POST", "/command", {"command": "pause"})[0], 404)
        # The same decision from the page itself is taken.
        code, _ = page.request("POST", "/decision", {"item": DEMO, "edit": "keep"},
                               headers={"Origin": f"http://127.0.0.1:{page.port}"})
        self.assertEqual(code, 200)


class EditsFileTest(unittest.TestCase):
    """`data.save_edits` writes only text that reads back as what it was given."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="so101-edits-")
        self.addCleanup(temp.cleanup)
        self.s = dataclasses.replace(settings.load(), data_dir=Path(temp.name))

    def test_control_characters_in_a_reason_are_dropped(self) -> None:
        # DEL is no TOML string's, escaped or not, and a lone surrogate can't be written as UTF-8 at all.
        edit = data.Edit(SESSION, 0, delete="dropped\x7f the\x00 cube\x1b\ud800\u0085")
        data.save_edits(self.s, TASK, {edit.key: edit})
        self.assertEqual(data.load_edits(self.s, TASK)[edit.key].delete, "dropped the cube")

    def test_text_that_wont_parse_leaves_the_file_as_it_was(self) -> None:
        kept = data.Edit(SESSION, 0, keep=(1.0, 2.0))
        data.save_edits(self.s, TASK, {kept.key: kept})
        before = data.edits_path(self.s, TASK).read_text()
        bad = data.Edit("20260101-\x7f", 1)  # a session name is a folder's, never stripped
        with self.assertRaises(ConfigError):
            data.save_edits(self.s, TASK, {kept.key: kept, bad.key: bad})
        self.assertEqual(data.edits_path(self.s, TASK).read_text(), before)


class ReviewPortTest(ConfigTest):
    """`check` covers `[control] review_port` as it does `port`, and the two must differ."""

    def problems(self, control: str) -> list[str]:
        self.config.write_text(ARMS + "\n[control]\n" + control)
        report = Report()
        check_control(report, settings.load(), "FAIL")
        return [r.detail for r in report.results if r.status == "FAIL" and r.group == "control"]

    def test_review_port_is_a_port_other_than_the_control_pages(self) -> None:
        self.assertEqual(self.problems(""), [])
        self.assertEqual(settings.load().control.review_port, 8766)
        for line, why in (("review_port = 8101", "the control page's port too"), ('review_port = "8766"', "port number"),
                          ("review_port = 0", "port number"), ("review_port = true", "port number")):
            with self.subTest(line):
                [problem] = self.problems(line + "\n")
                self.assertIn("review_port", problem)
                self.assertIn(why, problem)
        self.assertEqual(self.problems("port = 9000\nreview_port = 8101\n"), [])


if __name__ == "__main__":
    unittest.main()
