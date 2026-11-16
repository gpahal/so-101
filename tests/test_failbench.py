"""so101-failbench without any network: the labels file, subsets, answer parsing, what each adapter sends
(through stand-in SDKs), scoring, the runner's cache, resume, spending cap, Ctrl+C and failures, the
candidates' replay and merge, building from a recorded run, and the command line."""

import _thread
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from so_101.failbench import candidates, cli, dataset, metrics, runner
from so_101.grasp.detector import Detector
from so_101.vision import models, prompt, providers

# A made-up labels file: header comments, one wrong, two ok and one row nobody has labelled yet.
LABELS = """# Made up for the tests.
id,src,ep,t,moment,label,category,gripper,note
a,run1,0,10,after_close,wrong,missed_grasp,9.5,closed beside the block
b,run1,0,40,after_release,ok,none,1.2,block in the jar
c,run1,1,5,end,ok,none,1.0,
d,run1,1,8,approach,,,30.0,not labelled yet
"""


def sample(i, label="ok", category="none", moment="end"):
    return dataset.Sample(id=f"s{i}", src="run1", ep=0, t=i, moment=moment, label=label,
                          category=category, gripper=1.0, note="")


def write_labels(folder: Path, text: str = LABELS) -> Path:
    path = Path(folder) / "labels.csv"
    path.write_text(text)
    return path


class LabelsTest(unittest.TestCase):
    def test_a_labels_file_reads_and_rows_without_a_label_are_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_labels(d)
            samples = dataset.read_labels(path)
            self.assertEqual([s.id for s in samples], ["a", "b", "c"])
            self.assertEqual(dataset.unlabelled(path), 1)
            self.assertTrue(samples[0].wrong and not samples[1].wrong)

    def test_rejects_ok_with_a_category(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_labels(d, "id,src,ep,t,moment,label,category,gripper,note\na,x,0,1,end,ok,missed_jar,1,\n")
            with self.assertRaises(ValueError):
                dataset.read_labels(p)

    def test_the_rollout_labels_are_gone(self):
        self.assertFalse((Path(dataset.__file__).parent / "labels.csv").exists())
        self.assertFalse(hasattr(dataset.Sample, "policy"))
        self.assertNotIn("rollout", (dataset.__doc__ or "").lower())

    def test_subset_is_deterministic_and_keeps_both_labels(self):
        samples = [sample(i, "wrong", "missed_grasp", "after_close") for i in range(30)] + [sample(100 + i) for i in range(70)]
        a, b = dataset.subset(samples, 20), dataset.subset(samples, 20)
        self.assertEqual([s.id for s in a], [s.id for s in b])
        self.assertTrue(5 <= sum(s.wrong for s in a) <= 7)
        self.assertEqual(len(dataset.subset(samples, None)), 100)

    def test_every_missing_recording_is_named_in_one_message(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_labels(d, LABELS.replace("run1,1,", "run2,1,"))
            with self.assertRaises(dataset.MissingRecordings) as caught:
                dataset.build(Path(d) / "out", [Path(d) / "runs"], path)
            message = str(caught.exception)
            self.assertIn("run1", message)
            self.assertIn("run2", message)
            self.assertEqual(message.count("\n"), 0, "one line")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exited:
                cli.main(["--env-file", str(Path(d) / "none.env"), "--dataset", str(Path(d) / "out"), "build",
                          "--labels", str(path), "--runs", str(Path(d) / "runs")])
            self.assertEqual(str(exited.exception), message)


class ParseTest(unittest.TestCase):
    def test_plain_json(self):
        a = providers.parse_answer('{"observation":"x","wrong":true,"p_wrong":0.8,"category":"missed_jar"}')
        self.assertEqual((a["wrong"], a["p_wrong"], a["category"]), (True, 0.8, "missed_jar"))

    def test_fenced_json_and_unknown_category(self):
        a = providers.parse_answer('```json\n{"wrong": "false", "p_wrong": 1.7, "category": "spill"}\n```')
        self.assertEqual((a["wrong"], a["p_wrong"], a["category"]), (False, 1.0, "other"))

    def test_no_answer(self):
        with self.assertRaises(providers.ProviderError):
            providers.parse_answer("I can't tell.")

    def test_json_that_isnt_an_object_is_no_answer(self):
        for text in ("[1, 2]", '"wrong"', "true", "null", "3"):
            with self.subTest(text), self.assertRaises(providers.ProviderError):
                providers.parse_answer(text)


class DescribeTest(unittest.TestCase):
    def test_the_two_camera_check_names_top_then_wrist(self):
        self.assertEqual(prompt.describe([("top", b""), ("wrist", b"")]),
                         "The first image is the TOP camera, the second the WRIST camera.")
        self.assertTrue(prompt.decision_context("end").endswith(prompt.describe(("top", "wrist"))))

    def test_one_camera(self):
        self.assertEqual(prompt.describe(["top"]), "The image is the TOP camera.")
        self.assertEqual(prompt.describe([("wrist", b"")]), "The image is the WRIST camera.")

    def test_the_watch_check_s_four(self):
        self.assertEqual(prompt.describe(["top", "wrist", "top_2s_ago", "wrist_2s_ago"]),
                         "The first image is the TOP camera now, the second the WRIST camera now, "
                         "the third the TOP camera 2 s ago, the fourth the WRIST camera 2 s ago.")
        self.assertIn("the third the TOP camera 2.5 s ago", prompt.describe(["top", "wrist", "top_2_5s_ago", "wrist_2_5s_ago"]))
        self.assertEqual(prompt.label("top_2s_ago"), "TOP camera, 2 s ago")
        self.assertEqual(prompt.label("wrist"), "WRIST camera")

    def test_the_decision_context_says_what_its_request_carries(self):
        self.assertTrue(prompt.decision_context("end", ["top"]).endswith("The image is the TOP camera."))

    def test_the_start_moment_is_gone(self):
        self.assertNotIn("start", prompt.MOMENTS)


class SDKs:
    """Stand-ins for the provider SDKs: each records the arguments of every client it makes and every
    request it sends, and answers as the real one would. `usage=False` answers without usage."""

    def __init__(self, usage: bool = True):
        self.calls: list[tuple[str, dict]] = []
        self.usage = usage
        ns = types.SimpleNamespace
        sdk = self

        def record(name):
            def call(*args, **kwargs):
                sdk.calls.append((name, {"args": args, **kwargs}))
                return ns(**kwargs)
            return call

        class OpenAI:
            def __init__(self, **kwargs):
                sdk.calls.append(("openai.OpenAI", kwargs))
                self.chat = ns(completions=ns(create=self.chat_create))
                self.decisions = ns(create=self.decide)

            def chat_create(self, **kwargs):
                sdk.calls.append(("openai.chat", kwargs))
                usage = ns(prompt_tokens=1000, completion_tokens=100, cost=None, model_extra={}) if sdk.usage else None
                message = ns(content='{"observation":"x","wrong":false,"p_wrong":0.1,"category":"none"}', refusal=None)
                return ns(choices=[ns(message=message, finish_reason="stop")], usage=usage, provider="p", model_extra={})

            def decide(self, **kwargs):
                sdk.calls.append(("openai.decisions", kwargs))
                return ns(usage=ns(input_tokens=1000) if sdk.usage else None,
                          answers=[ns(type="predicate", probability=0.2), ns(type="choice", choice="none")])

        class Client:
            def __init__(self, **kwargs):
                sdk.calls.append(("genai.Client", kwargs))
                self.models = ns(generate_content=self.generate)
                self.decisions = ns(create=self.decide)

            def generate(self, **kwargs):
                sdk.calls.append(("genai.generate", kwargs))
                um = ns(prompt_token_count=1000, candidates_token_count=50, thoughts_token_count=50) if sdk.usage else None
                return ns(usage_metadata=um, candidates=[ns(finish_reason="STOP")], prompt_feedback=None,
                          text='{"observation":"x","wrong":true,"p_wrong":0.9,"category":"missed_jar"}')

            def decide(self, **kwargs):
                sdk.calls.append(("genai.decisions", kwargs))
                return ns(usage=ns(total_input_tokens=1000) if sdk.usage else None,
                          answers={"wrong": ns(noul=0.7), "category": ns(choice="dropped", confidence=0.6)})

        class TypeSafeClient:
            def __init__(self, **kwargs):
                sdk.calls.append(("typesafe.Client", kwargs))

            def system_one(self, **kwargs):
                sdk.calls.append(("typesafe.system_one", kwargs))
                return ns(usage=ns(input_tokens=1000) if sdk.usage else None, nouls={"wrong": ns(noul=0.3)},
                          choices={"category": ns(choice="none")}, model="jev")

        genai_types = types.ModuleType("google.genai.types")
        genai_types.Part = ns(from_bytes=record("genai.Part"))
        genai_types.GenerateContentConfig = record("genai.config")
        genai_types.ThinkingConfig = record("genai.thinking")
        genai = types.ModuleType("google.genai")
        genai.Client, genai.types = Client, genai_types
        google = types.ModuleType("google")
        google.genai = genai
        openai = types.ModuleType("openai")
        openai.OpenAI = OpenAI
        typesafe = types.ModuleType("typesafe_sdk")
        typesafe.TypeSafeClient = TypeSafeClient
        typesafe.Image, typesafe.Noul, typesafe.Choice = (record(f"typesafe.{n}") for n in ("Image", "Noul", "Choice"))
        self.modules = {"openai": openai, "google": google, "google.genai": genai, "google.genai.types": genai_types,
                        "typesafe_sdk": typesafe}

    def installed(self):
        return mock.patch.dict(sys.modules, self.modules)

    def sent(self, prefix: str) -> str:
        """Everything the requests named `prefix...` carried, as text."""
        return json.dumps([kw for name, kw in self.calls if name.startswith(prefix)], default=repr)


ADAPTED = {  # a model per adapter, and the name of its request call
    "gemini-3.8-flash": "openai.chat", "gemini-robotics-er-2": "genai.generate", "gemini-decision-1": "genai.decisions",
    "openai-decisions-luna": "openai.decisions", "jev": "typesafe.system_one",
}
FOUR = [("top", b"t"), ("wrist", b"w"), ("top_2s_ago", b"t0"), ("wrist_2s_ago", b"w0")]


class AdapterTest(unittest.TestCase):
    def setUp(self):
        keys = {name: "test-key" for name in set(models.KEYS.values())}
        patch = mock.patch.dict(os.environ, keys)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(providers._cache.clear)
        providers._cache.clear()

    def predict(self, key: str, sdks: SDKs, images=FOUR, moment="stuck") -> dict:
        with sdks.installed():
            return models.MODELS[key].adapter().predict(images, moment)

    def test_every_adapter_says_which_images_it_sends(self):
        for key, call in ADAPTED.items():
            for images in (FOUR, FOUR[:1], FOUR[:2]):
                with self.subTest(key=key, images=len(images)):
                    providers._cache.clear()
                    sdks = SDKs()
                    self.predict(key, sdks, images)
                    self.assertIn(prompt.describe(images), sdks.sent(call))
                    self.assertNotIn("TOP_2S_AGO", sdks.sent(call), "each image is labelled in words")

    def test_both_gemini_adapters_get_the_request_timeout(self):
        for key in ("gemini-robotics-er-2", "gemini-decision-1"):
            with self.subTest(key):
                sdks = SDKs()
                self.predict(key, sdks)
                [client] = [kw for name, kw in sdks.calls if name == "genai.Client"]
                self.assertEqual(client["http_options"]["timeout"], providers.TIMEOUT_S * 1000, "milliseconds")

    def test_an_answer_without_usage_has_no_cost(self):
        for key in ADAPTED:
            with self.subTest(key):
                providers._cache.clear()
                self.assertIsNone(self.predict(key, SDKs(usage=False)).get("cost_usd"))
                providers._cache.clear()
                self.assertGreater(self.predict(key, SDKs()).get("cost_usd"), 0)

    def test_a_construction_error_is_a_provider_error_and_isnt_kept(self):
        m = models.Model("broken", models.OPENROUTER, "x/y", 1, 1, "test")

        def refuse(model):
            raise KeyError("OPENROUTER_API_KEY")

        with mock.patch.dict(providers._ADAPTERS, {models.OPENROUTER: refuse}):
            with self.assertRaises(providers.ProviderError) as caught:
                m.adapter()
            self.assertFalse(caught.exception.retryable)
        self.assertNotIn("broken", providers._cache)
        with SDKs().installed():
            self.assertIsInstance(m.adapter(), providers.OpenRouterChat, "the next try builds it")

    def test_models_have_no_unused_options(self):
        self.assertNotIn("options", {f.name for f in __import__("dataclasses").fields(models.Model)})


class MetricsTest(unittest.TestCase):
    def test_scores(self):
        samples = [sample(1, "wrong", "missed_grasp", "after_close"), sample(2, "wrong", "bad_grasp", "after_close"),
                   sample(3), sample(4)]
        preds = {"s1": {"wrong": True, "p_wrong": 0.9, "category": "bad_grasp", "latency_s": 1.0, "cost_usd": 0.01},
                 "s2": {"error": "boom", "cost_usd": 0.0},
                 "s3": {"wrong": False, "p_wrong": 0.1, "category": "none", "latency_s": 2.0, "cost_usd": 0.01},
                 "s4": {"wrong": True, "p_wrong": 0.6, "category": "other", "latency_s": 3.0, "cost_usd": 0.01}}
        sc = metrics.score("m", samples, preds)
        self.assertEqual(sc.errors, 1)
        self.assertAlmostEqual(sc.balanced_accuracy, 0.5)  # caught 1 of 2 failures, passed 1 of 2 ok
        self.assertAlmostEqual(sc.category_accuracy, 1.0)  # bad_grasp is in missed_grasp's family
        self.assertAlmostEqual(sc.auroc, 1.0)
        self.assertIn("| m |", metrics.report("t", samples, [sc], []))

    def test_auroc_ties(self):
        self.assertAlmostEqual(metrics.auroc([True, False], [0.5, 0.5]), 0.5)

    def test_an_unasked_sample_is_neither_an_error_nor_wrong(self):
        samples = [sample(1, "wrong", "missed_grasp", "after_close"), sample(2), sample(3, "wrong", "dropped"), sample(4)]
        right = {"s1": {"wrong": True, "p_wrong": 0.9}, "s2": {"wrong": False, "p_wrong": 0.1}}
        sc = metrics.score("m", samples, right)
        self.assertEqual((sc.errors, sc.unasked), (0, 2))
        self.assertAlmostEqual(sc.balanced_accuracy, 1.0)

    def test_models_are_scored_on_the_samples_every_one_was_asked_about(self):
        samples = [sample(1, "wrong", "missed_grasp", "after_close"), sample(2), sample(3, "wrong", "dropped"), sample(4)]
        full = {s.id: {"wrong": s.wrong, "p_wrong": 0.9 if s.wrong else 0.1} for s in samples}
        capped = {"s1": {"error": "boom"}, "s2": {"wrong": False, "p_wrong": 0.1}}
        scored, scores = metrics.score_all(samples, {"full": full, "capped": capped})
        self.assertEqual([s.id for s in scored], ["s1", "s2"])
        self.assertEqual([(sc.n, sc.errors, sc.unasked) for sc in scores], [(2, 0, 0), (2, 1, 0)])
        text = metrics.report("t", scored, scores, metrics.coverage(samples, scored, {"full": full, "capped": capped}))
        self.assertIn("2 of the 4 samples", text)
        self.assertIn("capped was asked about 2", text)

    def test_metrics_at_the_supervisor_s_thresholds(self):
        wrong = [sample(i, "wrong", "missed_grasp", "after_close") for i in range(3)]
        ok = [sample(10 + i) for i in range(3)]
        ps = dict(zip([s.id for s in wrong + ok], [0.9, 0.5, 0.1, 0.85, 0.1, 0.2], strict=True))
        preds = {k: {"wrong": p >= 0.5, "p_wrong": p} for k, p in ps.items()}
        sc = metrics.score("m", wrong + ok, preds, sure=(0.2, 0.8))
        self.assertAlmostEqual(sc.share_unsure, 1 / 6)
        self.assertAlmostEqual(sc.precision_sure_wrong, 1 / 2)  # 0.9 is wrong, 0.85 is ok
        self.assertAlmostEqual(sc.missed_in_sure_ok, 1 / 3)  # 0.1, 0.1 and 0.2: one of them is wrong
        text = metrics.report("t", wrong + ok, [sc], [], sure=(0.2, 0.8))
        self.assertIn("p_wrong at or over 0.8", text)
        self.assertIn("| m | 17% | 50% | 33% |", text)


class FakeAdapter:
    def __init__(self, cost=0.01):
        self.cost, self.calls = cost, 0

    def predict(self, images, moment):
        self.calls += 1
        assert [cam for cam, _ in images] == ["top", "wrist"]
        return {"wrong": moment == "after_close", "p_wrong": 0.5, "category": "none", "cost_usd": self.cost, "latency_s": 0.01}


class Scripted:
    """Answers, or raises, from a list, one per call in order; repeats the last."""

    def __init__(self, *script):
        self.script, self.calls = list(script), 0

    def predict(self, images, moment):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if callable(step):
            return step()
        if isinstance(step, BaseException):
            raise step
        return {"wrong": False, "p_wrong": 0.1, "category": "none", "cost_usd": step, "latency_s": 0.01}


def bad_request():
    return providers.ProviderError("bad request", retryable=False)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        (self.root / "images").mkdir()
        self.samples = [sample(i, moment="after_close" if i % 2 else "end") for i in range(10)]
        for s in self.samples:
            for cam in dataset.CAMERAS:
                dataset.image_path(self.root, s, cam).write_bytes(b"jpeg")
        self.model = models.Model("fake", models.OPENROUTER, "fake/model", 1, 1, "test", estimate=0.01)
        self.fake = FakeAdapter()
        providers._cache[self.model.key] = self.fake

    def tearDown(self):
        providers._cache.pop(self.model.key, None)
        providers._cache.pop("fake2", None)
        self.dir.cleanup()

    def run_once(self, max_cost, ms=None, workers=2, **options):
        cache = runner.Cache(self.root / "predictions.jsonl")
        budget = runner.Budget(max_cost)
        self.logged = []
        self.result = runner.run(ms or [self.model], self.samples, self.root, cache, runner.Variant(), budget,
                                 workers=workers, log=self.logged.append, **options)
        self.budget = budget
        return cache

    def test_cap_then_resume_without_repeating(self):
        cache = self.run_once(0.045)  # room for 4 calls at $0.01
        self.assertEqual(len(cache.for_model("fake", "top+wrist")), 4)
        cache = self.run_once(1.0)
        self.assertEqual(len(cache.for_model("fake", "top+wrist")), 10)
        self.assertEqual(self.fake.calls, 10)  # nothing asked twice
        lines = (self.root / "predictions.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 10)
        self.assertTrue(all(json.loads(line)["model"] == "fake" for line in lines))

    def test_errors_are_recorded_and_retried_on_request(self):
        class Broken:
            def predict(self, images, moment):
                raise providers.ProviderError("bad request", retryable=False)
        providers._cache[self.model.key] = Broken()
        cache = self.run_once(1.0)
        self.assertTrue(all(r.get("error") for r in cache.for_model("fake", "top+wrist").values()))
        providers._cache[self.model.key] = self.fake
        cache = runner.Cache(self.root / "predictions.jsonl")
        runner.run([self.model], self.samples, self.root, cache, runner.Variant(), runner.Budget(1.0),
                   retry_errors=True, log=lambda *_: None)
        self.assertFalse(any(r.get("error") for r in cache.for_model("fake", "top+wrist").values()))

    def test_a_cap_leaves_every_model_the_same_samples(self):
        other = models.Model("fake2", models.OPENROUTER, "fake/model2", 1, 1, "test", estimate=0.01)
        providers._cache["fake2"] = FakeAdapter()
        cache = self.run_once(0.045, [self.model, other], workers=1)
        asked = [set(cache.for_model(m, "top+wrist")) for m in ("fake", "fake2")]
        self.assertEqual(asked[0], asked[1])
        self.assertEqual(len(asked[0]), 2)

    def test_a_call_without_usage_or_with_an_error_is_charged_the_estimate(self):
        providers._cache[self.model.key] = Scripted(None)
        self.run_once(0.035, workers=1)
        self.assertEqual(providers._cache[self.model.key].calls, 3)
        self.assertAlmostEqual(self.budget.spent, 0.03)
        providers._cache[self.model.key] = Scripted(bad_request())
        (self.root / "predictions.jsonl").unlink()
        self.run_once(0.025, workers=1)
        self.assertEqual(providers._cache[self.model.key].calls, 2)
        self.assertAlmostEqual(self.budget.spent, 0.02)

    def test_the_running_estimate_averages_answered_calls_only(self):
        # $0.02 answers between errors: each call after the first is reserved, and an error charged, at $0.02.
        providers._cache[self.model.key] = Scripted(0.02, bad_request(), 0.02, bad_request())
        self.run_once(0.05, workers=1)
        self.assertEqual(providers._cache[self.model.key].calls, 2)
        self.assertAlmostEqual(self.budget.spent, 0.04)

    def test_a_model_stops_after_errors_in_a_row_that_retrying_cant_fix(self):
        broken = Scripted(bad_request())
        providers._cache[self.model.key] = broken
        cache = self.run_once(1.0, workers=1)
        self.assertEqual(broken.calls, runner.GIVE_UP_AFTER)
        self.assertEqual(len(cache.for_model("fake", "top+wrist")), runner.GIVE_UP_AFTER)
        self.assertTrue(any("fake" in line and "stopped" in line for line in self.logged))
        flaky = Scripted(bad_request(), bad_request(), 0.01, bad_request(), bad_request(), 0.01)
        providers._cache[self.model.key] = flaky
        (self.root / "predictions.jsonl").unlink()
        self.run_once(1.0, workers=1)
        self.assertEqual(flaky.calls, 10, "an answer between errors starts the count again")

    def test_a_model_whose_adapter_cant_be_built_is_skipped_before_any_call(self):
        providers._cache.pop(self.model.key)

        def refuse(model):
            raise ImportError("No module named 'openai'")

        with mock.patch.dict(providers._ADAPTERS, {models.OPENROUTER: refuse}):
            cache = self.run_once(1.0)
        self.assertEqual(cache.rows, {}, "nothing is written for it")
        self.assertTrue(any("Skipping fake" in line and "openai" in line for line in self.logged))

    def test_a_stop_ends_the_run_and_cancels_what_is_queued(self):
        stop = threading.Event()
        providers._cache[self.model.key] = Scripted(0.01, 0.01, lambda: stop.set() or {"wrong": False, "cost_usd": 0.01})
        cache = self.run_once(1.0, workers=1, stop=stop)
        self.assertEqual(len(cache.for_model("fake", "top+wrist")), 3, "the call in flight is kept")
        self.assertTrue(self.result.interrupted)

    def test_a_stopping_run_doesnt_wait_out_the_backoff(self):
        stop = threading.Event()
        stop.set()
        busy = Scripted(providers.ProviderError("rate limited", retryable=True))
        started = time.monotonic()
        with self.assertRaises(providers.ProviderError):
            runner.call_with_retries(lambda: busy.predict([], None), first_wait=10, stop=stop)
        self.assertEqual(busy.calls, 1)
        self.assertLess(time.monotonic() - started, 1)
        providers._cache[self.model.key] = busy
        cache = self.run_once(1.0, workers=1, stop=stop)
        self.assertEqual(cache.rows, {}, "nothing was asked, so nothing is cached as an error")
        stop.clear()

        def stop_while_rate_limited():
            stop.set()
            raise providers.ProviderError("rate limited", retryable=True)

        first_call_stops = Scripted(stop_while_rate_limited)
        providers._cache[self.model.key] = first_call_stops
        cache = self.run_once(1.0, workers=1, stop=stop)
        self.assertEqual(first_call_stops.calls, 1)
        self.assertEqual(cache.rows, {}, "a call the stop cut short is left unasked, not cached as an error")

    def test_ctrl_c_stops_the_run_and_returns(self):
        def interrupt():  # Ctrl+C while a call is in flight, which takes longer than the main thread's wait
            _thread.interrupt_main()
            time.sleep(5 * runner.WAIT_S)
            return {"wrong": False, "p_wrong": 0.1, "cost_usd": 0.01}

        providers._cache[self.model.key] = Scripted(0.01, interrupt)
        cache = self.run_once(1.0, workers=1)
        self.assertEqual(len(cache.for_model("fake", "top+wrist")), 2, "the call in flight finishes and is kept")
        self.assertTrue(self.result.interrupted)
        self.assertTrue(any("Stopping" in line for line in self.logged))


class CommandTest(unittest.TestCase):
    """The command line, on a made-up dataset and a stand-in model."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images").mkdir(parents=True)
        self.samples = [sample(i, "wrong" if i % 2 else "ok", "missed_grasp" if i % 2 else "none") for i in range(6)]
        with open(self.dataset / "samples.jsonl", "w") as f:
            for s in self.samples:
                f.write(json.dumps(s.__dict__) + "\n")
                for cam in dataset.CAMERAS:
                    dataset.image_path(self.dataset, s, cam).write_bytes(b"jpeg")
        self.model = models.Model("stand-in", models.OPENROUTER, "x/y", 1, 1, "test", estimate=0.01)
        patches = [mock.patch.dict(models.MODELS, {"stand-in": self.model}),
                   mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(providers._cache.pop, "stand-in", None)

    def main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["--env-file", str(self.root / "none.env"), "--dataset", str(self.dataset), *args])
        return out.getvalue()

    def test_ctrl_c_still_writes_the_report(self):
        def interrupt():  # Ctrl+C while a call is in flight, which takes longer than the main thread's wait
            _thread.interrupt_main()
            time.sleep(5 * runner.WAIT_S)
            return {"wrong": True, "p_wrong": 0.9, "cost_usd": 0.01}

        providers._cache["stand-in"] = Scripted(interrupt)
        printed = self.main("run", "--models", "stand-in", "--workers", "1", "--out", str(self.root / "run"))
        report = (self.root / "run" / "report.md").read_text()
        self.assertIn("stand-in", report)
        self.assertIn("Stopped by Ctrl+C", report)
        self.assertIn("Report:", printed)

    def test_the_report_states_what_was_left_out(self):
        providers._cache["stand-in"] = Scripted(0.01)
        self.main("run", "--models", "stand-in", "--max-cost", "0.035", "--workers", "1", "--out", str(self.root / "run"))
        report = (self.root / "run" / "report.md").read_text()
        self.assertIn("3 of the 6 samples", report)

    def test_models_tells_a_group_from_a_provider(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.pick_models("gemini")
        self.assertIn("group:gemini", str(caught.exception))
        self.assertIn("provider:gemini", str(caught.exception))
        self.assertEqual({m.key for m in cli.pick_models("provider:gemini")}, {"gemini-robotics-er-2"})
        self.assertTrue(all(m.group == "gemini" for m in cli.pick_models("group:gemini")))
        self.assertEqual([m.key for m in cli.pick_models("jev")], ["jev"])
        self.assertTrue(all(m.group == "claude" for m in cli.pick_models("claude")), "a name that is only a group")

    def test_abbreviated_options_are_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.main("run", "--models", "stand-in", "--max", "0.01")
        self.assertEqual(caught.exception.code, 2)

    def test_check_s_help_says_it_bills(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.main(["check", "--help"])
        self.assertIn("bills", out.getvalue())

    def test_default_paths_come_from_the_settings(self):
        from so_101 import settings

        s = settings.load()
        paths = cli.Paths.of(s)
        self.assertEqual(paths.runs, s.data_dir / "block-container" / "runs")
        self.assertEqual(paths.env_file, s.path.parent / ".scratch" / "bench.env")
        self.assertEqual(paths.candidates, s.outputs_dir / "failbench" / "candidates.csv")

class ReadmeTest(unittest.TestCase):
    def test_the_readme_s_costs_match_the_models(self):
        readme = (Path(__file__).parent.parent / "README.md").read_text()
        per_sample = sum(m.estimate for m in models.MODELS.values())
        self.assertIn(f"${2 * per_sample:.2f} for all of them", readme, "check asks each model about two samples")
        self.assertIn(f"${100 * per_sample:.0f} per 100 samples", readme)


class EnvTest(unittest.TestCase):
    def test_env_file_does_not_override(self):
        from so_101.vision.env import load_env
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bench.env"
            p.write_text("# comment\nFAILBENCH_TEST_A=from_file\nFAILBENCH_TEST_B=\nFAILBENCH_TEST_C='x'\n"
                         "export FAILBENCH_TEST_D=\"y\"\n")
            os.environ["FAILBENCH_TEST_A"] = "already"
            try:
                names = load_env(p)
                self.assertEqual(os.environ["FAILBENCH_TEST_A"], "already")
                self.assertEqual(os.environ["FAILBENCH_TEST_C"], "x")
                self.assertEqual(os.environ["FAILBENCH_TEST_D"], "y")
                self.assertEqual(names, ["FAILBENCH_TEST_C", "FAILBENCH_TEST_D"])
                self.assertNotIn("FAILBENCH_TEST_B", os.environ)
            finally:
                for k in ("FAILBENCH_TEST_A", "FAILBENCH_TEST_C", "FAILBENCH_TEST_D"):
                    os.environ.pop(k, None)


# --- Candidates -----------------------------------------------------------------------------------

FPS = 30
NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
POLICY, WAITING, PLAYBACK, HOLD = 1, 2, 3, 4  # policy.base's drivers


def frames(seconds: float, pan: float, gripper: float, driver: int = POLICY, elbow: float = 0.0):
    n = round(seconds * FPS)
    return [[pan, 0.0, elbow, 0.0, 0.0, gripper]] * n, [driver] * n


def episode(*segments):
    """Autonomous frames from (states, drivers) segments, as a run's columns."""
    states = [row for s, _ in segments for row in s]
    drivers = [d for _, ds in segments for d in ds]
    return (np.ones(len(states), int), np.array([d != POLICY for d in drivers]), np.array(states, float),
            np.array(drivers))


class Recording(Detector):
    """The detector, recording each `resumed(moved, pose)` call."""

    calls: list = []

    def resumed(self, moved, pose):
        Recording.calls.append((moved, pose))
        Detector.resumed(self, moved, pose)


class CandidatesTest(unittest.TestCase):
    def setUp(self):
        from so_101.grasp import thresholds
        from so_101.supervisor import program

        self.program, self.grasp = program.load(), thresholds.load()

    def pick(self, phases, waiting, states, drivers):
        return candidates.pick("run1", 0, phases, waiting, states, NAMES, FPS, self.program, self.grasp, drivers=drivers)

    def test_a_layer_s_hold_is_left_out_of_the_replay(self):
        # Reach out with the jaws open, then the supervisor holds the arm there for 6 s: live, the
        # perception never sees the hold, so the hover rule can't fire on it.
        phases, waiting, states, drivers = episode(frames(2, 0, 40), frames(1, 30, 40), frames(6, 30, 40, HOLD),
                                                   frames(0.5, 31, 40))
        rows = self.pick(phases, waiting, states, drivers)
        self.assertNotIn("stuck", [r["moment"] for r in rows])
        old = self.pick(phases, waiting, states, None)
        self.assertNotIn("stuck", [r["moment"] for r in old],
                         "without the driver column, a hold sent no policy action, and hover counts only driven ticks")

    def test_no_candidate_lands_on_a_playback_and_the_detector_resumes_after_one(self):
        phases, waiting, states, drivers = episode(
            frames(2, 0, 40), frames(1, 20, 40), frames(0.5, 20, 10),  # reach and close on nothing
            frames(2.5, 0, 40, PLAYBACK),  # the supervisor plays it back
            frames(2, 20, 40), frames(1, 20, 18), frames(2, 20, 18, elbow=40), frames(1, 20, 40, elbow=40))
        Recording.calls = []
        with mock.patch("so_101.grasp.detector.Detector", Recording):
            rows = self.pick(phases, waiting, states, drivers)
        self.assertTrue(all(drivers[r["t"]] == POLICY for r in rows if r["moment"] != "end"))
        self.assertEqual([r["moment"] for r in rows].count("after_close"), 2)
        resumed_at = round(6 * FPS)
        self.assertEqual(len(Recording.calls), 1)
        moved, pose = Recording.calls[0]
        self.assertTrue(moved)
        self.assertEqual(pose, dict(zip(NAMES, states[resumed_at], strict=True)))

    def test_writing_again_keeps_every_label_by_id(self):
        row = lambda i, note="close 1": {"id": f"r-e0-t{i}", "src": "r", "ep": 0, "t": i, "moment": "after_close",  # noqa: E731
                                         "label": "", "category": "", "gripper": 9.0, "note": note}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candidates.csv"
            self.assertEqual(candidates.write([row(1), row(2), row(3)], path), 0)
            text = path.read_text()
            text = text.replace("r-e0-t1,r,0,1,after_close,,,9.0,close 1", "r-e0-t1,r,0,1,after_close,wrong,missed_grasp,9.0,mine")
            text = text.replace("r-e0-t3,r,0,3,after_close,,,9.0,close 1", "r-e0-t3,r,0,3,after_close,ok,none,9.0,held")
            path.write_text(text)
            self.assertEqual(candidates.write([row(1), row(2), row(4)], path), 2)
            rows = {s.id: s for s in dataset.read_labels(path)}
            self.assertEqual((rows["r-e0-t1"].label, rows["r-e0-t1"].note), ("wrong", "mine"))
            self.assertEqual(rows["r-e0-t3"].label, "ok", "a labelled row the rules no longer pick stays")
            self.assertEqual(dataset.unlabelled(path), 2, "t2 and t4 wait for a label")
            self.assertTrue(path.read_text().startswith("#"))


class RecordedRunTest(unittest.TestCase):
    """candidates and build on a run the fake arm recorded."""

    def test_candidates_then_build_from_a_recorded_run(self):
        from test_controller import Rig

        from so_101.grasp import thresholds
        from so_101.supervisor import program

        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder) / "runs"
            rig = Rig(runs / "r1")
            try:
                rig.start_policy(ticks=4 * FPS)  # past the 3 s at which a close-less attempt's approach is picked
                rig.reset("failed")
            finally:
                rig.close()
            rows = candidates.from_runs(runs, program.load(), thresholds.load())
            self.assertEqual({r["moment"] for r in rows}, {"approach", "end"})
            self.assertTrue(all(r["src"] == "r1" for r in rows))
            for r in rows:
                r["label"], r["category"] = "ok", "none"
            labels = Path(folder) / "labels.csv"
            candidates.write(rows, labels)
            out = Path(folder) / "dataset"
            self.assertEqual(dataset.build(out, [runs], labels), len(rows))
            loaded = dataset.load(out)
            self.assertEqual([s.id for s in loaded], [r["id"] for r in rows])
            from PIL import Image

            for s in loaded:
                for cam in dataset.CAMERAS:
                    with Image.open(dataset.image_path(out, s, cam)) as image:
                        self.assertEqual(image.size, (64, 48))


if __name__ == "__main__":
    unittest.main()
