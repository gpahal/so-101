"""so101-failbench without any network: the labels file, subsets, answer parsing, scoring, the runner's
cache, resume and spending cap, and reading the key file."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from so_101.failbench import dataset, metrics, runner
from so_101.vision import models, prompt, providers


def sample(i, label="ok", category="none", moment="end"):
    return dataset.Sample(id=f"s{i}", src="act_block-container_x", ep=0, t=i, moment=moment, label=label,
                          category=category, gripper=1.0, note="")


class LabelsTest(unittest.TestCase):
    def test_labels_file_is_valid(self):
        samples = dataset.read_labels()
        self.assertGreater(len(samples), 100)
        self.assertTrue(any(s.wrong for s in samples) and any(not s.wrong for s in samples))
        self.assertTrue({s.moment for s in samples} <= set(prompt.MOMENTS))

    def test_rejects_ok_with_a_category(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.csv"
            p.write_text("id,src,ep,t,moment,label,category,gripper,note\na,x,0,1,end,ok,missed_jar,1,\n")
            with self.assertRaises(ValueError):
                dataset.read_labels(p)

    def test_subset_is_deterministic_and_keeps_both_labels(self):
        samples = [sample(i, "wrong", "missed_grasp", "after_close") for i in range(30)] + [sample(100 + i) for i in range(70)]
        a, b = dataset.subset(samples, 20), dataset.subset(samples, 20)
        self.assertEqual([s.id for s in a], [s.id for s in b])
        self.assertTrue(5 <= sum(s.wrong for s in a) <= 7)
        self.assertEqual(len(dataset.subset(samples, None)), 100)


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


class FakeAdapter:
    def __init__(self, cost=0.01):
        self.cost, self.calls = cost, 0

    def predict(self, images, moment):
        self.calls += 1
        assert [cam for cam, _ in images] == ["top", "wrist"]
        return {"wrong": moment == "after_close", "p_wrong": 0.5, "category": "none", "cost_usd": self.cost, "latency_s": 0.01}


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
        self.dir.cleanup()

    def run_once(self, max_cost):
        cache = runner.Cache(self.root / "predictions.jsonl")
        runner.run([self.model], self.samples, self.root, cache, runner.Variant(), runner.Budget(max_cost),
                   workers=2, log=lambda *_: None)
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


class DescribeTest(unittest.TestCase):
    def test_the_two_camera_check_names_top_then_wrist(self):
        self.assertEqual(prompt.describe([("top", b""), ("wrist", b"")]),
                         "The first image is the TOP camera, the second the WRIST camera.")
        self.assertTrue(prompt.decision_context("end").endswith(prompt.describe(("top", "wrist"))))


class EnvTest(unittest.TestCase):
    def test_env_file_does_not_override(self):
        from so_101.vision.env import load_env
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bench.env"
            p.write_text("# comment\nFAILBENCH_TEST_A=from_file\nFAILBENCH_TEST_B=\nFAILBENCH_TEST_C='x'\n")
            os.environ["FAILBENCH_TEST_A"] = "already"
            try:
                names = load_env(p)
                self.assertEqual(os.environ["FAILBENCH_TEST_A"], "already")
                self.assertEqual(os.environ["FAILBENCH_TEST_C"], "x")
                self.assertEqual(names, ["FAILBENCH_TEST_C"])
                self.assertNotIn("FAILBENCH_TEST_B", os.environ)
            finally:
                for k in ("FAILBENCH_TEST_A", "FAILBENCH_TEST_C"):
                    os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
