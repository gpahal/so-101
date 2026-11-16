"""Skills without hardware: their definitions, the events and spans of made-up demos, a skills build from
demo sessions read back row by row, the skill layer switching on the carry, and the layer as serve's
policy on the fake arm.

The demos are 30 Hz traces in the follower's units (degrees, gripper 0 to 100) with the program's own
thresholds: reach with the jaws opening, close on the block at 18.4, lift it, carry it, let go.
"""

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from fakes import FakeCamera, FakeClock, make_follower

from so_101 import cli, data, remote, settings, skills
from so_101.build import build
from so_101.control.recorder import features
from so_101.skills import Point, SkillError, SkillPolicy
from so_101.supervisor import program
from so_101.tasks import Task

FPS = 30
PROGRAM = program.load()
NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
TABLE = {
    "pick": {"prompt": "Pick up the red block", "from": "start", "to": "carry + 0.5"},
    "place": {"prompt": "Put the red block in the cylindrical container", "from": "lift", "to": "widest + 0.2",
              "handover": "carry"},
}
SKILLS = skills.parse(TABLE, "test")
TASK = Task(name="test", prompt="test", cameras=("top", "wrist"), fps=FPS, episodes=1, episode_time_s=20,
            reset_time_s=5, props=[], setup=[], skills=SKILLS)


def trace(grasp: bool = True) -> np.ndarray:
    """A demo's joint readings, frames x NAMES: rest, reach opening the jaws, close on the block (or on
    nothing, back open), lift, carry, release, close again and return."""
    pose = np.array([0.0, -100.0, 90.0, 70.0, 0.0, 1.0])
    rows = [pose.copy()]

    def to(seconds: float, **target) -> None:
        start, end = rows[-1], rows[-1].copy()
        for name, value in target.items():
            end[NAMES.index(f"{name}.pos")] = value
        n = round(seconds * FPS)
        rows.extend(start + (end - start) * (i / n) for i in range(1, n + 1))

    to(0.5)
    to(2.0, shoulder_pan=20, gripper=40)
    to(0.5, gripper=18.4 if grasp else 10)
    if not grasp:
        to(0.5, gripper=40)
    to(0.5)
    to(1.5, shoulder_lift=-80)
    to(1.5, shoulder_pan=-10)
    to(0.5, gripper=45)
    to(0.5, gripper=1)
    to(1.0, shoulder_pan=0, shoulder_lift=-100)
    return np.array(rows, dtype=np.float32)


class DefinitionTest(unittest.TestCase):
    def test_skills_parse_in_order_with_their_points(self) -> None:
        self.assertEqual([s.name for s in SKILLS], ["pick", "place"])
        self.assertEqual(SKILLS[0].stop, Point("carry", 0.5))
        self.assertEqual(SKILLS[1].start, Point("lift"))
        self.assertEqual((SKILLS[0].handover, SKILLS[1].handover), (None, "carry"))
        self.assertEqual(str(Point("widest", 0.2)), "widest + 0.2")
        self.assertEqual(Point.parse("release - 1.5", "x"), Point("release", -1.5))

    def test_a_bad_definition_says_what_is_wrong(self) -> None:
        cases = {
            "an unknown event": ({**TABLE, "pick": {**TABLE["pick"], "to": "grasp + 1"}}, "Events:"),
            "a handover on the first skill": ({**TABLE, "pick": {**TABLE["pick"], "handover": "close"}}, "takes no handover"),
            "no handover on a later skill": ({**TABLE, "place": {k: v for k, v in TABLE["place"].items() if k != "handover"}},
                                             "needs a handover"),
            "a handover the policy can't see live": ({**TABLE, "place": {**TABLE["place"], "handover": "lift"}},
                                                     "needs a handover"),
            "two skills with one prompt": ({**TABLE, "place": {**TABLE["place"], "prompt": TABLE["pick"]["prompt"]}},
                                           "its own prompt"),
            "an unknown key": ({**TABLE, "pick": {**TABLE["pick"], "until": "end"}}, "unknown keys"),
        }
        for case, (table, message) in cases.items():
            with self.subTest(case), self.assertRaisesRegex(SkillError, message):
                skills.parse(table, "test")


class CutTest(unittest.TestCase):
    def test_a_grasp_has_every_event_in_order_and_both_spans_hold_the_handover(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, PROGRAM)
        order = [at[e] for e in skills.EVENTS]
        self.assertEqual(order, sorted(order))
        self.assertEqual((at["start"], at["end"]), (0, len(state) - 1))
        self.assertLess(state[at["close"], 5], PROGRAM.gripper.closed_below)
        self.assertAlmostEqual((at["carry"] - at["close"]) / FPS, PROGRAM.gripper.hold_s, delta=0.15)
        self.assertEqual(state[at["widest"], 5], state[at["release"]:, 5].max())
        spans = skills.spans(SKILLS, at, FPS, len(state))
        self.assertEqual(spans["pick"], (0, at["carry"] + round(0.5 * FPS) + 1))
        self.assertEqual(spans["place"], (at["lift"], at["widest"] + round(0.2 * FPS) + 1))
        self.assertTrue(spans["place"][0] <= at["carry"] < spans["pick"][1], "the handover is inside both")

    def test_a_demo_that_never_holds_the_block_is_refused(self) -> None:
        with self.assertRaisesRegex(SkillError, "no carry"):
            skills.events(trace(grasp=False), NAMES, FPS, PROGRAM)

    def test_a_handover_outside_the_spans_it_joins_is_refused(self) -> None:
        at = skills.events(trace(), NAMES, FPS, PROGRAM)
        late = skills.parse({**TABLE, "place": {**TABLE["place"], "from": "release"}}, "test")
        with self.assertRaisesRegex(SkillError, "isn't inside both"):
            skills.spans(late, at, FPS, len(trace()))
        short = skills.parse({**TABLE, "place": {**TABLE["place"], "to": "lift + 0.1"}}, "test")
        with self.assertRaisesRegex(SkillError, "under 0.5 s"):
            skills.spans(short, at, FPS, len(trace()))


def grey(self, *args, **kwargs) -> np.ndarray:
    return np.full((self.height, self.width, 3), 128, dtype=np.uint8)


def make_session(root: Path, states: list[np.ndarray]) -> None:
    """A demo session, one episode per state trace, with only the columns `lerobot-record` writes."""
    from lerobot.datasets import LeRobotDataset

    from test_controller import REST_STEPS

    robot, _ = make_follower(FakeClock(), REST_STEPS)
    try:
        demo = {k: f for k, f in features(robot).items() if k.startswith(("observation.", "action"))}
        dataset = LeRobotDataset.create("local/demo", FPS, demo, root=root, robot_type=robot.robot_type,
                                        streaming_encoding=False)
    finally:
        robot.inner.disconnect()
    for state in states:
        for row in state:
            image = np.full((48, 64, 3), 128, dtype=np.uint8)
            dataset.add_frame({"observation.state": row, "action": row, "task": "test",
                               "observation.images.top": image, "observation.images.wrist": image})
        dataset.save_episode()
    dataset.finalize()


def table(root: Path) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted(root.glob("data/*/*.parquet"))).sort_values("index")


class BuildTest(unittest.TestCase):
    def setUp(self) -> None:
        for name in ("read_latest", "read", "async_read"):
            patch = mock.patch.object(FakeCamera, name, grey)
            patch.start()
            self.addCleanup(patch.stop)
        temp = tempfile.TemporaryDirectory(prefix="so101-skills-")
        self.addCleanup(temp.cleanup)
        self.s = dataclasses.replace(settings.load(), data_dir=Path(temp.name))
        self.states = [trace(), trace(grasp=False), trace()]
        make_session(data.task_dir(self.s, TASK) / "sessions" / "20260101-100000" / "dataset", self.states)

    def test_each_demo_s_skills_become_episodes_with_their_prompt_and_one_hot(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            build(self.s, TASK, kind=data.SKILLS)
        root = data.dataset_root(self.s, TASK, data.SKILLS)
        manifest = json.loads((root / "build.json").read_text())
        self.assertEqual(manifest, json.loads(data.build_manifest_path(self.s, TASK, data.SKILLS).read_text()))
        # The demo that never held the block is left out, with a note; the other two give two episodes each.
        self.assertEqual([(e["episode"], e["skill"]) for e in manifest["episodes"]],
                         [(0, "pick"), (0, "place"), (2, "pick"), (2, "place")])
        self.assertEqual(len([n for n in manifest["notes"] if "episode 1: left out" in n and "no carry" in n]), 1)
        self.assertEqual(manifest["inputs"]["dataset"], data.SKILLS)
        self.assertEqual([c.get("error") is None for c in manifest["cuts"]], [True, False, True])
        built = table(root)
        info = json.loads((root / "meta" / "info.json").read_text())
        self.assertEqual(info["features"][skills.STATE_KEY]["names"], ["skill.pick", "skill.place"])
        tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
        prompt = dict(zip(tasks["task_index"], tasks.index, strict=True))
        for n, entry in enumerate(manifest["episodes"]):
            rows = built[built["episode_index"] == n]
            a, b = entry["frames"]
            np.testing.assert_array_equal(np.stack(rows["observation.state"]), self.states[entry["episode"]][a:b])
            index = [s.name for s in SKILLS].index(entry["skill"])
            np.testing.assert_array_equal(np.stack(rows[skills.STATE_KEY]),
                                          np.tile(skills.one_hot(SKILLS, index), (b - a, 1)))
            self.assertEqual({prompt[int(t)] for t in rows["task_index"]}, {SKILLS[index].prompt})
        # With no whole dataset, each attempt's start pose comes from the pick episodes: the demos' rest pose.
        start = data.demo_start_pose(self.s, TASK)
        np.testing.assert_allclose(list(start.values()), np.round(self.states[0][0], 1))
        self.assertTrue(data.build_is_current(self.s, TASK, kind=data.SKILLS))
        self.assertFalse(data.build_is_current(self.s, TASK), "the whole dataset is its own build")
        moved = dataclasses.replace(TASK, skills=skills.parse({**TABLE, "pick": {**TABLE["pick"], "to": "carry + 1"}}, "test"))
        self.assertFalse(data.build_is_current(self.s, moved, kind=data.SKILLS), "a changed skill rebuilds")

    def test_a_skills_build_takes_the_demos_only(self) -> None:
        with self.assertRaisesRegex(Exception, "demos only"):
            build(self.s, TASK, ("corrections",), kind=data.SKILLS)


class FakeEngine:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def set_task(self, task: str) -> bool:
        self.tasks.append(task)
        return True


class FakeDriver:
    def __init__(self) -> None:
        self.engine = FakeEngine()
        self.calls: list[str] = []
        self.raised = self.retrying = self.failure = None
        self.dataset_features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES}}
        self.observation_processor = lambda observation: dict(observation)

    def __getattr__(self, name):
        if name in ("start", "stop", "pause", "run", "begin_episode", "end_episode"):
            return lambda *args: self.calls.append(name)
        raise AttributeError(name)

    def asks(self):
        return None

    def maintain(self):
        return None

    def step(self, robot, observation):
        return {"action": 1}


class LayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logged: list[dict] = []
        self.now = 0.0
        self.driver = FakeDriver()
        self.layer = SkillPolicy(self.driver, SKILLS, PROGRAM, log=lambda type_, **f: self.logged.append(f),
                                 clock=lambda: self.now)
        self.layer.begin_episode(1)
        self.layer.run()

    def feed(self, state: np.ndarray) -> None:
        for row in state:
            self.now += 1 / FPS
            self.layer.step(None, dict(zip(NAMES, map(float, row), strict=True)))

    def test_it_starts_on_the_first_skill_and_switches_once_at_the_handover(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, PROGRAM)
        self.assertEqual(self.driver.engine.tasks, [SKILLS[0].prompt])
        self.assertEqual(self.layer.values(), {"skill.pick": 1.0, "skill.place": 0.0})
        self.feed(state[: at["carry"]])
        self.assertEqual(self.layer.skill.name, "pick")
        runs = self.driver.calls.count("run")
        self.feed(state[at["carry"]:])
        self.assertEqual(self.layer.skill.name, "place")
        self.assertEqual(self.driver.engine.tasks, [SKILLS[0].prompt, SKILLS[1].prompt])
        self.assertEqual(self.driver.calls.count("run"), runs + 1, "the switch clears the policy's queue")
        self.assertEqual(self.layer.values(), {"skill.pick": 0.0, "skill.place": 1.0})
        self.assertEqual([(e["skill"], e.get("previous"), e["cause"]) for e in self.logged],
                         [("pick", None, "start"), ("place", "pick", "carry")])

    def test_a_new_attempt_starts_on_the_first_skill_again(self) -> None:
        self.feed(trace())
        self.layer.begin_episode(2)
        self.assertEqual(self.layer.skill.name, "pick")
        self.assertEqual(self.driver.engine.tasks[-1], SKILLS[0].prompt)

    def test_the_one_hot_reaches_the_frame_lerobot_builds_for_the_policy(self) -> None:
        from lerobot.utils.feature_utils import build_dataset_frame

        observation = dict(zip(NAMES, map(float, trace()[0]), strict=True))
        frame = build_dataset_frame(self.driver.dataset_features, self.driver.observation_processor(observation),
                                    prefix="observation")
        np.testing.assert_array_equal(frame[skills.STATE_KEY], [1.0, 0.0])
        self.feed(trace())
        frame = build_dataset_frame(self.driver.dataset_features, self.driver.observation_processor(observation),
                                    prefix="observation")
        np.testing.assert_array_equal(frame[skills.STATE_KEY], [0.0, 1.0])


class InServeTest(unittest.TestCase):
    """The layer as serve's policy on the controller's fake arm, recording a run."""

    def test_serve_records_the_switch_and_the_engine_gets_the_new_prompt(self) -> None:
        from test_controller import START_POSE, Rig

        # Open the jaws, close them on the block at 18.4, and hold it there.
        opened = [{**START_POSE, "gripper.pos": min(10.0 + 2 * i, 40.0)} for i in range(1, 31)]
        closed = [{**opened[-1], "gripper.pos": max(40.0 - 3 * i, 18.4)} for i in range(1, 11)]
        actions = [[a[k] for k in START_POSE] for a in opened + closed]
        with tempfile.TemporaryDirectory() as folder:
            rig = Rig(Path(folder) / "run", actions=actions,
                      wrap=lambda driver, recorder, clock: SkillPolicy(driver, SKILLS, PROGRAM, log=recorder.event, clock=clock))
            rig.start_policy(ticks=4 * FPS)
            rig.reset("failed")
            rig.close()
            switches = [(e["skill"], e["cause"]) for e in rig.events("skill")]
            self.assertEqual(switches, [("pick", "start"), ("place", "carry")])
            self.assertEqual(rig.engine.task, SKILLS[1].prompt)


class CommandTest(unittest.TestCase):
    def test_skills_models_and_datasets_have_their_own_hub_names(self) -> None:
        with mock.patch("so_101.tasks.hf_user", return_value="me"):
            self.assertEqual(TASK.policy_for("act"), "me/act_so101_test")
            self.assertEqual(TASK.policy_for("pi05", skills=True), "me/pi05_so101_test_skills")
            self.assertEqual((TASK.dataset, TASK.dataset_repo(skills=True)), ("me/so101_test", "me/so101_test_skills"))

    def test_local_training_of_a_skills_model_is_named_for_it(self) -> None:
        s = mock.Mock(outputs_dir=Path("/outputs"), run=mock.Mock(device="mps"))
        args, output = cli.local_train_args(s, TASK, "act", Path("/data/datasets/skills"), skills=True)
        self.assertIn("--dataset.repo_id=local/so101_test_skills", args)
        self.assertTrue(output.name.startswith("act_so101_test_skills_"))

    def test_a_server_for_one_model_never_serves_a_run_of_the_other(self) -> None:
        whole = remote._labels("test", "pi05", "main", 50, True)
        job = mock.Mock(labels={"so101": remote.SERVER_LABEL, **whole}, status=mock.Mock(stage="RUNNING"), id="j",
                        flavor="h200")
        with mock.patch.object(remote, "active_server", return_value=job):
            self.assertIs(remote.require_server("test", "pi05", "main", 50, True), job)
            with self.assertRaises(SystemExit) as stop:
                remote.require_server("test", "pi05", "main", 50, True, "skills")
            self.assertIn("--skills", str(stop.exception))
            job.labels = {k: v for k, v in job.labels.items() if k != "dataset"}  # started before the label
            self.assertIs(remote.require_server("test", "pi05", "main", 50, True), job)

    def test_a_task_without_skills_says_how_to_add_them(self) -> None:
        with self.assertRaisesRegex(SystemExit, r"\[test.skills\]"):
            dataclasses.replace(TASK, skills=()).require_skills()


if __name__ == "__main__":
    unittest.main()
