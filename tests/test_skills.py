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
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from fakes import FakeCamera, FakeClock, make_follower

from so_101 import cli, data, remote, settings, skills
from so_101.build import build
from so_101.control.recorder import features
from so_101.policy import stack
from so_101.policy.base import (LAYER, PERSON, POLICY, START, Conditioning, Handback, Health,
                                 LayerContext, Output, Tick)
from so_101.skills import Point, SkillError
from so_101.supervisor import judges as J
from so_101.grasp import thresholds
from so_101.supervisor import program
from so_101.tasks import Task

FPS = 30
PROGRAM = program.load()
GRASP = thresholds.load()
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
        at = skills.events(state, NAMES, FPS, GRASP)
        order = [at[e] for e in skills.EVENTS]
        self.assertEqual(order, sorted(order))
        self.assertEqual((at["start"], at["end"]), (0, len(state) - 1))
        self.assertLess(state[at["close"], 5], GRASP.gripper.closed_below)
        self.assertAlmostEqual((at["carry"] - at["close"]) / FPS, GRASP.gripper.hold_s, delta=0.15)
        self.assertEqual(state[at["widest"], 5], state[at["release"]:, 5].max())
        spans = skills.spans(SKILLS, at, FPS, len(state))
        self.assertEqual(spans["pick"], (0, at["carry"] + round(0.5 * FPS) + 1))
        self.assertEqual(spans["place"], (at["lift"], at["widest"] + round(0.2 * FPS) + 1))
        self.assertTrue(spans["place"][0] <= at["carry"] < spans["pick"][1], "the handover is inside both")

    def test_a_demo_that_never_holds_the_block_is_refused(self) -> None:
        with self.assertRaisesRegex(SkillError, "no carry"):
            skills.events(trace(grasp=False), NAMES, FPS, GRASP)

    def test_a_handover_outside_the_spans_it_joins_is_refused(self) -> None:
        at = skills.events(trace(), NAMES, FPS, GRASP)
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

    def record_run(self, name: str, with_skills: bool) -> Path:
        """A serve run on the fake arm that closes on the block, holds it past the carry and succeeds:
        through the perception alone (a whole-model run), or with the skill sequencer too."""
        from test_controller import START_POSE, Rig

        opened = [{**START_POSE, "gripper.pos": min(10.0 + 2 * i, 40.0)} for i in range(1, 31)]
        closed = [{**opened[-1], "gripper.pos": max(40.0 - 3 * i, 18.4)} for i in range(1, 11)]
        actions = [[a[k] for k in START_POSE] for a in opened + closed]
        run_dir = data.task_dir(self.s, TASK) / "runs" / name
        rig = Rig(run_dir, actions=actions, wrap=lambda driver, recorder, clock: stack.build(
            driver, LayerContext(recorder.event, clock, GRASP, PROGRAM), skills=SKILLS if with_skills else ()))
        rig.start_policy(ticks=5 * FPS)
        rig.reset("succeeded")
        rig.close()
        policy = {"skills": [skill.record() for skill in SKILLS]} if with_skills else {}
        (run_dir / "run.json").write_text(json.dumps({"policy": policy}))
        return run_dir

    def test_run_cuts_are_split_where_the_skill_changes(self) -> None:
        for name, with_skills in (("20260101-110000", False), ("20260101-120000", True)):
            with self.subTest("skills run" if with_skills else "whole-model run"):
                run_dir = self.record_run(name, with_skills)
                with contextlib.redirect_stdout(io.StringIO()):
                    build(self.s, TASK, ("policy-successes",), kind=data.SKILLS)
                manifest = json.loads(data.build_manifest_path(self.s, TASK, data.SKILLS).read_text())
                pieces = [e for e in manifest["episodes"] if e.get("run") == name]
                self.assertEqual([e["skill"] for e in pieces], ["pick", "place"])
                events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
                carry = next(e["frame"] for e in events if e.get("type") == "grasp" and e["grasp"] == "carry")
                self.assertEqual(pieces[0]["frames"][1], pieces[1]["frames"][0], "the pieces meet")
                self.assertLessEqual(abs(pieces[1]["frames"][0] - carry), 1, "they meet at the carry")
                built = table(data.dataset_root(self.s, TASK, data.SKILLS))
                n = manifest["episodes"].index(pieces[1])
                np.testing.assert_array_equal(np.stack(built[built["episode_index"] == n][skills.STATE_KEY]).max(axis=0),
                                              [0.0, 1.0])
                shutil.rmtree(run_dir)


class FakeLeaf:
    """The model at the bottom of the stack: notes what it's told and handed; each step drives."""

    def __init__(self) -> None:
        self.conditions: list[Conditioning] = []
        self.handbacks: list[Handback] = []
        self.calls: list[str] = []
        self.failure: str | None = None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def begin_episode(self, number: int) -> None: ...
    def end_episode(self) -> None: ...
    def hold(self, robot) -> None: ...

    def pause(self) -> None:
        self.calls.append("pause")

    def run(self, handback: Handback) -> None:
        self.calls.append("run")
        self.handbacks.append(handback)

    def condition(self, conditioning: Conditioning) -> None:
        self.conditions.append(conditioning)

    def maintain(self) -> list[str]:
        return []

    def health(self) -> Health:
        return Health(self.failure)

    def record_failure(self, detail: str) -> None:
        self.failure = detail

    def step(self, robot, tick: Tick) -> Output:
        return Output({"shoulder_pan.pos": tick.pose["shoulder_pan.pos"]}, POLICY)


class Stack:
    """Perception and the skill sequencer over a FakeLeaf, stepped by hand as serve steps the stack."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.now, self.frame = 0.0, -1
        self.leaf = FakeLeaf()
        ctx = LayerContext(lambda type_, **fields: self.events.append({"type": type_, **fields}), lambda: self.now, GRASP, PROGRAM)
        self.top = stack.build(self.leaf, ctx, skills=SKILLS)
        self.top.begin_episode(1)
        self.top.run(Handback(START))
        self.outputs: list[Output] = []

    def feed(self, state: np.ndarray) -> None:
        for row in state:
            self.now += 1 / FPS
            self.frame += 1
            pose = dict(zip(NAMES, map(float, row), strict=True))
            self.outputs.append(self.top.step(None, Tick(self.now, self.frame, pose, pose)))

    def switches(self) -> list[tuple]:
        return [(e["skill"], e["cause"]) for e in self.events if e["type"] == "skill"]


class SequencerTest(unittest.TestCase):
    def test_it_starts_on_the_first_skill_and_switches_once_at_the_carry(self) -> None:
        h = Stack()
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["carry"]])
        self.assertEqual([c.prompt for c in h.leaf.conditions], [SKILLS[0].prompt])
        self.assertEqual(h.leaf.conditions[0].values(), {"skill.pick": 1.0, "skill.place": 0.0})
        h.feed(state[at["carry"]:])
        self.assertEqual([c.prompt for c in h.leaf.conditions], [SKILLS[0].prompt, SKILLS[1].prompt])
        self.assertEqual(h.leaf.conditions[1].values(), {"skill.pick": 0.0, "skill.place": 1.0})
        self.assertEqual(h.switches(), [("pick", "start"), ("place", "carry")])
        skills_seen = [o.skill for o in h.outputs]
        self.assertEqual(skills_seen[: at["carry"]], [0] * at["carry"], "each Output names the current skill")
        self.assertEqual(skills_seen[-1], 1)

    def test_the_perception_logs_every_grasp_for_the_run(self) -> None:
        h = Stack()
        h.feed(trace())
        grasps = [e["grasp"] for e in h.events if e["type"] == "grasp"]
        self.assertEqual(grasps, ["close", "carry", "release"])

    def test_a_handback_after_the_arm_moved_with_the_jaws_open_goes_back_to_the_first_skill(self) -> None:
        h = Stack()
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["carry"] + 10])  # on place now
        h.top.pause()
        h.top.run(Handback(LAYER, moved=True))  # a playback undid the grasp: the jaws are open again
        h.feed(state[at["close"] - 5: at["close"] - 2])  # the reach before the close, jaws open
        self.assertEqual(h.switches()[-1], ("pick", "reopened"))
        self.assertEqual(h.leaf.conditions[-1].prompt, SKILLS[0].prompt)

    def test_a_handback_with_the_block_still_held_keeps_the_skill(self) -> None:
        h = Stack()
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["carry"] + 10])
        h.top.pause()
        h.top.run(Handback(PERSON, moved=True))
        h.feed(state[at["carry"] + 10: at["carry"] + 15])  # still holding at 18.4
        self.assertEqual(h.switches(), [("pick", "start"), ("place", "carry")])

    def test_a_playback_that_undoes_the_grasp_sends_the_skill_back_to_pick(self) -> None:
        """The gap the uniform design closes: the carry switches to place, then the after-close check, asked
        after the carry, finds the jaws empty, and the supervisor plays back to before the close."""
        from test_supervisor import FakeModel, FakeEscalation, Pool

        events, now = [], [0.0]
        leaf = FakeLeaf()
        # The check looks only after the carry has switched skills (an empty carry), and both judges call
        # it wrong; the first playback reaches back past the close.
        program_ = dataclasses.replace(PROGRAM, after_close_s=2.5, miss_rewind_s=(3.0, 5.0))
        pool = Pool()
        made = J.Judges(program_, adapters={"jev": FakeModel({"after_close": 0.95}),
                                            "gemini-3.8-flash": FakeModel({"after_close": 0.95}),
                                            "claude-opus-5.5": FakeEscalation()}, pool=pool)
        ctx = LayerContext(lambda type_, **fields: events.append({"type": type_, **fields}), lambda: now[0], GRASP, program_)
        top = stack.build(leaf, ctx, skills=SKILLS, judges=made, mode="act", cameras=("top", "wrist"))
        top.begin_episode(1)
        top.run(Handback(START))
        frame = np.zeros((48, 64, 3), np.uint8)
        robot = types.SimpleNamespace(goals=[], send_action=lambda goal: robot.goals.append(dict(goal)))
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        pose = None
        for n, row in enumerate(state[: at["carry"] + 5]):
            now[0] += 1 / FPS
            pose = dict(zip(NAMES, map(float, row), strict=True))
            top.step(robot, Tick(now[0], n, {**pose, "top": frame, "wrist": frame}, pose))
        self.assertEqual(top.inner.skill.name, "place", "the carry switched to place first")
        n = len(state)
        for _ in range(6 * FPS):  # the playback runs, then the stack has the arm again
            now[0] += 1 / FPS
            n += 1
            pose = robot.goals[-1] if top.doing != "pass" and robot.goals else pose
            top.maintain()
            top.step(robot, Tick(now[0], n, {**pose, "top": frame, "wrist": frame}, pose))
            if top.doing == "pass" and any(e["type"] == "playback" for e in events):
                break
        top.step(robot, Tick(now[0] + 1 / FPS, n + 1, {**pose, "top": frame, "wrist": frame}, pose))
        self.assertEqual([e["type"] for e in events if e["type"] == "playback"], ["playback"])
        self.assertEqual([(e["skill"], e["cause"]) for e in events if e["type"] == "skill"],
                         [("pick", "start"), ("place", "carry"), ("pick", "reopened")])
        self.assertEqual(leaf.conditions[-1].prompt, SKILLS[0].prompt)


class EngineConditioningTest(unittest.TestCase):
    def test_the_model_gets_the_prompt_and_the_one_hot_in_the_frame_it_builds(self) -> None:
        from fakes import make_engine

        from so_101.policy.engine import EngineDriver

        engine = make_engine([[0.0] * 6] * 10)
        seen = []
        get_action = engine.get_action
        engine.get_action = lambda frame: (seen.append(frame), get_action(frame))[1]
        driver = EngineDriver(engine, NAMES, dataset_features={
            "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES}})
        robot = types.SimpleNamespace(supports_hold=False, send_action=lambda goal: None)
        driver.condition(Conditioning(SKILLS[0].prompt, ("pick", "place"), 0))
        driver.run(Handback(START))
        pose = dict(zip(NAMES, map(float, trace()[0]), strict=True))
        driver.step(robot, Tick(0.0, 0, pose, pose))
        self.assertEqual(engine.task, SKILLS[0].prompt)
        np.testing.assert_array_equal(seen[-1][skills.STATE_KEY], [1.0, 0.0])
        driver.condition(Conditioning(SKILLS[1].prompt, ("pick", "place"), 1))
        driver.step(robot, Tick(0.03, 1, pose, pose))
        self.assertEqual(engine.task, SKILLS[1].prompt)
        np.testing.assert_array_equal(seen[-1][skills.STATE_KEY], [0.0, 1.0])


class InServeTest(unittest.TestCase):
    """The stack as serve's policy on the controller's fake arm, recording a run."""

    def test_serve_records_the_switch_and_each_frame_s_skill(self) -> None:
        from test_controller import START_POSE, Rig

        # Open the jaws, close them on the block at 18.4, and hold it there.
        opened = [{**START_POSE, "gripper.pos": min(10.0 + 2 * i, 40.0)} for i in range(1, 31)]
        closed = [{**opened[-1], "gripper.pos": max(40.0 - 3 * i, 18.4)} for i in range(1, 11)]
        actions = [[a[k] for k in START_POSE] for a in opened + closed]
        with tempfile.TemporaryDirectory() as folder:
            rig = Rig(Path(folder) / "run", actions=actions,
                      wrap=lambda driver, recorder, clock: stack.build(driver, LayerContext(recorder.event, clock, GRASP, PROGRAM),
                                                                       skills=SKILLS))
            rig.start_policy(ticks=4 * FPS)
            rig.reset("failed")
            rig.close()
            switches = [(e["skill"], e["cause"]) for e in rig.events("skill")]
            self.assertEqual(switches, [("pick", "start"), ("place", "carry")])
            self.assertEqual(rig.engine.task, SKILLS[1].prompt)
            self.assertEqual([e["grasp"] for e in rig.events("grasp")][:2], ["close", "carry"])
            rows = table(rig.run_dir / "dataset").sort_values("frame_index")
            skill = np.ravel(np.stack(rows["control.skill"]))
            switch = next(e for e in rig.events("skill") if e["cause"] == "carry")["frame"]
            self.assertTrue((skill[switch:][np.ravel(np.stack(rows["control.phase"]))[switch:] == 1] == 1).all())
            self.assertTrue((skill[: switch - 1] <= 0).all())


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
