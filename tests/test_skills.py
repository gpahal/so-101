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
RETURN = {"prompt": "Return to the rest pose", "from": "release", "to": "end", "handover": "release"}
THREE = skills.parse({**TABLE, "return": RETURN}, "test")
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
            # At runtime a close that misses reopens, and a switch on it would already have told the model to place.
            "a handover on the close": ({**TABLE, "place": {**TABLE["place"], "handover": "close"}},
                                        "can't hand over on the close"),
        }
        for case, (table, message) in cases.items():
            with self.subTest(case), self.assertRaisesRegex(SkillError, message):
                skills.parse(table, "test")


    def test_block_container_returns_to_rest_as_its_third_skill_switched_to_at_the_release(self) -> None:
        from so_101 import tasks

        found = tasks.load()["block-container"].skills
        self.assertEqual([skill.name for skill in found], ["pick", "place", "return"])
        self.assertEqual((found[2].start, found[2].stop, found[2].handover), (Point("release"), Point("end"), "release"))
        self.assertEqual(found[2].prompt, "Return to the rest pose")


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

    def test_the_return_runs_from_the_release_to_the_end_overlapping_place(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        spans = skills.spans(THREE, at, FPS, len(state))
        self.assertEqual(spans["return"], (at["release"], len(state)))
        self.assertTrue(spans["place"][0] <= at["release"] < spans["place"][1], "place runs on past the release")

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


class StalenessTest(unittest.TestCase):
    """A skills build is current only while what cuts it is unchanged, by content, never by path."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="so101-stale-")
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.s = dataclasses.replace(settings.load(), data_dir=self.folder / "data")
        root = data.dataset_root(self.s, TASK, data.SKILLS)
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}")
        self.inputs = data.build_inputs(self.s, TASK, kind=data.SKILLS)
        data.build_manifest_path(self.s, TASK, data.SKILLS).write_text(json.dumps({"inputs": self.inputs}))

    def thresholds(self, old: str, new: str) -> settings.Settings:
        """Settings whose [perception] thresholds file is the package's with `old` replaced by `new`."""
        text = thresholds.DEFAULT.read_text()
        self.assertIn(old, text)
        path = self.folder / f"grasp-{len(list(self.folder.glob('grasp-*')))}.toml"
        path.write_text(text.replace(old, new, 1))
        return dataclasses.replace(self.s, perception=dataclasses.replace(self.s.perception, thresholds=path))

    def test_the_record_holds_no_path(self) -> None:
        text = json.dumps(self.inputs)
        self.assertNotIn(str(self.folder), text)
        self.assertNotIn(str(thresholds.DEFAULT.parent), text)
        self.assertEqual(self.inputs["gripper_sha256"], skills.gripper_hash(GRASP.gripper))
        self.assertEqual(self.inputs["cut_rules"], skills.CUT_RULES_VERSION)

    def test_each_input_that_changes_a_cut_makes_the_build_stale(self) -> None:
        self.assertTrue(data.build_is_current(self.s, TASK, kind=data.SKILLS))
        renamed = {"grab" if k == "pick" else k: v for k, v in TABLE.items()}
        changes = {
            "a gripper threshold": (self.thresholds("hold_s = 1.5", "hold_s = 1.6"), TASK),
            "the gripper table's version": (self.thresholds("[gripper]\nversion = 1", "[gripper]\nversion = 2"), TASK),
            "a skill's name": (self.s, dataclasses.replace(TASK, skills=skills.parse(renamed, "test"))),
            "a skill's prompt": (self.s, dataclasses.replace(TASK, skills=skills.parse(
                {**TABLE, "pick": {**TABLE["pick"], "prompt": "Grab the red block"}}, "test"))),
            "a skill's bound": (self.s, dataclasses.replace(TASK, skills=skills.parse(
                {**TABLE, "place": {**TABLE["place"], "to": "widest + 0.3"}}, "test"))),
            "a skill's handover": (self.s, dataclasses.replace(TASK, skills=skills.parse(
                {**TABLE, "place": {**TABLE["place"], "handover": "release", "from": "carry"}}, "test"))),
            "the skills' order": (self.s, dataclasses.replace(TASK, skills=THREE)),
        }
        for change, (s, task) in changes.items():
            with self.subTest(change):
                self.assertFalse(data.build_is_current(s, task, kind=data.SKILLS))
        with self.subTest("the cut rules"), mock.patch.object(skills, "CUT_RULES_VERSION", skills.CUT_RULES_VERSION + 1):
            self.assertFalse(data.build_is_current(self.s, TASK, kind=data.SKILLS))

    def test_the_run_labelling_version_stales_only_a_build_with_runs(self) -> None:
        self.assertNotIn("run_labels", self.inputs, "a demos-only build labels no run")
        runs = ("policy-successes",)
        with_runs = data.build_inputs(self.s, TASK, runs, kind=data.SKILLS)
        self.assertEqual(with_runs["run_labels"], skills.RUN_LABEL_VERSION)
        with mock.patch.object(skills, "RUN_LABEL_VERSION", skills.RUN_LABEL_VERSION + 1):
            self.assertTrue(data.build_is_current(self.s, TASK, kind=data.SKILLS), "a demos-only build stays current")
        data.build_manifest_path(self.s, TASK, data.SKILLS).write_text(json.dumps({"inputs": with_runs}))
        self.assertTrue(data.build_is_current(self.s, TASK, runs, kind=data.SKILLS))
        with mock.patch.object(skills, "RUN_LABEL_VERSION", skills.RUN_LABEL_VERSION + 1):
            self.assertFalse(data.build_is_current(self.s, TASK, runs, kind=data.SKILLS), "a build with runs goes stale")
        # A skills build with runs from before run_labels was recorded is stale too.
        before = {k: v for k, v in with_runs.items() if k != "run_labels"}
        data.build_manifest_path(self.s, TASK, data.SKILLS).write_text(json.dumps({"inputs": before}))
        self.assertFalse(data.build_is_current(self.s, TASK, runs, kind=data.SKILLS))

    def test_the_build_records_the_skills_contract_a_model_is_checked_against(self) -> None:
        for runs in ((), ("policy-successes",)):
            with self.subTest(runs=runs):
                inputs = data.build_inputs(self.s, TASK, runs, kind=data.SKILLS)
                want = skills.contract(TASK.skills, GRASP.gripper, runs=bool(runs))
                self.assertEqual({k: inputs.get(k) for k in want}, want)

    def test_the_stuck_rules_and_where_the_thresholds_live_change_no_build(self) -> None:
        for change, s in {"a stuck rule": self.thresholds("seconds = 14.0", "seconds = 20.0"),
                          "the same thresholds in another file": self.thresholds("hold_s = 1.5", "hold_s = 1.5")}.items():
            with self.subTest(change):
                self.assertTrue(data.build_is_current(s, TASK, kind=data.SKILLS))


class RunLabelTest(unittest.TestCase):
    """Each frame of a run's attempt labelled by what happened (`skills.label_run`), as a demo is cut."""

    def label(self, state, skills_=SKILLS, scripted=None, told=None, teleop=None, fed=None, drove=None) -> np.ndarray:
        scripted = np.zeros(len(state), dtype=bool) if scripted is None else scripted
        return skills.label_run(skills_, state, NAMES, FPS, GRASP, scripted, told, teleop=teleop, fed=fed, drove=drove)

    def held_after_close(self) -> tuple[np.ndarray, int]:
        """A reach that closes on the block at 18 and then holds it still for 6 s, with no elbow travel, so
        the carry comes from holding steady: the trace, and the frame the jaws reach 18."""
        rows = [np.array([0.0, -100.0, 90.0, 70.0, 0.0, 1.0])]

        def to(seconds: float, **target) -> None:
            start, end = rows[-1], rows[-1].copy()
            for name, value in target.items():
                end[NAMES.index(f"{name}.pos")] = value
            n = round(seconds * FPS)
            rows.extend(start + (end - start) * (i / n) for i in range(1, n + 1))

        to(0.5)
        to(2.0, shoulder_pan=20, gripper=40)
        to(0.5, gripper=18)
        closed = len(rows) - 1
        to(6.0)
        return np.array(rows, dtype=np.float32), closed

    def test_a_pause_right_after_the_close_is_no_carry_and_the_hold_counts_again_after_it(self) -> None:
        state, closed = self.held_after_close()
        hold = round(GRASP.gripper.hold_s * FPS)
        unpaused = int(np.flatnonzero(self.label(state) == 1)[0])
        pause = slice(closed + 1, closed + 1 + 2 * FPS)  # 2 s Paused with the jaws on the block
        self.assertTrue(pause.start < unpaused < pause.stop, "without the pause, the carry falls inside it")
        fed = np.ones(len(state), dtype=bool)
        fed[pause] = False
        labels = self.label(state, fed=fed)
        np.testing.assert_array_equal(labels[: pause.stop], 0, "no carry while paused: the detector saw none of it")
        carry = int(np.flatnonzero(labels == 1)[0])
        self.assertLessEqual(abs(carry - (pause.stop + hold)), 1, "the carry comes hold_s of fed frames after the pause")
        # As at runtime: the perception sees no paused tick and is handed back with moved=False.
        h = Stack()
        h.feed(state[: pause.start])
        h.top.pause()
        h.top.run(Handback(PERSON))
        h.feed(state[pause.stop:])
        ran = pause.stop + [o.skill for o in h.outputs[pause.start:]].index(1)
        self.assertLessEqual(abs(carry - ran), 1, f"the carry at frame {carry}, at runtime at {ran}")

    def test_a_pause_and_resume_in_one_tick_restarts_the_hold_as_it_did_live(self) -> None:
        # Every frame stays Autonomous, but the perception heard the handback (resumed with nothing moved),
        # which restarts the steady hold the carry waits for: `analysis.Attempt.handbacks` marks the frame.
        state, closed = self.held_after_close()
        hold = round(GRASP.gripper.hold_s * FPS)
        unmarked = int(np.flatnonzero(self.label(state) == 1)[0])
        at = closed + 10
        self.assertLess(at, unmarked, "the handback comes during the hold")
        marks = np.zeros(len(state), dtype=bool)
        marks[at] = True
        labels = skills.label_run(SKILLS, state, NAMES, FPS, GRASP, np.zeros(len(state), dtype=bool), handbacks=marks)
        carry = int(np.flatnonzero(labels == 1)[0])
        self.assertGreater(carry, unmarked, "the hold started again at the handback")
        self.assertLessEqual(abs(carry - (at + hold)), 1, "the carry comes hold_s after the handback")

    def test_waiting_frames_count_on_the_detector_s_clock(self) -> None:
        state, closed = self.held_after_close()
        unpaused = int(np.flatnonzero(self.label(state) == 1)[0])
        waiting = slice(closed + 1, closed + 1 + 2 * FPS)  # the model sent nothing for 2 s; the frames are fed
        drove = np.ones(len(state), dtype=bool)
        drove[waiting] = False
        labels = self.label(state, fed=np.ones(len(state), dtype=bool), drove=drove)
        self.assertEqual(int(np.flatnonzero(labels == 1)[0]), unpaused, "the carry comes as if the model drove")

    def test_a_pause_then_a_rewind_with_the_jaws_open_starts_again_on_pick(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        # After the carry the person pauses, then a Rewind plays the arm back to before the close: moved wins.
        held = state[: at["carry"] + 15]
        paused = np.repeat(held[-1][None], FPS, axis=0)
        back = state[at["close"] - 20: at["carry"] + 15][::-1][:40]
        again = state[at["close"] - 20:]
        joined = np.concatenate([held, paused, back, again])
        scripted = np.zeros(len(joined), dtype=bool)
        scripted[len(held) + len(paused): len(held) + len(paused) + len(back)] = True
        fed = np.ones(len(joined), dtype=bool)
        fed[len(held): len(held) + len(paused)] = False
        labels = self.label(joined, THREE, scripted=scripted, fed=fed)
        restart = len(held) + len(paused) + len(back)
        np.testing.assert_array_equal(labels[len(held): restart], 1, "paused and rewinding frames keep the skill")
        self.assertEqual(labels[restart], 0, "the jaws read open after the rewind: back on pick")
        self.assertEqual(labels[-1], 2)

    def test_a_regrip_in_teleop_during_the_carry_stays_in_place(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        # The model carries the block; a person takes Teleop, opens the jaws and closes them on it again,
        # and hands back; the model carries it on and lets go over the jar.
        split, n = at["carry"] + 15, 30
        self.assertLess(state[split, 5], GRASP.gripper.closed_below, "the block is held when the person takes over")
        regrip = np.repeat(state[split][None], n, axis=0)
        regrip[:, 5] = np.concatenate([np.linspace(state[split, 5], 40, n // 2), np.linspace(40, state[split, 5], n // 2)])
        joined = np.concatenate([state[:split], regrip, state[split:]])
        teleop = np.zeros(len(joined), dtype=bool)
        teleop[split: split + n] = True
        labels = self.label(joined, THREE, teleop=teleop)
        release = at["release"] + n
        np.testing.assert_array_equal(labels[: at["carry"]], 0)
        np.testing.assert_array_equal(labels[at["carry"]: release], 1, "the re-open isn't a release: still place")
        np.testing.assert_array_equal(labels[release:], 2, "the model's own release switches to the return")

    def test_a_grasp_in_teleop_during_pick_is_place_from_its_carry(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        # The model reaches; a person takes Teleop before the close, grasps and carries, and hands back
        # with the block held; the model lets go over the jar.
        takeover, back = at["close"] - 10, at["carry"] + 20
        teleop = np.zeros(len(state), dtype=bool)
        teleop[takeover:back] = True
        labels = self.label(state, THREE, teleop=teleop)
        np.testing.assert_array_equal(labels[: at["carry"]], 0)
        np.testing.assert_array_equal(labels[at["carry"]: at["release"]], 1, "a person's grasp counts")
        np.testing.assert_array_equal(labels[at["release"]:], 2)

    def test_a_person_who_finishes_in_teleop_is_labelled_return_from_their_release(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        # The model reaches; a person takes Teleop before the close, grasps, carries, lets go over the
        # jar and brings the arm back, and the attempt ends in Teleop.
        teleop = np.zeros(len(state), dtype=bool)
        teleop[at["close"] - 10:] = True
        labels = self.label(state, THREE, teleop=teleop)
        np.testing.assert_array_equal(labels[: at["carry"]], 0)
        np.testing.assert_array_equal(labels[at["carry"]: at["release"]], 1)
        np.testing.assert_array_equal(labels[at["release"]:], 2, "the person's release starts the return")
        two = self.label(state, teleop=teleop)
        bound = at["widest"] + round(0.2 * FPS)
        np.testing.assert_array_equal(two[bound + 1:], -1, "without a return, past place's bound is in none")

    def test_a_person_alone_is_labelled_by_the_events_of_what_they_did(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        labels = self.label(state, THREE)
        np.testing.assert_array_equal(labels[: at["carry"]], 0)
        np.testing.assert_array_equal(labels[at["carry"]: at["release"]], 1)
        np.testing.assert_array_equal(labels[at["release"]:], 2)

    def test_without_a_return_skill_the_way_back_after_place_is_in_none(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        labels = self.label(state)
        bound = at["widest"] + round(0.2 * FPS)
        np.testing.assert_array_equal(labels[at["carry"]: bound + 1], 1)
        np.testing.assert_array_equal(labels[bound + 1:], -1)

    def test_a_correction_that_starts_in_pick_is_place_from_the_carry_whatever_the_model_was_told(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        # The model drove the reach under pick; a person took over before the close and carried the block.
        takeover = at["close"] - 10
        told = np.full(len(state), -1, dtype=np.int8)
        told[:takeover] = 0
        labels = self.label(state, THREE, told=told)
        np.testing.assert_array_equal(labels[: at["carry"]], 0)
        np.testing.assert_array_equal(labels[at["carry"]: at["release"]], 1)
        np.testing.assert_array_equal(labels[at["release"]:], 2)

    def test_the_frames_the_model_drove_keep_the_skill_it_was_told(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        told = np.zeros(len(state), dtype=np.int8)  # it was never switched: told pick throughout
        labels = self.label(state, THREE, told=told)
        bound = at["carry"] + round(0.5 * FPS)
        np.testing.assert_array_equal(labels[: bound + 1], 0)
        np.testing.assert_array_equal(labels[bound + 1:], -1, "pick past its bound is in no skill, as in a demo")

    def test_a_rewind_that_opens_the_jaws_starts_again_on_pick(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        # After the carry a Rewind plays the arm back to before the close, then a person grasps again.
        held = state[: at["carry"] + 15]
        back = state[at["close"] - 20: at["carry"] + 15][::-1][:40]
        again = state[at["close"] - 20:]
        joined = np.concatenate([held, back, again])
        scripted = np.zeros(len(joined), dtype=bool)
        scripted[len(held): len(held) + len(back)] = True
        labels = self.label(joined, THREE, scripted=scripted)
        restart = len(held) + len(back)
        self.assertEqual(labels[restart], 0, "the jaws read open after the rewind: back on pick")
        second = restart + int(np.flatnonzero(labels[restart:] == 1)[0])
        np.testing.assert_array_equal(labels[restart: second], 0)
        self.assertLessEqual(abs(second - (restart + at["carry"] - (at["close"] - 20))), 1,
                             "the second carry switches to place again")
        self.assertEqual(labels[-1], 2)

    def dropped_and_rewound(self) -> tuple[np.ndarray, np.ndarray, int]:
        """The model lets go outside the jar, a Rewind plays the arm back to before the close with the jaws
        open, and the model reaches, grasps and lets go again: the trace, its scripted frames, and the
        first frame after the Rewind."""
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        dropped = state[: at["release"] + 10]
        back = state[at["close"] - 20: at["release"] + 10][::-1]
        again = state[at["close"] - 20:]
        joined = np.concatenate([dropped, back, again])
        scripted = np.zeros(len(joined), dtype=bool)
        scripted[len(dropped): len(dropped) + len(back)] = True
        return joined, scripted, len(dropped) + len(back)

    def test_a_rewind_that_undoes_the_release_starts_again_on_pick(self) -> None:
        joined, scripted, restart = self.dropped_and_rewound()
        labels = self.label(joined, THREE, scripted=scripted)
        self.assertEqual(labels[restart - 1], 2, "the return, until the model drives again")
        self.assertEqual(labels[restart], 0, "the jaws read open after the rewind, and nothing is held: back on pick")
        second = restart + int(np.flatnonzero(labels[restart:] == 1)[0])
        np.testing.assert_array_equal(labels[restart: second], 0)
        self.assertEqual(labels[-1], 2, "the second release switches to the return again")

    def runtime(self, state: np.ndarray, scripted: np.ndarray, skills_=THREE) -> np.ndarray:
        """Each frame's skill as serve records it (control.skill), with the stack paused through the
        scripted frames and handed back with `moved` after them."""
        h = Stack(skills_)
        skill, paused = [], False
        for row, moved in zip(state, scripted, strict=True):
            if moved:
                if not paused:
                    h.top.pause()
                    paused = True
                skill.append(skill[-1])
                continue
            if paused:
                h.top.run(Handback(PERSON, moved=True))
                paused = False
            h.feed(row[None])
            skill.append(h.outputs[-1].skill)
        return np.array(skill, dtype=np.int8)

    def test_labels_switch_where_the_runtime_does(self) -> None:
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)

        def rewound(stop: int, back_to: int) -> tuple[np.ndarray, np.ndarray]:
            """The trace to `stop`, a Rewind back to `back_to`, and the trace again from there."""
            back = state[back_to:stop][::-1]
            joined = np.concatenate([state[:stop], back, state[back_to:]])
            scripted = np.zeros(len(joined), dtype=bool)
            scripted[stop: stop + len(back)] = True
            return joined, scripted

        cases = {
            "the drop and the rewind to before the close": self.dropped_and_rewound()[:2],
            "a rewind that keeps the block held": rewound(at["carry"] + 15, at["carry"]),
            "a rewind in pick from after the close to before it": rewound(at["close"] + 20, at["close"] - 20),
            # The replay's hold timer mustn't run on across the frames the Rewind skipped.
            "a rewind in pick that keeps the jaws closed": rewound(at["close"] + 40, at["close"] + 5),
        }

        def switches(skill: np.ndarray) -> list[tuple[int, int]]:
            return [(i, int(skill[i])) for i in range(len(skill)) if i == 0 or skill[i] != skill[i - 1]]

        for case, (joined, scripted) in cases.items():
            with self.subTest(case):
                labelled, ran = switches(self.label(joined, THREE, scripted=scripted)), switches(self.runtime(joined, scripted))
                self.assertEqual([s for _, s in labelled], [s for _, s in ran], "the same skills in the same order")
                # The perception sums its clock tick by tick, so float rounding can move a switch on a
                # threshold of time, such as the carry's hold, by one frame.
                for (a, _), (b, _) in zip(labelled, ran, strict=True):
                    self.assertLessEqual(abs(a - b), 1, f"a switch at frame {a}, at runtime at {b}")


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

    def test_when_no_demo_can_be_cut_the_build_says_so_rather_than_to_record_demos(self) -> None:
        missed = dataclasses.replace(TASK, name="missed")
        make_session(data.task_dir(self.s, missed) / "sessions" / "20260101-100000" / "dataset", [trace(grasp=False)])
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(settings.ConfigError) as error:
            build(self.s, missed, kind=data.SKILLS)
        self.assertIn("none of its 1 kept demos could be cut into skills", str(error.exception))
        self.assertNotIn("record some demos", str(error.exception))
        self.assertIn("no carry", out.getvalue())

    def record_run(self, name: str, with_skills: bool) -> Path:
        """A serve run on the fake arm that closes on the block, holds it past the carry and succeeds:
        through the perception alone (a whole-model run), or with the skill sequencer too."""
        from test_controller import START_POSE, Rig

        opened = [{**START_POSE, "gripper.pos": min(10.0 + 2 * i, 40.0)} for i in range(1, 31)]
        closed = [{**opened[-1], "gripper.pos": max(40.0 - 3 * i, 18.4)} for i in range(1, 11)]
        actions = [[a[k] for k in START_POSE] for a in opened + closed]
        run_dir = data.task_dir(self.s, TASK) / "runs" / name
        rig = Rig(run_dir, actions=actions, wrap=lambda driver, recorder, clock: stack.build(
            driver, LayerContext(recorder.event, clock, GRASP, PROGRAM, FPS), skills=SKILLS if with_skills else ()))
        rig.start_policy(ticks=5 * FPS)
        rig.reset("succeeded")
        rig.close()
        policy = {"skills": [skill.record() for skill in SKILLS]} if with_skills else {}
        (run_dir / "run.json").write_text(json.dumps({"policy": policy}))
        return run_dir

    def test_a_runs_teleop_frames_reach_the_replay_as_a_persons(self) -> None:
        from test_controller import Rig

        from so_101 import build as builder
        from so_101.control import analysis
        from so_101.control.recorder import Phase

        name = "20260101-130000"
        run_dir = data.task_dir(self.s, TASK) / "runs" / name
        rig = Rig(run_dir, wrap=lambda driver, recorder, clock: stack.build(
            driver, LayerContext(recorder.event, clock, GRASP, PROGRAM, FPS)))
        rig.start_policy(ticks=FPS)
        rig.teleop(1.0)
        rig.reset("succeeded")
        rig.close()
        (run_dir / "run.json").write_text(json.dumps({"policy": {}}))
        run = next(r for r in data.runs(self.s, TASK) if r.name == name)
        attempt = analysis.analyse(run, self.s.control.blame_s).attempts[0]
        with mock.patch.object(skills, "label_run", wraps=skills.label_run) as label:
            per_frame, why = builder.run_skills(run, attempt, TASK, GRASP)
        self.assertEqual(why, "")
        teleop = label.call_args.kwargs["teleop"]
        np.testing.assert_array_equal(teleop, attempt.phases == Phase.TELEOP)
        self.assertIs(label.call_args.kwargs["handbacks"], attempt.handbacks, "same-tick handbacks reach the replay")
        self.assertGreaterEqual(teleop.sum(), FPS - 1, "the second of Teleop")

    def test_a_runs_paused_frames_are_not_replayed_and_only_the_models_drive(self) -> None:
        from test_controller import Rig

        from so_101 import build as builder
        from so_101.control import analysis
        from so_101.control.recorder import Phase
        from so_101.policy.base import WAITING

        name = "20260101-140000"
        run_dir = data.task_dir(self.s, TASK) / "runs" / name
        rig = Rig(run_dir, wrap=lambda driver, recorder, clock: stack.build(
            driver, LayerContext(recorder.event, clock, GRASP, PROGRAM, FPS)))
        rig.start_policy(ticks=FPS)
        rig.command("pause")
        rig.tick(FPS)
        rig.command("resume")
        rig.tick_until("autonomous")
        rig.tick(FPS)
        rig.teleop(0.5)
        rig.reset("succeeded")
        rig.close()
        (run_dir / "run.json").write_text(json.dumps({"policy": {}}))
        run = next(r for r in data.runs(self.s, TASK) if r.name == name)
        attempt = analysis.analyse(run, self.s.control.blame_s).attempts[0]
        with mock.patch.object(skills, "label_run", wraps=skills.label_run) as label:
            per_frame, why = builder.run_skills(run, attempt, TASK, GRASP)
        self.assertEqual(why, "")
        driver = np.array([int(np.ravel(v)[0]) for v in table(run.dataset_root)["control.driver"]])
        autonomous, teleop = attempt.phases == Phase.AUTONOMOUS, attempt.phases == Phase.TELEOP
        fed, drove = label.call_args.kwargs["fed"], label.call_args.kwargs["drove"]
        np.testing.assert_array_equal(fed, teleop | (autonomous & np.isin(driver, (POLICY, WAITING))))
        np.testing.assert_array_equal(drove, teleop | (driver == POLICY))
        self.assertGreaterEqual((attempt.phases == Phase.PAUSED).sum(), FPS - 1, "the second Paused")
        self.assertFalse(fed[attempt.phases == Phase.PAUSED].any(), "no paused frame is replayed")
        self.assertTrue(fed[teleop].all() and teleop.sum() >= FPS // 2 - 1, "Teleop frames are")

    def test_the_replay_labels_a_run_as_the_live_sequencer_did_after_a_layers_playback(self) -> None:
        """A layer plays the model's motion back and hands the arm back itself; the model then closes on the
        block, carries it and lets it go, and a person pauses and resumes. The live sequencer stays on
        return across that plain pause, and the replay of the recorded run says the same on every frame
        the model drove."""
        from test_controller import START_POSE, PlayingBack, Rig

        from so_101 import build as builder
        from so_101.control import analysis
        from so_101.control.recorder import Phase

        three = dataclasses.replace(TASK, skills=THREE)
        opened = [{**START_POSE, "gripper.pos": min(10.0 + 2 * i, 40.0)} for i in range(1, 31)]
        closed = [{**opened[-1], "gripper.pos": max(40.0 - 3 * i, 18.4)} for i in range(1, 11)]
        held = [closed[-1]] * (2 * FPS)
        let_go = [{**closed[-1], "gripper.pos": min(18.4 + 3 * i, 40.0)} for i in range(1, 11)]
        actions = [[a[k] for k in START_POSE] for a in opened + closed + held + let_go]
        name = "20260101-150000"
        run_dir = data.task_dir(self.s, three) / "runs" / name
        rig = Rig(run_dir, actions=actions, wrap=lambda driver, recorder, clock: PlayingBack(
            stack.build(driver, LayerContext(recorder.event, clock, GRASP, PROGRAM, FPS), skills=THREE),
            drive=FPS // 2, back_s=0.3, clock=clock, hand_back=True))
        rig.start_policy(ticks=6 * FPS)
        rig.command("pause")
        rig.tick(FPS)
        rig.command("resume")
        rig.tick_until("autonomous")
        rig.tick(FPS)
        rig.reset("succeeded")
        rig.close()
        (run_dir / "run.json").write_text(json.dumps({"policy": {"skills": [s.record() for s in THREE]}}))
        run = next(r for r in data.runs(self.s, three) if r.name == name)
        attempt = analysis.analyse(run, self.s.control.blame_s).attempts[0]
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e.get("type") == "grasp" and e["grasp"] == "release" for e in events), "the model let go")
        self.assertIn(Phase.PAUSED, set(attempt.phases), "a person paused after it")
        live = attempt.skills
        resumed = int(np.flatnonzero(attempt.phases == Phase.PAUSED)[-1]) + 1
        driven_after = np.flatnonzero((attempt.phases == Phase.AUTONOMOUS) & (np.arange(len(live)) >= resumed))
        self.assertTrue(len(driven_after) and (live[driven_after] == 2).all(),
                        "the live sequencer stays on return across the plain pause")
        with mock.patch.object(skills, "label_run", wraps=skills.label_run) as label:
            builder.run_skills(run, attempt, three, GRASP)
        args, kwargs = label.call_args
        replayed = skills.label_run(*args[:6], None, **{k: v for k, v in kwargs.items() if k != "told"})
        driven = (attempt.phases == Phase.AUTONOMOUS) & kwargs["drove"] & (replayed >= 0)
        # The live perception's clock counts each tick fed as 1/fps, as the replay counts each frame (it runs the
        # replay's own LiveFeed), so the two agree exactly: the same skills, in the same order, switching on the
        # same frames, and every driven frame the same. This was within a frame while the perception summed real
        # tick gaps (a fake clock at 1000.0 plus 1/30 a tick, which rounds), and is exact again now.
        def switches(labels: np.ndarray) -> list[tuple[int, int]]:
            frames = np.flatnonzero(driven)
            return [(int(f), int(labels[f])) for k, f in enumerate(frames) if k and labels[f] != labels[frames[k - 1]]]

        self.assertTrue(switches(live), "the attempt switches skills, so the switches are compared")
        self.assertEqual(switches(replayed), switches(live), "the same skills, in the same order, on the same frames")
        np.testing.assert_array_equal(replayed[driven], live[driven])
        self.assertTrue((replayed[driven_after] == 2).all(), "the replay stays on return across the plain pause too")

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

    def __init__(self, skills_: tuple = SKILLS) -> None:
        self.events: list[dict] = []
        self.now, self.frame = 0.0, -1
        self.leaf = FakeLeaf()
        ctx = LayerContext(lambda type_, **fields: self.events.append({"type": type_, **fields}), lambda: self.now, GRASP, PROGRAM, FPS)
        self.top = stack.build(self.leaf, ctx, skills=skills_)
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

    def test_the_release_switches_to_the_return_and_a_handback_after_a_move_with_the_jaws_open_goes_back_to_pick(self) -> None:
        h = Stack(THREE)
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["release"] + 5])
        self.assertEqual(h.switches(), [("pick", "start"), ("place", "carry"), ("return", "release")])
        self.assertEqual(h.leaf.conditions[-1].prompt, "Return to the rest pose")
        self.assertEqual(h.leaf.conditions[-1].values(), {"skill.pick": 0.0, "skill.place": 0.0, "skill.return": 1.0})
        h.top.pause()
        h.top.run(Handback(PERSON))  # a pause that moved nothing: the release stands
        h.feed(state[at["release"] + 5: at["release"] + 10])
        self.assertEqual(h.switches()[-1], ("return", "release"))
        self.assertEqual(h.outputs[-1].skill, 2)
        h.top.pause()
        # A person moved the arm with the jaws open: the perception can't tell a release from a block put
        # back on the mat, so the attempt starts again on pick, as the user decided.
        h.top.run(Handback(PERSON, moved=True))
        h.feed(state[at["release"] + 10:])
        self.assertEqual(h.switches()[-1], ("pick", "reopened"))
        self.assertEqual(h.leaf.conditions[-1].prompt, SKILLS[0].prompt)
        self.assertEqual(h.outputs[-1].skill, 0)

    def test_a_handback_above_the_hold_band_after_a_release_goes_back_to_pick(self) -> None:
        # Jaws at 23.8 after a moved handback: under closed_below, but above the hold band, so nothing is held.
        h = Stack(THREE)
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["release"] + 5])
        self.assertEqual(h.switches()[-1], ("return", "release"))
        h.top.pause()
        h.top.run(Handback(PERSON, moved=True))
        ajar = state[at["release"] + 5: at["release"] + 10].copy()
        ajar[:, NAMES.index("gripper.pos")] = 23.8
        h.feed(ajar)
        self.assertEqual(h.switches()[-1], ("pick", "reopened"))
        self.assertEqual(h.outputs[-1].skill, 0)

    def test_a_drop_outside_the_jar_rewound_to_before_the_close_goes_back_to_pick(self) -> None:
        h = Stack(THREE)
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["release"] + 10])  # let go outside the jar: on the return
        self.assertEqual(h.outputs[-1].skill, 2)
        h.top.pause()  # a Rewind plays the arm back to before the close, the jaws open
        h.top.run(Handback(PERSON, moved=True))
        h.feed(state[at["close"] - 20: at["close"] - 18])
        self.assertEqual(h.switches()[-1], ("pick", "reopened"))
        self.assertEqual(h.leaf.conditions[-1].prompt, SKILLS[0].prompt)
        self.assertEqual(h.leaf.conditions[-1].values(), {"skill.pick": 1.0, "skill.place": 0.0, "skill.return": 0.0})
        self.assertEqual([o.skill for o in h.outputs[-2:]], [0, 0], "from the first tick the model drives again")
        h.feed(state[at["close"] - 18:])  # the model grasps again and lets go
        self.assertEqual([s for s, _ in h.switches()], ["pick", "place", "return", "pick", "place", "return"])

    def test_a_handback_with_the_jaws_closed_after_the_release_keeps_the_return(self) -> None:
        h = Stack(THREE)
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["release"] + 10])
        h.top.pause()
        h.top.run(Handback(PERSON, moved=True))  # a person closed the empty jaws: the release stands
        h.feed(state[-5:])
        self.assertEqual(h.switches()[-1], ("return", "release"))

    def test_an_error_conditioning_the_model_is_the_models_and_reaches_the_loop(self) -> None:
        h = Stack()
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["carry"]])

        def refuse(conditioning: Conditioning) -> None:
            raise RuntimeError("the engine refused the task")

        h.leaf.condition = refuse
        with self.assertRaisesRegex(RuntimeError, "refused the task"):
            h.feed(state[at["carry"]: at["carry"] + 5])
        self.assertIsNone(h.top.broken, "the sequencer's own logic didn't fail")
        self.assertEqual([e for e in h.events if e["type"] == "skills"], [])

    def test_after_the_jaws_reopen_a_second_grasp_reaches_place_again(self) -> None:
        h = Stack()
        state = trace()
        at = skills.events(state, NAMES, FPS, GRASP)
        h.feed(state[: at["carry"] + 10])
        h.top.pause()
        h.top.run(Handback(LAYER, moved=True))  # a playback undid the grasp: the jaws are open again
        h.feed(state[at["close"] - 30:])  # the reach again, then a second close, lift and carry
        self.assertEqual(h.switches(), [("pick", "start"), ("place", "carry"), ("pick", "reopened"), ("place", "carry")])
        self.assertEqual(h.leaf.conditions[-1].prompt, SKILLS[1].prompt)

    def test_a_playback_that_undoes_the_grasp_sends_the_skill_back_to_pick(self) -> None:
        self.playback_undoes_the_grasp()

    def playback_undoes_the_grasp(self):
        """The gap the uniform design closes: the carry switches to place, then the after-close check, asked
        after the carry, finds the jaws empty, and the supervisor plays back to before the close. Returns
        the stack and what the next ticks need."""
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
        ctx = LayerContext(lambda type_, **fields: events.append({"type": type_, **fields}), lambda: now[0], GRASP, program_, FPS)
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
        return top, events, leaf, robot, state, at, frame, now, n

    def test_after_a_playback_a_second_grasp_reaches_place_again(self) -> None:
        top, events, leaf, robot, state, at, frame, now, n = self.playback_undoes_the_grasp()
        for row in state[at["close"] - 30:]:  # the model reaches again and grasps; no verdict comes back
            now[0] += 1 / FPS
            n += 1
            pose = dict(zip(NAMES, map(float, row), strict=True))
            top.step(robot, Tick(now[0], n, {**pose, "top": frame, "wrist": frame}, pose))
        self.assertEqual([(e["skill"], e["cause"]) for e in events if e["type"] == "skill"],
                         [("pick", "start"), ("place", "carry"), ("pick", "reopened"), ("place", "carry")])
        self.assertEqual(leaf.conditions[-1].prompt, SKILLS[1].prompt)


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
                      wrap=lambda driver, recorder, clock: stack.build(driver, LayerContext(recorder.event, clock, GRASP, PROGRAM, FPS),
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


class RunSkillsTest(unittest.TestCase):
    """so101-control checks again, after it reloads tasks.toml and the grasp thresholds, that the skills and
    `[gripper]` thresholds are the ones `serve` or `run` checked the model against and wrote in run.json
    (`control.main.require_run_skills`): either file may change while the policy downloads."""

    def policy(self, **changes) -> dict:
        return {"skills": json.loads(json.dumps([skill.record() for skill in SKILLS])),
                "gripper_sha256": skills.gripper_hash(GRASP.gripper), **changes}

    def refused(self, policy: dict, task: Task = TASK, gripper=GRASP.gripper) -> str:
        from so_101.control.main import require_run_skills

        with self.assertRaises(SystemExit) as stop:
            require_run_skills(policy, task, gripper)
        return str(stop.exception.code)

    def test_the_skills_and_thresholds_checked_run(self) -> None:
        from so_101.control.main import require_run_skills

        require_run_skills(self.policy(), TASK, GRASP.gripper)
        require_run_skills({"skills": None}, dataclasses.replace(TASK, skills=()), GRASP.gripper)  # a whole-demo model

    def test_skills_changed_in_tasks_toml_since_the_check_are_refused(self) -> None:
        reworded = dataclasses.replace(TASK, skills=tuple(dataclasses.replace(k, prompt="Grab it") if k.name == "pick"
                                                          else k for k in SKILLS))
        message = self.refused(self.policy(), task=reworded)
        self.assertIn("[test.skills]", message)
        self.assertIn("pick", message)
        self.assertIn("start the run again", message)
        message = self.refused(self.policy(), task=dataclasses.replace(TASK, skills=()))
        self.assertIn("[test.skills]", message)

    def test_gripper_thresholds_changed_since_the_check_are_refused(self) -> None:
        message = self.refused(self.policy(), gripper=dataclasses.replace(GRASP.gripper, hold_s=1.6))
        self.assertIn("[gripper]", message)
        self.assertNotIn("[test.skills]", message)
        self.assertIn("[gripper]", self.refused(self.policy(gripper_sha256=None)), "a run.json that recorded none")

    def test_both_changes_are_named_in_one_refusal(self) -> None:
        message = self.refused(self.policy(), task=dataclasses.replace(TASK, skills=SKILLS[::-1]),
                               gripper=dataclasses.replace(GRASP.gripper, hold_s=1.6))
        self.assertIn("[test.skills]", message)
        self.assertIn("[gripper]", message)


class ModelTest(unittest.TestCase):
    """A skills model runs only with the skills and gripper thresholds its dataset was cut with
    (`commands.policy.check_skills_model`), found through its train_config.json and the dataset's build.json."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="so101-model-")
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.build = {"inputs": {"dataset": data.SKILLS, "sources": ["demos"],
                                 "skills": [skill.record() for skill in SKILLS],
                                 "gripper_sha256": skills.gripper_hash(GRASP.gripper),
                                 "cut_rules": skills.CUT_RULES_VERSION}}
        # outputs/ beside a so101.toml in the temporary folder, where `train` keeps the build a local model
        # trained on (`data.trained_build_path`): never the repo's own outputs/.
        self.s = dataclasses.replace(settings.load(), path=self.folder / "so101.toml")

    def check(self, build: dict | None = None, task: Task = TASK, one_hot: int | None = 2, with_skills: bool = True,
              policy: str = "act", pinned: bool = True, rebuilt: bool = False, kept: str = "kept") -> None:
        """Check a local checkpoint trained on `build`, whose train_config.json records that build, and whose
        build.json is kept, as `train --target local` does (`pinned`), and whose dataset was then built again
        (`rebuilt`). `kept` is what became of the kept copy: "kept", "changed" or "missing"."""
        from so_101.commands.policy import check_skills_model

        root = self.folder / "dataset"
        root.mkdir(exist_ok=True)
        (root / "build.json").write_text(json.dumps({"built": "first", **(self.build if build is None else build)}))
        model = self.folder / "model"
        model.mkdir(exist_ok=True)
        dataset = {"repo_id": "local/so101_test_skills", "root": str(root), "revision": None}
        if pinned:
            dataset["revision"] = data.snapshot_build(self.s, root / "build.json")
            copy = data.trained_build_path(self.s, dataset["revision"])
            if kept == "changed":
                copy.write_text(copy.read_text() + " ")
            elif kept == "missing":
                copy.unlink()
        (model / "train_config.json").write_text(json.dumps({"dataset": dataset}))
        if rebuilt:  # the same skills and thresholds, built again after training
            (root / "build.json").write_text(json.dumps({"built": "second", **(self.build if build is None else build)}))
        inputs = {} if one_hot is None else {skills.STATE_KEY: {"type": "ENV", "shape": [one_hot]}}
        ns = types.SimpleNamespace(policy=policy, revision=None)
        check_skills_model(self.s, ns, task, {"input_features": inputs}, model, with_skills)

    def refused(self, message: str, **kwargs) -> None:
        with self.assertRaises(SystemExit) as stop:
            self.check(**kwargs)
        self.assertIn(message, str(stop.exception.code))

    def test_a_model_trained_on_the_tasks_skills_runs(self) -> None:
        self.check()
        self.check(one_hot=None, policy="pi05")  # pi0.5 may read the prompt alone

    def test_a_model_is_refused_when_its_skills_differ_from_the_tasks(self) -> None:
        inputs = self.build["inputs"]
        renamed = [{**r, "name": "grab"} if r["name"] == "pick" else r for r in inputs["skills"]]
        reworded = [{**r, "prompt": "Grab it"} if r["name"] == "pick" else r for r in inputs["skills"]]
        for case, changed in {"a name": renamed, "the order": inputs["skills"][::-1]}.items():
            with self.subTest(case):
                self.refused("doesn't match this task's skills: its skills were",
                             build={"inputs": {**inputs, "skills": changed}})
        self.refused("doesn't match this task's skills: pick: prompt 'Pick up the red block' here, 'Grab it' in the "
                     "model's build", build={"inputs": {**inputs, "skills": reworded}})

    def test_a_model_is_refused_when_its_skills_switch_or_end_elsewhere(self) -> None:
        inputs = self.build["inputs"]

        def place(**change) -> list[dict]:
            return [{**r, **change} if r["name"] == "place" else r for r in inputs["skills"]]

        cases = {
            "the handover": (place(handover="release"), "place: handover carry here, release in the model's build"),
            "from": (place(**{"from": "carry"}), "place: from lift here, carry in the model's build"),
            "to": (place(to="widest + 0.3"), "place: to widest + 0.2 here, widest + 0.3 in the model's build"),
        }
        for case, (changed, message) in cases.items():
            with self.subTest(case):
                self.refused(message, build={"inputs": {**inputs, "skills": changed}})
        # Every difference is named in the one refusal.
        both = [{**r, "handover": "release", "to": "widest + 0.3"} if r["name"] == "place" else r
                for r in inputs["skills"]]
        with self.assertRaises(SystemExit) as stop:
            self.check(build={"inputs": {**inputs, "skills": both}})
        self.assertIn("place: to widest + 0.2 here, widest + 0.3 in the model's build; "
                      "place: handover carry here, release in the model's build", str(stop.exception.code))

    def test_a_model_is_refused_when_its_demos_were_cut_under_other_rules(self) -> None:
        inputs = self.build["inputs"]
        self.refused(f"its demos were cut under cut rules {skills.CUT_RULES_VERSION - 1}, and this checkout cuts under "
                     f"{skills.CUT_RULES_VERSION}",
                     build={"inputs": {**inputs, "cut_rules": skills.CUT_RULES_VERSION - 1}})
        self.refused("its build recorded no cut_rules",
                     build={"inputs": {k: v for k, v in inputs.items() if k != "cut_rules"}})

    def test_a_model_trained_with_runs_is_refused_when_they_were_labelled_under_other_rules(self) -> None:
        inputs = {**self.build["inputs"], "sources": ["demos", "policy-successes"]}
        self.check(build={"inputs": {**inputs, "run_labels": skills.RUN_LABEL_VERSION}})
        self.refused(f"its runs were labelled under run labels {skills.RUN_LABEL_VERSION + 1}, and this checkout labels "
                     f"them under {skills.RUN_LABEL_VERSION}",
                     build={"inputs": {**inputs, "run_labels": skills.RUN_LABEL_VERSION + 1}})
        self.refused("its build took runs and recorded no run_labels", build={"inputs": inputs})

    def test_a_model_is_accepted_when_its_build_matches_this_checkout(self) -> None:
        # A demos-only build records no run_labels, and needs none.
        self.assertNotIn("run_labels", self.build["inputs"])
        self.check()
        temp = tempfile.TemporaryDirectory(prefix="so101-model-data-")
        self.addCleanup(temp.cleanup)
        s = dataclasses.replace(self.s, data_dir=Path(temp.name))
        for runs in ((), ("policy-successes",)):
            with self.subTest(runs=runs):
                self.check(build={"inputs": data.build_inputs(s, TASK, runs, kind=data.SKILLS)})

    def test_a_model_is_refused_when_its_demos_were_cut_with_other_gripper_thresholds(self) -> None:
        inputs = self.build["inputs"]
        other = skills.gripper_hash(dataclasses.replace(GRASP.gripper, hold_s=1.6))
        self.refused("other [gripper] thresholds", build={"inputs": {**inputs, "gripper_sha256": other}})
        self.refused("recorded no [gripper] thresholds",
                     build={"inputs": {k: v for k, v in inputs.items() if k != "gripper_sha256"}})

    def test_the_one_hot_must_fit_the_tasks_skills_and_needs_skills(self) -> None:
        self.refused("reads a one-hot of 2 skills, and [test.skills] in tasks.toml lists 3",
                     task=dataclasses.replace(TASK, skills=THREE))
        self.refused("it's a skills model. Run it with --skills", with_skills=False)
        self.refused("can't be told which skill it's on", one_hot=None)
        self.check(one_hot=None, with_skills=False)  # a whole-demo model, run as one

    def test_a_local_checkpoint_is_checked_against_the_build_it_was_trained_on(self) -> None:
        self.check()
        # Rebuilt since: `train` kept a copy of the build.json it trained on, so the model still runs. This was a
        # refusal ("was rebuilt after this model was trained") until the user chose to keep that copy, since any
        # rebuild, the one `train` runs when the build is stale included, retired a model that took days to train.
        self.check(rebuilt=True)
        # Trained before `train` recorded its build: its train_config.json has no dataset.revision.
        self.refused("recorded no build", pinned=False)

    def test_a_local_skills_model_without_its_build_snapshot_is_refused(self) -> None:
        copy = data.trained_build_path(self.s, data.build_revision(self.write_build()))
        self.refused(f"whose copy {copy} is missing", kept="missing")
        self.refused(f"whose copy {copy} is changed", kept="changed")
        self.refused("Copy it there from the computer that trained the model", kept="missing")

    def write_build(self) -> Path:
        """The build.json `check` writes, for its name."""
        path = self.folder / "built.json"
        path.write_text(json.dumps({"built": "first", **self.build}))
        return path

    def test_snapshot_build_copies_the_build_once(self) -> None:
        build = self.folder / "build.json"
        build.write_text(json.dumps({"built": "first", **self.build}))
        revision = data.snapshot_build(self.s, build)
        kept = data.trained_build_path(self.s, revision)
        self.assertEqual(revision, data.build_revision(build))
        self.assertEqual(kept.read_bytes(), build.read_bytes(), "the same bytes as the build")
        self.assertEqual(list(kept.parent.iterdir()), [kept], "no partial file left behind")
        kept.write_text("kept")  # an existing snapshot is never rewritten
        self.assertEqual(data.snapshot_build(self.s, build), revision)
        self.assertEqual(kept.read_text(), "kept")

    def hub_model(self, revision: str | None) -> mock.MagicMock:
        """Check a Hub model whose train_config.json records the dataset `revision`, as LeRobot writes it;
        returns the Hub download mock."""
        from so_101.commands.policy import skills_build

        (self.folder / "train_config.json").write_text(json.dumps({"dataset": {
            "repo_id": "me/so101_test_skills", "root": None, "revision": revision, "episodes": None}}))
        (self.folder / "build.json").write_text(json.dumps(self.build))

        def download(repo, name, **kwargs):
            return str(self.folder / name)

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=download) as hub:
            self.assertEqual(skills_build(self.s, None, "me/act_so101_test_skills", "v1"), self.build)
        return hub

    def test_a_hub_model_is_checked_against_the_build_uploaded_at_the_commit_it_trained_from(self) -> None:
        commit = "0123456789abcdef0123456789abcdef01234567"
        hub = self.hub_model(commit)
        self.assertEqual(hub.call_args_list, [
            mock.call("me/act_so101_test_skills", "train_config.json", revision="v1"),
            mock.call("me/so101_test_skills", "build.json", repo_type="dataset", revision=commit)])

    def test_a_hub_model_trained_before_train_pinned_its_dataset_is_refused(self) -> None:
        # As train_config.json reads for a model trained before this: LeRobot read the dataset's moving
        # format tag, so the dataset's build.json may since describe another build.
        for revision in (None, "v3.0", "main"):
            with self.subTest(revision=revision):
                with self.assertRaises(SystemExit) as stop:
                    self.hub_model(revision)
                self.assertIn("recorded no build", str(stop.exception.code))
                self.assertIn("Train it again with `so101 train --skills`", str(stop.exception.code))


class CommandTest(unittest.TestCase):
    def test_a_model_missing_from_the_hub_says_to_train_it_first(self) -> None:
        from huggingface_hub.errors import RepositoryNotFoundError

        with mock.patch("so_101.tasks.hf_user", return_value="me"), \
                mock.patch("huggingface_hub.hf_hub_download", side_effect=RepositoryNotFoundError("404", response=mock.Mock(status_code=404))), \
                self.assertRaises(settings.ConfigError) as error:
            TASK.policy_config(None, "act", skills=True)
        self.assertIn("There's no model me/act_so101_test_skills on the Hub", str(error.exception))
        self.assertIn("Train it first: uv run so101 train test --policy act --skills", str(error.exception))

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
