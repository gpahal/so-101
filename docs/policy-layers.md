# Policy layers: one interface from the model up, and what each frame records

Status: built, tested on the fake arm and stand-in models, not yet run on the arm. Replaced the ad hoc driver
methods that `serve`'s supervisor and skill layers shared in the first build. `serve` and `run` both step it.
How to use what it records is in the README, under What serve records. `run` on the control center is in [`run.md`](run.md); the web reviewer that
shows what these record is in [`reviewer.md`](reviewer.md).

## The idea

A **policy** is anything the control loop steps. A **layer** is a policy that wraps a policy: it inherits
every method from the one it wraps and overrides only what it adds. The loop sees one policy, the top of a
stack that one function builds:

```
EngineDriver   the model: LeRobot's inference engine, local or on the cloud GPU (the only LeRobot code)
Perception     the grasp detector (grasp/) over each tick the model drove: close, carry, release, stuck rules
SkillSequencer with --skills: which skill the model is told to do, switched at each handover
Supervisor     with [supervisor] mode shadow or act: vision checks, playback, hold, asking for a person
```

`Perception` is always in the stack, so every `serve` and `run` logs its grasps, whatever model runs.

## The interface (`policy/base.py`)

| Method | Called | A layer's default |
|---|---|---|
| `start()`, `stop()`, `close()` | once each, by the process | forwards |
| `begin_episode(n)`, `end_episode()` | when an attempt opens and closes | forwards |
| `run(handback)` | the arm is handed to the stack: an attempt starts, a person resumes, or a layer above gives it back | forwards |
| `pause()` | the arm is taken from the stack | forwards |
| `hold(robot)` | each tick the stack doesn't drive | forwards |
| `step(robot, tick) -> Output` | each tick the stack drives | forwards |
| `condition(conditioning)` | which skill the model is told to do | forwards |
| `maintain() -> list[str]` | every tick, in every state: messages for the operator | forwards |
| `health() -> Health`, `record_failure(detail)` | the model's failure and reconnecting, read; a failure the loop saw, written | forwards |

- **`Tick`**, built by the loop once per tick: `now` (the loop's clock), `frame` (the open episode's frame
  number, or -1), `observation` (the robot's raw observation: joints and camera frames), `pose` (the
  measured joints) and `at_rest` (every joint within `rollout.TOLERANCE` of the rest pose).
- **`Output`**, returned by `step`: `action` (the model's action when it reached the arm, else None),
  `driver` (`POLICY`, `WAITING` when the model sent nothing, `PLAYBACK` or `HOLD` when a layer drove),
  `signals` (the perception's events this tick), `readings` (the perception's state: policy seconds,
  rules near firing, closes, regrasps, carried, released, fired) and `asks` (an `Ask(kind, text)` when a layer
  wants a person, else None: `kind` is `stuck`, `done` or `layer`, and the loop logs it as `ask` in the
  `state` event for `policy_asked`) and `skill` (the skill the model is conditioned on, or None for no change: the controller keeps the
  last one for the attempt, so a playback, a hold or a person's Teleop is recorded under the skill it
  interrupted). Outputs flow up the stack, so a layer reads the signals of the one below and adds its own.
- **`Handback(why, moved)`**: `why` is `START` (the stack's first handback in an attempt, even one a person
  began in Teleop), `PERSON` (a person resumed) or `LAYER` (a layer above gave the arm back after a
  playback or a hold); `moved` is whether the arm may have moved while someone else had it. The controller
  sets `moved` when the attempt had Teleop, Rewind or Returning ticks, or a layer's playback, since the
  stack last drove. A layer's playback takes back the controller's history of the motion it undid, and its
  hold adds none, so a person's Rewind afterwards never replays either.
- **`Conditioning(prompt, skills, index)`**: what the model is told. Only `EngineDriver` acts on it: it
  sets the engine's task (`set_task`, which pi0.5 reads), adds the one-hot `observation.environment_state`
  to the frame it builds (which ACT reads), and, while the model drives, clears the queued actions, so no
  action for the old skill reaches the arm.
- **`LayerContext`**, the same for every layer: `event(type, **fields)` (the run's event log), `clock`,
  `grasp` (the gripper and rule thresholds, `grasp/grasp.toml`) and `program` (the supervisor's program,
  `supervisor/program.toml`, whose `watch.near` the perception reads).
- A layer's own exceptions are its own (`Layer.safely`): logged once as an `error` event, announced once
  through `maintain`, and the layer becomes a pass-through. `safely` wraps only the layer's own logic:
  every call down the stack runs outside it, so an exception from the arm or the model reaches the loop
  unchanged. In a skills run the perception and the sequencer are `essential` (`stack.build` marks them):
  either one broken reports itself in `health().failure`, so the loop faults and Resume is refused, rather
  than the model staying on its first skill.

## The layers

- **`EngineDriver`** (`policy/engine.py`, from `control/policy.py`): unchanged inside, plus `condition`.
- **`Perception`** (`policy/perception.py`): one detector (`grasp/detector.py`, a pure module that
  `so101 skills`, `build --skills` and `so101-failbench candidates` run too) per attempt, fed on each tick
  the model drove,
  with the policy's own seconds. Logs a `grasp` event for each close, carry, regrasp and release, and a
  `rule` event when a stuck rule fires. After a handback, on the next tick it feeds, it calls
  `Detector.resumed(moved, pose)`: the detector's timers start again, and after a move with the jaws open
  it forgets the open close, the carry and the release, so a close a playback undid is no regrasp and a
  second grasp is seen in full; forgetting a carry or release also gives `no_carry` its full 14 s of
  driving again, from the handback, since the time spent carrying was never counted. The detector (`grasp/detector.py`) counts hover and frozen time only on
  ticks the model's action reached the arm.
- **`SkillSequencer`** (`policy/skills.py`): starts each attempt on the first skill; at the next skill's
  handover signal it conditions the model on that skill and logs a `skill` event. Deciding to switch runs
  under `safely`; telling the model (`inner.condition`) runs outside it, right after in the same call, so
  an error from the model reaches the loop as the model's, not as a broken sequencer. On a handback with
  `moved`, it reads the perception's readings on the first tick the perception feeds, after its
  `Detector.resumed`: if neither a carry nor a release still stands, it goes back to the first skill,
  whatever skill it was on, since whatever was held or let go was undone. This covers a playback that
  undid a grasp, your own Rewind, and a block let go outside the jar and rewound to before the close.
- **`Supervisor`** (`supervisor/layer.py`): as before, but reads the perception's signals and readings
  instead of keeping its own detector, hands the arm back with `Handback(LAYER, moved=True)` after a
  playback and `moved=False` after a hold, and returns its requests for a person in `Output.asks`.

`policy/stack.py`'s `build(driver, ctx, skills=, judges=, mode=, cameras=)` composes them in this order and
is the only place that knows the order. `control/main.py` calls it.

## What each frame and event records

Every `serve` and `run` recording gets two columns:

| Column | Values | Read by |
|---|---|---|
| `control.skill` | the index of the skill the model was conditioned on in `run.json`'s `policy.skills`, or -1 | the reviewer, `build --skills --with` |
| `control.driver` | 0 not the stack (a scripted move, a person, a hold outside Autonomous), 1 the model, 2 waiting, 3 a layer's playback, 4 a layer's hold | analysis, the reviewer |

- Analysis takes waiting frames from `control.driver` (2, 3 or 4 in Autonomous), and from an all-NaN
  `control.policy_action` in runs recorded before the column. Neither is ever trained on as the model's.
- Events, generic by layer: `grasp` and `rule` (Perception), `skill` (SkillSequencer), `playback` (any
  layer that plays motion back: `start_frame`, `end_frame`, `seconds`, `cause`, as serve's `rewind`
  event), `supervisor` (the supervisor's checks and decisions). Analysis reads `playback` events without
  knowing which layer wrote them, and treats a `policy_asked` pause as an interruption.

## Training on runs per skill

`build --skills --with corrections,policy-successes,...` cuts runs as `build --with` does, then splits each
cut where the skill changes, by what happened (`build.run_skills`, `skills.label_run`): the attempt's joint
readings go through one detector as the perception's do, frames a script or a layer's playback moved aren't
replayed, Teleop frames are (a re-grip in Teleop, the jaws back on the block when it ends, is rolled
back to place), the sequencer's handback rule applies after each,
each skill's `to` bound applies, and only the frames a skills model drove keep the skill it was told
(`control.skill`). So a person's correction is cut by what the arm did, whatever the model was told, and a
person's re-grip during the carry is never a release.
[`skills.md`](skills.md) has the steps.

Each piece is an episode with its skill's prompt and one-hot. A piece under 0.5 s is left out.

## Tests

A contract suite in `tests/test_policy.py` runs against every policy (the leaf over a scripted engine, each layer over a fake leaf,
and the full stack): nothing reaches the arm after `pause`; `begin_episode`, `end_episode`, `condition`,
`health` and `record_failure` reach the leaf; a handback reaches the leaf; `maintain` returns a list; a
layer's own error never escapes, checked by making each of every layer's own methods raise in turn while a
demo's trace is stepped through, with its pauses and resumes; `stop` reaches the leaf. Each layer has its own tests on top, including the playback that undoes a
grasp sending the skill back to the first.
