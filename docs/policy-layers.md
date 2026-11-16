# Policy layers: one interface from the model up, and what each frame records

Status: built, tested on the fake arm and stand-in models, not yet run on the arm. Replaced the ad hoc driver
methods that `serve`'s supervisor and skill layers shared in the first build. How to use what it records is
in the README, under What serve records. `run` on the control center is in [`run.md`](run.md); the web reviewer that
shows what these record is in [`reviewer.md`](reviewer.md).

## The idea

A **policy** is anything the control loop steps. A **layer** is a policy that wraps a policy: it inherits
every method from the one it wraps and overrides only what it adds. The loop sees one policy, the top of a
stack that one function builds:

```
EngineDriver   the model: LeRobot's inference engine, local or on the cloud GPU (the only LeRobot code)
Perception     the gripper rules over each tick the model drove: close, lift, carry, release, stuck rules
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
  number, or -1), `observation` (the robot's raw observation: joints and camera frames) and `pose` (the
  measured joints).
- **`Output`**, returned by `step`: `action` (the model's action when it reached the arm, else None),
  `driver` (`POLICY`, `WAITING` when the model sent nothing, `PLAYBACK` or `HOLD` when a layer drove),
  `signals` (the perception's events this tick), `readings` (the perception's state: policy seconds,
  rules near firing, closes, regrasps, carried, released, fired) and `asks` (why a layer wants a person,
  or None) and `skill` (the skill the model is conditioned on, or None for no change: the controller keeps the
  last one for the attempt, so a playback, a hold or a person's Teleop is recorded under the skill it
  interrupted). Outputs flow up the stack, so a layer reads the signals of the one below and adds its own.
- **`Handback(why, moved)`**: `why` is `START` (an attempt starts), `PERSON` (a person resumed) or `LAYER`
  (a layer above gave the arm back after a playback or a hold); `moved` is whether the arm may have moved
  while someone else had it. The controller sets `moved` when the attempt had Teleop or Rewind ticks since
  the stack last drove.
- **`Conditioning(prompt, skills, index)`**: what the model is told. Only `EngineDriver` acts on it: it
  sets the engine's task (`set_task`, which pi0.5 reads), adds the one-hot `observation.environment_state`
  to the frame it builds (which ACT reads), and, while the model drives, clears the queued actions, so no
  action for the old skill reaches the arm.
- **`LayerContext`**, the same for every layer: `event(type, **fields)` (the run's event log), `clock`,
  `grasp` (the gripper and rule thresholds, `grasp/grasp.toml`) and `program` (the supervisor's program,
  `supervisor/program.toml`, whose `watch.near` the perception reads).
- A layer's own exceptions are its own: logged once, and the layer becomes a pass-through. An exception
  from the arm or the model reaches the loop unchanged.

## The layers

- **`EngineDriver`** (`policy/engine.py`, from `control/policy.py`): unchanged inside, plus `condition`.
- **`Perception`** (`policy/perception.py`): one detector per attempt, fed on each tick the model drove,
  with the policy's own seconds. Logs a `grasp` event for each close, carry, regrasp and release, and a
  `rule` event when a stuck rule fires. On a handback it restarts the detector's timers.
- **`SkillSequencer`** (`policy/skills.py`): starts each attempt on the first skill; at the next skill's
  handover signal it conditions the model on that skill and logs a `skill` event. On a handback with
  `moved`, it checks the gripper on its next tick: reading open, it goes back to the first skill, since
  whatever was held was let go or undone. This covers a playback that undid a grasp and your own Rewind.
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
cut where the skill changes:
- a skills run: by its `control.skill` column;
- a whole-model run: by its `grasp` events, the first skill until the next skill's handover event (the
  carry), the next skill after it.

Each piece is an episode with its skill's prompt and one-hot. A piece under 0.5 s is left out.

## Tests

A contract suite runs against every policy (the leaf over a scripted engine, each layer over a fake leaf,
and the full stack): nothing reaches the arm after `pause`; `begin_episode`, `end_episode`, `condition`,
`health` and `record_failure` reach the leaf; a handback reaches the leaf; `maintain` returns a list; a
layer's own error never escapes. Each layer has its own tests on top, including the playback that undoes a
grasp sending the skill back to the first.
