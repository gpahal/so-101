# Skills: one model trained on the parts of a task

Status: built, not yet run on the arm or a remote server. How to use it is in the README, under Training one model on skills; this note keeps the design and its measurements. Not built: skills models in `sim eval`.
pick, place and combined dataset were made from them by a scratch script and checked (every row equal to
the raw session's, every cut confirmed on the camera frames). Nothing in `so101` builds, trains or runs
skills yet.

A task's demos can be cut into skills (for block-container, **pick** and **place**) and one model trained
on both, told which skill to do. At runtime a layer over the policy starts with the first skill and
switches to the next when the arm's readings show the handover. It's the "named skills" design from the
System 1 / System 2 research behind [`supervisor.md`](supervisor.md): the supervisor can then retry one
skill instead of the whole task.

## The words

| Word | What it is | Example |
|---|---|---|
| **task** | What you record demos of, in `tasks.toml` | `block-container` |
| **recordings** | What `record`, `serve` and `sim nudge` saved: never changed | `sessions/`, `runs/`, `nudges/` |
| **edits** | Your decisions about the recordings | `edits.toml`, each run's `labels.toml` |
| **dataset** | What a model trains on, built from recordings and edits. Two kinds per task | `whole`, `skills` |
| **skill** | A named part of every demo, cut where the arm's readings show an event | `pick`, `place` |
| **model** | A policy type trained on one of the datasets | `act_so101_block-container_skills` |

One review serves both datasets: `skills` is cut from the same kept ranges as `whole`, so a demo you trim
or delete in `review` is trimmed or gone in both.

## Defining skills

In `tasks.toml`, under the task, in the order they run:

```toml
[block-container.skills]
# Each skill is a part of every demo, from one event to another, plus or minus seconds. The events
# come from the arm's readings (src/so_101/skills.py): start, close, lift, carry, release, widest, end.
#   start    the kept demo's first frame (the arm at rest)
#   close    the gripper drops below 25: it closes on the block
#   lift     shoulder_lift has moved 4 deg since the close: the block leaves the mat (0.63 s after, median)
#   carry    the gripper has held steady in its holding band for 1.5 s (1.57 s after the close)
#   release  the gripper opens 4 past where it held the block
#   widest   the gripper's widest opening in the 2 s after the release (0.5 s after it, median)
#   end      the kept demo's last frame
# `handover` is the event that switches to this skill at runtime. It must fall inside both this
# skill's span and the last one's, which `so101 skills` checks on every demo.
pick  = { prompt = "Pick up the red block", from = "start", to = "carry + 0.5" }
place = { prompt = "Put the red block in the cylindrical container", from = "lift", to = "widest + 0.2", handover = "carry" }
```

Why these spans, from the 144 demos:
- **pick ends 0.5 s after the carry**, about 2 s after the close: the block is lifted and has held, so a
  slip or an edge pinch shows before the handover. At the close + 2 s, 43% of the carry is already done,
  which would leave the place skill 1.1 s in the quickest demos.
- **place starts at the lift**, before the handover, so it has seen every state from the handover on.
  The two overlap by about 1.4 s, and the frames in the overlap are in both skills with the same actions.
- **place ends 0.2 s after the widest opening**: the demos open about 0.5 s after the release and start
  closing for the return 0.57 s later, so the release + 1 s ended a third of them closing. At the widest
  + 0.2 s every demo's jaws are at least 12.6 above the hold.
- **The return to rest is in no skill.** `serve`'s Reset drives it.

| Skill | Frames (144 demos) | Median length |
|---|---|---|
| pick | 40,159 | 9.3 s |
| place | 17,269 | 4.0 s |
| `whole`, for comparison | 68,111 | 15.5 s |

## Commands

Every one takes the task like the rest of `so101`, and `--skills` picks the skills dataset and model
wherever there is a choice. Without it, everything works on `whole`, as today.

| Command | What it does | Touches |
|---|---|---|
| `so101 skills [task]` | Finds every skill's span in every kept demo, checks the handover falls inside both spans, prints the phase timings and any demo it couldn't cut, and draws a contact sheet of each cut | Reads `data/`; writes `outputs/skills/<task>/` |
| `so101 build [task] --skills` | Builds `datasets/skills/`: each demo's pick episode, then its place episode | `data/<task>/datasets/skills/` |
| `so101 train [task] --skills [--policy act\|pi05]` | Builds if stale, uploads to `<you>/so101_<task>_skills`, trains `<you>/<policy>_so101_<task>_skills` | Hub, HF Jobs credit |
| `so101 serve [task] --skills [--policy …] [--remote]` | Runs the skills model with the skill layer switching skills | The arm |
| `so101 server start [task] --skills --policy pi05` | The cloud GPU server for the skills model | HF Jobs credit |
| `so101 tasks` | Also lists each task's skills, and which datasets are built and current | Reads only |

A session, start to finish:

```
uv run so101 review                       # as today: trims apply to both datasets
uv run so101 skills                       # check the cuts: 144 of 144 cut, sheets in outputs/skills/
uv run so101 train --skills --policy pi05 # builds datasets/skills, uploads, trains one model on both skills
uv run so101 server start --skills --policy pi05
uv run so101 serve --skills --policy pi05 --remote
```

- **`build --skills --with ...`** adds runs, each cut split where the skill changes: by the run's
  `control.skill` column for a skills model's run, by its `grasp` events for any other
  ([`policy-layers.md`](policy-layers.md)).
- **`run --skills`** runs a skills model with `run`'s autopilot ([`run.md`](run.md)).
- **`sim eval`** doesn't: it runs LeRobot's own loop, which the policy stack doesn't fit in without a
  wrapper around LeRobot's policy.

## On disk

`train/` and `build.json` become one of two datasets, each with its build record inside it:

```
data/<task>/
  sessions/<time>/          recordings, never changed (unchanged)
  runs/<time>/              recordings, never changed (unchanged)
  nudges/<time>/            recordings, never trained on (unchanged)
  edits.toml                your trims and deletions (unchanged)
  datasets/
    whole/                  was train/: the whole demos, plus runs with --with
      meta/ data/ videos/   a LeRobot dataset
      build.json            was ../build.json: what it was built from
    skills/                 every demo's pick and place episodes, in order
      meta/ data/ videos/
      build.json            what it was built from, the skill definitions and every demo's cuts
outputs/
  skills/<task>/            `so101 skills`' contact sheets (new)
  policies/ rollouts/ train/ check/ runs/ sim/ failbench/   (unchanged)
```

Why `build.json` moves inside: a build writes its dataset to a temporary folder and renames it into
place. With the record inside, the rename swaps both at once, so a kill can never leave a dataset beside
a record of another build. Today `build` has to delete `build.json` before the swap to get the same
guarantee.

## On the Hub

The `_skills` suffix follows the dataset everywhere; the names of everything you have today stay.

| Repo (all private) | What | Written by |
|---|---|---|
| `<you>/so101_<task>_raw` | Backup of the recordings and edits, under the same paths | `backup` (unchanged) |
| `<you>/so101_<task>_nudges` | The pushes `sim nudge` records, for the sim's friction | `sim physics` (unchanged) |
| `<you>/so101_<task>` | The `whole` dataset | `train` (unchanged) |
| `<you>/so101_<task>_skills` | The `skills` dataset | `train --skills` |
| `<you>/<policy>_so101_<task>` | A model trained on `whole` | `train` (unchanged) |
| `<you>/<policy>_so101_<task>_skills` | A model trained on `skills` | `train --skills` |

Datasets aren't backed up: `build` remakes them from the recordings and edits, which `backup` covers.
Tag models as today (`hf repos tag create <repo> <tag>`), and `--revision` runs a tag.

## One model for both skills

The skills dataset carries the skill two ways, so the same dataset trains either policy:

- **A prompt per episode** ("Pick up the red block", "Put the red block in the cylindrical
  container"), LeRobot's `task`. π0.5 reads it. At runtime the layer changes it with the inference
  engine's `set_task`, which LeRobot 0.7's local and remote engines both have.
- **A skill one-hot**, `observation.environment_state`: `[1, 0]` for pick, `[0, 1]` for place. ACT has
  no text input but takes an environment state as one more encoder token (`ACTConfig.env_state_feature`).
  At runtime the layer adds it to each observation.

Checked here: LeRobot 0.7's `ACTConfig` and `PI05Config` both accept a dataset with that column, and π0.5's
model code never reads it. Not yet measured: whether ACT learns to switch on the one-hot. No published
use was found, so it's the first thing to test; the fallback is a model per skill, which the same
dataset gives by filtering on the prompt.

Steps follow the demo count, as for `whole`, so time and cost are the same: the skills dataset has 84% of
`whole`'s frames (57,428 against 68,111) because it leaves out the return.

## Running a skills model

`--skills` adds the skill sequencer to the policy stack ([`policy-layers.md`](policy-layers.md)):
`EngineDriver` → `Perception` → `SkillSequencer` → `Supervisor`. Like every layer it sends the controller no
commands; it conditions the model through the stack, and only the model acts on that.

1. Each attempt starts with the first skill: the prompt and one-hot are pick's.
2. The perception's detector runs on every tick the model drives. When the next skill's `handover` event fires (the carry, about
   1.57 s after the close), the layer switches: it sets the new prompt and one-hot and clears the
   policy's queue, so no pick action from before the switch reaches the arm. The cost per switch: one
   ACT inference locally (44 ms, measured); for remote π0.5, an estimated stop of 0.7 to 1.3 s while a new
   plan arrives, because a request was in flight about 90% of the time in earlier runs.
3. After the last skill, the attempt ends as today: the supervisor's done check, or your Reset.
4. Each switch is a `skill` event in `events.jsonl`, with the frame, so a run says which skill each
   frame came from.
5. With the supervisor on, a missed grasp still plays back and retries inside pick. Later, a block dropped
   during place could switch back to pick from where the arm is; that needs pick demos that start from
   such states, which the demos don't have.

## Moving the files

Done for block-container: `data/block-container/train/` moved to `datasets/whole/` with `build.json` inside
it, unchanged, so the build stayed current and nothing retrained. On the Hub nothing moved. A task folder
still laid out the old way just builds `datasets/whole/` afresh the next time it's built, and `backup`
leaves out both layouts' datasets.

## How it's built

- `src/so_101/skills.py`: the skill definitions from `tasks.toml` and their checks, the events of a
  demo's frames (the supervisor's detector, plus lift and widest), and each demo's spans. The runtime
  layer is `policy/skills.py`'s `SkillSequencer`.
- `tasks.py`: parse `[<task>.skills]`.
- `data.py`: `dataset_root(s, task, kind)` and `build_manifest_path(s, task, kind)` in place of
  `train_root`; the layout docstring. Every user of `train_root` goes through these, so `sim`'s demos
  folder follows.
- `build.py`: a `kind`; for skills, plan two episodes per kept demo from `skills.py`'s spans, write
  `task` and `observation.environment_state` per frame, and keep `build.json` inside the dataset.
- `cli.py`: the `skills` command, `--skills` on `build`, `train`, `serve` and `server start`, and the
  `_skills` repo names (`tasks.Task.policy_for` and `dataset` take the kind).
- `policy/stack.py`: adds the `SkillSequencer` when the run is a skills run.
- Tests (`tests/test_skills.py`): the definitions and their refusals; the events and spans on made-up
  traces, with a demo with no carry and a handover outside its spans refused; a skills build on fake
  sessions with every row, one-hot and prompt checked against the source; the layer switching once on
  the carry, clearing the queue, setting the prompt and one-hot and logging the `skill` event; the
  one-hot reaching the frame LeRobot builds for the policy; the layer as serve's policy on the fake arm;
  the Hub names and the server labels.
- README: the on-disk tree, the Hub names, and a "Training on skills" section.

## Open questions

- Does ACT switch behaviour on a two-value one-hot? Train both policies and compare against the `whole`
  models on the same block and jar places.
- Is ACT's 100-step chunk (3.3 s) too long for a 4 s place? The switch clears the queue, so place starts
  fresh, but a shorter chunk may follow the jar better.
- How many retry-start pick demos does a retry need? No source measured it on an SO arm.
