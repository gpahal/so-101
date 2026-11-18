# Skills: one model trained on the parts of a task

Status: built. ACT and π0.5 skills models (pick, place and return) have run on the arm in `so101 run`,
π0.5's on the cloud GPU ([`results/act-pi05-block-container-supervisor-shadow.md`](../results/act-pi05-block-container-supervisor-shadow.md)). How to use it is in the README, under
[Training one model on skills](../README.md#training-one-model-on-skills); this note keeps the design and its
measurements. Not built: skills models in `sim eval`.

A task's demos can be cut into skills (for block-container, **pick**, **place** and **return**) and one
model trained on all of them, told which skill to do. At runtime a layer over the policy starts with the first skill and
switches to the next when the arm's readings show the handover. It's the "named skills" design from the
System 1 / System 2 research behind [`supervisor.md`](supervisor.md): the supervisor can then retry one
skill instead of the whole task.

## The words

| Word | What it is | Example |
|---|---|---|
| **task** | What you record demos of, in `tasks.toml` | `block-container` |
| **recordings** | What `record`, `serve`, `run` and `sim nudge` saved: never changed | `sessions/`, `runs/`, `nudges/` |
| **edits** | Your decisions about the recordings | `edits.toml`, each run's `labels.toml` |
| **dataset** | What a model trains on, built from recordings and edits. Two kinds per task | `whole`, `skills` |
| **skill** | A named part of every demo, cut where the arm's readings show an event | `pick`, `place` |
| **model** | A policy type trained on one of the datasets | `act_so101_block-container_skills` |

One review serves both datasets: `skills` is cut from the same kept ranges as `whole`, so a demo you trim
or delete in `review` is trimmed or gone in both.

## Defining skills

In `tasks.toml`, under the task, in the order they run (the comments explain the events):

```toml
[block-container.skills]
# Each skill is a part of every demo, from one event to another, plus or minus seconds. The events
# come from the arm's readings (src/so_101/skills.py): start, close, lift, carry, release, widest, end.
#   start    the kept demo's first frame (the arm at rest)
#   close    the gripper drops below 25: it closes on the block
#   lift     shoulder_lift has moved 4 deg since the close: the block leaves the mat (0.63 s after, median)
#   carry    the gripper has held steady in its holding band for 1.5 s (1.57 s after the close), or the
#            elbow moved 25 deg with the gripper above 12
#   release  the gripper opens 4 past where it held the block, and above the hold band's top (22)
#   widest   the gripper's widest opening in the 2 s after the release (0.5 s after it, median)
#   end      the kept demo's last frame
# `handover` is the event that switches to this skill at runtime: a carry or a release. It must fall
# inside both this skill's span and the last one's, which `so101 skills` checks on every demo.
pick   = { prompt = "Pick up the red block", from = "start", to = "carry + 0.5" }
place  = { prompt = "Put the red block in the cylindrical container", from = "lift", to = "widest + 0.2", handover = "carry" }
return = { prompt = "Return to the rest pose", from = "release", to = "end", handover = "release" }
```

`tasks.load` refuses a handover on `close`: as the policy runs, a close that misses the block reopens, and
by then the switch would already have told the model to place.

Why these spans:
- **pick ends 0.5 s after the carry**, about 2 s after the close: the block is lifted and has held, so a
  slip or an edge pinch shows before the handover. At the close + 2 s, 43% of the carry is already done,
  which would leave the place skill 1.1 s in the quickest demos.
- **place starts at the lift**, before the handover, so it has seen every state from the handover on.
  The two overlap by about 1.5 s, and the frames in the overlap are in both skills with the same actions.
- **place ends 0.2 s after the widest opening**: the demos open about 0.5 s after the release and start
  closing for the return 0.57 s later, so the release + 1 s ended a third of them closing. At the widest
  + 0.2 s every demo's jaws are at least 12.6 above the hold.
- **return runs from the release to the demo's end**, so the model drives the arm home as the demos do,
  rather than leave it to Reset. It starts at its handover, the release, which place runs on past by 0.70 s
  (median; 0.57 to 0.80 s for 90% of the demos), so the switch is inside both. A handback after it
  that moved the arm with the jaws open sends the run back to pick (step 6 of
  [Running a skills model](#running-a-skills-model)).

Measured with `so101 skills` on the 125 demos kept of the 180 recorded (144 at six jar places, 24 each, and
36 edge cases), all of which cut. The other 55 are deleted in `edits.toml`, each with its reason: mostly a jaw
within about 2 mm of the block before closing, a block well off square to the jaws, a jar the drop pushed, an
arm still moving as the jaws opened, or a reach with long stops. 14 to 20 of the 24 demos at each place are
kept; the two places farthest from the arm, where faults were commonest, keep 11 with a moderate fault.

| Skill | Frames (125 demos) | Median length |
|---|---|---|
| pick | 35,405 | 9.5 s |
| place | 14,981 | 3.8 s |
| return | 17,481 | 4.7 s |
| `whole`, for comparison | 59,580 | 15.8 s |

The release to the widest opening takes 0.50 s, and the widest opening to the demo's end 4.13 s (median).

The hold is the median reading over the last 1.5 s the gripper held steady (`Detector._steady`), which a
block settling in the jaws still moves and opening jaws don't. One edge case opens the jaws only 4.9 past its
hold, from 18.9 to 23.8 over 0.3 s, as the block slides into the jar: the last reading inside the hold band
would follow those jaws up to 21.6 and ask for 25.6, so it releases only with the median hold (it is among the
55 deleted, for that narrow release). A release also needs the reading above the hold band's top (22),
because a held block keeps the gripper inside the band: a carried block whose reading rises 4.5 inside it
within 2 s, from 17.0 to 21.5, is still held. Jaws that open slower than about `hold_steady` per `hold_s`
(1.3 a second) carry the hold with them, as a settling block does, so such an opening is a release only once
it passes 4 over where the hold stopped: 18.9 to 24 over 4 s is none.

## Commands

Every one takes the task like the rest of `so101`, and `--skills` picks the skills dataset and model
wherever there is a choice. Without it, everything works on `whole`. The commands are listed in the README,
under [Training one model on skills](../README.md#training-one-model-on-skills). A session, start to finish:

```
uv run so101 review                       # trims apply to both datasets
uv run so101 skills                       # check the cuts: 125 of 125 cut, sheets in outputs/skills/
uv run so101 train --skills --policy pi05 # builds datasets/skills, uploads, trains one model on all three
uv run so101 server start --skills --policy pi05
uv run so101 serve --skills --policy pi05 --remote
```

- **`build --skills --with ...`** adds runs, each cut split where the skill changes, labelled by what
  happened (`skills.label_run`, called from `build.run_skills`):
  1. The attempt's joint readings go through one detector frame by frame, as the perception's do at
     runtime. The attempt starts on the first skill and moves to the next at its handover event.
  2. Frames a script moved aren't replayed (the move to the start pose, a Rewind, the return to rest, by
     `control.phase`, and a layer's playback, by `control.driver`), nor frames the running model's detector
     never sees: Paused, Handover and a layer's hold. Teleop frames and Autonomous frames the model drove or
     waited on are fed through `grasp.replay.LiveFeed`, which the perception also runs
     (`tests/test_replay.py`), on the model's clock: 1/fps per fed frame, and nothing for the first fed
     frame after a handback. So a 2 s pause with the jaws on the block right after a close is no carry: the
     carry comes `hold_s` of fed frames after it. After a paused or held stretch the next fed frame tells the
     detector the arm was held (`resumed(False)`), as after a plain Pause; a scripted or Teleop stretch in
     the same gap wins and says it moved. A Pause and Resume in one tick leave no Paused frame but are
     replayed from the run's `state` events (`analysis.Attempt.handbacks`). Only frames the model drove, and
     Teleop, count as driven time for the stuck rules. A person's close, carry and release in Teleop count,
     so a person who finishes the task in Teleop is in the return from their release; one who opens the jaws
     on a carried block and has the gripper back in the hold band (16.5 to 22) when the Teleop stretch ends
     re-gripped it, so the replay goes back to before that release and the stretch stays place.
  3. After a stretch that moved the arm (a script, a playback or Teleop), the next frame replayed tells the detector the arm moved (`Detector.resumed`), as the
     perception does after a handback, so its timers start again. If it then shows neither a carry nor a
     release, the replay starts again on the first skill, as the sequencer does at runtime. The tests check
     that the labels switch within a frame of where the sequencer does, across Rewinds.
  4. A frame a skills model drove (an Autonomous frame with a `control.skill`) keeps the skill it was told;
     every other frame takes the replay's. So a correction that starts in pick and carries the block is
     place from the carry, whatever the model was told, and a person's attempt alone is cut like a demo.
  5. Each skill's `to` bound applies, from the events of the replay its frame is in: a frame past it is in
     no skill and is left out. A bound whose event didn't happen
     (pick's carry, in an attempt that never held the block) doesn't apply.

- **`run --skills`** runs a skills model with `run`'s autopilot ([`run.md`](run.md)).
- **`sim eval`** doesn't: it runs LeRobot's own loop, which the policy stack doesn't fit in without a
  wrapper around LeRobot's policy.

## On the Hub

The `_skills` suffix follows the dataset everywhere.

| Repo (all private) | What | Written by |
|---|---|---|
| `<you>/so101_<task>_raw` | Backup of the recordings and edits, under the same paths | `backup` |
| `<you>/so101_<task>_nudges` | The pushes `sim nudge` records, for the sim's friction | `sim physics` |
| `<you>/so101_<task>` | The `whole` dataset | `train` |
| `<you>/so101_<task>_skills` | The `skills` dataset | `train --skills` |
| `<you>/<policy>_so101_<task>` | A model trained on `whole` | `train` |
| `<you>/<policy>_so101_<task>_skills` | A model trained on `skills` | `train --skills` |

Datasets aren't backed up: `build` remakes them from the recordings and edits, which `backup` covers.
Tag models with `hf repos tag create <repo> <tag>`, and `--revision` runs a tag.

## One model for all the skills

The skills dataset carries the skill two ways, so the same dataset trains either policy:

- **A prompt per episode** ("Pick up the red block", "Put the red block in the cylindrical
  container", "Return to the rest pose"), LeRobot's `task`. π0.5 reads it. At runtime the layer changes it with the inference
  engine's `set_task`, which LeRobot 0.7's local and remote engines both have.
- **A skill one-hot**, `observation.environment_state`: `[1, 0, 0]` for pick, `[0, 1, 0]` for place and
  `[0, 0, 1]` for return. ACT has no text input but takes an environment state as one more encoder
  token (`ACTConfig.env_state_feature`). At runtime the driver adds it to each observation, from the
  first tick of a run.

LeRobot 0.7's `ACTConfig` and `PI05Config` both accept a dataset with that column, and π0.5's model code
never reads it. ACT's skills model, which has only the one-hot to switch on, placed the block in 5 of 6
attempts on the arm; 6 attempts can't say whether it does as well as the whole-task model. If it doesn't,
the fallback is a model per skill, which the same dataset gives by filtering on the prompt.

Steps follow the demo count, as for `whole`, so time and cost are the same. The skills dataset has 114% of
`whole`'s frames (67,867 against 59,580 on the 125 kept demos): place overlaps pick by about 1.5 s and return by
0.7 s. Without return as a skill it would have 85% (50,386), because it would leave the return out.

## Running a skills model

`--skills` adds the skill sequencer to the policy stack ([`policy-layers.md`](policy-layers.md)):
`EngineDriver` → `Perception` → `SkillSequencer` → `Supervisor`. Like every layer it sends the controller no
commands; it conditions the model through the stack, and only the model acts on that.

1. Each attempt starts with the first skill: the prompt and one-hot are pick's.
2. The perception's detector runs on every tick the model drives. When the next skill's `handover` event fires (the carry, about
   1.57 s after the close), the layer switches: it sets the new prompt and one-hot and clears the
   policy's queue, so no pick action from before the switch reaches the arm. The cost per switch: one
   ACT inference locally (44 ms, measured); for remote π0.5, 0.8 to 1.3 s with no action while a new plan
   arrives, measured on the arm.
3. At the release it switches to return, and the model drives the arm home. The attempt ends on
   the supervisor's done check, the time limit or your Reset.
4. Each switch is a `skill` event in `events.jsonl`, with the frame, so a run says which skill each
   frame came from.
5. Deciding to switch is the sequencer's own logic, and a failure there turns it into a pass-through
   (`Layer.safely`); telling the model below about the switch runs outside that, so an error from the
   model reaches the control loop as the model's.
6. After a handback that moved the arm (a Teleop, a Rewind, a layer's playback), the sequencer follows the
   perception: on the first tick the model drives, the detector is told the arm moved
   (`Detector.resumed`), and if it then shows neither a carry nor a release, the attempt starts again on
   the first skill, whatever skill it was on. So a missed grasp the supervisor plays back retries inside
   pick, and a block let go outside the jar and rewound to before the close is picked again rather than
   left to return. Open means at `closed_below` (25) or above, or, once the block was carried or let go,
   above `hold_band`'s top (22), where no block is held: the line a release crosses, so a narrow release
   (18.9 to 23.8) that a person undoes re-arms too. With the jaws closed after the move nothing changes: a block still held keeps place,
   and jaws a person closed on the block count as a close. A person who teleoperates after a real release
   and resumes with the jaws open also sends it back to pick, because the detector can't tell that from a
   block put back; that was chosen over leaving a dropped block on return. The model drives that first
   tick on the skill it had, and the switch conditions it from the next tick on.

## When a build is current

A skills build records in its `build.json` what decides its cuts (`data.build_inputs`), and is current
only while they're unchanged:

- each skill's name, prompt, bounds and handover, in order;
- the `[gripper]` thresholds, by the sha256 of their values (`skills.gripper_hash`), not the file's path:
  the same values in another file cut the same skills;
- `skills.CUT_RULES_VERSION`, raised with any change to `LIFT_DEG`, `WIDEST_WITHIN_S`, `MIN_SPAN_S`, the
  code that cuts a demo, or how the detector finds the events a cut uses;
- for a build with runs only (`--with`), `skills.RUN_LABEL_VERSION`, raised with any change to how
  `label_run` labels a run, to what `build.run_skills` gives it, or to how the grasp detector finds events
  from the same thresholds, which can move a run's labels without moving any demo's cut. A demos-only build
  doesn't record it.

`skills.contract` is the one definition of these fields: `data.build_inputs` records it and the
skills-model check below compares it, so the two can't drift.

The stuck rules (`[rules]`) cut nothing, so retuning them leaves the build current. `build.json` holds no
folder of this computer, since `train` uploads it with the dataset: a note that quoted one writes the data
folder as `<data_dir>` and the home folder as `~`.

A rebuild renames the old dataset aside, renames the new one in, and only then deletes the old
(`build.swap`), so a failed rename puts the old one back and a kill never leaves half of one. Each dataset
keeps its `build.json` inside it, so the rename swaps both at once, and a kill never leaves a dataset beside
another build's record.

## A model tied to its skills

`run` and `serve` refuse a skills model that doesn't fit the task before anything connects
(`commands.policy.check_skills_model`):

- a model that reads the one-hot without `--skills`, and an ACT model without it with `--skills`;
- a one-hot of another size than `[<task>.skills]`;
- a model whose build's `build.json` records another `skills.contract` than this checkout's: any skill's
  name, prompt, `from`, `to` or `handover`, or their order; the `[gripper]` thresholds' hash; the cut
  rules (`CUT_RULES_VERSION`); and, when the build took runs, the run labelling, the grasp detector
  included (`RUN_LABEL_VERSION`).
  One refusal names every difference, by skill and field (`place: handover carry here, release in the
  model's build`), and a `build.json` that leaves any of them out is refused, naming it.

`run.json`'s `policy` records the skills and the `[gripper]` thresholds' hash the model was checked
against (`skills`, `gripper_sha256`). `so101-control` loads `tasks.toml` and the grasp thresholds again
when it starts, which can be minutes later after a download, and exits before anything connects if
either no longer matches (`control.main.require_run_skills`), so a run never switches skills where its
model wasn't trained to, or records skills it didn't use.

The build is the one the model was trained on, which `train --skills` records as `dataset.revision` in
the model's `train_config.json` (`commands.policy.skills_build`):

- On Hugging Face Jobs, `train` uploads the dataset with its `build.json`, and the job trains from that
  upload's commit, not the moving format tag, so the record names the very data it trained on. The
  check reads `build.json` at that commit of the Hub dataset `dataset.repo_id` names. A later
  `train --skills` uploads over `main` and leaves that commit as it was.
- With `--target local`, the record is `so101-build-` and the sha256 of the `build.json` it trained on
  (`data.build_revision`). Every build writes that file anew, with the time it was built, so the sha256
  names one build. LeRobot reads a local dataset from `dataset.root` and never uses the revision.
  Before training, `train` copies that `build.json` to `outputs/train/builds/<revision>.json`
  (`data.snapshot_build`, written to a temporary name and renamed, and never rewritten once there). The
  check reads that copy (`data.trained_build_path`) and refuses the model, naming the file, when it's
  missing or its sha256 no longer matches. The dataset folder holds only the newest build, so the copy
  keeps a rebuild from retiring the model. A model moved to another computer needs its copy moved too.
- A model whose `dataset.revision` is null is refused, saying so: train it again.

The prompts, bounds, handovers, thresholds and rules are what the model was trained on and when it switches at runtime, so a model
trained on other ones would be told the wrong skill at the wrong time. `train --skills --base` refuses a
whole-demo ACT base, which has no input for the one-hot, and `train --base` a skills one.

## On the cloud GPU

LeRobot's policy server serves a checkpoint only when the inputs its config names equal the checkpoint's
`input_features`, by name, kind and shape. A skills checkpoint that reads the one-hot has one more input
than a whole-demo one, so:

- `server start --skills` names it in the server's config (`remote.server_config`, float32 of one value
  per skill, named `skill.<name>`) when the checkpoint's `config.json` lists it, and refuses one of another
  size before the job starts. ACT's validator and π0.5's (`remote.PI05_SERVER`) then both pass, which
  `tests/test_remote.py` checks with LeRobot's own validators.
- On the Mac, LeRobot works out the features it admits and sends from the robot alone, and its remote
  engine drops anything the server didn't name. `control.main.remote_context` adds the one-hot to those
  features while LeRobot builds the context, so the admission, the first engine and every reconnected one
  carry it, and each tick's frame sends it. A reconnected session is admitted with the server's own
  contract (`client.capabilities`) and told the prompt of the skill the run was on.

π0.5's skills model, whose checkpoint lists the one-hot, has run this way on the cloud GPU.

## How it's built

- `skills.py`: the definitions from `tasks.toml` and their checks, a demo's events (the detector's, plus
  lift and widest) and spans, a run attempt's labels (`label_run`), and what a skills model is tied to
  (`contract`, `gripper_hash`). `policy/skills.py`: the runtime `SkillSequencer`, which `policy/stack.py`
  adds for a skills run. `tasks.py` parses `[<task>.skills]`.
- `data.py`: `dataset_root` and `build_manifest_path` by kind, `build_inputs`, and the local build snapshots
  (`build_revision`, `snapshot_build`, `trained_build_path`).
- `build.py`: an episode per skill per kept demo, each frame's `task` and `observation.environment_state`,
  run cuts labelled by `run_skills`, and `swap`.
- `commands/policy.py`: the checks that tie a model to the task's skills; `remote.py` and
  `control/main.py`'s `remote_context`: the one-hot on the cloud GPU.
- `cli.py`: the `skills` command and `--skills` on `build`, `train`, `serve`, `run` and `server start`;
  `tasks.Task.policy_for` and `dataset_repo` give the `_skills` names.
- Tests (`tests/test_skills.py`, `tests/test_remote.py`): the definitions and their refusals; the events and
  spans on made-up traces, with a demo with no carry and a handover outside its spans refused; a skills build
  on fake sessions with every row, one-hot and prompt checked against the source; the layer switching once
  on the carry, clearing the queue, setting the prompt and one-hot and logging the `skill` event; the one-hot
  reaching the frame LeRobot builds for the policy; the layer as serve's policy on the fake arm; the Hub
  names and the server labels; the return; what makes a build stale; run labels for a person's attempt, a
  correction that starts in pick, the model's own frames and a Rewind; the model checks; both validators
  with the one-hot, the client's admission and the remote engine's frame.

## Open questions

- Does a skills model do as well as the `whole` one? Compare both policies on the same block and jar places,
  over more than 3 attempts a run.
- Is ACT's 100-step chunk (3.3 s) too long for a 4 s place? The switch clears the queue, so place starts
  fresh, but a shorter chunk may follow the jar better.
- How many retry-start pick demos does a retry need? No source measured it on an SO arm.
- Does return bring the arm home as reliably as Reset's scripted move?
- Could a block dropped during place switch back to pick from where the arm is, without a Rewind? That needs
  pick demos that start from such states.
