# Using run data beyond `build --with`

Status: ideas, none built. Written in November 2026 against LeRobot 0.7.0 and this repo's control center (`so101 serve`). The README covers what `build --with` does today; this note covers what else the same data could train, and what each would take.

## What a build uses today

`uv run so101 build --with corrections,human-attempts` is HG-DAgger: a human takes over when the policy goes wrong, and only what the human did is added to the demos. HG-DAgger never trains on the policy's own frames, so the steps before a takeover get weight 0.

| In the build | Left out |
| --- | --- |
| The demos, after `edits.toml` | Suspect frames: the policy's actions that led to an interruption |
| Corrections: Teleop after the policy drove in that attempt, whatever the outcome, minus suspect frames | Which failure each fix answered: `runs` exports failure-fix pairs, but the build keeps only the fix |
| Human attempts that succeeded | The policy's frames in failed attempts, and in successful ones unless `policy-successes` is named |
| | Interruptions the operator resumed without correcting |
| | Pause reasons (the page's chips), and the time-limit, fault and lost-link events |
| | `control.policy_action`, recorded only while the policy drives, so no frame pairs the policy's action with a human's at the same state |
| | `control.leader` outside Teleop, and every scripted frame (start, pause, handover, rewind, return), which the controller drove |

Leaving out the scripted frames is deliberate: no demonstrator chose those actions.

## The frames before a correction

The suspect frames hold bad actions in exactly the states where the policy fails. Most uses keep the states and drop the actions.

### Why weight 0, and never negative

ACT trains with an L1 loss and π0.5 and diffusion policies with an MSE loss. A negative weight turns "move toward this action" into "move away from it", which has no bound: the loss keeps falling as the predicted action moves further away, so training diverges instead of learning "not that".

0 is right for the suspect range a rewind marks, because the operator chose it. The range without a rewind is a guess, the last `blame_s` (2.5 s) of frames where the policy drove, not counting waiting frames, where it sent no action. If the actual mistake was its last 0.5 s, the first 2 s were probably fine. Sirius gives the 15 robot actions before each intervention a weight of 0 too. Here, shortening `blame_s` or fixing the range in `so101 review --runs` is cleaner than a small positive weight. Try one only if review shows the window often catching good motion.

[`learning-from-corrections.md`](learning-from-corrections.md) covers the bounded ways to learn from these frames (a quality tag, pairwise losses such as FlowPRO's RPRO, and sampling-time guidance), and the ways to weight corrections, in more detail.

### Ways to use them

1. **Find where the policy fails, then record demos there.** Group suspect ranges by pause reason, camera view and arm pose to find each failure mode, such as missing the grasp when the object is near the left edge. Then record demos of that case. Compare `control.policy_action` with `action`: if the `max_relative_target` cap clipped the policy's output, the problem is jerky outputs rather than a wrong plan.
2. **Start more corrections from those states.** Rewind followed by Teleop already makes the human demonstrate from the state where the policy failed. More of these make corrections cover the policy's own failure states, which is the core idea of DAgger.
3. **Train a failure detector.** Train a classifier on suspect frames against normal ones, from cameras and joints only. In `serve` it could pause before the operator has to, warn the operator, or flag attempts for review. LeRobot's reward classifier (`lerobot/rewards/classifier`) is a starting point.
4. **Train a value or progress model.** Suspect frames are examples of low value. Such a model drives RA-BC weights, RECAP's good and bad labels, and HIL-SERL's reward (see the training table below).
5. **Keep the actions with a quality tag in the prompt.** For a prompt-conditioned policy such as π0.5, train on suspect frames with "Quality: bad" in the task string and run with "Quality: good" (see the π0.5 section).
6. **Train on preference pairs.** A failure-fix pair gives a bad action chunk and a good one from the same pose, which DPO-style training needs. Only those pairs qualify, because only there do the states match. For flow and diffusion policies this is research-level.
7. **Simulate from the failure state** (see [Simulating from failure states](#simulating-from-failure-states)).

Relabeling every suspect frame with the right action, as classic DAgger does, doesn't work here. After the first few frames the arm is in states the human never reached, so nothing labels them.

Uses 1 and 3 give the most for the least work. Analysis says which demos to record next. A detector catches failures earlier, which gives cleaner corrections for every method below.

## Training methods

Every LeRobot policy trains on a `--with` build, because no `control.*` column starts with `observation.` and so none becomes an input. LeRobot 0.7.0 has ACT, Diffusion, VQ-BeT, SmolVLA, π0, π0-FAST, π0.5, GR00T, X-VLA, Wall-X and newer ones. The methods differ in how much of the run data they use. They are ordered from least to most work:

| Method | Uses | Work |
| --- | --- | --- |
| HG-DAgger | `build --with corrections,human-attempts` | Built |
| Filtered BC | `--with ...,policy-successes`: the policy's successes, minus suspect frames | Built. It can also reinforce the policy's own habits |
| Intervention weighting (IWR, Sirius) | Corrections weighted up, the frames before them at 0. `--keep-labels` already writes `control.label` | A custom weighter of about 50 lines, registered by patching LeRobot, whose sample-weighting hook (`lerobot/utils/sample_weighting.py`) offers only `rabc` and `uniform`. ACT also needs a patched `forward` that returns a loss per sample, and a job runs only the code in its image |
| RA-BC | A progress model's score per frame; frames that move backwards get 0 | In LeRobot (`lerobot/rewards/sarm`, `robometer`, `topreward`), after training or downloading a progress model |
| HIL-SERL | Outcomes train a success classifier, then RL on the real arm with a human intervening | In LeRobot (`lerobot/rl`), but with its own actor and learner loop, separate from `serve`. Suits one precise task |
| RECAP (π\*0.6) | Demos, corrections, and the policy's successes and failures, each step labelled good or bad by a learned value model | Not in LeRobot. The community versions are partial |

The weighting hook works with π0.5: LeRobot's π0.5 computes a loss per sample. LeRobot's ACT doesn't: the hook calls `policy(batch, reduction="none")`, and `ACTPolicy.forward` takes no `reduction` and returns one scalar.

## With π0.5

π0.5 takes a text prompt with every frame, and LeRobot stores the task per frame. That makes the prompt an input the build can tag data with.

- **A quality tag: a cheap version of RECAP.** Train on everything, with "Quality: good" on demos, corrections and the policy's successes, and "Quality: bad" on suspect frames and the policy's frames in failed attempts. At run time, prompt with "Quality: good". The model learns what both look like, without the divergence of a negative weight. π\*0.6 gives its advantage label the same way, as text, and always marks human corrections good. The difference is that π\*0.6 learns a value model to label each step, where this uses the run's labels. At worst the model ignores the tag and the result is close to HG-DAgger. It needs a build that keeps suspect and failed frames and writes a task string per segment.
- **Recovery prompts.** Label each correction with what the human did, such as "open the gripper and grasp the block again". The pause-reason chips are a start, and LeRobot's annotation pipeline (`lerobot/annotations/steerable_pipeline`) writes subtask labels with a VLM. The policy then learns recovery skills that a prompt can call, so an operator without a leader arm could fix some failures with a text command, as in the Yell At Your Robot paper. `serve` would need a "change prompt" command; LeRobot's `RolloutController.set_task` exists, but the control center doesn't expose it.
- **Short-term memory.** LeRobot's π0.5 has `use_visual_memory` and `use_proprioceptive_memory` (from the MEM paper): the last 6 observations, 1 s apart by default. A policy without memory that just missed a grasp often repeats the same move. The build cuts each correction into its own episode, so a correction's first frames have no history before them. Memory would need the build to keep a few seconds before each cut as context the policy sees but isn't trained to imitate.
- **RL or preference fine-tuning** of π0.5's action head is research-level. RLinf has run RL on π0.5 in simulation.

## Simulating from failure states

The sim twin on the `sim-twin` branch (PR #13) is a MuJoCo scene fitted to an earlier task's demos on a wooden table: camera poses, motor response, colours and softness. For the block-container task it needs the black mat, the block and the container modelled, and the top camera's pose fitted again. A run records the arm's joints and both cameras at every frame, so each interruption marks a real state where the policy failed. Starting the sim there turns one real failure into many sim attempts.

### Setting the sim to a recorded state

- **The arm:** `observation.state` at the first suspect frame, through the scene's `to_ctrl`, in degrees.
- **The object:** not recorded. It has to be found in that frame's images. The sim's cameras are fitted to the real ones and the table plane is known, so one pixel on the object in the top view, or `wrist_pixel_on_table` in the wrist view, gives its position on the table. `reset(options={"object_xy": ...})` already takes that position.
- **What it can't set:** the object in the gripper, tipped over, or in contact with the arm. `set_object` places it upright on the table. Failures with the object in hand need a grasp state the sim doesn't have yet.

### Uses

1. **A failure suite.** Keep every real failure state and run each new policy from all of them in the sim before it touches the arm. It answers whether a new policy still fails where the last one did.
2. **Reproduce and probe a failure.** Run the same policy from the state just before the suspect frames. If the sim reproduces the failure, move the object by 1 to 2 cm, or change the start pose, to see how narrow it is. If the sim doesn't reproduce it, the cause may be outside the policy: lighting, the camera, or motor behaviour the sim doesn't model.
3. **Generate recovery demos around each failure.** The sim knows where the object is, so a scripted grasp, or the human's fix from a failure-fix pair adapted to a moved object (the MimicGen idea), can produce many recoveries per real failure. Train on them mixed with real corrections, as a small share and labelled as sim, because of the visual gap below.
4. **RL from failure states.** Start sim episodes from the real failure states, where the policy is weakest, and reward the sim's own success check (the block settled in the container). This is heavier, and every gap between the sim and the arm becomes something the policy can exploit.
5. **Test the failure detector.** Check that it fires on failures the sim reproduces, and not on successful sim attempts.

### Limits

PR #13 lists these, and they apply here too:

- The wrist camera is the weakest part of the sim: the object appears 32 px from where it is drawn.
- Friction is uncertain: at 0.3, only 18 of 50 demo replays succeed, so sim grasps may rely on friction that is higher than real.
- Sim success rates haven't been checked against the real arm, and the sim looks more forgiving: ACT succeeded in 9 of 20 sim attempts but got nowhere close in 3 real ones.

Uses 1 and 2 only compare policies with each other, so they tolerate these gaps best. Uses 3 and 4 train on the sim and need the gaps measured first.

### What it needs

- PR #13 merged. Its code is git-ignored on other branches.
- A command that reads an interruption from a run (`control.analysis`), sets the sim to it, and runs a policy from there, for example `so101 sim eval --from-runs`.
- An object locator for real frames: a click on the object during `review --runs`, or a segmentation model.

## Order

1. Collect runs with `serve` and train HG-DAgger, which is built.
2. Group the suspect frames by failure mode and record demos of the common modes.
3. Add intervention weighting.
4. Build the sim failure suite once PR #13 merges, so each new policy is compared on real failure states before it runs on the arm.
5. Train a failure detector and use it in `serve`.
6. Add the π0.5 quality tag, if π0.5 is the policy.
7. Then recovery prompts, sim recovery demos, RL and RECAP, as the earlier steps show where the policy still fails.

## Sources

- HG-DAgger: Kelly et al., 2019, [arXiv:1810.02890](https://arxiv.org/abs/1810.02890)
- Intervention weighted regression (IWR): Mandlekar et al., 2020, [arXiv:2012.06733](https://arxiv.org/abs/2012.06733)
- Sirius: Liu et al., 2022, [arXiv:2211.08416](https://arxiv.org/abs/2211.08416)
- π\*0.6 and RECAP: Physical Intelligence, 2025, [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)
- SARM and RA-BC: [arXiv:2509.25358](https://arxiv.org/abs/2509.25358)
- HIL-SERL: [hil-serl.github.io](https://hil-serl.github.io/)
- Genie Centurion, rewind and refine: [arXiv:2505.18793](https://arxiv.org/abs/2505.18793)
- Yell At Your Robot: Shi et al., 2024, [arXiv:2403.12910](https://arxiv.org/abs/2403.12910)
- MimicGen: Mandlekar et al., 2023, [arXiv:2310.17596](https://arxiv.org/abs/2310.17596)
- MEM, short-horizon memory for π0.5: [arXiv:2603.03596](https://arxiv.org/abs/2603.03596)
- RLinf RECAP and RL examples: [rlinf.readthedocs.io](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)
- LeRobot rollout and DAgger docs: [huggingface.co/docs/lerobot/inference](https://huggingface.co/docs/lerobot/inference.md)
