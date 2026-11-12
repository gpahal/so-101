# Learning from corrections and rewinds

Status: research, none built beyond HG-DAgger (`build --with corrections,human-attempts`). Written in November 2026 against LeRobot 0.7.0, from the papers listed under [Sources](#sources). Most of the methods below are 2025 and 2026 preprints with no released code, each tested on a single robot, so no one has reproduced their numbers. None of them was tested on ACT.

This note covers training methods that use what `so101 serve` records when a human steps in on the policy. There are two groups:

- **The simpler group** trains on the human's corrections as extra demos and weights them differently from the original demos. It never uses the policy's bad actions as a training signal. HG-DAgger, which is built, is the simplest of these.
- **The second group** also uses the bad actions as a sign of what not to do: the frames a rewind undid, or the last `blame_s` before an intervention.

[`using-run-data.md`](using-run-data.md) covers other uses of the same data, such as finding failure modes, a failure detector and simulating from failure states.

## What a run gives you

The rules are in the `control/analysis.py` docstring. In short:

| Data | What it is | Used today |
| --- | --- | --- |
| Corrections | Teleop segments after the policy drove in that attempt | Yes, with `--with corrections`, at weight 1 |
| Human attempts | Teleop before the policy first drove, in attempts that succeeded | Yes, with `--with human-attempts`, at weight 1 |
| Policy successes | The policy's frames in attempts that succeeded, minus suspect frames | Yes, with `--with policy-successes`, at weight 1 |
| Suspect frames | The policy frames a rewind undid. Without a rewind, the last `blame_s` (2.5 s) of policy frames before a Pause, Rewind or Teleop | No: weight 0 |
| Failure-fix pairs | A rewind that undid policy frames, followed by Teleop from the pose it went back to | No. `so101 runs` exports them |
| Outcomes | succeeded, failed or ignored, from the operator's answer or `labels.toml` | Only to select attempts |

A failure-fix pair is the data most of the second group needs: a bad action chunk and a good one from the same state. Its `action` column is what the arm was sent, after the `max_relative_target` cap. `control.policy_action` holds the policy's raw output. Both are recorded only while the policy drives, so no frame shows what the policy would have done while the human drove.

## What LeRobot 0.7.0 provides

Paths below are under `.venv/lib/python3.13/site-packages/lerobot/`.

- **Sample weighting.** `lerobot-train` takes a `sample_weighting` config. Each step it computes `loss = (per_sample_loss * w).sum() / (w.sum() + eps)` (`scripts/lerobot_train.py`, lines 247 to 257). Only two weighters exist, `rabc` and `uniform`, and `make_sample_weighter` raises an error on any other type (`utils/sample_weighting.py`). A new weighter therefore means patching LeRobot or calling the training loop from this repo's own entry point.
- **ACT doesn't give a loss per sample.** The weighted path calls `policy(batch, reduction="none")`, but `ACTPolicy.forward(self, batch)` has no `reduction` argument and returns one scalar (`policies/act/modeling_act.py`, line 140). π0.5's `forward(batch, reduction="mean")` accepts `"none"` and returns the flow loss of each sample (`policies/pi05/modeling_pi05.py`). Any weighting with ACT needs a patched forward that returns the L1 loss per sample, and the KL term per sample or averaged.
- **No preference or pairwise losses.** LeRobot has no DPO, preference, advantage or RECAP code for ACT or π0.5.
- **Training jobs run LeRobot's image.** `so101 train` runs `lerobot-train --job.target=...`, which starts a Hugging Face job running `lerobot-train` in `huggingface/lerobot-gpu:latest` (`configs/default.py`, `jobs/hf.py`). Code that isn't in LeRobot, such as a weighter, a patched ACT or a new loss, has to be in a custom image passed with `--job.image`. Training with `--target local` avoids that, but only works for ACT.

## Weighting corrections as demos

These methods keep imitation learning as it is and change only how much each frame counts. Every weight is 0 or more, so none of them can diverge. All of them can be built with a weighter that reads `control.label` and `intervention` from a `--keep-labels` build.

### HG-DAgger (built)

The policy runs, a human takes over when it goes wrong, and only the human's frames are added to the dataset, each at weight 1 like a demo ([Kelly et al., 2019](https://arxiv.org/abs/1810.02890)). That is `build --with corrections,human-attempts`.

The weakness is volume. A correction is a few seconds, so 50 of them add a few minutes to hours of demos, and the policy sees them rarely in training. The methods below raise their share.

### Filtered BC (built)

Add the policy's own successful attempts, minus suspect frames: `--with policy-successes`. It adds data in the states the policy really reaches. It can also reinforce the policy's own habits, such as a slow or roundabout grasp, because nothing marks a success as good or merely lucky.

### Intervention weighted regression (IWR)

IWR weights intervention samples so that the intervention data and everything else each make up half of every batch: the intervention weight is α = |D_R| / |D_I|, the size of the non-intervention data over the size of the intervention data ([Mandlekar et al., 2020](https://arxiv.org/abs/2012.06733)). It does nothing special with the frames before an intervention. In simulation it beat HG-DAgger, 87.3% against 75.3% on Threading and 87.5% against 69.6% on Coffee. It had no real-robot results.

Here it would mean giving corrections the weight that makes their total equal the demos' total. With 100 demos of 30 s and 60 corrections of 3 s, that is a weight of about 17 per correction frame.

### Sirius

Sirius sorts every sample into four classes: demo, robot (the policy driving), intervention, and pre-intervention, the 15 robot actions before each intervention ([Liu et al., 2022](https://arxiv.org/abs/2211.08416)). Each class c gets the weight w = P\*(c) / P(c), where P(c) is the class's share of the data and P\*(c) its target share. It sets P\*(intervention) = 1/2 and **P\*(pre-intervention) = 0**, so the frames before an intervention are dropped, not given a small weight. It started from 30 to 80 demos and ran 3 rounds, with a BC-RNN-GMM policy, and its abstract reports +27% success on real hardware.

This repo already has every class: demos, policy successes, corrections and suspect frames (a range a rewind marks, otherwise `blame_s` of 2.5 s against Sirius's 15 steps). Sirius at its paper's settings is IWR plus training on the policy's frames outside the suspect range.

### RA-BC with a progress model

RA-BC weights each frame by how much a learned progress model says the task advanced ([SARM, arXiv 2509.25358](https://arxiv.org/abs/2509.25358)). The weight is 1 when progress rose by more than κ (0.01), between 0 and 1 for smaller gains, and 0 when progress fell (`rewards/sarm/rabc.py`, lines 270 to 293). SARM is trained only on trajectories without serious mistakes. With π0 and LoRA on medium T-shirt folding, from 200 hours of demos, it reached 10/12 against 1/12 or 7/12 for plain BC.

It is the one weighter LeRobot ships, so it works with π0.5 today. It needs a progress model trained or downloaded first (`rewards/sarm`, `robometer`, `topreward`). Its weights never go below 0. It would down-weight the slow or backward parts of corrections, as well as of demos.

### Advantage weights (AWR, IQL)

Offline RL methods weight each action by exp(β · advantage), where the advantage is how much better the action did than the average from that state. IQL clips the exponentiated advantage at 100, and uses β from 0.5 to 10 ([Kostrikov et al., 2021](https://arxiv.org/abs/2110.06169)). A bad action gets a small positive weight, never a negative one. Applied as a weight on ACT's L1 or π0.5's flow loss, it works with either policy. It needs a reward and a learned value function. π\*0.6 tried AWR on its T-shirt task and found it "fall far short" of advantage conditioning (below), with slower policies ([arXiv 2511.14759, §VI-C3](https://arxiv.org/abs/2511.14759)).

### A caution from PHIL

PHIL found that corrections made after a failure contain recovery motions, and that BC on them teaches the policy to make those motions "even when no recovery is needed" ([arXiv 2611.05667, §I](https://arxiv.org/abs/2611.05667)). Its HG-DAgger policies made 214 unnecessary recoveries against 21 when trained on corrections made before the failure. Weighting corrections up makes this effect stronger. Watch for it in `serve` after training on weighted corrections, and prefer stepping in before a mistake over waiting for it.

### Comparison

| Method | Weights | Needs | Policies | Built |
| --- | --- | --- | --- | --- |
| HG-DAgger | corrections and demos at 1 | nothing new | both | yes |
| Filtered BC | policy successes at 1 | outcomes | both | yes |
| IWR | corrections up to half of each batch | a weighter | π0.5; ACT after the patch | no |
| Sirius | IWR, suspect frames at 0 | a weighter | π0.5; ACT after the patch | no |
| RA-BC | 0 to 1 from progress | a progress model | π0.5; ACT after the patch | weighter in LeRobot |
| AWR / IQL | exp(β · advantage), clipped | rewards and a value function | both, after the patch | no |

## Learning from the bad actions

### Why the weight is 0 and never negative

A negative weight turns "move toward this action" into "move away from it". ACT's L1 loss and π0.5's flow-matching loss both measure distance, and distance has no upper bound, so the loss keeps falling as the prediction moves further from the bad action, toward anything at all. Training then makes the policy worse everywhere while the number improves. In LeRobot's loop a negative weight also shrinks or flips the `w.sum()` it divides by. CARF reports that a direct minus-loss term "may lead to unstable optimization" for this reason ([arXiv 2609.21982, §3](https://arxiv.org/abs/2609.21982)).

Every method below bounds the signal in one of four ways:

1. **A condition label.** Train the bad actions with a "bad" input instead of a negative weight, and run with "good" (the quality tag).
2. **A pairwise loss.** Ask only that the fix score better than the mistake from the same state, through a log-sigmoid that stops pushing once the fix clearly wins, and keep an imitation loss as an anchor (RPRO, PHIL).
3. **A margin.** Push away from bad actions only until their loss passes a reference (CARF).
4. **Sampling-time guidance.** Train only ordinary losses, and use the failure data to steer the policy's sampler (AFIL, PreferenceFlow).

### The flow loss as a likelihood

π0.5's action expert is trained with flow matching. Its loss on one chunk a in state s is

ℓθ(s, a) = E over t ~ U[0, 1] and ε ~ N(0, I) of ‖vθ(aₜ, t | s) − (a − ε)‖², with aₜ = (1 − t)ε + ta.

FlowPRO uses it as a stand-in for −log πθ(a | s), up to a constant: a lower loss means the policy is more likely to produce that chunk. The pairwise methods compare chunks through it. Each value is an estimate from one draw of t and ε, so it is noisy. ACT's L1 loss is the negative log-likelihood of a Laplace distribution up to a constant, so the same comparison works mechanically for ACT, but no paper tests it.

### The quality tag (RECAP, CFGRL)

π\*0.6 gives every training frame a text indicator, "Advantage: positive" or "Advantage: negative", before the actions, and runs with "positive" ([arXiv 2511.14759, §IV-B, §V](https://arxiv.org/abs/2511.14759)).

- **The labels.** A learned value model labels a frame positive when its advantage is above the task's 30th percentile. Human corrections and demos are always positive.
- **Dropping the indicator.** It is left out at random in training, so the model also learns without it, and classifier-free guidance on the indicator is possible at run time.
- **The loss.** Both terms are ordinary flow losses: −log π(a | o, ℓ) − α · log π(a | I, o, ℓ). The bad frames train the "negative" mode instead of being pushed away.
- **Its scale.** It used 300 to 600 autonomous trajectories per iteration, and reached over 90% success on every task but diverse laundry, with throughput more than doubled on some.

CFGRL is the same idea with an RL grounding: a flow policy conditioned on o = 1 for actions with advantage at least 0, o = 0 for the rest, and o = none for about 10% of samples. At run time it blends the "none" and "good" velocities with a weight w ([arXiv 2505.23458](https://arxiv.org/abs/2505.23458)). It beat AWR on OGBench and ExORL, in simulation.

**For this repo** it fits π0.5, whose task string LeRobot stores per frame:

- **Positive:** demos, corrections and policy successes.
- **Negative:** suspect frames and the policy's frames in failed attempts.

Only `build` changes. Training stays LeRobot's own, the loss stays bounded, and suspect frames without a fix after them still count. In FlowPRO's rollback experiments, a π\*0.6-style baseline came within 3 to 7 points of RPRO. Two things are untested:

- labels from rewinds and `blame_s` instead of a value model;
- this repo's scale, about 100 demos with tens to hundreds of interventions.

ACT has no text input. A binary input feature might play the tag's part, but no source tests that.

### FlowPRO and RPRO

FlowPRO collects data the way `serve` does, and trains π0 and π0.5 on it ([arXiv 2606.05468](https://arxiv.org/abs/2606.05468)).

**The data.**

1. **Train on demos.** It fine-tunes on 298 to 773 demos per task.
2. **Collect pairs.** It runs the policy with an operator watching. On an error, the robot rolls back Δ steps to the state before the mistake. The operator chooses Δ each time. A headset shows the camera image from that moment, so the operator can put the scene back as it was, then teleoperate the fix. The executed bad segment is the losing trajectory τˡ, and the fix is the winning trajectory τʷ.
3. **Repeat in rounds.** It retrains, redeploys, and collects pairs from the new policy's mistakes, for 3 rounds of 30 to 89 pairs each. The data always comes from the current policy, which is the DAgger idea.

**Pairs at every state.** One trajectory pair gives one comparison, and the paper says that "weakens the per-state learning signal". So it builds a pair at each state M on the bad segment:

- **The negative** is the next H actions the robot executed from M.
- **The positive** is made up, because no human acted from M. It finds the nearest point on the fix, weighting position, rotation and gripper 1, 0.5 and 0.2, and draws a cubic Bézier curve from M that joins the fix and then follows it. It interpolates rotation with Slerp and the gripper linearly. The positive says "from here, curve back onto the human's path".

No ablation tests the curves alone.

**The loss.** The implicit reward of a chunk measures how much more likely the trained policy finds it than a frozen reference copy:

r(s, a) = (β/2) · (ℓ_ref(s, a) − ℓθ(s, a)).

The objective has three terms:

1. **The contrastive term,** −log σ(r(s, aʷ) − r(s, aˡ)), is DPO: rank the fix above the mistake.
2. **The proximal term,** −Σ over both chunks of ½ · (log σ(r) + log σ(−r)), is smallest at r = 0 and grows equally in both directions. It holds each chunk's loss near the reference's, so the policy can't move either chunk far. The paper compares it to RLHF's KL penalty, but it acts on each chunk, not on whole distributions.
3. **The SFT term** is the ordinary flow loss on the winning chunks.

The total is ⅓ · L_PRO + 1 · L_SFT, with β = 3.5. When the two chunks are the same, the contrastive gradient cancels and only the bounded proximal gradient is left (Eq. 7 and 8).

**Why it works.** DPO constrains only the difference between the two chunks. The paper says the optimizer "can drive both π_θ(a^w|s) and π_θ(a^l|s) down together". In its runs, plain DPO's rewards fell to about −100 while the margin stayed positive, and the policy produced "implausible control actions" far from both chunks. Adding SFT anchors only the winning chunk, so the losing one still ran away. The proximal term anchors both. The ablation on the Pack task, one round, success in distribution and out of distribution:

| Variant | Success |
| --- | --- |
| DPO alone | 13% / 5% |
| DPO + SFT | 87% / 70% |
| SFT only (HG-DAgger) | 90% / 75% |
| Contrastive + proximal, no SFT | 88% / 81% |
| RPRO, all three terms | 94% / 91% |

The proximal term is what raised out-of-distribution success from 70 to 75% up to 91%.

**Setup and results.** It ran on a bimanual Dobot XTrainer with three cameras, with H = 50 steps (1 s at 50 Hz). Each round was 25k steps at learning rate 1e-5 and batch 20, on 4 H20 GPUs. Each batch was 70 to 80% preference data and 15 to 20% original demos. It reached 92 to 99% success on four tasks, ahead of DAgger by 8 to 15 points. No code is released. The paper isn't consistent about the reference model: one place says the previous round's policy, another the original SFT policy.

**For this repo:**

- **A training step built for pairs.** Each sample carries two chunks, each needs a forward pass on the policy and on a frozen π0.5 copy, and the frozen copy roughly doubles the weights in memory.
- **The curves are optional.** Failure-fix pairs already hold the negative at the rewound state, and the curves would be new code, simpler in joint space than in FlowPRO's Cartesian space.
- **The object isn't rewound.** A rewind puts the arm back, not the block. If the policy pushed the block before the rewind, the fix starts from a different scene than the mistake did. See [Recording pairs worth training on](#recording-pairs-worth-training-on).

### PHIL

PHIL is a simpler pairwise loss with no reference model, tested on a diffusion policy (BESO) and on X-VLA ([arXiv 2611.05667](https://arxiv.org/abs/2611.05667)).

**The data.** While the human drives, the policy keeps predicting every B steps in "shadow mode": its chunks are shown but not executed. The negative is the shadow chunk at a query state, and the positive is the next H actions the human executed from that same state. Its experiments used H = 30, B = 10 at 20 Hz, and five correction episodes per condition.

**The loss.**

L = L_base + λ · (−log σ(β · (ℓθ(A⁻) − ℓθ(A⁺)))),

where ℓθ is the policy's own imitation loss averaged over the chunk, and L_base is that loss on the original demos. The paper gives no values for β and λ.

**Why it works, per the authors.** Corrections made after a failure teach recovery motions the policy then makes when nothing has gone wrong. PHIL's operators step in before the failure, and the pairwise loss teaches "this, not that" at the moment of decision. On the same proactive data, PHIL beat HG-DAgger on Towel (68% against 40%) and Carrot (52% against 28%). It tied on Banana and was close on Lemon (40% against 36%). With BC alone, proactive data gave no consistent gain. The mean went from 44.0% to 70.3% for the diffusion policy and from 57.8% to 76.9% for X-VLA. Mixing reactive and proactive data dropped PHIL to 0 to 20%.

**The risk.** Only the demo loss anchors the policy, and nothing bounds the loss on the rejected chunk. FlowPRO's ablation suggests that can drift, as DPO + SFT did. The paper doesn't measure it. Log ℓθ(A⁻) during training and stop if it keeps rising.

**For this repo.** `serve` records no shadow chunks. The policy could be run offline on the recorded observations at the rewound state. ACT is deterministic given an observation, and π0.5 would be sampled. PHIL needs no frozen copy, so it costs about one extra forward pass per pair. It works with ACT mechanically, but untested.

### CARF

CARF trains on natural failures, without pairs ([arXiv 2609.21982](https://arxiv.org/abs/2609.21982)). A progress scorer trained only on successes marks the failure segments. The loss adds λ₂ · ReLU(stopgrad(L_suc on failure states) − L_fail), which pushes the policy's loss on failure actions up only until it passes the success loss on the same states, then stops. It used λ₁ = 0.05 and λ₂ = 0.01. With 50 to 200 successes and 26 to 110 failures per task, it reached 0.85 against 0.50 on mug hanging and 0.70 against 0.55 on a cooking sequence. No code is linked. It fits suspect frames that have no fix after them.

### AFIL

AFIL trains a failure head and steers π0.5's sampler away from it ([arXiv 2605.08434](https://arxiv.org/abs/2605.08434)).

**Training.** Two action heads share π0.5's VLM backbone. One is trained only on successful or corrected trajectories, the other only on failures, each with the ordinary flow loss. Nothing is pushed away in training, so nothing can diverge. The second head adds 12.76% to π0.5's parameters.

**Sampling.** At each denoising step both heads predict a velocity, and the step moves away from the failure head's:

v = v_succ − λ̂ · (v_fail − v_succ), with λ̂ = α · (1 − cos(v_succ, v_fail)) and α = 1.

This is classifier-free guidance with "failure" as the negative prompt. The paper's Eq. 6 and Eq. 7 differ by a factor, λ̂ = λ / (1 + λ), that its "⇔" hides. Check it before implementing.

**Why λ̂ adapts.** When the two heads agree, success and failure are "locally ambiguous", as when both reach for the object at the start of an attempt, and strong guidance there "may suppress valid actions". λ̂ is then near 0. It grows, up to 2α, where the heads disagree.

**Results.** In simulation, from success data only, to a fixed λ of 0.05, to the adaptive λ̂:

- Stack: 71.3% → 80.0% → 84.7%;
- Lift peg: 70.7% → 86.7% → 90.7%.

One shared head instead of two scored 72.0% on Stack and 75.3% on Lift. A fixed λ above 0.05 did worse. On a real ViperX arm, 50 successful and 50 failed trajectories took success from 0.50 to 0.92 over 50 trials. The failures came from rollouts and the corrections from a motion planner. Training took about 8 hours on 4 A100s. No code is linked.

**For this repo.** It needs no pairs: suspect frames and the policy's frames in failed attempts can train the failure head, including interventions with no fix after them. It needs a second head in LeRobot's π0.5, and a guided sampler in the remote policy server, which means a custom server image (`remote.SERVER_IMAGE`). It doesn't apply to ACT, which predicts its chunk in one pass with no sampler to steer.

### PreferenceFlow

PreferenceFlow trains a small scorer and uses it to steer a frozen π0.5 ([arXiv 2609.36872](https://arxiv.org/abs/2609.36872)).

**The scorer.**

- **Its pairs.** For each 10-step window inside a human takeover, it compares the human's chunk with a chunk freshly sampled from the frozen policy at the same state, and labels the human's chunk better. Its negatives are policy samples, not executed bad chunks.
- **Its model.** The scorer F reads frozen π0.5 features through cross-attention and scores each action step.
- **Its loss.** A Bradley-Terry (log-sigmoid) loss, plus two regularisers:
  - a cap on ‖∇ₐF‖, so the scorer's slope with respect to actions stays small;
  - a zero-gradient loss on noisy demo actions, so the scorer has no opinion around the routine moves the demos cover.

**Sampling.** At each Euler step it estimates the clean action â₁ = aₜ + (1 − t) · v, then steps aₜ₊δ = aₜ + δ · (v + λ_guide · ∇F(s, â₁)). λ_guide = 0 gives back the base policy.

**Why it works.** The ablation, mean success over four insertion tasks on a Franka:

| Variant | Success |
| --- | --- |
| Frozen π0.5 | 69.0% |
| BC on the interventions | 84.5% |
| Full PreferenceFlow | 90.5% |
| No gradient cap | 15.0% (81.0% after lowering λ_guide) |
| No zero-gradient loss | 5.0% (74.5% after lowering λ_guide) |
| Ranking loss only | 69.0% after lowering λ_guide |
| Labels shuffled at random | 63.5% |

A raw learned scorer has steep slopes that mean nothing, and following them wrecks the actions. The regularisers keep the guidance small where the policy is already right. Success rose with intervention episodes per task: 69.0% with none, 78.5% with 10, 87% with 30 and 90.5% with 50. It used 50 demos per task, interventions of 2 to 5 s per episode, and one RTX 5090. No code was found.

**For this repo.** It never retrains π0.5. It needs policy samples at the intervention states, which can be made offline from the recorded observations, and a guided sampler in the remote policy server. It doesn't apply to ACT.

### Methods that don't fit this data

- **They need online rollouts with rewards:** RLIF, which gives −1 at each intervention ([arXiv 2311.12996](https://arxiv.org/abs/2311.12996)), HIL-SERL ([arXiv 2410.21845](https://arxiv.org/abs/2410.21845)), Diffusion Policy Policy Optimization ([arXiv 2409.00588](https://arxiv.org/abs/2409.00588)), FDPP ([arXiv 2501.08259](https://arxiv.org/abs/2501.08259)), Q-chunking ([arXiv 2507.07969](https://arxiv.org/abs/2507.07969)) and DSRL ([arXiv 2506.15799](https://arxiv.org/abs/2506.15799)). LeRobot has HIL-SERL, but with its own actor and learner loop and a Gaussian policy, separate from `serve`.
- **They assume a policy that outputs action tokens with likelihoods:** GRAPE ([arXiv 2411.19309](https://arxiv.org/abs/2411.19309)) and APO ([arXiv 2506.07127](https://arxiv.org/abs/2506.07127)). APO labels the last 10 actions before an intervention "undesirable", which is close to suspect frames, but it was tested only on OpenVLA and π0-FAST.
- **They replace the policy class:** IBC, which trains an energy model against random counter-examples ([arXiv 2109.00137](https://arxiv.org/abs/2109.00137)), and its follow-ups that use correction negatives, CLIC ([arXiv 2502.07645](https://arxiv.org/abs/2502.07645)) and Set-Supervised Diffusion Policy ([arXiv 2606.01865](https://arxiv.org/abs/2606.01865)).
- **Two papers are called DPPO.** Diffusion Policy Policy Optimization is above. The other is Direct Preference-based Policy Optimization ([arXiv 2301.12842](https://arxiv.org/abs/2301.12842)), an offline preference RL method with no robot or diffusion results.

### Comparison

| Method | Data needed | What keeps it stable | Policies | Real-robot scale | Work here |
| --- | --- | --- | --- | --- | --- |
| Quality tag | good and bad labels per frame | ordinary loss, bad frames as a condition | π0.5 | hundreds of trajectories per iteration | `build` only |
| RPRO | failure-fix pairs, frozen copy | proximal and SFT terms | π0.5; ACT untested | 30 to 89 pairs per round | custom training step |
| PHIL | same-state shadow pairs | demo loss only | diffusion, X-VLA; ACT untested | 5 correction episodes | custom training step, offline shadow chunks |
| CARF | failure segments, progress scorer | ReLU margin | backbone not checked | 26 to 110 failures | custom loss and scorer |
| AFIL | unpaired failures | ordinary losses, guidance only at sampling | π0.5 | 50 failures | second head, custom server |
| PreferenceFlow | human chunks against policy samples | gradient cap and zero-gradient loss | π0.5 (frozen) | 50 episodes per task | scorer, custom server |

## Recording pairs worth training on

The pair methods assume the fix starts from the state the mistake started from. In `serve`:

- **Rewind before taking over.** Pause, then Rewind (`b`) to just before the mistake, then Teleop (`t`). The rewind range becomes the suspect frames exactly, with no `blame_s` guess, and the analysis records a failure-fix pair. Each `b` goes back 2.5 s (`rewind_s`).
- **Put the object back.** A rewind retraces the arm's path, and the gripper opens if it closed in that time, but the object stays where the mistake left it. Before taking over, put it back where it was before the mistake, as FlowPRO's operators do from the headset image.
- **Step in early.** PHIL's results favour corrections made just before a failure over recoveries after it.
- **Label every attempt.** Outcomes decide filtered BC, the quality tag's labels and which human attempts count. Pause reasons can group failures for analysis.

## What to build, in order

1. **Weighting for both policies.** Patch ACT's forward to return a loss per sample, add a weighter that reads `control.label` and `intervention`, and run it in a custom job image. Start with Sirius's settings: corrections at half of each batch, suspect frames at 0. Compare with HG-DAgger in `serve`.
2. **The quality tag, if the policy is π0.5.** It needs only a `build` option that writes "Advantage: positive" or "Advantage: negative" into each segment's task string, plus a runtime prompt with "positive".
3. **RPRO as a fine-tuning stage on π0.5,** from the HG-DAgger checkpoint, on failure-fix pairs, if steps 1 and 2 leave failures the pairs cover. Use PHIL's loss instead to avoid the frozen copy, with the rejected-chunk loss logged.
4. **AFIL or PreferenceFlow,** if the policy still repeats known failures. They need the remote server to run a custom sampler.

Each step needs a fair comparison: the same attempts in `serve` from the same start positions, against the previous policy.

## Open questions

- **ACT.** No source tests a pairwise loss on ACT's L1 with its CVAE, or a binary input standing in for the quality tag.
- **Labels and scale.** No source tests the quality tag with heuristic labels (rewind ranges and `blame_s`) instead of a value model, or at about 100 demos with tens to hundreds of interventions.
- **Imperfect pairs.** No source says how much an object moved before the rewind hurts a failure-fix pair.
- **Not read in full:** FlowPRO's choice of Δ, its baselines' configurations and its code release; Sirius's per-method real-robot numbers, which are only in a figure; and the AWR, AWAC and CQL papers.
- **Cost here.** Training time and memory were not measured on this repo's jobs. RPRO roughly doubles the weights in memory, and AFIL doubles the action-head passes when sampling.

## Sources

- HG-DAgger: Kelly et al., 2019, [arXiv:1810.02890](https://arxiv.org/abs/1810.02890)
- IWR: Mandlekar et al., 2020, [arXiv:2012.06733](https://arxiv.org/abs/2012.06733)
- Sirius: Liu et al., 2022, [arXiv:2211.08416](https://arxiv.org/abs/2211.08416)
- SARM and RA-BC: [arXiv:2509.25358](https://arxiv.org/abs/2509.25358)
- IQL: Kostrikov et al., 2021, [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
- π\*0.6 and RECAP: Physical Intelligence, 2025, [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)
- CFGRL: [arXiv:2505.23458](https://arxiv.org/abs/2505.23458)
- FlowPRO and RPRO: [arXiv:2606.05468](https://arxiv.org/abs/2606.05468)
- PHIL: [arXiv:2611.05667](https://arxiv.org/abs/2611.05667)
- CARF: [arXiv:2609.21982](https://arxiv.org/abs/2609.21982)
- AFIL: [arXiv:2605.08434](https://arxiv.org/abs/2605.08434)
- PreferenceFlow: [arXiv:2609.36872](https://arxiv.org/abs/2609.36872)
- Diffusion-DPO: Wallace et al., 2023, [arXiv:2311.12908](https://arxiv.org/abs/2311.12908)
- LeRobot 0.7.0: `scripts/lerobot_train.py`, `utils/sample_weighting.py`, `rewards/sarm/rabc.py`, `policies/act/modeling_act.py`, `policies/pi05/modeling_pi05.py`, `configs/default.py`, `jobs/hf.py`
