# The simulated twin: how close it is

Status: built. `so101 sim` is described in the README under "Try a policy in simulation". This note holds what was measured when the sim for `block-container` was first built, on 2026-11-13 from the 100 demos recorded on 2026-11-12, and what limits it. Nothing here ran on the arm: the real numbers it quotes are from the demos and from `results/act-block-container-baseline.md`.

## What the fits reached

Each fit used 72 of the 96 demos that hold the block; `score` used the other 24.

| | Result |
| --- | --- |
| Top camera's pose and lens, joint offsets | The drawn arm's outline is 3.8 px from the real one. The block appears 4.4 px from where the jaws closed on it |
| Joint offsets | shoulder_lift -1.0°, elbow_flex -6.5°, wrist_flex -0.8°, wrist_roll +3.1° |
| Block's start, seen by the top camera | 4.9 mm (median) from where the jaws later closed on it |
| Jar | 68 mm across; it stood within 5 x 6 mm (1 SD) over the demos |
| Wrist camera | Its drawn outlines are 5.8 px from the real jaws, 7.0 px from the block, 9.5 px from the jar |
| Cameras' timing | The top camera's picture is 3 control ticks (100 ms) older than the joints saved with it, and it gives 25 new pictures a second. The wrist camera's delay doesn't show: 0 to 2 ticks fit equally |
| Motors, on demos they weren't fitted to | Joint error 0.2, 1.1, 0.4, 0.2, 0.2 and 0.1, shoulder_pan to gripper (degrees; the gripper in its own units) |
| Gripper | Holding the block, it reads the real gripper's 18.8 with its zero 0.2° from the arm model's end stop, where the real gripper rests. That needed stiffer contacts than the arm model's: with those the jaws sank 2 to 3 mm into the block |

## What the checks found

| Check | Result |
| --- | --- |
| `sim replay`: every demo's real commands, nothing correcting the arm | 87 of 96 lift the block, 84 put it in the jar |
| Jaw tips at the grasp in those replays, sim against real joints | 1.1 mm short and 1.6 mm high on average, within 3.1 mm (1 SD) |
| `sim score`, top camera | Pixels 9.3 of 255 from the real frame, against 7.2 between two real frames of the same pose. Block drawn 1 px from the real one |
| `sim score`, wrist camera | Pixels 21.2, against 13.7. Block 11 px from the real one |
| `sim check` | The policy's actions are 0.6° from the demo's on real images and 2.8° on sim images: 1.8° with only the top picture the sim's, 1.3° with only the wrist picture |
| `sim eval`, 25 attempts of the ACT policy, 100 action steps, 30 s each | 17 placed (68%; 48% to 83% at 95%). The same policy placed 24 of 25 on the arm |

## Why the policy does worse in the sim

The sim's physics isn't what holds the policy back: replayed demos grasp and drop the block, and the jaw tips arrive within millimetres of where the real ones did. The policy fails the way its real failures went, with the block against the fixed jaw instead of between the jaws, but far more often, because the sim's pictures move its aim.

`sim check` on demo frames up to just after the grasp, with the real wrist picture and the top picture changed one part at a time, gave these shifts in the policy's actions from what it predicts on the real picture:

| Top camera picture | All joints | shoulder_lift | elbow_flex |
| --- | --- | --- | --- |
| Real, with the sim's block pasted in | 0.7° | 1.0° | 1.0° |
| Real, with the sim's jar | 0.6° | 1.2° | 1.1° |
| Real, with the sim's background | 0.7° | 1.4° | 1.2° |
| Real, with the sim's arm | 1.3° | 2.3° | 2.1° |
| Real, blurred by 1 px | 1.0° | 1.9° | 1.7° |
| The sim's | 2.0° | 3.5° | 3.5° |

Two degrees at the shoulder moves the jaws about a centimetre, and the block is 32 mm wide.

- **The arm matters most.** The drawn arm has the right outline and shading, but none of the real one's cables, print lines or dark holes.
- **This policy reads fine detail.** Blurring a real picture by a single pixel moves it as much as the sim's block, jar and background together. A policy trained with image augmentation should care less, and `sim check` will show it.
- **Shading and timing aren't the gap.** Lighting the arm from above with shaded sides, and drawing the top picture 3 ticks old as the real camera's is, each changed the arm's row by under 0.1°.

So a sim result for this ACT policy is a lower bound, not a prediction. Compare models in the sim only after `sim check` shows they react to sim pictures alike.

## What would close the gap

- **Texture the arm from real frames.** The outline fit already says which pixels of each top camera frame are which link, so each link's surface could be painted from the demos' frames rather than drawn in one colour. The cables would still be missing.
- **Train with augmentation.** Blur, noise and colour jitter in training would make a policy less sensitive to exactly the detail the sim lacks, and should help on the arm too.
- **Measure what's guessed.** The block's and jar's weight and the mat's height above the surface the base stands on, in `[sim]` in `so101.toml`, and pushes for the mat's friction (`sim nudge`, `sim physics`).

## What was tried and dropped

- **Fitting the mat's height.** The top camera draws the block about 9% larger than its red pixels cover, and the wrist camera sees the jar about 15% larger than the top camera's fit of it. Both would follow from a mat a few centimetres lower than the 3 mm the build assumes. Fitted from the block's size, the mat came out 7 to 16 mm below the base's plane, but the arm's outline then fitted worse (5.1 to 6.0 px against 3.8) and the optimiser's result depended on where it started. `mat_height = "fit"` still does it. A ruler settles it.
- **A strong pull on the joint offsets.** The follower's calibrated ranges match the arm model's within 2° at the elbow and wrist, yet the outline fit wants -6.5° at the elbow. Holding it under 2° left the outline 5.1 px out. The offset may stand in for something else, such as a forearm that isn't quite the model's.
- **Cast shadows.** MuJoCo's leave what they fall on nearly black, and the real ones are soft and grey.
