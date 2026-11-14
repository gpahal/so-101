# The simulated twin: how close it is

Status: built. `so101 sim` is described in the README under "Try a policy in simulation". This note holds what was measured when the sim for `block-container` was built, on 2026-11-13 and 14 from the 100 demos recorded on 2026-11-12, what a ruler and nine photos of the setup added on 2026-11-14, what limits it, and what to measure at the arm to close the gap. Nothing here ran on the arm: the real numbers it quotes are from the demos and from `results/act-block-container-baseline.md`.

## What the fits reached

Each fit used 72 of the 96 demos that hold the block; `score` and `check` used the other 24. These are from the build of 2026-11-14, with the jar as measured.

| | Result |
| --- | --- |
| Top camera's pose and lens, joint offsets | The drawn arm's outline is 3.8 px from the real one. The block appears 3.9 px from where the jaws closed on it |
| Joint offsets | shoulder_lift -0.4°, elbow_flex -6.1°, wrist_flex -2.6°, wrist_roll +3.9° |
| Block's start, seen by the top camera | 5.0 mm (median) from where the jaws later closed on it |
| Jar | Measured: 63 mm across, 70 mm tall, taped over its lower 52 mm. It stood within 5 x 6 mm (1 SD) over the demos |
| Wrist camera | Its drawn outlines are 6.6 px from the real jaws, 6.5 px from the block, 9.3 px from the jar |
| Cameras' timing | The top camera's picture is 3 control ticks (100 ms) older than the joints saved with it. The wrist camera's delay barely shows: 0 to 2 ticks fit alike, and 0 was taken. Both give 25 new pictures a second |
| Motors, on demos they weren't fitted to | Joint error 0.2, 1.1, 0.4, 0.2, 0.2 and 0.1, shoulder_pan to gripper (degrees; the gripper in its own units) |
| Gripper | Holding the block, it reads the real gripper's 18.8 with its zero 0.2° from the arm model's end stop, where the real gripper rests. That needed stiffer contacts than the arm model's: with those the jaws sank 2 to 3 mm into the block |

## What the checks found

"Rendered" draws everything from the arm model and fitted colours. "Real pixels" is what the sim does when the demos are on the computer: the arm in the top camera, the wrist camera's view, and each attempt's block and jar come from the demos' frames (README, "Real pixels"). The first two columns are from 2026-11-13, when the jar's height was a guess (63 mm) and the jaws' friction was the arm model's 1.0.

| Check | Rendered, 13 Nov | Real pixels, 13 Nov | Real pixels, 14 Nov |
| --- | --- | --- | --- |
| `sim score`, top camera, pixels from the real frame (two real frames of the same pose: 7.2) | 9.3 | 5.4 | 5.3 |
| `sim score`, wrist camera (two real frames: 13.7) | 21.2 | 11.1 | 10.0 |
| `sim score`, wrist camera's sharpness against a real frame's | | 3.7 | 2.3 |
| `sim check`: the policy's actions from the demo's, on sim images (on real images: 0.6°) | 3.1° | 2.2° | 2.0° |
| ... with only the top picture the sim's | 2.2° | 1.3° | 1.3° |
| ... with only the wrist picture the sim's | 1.5° | 1.4° | 1.2° |
| `sim replay`: of 96 demos' real commands, the ones that put the block in the jar | | 85 | 88 |
| `sim eval`, ACT policy, 100 action steps, 30 s an attempt (on the arm: 24 of 25) | 17 of 25 (68%) | 19 of 25 (76%; 57% to 89% at 95%) | 15 of 25 (60%; 41% to 77% at 95%) |

Every eval starts its attempts from the same 25 layouts in the same order, so the runs can be compared attempt by attempt:

- **The same layouts fail.** Both real-pixels runs placed 2 of their last 7. On 14 November, 5 of the 6 attempts whose block lay more than 24° to the arm's right failed, and 14 of the other 19 placed. Of the demos, 16 of 96 have their block that far right, the edge of where they put it.
- **The two real-pixels runs don't differ by more than chance.** 19 and 15 of 25 are 76% and 60% with intervals of 57% to 89% and 41% to 77%, and an earlier run of 15 attempts, a build before, placed 12. Which of the day's changes moved which attempt wasn't separated: 25 attempts can't show a change of 4.
- **Half the failures seen were out of time, not out of reach.** Videos are kept for the first 10 attempts. Of the 4 that failed on 14 November, 2 never closed on the block, and 2 hovered over it, grasped late and were holding it over the jar when the 30 s ran out.

Physics, which doesn't depend on how things are drawn:

| Check | Result |
| --- | --- |
| `sim replay`: every demo's real commands, nothing correcting the arm | 90 of 96 lift the block, 88 put it in the jar |
| Jaw tips at the release, sim against real joints, in 32 of those replays | 1.7 mm low and 1.1 mm across (medians), with every joint within 0.3° |

## What the measurements and photos of 14 November settled

A ruler gave the block, the jar and the mat, and nine photos showed the setup from around the table. The follower wasn't connected to the computer with the sim, so no joint readings go with the photos.

- **The jar is 70 mm tall, not 63.** The top camera's fit had its diameter (64 mm against 62 to 64 measured) and its tinted band (50 mm against 52), which checks that camera's scale to about 3%. The clear neck above the band was a guess, 13 mm where it's 18. With the measured jar and nothing else changed, the wrist camera's drawn jar fitted 1 px closer, its picture was 9.4 from the real frame instead of 11.1, and the policy's actions moved 1.9° instead of 2.2°. The build after it, which refitted the top camera with the mat against the base's underside, reads 10.0 and 2.0°: the difference is the size of what a refit changes.
- **The block is the DUPLO brick assumed.** The ruler read 32 x 32 x 19 mm against 31.8 and 19.2.
- **The base stands on the table the mat lies on.** It's clamped there by its plate, with nothing under it. So the mat's top is the mat's thickness above the base's underside, which the arm model has 2.4 mm below its origin; the build now puts the mat there (`arm.base_underside`). The mat is 3 mm thick, as assumed, and 30 cm from front to back.
- **The elbow's offset is real.** The fit's -6° at the elbow looked wrong, because the follower's calibrated range is as wide as the arm model's. But the model's range isn't where its parts meet: its forearm meets its upper arm at 91°, 6° before its limit of 96.8°. In the photos the follower rests with its forearm lying on its upper arm, where it reads 96.3°. So the real elbow reads about 5° more than the model's angle, as fitted.
- **The demos can't say how the offsets split between the joints.** Holding the elbow's offset and refitting gives the two cameras the same fit:

  | Elbow offset | shoulder_lift | wrist_flex | wrist_roll | Top camera's outline, block | Wrist camera's jaws, block, jar |
  | --- | --- | --- | --- | --- | --- |
  | -6.4° (free) | -0.9° | -1.3° | +4.8° | 3.8, 3.8 px | 6.4, 6.6, 9.1 px |
  | -3° (held) | -2.9° | -4.2° | +2.0° | 3.8, 3.8 px | 6.4, 6.8, 9.3 px |
  | 0° (held) | -4.7° | -6.5° | +1.5° | 4.0, 3.7 px | 6.5, 7.0, 8.8 px |

  Their sum, how far the gripper is pitched from its readings, stays near -10°. The splits draw the same pictures, so the choice matters little to a policy; the photos favour the first.
- **The top camera's scale is right to about 1%.** Its fit lays the mat out 300 mm from front to back where it's clear of the arm (301, 300 and 300 mm at three places), against 30 cm by ruler. Nothing in the fit knew the mat's size.
- **Nor can the demos place the mat.** With the mat's top held 15, 30 and 43 mm lower, the top camera's outline fits within 0.25 px (3.8, 3.9, 4.1 and 4.0 px) and the offsets move by up to 16° to keep each grasp on the mat. Only the photos say where it is.
- **The wrist camera is where the arm model mounts one.** The fit puts its lens 9 mm from the model's own wrist camera and pitches it 5° from it, and the photos show the follower's camera on that mount.
- **The jaws' friction was too high.** See the next section.

## What replaying the demos showed about the jaws

With the measured jar, 82 of 96 replayed demos put the block in the jar, 3 fewer than with the shorter one. The replays that lifted the block and still missed went wrong in two ways, both at the release, and both because the jaws gripped harder than printed plastic does:

- **The block wedged.** Released a centimetre above the rim, it caught between the jar's wall and the fixed jaw, and the jaws picked it up again as they closed.
- **The block never left.** In a few demos the real gripper opened by about 1 unit of its reading (1.5 mm at the tips) and the block fell. In the sim it stayed, held by what little the jaws still pressed.

The arm model gives its jaws a friction of 1.0. Printed plastic against a smooth brick is 0.3 to 0.5. Replaying all 96 demos at each value, on the build with the measured jar and on the one after it:

| Jaws' friction | 1.0 | 0.8 | 0.6 | 0.5 | 0.45 | 0.4 | 0.35 | 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Blocks in the jar, of 96 | 82, 79 | 82 | 85, 85 | 84, 84 | 90, 87 | 89, 83 | 89, 88 | 87, 86 |

The count moves by up to 6 between builds, so it only says that 0.3 to 0.45 is better than the model's value; the build uses 0.35 (`jaw_friction`). The same six demos fail to lift the block at nearly every value (14, 34, 45, 61, 75 and 92 on the last build): their blocks start a few millimetres from where the real ones lay, and the jaws close beside them.

## Why the policy still does worse in the sim

The physics isn't what holds the policy back: replayed demos grasp and drop the block, and the jaw tips arrive within millimetres of where the real ones did. The policy fails the way its real failures went, with the block against the fixed jaw instead of between the jaws, but more often, because the sim's pictures move its aim.

`sim check` on demo frames up to just after the grasp, with one picture real and the other changed one part at a time, gave these shifts in the policy's actions from what it predicts on the real pictures, on the build of 13 November. Two degrees at the shoulder moves the jaws about a centimetre, and the block is 32 mm wide.

| Top camera picture | All joints, rendered | All joints, real pixels | shoulder_lift, real pixels |
| --- | --- | --- | --- |
| Real, with the sim's jar | 0.6° | 0.1° | 0.2° |
| Real, with the sim's block | 0.7° | 0.4° | 0.8° |
| Real, with the sim's background | 0.7° | 0.4° | 0.8° |
| Real, with the sim's arm | 1.3° | 0.7° | 1.2° |
| The sim's | 2.0° | 1.0° | 1.7° |
| Real, blurred by 1 px | 1.0° | 1.0° | 1.9° |

| Wrist camera picture | All joints, rendered | All joints, real pixels | shoulder_lift, real pixels |
| --- | --- | --- | --- |
| Real, with the sim's mat and table | 0.2° | 0.4° | 0.7° |
| Real, with the sim's jaws | 0.6° | 0.3° | 0.6° |
| Real, with the sim's wall and floor | 0.4° | 0.4° | 0.7° |
| Real, with the sim's jar | 0.5° | 0.4° | 1.0° |
| Real, with the sim's block | 1.0° | 1.1° | 1.9° |
| The sim's | 1.6° | 1.5° | 2.8° |

- **This policy reads fine detail.** Blurring a real picture by a single pixel moves it 1.0°, as much as the whole of the sim's top picture now does. A policy trained with image augmentation should care less, and `sim check` will show it.
- **How things look is mostly solved for the top camera.** Its picture is closer to the real frame than a second real frame of the same pose is.
- **Where the block sits in the wrist picture is the largest piece left.** Neither real pixels nor the measured jar changed it: the block is drawn 19 px (median) from where the real camera saw it, on average 10 px to its right. What causes it isn't known: a lens centre and freer joint offsets in the wrist camera's fit didn't remove it (below). The wrist camera's lens is the least pinned thing left. Its focal length came out 338, 390 and 360 px on three builds, the outlines fitting within 1 px of each other, and a lens that's wrong puts things in the wrong place away from the picture's middle. A checkerboard pins it (below).
- **The wrist picture has seams.** Where a patch from a real frame meets a rendered one there's an edge no real picture has: its sharpness reads 2.3 times a real frame's (3.7 before the jar was measured). Blending the seams didn't change the policy's actions.

So a sim result for this ACT policy is still a lower bound, not a prediction. Compare models in the sim only after `sim check` shows they react to sim pictures alike.

## What to measure at the arm

In order of what they would fix. Each `[sim]` setting is in `so101.toml`. Measured on 2026-11-14 and no longer needed: the block's, the jar's and the mat's sizes, and photos of the whole setup.

1. **A checkerboard on the mat.** A printed checkerboard (squares of 25 mm, measured after printing) flat on the mat, and `uv run so101 check` run once to save a frame from each camera; then the same at two more places, one of them with the board propped at a slope so the wrist camera sees it at an angle. This gives each camera's lens from points known to the millimetre, which is what the wrist camera's fit lacks. With the distance from the shoulder's pan axis to the board's near corner measured, it also gives the top camera's pose.
2. **The arm at known poses.** `uv run so101 pose` (read-only) with the arm's power on and the jaws closed, their tips resting on the mat, at three distances from the base measured with the ruler. A tip on the mat at a known distance is two numbers the joints must agree with, so three of them give the offsets of shoulder_lift, elbow_flex and wrist_flex, which the demos can't split.
3. **The weights.** The block and the jar on kitchen scales (`block_mass`, `jar_mass`), and the jar's opening across.
4. **Pushes for the mat's friction.** `uv run so101 sim nudge --episodes 3` (README). This moves the arm and hasn't been run yet.

## What else would close the gap

- **Train with augmentation.** Blur, noise and colour jitter in training would make a policy less sensitive to exactly the detail a sim lacks, and should help on the arm too.
- **A held block from real pixels.** Once the block is grasped, the top camera draws it rendered; in the wrist camera it comes from demo frames of a carried block, placed by where the model thinks that demo's block was.

## What was tried and dropped

- **Fitting the mat's height.** The top camera draws the block about 9% larger than its red pixels cover, and the wrist camera saw the jar about 15% larger than the top camera's fit of it. A lower mat would explain the first; the second was the jar's height, 70 mm where 63 was assumed. Fitted from the block's size, the mat came out 7 to 16 mm below the base's plane, the arm's outline fitted worse (5.1 to 6.0 px against 3.8) and the optimiser's result depended on where it started. The photos show the base on the same table as the mat, and holding the mat lower fits the top camera as well, so the demos can't place it. `mat_height = "fit"` still does it.
- **A strong pull on the joint offsets.** Holding the elbow's offset under 2°, as its calibrated range seemed to ask, left the outline 5.1 px out with the other offsets nearly free of it; refitted with the elbow held and the others free, the outline is back at 3.8 to 4.0 px (the table above). The range argument was wrong anyway: the model's limits aren't where its parts meet.
- **A lens centre and freer joint offsets in the wrist camera's fit.** Letting the lens's axis sit off the picture's middle, and then letting the four offsets move, left the block's centre 15.8 px from the real one on held-back demos against 16.9 px without them, and the offsets didn't move. The 10 px that the block is drawn to the right of the real one stayed.
- **The wrist camera's lens from straight edges.** A straight edge must come out straight through the right lens, and the wrist camera sees the strip of bare table past the mat and the mat against the wall. But it sees them in too few places: 46 edges from 72 demos, all the same strip near the top of the picture, and the lens that straightened them best ran to the limits it was given. A checkerboard does this properly.
- **Fitting the top camera on still frames only**, so that its picture's delay couldn't matter. Too few different poses are still: the outline fitted 5.4 px out and the wrist's roll came out 19° off. The fit instead pairs each picture with the joints 3 ticks earlier.
- **Starting each block where the wrist camera saw it**, 3.9 mm (median) from where the top camera did. Fewer replayed demos then put the block in the jar (51 of 72 against 61), so the build keeps the top camera's places; it measures both each time and keeps the better.
- **Cast shadows.** MuJoCo's leave what they fall on nearly black, and the real ones are soft and grey.
- **Shading the rendered arm, and blending the wrist picture's seams.** Neither changed the policy's actions by 0.1°.
