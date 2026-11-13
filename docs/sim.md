# The simulated twin: how close it is

Status: built. `so101 sim` is described in the README under "Try a policy in simulation". This note holds what was measured when the sim for `block-container` was built, on 2026-11-13 and 14 from the 100 demos recorded on 2026-11-12, what limits it, and what to measure at the arm to close the gap. Nothing here ran on the arm: the real numbers it quotes are from the demos and from `results/act-block-container-baseline.md`.

## What the fits reached

Each fit used 72 of the 96 demos that hold the block; `score` and `check` used the other 24.

| | Result |
| --- | --- |
| Top camera's pose and lens, joint offsets | The drawn arm's outline is 3.8 px from the real one. The block appears 3.8 px from where the jaws closed on it |
| Joint offsets | shoulder_lift -0.9°, elbow_flex -6.4°, wrist_flex -1.3°, wrist_roll +4.8° |
| Block's start, seen by the top camera | 5.4 mm (median) from where the jaws later closed on it |
| Jar | 64 mm across; it stood within 5 x 5 mm (1 SD) over the demos |
| Wrist camera | Its drawn outlines are 6.8 px from the real jaws, 7.0 px from the block, 10.2 px from the jar |
| Cameras' timing | The top camera's picture is 3 control ticks (100 ms) older than the joints saved with it. The wrist camera's delay barely shows: 0 to 2 ticks fit alike, and 1 was taken. Both give 25 new pictures a second |
| Motors, on demos they weren't fitted to | Joint error 0.2, 1.1, 0.4, 0.2, 0.2 and 0.1, shoulder_pan to gripper (degrees; the gripper in its own units) |
| Gripper | Holding the block, it reads the real gripper's 18.8 with its zero 0.2° from the arm model's end stop, where the real gripper rests. That needed stiffer contacts than the arm model's: with those the jaws sank 2 to 3 mm into the block |

## What the checks found

"Rendered" draws everything from the arm model and fitted colours; its score and its eval are from the first build, before the cameras were refitted with their pictures' delay. "Real pixels" is what the sim does when the demos are on the computer: the arm in the top camera, the wrist camera's view, and each attempt's block and jar come from the demos' frames (README, "Real pixels").

| Check | Rendered | Real pixels |
| --- | --- | --- |
| `sim score`, top camera, pixels from the real frame (two real frames of the same pose: 7.2) | 9.3 | 5.4 |
| `sim score`, wrist camera (two real frames: 13.7) | 21.2 | 11.1 |
| `sim check`: the policy's actions from the demo's, on sim images (on real images: 0.6°) | 3.1° | 2.2° |
| ... with only the top picture the sim's | 2.2° | 1.3° |
| ... with only the wrist picture the sim's | 1.5° | 1.4° |
| `sim eval`, ACT policy, 100 action steps, 30 s an attempt (on the arm: 24 of 25) | 17 of 25 (68%) | 19 of 25 (76%; 57% to 89% at 95%) |

The real-pixels eval placed 17 of its first 18 and 2 of its last 7; an earlier run of 15, a build before, placed 12. It ran just before the last change to the wrist camera's drawing, which leaves the block rendered when a real frame can't supply it whole; `sim check` read the same within 0.05° either side of that change.

Physics, which doesn't depend on how things are drawn:

| Check | Result |
| --- | --- |
| `sim replay`: every demo's real commands, nothing correcting the arm | 90 of 96 lift the block, 85 put it in the jar |
| Jaw tips at the grasp in those replays, sim against real joints | 1.1 mm short and 1.6 mm high on average, within 3.1 mm (1 SD) |

## Why the policy still does worse in the sim

The physics isn't what holds the policy back: replayed demos grasp and drop the block, and the jaw tips arrive within millimetres of where the real ones did. The policy fails the way its real failures went, with the block against the fixed jaw instead of between the jaws, but more often, because the sim's pictures move its aim.

`sim check` on demo frames up to just after the grasp, with one picture real and the other changed one part at a time, gave these shifts in the policy's actions from what it predicts on the real pictures. Two degrees at the shoulder moves the jaws about a centimetre, and the block is 32 mm wide.

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
- **Where the block sits in the wrist picture is the largest piece left.** Real pixels didn't change it: the block is drawn 18 px (median) from where the real camera saw it. That's the wrist camera's place on the gripper, the joint offsets and the block's place on the mat, each a few millimetres out. The measurements below are for this.
- **The wrist picture has seams.** Where a patch from a real frame meets a rendered one there's an edge no real picture has: its sharpness reads 3.7 times a real frame's. Blending the seams didn't change the policy's actions.

So a sim result for this ACT policy is still a lower bound, not a prediction. Compare models in the sim only after `sim check` shows they react to sim pictures alike.

## What to measure at the arm

In order of what they would fix. Each `[sim]` setting is in `so101.toml`.

1. **Where the arm's base stands relative to the mat.** With a ruler, how far the mat's top is above (or below) the surface the base's bottom stands on, and what is under the base. The fit assumes 3 mm (`mat_height`). The elbow's fitted offset, -6.4°, is more than its calibrated range allows, and a mat lower than assumed would explain it.
2. **The arm at known poses.** Side-on photos, square to the arm, of the follower at the start pose, with its jaws closed on the block on the mat, and at one stretched pose, each with the output of `uv run so101 pose` (read-only). The link angles in the photos against the readings give the joint offsets directly, and the grasp photo shows how high the jaw tips are above the mat and how deep the block sits between them, which the fit assumes are 6 mm each.
3. **A checkerboard on the mat.** A printed checkerboard (squares of 25 mm, measured after printing) flat on the mat, with the distance from the shoulder's pan axis to its near corner measured, and `uv run so101 check` run once to save a frame from each camera. Then a second position. This gives the top camera's pose and lens from points known to the millimetre, where the fit now has the arm's outline, and it shows the wrist camera the same points.
4. **The wrist camera on the gripper.** Close-up photos from the side and from above with a ruler in view: where the lens is relative to the jaw tips, and how far it's tilted.
5. **The block and the jar.** With calipers or a ruler: the block's width and its height without studs; the jar's outer diameter, height, the height of its tinted band and its opening. On kitchen scales: both weights (`block_width`, `block_height`, `block_mass`, `jar_diameter`, `jar_height`, `jar_tint_height`, `jar_mass`). The wrist camera sees the jar about 15% larger than the top camera's fit of it.
6. **The top camera.** Its lens's height above the mat and its place over the table by tape, and the model on its label, for its field of view.
7. **Pushes for the mat's friction.** `uv run so101 sim nudge --episodes 3` (README). This moves the arm and hasn't been run yet.
8. **Photos of the whole setup** from the front, the side and above, which settle anything the others leave unclear.

## What else would close the gap

- **Train with augmentation.** Blur, noise and colour jitter in training would make a policy less sensitive to exactly the detail a sim lacks, and should help on the arm too.
- **A held block from real pixels.** Once the block is grasped, the top camera draws it rendered; in the wrist camera it comes from demo frames of a carried block, placed by where the model thinks that demo's block was.

## What was tried and dropped

- **Fitting the mat's height.** The top camera draws the block about 9% larger than its red pixels cover, and the wrist camera sees the jar about 15% larger than the top camera's fit of it. Both would follow from a mat a few centimetres lower than the 3 mm the build assumes. Fitted from the block's size, the mat came out 7 to 16 mm below the base's plane, but the arm's outline then fitted worse (5.1 to 6.0 px against 3.8) and the optimiser's result depended on where it started. `mat_height = "fit"` still does it. Measurement 1 settles it.
- **A strong pull on the joint offsets.** The follower's calibrated ranges match the arm model's within 2° at the elbow and wrist, yet the outline fit wants -6.4° at the elbow. Holding it under 2° left the outline 5.1 px out. The offset may stand in for something else, such as a forearm that isn't quite the model's, or the mat's height.
- **Fitting the top camera on still frames only**, so that its picture's delay couldn't matter. Too few different poses are still: the outline fitted 5.4 px out and the wrist's roll came out 19° off. The fit instead pairs each picture with the joints 3 ticks earlier.
- **Starting each block where the wrist camera saw it**, 3.9 mm (median) from where the top camera did. Fewer replayed demos then put the block in the jar (51 of 72 against 61), so the build keeps the top camera's places; it measures both each time and keeps the better.
- **Cast shadows.** MuJoCo's leave what they fall on nearly black, and the real ones are soft and grey.
- **Shading the rendered arm, and blending the wrist picture's seams.** Neither changed the policy's actions by 0.1°.
