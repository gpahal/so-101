# The simulated twin: how close it is

Status: built. `so101 sim` is described in the README under "Try a policy in simulation". This note holds what the fits reach on the `block-container` demos, how close the sim comes to the arm, what limits it, what was tried and didn't help, and what to measure at the arm to close the gap. Nothing here ran on the arm: the real numbers it quotes are from the demos and from the runs reviewed in `results/`.

No policy enters any fit or any choice in the build. Every number below that judges the sim compares it with the demos' own frames and joints, or replays commands that ran on the arm. `sim check` and `sim eval` report how a policy reacts to the sim; they are never used to tune it.

## The setup the sim is built for

125 demos hold the block. The build fits on 94 of them and holds 31 back for `score` and `check`.

- **The arm** is the arm model's size: from the shoulder-lift axis to the elbow's, 115 to 117 mm by ruler (the model: 116); elbow to wrist-flex, 134 to 136 (135); wrist-flex axis to the closed jaws' tips, 165 to 170 (165.9); the shoulder-lift axis above the table, 118 to 120 (119). Its base is clamped to the table the mat lies on, with nothing under it.
- **The block** is a DUPLO 2x2 brick: 32 x 32 x 19 mm by ruler, against the catalogue's 31.8 and 19.2. Its weight is BrickLink's 6.75 g.
- **The jar** is 62 to 64 mm across and 70 mm tall, taped over its lower 52 mm, with a clear threaded neck above. In the demos it stands in two places about 13 cm apart (1 SD over the demos: 8 x 67 mm).
- **The mat** is a smooth cloth mouse pad 3 mm thick and 30 cm from front to back. Its far edge is 0.40 m in front of the arm's base, its left edge 10.1 cm to the base's left. The table's edge runs along the mat's left edge, with the floor's carpet past it; the wrist camera sees only a thin rim of wood there (`table_past_mat = 0`). The jar often stands past the mat's far edge, so the wall is placed behind the furthest demo's jar, 46.2 cm out.
- **The cameras** saturate colours: the block reads about (171, 1, 1), the wood (186, 56, 1), the jar (150, 77, 22), the wall and arm a beige of saturation 0.2 to 0.4. Every colour test the fits use is by hue and saturation (`looks.py`): over 120 frames of each camera, pixels redder than 2.1 times their green fell into a block peak at -2° to 2° of hue and a wood peak from 8°, so the block is under 7°; the arm and wall are bright with saturation under 0.45, the wood and the jar over 0.6.

## What the fits reach

| | Result |
| --- | --- |
| Top camera's pose and lens | The drawn arm's outline is 5.4 px from the real one. The block appears 4.4 px from where it lay before the closing jaws slid it, 5.8 mm (median) towards the fixed jaw. The lens is close to a fisheye, as its picture shows. The fit starts from the scene's last camera as well as from guesses, because from guesses alone it can settle on a lens bent nearly like a plain pinhole (the outline then fits 5.9 px out, the block 5.4) |
| Joint offsets | shoulder_lift -0.7°, elbow_flex -5.8°, wrist_flex -1.7°, wrist_roll +5.2°, held by `joint_offsets_deg` (below) |
| Jar | Measured, so only its places are fitted. Each lies 4 mm (median) from where the arm's joints say its demo let go of the block, 12 mm at the 90th percentile |
| Block's start | The top camera's place, moved clear of where the demo's real jaws came down: 8 of 94 move, by 4 mm (median). The wrist camera sees each block 2.4 mm (median) from the top camera's place. Replayed, the demos put 92 of 94 in the jar from the top camera's places and 91 from the wrist camera's |
| Wrist camera | The drawn fixed jaw is 2.9 px from the real one, each block's centre 5.0 px (median) from the real one's at 4% under its size, and placed from far, near blocks are drawn 14.4 px from the real ones |
| Cameras' timing | Both pictures are 2 control ticks (67 ms) older than the joints saved with them. The top camera gives 25 new pictures a second, the wrist camera 24.5 |
| Motors, on demos they weren't fitted to | Joint error 0.17, 1.09, 0.41, 0.14, 0.16 and 0.16, shoulder_pan to gripper (degrees; the gripper in its own units). Each motor is searched on a grid and then refined continuously, which takes the error 10% to 40% below the grid's alone |
| Gripper | Holding the block, it reads the real gripper's 18.4 |

**The joint offsets are held, not fitted.** The demos can't say how the offsets split between the joints: fitted freely, the top camera draws the arm as well with the gripper pitched 5° differently (the offsets' sum -6.9° against -8.2°), about 8 mm at the jaw tips, and 34 of 94 blocks then start under a real jaw instead of 8. The held offsets were checked at the arm: with the closed jaw tips touched to the four corners of a tissue on the mat, from 170 to 350 mm out, the model's jaws come to rest 0.3 mm below, 0.3 below, 1.5 above and 0.6 above the mat. The elbow's -6° is real: the model's forearm meets its upper arm at 91°, 6° before its limit of 96.8°, and the follower folded flat reads 96.3°. Holding them, the top camera fits as well as with them free. They stay right for as long as the follower's calibration does; after recalibrating, remove `joint_offsets_deg` and rebuild with `--refit-cameras`.

**Closing jaws slide the block.** The moving jaw meets the block first and shoves it across until the fixed jaw stops it, which the wrist camera's frames show with the block's own width as the ruler (`fit.push_while_closing`). So a block the top camera sees on the mat is matched to where it lay before the slide (`arm.block_before_push`), not to where the jaws held it.

## How close it is

"Real pixels" is what the sim draws when the demos are on the computer: the arm in the top camera, the wrist camera's view, and each attempt's block and jar come from the demos' frames (README, "Real pixels"). Every image check below uses them.

| Check | Result |
| --- | --- |
| `sim replay`: every demo's real commands, nothing correcting the arm | 125 of 125 lift the block, 123 put it in the jar, including all 31 held back |
| `sim replay --recording`: the 24 attempts that ACT and π0.5, whole-task and skills models, drove on the arm, all of which held the block there and 22 placed it | 22 lift it, 20 put it in the jar, and 20 of the 24 end as on the arm |
| `sim score`, top camera, pixels from the real frame (two real frames of the same pose: 14.4) | 7.4 |
| `sim score`, wrist camera (two real frames: 17.7) | 11.7 |
| `sim score`, wrist camera's sharpness against a real frame's | 3.2 |
| `sim check`, the task's ACT policy: error from the demo on real images, on sim images | 0.73°, 2.25° |
| ... with only the top picture the sim's, only the wrist picture | 1.22°, 1.66° |
| `sim eval`, that ACT policy, 25 attempts (on the arm: 5 of 6) | 21 of 25 (65% to 94% at 95%) |

Physics doesn't depend on how things are drawn, and the replays say it's close: demos that worked on the arm work in the sim, and commands no fit saw end as on the arm in 20 of 24 attempts. The policy's reaction to the pictures is the larger gap: its actions move three times as far on sim pictures as on real ones, most of it from the wrist camera's.

## What limits it

- **Releases over the rim near the base.** Of the 4 run attempts the sim ends differently from the arm, 3 are at the block spot near the arm's base, where the policies let go with the block's centre 1 to 3 cm from the jar's middle and the arm itself missed 2 of 8: 2 that fell in on the arm miss in the sim, and 1 the arm missed falls in. The fourth, at the middle spot, misses in the sim where the arm placed it. Tuning the rim's contact to these attempts would fit the sim to them, so it isn't done.
- **The wrist picture.** It moves the policy further than the top picture does, and its sharpness reads 3.2 times a real frame's: its seams, where patches from different real frames meet. Its block is drawn 14.4 px (median) from the real one near a grasp when placed from far.
- **The top camera's outline.** 5.4 px: the camera is close to the arm, whose upper arm, lifted, fills a corner of its picture, where the lens is least sure.
- **The shoulder's lift.** Its error is 1°, almost all of it in fast moves: 2.8° while it moves faster than 20° a second, 0.4° when still. It's timing, not a sag.
- **Physics values that are typical, not measured:** the block's 6.75 g, the jar's 30 g, the mat's friction against the block (0.4, for plastic on a smooth cloth pad) and the jaws' (0.35: the arm model's 1.0 keeps hold of blocks the real jaws let go, and 0.3 to 0.45 lets the most demos replay).

## Tried, and not used

Each was measured on these demos the way the tables above measure the build.

- **A delay of its own for each motor**, within a control tick, which MuJoCo's actuator delay can model. Fitted with the rest of each motor, it made the shoulder's lift worse (1.09° against 0.96°) and changed the others by under 0.01.
- **Dropping small pale specks from the arm's outline.** The top camera then fitted worse (6.0 px and 5.4 px against 5.3 and 4.5).
- **Another delay for the top camera's picture.** The outline frames are nearly still, so 0 to 8 ticks fit within 0.02 px of each other.
- **Fitting each jar's place to the wrist camera's frames before each release.** From that close the real jar's amber covers only part of it, so the overlap stays low (0.33 at best) and says little. The top camera's places agree with where the arm lets go instead.
- **A strong opening on the jar's amber**, to drop the glow along the mat's edge against the wall. It erodes the jar's own patchy amber: the jars' places then lay 23 mm (median) from where their demos let go. The jar is found first as the largest solid patch over a wide area instead, and its outline is matched to the amber round it.
- **Cast shadows.** MuJoCo's leave what they fall on nearly black, and the real ones are soft and grey.
- **Fitting the mat's height.** The demos can't place it: held lower, the top camera's outline fits about as well and the offsets move to keep each grasp on the mat. The ruler says where it is.

## What to measure at the arm

1. **Pushes for the mat's friction, with the jaws.** `uv run so101 sim nudge --episodes 3` records where no training reads (README): the block on the mat, pushed a few centimetres with the jaws two or three times. Then `sim physics`.
2. **The weights.** The block and the jar on kitchen scales (`block_mass`, `jar_mass`), and the jar's opening across.
3. **The joint offsets again after recalibrating the follower**, with the closed jaws touched to a tissue's corners, as above.
4. **The top camera kept still.** A policy and the sim expect the demos' view; `sim aim` puts the camera and the mat back.
