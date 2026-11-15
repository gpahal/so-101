# The simulated twin: how close it is

Status: built. `so101 sim` is described in the README under "Try a policy in simulation". This note holds what was measured when the sim for `block-container` was built, on 2026-11-13 and 14 from the 100 demos recorded on 2026-11-12, what a ruler, nine photos of the setup and a recording of a tissue on the mat added on 2026-11-14, what limits it, and what to measure at the arm to close the gap. Nothing here ran on the arm: the real numbers it quotes are from the demos and from `results/act-block-container-baseline.md`.

## What the fits reached

Each fit used 72 of the 96 demos that hold the block; `score` and `check` used the other 24. These are from the last build of 2026-11-14, with the jar as measured and each block placed where it lay before the jaws slid it.

| | Result |
| --- | --- |
| Top camera's pose and lens, joint offsets | The drawn arm's outline is 3.8 px from the real one. The block appears 3.2 px from where it lay before the closing jaws slid it, 4.0 mm (median) towards the fixed jaw; matched to where the jaws held it instead, 3.9 px |
| Joint offsets | shoulder_lift -0.7°, elbow_flex -5.8°, wrist_flex -1.7°, wrist_roll +5.2° |
| Block's start | The top camera saw it 6.7 mm (median) from where the jaws later closed on it. 22 of the 96 lay under where a real jaw came down, and were moved clear by 3 mm (median); 37 did before the slide was allowed for |
| Jar | Measured: 63 mm across, 70 mm tall, taped over its lower 52 mm. It stood within 5 x 6 mm (1 SD) over the demos |
| Wrist camera | Fitted to the fixed jaw and to each demo's block seen from far and near ("The wrist camera, refitted"). The drawn fixed jaw is 2.9 px from the real one, each block's centre 7.2 px (median) from the real one's at 1% under its size, and placed from far, near blocks are drawn 13.0 px from the real ones (30 px on the build before) |
| Cameras' timing | The top camera's picture is 3 control ticks (100 ms) older than the joints saved with it. The wrist camera's delay barely shows: 0 to 2 ticks fit alike, and the last two builds took 0 and 1. Both give 25 new pictures a second |
| Motors, on demos they weren't fitted to | Joint error 0.2, 1.1, 0.4, 0.2, 0.2 and 0.1, shoulder_pan to gripper (degrees; the gripper in its own units) |
| Gripper | Holding the block, it reads the real gripper's 18.8 with its zero 0.2° from the arm model's end stop, where the real gripper rests. That needed stiffer contacts than the arm model's: with those the jaws sank 2 to 3 mm into the block |

## What the checks found

"Rendered" draws everything from the arm model and fitted colours. "Real pixels" is what the sim does when the demos are on the computer: the arm in the top camera, the wrist camera's view, and each attempt's block and jar come from the demos' frames (README, "Real pixels"). The first two columns are from 2026-11-13, when the jar's height was a guess (63 mm) and the jaws' friction was the arm model's 1.0.

| Check | Rendered, 13 Nov | Real pixels, 13 Nov | 14 Nov, jar measured and jaws' friction lowered | 14 Nov, and blocks moved clear of the jaws | 14 Nov, and blocks placed before the jaws' slide | 14 Nov, and the wrist camera refitted | 15 Nov, and the wrist picture's block always painted from real frames |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sim score`, top camera, pixels from the real frame (two real frames of the same pose: 7.2) | 9.3 | 5.4 | 5.3 | 5.3 | 5.3 | 5.3 | 5.3 |
| `sim score`, wrist camera (two real frames: 13.7) | 21.2 | 11.1 | 10.0 | 9.9 | 10.2 | 9.9 | 9.8 |
| `sim score`, wrist camera's sharpness against a real frame's | | 3.7 | 2.3 | 2.3 | 2.5 | 2.9 | 3.0 |
| `sim check`: the policy's actions from the demo's, on sim images (on real images: 0.6°) | 3.1° | 2.2° | 2.0° | 2.0° | 2.0° | 1.9° | 1.8° |
| ... with only the top picture the sim's | 2.2° | 1.3° | 1.3° | 1.3° | 1.3° | 1.3° | 1.3° |
| ... with only the wrist picture the sim's | 1.5° | 1.4° | 1.2° | 1.2° | 1.2° | 1.2° | 1.1° |
| `sim replay`: of 96 demos' real commands, the ones that put the block in the jar | | 85 | 88 | 90 | 90 | 90 | 90 |
| `sim replay --recording`: of the 24 attempts of the real ACT run that held the block, the ones the sim puts in the jar | | | 17 | 20 | 19 | 19 | 19 |
| `sim eval`, ACT policy, 100 action steps, 30 s an attempt (on the arm: 24 of 25) | 17 of 25 (68%) | 19 of 25 (76%; 57% to 89% at 95%) | 15 of 25 (60%; 41% to 77%) | 17 of 25 (68%; 48% to 83%) | 18 of 25 (72%; 52% to 86%) | 36 of 50 (72%; 58% to 83%), as the column before on the same 50 | 41 of 50 (82%; 69% to 90%) |

Every eval starts its attempts from the same 25 layouts in the same order, so the runs can be compared attempt by attempt:

- **The same layouts fail, at both edges of where the demos put the block.** Over the five real-pixels runs of 25, the attempts whose block lay 27° to 30° to the arm's right placed 4 of 20, those at 12° to 14° to its left 12 of 25, and those from 17° right to straight ahead 58 of 65. One far-right attempt and one on the left failed in all five. On the arm the far right was the hardest spot too, with 4 of 5. Of the demos, 16 of 96 have their block more than 23° right, the edge of where they put it.
- **The real-pixels runs don't differ by more than chance.** 19, 15, 17 and 18 of 25 have intervals that overlap from 57% to 77%, and an earlier run of 15 attempts, a build before, placed 12. On 50 attempts with the same layouts, the build of `sim-checkpoint-1` placed 36, the same with the wrist camera fitted from far and near 39, and the build that fits it that way 36: the camera moved the left of the mat from 7 of 12 to 10 both times, and nothing else by more than chance. Which change moved which attempt wasn't separated: 25 attempts can't show a change of 4.
- **Half the failures seen were out of time, not out of reach.** Videos are kept for the first 10 attempts. Of the 4 that failed in the first run of 14 November, 2 never closed on the block, and 2 hovered over it, grasped late and were holding it over the jar when the 30 s ran out.

Physics, which doesn't depend on how things are drawn:

| Check | Result |
| --- | --- |
| `sim replay`: every demo's real commands, nothing correcting the arm | 96 of 96 lift the block, 90 put it in the jar (93 and 90 before the slide was allowed for) |
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

## What the tissue recording of 14 November settled

`sim nudge --episodes 3` recorded three takes at 15:06 on 14 November, with a tissue 20 cm square laid on the mat: the closed jaw tips touched to its four corners (30 s), the gripper swept over it (17 s), and with the tissue gone the block pushed about by hand (27 s). A ruler then measured the arm itself.

- **The arm is the arm model's size.** From the shoulder-lift axis to the elbow's, 115 to 117 mm (the model: 116); elbow to wrist-flex, 134 to 136 (135); wrist-flex axis to the closed jaws' tips, 165 to 170 (the model's jaw mesh reaches 165.9); the shoulder-lift axis above the table, 118 to 120 (119).
- **The joint offsets and the mat's height agree to 1.5 mm.** At the four corner touches, from 170 to 350 mm out, the model's jaws come to rest 0.3 mm below, 0.3 below, 1.5 above and 0.6 above the mat.
- **The touches don't say where the corners were.** Laid out flat they make a square 190 mm across, against 197 mm for the tissue in the top camera's picture, and in the wrist camera's frames one corner lies a centimetre or two past the jaws' tips. So they check heights, not places.
- **The wrist camera hasn't moved on the gripper.** The fixed jaw's edges sit within 1 px of where they were in the demos' frames, row by row.
- **The top camera has moved since the demos,** so today's picture can't check the demos' geometry directly. Through the demos' camera, today's frame puts the tissue 7 to 9 mm further from the arm than the wrist camera and the arm's joints do, and the mat's far edge 7 mm nearer than on 12 November.
- **The tissue doesn't pin the wrist camera's lens better than the demos do.** Fitted to the tissue's edges alone, the lens traded focal length for distance (444 px, the jaws drawn 30 px off). Fitted to the tissue and the jaws together, it drew the tissue's edges and the jaws right in every frame, but on the demos it put the block 9.1 px and the jar 15 to 16 px off, against 7.2 and 9 to 10 for the demos' own fit. The tissue's map comes from the moved top camera, which may be why. The build keeps the demos' fit.
- **The third take has no friction in it.** A finger pushed the block, not the jaws, so `sim physics` has nothing to fit; don't run it on this recording.

**What it led to: closing jaws slide the block.** The moving jaw meets the block first and shoves it across until the fixed jaw stops it. In the wrist camera's frames, between the frame the jaws start to close and the hold, the block crosses towards the fixed jaw by 4.0 mm (median; quartiles 2.4 and 5.6) in 95 of 96 demos, measured with the block's own width as the ruler (`fit.push_while_closing`), and about 3 mm over the mat. The top camera's fit had matched each block's first sighting to where the jaws held it, so it drew each block about 3 mm towards where the fixed jaw comes down, which is how the sim's attempts fail. It now matches the sighting to where the block lay before the slide (`arm.block_before_push`). The block then appears 3.2 px from that place against 3.9, 22 of 96 starting places lie under a real jaw against 37, and the replayed demos lift 96 of 96 against 93; the wrist camera fits 1 px worse on the block, and the ACT policy placed 18 of 25, against 17 on the build before. That last difference is chance.

## The wrist camera, refitted

The build used to fit the wrist camera to the jaws, the block and the jar at the places the top camera gave them. That took on whatever error those places had, and gave a different lens on every build: focal lengths of 338, 390, 360 and 376 px, each fitting its own data within a pixel. A test that never uses the top camera tells them apart. Each held-back demo's block is placed from the wrist camera's own sightings more than 1 s before the jaws close, and the camera then has to draw it right in the sightings under 0.5 s.

| Wrist camera | Near block, from where the far ones put it | Its size there |
| --- | --- | --- |
| Build of 13 November | 13.2 px | -1.7% |
| Build of 14 November (`sim-checkpoint-1`) | 30.1 px | +14.2% |
| Fitted to the tissue recording and the jaws | 9.0 px | +3.4% |
| Fitted to the fixed jaw and each block seen from far and near, as the build now does | 13.7 px | +2.9% |

The build now fits it the second way (`fit.fit_wrist_camera`). The fixed jaw is part of the gripper the camera rides on, so its pixels never move: the build finds them in the demos' frames over the dark mat (`fit.fixed_jaw_core`; they match the tissue recording's jaw to 0.85 overlap) and they pin where the camera sits. Each demo's block, seen as the gripper comes down to it, pins how the lens draws one thing at many distances, and its place is fitted too. The jar is left out: with it, the same fit drew near blocks 28 px off. Several lenses still fit about as well, so the fit follows five starts to the end and keeps the one that draws near blocks best from far, on the demos it's fitted to.

With the new camera, blocks placed by the wrist camera replay into the jar as often as the top camera's (66 and 64 of 72 against 66 and 65); with the old one, 61 and 58.

Measured with the ACT policy as an instrument only, on frames of the demos' approach and grasp, the old camera was most of what moved its actions on sim pictures. Its aim moved 1.25° from the one on real pictures with the block at the mat's far right, 0.80° in the middle and 0.92° at the left, mostly at the shoulder, which aimed 2.1° short at the far right; with the camera fitted from far and near, 0.87°, 0.63° and 0.71°, and 0.8° short.

## Following attempts one at a time

Changing the sim and counting placements over 25 or 50 attempts didn't say why attempts failed. The ACT run of 2026-11-13 13:09 (24 of 25 on the arm) did, attempt by attempt. It saved every frame, joint and command, the policy carried out 100-action chunks as the sim's eval does, and ACT predicts the same chunk from the same input. So:

- **Along each real attempt,** at every chunk's start, the policy predicts from the real frames, from the sim's pictures with the arm, block and jar where they really were, and from one of each. From the real frames it reproduces the chunk the arm carried out to 1 to 3 mm at the jaws' tips (median per phase; up to 14 mm on an attempt's first chunk), so whatever the sim's pictures change is theirs.
- **From each real attempt's start,** the policy runs in the sim, closed loop, with every contact between the jaws and the block recorded, and each sim attempt is set beside the real one.

| Chunk | Real frames against what the arm did | Sim pictures | Sim top picture only | Sim wrist picture only |
| --- | --- | --- | --- | --- |
| The first, at the start pose | 6.0 mm | 26.4 mm | 16.2 | 12.7 |
| Approach | 1.5 mm | 9.9 mm | 8.1 | 5.4 |
| The one with the grasp | 2.0 mm | 20.8 mm | 9.0 | 16.2 |

(Medians of how far the jaws' tips move from the real-frame chunk, averaged over each chunk.)

**At the grasp, the wrist camera's block decides it.** With the real block's pixels put into the sim's wrist picture, the grasp chunk moved 6.5 mm instead of 16.2; with only the sim's block put into the real frame, 12.5. And it was the block's look more than its place: the sim's block pasted where the real one was moved the chunk 10.3 mm, the real block moved to where the sim drew it, 5.0. The painter (`photo.WristPhotos`) used to leave the block rendered whenever the nearest demo frame couldn't paint 80% of it spot by spot, as up close at a grasp it couldn't about half the time; a rendered block moved the grasp chunk 22 mm (median, 13 chunks) and a painted one 5 mm (11 chunks). It also read the mat round where the real block stood in that frame from the block's dark, shaded edges, which left a dark outline of it on the mat.

**In the sim the fixed jaw came down on the block.** Closed loop from the 25 real starts, the sim put 18 in the jar. In every failure the policy commanded the descent as on the arm, but the fixed jaw met the block's top and held the arm 30 to 35 mm up: when the jaws began to close, the block lay 22 mm below their midpoint and 15 to 30 mm towards the fixed jaw, where on the arm it lay level with them and 9 mm towards it. Replaying the arm's own commands in the sim, the same attempts succeed, so it's the policy's aim from the sim's pictures, not the physics or where the block is.

**The fix.** A block the painter can't paint spot by spot now gets the nearest frame's real block, scaled and moved onto the drawn one (`photo.BLOCK_PATCH`), and mat within 9 px of the real block's red isn't read from that frame (`photo.BLOCK_EDGE`). The grasp chunk then moves 10.1 mm (median) instead of 16.2, and the approach chunks are unchanged (5.3 against 5.4).

| From the 25 starts of the real run | Sim puts in the jar | Outcome as on the arm |
| --- | --- | --- |
| Before | 18 | 18 |
| With the block painted as above | 20 | 21 |
| And the block's and jar's top camera pixels from the run's own frame 3 s in, as an eval takes a demo's | 21 | 22 |

Over the eval's 50 attempts with the same layouts, it placed 41 against 36: 6 placed that the build before missed, and 1 missed that it placed (p = 0.13 by an exact sign test).

What's left, each measured the same way:

- **Real attempts 18 and 23 were too close to call.** With the blocks where the sim places them, the arm's own fixed jaw passed the block's top 0.0 and 0.4 mm to its side, measured from the jaw's mesh at the real joint readings. In 12 of the 24 real grasps it passed within 1 mm; over its three runs the sim placed 26 of those 36 attempts, against 33 of 36 of the others. The sim knows a block's place to 3 to 5 mm and the policy's grasp from its pictures to about 10, so a grasp that tight goes either way. The run's own wrist frames put its blocks 3.5 mm (median) further from the fixed jaw than the sim does, in 20 of 21 attempts, but placing them there made it worse: 17 of the 25 in the jar instead of 21 (its top picture then showed the block's real pixels up to 10 mm from where the block lay).
- **The block drawn 14 px high at the grasp is that run's alone.** On the demos' layouts, the ones the eval uses, the wrist picture's block is within a pixel of the real one's on average in the last 0.8 s before the jaws close (-0.9 px across, +0.7 px down, 10 to 11 px apart in any one frame), drawn 7% too big. The run's blocks were placed through a top camera that had moved since the demos.
- **The table's edge past the mat and the floor don't matter.** They're a third of the wrist picture on the approach, and with the real frame's pixels there instead, the approach chunks moved 5.3 mm and the grasp chunks 9.5 mm, against 5.3 and 10.1 with the sim's.
- **The block's look is still half of what the wrist picture does at the grasp.** With the real block's pixels, the grasp chunk moves 5.8 mm instead of 10.1; with only the sim's block in the real frame, 7.3. The rest is spread over the jaws and the mat.

## Four more things tried on 15 November

Each was measured as the wrist painter's fix was: first by how far it moves the ACT policy's plan from the plan it makes on real frames, at the same arm pose, then in closed loop.

- **The top picture's arm.** Pasting one region of the sim's top picture into a real frame at a time (held-back demos, wrist picture real), the arm moved the plan 9.4 mm at the grasp and 8.4 at the start, against 9.9 and 12.2 for the whole picture; the block and the jar, pasted from real frames, 1.5 mm or less, and the background 3.3 and 6.2. Shifting the drawn arm 3 to 8 px either way made it worse (10.5 to 17.9 mm), so it's drawn where it should be, and what moves the policy is its look, copied from the nearest-posed demo frame. Copying from all 96 demos instead of the 72 the fits used didn't help (start 19.8 mm against 16.2, grasp 8.1 against 9.0).
- **The wrist picture's block, again.** Always pasting the frame's whole real block, instead of painting it spot by spot when 80% of it maps, moved the grasp chunk 10.6 mm against 10.1. Copying a block from another frame doesn't get closer than this.
- **The block's size and the jaws in the wrist picture.** The sim draws the block 1.08 times the real one's size just before a grasp, but scaling it down moved the plan more, not less (18.8 mm against 13.6). Taking the jaws from the demo frame with the nearest gripper opening instead of the one the rest comes from changed nothing (14.6 against 13.6 on held-back demos, 5.1 against 4.4 on the real run).
- **Grasp physics.** The mat is a smooth cloth mouse pad: 0.4 against the block, where 0.6 was assumed (no measured figure for plastic on one was found; plastic on woven fabric is usually 0.3 to 0.5). The block weighs 6.75 g, BrickLink's weight for the DUPLO 2x2 brick, where 5.5 g was assumed. Replaying the real ACT run's own commands with each block moved 0, 2 and 4 mm towards the fixed jaw, these put 19, 18 and 14 of the 24 grasps in the jar, against 19, 17 and 15 before; fitting the jaws' collision boxes inside the printed part as well (the fixed jaw's faces stood out of it by up to 1.2 mm, one of the moving jaw's by 10 mm) gave 20, 18 and 14. A grazing jaw rests on the block whatever these are, and it didn't make the real arm's grasps any less tight: those cleared the block by under 1 mm in half the attempts. The build uses the two values (`block_mass`, `table_friction` in `so101.toml`) because they're the better-founded ones, not for a score: on them the ACT policy placed 39 of the eval's 50 against 41 (2 placed that the build before missed, 4 missed that it placed), and from the real run's 25 starts 20 against 21, both within chance. The demos replay 96 lifted and 89 in the jar (90 before). The arm model's collision boxes are as they were.

So none of the four moved the sim closer to the arm. What's left is the policy's: it grasped with under 1 mm to spare in half its real attempts, and it reacts to detail no patch of another frame reproduces (a real frame blurred by 1 px moves its actions 1°). A block placed to 1 mm, which needs demos and a calibration recorded with the top camera fixed in place, would shrink the first; a policy trained with image augmentation would shrink the second.

## What replaying a real policy run showed

`so101 run` saves every attempt's frames, joints and commands, so the run of 2026-11-13 that placed 24 of 25 on the arm can be replayed like a demo (`sim replay --recording`). No fit saw it and no model is in the loop: its commands worked on the arm, so they should work in the sim.

At first they didn't. Of the 24 attempts that held the block on the arm, the sim put 17 in the jar, and missed all 3 at the far-right spot. There the sim's fixed jaw came to rest on top of the block, 25 mm up, where the real one went down beside it to 7 mm, and the jaws closed on nothing.

- **The policy leaves the block almost no room.** It brings its fixed jaw down within a few millimetres of the block's side, most of all at the far right; a demo's grasp is centred. On the arm that works, and it's also where the run's one failure was. In the sim a block that starts a few millimetres off lies under the jaw.
- **The top camera can't place a block that closely.** It and the arm's joints disagree by about 5 mm on where a block is, and by more for this run: the top camera had moved 1 to 2.5 px between the demos, on 12 November, and the run a day later (its picture of the table's edge and of the mat's corner, lined up with the demos'). By 14 November it had moved 4 to 13 px. The π0.5 run of that morning is replayed all the same, and a moved camera is part of why it replays worse.
- **Each recording says where its block wasn't.** The real jaws never passed through the real block. So each block is moved to the nearest spot to where the top camera saw it that the real arm, at its real joints over the 5 s before its jaws begin to close, doesn't touch (`fit.clear_of_jaws`). After the jaws close they push the block, so nothing later counts.

With each attempt's block moved clear of that attempt's own jaws, the sim lifts 22 of the 24 and puts 20 in the jar; 11 blocks moved, by 5 mm (median), and the far-right ones by 6 to 8 mm. The attempt that failed on the arm fails in the sim too. The build now does the same for each demo (37 of 96 move, by 2 mm), and the replayed demos went from 88 of 96 in the jar to 90.

Other runs, the same way:

| Run on the arm | Held the block | Sim lifts it | Sim puts it in the jar |
| --- | --- | --- | --- |
| ACT, 2026-11-13 13:09, 25 attempts (24 placed on the arm) | 24 | 22 | 20; 19 on the last build |
| ACT, 2026-11-13 12:49, 25 attempts | 20 | 14 | 9 |
| π0.5, 2026-11-13 13:28 and 14:23, 5 attempts each | 7 | 4 | 2 |
| π0.5, 2026-11-14 10:37, 25 attempts | 23 | 17 | 15 |

The 12:49 run's results on the arm weren't written down; its last frames show most of its blocks in the jar. Its first frames often show a hand still putting the block down, which is why a recording's block is placed from the frame 3 s in.

What this doesn't fix is a policy running in the sim. There the block starts where a demo's did, moved clear of that demo's jaws, not of the policy's, and the policy comes down closer to the block than a demo did. In an eval of 14 November the attempts that placed the block first touched it 8 to 10 s in, as on the arm, and had it in the jar by 11 to 22 s (on the arm the jaws open over the jar at 12 to 15 s); the ones that failed closed their jaws up to 5 times without lifting it.

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
- **Where the block sits in the wrist picture was the largest piece left,** until the wrist camera was refitted ("The wrist camera, refitted"). On the builds before that: Neither real pixels nor the measured jar changed it: the block is drawn 19 to 22 px (median, over the last builds) from where the real camera saw it, on average 10 px to its right. What causes it isn't known: a lens centre and freer joint offsets in the wrist camera's fit didn't remove it (below). The wrist camera's lens is the least pinned thing left. Its focal length came out 338, 390, 360 and 376 px on four builds, and its bend `k1` 0.32 on the third and 0.01 on the fourth, the outlines fitting within 1 px of each other, and a lens that's wrong puts things in the wrong place away from the picture's middle. A checkerboard pins it (below); a tissue on the mat didn't (above).
- **The wrist picture has seams.** Where a patch from a real frame meets a rendered one there's an edge no real picture has: its sharpness reads 2.3 times a real frame's (3.7 before the jar was measured). Blending the seams didn't change the policy's actions.

So a sim result for this ACT policy is still a lower bound, not a prediction. Compare models in the sim only after `sim check` shows they react to sim pictures alike.

## What to measure at the arm

In order of what they would fix. Each `[sim]` setting is in `so101.toml`. Measured on 2026-11-14 and no longer needed: the block's, the jar's, the mat's and the arm's sizes, photos of the whole setup, and a tissue touched and swept over.

What stands between the sim and the arm's 24 of 25 is a few millimetres at the edges of where the demos put the block: the policy's jaw clears the block by that much, and nearly every failure is at the far right or the left.

1. **Pushes for the mat's friction, with the jaws.** `uv run so101 sim nudge --episodes 3` teleoperates and records where no training reads (README): the block on the mat, pushed a few centimetres with the jaws two or three times. The recording of 14 November pushed it by hand, which `sim physics` can't use.
2. **The weights.** The block and the jar on kitchen scales (`block_mass`, `jar_mass`), and the jar's opening across.
3. **New demos if the top camera has moved.** It had by 14 November, by 4 to 13 px. The sim is built for where the camera was on 12 November, and so is a policy trained on those demos.

## What else would close the gap

- **Train with augmentation.** Blur, noise and colour jitter in training would make a policy less sensitive to exactly the detail a sim lacks, and should help on the arm too.
- **A held block from real pixels.** Once the block is grasped, the top camera draws it rendered; in the wrist camera it comes from demo frames of a carried block, placed by where the model thinks that demo's block was.

## What was tried and dropped

- **The arm's shadow in the top camera's picture.** With the block at the far right the arm is over the bare wood, where its shadow shows, and the sim's top picture is 6.3 from the real frame there against 3.9 with the block in the middle. A soft shadow on the table and the wall, with the light's direction, softness and darkness fitted to the demos' frames, fitted best with no darkness at all: 15 px out from the arm the background is already within 3.9 of the real frame, and nearer than that the picture comes from real frames.
- **Fitting the mat's height.** The top camera draws the block about 9% larger than its red pixels cover, and the wrist camera saw the jar about 15% larger than the top camera's fit of it. A lower mat would explain the first; the second was the jar's height, 70 mm where 63 was assumed. Fitted from the block's size, the mat came out 7 to 16 mm below the base's plane, the arm's outline fitted worse (5.1 to 6.0 px against 3.8) and the optimiser's result depended on where it started. The photos show the base on the same table as the mat, and holding the mat lower fits the top camera as well, so the demos can't place it. `mat_height = "fit"` still does it.
- **A strong pull on the joint offsets.** Holding the elbow's offset under 2°, as its calibrated range seemed to ask, left the outline 5.1 px out with the other offsets nearly free of it; refitted with the elbow held and the others free, the outline is back at 3.8 to 4.0 px (the table above). The range argument was wrong anyway: the model's limits aren't where its parts meet.
- **A lens centre and freer joint offsets in the wrist camera's fit.** Letting the lens's axis sit off the picture's middle, and then letting the four offsets move, left the block's centre 15.8 px from the real one on held-back demos against 16.9 px without them, and the offsets didn't move. The 10 px that the block is drawn to the right of the real one stayed.
- **Fitting the wrist camera with each block's place free but the jar kept.** The jar's places come from the top camera, and with them the fit drew near blocks 28 px off and 14% too big; without the jar, 17 px, and choosing among starts by how well far blocks predict near ones, 13.
- **The wrist camera's lens from the tissue recording.** It drew the tissue and the jaws right and the demos' block and jar worse ("What the tissue recording settled").
- **The wrist camera's lens from straight edges.** A straight edge must come out straight through the right lens, and the wrist camera sees the strip of bare table past the mat and the mat against the wall. But it sees them in too few places: 46 edges from 72 demos, all the same strip near the top of the picture, and the lens that straightened them best ran to the limits it was given. A checkerboard does this properly.
- **Fitting the top camera on still frames only**, so that its picture's delay couldn't matter. Too few different poses are still: the outline fitted 5.4 px out and the wrist's roll came out 19° off. The fit instead pairs each picture with the joints 3 ticks earlier.
- **Starting each block where the wrist camera saw it**, 3.9 mm (median) from where the top camera did. Fewer replayed demos then put the block in the jar (51 of 72 against 61), so the build keeps the top camera's places; it measures both each time and keeps the better.
- **Cast shadows.** MuJoCo's leave what they fall on nearly black, and the real ones are soft and grey.
- **Shading the rendered arm, and blending the wrist picture's seams.** Neither changed the policy's actions by 0.1°.
