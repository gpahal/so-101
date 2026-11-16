"""`so101 skills`: find each skill's span in every kept demo, check the handovers, and draw contact sheets.

Read-only on data/: it reads the raw sessions as `so101 build` does, with the same edits, and writes only
the contact sheets, under outputs/skills/<task>/<time>/. Run it before `build --skills` or `train --skills`
to see what each skill will hold: a sheet per skill and end, each tile the top camera over the wrist camera
at that frame, one tile per demo.
"""

import time

import numpy as np

from so_101 import skills as S
from so_101.grasp import thresholds

TILE_W, TILE_H, PER_ROW = 160, 120, 12


def _quantiles(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{np.percentile(v, 5):5.2f} {np.median(v):5.2f} {np.percentile(v, 95):5.2f}"


def inspect(s, task, sheets: bool = True) -> None:
    skills = task.require_skills()
    found = thresholds.load(s.perception.thresholds)
    print(f"{task.name}: {len(skills)} skills, events found with {found.path} (gripper v{found.gripper.version})")
    for skill in skills:
        handover = f", switched to at the {skill.handover}" if skill.handover else ", where every attempt starts"
        print(f"  {skill.name}: {skill.start} to {skill.stop}{handover}: \"{skill.prompt}\"")
    cuts, notes = S.cut_demos(s, task, found)
    for note in notes:
        print(f"  {note}")
    good = [c for c in cuts if c.error is None]
    bad = [c for c in cuts if c.error is not None]
    print(f"\n{len(good)} of {len(cuts)} kept demos cut" + (f"; {len(bad)} left out of a skills build:" if bad else "."))
    for c in bad:
        print(f"  session {c.plan.recording.name} episode {c.plan.episode}: {c.error}")
    if not good:
        return
    fps = good[0].plan.meta.fps
    print("\nseconds, 5% / median / 95%:")
    for skill in skills:
        print(f"  {skill.name + ' span':34s} {_quantiles([(c.spans[skill.name][1] - c.spans[skill.name][0]) / fps for c in good])}")
    for n in range(1, len(skills)):
        prev, nxt = skills[n - 1], skills[n]
        before = [(c.events[nxt.handover] - c.spans[nxt.name][0]) / fps for c in good]
        after = [(c.spans[prev.name][1] - 1 - c.events[nxt.handover]) / fps for c in good]
        print(f"  {(nxt.name + " starts before its handover"):34s} {_quantiles(before)}")
        print(f"  {(prev.name + " runs on after the handover"):34s} {_quantiles(after)}")
    for a, b in (("start", "close"), ("close", "lift"), ("close", "carry"), ("carry", "release"), ("release", "widest")):
        print(f"  {a + ' to ' + b:34s} {_quantiles([(c.events[b] - c.events[a]) / fps for c in good])}")
    if sheets:
        draw_sheets(s, task, good)


def draw_sheets(s, task, cuts) -> None:
    from PIL import Image, ImageDraw

    from so_101.dataset_io import episode_frames

    folder = s.outputs_dir / "skills" / task.name / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    rows = -(-len(cuts) // PER_ROW)
    total = len(task.skills) * 2 * len(cuts)
    done = 0
    print(f"\nDrawing contact sheets ({total} tiles) into {folder}...")
    for skill in task.skills:
        for end in ("first", "last"):
            sheet = Image.new("RGB", (TILE_W * PER_ROW, 2 * TILE_H * rows + 24), "white")
            ImageDraw.Draw(sheet).text(
                (6, 6), f"{skill.name}, {end} frame ({skill.start if end == 'first' else skill.stop}): "
                        "top camera over wrist camera, one tile per demo", fill=(0, 0, 0))
            for n, c in enumerate(cuts):
                p = c.plan
                start, stop = c.spans[skill.name]
                frame = p.start + (start if end == "first" else stop - 1)
                tile = Image.new("RGB", (TILE_W, 2 * TILE_H))
                for k, key in enumerate(("observation.images.top", "observation.images.wrist")):
                    if key in p.meta.video_keys:
                        image = next(episode_frames(p.meta, p.episode, key, frame, frame + 1))[1]
                        tile.paste(Image.fromarray(image).resize((TILE_W, TILE_H)), (0, k * TILE_H))
                ImageDraw.Draw(tile).text((3, 3), f"{p.recording.name[-6:]}/{p.episode} {(frame - p.start) / p.meta.fps:.1f}s",
                                          fill=(255, 255, 0))
                sheet.paste(tile, ((n % PER_ROW) * TILE_W, 24 + (n // PER_ROW) * 2 * TILE_H))
                done += 1
                print(f"\r  {done}/{total} tiles", end="", flush=True)
            sheet.save(folder / f"{skill.name}-{end}.jpg", quality=80)
    print()
    for path in sorted(folder.glob("*.jpg")):
        print(f"  {path}")
