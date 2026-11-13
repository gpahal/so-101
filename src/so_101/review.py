"""`so101 review`: step through recorded episodes and decide what to keep, trim or delete.

Each decision is written to edits.toml straight away; the recordings themselves are never changed.
`so101 build` applies the edits.
"""

import subprocess
import sys
from pathlib import Path

from so_101 import data
from so_101.build import MIN_EPISODE_S
from so_101.dataset_io import column, episode_table
from so_101.settings import Settings
from so_101.tasks import Task
from so_101.validate import idle_trim

HELP = """\
  Enter        accept the suggestion
  k            keep the whole episode
  t START END  keep only START..END seconds, e.g. `t 1.2 14.5`
  d [REASON]   delete the episode from training, e.g. `d dropped the cube`
  v            open the episode in the viewer again
  s            skip for now (no decision)
  q            quit; decisions so far are saved
"""


def view(repo_id: str, root: Path, episode: int) -> None:
    """Open episode `episode` of the dataset at `root` in Rerun, without waiting for it: both cameras
    plus joint plots, with a timeline in seconds."""
    exe = Path(sys.executable).parent / "lerobot-dataset-viz"
    subprocess.Popen(
        [str(exe), "--repo-id", repo_id, "--root", str(root), "--episode-index", str(episode), "--num-workers", "0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def ask(prompt: str) -> str:
    """The operator's answer to `prompt`; "q", to quit, at the end of input or Ctrl+C."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def describe(edit: data.Edit | None) -> str:
    if edit is None:
        return "not reviewed"
    if edit.delete is not None:
        return f"deleted ({edit.delete or 'no reason'})"
    if edit.keep:
        return f"keep {edit.keep[0]:.2f}-{edit.keep[1]:.2f} s"
    return "keep all"


def parse(answer: str, duration: float, suggestion: data.Edit) -> data.Edit | str | None:
    """Turn an answer into an edit, a command ("v", "s", "q", "?"), or an error message string."""
    words = answer.split()
    if not words:
        return suggestion
    cmd, args = words[0].lower(), words[1:]
    if cmd in ("v", "s", "q", "?", "h", "help"):
        return cmd
    if cmd == "k":
        return data.Edit(suggestion.session, suggestion.episode)
    if cmd == "d":
        return data.Edit(suggestion.session, suggestion.episode, delete=" ".join(args))
    if cmd == "t":
        try:
            start, end = (float(a) for a in args)
        except ValueError:
            return "Usage: t START END, in seconds, e.g. t 1.2 14.5"
        end = min(end, duration)
        if not 0 <= start < end or end - start < MIN_EPISODE_S:
            return f"Need 0 <= START < END <= {duration:.2f}, at least {MIN_EPISODE_S} s apart."
        return data.Edit(suggestion.session, suggestion.episode, keep=(round(start, 2), round(end, 2)))
    return f"Unknown answer {answer!r}. Type ? for help."


def review(s: Settings, task: Task, only: str | None, everything: bool, viewer: bool) -> None:
    edits = data.load_edits(s, task)
    episodes = [
        (session, ep)
        for session in data.sessions(s, task)
        if session.meta
        for ep in range(session.episodes)
        if (only is None or only in (session.name, f"{session.name}/{ep}"))
    ]
    if only is None and not everything:
        episodes = [(session, ep) for session, ep in episodes if (session.name, ep) not in edits]
    if not episodes:
        if everything or only:
            print(f"No episodes for {task.name}.")
        else:
            print(f"No unreviewed episodes for {task.name}; --all shows every one.")
        return
    print(f"Reviewing {len(episodes)} episodes of {task.name}. Decisions are saved to {data.edits_path(s, task)}.")
    print("In the viewer, the timeline shows each frame's time in seconds.\n" + HELP)

    for n, (session, ep) in enumerate(episodes, 1):
        meta = session.meta
        action = column(episode_table(meta, ep), "action")
        duration = len(action) / meta.fps
        suggested = idle_trim(action, meta.fps)
        if suggested is None:
            suggestion = data.Edit(session.name, ep, delete="nothing moved")
            hint = "nothing moved: delete it"
        elif suggested[0] < 0.3 and suggested[1] > duration - 0.3:
            suggestion = data.Edit(session.name, ep)
            hint = "no idle time to cut: keep all"
        else:
            suggestion = data.Edit(session.name, ep, keep=suggested)
            hint = (f"keep {suggested[0]:.2f}-{suggested[1]:.2f} s (cuts {suggested[0]:.1f} s idle at the start, "
                    f"{duration - suggested[1]:.1f} s at the end)")
        print(f"[{n}/{len(episodes)}] session {session.name}, episode {ep}: {duration:.1f} s")
        print(f"  now:        {describe(edits.get((session.name, ep)))}")
        print(f"  suggestion: {hint}")
        if viewer:
            view(session.repo_id, session.dataset_root, ep)
        while True:
            result = parse(ask("  > "), duration, suggestion)
            if isinstance(result, data.Edit):
                edits[result.key] = result
                data.save_edits(s, task, edits)
                print(f"  saved: {describe(result)}\n")
                break
            if result == "v":
                view(session.repo_id, session.dataset_root, ep)
            elif result == "s":
                print()
                break
            elif result == "q":
                print(f"Stopped. Run `uv run so101 build {task.name}` to apply the edits.")
                return
            elif result in ("?", "h", "help"):
                print(HELP)
            else:
                print(f"  {result}")
    print(f"Done. Run `uv run so101 build {task.name}` to apply the edits.")
