"""`so101 review`: step through recorded episodes and decide what to keep, trim or delete.

The decisions are taken on the reviewer page (`web/review.py`), or with `--terminal` at the text
prompts below. Both write through `save`, from edits built by `keep_all`, `delete` and `keep_range`,
so each writes the same edits.toml. Each decision is written straight away; the recordings
themselves are never changed. `so101 build` applies the edits.
"""

from dataclasses import dataclass

import numpy as np

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
  s            skip for now (no decision)
  q            quit; decisions so far are saved
"""


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


@dataclass(frozen=True)
class Demo:
    """One recorded episode to review."""

    session: data.Session
    episode: int

    @property
    def id(self) -> str:
        """How `--episode` and the reviewer page name it: SESSION/N."""
        return f"{self.session.name}/{self.episode}"

    @property
    def key(self) -> tuple[str, int]:
        return (self.session.name, self.episode)


@dataclass(frozen=True)
class Suggestion:
    """What `review` suggests for an episode `duration` seconds long: `edit`, described by `hint`."""

    duration: float
    edit: data.Edit
    hint: str


def demos(s: Settings, task: Task, only: str | None, everything: bool) -> list[Demo]:
    """The task's episodes to review, oldest first: those edits.toml has no decision for, or with
    `everything` all of them. `only` is a session (YYYYmmdd-HHMMSS) or one episode (SESSION/N), shown
    whether reviewed or not."""
    edits = data.load_edits(s, task)
    found = [
        Demo(session, ep)
        for session in data.sessions(s, task)
        if session.meta
        for ep in range(session.episodes)
        if (only is None or only in (session.name, f"{session.name}/{ep}"))
    ]
    if only is None and not everything:
        found = [demo for demo in found if demo.key not in edits]
    return found


def nothing_left(task: Task, only: str | None, everything: bool) -> str:
    """What to say when `demos` finds none."""
    if everything or only:
        return f"No episodes for {task.name}."
    return f"No unreviewed episodes for {task.name}; --all shows every one."


def suggest(demo: Demo, action: np.ndarray, fps: float) -> Suggestion:
    """The suggestion for `demo`, from its `action` column: cut the idle time at its start and end
    (`validate.idle_trim`), keep it all when there's none to cut, or delete it when nothing moved."""
    session, ep = demo.key
    duration = len(action) / fps
    suggested = idle_trim(action, fps)
    if suggested is None:
        return Suggestion(duration, data.Edit(session, ep, delete="nothing moved"), "nothing moved: delete it")
    if suggested[0] < 0.3 and suggested[1] > duration - 0.3:
        return Suggestion(duration, data.Edit(session, ep), "no idle time to cut: keep all")
    return Suggestion(duration, data.Edit(session, ep, keep=suggested),
                      f"keep {suggested[0]:.2f}-{suggested[1]:.2f} s (cuts {suggested[0]:.1f} s idle at the start, "
                      f"{duration - suggested[1]:.1f} s at the end)")


def read(demo: Demo):
    """The episode's rows (`dataset_io.episode_table`)."""
    return episode_table(demo.session.meta, demo.episode)


def keep_all(suggestion: Suggestion) -> data.Edit:
    return data.Edit(suggestion.edit.session, suggestion.edit.episode)


def delete(suggestion: Suggestion, reason: str) -> data.Edit:
    return data.Edit(suggestion.edit.session, suggestion.edit.episode, delete=reason)


def keep_range(suggestion: Suggestion, start: float, end: float) -> data.Edit | str:
    """Keep only `start`..`end` seconds, `end` capped at the episode's length, or an error message
    string when that isn't a range a build can train on."""
    end = min(end, suggestion.duration)
    if not 0 <= start < end or end - start < MIN_EPISODE_S:
        return f"Need 0 <= START < END <= {suggestion.duration:.2f}, at least {MIN_EPISODE_S} s apart."
    return data.Edit(suggestion.edit.session, suggestion.edit.episode, keep=(round(start, 2), round(end, 2)))


def save(s: Settings, task: Task, edit: data.Edit) -> None:
    """Write one decision to edits.toml, keeping every other one, read again in case it changed."""
    edits = data.load_edits(s, task)
    edits[edit.key] = edit
    data.save_edits(s, task, edits)


def parse(answer: str, suggestion: Suggestion) -> data.Edit | str:
    """Turn an answer into an edit, a command ("s", "q", "?"), or an error message string."""
    words = answer.split()
    if not words:
        return suggestion.edit
    cmd, args = words[0].lower(), words[1:]
    if cmd in ("s", "q", "?", "h", "help"):
        return cmd
    if cmd == "k":
        return keep_all(suggestion)
    if cmd == "d":
        return delete(suggestion, " ".join(args))
    if cmd == "t":
        try:
            start, end = (float(a) for a in args)
        except ValueError:
            return "Usage: t START END, in seconds, e.g. t 1.2 14.5"
        return keep_range(suggestion, start, end)
    return f"Unknown answer {answer!r}. Type ? for help."


def review(s: Settings, task: Task, only: str | None, everything: bool) -> None:
    """`review --terminal`: each episode's suggestion, then a prompt for the decision."""
    episodes = demos(s, task, only, everything)
    if not episodes:
        print(nothing_left(task, only, everything))
        return
    print(f"Reviewing {len(episodes)} episodes of {task.name}. Decisions are saved to {data.edits_path(s, task)}.")
    print(HELP)

    for n, demo in enumerate(episodes, 1):
        meta = demo.session.meta
        suggestion = suggest(demo, column(read(demo), "action"), meta.fps)
        print(f"[{n}/{len(episodes)}] session {demo.session.name}, episode {demo.episode}: {suggestion.duration:.1f} s")
        print(f"  now:        {describe(data.load_edits(s, task).get(demo.key))}")
        print(f"  suggestion: {suggestion.hint}")
        while True:
            result = parse(ask("  > "), suggestion)
            if isinstance(result, data.Edit):
                save(s, task, result)
                print(f"  saved: {describe(result)}\n")
                break
            if result == "s":
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
