"""`so101 review --runs`: step through the interruptions in a task's `serve` runs, fix attempt
outcomes and adjust the suspect frames before each interruption.

Each decision is written to the run's labels.toml straight away, as `review` writes demo decisions
to edits.toml. The run's dataset, events.jsonl and run.json are never changed. `so101 runs` and
`so101 build` read the labels through `control.analysis`.

Each run's files are read once (`analysis.read_run`), and the record is kept across the decisions:
after each one, the analysis is derived anew from the labels it wrote.
"""

from dataclasses import dataclass
from itertools import pairwise

from so_101 import data
from so_101.control import analysis
from so_101.control.analysis import Attempt, Interruption, Labels, RunAnalysis, RunRecord
from so_101.control.recorder import Phase
from so_101.review import ask, view
from so_101.settings import ConfigError, Settings
from so_101.tasks import Task

HELP = """\
  Enter               next: keep what is shown
  f START STOP        set the suspect frames to START..STOP, stop not included, e.g. `f 380 412`;
                      seconds work too, as the viewer shows them: `f 12.7s 13.7s`
  c                   clear the suspect frames: nothing before this interruption is suspect
  a                   back to the automatic suspect frames (the rewound range, or the last blame_s seconds)
  o OUTCOME           at the outcome prompt, set the attempt's outcome: succeeded, failed, ignored or
                      unlabelled, or the first letter, e.g. `o s`; `o i` ignores the attempt, and any
                      other outcome un-ignores it
  v                   open the attempt in the viewer again
  n                   skip to the next attempt
  q                   quit; decisions so far are saved
"""

OUTCOME_CHOICES = (*analysis.OUTCOMES, analysis.UNLABELLED)
CONTEXT_S = 3.0  # seconds of frames shown before the suspect range and after the interruption

PHASE_NAMES = {
    Phase.STARTING: "moving to start",
    Phase.AUTONOMOUS: "policy",
    Phase.PAUSED: "paused",
    Phase.HANDOVER: "handover",
    Phase.TELEOP: "teleop",
    Phase.REWINDING: "rewinding",
    Phase.RETURNING: "returning",
}


@dataclass(frozen=True)
class Suspect:
    """Set (`frames` a range), clear (`frames` None) or, with `auto`, drop the labels.toml override."""

    frames: analysis.Range | None
    auto: bool = False


def parse_outcome(word: str) -> str | None:
    """The outcome a word or a unique prefix of one names, or None."""
    matches = [o for o in OUTCOME_CHOICES if o.startswith(word.lower())]
    return matches[0] if word and len(matches) == 1 else None


def parse_frame(word: str, fps: float) -> int:
    if word.lower().endswith("s"):
        return round(float(word[:-1]) * fps)
    return int(word)


def parse(answer: str, frames: int, fps: float, at_interruption: bool) -> Suspect | str:
    """Turn an answer into a `Suspect`, an outcome (one of OUTCOME_CHOICES), a command ("", "v", "n",
    "q", "?"), or an error message starting with "!". `frames` is the attempt's episode length."""
    words = answer.split()
    if not words:
        return ""
    cmd, args = words[0].lower(), words[1:]
    if cmd in ("v", "n", "q", "?", "h", "help"):
        return cmd
    if cmd == "o":
        if at_interruption:
            return "!Set the outcome at the outcome prompt, after this attempt's interruptions: Enter goes on to the next."
        outcome = parse_outcome(args[0]) if len(args) == 1 else None
        if outcome is None:
            return f"!Usage: o OUTCOME, one of {', '.join(OUTCOME_CHOICES)}, e.g. `o failed` or `o f`."
        return outcome
    if cmd in ("f", "c", "a") and not at_interruption:
        return "!This attempt has no interruption left to set suspect frames on. Set its outcome with `o`, or Enter."
    if cmd == "c":
        return Suspect(None)
    if cmd == "a":
        return Suspect(None, auto=True)
    if cmd == "f":
        try:
            start, stop = (parse_frame(a, fps) for a in args)
        except (ValueError, OverflowError):  # OverflowError: `f infs 2s`
            return "!Usage: f START STOP, in frames (`f 380 412`) or seconds (`f 12.7s 13.7s`)."
        if not 0 <= start < stop <= frames:
            return f"!Need 0 <= START < STOP <= {frames}, the attempt's frames."
        return Suspect((start, stop))
    return f"!Unknown answer {answer!r}. Type ? for help."


def describe_outcome(attempt: Attempt) -> str:
    now = attempt.outcome or analysis.UNLABELLED
    recorded = attempt.recorded_outcome or analysis.UNLABELLED
    return now if now == recorded else f"{now} (labels.toml; recorded as {recorded})"


def describe_suspect(i: Interruption, fps: float) -> str:
    source = {"rewind": "the range the rewind undid", "blame": "the last blame_s seconds of policy frames",
              "labels": "set in labels.toml"}[i.suspect_source]
    if i.suspect is None:
        return f"none ({source})"
    a, b = i.suspect
    return f"frames {a}-{b}, {a / fps:.2f}-{b / fps:.2f} s, {(b - a) / fps:.1f} s ({source})"


def timeline(attempt: Attempt, i: Interruption, fps: float) -> list[str]:
    """The segments from CONTEXT_S before the suspect range to CONTEXT_S after the interruption, with
    the suspect frames split out and marked, one line each."""
    context = round(CONTEXT_S * fps)
    first = i.suspect[0] if i.suspect else i.frame
    lo, hi = max(0, first - context), min(attempt.frames, i.frame + context)
    cuts = {lo, hi, i.frame}
    if i.suspect:
        cuts |= {max(lo, i.suspect[0]), min(hi, i.suspect[1])}
    lines = []
    for segment in attempt.segments:
        a, b = max(segment.start, lo), min(segment.stop, hi)
        if a >= b:
            continue
        edges = sorted({a, b} | {c for c in cuts if a < c < b})
        for start, stop in pairwise(edges):
            mark = []
            if i.suspect and i.suspect[0] <= start and stop <= i.suspect[1]:
                mark.append("suspect")
            if start == i.frame:
                mark.append(f"<- {i.command}" + (f" ({i.reason})" if i.reason else ""))
            lines.append(f"    {start / fps:6.2f}-{stop / fps:6.2f} s  frames {start:>5}-{stop:<5} "
                         f"{PHASE_NAMES[segment.phase]:<15} {'  '.join(mark)}".rstrip())
    return lines


def show_interruption(attempt: Attempt, i: Interruption, n: int, fps: float) -> None:
    why = i.reason or "no reason given"
    print(f"  interruption {n}/{len(attempt.interruptions)} at {i.frame / fps:.2f} s (frame {i.frame}): "
          f"{i.command} ({why}), then {i.then}")
    print(f"  suspect: {describe_suspect(i, fps)}")
    print(*timeline(attempt, i, fps), sep="\n")


def refuse_unknown(run: data.Run, labels: Labels) -> Labels:
    """A save rewrites the whole labels.toml, so refuse a file with a top-level key review --runs
    doesn't know, rather than drop it. Returns `labels`."""
    if labels.unknown:
        keys = ", ".join(repr(key) for key in labels.unknown)
        raise ConfigError(f"{run.labels_path} has keys review --runs doesn't know: {keys}. A save rewrites the "
                          "whole file and would drop them, so review --runs saves nothing to it: fix or remove "
                          "them, then run review --runs again.")
    return labels


def apply(run: data.Run, decision: Suspect | str, attempt: Attempt, i: Interruption | None) -> Labels:
    """Write one decision, a `Suspect` or an outcome, to the run's labels.toml, keeping every other
    one, and return the labels as written."""
    labels = refuse_unknown(run, analysis.load_labels(run))  # again: the file may have changed since
    outcomes, suspects = dict(labels.outcomes), dict(labels.suspects)
    if isinstance(decision, str):
        outcomes[attempt.number] = None if decision == analysis.UNLABELLED else decision
    elif decision.auto:
        suspects.pop((attempt.number, i.frame), None)
    else:
        suspects[(attempt.number, i.frame)] = decision.frames
    labels = Labels(outcomes, suspects)
    analysis.save_labels(run, labels)
    return labels


def find(record: RunRecord, labels: Labels, blame_s: float, attempt: int) -> tuple[RunAnalysis, Attempt]:
    """The run's analysis with `labels`, and its attempt numbered `attempt`."""
    a = analysis.analyse_record(record, labels, blame_s)
    return a, next(x for x in a.attempts if x.number == attempt)


def review_runs(s: Settings, task: Task, only: str | None, everything: bool, viewer: bool) -> None:
    """Step through each saved attempt of each run: its interruptions, then its outcome. Without
    `everything`, attempts whose outcome labels.toml already sets are skipped. `only` is a run
    (YYYYmmdd-HHMMSS) or one attempt (RUN/N)."""
    blame_s = analysis.check_blame_s(s.control.blame_s)  # before any run is read
    runs = [r for r in data.runs(s, task) if only is None or only.split("/")[0] == r.name]
    todo: list[tuple[data.Run, int]] = []
    records: dict[str, RunRecord] = {}  # each run's files, read once
    # The Rerun viewer (lerobot-dataset-viz) loads every data file of a run, so one it can't read
    # stops it opening any of the run's attempts, silently.
    unviewable: set[str] = set()
    for run in runs:
        record = records[run.name] = analysis.read_run(run)
        labels = refuse_unknown(run, analysis.load_labels(run))  # before any review, not at its first save
        a = analysis.analyse_record(record, labels, blame_s)
        for note in a.notes:
            print(f"{run.name}: note: {note}")
        if record.skipped:
            unviewable.add(run.name)
            files = ", ".join(str(path.relative_to(run.path)) for path, _ in record.skipped)
            print(f"{run.name}: note: the viewer loads every data file of a run, so it can't open this run's attempts "
                  f"until {files} is removed or repaired; they're reviewed here without it.")
        for attempt in a.attempts:
            if attempt.episode is None:
                continue  # not saved: no frames to review
            if only is not None and "/" in only and only != f"{run.name}/{attempt.number}":
                continue
            if only is None and not everything and attempt.number in labels.outcomes:
                continue
            todo.append((run, attempt.number))
    if not todo:
        if not data.runs(s, task):
            print(f"No runs for {task.name}. Record some with `uv run so101 serve {task.name}`.")
        elif everything or only:
            print(f"No saved attempts for {task.name}{f' in {only}' if only else ''}.")
        else:
            print(f"No unreviewed attempts for {task.name}; --all shows every one.")
        return
    print(f"Reviewing {len(todo)} attempt{'' if len(todo) == 1 else 's'} of {task.name}'s runs. Decisions are saved to each run's "
          f"{analysis.LABELS_FILE}; the runs themselves are never changed.")
    print("In the viewer, the timeline shows each frame's time in seconds.\n" + HELP)

    for n, (run, number) in enumerate(todo, 1):
        a, attempt = find(records[run.name], analysis.load_labels(run), blame_s, number)
        fps = a.fps
        count = len(attempt.interruptions)
        print(f"[{n}/{len(todo)}] run {run.name}, attempt {number} (episode {attempt.episode}): "
              f"{attempt.frames / fps:.1f} s, ended by {attempt.ended_by}, "
              f"{count} interruption{'' if count == 1 else 's'}")
        print(f"  outcome: {describe_outcome(attempt)}")
        if viewer and run.name not in unviewable:
            view(run.repo_id, run.dataset_root, attempt.episode)

        k = 0  # the interruption being reviewed; == count is the attempt's outcome step
        shown = -1
        while k <= count:
            at_interruption = k < count
            if at_interruption and shown != k:
                show_interruption(attempt, attempt.interruptions[k], k + 1, fps)
                shown = k
            prompt = "  > " if at_interruption else f"  outcome {attempt.outcome or analysis.UNLABELLED}; Enter keeps it > "
            result = parse(ask(prompt), attempt.frames, fps, at_interruption)
            if isinstance(result, Suspect) or result in OUTCOME_CHOICES:
                i = attempt.interruptions[k] if at_interruption else None
                a, attempt = find(records[run.name], apply(run, result, attempt, i), blame_s, number)
                if isinstance(result, Suspect):
                    print(f"  saved: suspect {describe_suspect(attempt.interruptions[k], fps)}")
                    k += 1
                else:
                    print(f"  saved: outcome {describe_outcome(attempt)}")
                    if not at_interruption:
                        print()
                        break
                continue
            if result == "":
                if not at_interruption:
                    # Confirm the outcome, so the attempt counts as reviewed.
                    apply(run, attempt.outcome or analysis.UNLABELLED, attempt, None)
                    print(f"  saved: outcome {describe_outcome(attempt)}\n")
                    break
                k += 1
            elif result == "v" and run.name in unviewable:
                print(f"  The viewer can't open run {run.name}'s attempts: see its note above.")
            elif result == "v":
                view(run.repo_id, run.dataset_root, attempt.episode)
            elif result == "n":
                print()
                break
            elif result == "q":
                print(f"Stopped. `uv run so101 runs {task.name}` shows the runs with these labels.")
                return
            elif result in ("?", "h", "help"):
                print(HELP)
            else:
                print(f"  {result[1:]}")
    print(f"Done. `uv run so101 runs {task.name}` shows the runs with these labels.")

