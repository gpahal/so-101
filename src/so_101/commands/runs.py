"""`so101 runs`: summarise a task's serve runs and export their failure-fix pairs."""

import argparse
import sys

from so_101 import data, settings
from so_101.tasks import get_task


def _share(part: float | None) -> str:
    return "-" if part is None else f"{part:.0%}"


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


def runs_summary(metrics, pairs: int) -> str:
    """One line of run metrics, for example: 5 attempts: 3 succeeded, 1 failed, 1 unlabelled (75% of
    labelled, 95% interval 30-95%), 1 ignored. The policy drove 82% of 4.1 min of driving. 0.8
    interruptions per attempt. 2 failure-fix pairs."""
    m = metrics
    rate = ""
    if m.success_interval is not None:
        low, high = m.success_interval
        rate = f" ({_share(m.success_rate)} of labelled, 95% interval {low:.0%}-{high:.0%})"
    parts = [f"{m.attempts} attempt{'' if m.attempts == 1 else 's'}: {m.succeeded} succeeded, {m.failed} failed, "
             f"{m.unlabelled} unlabelled{rate}"]
    if m.ignored:
        parts.append(f"{m.ignored} ignored")
    if m.unsaved:
        parts.append(f"{m.unsaved} not saved")
    line = ", ".join(parts) + "."
    if m.policy_s + m.human_s:
        line += f" The policy drove {_share(m.policy_share)} of {_minutes(m.policy_s + m.human_s)} of driving."
    if m.attempts:
        line += f" {m.interruptions_per_attempt:.1f} interruptions per attempt."
    return line + f" {pairs} failure-fix pair{'' if pairs == 1 else 's'}."


def cmd_runs(s: settings.Settings, ns: argparse.Namespace, _: list[str]) -> None:
    from so_101.control import analysis

    task = get_task(ns.task)
    runs = data.runs(s, task)
    if not runs:
        sys.exit(f"No runs for {task.name} in {data.task_dir(s, task) / 'runs'}. Record some with "
                 f"`uv run so101 serve {task.name}`.")
    analyses = []
    for run in runs:
        a = analysis.analyse(run, s.control.blame_s)
        analyses.append(a)
        print(f"{run.name}: {runs_summary(analysis.Metrics.of([a]), len(a.failure_fixes))}")
        for note in a.notes:
            print(f"  note: {note}")
        for attempt in a.attempts:
            if attempt.episode is None:
                print(f"  attempt {attempt.number}: not saved (ended by {attempt.ended_by or 'a crash'})")
                continue
            secs = attempt.seconds(a.fps)
            outcome = attempt.outcome or "unlabelled"
            if attempt.outcome != attempt.recorded_outcome:
                outcome += " (labels.toml)"
            print(f"  attempt {attempt.number}: {outcome}, ended by {attempt.ended_by}; policy {secs['policy']:.1f} s, "
                  f"human {secs['human']:.1f} s, script {secs['script']:.1f} s; "
                  f"{len(attempt.interruptions)} interruption{'' if len(attempt.interruptions) == 1 else 's'}")
            for i in attempt.interruptions:
                why = f" ({i.reason})" if i.reason else ""
                suspect = f"frames {i.suspect[0]}-{i.suspect[1]} ({i.suspect_source})" if i.suspect else "none"
                print(f"    {i.frame / a.fps:5.1f} s (frame {i.frame}): {i.command}{why}, then {i.then}; suspect {suspect}")
    total = analysis.Metrics.of(analyses)
    pairs = sum(len(a.failure_fixes) for a in analyses)
    print(f"\nAll {len(analyses)} run{'' if len(analyses) == 1 else 's'}: {runs_summary(total, pairs)}")
    print(f"Ignored attempts count in nothing but the ignored total. Fix outcomes and suspect frames with `uv run so101 review --runs {task.name}`.")
    # Written even with no pairs, so pairs of attempts since ignored aren't left in it.
    out = s.outputs_dir / "runs" / task.name / "failure_fixes.jsonl"
    analysis.export_failure_fixes(analyses, out)
    print(f"Wrote {pairs} failure-fix pair{'' if pairs == 1 else 's'} to {out}.")
