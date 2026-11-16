"""The `so101-failbench` command: build the dataset, check the models answer, run, report."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from so_101.failbench import dataset, metrics, models, runner

DATASET = Path("outputs/failbench/dataset")
RUNS = Path("outputs/failbench/runs")
SERVE_RUNS = Path("data/block-container/runs")
CANDIDATES = Path("outputs/failbench/candidates.csv")
ENV_FILE = Path(".scratch/bench.env")


def load_env(path: Path) -> list[str]:
    """Read KEY=value lines into the environment, without overriding variables already set.
    Returns the names it set (never the values)."""
    if not path.exists():
        return []
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if value and not os.environ.get(name):
            os.environ[name] = value
            names.append(name)
    return names


def pick_models(names: str | None) -> list[models.Model]:
    if not names or names == "all":
        return list(models.MODELS.values())
    chosen = []
    for name in names.split(","):
        name = name.strip()
        matches = [m for m in models.MODELS.values() if name in (m.key, m.provider, m.group)]
        if not matches:
            sys.exit(f"Unknown model {name!r}. `so101-failbench models` lists them.")
        chosen += [m for m in matches if m not in chosen]
    return chosen


def available(ms: list[models.Model], log=print) -> list[models.Model]:
    ok = [m for m in ms if m.has_key()]
    for m in ms:
        if not m.has_key():
            log(f"Skipping {m.key}: {m.env} isn't set (in the environment or {ENV_FILE}).")
    return ok


def cmd_build(a) -> None:
    n = dataset.build(Path(a.dataset), [Path(a.runs)], Path(a.labels))
    samples = dataset.load(Path(a.dataset))
    wrong = sum(s.wrong for s in samples)
    print(f"Built {n} samples in {a.dataset}: {wrong} wrong, {n - wrong} ok.")


def cmd_candidates(a) -> None:
    from so_101.failbench import candidates
    from so_101.grasp import thresholds
    from so_101.supervisor import program

    if not Path(a.runs).is_dir():
        sys.exit(f"No serve runs at {a.runs}. Record some with `uv run so101 serve`.")
    rows = candidates.from_runs(Path(a.runs), program.load(Path(a.program) if a.program else None),
                                thresholds.load(Path(a.thresholds) if a.thresholds else None))
    candidates.write(rows, Path(a.out))
    counts = {}
    for row in rows:
        counts[row["moment"]] = counts.get(row["moment"], 0) + 1
    print(f"Wrote {len(rows)} candidates to {a.out}: " + ", ".join(f"{n} {m}" for m, n in sorted(counts.items()))
          + ". Label them, then: so101-failbench build --labels " + str(a.out))


def cmd_models(a) -> None:
    for m in models.MODELS.values():
        key = "key set" if m.has_key() else f"needs {m.env}"
        print(f"{m.key:28} {m.provider:18} {m.model_id:36} ${m.price_in:g}/${m.price_out:g} per M tokens  "
              f"~${m.estimate:.4f}/sample  {key}")


def cmd_check(a) -> None:
    """One sample of each label per model: does every model answer, how fast, at what cost."""
    samples = dataset.load(Path(a.dataset))
    ms = available(pick_models(a.models))
    picks = [next(s for s in samples if s.wrong), next(s for s in samples if not s.wrong)]
    variant = runner.Variant()
    results = []
    for m in ms:
        for s in picks:
            started = time.monotonic()
            try:
                r = runner.call_with_retries(lambda: m.adapter().predict(runner.images_for(Path(a.dataset), s, variant), s.moment), attempts=2)
                status = "OK" if r.get("wrong") is not None else "NO ANSWER"
                detail = f"wrong={r.get('wrong')} p_wrong={r.get('p_wrong')} category={r.get('category')} ${r.get('cost_usd') or 0:.4f}"
            except Exception as e:
                status, detail = "FAIL", f"{type(e).__name__}: {e}"[:300]
            secs = time.monotonic() - started
            results.append({"model": m.key, "sample": s.id, "label": s.label, "status": status, "secs": round(secs, 2), "detail": detail})
            print(f"{status:9} {m.key:28} {s.label:5} {secs:5.1f} s  {detail}", flush=True)
    failed = sorted({r["model"] for r in results if r["status"] != "OK"})
    print(f"\n{len(ms) - len(failed)} of {len(ms)} models answered both samples." + (f" Failed: {', '.join(failed)}" if failed else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))


def cmd_run(a) -> None:
    root = Path(a.dataset)
    samples = dataset.subset(dataset.load(root), a.limit, a.seed)
    ms = available(pick_models(a.models))
    if not ms:
        sys.exit("No model has its API key set.")
    out = Path(a.out) if a.out else RUNS / (a.name or time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    variant = runner.Variant(tuple(a.cameras.split("+")), not a.no_moment)
    cache = runner.Cache(out / "predictions.jsonl")
    budget = runner.Budget(a.max_cost)
    print(f"{len(ms)} models x {len(samples)} samples, variant {variant.name}, cap ${a.max_cost:.2f}, into {out}")
    runner.run(ms, samples, root, cache, variant, budget, workers=a.workers, retry_errors=a.retry_errors)
    write_report(out, samples, ms, variant, cache, extra=[f"This invocation spent ${budget.spent:.2f}."])


def write_report(out: Path, samples, ms, variant, cache, extra=()) -> None:
    scores = [metrics.score(m.key, samples, cache.for_model(m.key, variant.name)) for m in ms]
    notes = [f"Cameras: {' and '.join(variant.cameras)}. Moment given to the model: {'yes' if variant.moment_hint else 'no'}.",
             *extra]
    (out / "report.md").write_text(metrics.report(f"Failure detection: {out.name}", samples, scores, notes))
    with open(out / "scores.json", "w") as f:
        json.dump([s.__dict__ for s in scores], f, indent=2, default=str)
    print(f"\nReport: {out / 'report.md'}")
    for sc in sorted(scores, key=lambda s: -(s.balanced_accuracy if s.balanced_accuracy == s.balanced_accuracy else -1)):
        print(f"  {sc.model:28} balanced accuracy {metrics._f(sc.balanced_accuracy, pct=True):>4}  "
              f"caught {metrics._f(sc.recall_wrong, pct=True):>4}  ok passed {metrics._f(sc.recall_ok, pct=True):>4}  "
              f"errors {sc.errors}  p50 {metrics._f(sc.latency_p50, 1)} s  ${sc.cost_total:.2f}")


def cmd_report(a) -> None:
    out = Path(a.run)
    samples = dataset.subset(dataset.load(Path(a.dataset)), a.limit, a.seed)
    cache = runner.Cache(out / "predictions.jsonl")
    keys = sorted({k[0] for k in cache.rows})
    ms = [models.MODELS[k] for k in keys if k in models.MODELS]
    variant = runner.Variant(tuple(a.cameras.split("+")), not a.no_moment)
    answered = {s for (m, v, s) in cache.rows if v == variant.name}
    samples = [s for s in samples if s.id in answered] if a.answered_only else samples
    write_report(out, samples, ms, variant, cache)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="so101-failbench", description=(
        "Benchmark vision models on telling when something has gone wrong in a frame of a block-container "
        "attempt. Run it from the repo root. API keys come from the environment or .scratch/bench.env: "
        "OPENROUTER_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, TYPESAFE_API_KEY. Models whose key is missing "
        "are skipped."))
    p.add_argument("--env-file", default=str(ENV_FILE), help="KEY=value file with API keys (default %(default)s)")
    p.add_argument("--dataset", default=str(DATASET), help="dataset folder (default %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="decode the labelled frames from the recordings into the dataset folder")
    b.add_argument("--runs", default=str(SERVE_RUNS), help="so101 serve and run runs (default %(default)s)")
    b.add_argument("--labels", default=str(dataset.LABELS), help="the labels file (default: the one in the package)")
    b.set_defaults(fn=cmd_build)

    cd = sub.add_parser("candidates", help="pick frames to label from so101 serve runs, where the supervisor would check")
    cd.add_argument("--runs", default=str(SERVE_RUNS), help="so101 serve and run runs (default %(default)s)")
    cd.add_argument("--program", help="the supervisor's program file, for its check delays (default: the one in the package)")
    cd.add_argument("--thresholds", help="the grasp thresholds file (default: the one in the package)")
    cd.add_argument("--out", default=str(CANDIDATES), help="where to write the rows (default %(default)s)")
    cd.set_defaults(fn=cmd_candidates)

    sub.add_parser("models", help="list the models, their prices and whether their key is set").set_defaults(fn=cmd_models)

    c = sub.add_parser("check", help="ask each model about one wrong and one ok sample, to see that it answers")
    c.add_argument("--models", help="comma-separated keys, providers or groups (default: all)")
    c.add_argument("--json", help="also write the results here")
    c.set_defaults(fn=cmd_check)

    def common(x):
        x.add_argument("--limit", type=int, help="use a fixed, stratified subset of about this many samples")
        x.add_argument("--seed", type=int, default=0, help="which subset --limit picks (default %(default)s)")
        x.add_argument("--cameras", default="top+wrist", choices=["top+wrist", "top", "wrist"],
                       help="which frames each model sees (default %(default)s)")
        x.add_argument("--no-moment", action="store_true", help="don't tell the model which moment of the attempt it is")

    r = sub.add_parser("run", help="run models over the dataset; answers are cached so a rerun resumes")
    r.add_argument("--models", help="comma-separated keys, providers or groups (default: all)")
    r.add_argument("--max-cost", type=float, default=5.0, help="stop before this invocation spends more, in USD (default %(default)s)")
    r.add_argument("--workers", type=int, default=4, help="parallel calls per provider (default %(default)s)")
    r.add_argument("--name", help="run folder name under outputs/failbench/runs (default: the time)")
    r.add_argument("--out", help="run folder (overrides --name)")
    r.add_argument("--retry-errors", action="store_true", help="ask again for samples whose answer was an error")
    common(r)
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="rewrite a run's report from its cached answers")
    rp.add_argument("run", help="the run folder")
    rp.add_argument("--answered-only", action="store_true", help="score only samples some model answered")
    common(rp)
    rp.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    load_env(Path(a.env_file))
    a.fn(a)


if __name__ == "__main__":
    main()
