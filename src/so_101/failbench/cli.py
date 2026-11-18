"""The `so101-failbench` command: pick candidates, build the dataset, check the models answer, run, report."""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from so_101 import settings
from so_101.failbench import dataset, metrics, runner
from so_101.vision import models
from so_101.vision.env import load_env
from so_101.vision.prompt import TASK_NAME


@dataclass(frozen=True)
class Paths:
    """Where the command reads and writes by default, from so101.toml: the runs in its data folder, the keys
    in `.scratch/bench.env` beside it, and everything it writes under `outputs/failbench/`."""

    runs: Path
    env_file: Path
    candidates: Path
    dataset: Path
    results: Path

    @classmethod
    def of(cls, s: settings.Settings) -> "Paths":
        from so_101.data import Run

        out = s.outputs_dir / "failbench"
        return cls(runs=s.data_dir / TASK_NAME / Run.FOLDER, env_file=s.path.parent / ".scratch" / "bench.env",
                   candidates=out / "candidates.csv", dataset=out / "dataset", results=out / "runs")


def pick_models(names: str | None) -> list[models.Model]:
    """The models `--models` names: keys, `group:NAME` or `provider:NAME`, comma-separated, or `all`. A bare
    name that is only a group or only a provider means that one; one that is both must say which."""
    if not names or names == "all":
        return list(models.MODELS.values())
    chosen = []
    for name in names.split(","):
        name = name.strip()
        kind, _, value = name.rpartition(":")
        groups = [m for m in models.MODELS.values() if m.group == value]
        providers = [m for m in models.MODELS.values() if m.provider == value]
        if kind == "group":
            matches = groups
        elif kind == "provider":
            matches = providers
        elif kind:
            sys.exit(f"{name!r}: say group:NAME or provider:NAME, or give a model's name.")
        elif value in models.MODELS:
            matches = [models.MODELS[value]]
        elif groups and providers and groups != providers:
            sys.exit(f"{value!r} is both a group ({', '.join(m.key for m in groups)}) and a provider "
                     f"({', '.join(m.key for m in providers)}): say group:{value} or provider:{value}.")
        else:
            matches = groups or providers
        if not matches:
            sys.exit(f"Unknown model {name!r}. `so101-failbench models` lists them.")
        chosen += [m for m in matches if m not in chosen]
    return chosen


def max_cost(text: str) -> float:
    """`--max-cost` as a number of dollars above 0. NaN, infinity and 0 or less are refused, because the cap
    compares with it before each sample: NaN and infinity would turn the cap off, and 0 or less would ask nothing."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a number of dollars.") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a spending cap: give a number of dollars above 0, "
                                         "such as 5.")
    return value


def available(ms: list[models.Model], env_file: Path, log=print) -> list[models.Model]:
    ok = [m for m in ms if m.has_key()]
    for m in ms:
        if not m.has_key():
            log(f"Skipping {m.key}: {m.env} isn't set (in the environment or {env_file}).")
    return ok


def cmd_build(a) -> None:
    labels = Path(a.labels)
    if not labels.is_file():
        sys.exit(f"No labels file at {labels}. Write one with `so101-failbench candidates`, fill in its labels, "
                 "then build.")
    try:
        n = dataset.build(Path(a.dataset), [Path(a.runs)], labels)
    except dataset.MissingRecordings as e:
        sys.exit(str(e))
    samples = dataset.load(Path(a.dataset))
    wrong = sum(s.wrong for s in samples)
    left = dataset.unlabelled(labels)
    print(f"Built {n} samples in {a.dataset}: {wrong} wrong, {n - wrong} ok."
          + (f" {left} rows of {labels} have no label yet and were left out." if left else ""))


def cmd_candidates(a) -> None:
    from so_101.failbench import candidates
    from so_101.grasp import thresholds
    from so_101.supervisor import program

    if not Path(a.runs).is_dir():
        sys.exit(f"No serve or run runs at {a.runs}. Record some with `uv run so101 serve` or `uv run so101 run`.")
    rows = candidates.from_runs(Path(a.runs), program.load(Path(a.program) if a.program else None),
                                thresholds.load(Path(a.thresholds) if a.thresholds else None))
    stale = candidates.unpicked(rows, Path(a.out))
    kept = candidates.write(rows, Path(a.out))
    counts = {}
    for row in rows:
        counts[row["moment"]] = counts.get(row["moment"], 0) + 1
    print(f"Wrote {len(rows)} candidates to {a.out}: " + ", ".join(f"{n} {m}" for m, n in sorted(counts.items()))
          + (f". It kept the {kept} labels already there" if kept else "")
          + ". Label them, then: so101-failbench build --labels " + str(a.out))
    if stale:
        print(f"{stale} labelled row{'' if stale == 1 else 's'} at the end of the file {'is' if stale == 1 else 'are'} "
              "not picked by the current rules: a moment the supervisor doesn't check (a check not yet sent when a "
              "pause or resume came, or for a close a newer one replaced), or a run that's gone. Delete any that "
              "shouldn't be scored.", flush=True)


def cmd_models(a) -> None:
    for m in models.MODELS.values():
        key = "key set" if m.has_key() else f"needs {m.env}"
        print(f"{m.key:28} {m.provider:18} {m.group:9} {m.model_id:36} ${m.price_in:g}/${m.price_out:g} per M tokens  "
              f"~${m.estimate:.4f}/sample  {key}")


def cmd_check(a) -> None:
    """One sample of each label per model: does every model answer, how fast, at what cost."""
    samples = dataset.load(Path(a.dataset))
    ms = available(pick_models(a.models), Path(a.env_file))
    picks = [s for s in (next((s for s in samples if s.wrong), None), next((s for s in samples if not s.wrong), None)) if s]
    variant = runner.Variant()
    results = []
    for m in ms:
        for s in picks:
            started = time.monotonic()
            try:
                r = runner.call_with_retries(lambda: m.adapter().predict(runner.images_for(Path(a.dataset), s, variant), s.moment), attempts=2)
                status = "OK" if r.get("wrong") is not None else "NO ANSWER"
                cost = f"${r['cost_usd']:.4f}" if r.get("cost_usd") is not None else "no usage reported"
                detail = f"wrong={r.get('wrong')} p_wrong={r.get('p_wrong')} category={r.get('category')} {cost}"
            except Exception as e:
                status, detail = "FAIL", f"{type(e).__name__}: {e}"[:300]
            secs = time.monotonic() - started
            results.append({"model": m.key, "sample": s.id, "label": s.label, "status": status, "secs": round(secs, 2), "detail": detail})
            print(f"{status:9} {m.key:28} {s.label:5} {secs:5.1f} s  {detail}", flush=True)
    failed = sorted({r["model"] for r in results if r["status"] != "OK"})
    print(f"\n{len(ms) - len(failed)} of {len(ms)} models answered all {len(picks)} samples." + (f" Failed: {', '.join(failed)}" if failed else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))


def _sure(a) -> tuple[float, float]:
    """The supervisor's (sure_ok_below, sure_wrong_above), from its program."""
    from so_101.supervisor import program

    j = program.load(Path(a.program) if a.program else None).judges
    return j.sure_ok_below, j.sure_wrong_above


def cmd_run(a) -> None:
    root = Path(a.dataset)
    samples = dataset.subset(dataset.load(root), a.limit, a.seed)
    ms = available(pick_models(a.models), Path(a.env_file))
    if not ms:
        sys.exit("No model has its API key set.")
    out = Path(a.out) if a.out else Path(a.results) / (a.name or time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    variant = runner.Variant(tuple(a.cameras.split("+")), not a.no_moment)
    sure = _sure(a)
    cache = runner.Cache(out / "predictions.jsonl")
    budget = runner.Budget(a.max_cost)
    print(f"{len(ms)} models x {len(samples)} samples, variant {variant.name}, cap ${a.max_cost:.2f}, into {out}")
    result = runner.run(ms, samples, root, cache, variant, budget, workers=a.workers, retry_errors=a.retry_errors)
    if result.stale:
        sys.exit(f"{out} holds answers for {', '.join(result.stale)} asked under another prompt, adapter or model settings: "
                 "run into a new --out folder, so the report compares answers to one question.")
    if len(result.skipped) == len(ms) and not cache.rows:
        sys.exit("No model could be asked: " + "; ".join(f"{key}: {why}" for key, why in result.skipped.items()))
    extra = [f"This invocation spent ${budget.spent:.2f}."]
    if result.interrupted:
        extra.append("Stopped by Ctrl+C before every sample was asked; running the same command again asks the rest.")
    elif result.capped:
        extra.append(f"Stopped at the spending cap of ${budget.max_cost:.2f} before every sample was asked.")
    extra += [f"Left out: {key}, {why}." for key, why in result.skipped.items()]
    write_report(out, samples, [m for m in ms if m.key not in result.skipped], variant, cache, sure, extra)


def write_report(out: Path, samples, ms, variant, cache, sure, extra=()) -> None:
    predictions = {m.key: cache.for_model(m.key, variant.name) for m in ms}
    scored, scores = metrics.score_all(samples, predictions, sure)
    notes = [f"Cameras: {' and '.join(variant.cameras)}. Moment given to the model: {'yes' if variant.moment_hint else 'no'}.",
             *metrics.coverage(samples, scored, predictions), *extra]
    (out / "report.md").write_text(metrics.report(f"Failure detection: {out.name}", scored, scores, notes, sure))
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
    variant = runner.Variant(tuple(a.cameras.split("+")), not a.no_moment)
    asked = {m for (m, v, _) in cache.rows if v == variant.name}
    ms = [m for m in pick_models(a.models) if m.key in asked]
    if not ms:
        sys.exit(f"{out / 'predictions.jsonl'} has no answers for variant {variant.name} from these models.")
    write_report(out, samples, ms, variant, cache, _sure(a))


def main(argv=None) -> None:
    try:
        paths = Paths.of(settings.load())
    except settings.ConfigError as e:
        sys.exit(str(e))
    p = argparse.ArgumentParser(prog="so101-failbench", allow_abbrev=False, description=(
        f"Benchmark vision models on telling when something has gone wrong in a frame of a {TASK_NAME} "
        f"attempt. API keys come from the environment or {paths.env_file}: OPENROUTER_API_KEY, GEMINI_API_KEY, "
        "OPENAI_API_KEY, TYPESAFE_API_KEY. Models whose key is missing are skipped. check and run bill those keys "
        "and send the dataset's frames to the providers."))
    p.add_argument("--env-file", default=str(paths.env_file), help="KEY=value file with API keys (default %(default)s)")
    p.add_argument("--dataset", default=str(paths.dataset), help="dataset folder (default %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def command(name, help, **kwargs):
        return sub.add_parser(name, help=help, description=help[0].upper() + help[1:] + ".", allow_abbrev=False, **kwargs)

    model_help = ("comma-separated model names, group:NAME or provider:NAME; a bare group or provider name works "
                  "when it isn't both (default: all)")

    b = command("build", "decode the labelled frames from the runs into the dataset folder; rows without a label are left out")
    b.add_argument("--runs", default=str(paths.runs), help="so101 serve and run runs (default %(default)s)")
    b.add_argument("--labels", default=str(paths.candidates), help="the labels file (default %(default)s)")
    b.set_defaults(fn=cmd_build)

    cd = command("candidates", "pick frames to label from so101 serve and run runs, where the supervisor would "
                 "check; labels already in the file are kept, by id")
    cd.add_argument("--runs", default=str(paths.runs), help="so101 serve and run runs (default %(default)s)")
    cd.add_argument("--program", help="the supervisor's program file, for its check delays (default: the one in the package)")
    cd.add_argument("--thresholds", help="the grasp thresholds file (default: the one in the package)")
    cd.add_argument("--out", default=str(paths.candidates), help="where to write the rows (default %(default)s)")
    cd.set_defaults(fn=cmd_candidates)

    command("models", "list the models, their provider, group and prices, and whether their key is set").set_defaults(fn=cmd_models)

    per_sample = sum(m.estimate for m in models.MODELS.values())
    c = command("check", "ask each model about one wrong and one ok sample, to see that it answers. It bills each "
                f"model's API key, about ${2 * per_sample:.2f} for all of them, and sends those frames to the providers")
    c.add_argument("--models", help=model_help)
    c.add_argument("--json", help="also write the results here")
    c.set_defaults(fn=cmd_check)

    def common(x):
        x.add_argument("--limit", type=int, help="use a fixed, stratified subset of about this many samples")
        x.add_argument("--seed", type=int, default=0, help="which subset --limit picks (default %(default)s)")
        x.add_argument("--cameras", default="top+wrist", choices=["top+wrist", "top", "wrist"],
                       help="which frames each model sees (default %(default)s)")
        x.add_argument("--no-moment", action="store_true", help="don't tell the model which moment of the attempt it is")
        x.add_argument("--program", help="the supervisor's program file, for the thresholds the report scores at "
                       "(default: the one in the package)")
        x.add_argument("--models", help=model_help)

    r = command("run", "run models over the dataset; answers are cached so a rerun resumes. It bills each model's "
                "API key, up to --max-cost, and sends the frames to the providers")
    r.add_argument("--max-cost", type=max_cost, default=5.0, help="stop before this invocation spends more, in USD, admitting whole samples so each one asked "
                        "is asked of every model (default %(default)s)")
    r.add_argument("--workers", type=int, default=4, help="parallel calls per provider (default %(default)s)")
    r.add_argument("--name", help=f"run folder name under {paths.results} (default: the time)")
    r.add_argument("--out", help="run folder (overrides --name)")
    r.add_argument("--retry-errors", action="store_true", help="ask again for samples whose answer was an error")
    r.set_defaults(fn=cmd_run, results=str(paths.results))
    common(r)

    rp = command("report", "rewrite a run's report from its cached answers, scoring every model on the samples all "
                 "of them were asked about")
    rp.add_argument("run", help="the run folder")
    common(rp)
    rp.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    load_env(Path(a.env_file))
    a.fn(a)


if __name__ == "__main__":
    main()
