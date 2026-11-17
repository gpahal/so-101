"""Scores for each model and the Markdown report.

A prediction is right when its `wrong` matches the label. The headline is balanced accuracy, the mean
of recall on wrong samples and recall on ok samples, because the dataset has more of one than the other.
Missing a failure (a false "ok") and crying wolf (a false "wrong") are reported separately because they
cost different things: a missed failure leaves a dropped block unhandled, a false alarm wastes a retry.
A prediction that errored or refused counts as wrong in `balanced_accuracy` and is counted in `errors`;
`answered_balanced_accuracy` scores only the samples the model answered. A sample the model was never asked
about (a run the cap or Ctrl+C stopped) is neither: it's counted in `unasked` and left out, and
`score_all` scores every model on the samples all of them were asked about, so a capped model isn't
marked down for what it wasn't asked.

With the supervisor's thresholds (`sure`, its program's `sure_ok_below` and `sure_wrong_above`), three
more say how a model would do as its judge: the share of answers it would call unsure, the share of its
sure-wrong calls that were wrong (a sure wrong after a close plays the attempt back), and the share of
its sure-ok calls that were in fact wrong (failures the supervisor would let pass).
"""

import math
import random
from dataclasses import dataclass

from so_101.vision import prompt


@dataclass
class Scores:
    model: str
    n: int  # samples scored: those the model was asked about
    errors: int
    balanced_accuracy: float
    ba_low: float  # 95% bootstrap interval
    ba_high: float
    answered_balanced_accuracy: float
    recall_wrong: float  # share of wrong samples flagged
    recall_ok: float  # share of ok samples passed
    precision_wrong: float
    auroc: float | None
    ece: float | None
    category_accuracy: float  # on wrong samples the model flagged as wrong
    by_moment: dict[str, float]  # accuracy per moment
    by_category: dict[str, float]  # recall per wrong category
    latency_p50: float
    latency_p95: float
    cost_total: float
    cost_per_sample: float
    unasked: int = 0  # samples given but never asked about, left out of every score
    share_unsure: float = math.nan  # at the supervisor's thresholds, of the answers with a probability
    precision_sure_wrong: float = math.nan  # of the sure-wrong calls, the share that were wrong
    missed_in_sure_ok: float = math.nan  # of the sure-ok calls, the share that were wrong


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return math.nan
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def _ba(pairs: list[tuple[bool, bool | None]]) -> float:
    pos = [p for t, p in pairs if t]
    neg = [p for t, p in pairs if not t]
    if not pos or not neg:
        return math.nan
    return (sum(p is True for p in pos) / len(pos) + sum(p is False for p in neg) / len(neg)) / 2


def auroc(truth: list[bool], score: list[float]) -> float | None:
    pos = [s for t, s in zip(truth, score) if t]
    neg = [s for t, s in zip(truth, score) if not t]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def ece(truth: list[bool], prob: list[float], bins: int = 10) -> float | None:
    if not truth:
        return None
    total = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(prob) if (b / bins <= p < (b + 1) / bins) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(prob[i] for i in idx) / len(idx)
            acc = sum(truth[i] for i in idx) / len(idx)
            total += len(idx) / len(truth) * abs(conf - acc)
    return total


def _share(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def score(model: str, samples: list, predictions: dict, sure: tuple[float, float] | None = None,
          seed: int = 0) -> Scores:
    """`predictions` maps sample id to a prediction dict (an adapter's, as the runner caches it) for this
    model. Samples it has none for are unasked: counted, not scored. `sure` is the supervisor's
    (sure_ok_below, sure_wrong_above)."""
    unasked = sum(s.id not in predictions for s in samples)
    samples = [s for s in samples if s.id in predictions]
    pairs, probs, truth_p, lat, cost = [], [], [], [], 0.0
    errors = 0
    cat_hits = cat_total = 0
    moment_hits: dict[str, list[bool]] = {}
    cat_recall: dict[str, list[bool]] = {}
    for s in samples:
        p = predictions.get(s.id)
        said = None if p is None or p.get("error") else p.get("wrong")
        if said is None:
            errors += 1
        pairs.append((s.wrong, said))
        moment_hits.setdefault(s.moment, []).append(said == s.wrong)
        if s.wrong:
            cat_recall.setdefault(s.category, []).append(said is True)
            if said is True:
                cat_total += 1
                cat_hits += prompt.FAMILIES.get(p.get("category"), "other") == prompt.FAMILIES[s.category]
        if p and not p.get("error"):
            if p.get("p_wrong") is not None:
                probs.append(float(p["p_wrong"]))
                truth_p.append(s.wrong)
            if p.get("latency_s") is not None:
                lat.append(float(p["latency_s"]))
        if p:
            cost += float(p.get("cost_usd") or 0)
    ba = _ba(pairs)
    rng = random.Random(seed)
    boots = sorted(_ba([rng.choice(pairs) for _ in pairs]) for _ in range(1000))
    boots = [b for b in boots if not math.isnan(b)]
    answered = [(t, p) for t, p in pairs if p is not None]
    pos = [p for t, p in pairs if t]
    neg = [p for t, p in pairs if not t]
    flagged = [t for t, p in pairs if p is True]
    unsure = sure_wrong = sure_ok = []
    if sure is not None:
        ok_below, wrong_above = sure
        unsure = [ok_below < p < wrong_above for p in probs]
        sure_wrong = [t for t, p in zip(truth_p, probs, strict=True) if p >= wrong_above]
        sure_ok = [t for t, p in zip(truth_p, probs, strict=True) if p <= ok_below]
    return Scores(
        model=model, n=len(samples), errors=errors, balanced_accuracy=ba,
        ba_low=_pct(boots, 0.025), ba_high=_pct(boots, 0.975),
        answered_balanced_accuracy=_ba(answered),
        recall_wrong=sum(p is True for p in pos) / len(pos) if pos else math.nan,
        recall_ok=sum(p is False for p in neg) / len(neg) if neg else math.nan,
        precision_wrong=sum(flagged) / len(flagged) if flagged else math.nan,
        auroc=auroc(truth_p, probs) if len(set(probs)) > 1 else None,
        ece=ece(truth_p, probs), category_accuracy=cat_hits / cat_total if cat_total else math.nan,
        by_moment={m: sum(v) / len(v) for m, v in sorted(moment_hits.items())},
        by_category={c: sum(v) / len(v) for c, v in sorted(cat_recall.items())},
        latency_p50=_pct(lat, 0.5), latency_p95=_pct(lat, 0.95),
        cost_total=cost, cost_per_sample=cost / max(1, len(samples) - errors), unasked=unasked,
        share_unsure=_share(unsure), precision_sure_wrong=_share(sure_wrong), missed_in_sure_ok=_share(sure_ok),
    )


def common(samples: list, predictions: list[dict]) -> list:
    """The samples every one of `predictions` (one dict per model) has an answer or error for."""
    return [s for s in samples if all(s.id in p for p in predictions)]


def score_all(samples: list, predictions: dict[str, dict], sure: tuple[float, float] | None = None,
              ) -> tuple[list, list[Scores]]:
    """Every model in `predictions` (model key -> its predictions) scored on the samples all of them were
    asked about. Returns those samples and the scores, in `predictions`' order."""
    scored = common(samples, list(predictions.values()))
    return scored, [score(model, scored, p, sure) for model, p in predictions.items()]


def coverage(samples: list, scored: list, predictions: dict[str, dict]) -> list[str]:
    """The report's note on samples left out because not every model was asked about them, or none."""
    if len(scored) == len(samples):
        return []
    asked = ", ".join(f"{model} was asked about {sum(s.id in p for s in samples)}" for model, p in predictions.items())
    return [f"Scored on the {len(scored)} of the {len(samples)} samples every model was asked about: a run the "
            f"spending cap or Ctrl+C stopped leaves the rest unasked, and an unasked sample isn't counted wrong. "
            f"Of the {len(samples)}, {asked}."]


def _f(x, digits=2, pct=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.{digits}f}"


def report(title: str, samples: list, scores: list[Scores], notes: list[str],
           sure: tuple[float, float] | None = None) -> str:
    wrong = sum(s.wrong for s in samples)
    lines = [f"# {title}", "",
             f"{len(samples)} samples: {wrong} wrong, {len(samples) - wrong} ok. "
             "Sorted by balanced accuracy (the mean of the two recalls). The interval is a 95% bootstrap "
             "interval: with this few samples, models whose intervals overlap aren't shown to differ.", ""]
    lines += notes + [""] if notes else []
    lines += ["| Model | Balanced accuracy | 95% interval | Failures caught | OK passed | Precision on wrong | AUROC | Calibration error | Category right | Errors | p50 latency | p95 latency | Cost/sample | Cost |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc in sorted(scores, key=lambda s: -(s.balanced_accuracy if not math.isnan(s.balanced_accuracy) else -1)):
        lines.append(f"| {sc.model} | {_f(sc.balanced_accuracy, pct=True)} | {_f(sc.ba_low, pct=True)}–{_f(sc.ba_high, pct=True)} | "
                     f"{_f(sc.recall_wrong, pct=True)} | {_f(sc.recall_ok, pct=True)} | {_f(sc.precision_wrong, pct=True)} | "
                     f"{_f(sc.auroc)} | {_f(sc.ece)} | {_f(sc.category_accuracy, pct=True)} | {sc.errors} | "
                     f"{_f(sc.latency_p50, 1)} s | {_f(sc.latency_p95, 1)} s | ${sc.cost_per_sample:.4f} | ${sc.cost_total:.2f} |")
    moments = sorted({m for sc in scores for m in sc.by_moment})
    lines += ["", "## Accuracy by moment", "", "| Model | " + " | ".join(moments) + " |", "|---|" + "---|" * len(moments)]
    for sc in scores:
        lines.append(f"| {sc.model} | " + " | ".join(_f(sc.by_moment.get(m), pct=True) for m in moments) + " |")
    if sure is not None:
        lines += ["", "## At the supervisor's thresholds", "",
                  f"The supervisor calls a check sure wrong at p_wrong at or over {sure[1]:g}, sure ok at or under "
                  f"{sure[0]:g}, and unsure between (its program's `sure_wrong_above` and `sure_ok_below`). "
                  "*Unsure* is the share of answers it would call unsure; *Sure wrong, right* the share of sure-wrong "
                  "calls that were wrong, since a sure wrong after a close plays the attempt back; *Wrong among sure "
                  "ok* the share of sure-ok calls that were wrong, failures it would let pass. Only answers with a "
                  "probability count.", "",
                  "| Model | Unsure | Sure wrong, right | Wrong among sure ok |", "|---|---|---|---|"]
        for sc in scores:
            lines.append(f"| {sc.model} | {_f(sc.share_unsure, pct=True)} | {_f(sc.precision_sure_wrong, pct=True)} | "
                         f"{_f(sc.missed_in_sure_ok, pct=True)} |")
    cats = sorted({c for sc in scores for c in sc.by_category})
    counts = {c: sum(s.category == c for s in samples) for c in cats}
    lines += ["", "## Failures caught, by category", "",
              "| Model | " + " | ".join(f"{c} ({counts[c]})" for c in cats) + " |", "|---|" + "---|" * len(cats)]
    for sc in scores:
        lines.append(f"| {sc.model} | " + " | ".join(_f(sc.by_category.get(c), pct=True) for c in cats) + " |")
    lines += ["", "Columns: *Failures caught* is recall on wrong samples; *OK passed* is recall on ok samples; "
              "*AUROC* ranks samples by the model's probability that something is wrong (– when it gives none); "
              "*Calibration error* is the expected calibration error of that probability over 10 bins; "
              "*Category right* is the share of caught failures given a category in the right family (grasp failed; block not in the jar; still holding; wrong target; other); *Errors* are "
              "samples with no usable answer (API error, refusal, unparseable), scored as wrong answers.",
              "", "Categories: " + "; ".join(f"`{k}`: {v.rstrip('.')}" for k, v in prompt.CATEGORIES.items() if k != "none") + "."]
    return "\n".join(lines) + "\n"
