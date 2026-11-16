"""Run models over samples: in parallel per provider, every answer cached, within a spending cap.

Answers are appended to `<run>/predictions.jsonl` as they arrive, one JSON object per line, keyed by
model, variant and sample id. Running the same command again skips everything already answered, so a
crash, a Ctrl+C or the cap stopping a run costs nothing already paid for; `--retry-errors` asks again
for samples that errored.

The cap (`max_cost`) counts only what this invocation spends. Before each call the runner adds the
model's average cost so far (or its `estimate` from `models.py` before the first answer) and doesn't start
the call if that would pass the cap, so it can end a little under the cap but not far over it.
"""

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from so_101.failbench import dataset
from so_101.failbench.models import Model


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Variant:
    cameras: tuple[str, ...] = ("top", "wrist")
    moment_hint: bool = True

    @property
    def name(self) -> str:
        return "+".join(self.cameras) + ("" if self.moment_hint else "-nomoment")


@dataclass
class Budget:
    max_cost: float
    spent: float = 0.0
    stopped: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, estimate: float) -> bool:
        with self.lock:
            if self.spent + estimate > self.max_cost:
                self.stopped = True
                return False
            self.spent += estimate
            return True

    def settle(self, estimate: float, actual: float) -> None:
        with self.lock:
            self.spent += actual - estimate


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[tuple[str, str, str], dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[(r["model"], r["variant"], r["sample"])] = r  # later lines win

    def get(self, model: str, variant: str, sample: str) -> dict | None:
        return self.rows.get((model, variant, sample))

    def put(self, row: dict) -> None:
        with self.lock:
            self.rows[(row["model"], row["variant"], row["sample"])] = row
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def for_model(self, model: str, variant: str) -> dict[str, dict]:
        return {s: r for (m, v, s), r in self.rows.items() if m == model and v == variant}


def images_for(root: Path, s: dataset.Sample, variant: Variant) -> list[tuple[str, bytes]]:
    return [(cam, dataset.image_path(root, s, cam).read_bytes()) for cam in variant.cameras]


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode()


def call_with_retries(fn, attempts: int = 5, first_wait: float = 2.0):
    wait = first_wait
    for i in range(attempts):
        try:
            return fn()
        except ProviderError as e:
            if not e.retryable or i == attempts - 1:
                raise
            time.sleep(wait)
            wait *= 2


def run(models: list[Model], samples: list[dataset.Sample], root: Path, cache: Cache, variant: Variant,
        budget: Budget, workers: int = 4, retry_errors: bool = False, log=print) -> None:
    """Ask every model about every sample not yet in the cache."""
    by_provider: dict[str, list[tuple[Model, dataset.Sample]]] = {}
    for m in models:
        for s in samples:
            row = cache.get(m.key, variant.name, s.id)
            if row and not (retry_errors and row.get("error")):
                continue
            by_provider.setdefault(m.provider, []).append((m, s))
    if not by_provider:
        log("Nothing to do: every sample already has an answer from every model.")
        return
    totals = {m.key: [0, 0.0] for m in models}  # answers, cost, for the running estimate
    done = [0]
    todo = sum(len(v) for v in by_provider.values())

    def one(m: Model, s: dataset.Sample) -> None:
        n, c = totals[m.key]
        estimate = c / n if n else m.estimate
        if not budget.reserve(estimate):
            return
        started = time.monotonic()
        row = {"model": m.key, "variant": variant.name, "sample": s.id, "time": time.time()}
        actual = 0.0
        try:
            result = call_with_retries(lambda: m.adapter().predict(images_for(root, s, variant),
                                                                  s.moment if variant.moment_hint else None))
            actual = float(result.get("cost_usd") or 0.0)
            row.update(result)
        except ProviderError as e:
            row["error"] = str(e)[:500]
        except Exception as e:  # an adapter bug or an SDK surprise: record it, keep going
            row["error"] = f"{type(e).__name__}: {e}"[:500]
        row["latency_s"] = row.get("latency_s", round(time.monotonic() - started, 3))
        budget.settle(estimate, actual)
        totals[m.key][0] += 1
        totals[m.key][1] += actual
        cache.put(row)
        done[0] += 1
        mark = "ERROR " + row["error"][:120] if row.get("error") else f"wrong={row.get('wrong')} p={row.get('p_wrong')}"
        log(f"[{done[0]}/{todo}] {m.key} {s.id} ({s.label}): {mark}  ${budget.spent:.3f} spent")

    def provider_queue(items):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, m, s) for m, s in items]
            for f in as_completed(futures):
                f.result()

    with ThreadPoolExecutor(max_workers=len(by_provider)) as outer:
        for f in as_completed([outer.submit(provider_queue, items) for items in by_provider.values()]):
            f.result()
    if budget.stopped:
        log(f"Stopped at the spending cap of ${budget.max_cost:.2f}: some samples weren't asked. "
            "Run again with a higher --max-cost to finish; answers so far are kept.")
