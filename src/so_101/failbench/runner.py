"""Run models over samples: in parallel per provider, every answer cached, within a spending cap.

Answers are appended to `<run>/predictions.jsonl` as they arrive, one JSON object per line, keyed by
model, variant and sample id. Running the same command again skips everything already answered, so a
crash, a Ctrl+C or the cap stopping a run costs nothing already paid for; `--retry-errors` asks again
for samples that errored.

The queue is sample-major: every model is asked about one sample before any is asked about the next, so
a run the cap or a Ctrl+C stops leaves the models a shared subset to compare on.

The cap (`max_cost`) counts only what this invocation spends. Before each call the runner reserves the
model's average cost so far over the calls that answered with a cost (or its `estimate` from `models.py`
before the first) and doesn't start the call if that would pass the cap, so it can end a little under the
cap but not far over it. A call that errored, or answered without saying what it used, is charged what
was reserved, since the provider may have billed it anyway.

Ctrl+C (or `stop`) stops the run: nothing more starts, the calls in flight finish and are cached, and the
run returns, so its report is still written. A second Ctrl+C stops waiting for them.

Failures that retrying can't fix: each model's adapter is made before anything is queued, and a model
whose adapter can't be made is skipped with one line, nothing cached for it. A model that fails
`GIVE_UP_AFTER` calls in a row with errors that aren't retryable (a bad request, a wrong key) is asked
nothing more in this run.
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from so_101.failbench import dataset
from so_101.vision.models import Model
from so_101.vision.providers import ProviderError

GIVE_UP_AFTER = 3  # non-retryable errors in a row after which a model is asked nothing more
WAIT_S = 0.1  # how often the main thread looks up from waiting, so Ctrl+C reaches it


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
        with self.lock:
            return {s: r for (m, v, s), r in self.rows.items() if m == model and v == variant}


def images_for(root: Path, s: dataset.Sample, variant: Variant) -> list[tuple[str, bytes]]:
    return [(cam, dataset.image_path(root, s, cam).read_bytes()) for cam in variant.cameras]


def call_with_retries(fn, attempts: int = 5, first_wait: float = 2.0, stop: threading.Event | None = None):
    """`fn()`, tried again after a retryable `ProviderError`, waiting twice as long each time. Once `stop`
    is set it tries no more, so a stopping run doesn't wait out the backoff."""
    stop = stop or threading.Event()
    wait = first_wait
    for i in range(attempts):
        try:
            return fn()
        except ProviderError as e:
            if not e.retryable or i == attempts - 1 or stop.wait(wait):
                raise
            wait *= 2


@dataclass
class Result:
    interrupted: bool = False  # Ctrl+C or `stop` ended it before everything was asked
    capped: bool = False  # the spending cap did
    skipped: dict[str, str] = field(default_factory=dict)  # model key -> why it was asked nothing, or stopped


def run(models: list[Model], samples: list[dataset.Sample], root: Path, cache: Cache, variant: Variant,
        budget: Budget, workers: int = 4, retry_errors: bool = False, log=print,
        stop: threading.Event | None = None) -> Result:
    """Ask every model about every sample not yet in the cache. Returns how it ended; never raises
    KeyboardInterrupt."""
    result = Result()
    stop = stop or threading.Event()
    adapters = {}
    for m in models:
        try:
            adapters[m.key] = m.adapter()
        except ProviderError as e:
            result.skipped[m.key] = str(e)
            log(f"Skipping {m.key}: {e}")
    queues: dict[str, deque[tuple[Model, dataset.Sample]]] = {}
    for s in samples:
        for m in models:
            if m.key not in adapters:
                continue
            row = cache.get(m.key, variant.name, s.id)
            if row and not (retry_errors and row.get("error")):
                continue
            queues.setdefault(m.provider, deque()).append((m, s))
    if not queues:
        if adapters:
            log("Nothing to do: every sample already has an answer from every model.")
        return result
    lock = threading.Lock()
    answered = {m.key: [0, 0.0] for m in models}  # calls that answered with a cost, and that cost
    failing = {m.key: 0 for m in models}  # non-retryable errors in a row
    done = [0]
    todo = sum(len(q) for q in queues.values())

    def one(m: Model, s: dataset.Sample) -> None:
        with lock:
            n, c = answered[m.key]
        estimate = c / n if n else m.estimate
        if not budget.reserve(estimate):
            return
        started = time.monotonic()
        row = {"model": m.key, "variant": variant.name, "sample": s.id, "time": time.time()}
        hopeless = False
        try:
            answer = call_with_retries(lambda: adapters[m.key].predict(images_for(root, s, variant),
                                                                       s.moment if variant.moment_hint else None),
                                       stop=stop)
            row.update(answer)
        except ProviderError as e:
            if e.retryable and stop.is_set():  # cut short by the stop: left unasked, so a rerun asks it
                budget.settle(estimate, estimate)
                return
            row["error"], hopeless = str(e)[:500], not e.retryable
        except Exception as e:  # an adapter bug or an SDK surprise: record it, keep going
            row["error"], hopeless = f"{type(e).__name__}: {e}"[:500], True
        if row.get("cost_usd") is None:  # an exception, or an answer without usage
            row["cost_usd"], row["cost_estimated"] = round(estimate, 6), True
        actual = float(row["cost_usd"])
        row["latency_s"] = row.get("latency_s", round(time.monotonic() - started, 3))
        budget.settle(estimate, actual)
        with lock:
            if not row.get("cost_estimated") and not row.get("error"):
                answered[m.key][0] += 1
                answered[m.key][1] += actual
            failing[m.key] = failing[m.key] + 1 if hopeless else 0
            if failing[m.key] >= GIVE_UP_AFTER and m.key not in result.skipped:
                result.skipped[m.key] = f"{GIVE_UP_AFTER} errors in a row that retrying can't fix"
                log(f"{m.key} stopped after {GIVE_UP_AFTER} errors in a row that retrying can't fix; the last: "
                    f"{row['error'][:200]}")
            done[0] += 1
            count = done[0]
        cache.put(row)
        mark = "ERROR " + row["error"][:120] if row.get("error") else f"wrong={row.get('wrong')} p={row.get('p_wrong')}"
        log(f"[{count}/{todo}] {m.key} {s.id} ({s.label}): {mark}  ${budget.spent:.3f} spent")

    def worker(queue: deque) -> None:
        while not stop.is_set() and not budget.stopped:
            with lock:
                while queue and queue[0][0].key in result.skipped:
                    queue.popleft()
                if not queue:
                    return
                m, s = queue.popleft()
            one(m, s)

    threads = [threading.Thread(target=worker, args=(q,), name=f"failbench-{provider}", daemon=True)
               for provider, q in queues.items() for _ in range(workers)]
    waiting = [True]

    def interrupted() -> None:
        if stop.is_set():
            log("Not waiting for the calls in flight: their answers are lost.")
            waiting[0] = False
        else:
            stop.set()
            log("Stopping: nothing more starts, and the calls in flight finish and are kept. "
                "Ctrl+C again to stop waiting for them.")

    for t in threads:
        try:
            t.start()
        except KeyboardInterrupt:
            interrupted()
    while True:
        try:  # everything in here, so a Ctrl+C anywhere in the wait is caught
            alive = [t for t in threads if t.is_alive()]
            if not alive or not waiting[0]:
                break
            alive[0].join(WAIT_S)
        except KeyboardInterrupt:
            interrupted()
    result.interrupted = stop.is_set()
    result.capped = budget.stopped
    if result.capped:
        log(f"Stopped at the spending cap of ${budget.max_cost:.2f}: some samples weren't asked. "
            "Run again with a higher --max-cost to finish; answers so far are kept.")
    elif result.interrupted:
        log("Stopped before every sample was asked; answers so far are kept, and running the same command "
            "again asks the rest.")
    return result
