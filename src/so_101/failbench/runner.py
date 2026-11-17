"""Run models over samples: in parallel per provider, every answer cached, within a spending cap.

Answers are appended to `<run>/predictions.jsonl` as they arrive, one JSON object per line, keyed by
model, variant and sample id, each with what it was asked under (`asked`: the sha256 of the shared prompt
and of the provider adapters, and the model's id and effort). A run refuses to resume a folder holding a
selected model's answers asked under anything else, before asking anything (`Result.stale`): mixed with new ones, they would be scored as one
experiment, and the supervisor's thresholds are tuned from that report. Use a new folder instead. Running the same command again skips everything already answered, so a
crash, a Ctrl+C or the cap stopping a run costs nothing already paid for; `--retry-errors` asks again
for samples that errored. A line cut off by a kill is skipped with a warning and its answer asked again
(`Cache`); the file is never rewritten.

The queue is sample-major: every model is asked about one sample before any is asked about the next, so
a run the cap or a Ctrl+C stops leaves the models a shared subset to compare on.

The cap (`max_cost`, a finite number above 0: `cli.max_cost` refuses anything else, since NaN would
compare as never reached) counts only what this invocation spends. Before each call the runner reserves the
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

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from so_101.failbench import dataset
from so_101.vision.budget import Budget  # noqa: F401  `runner.Budget`, as `cli` and the tests name it
from so_101.vision.models import Model
from so_101.vision.providers import ProviderError

def asked(m: Model) -> dict:
    """What a model's answer was asked under: the shared prompt (all of `vision/prompt.py`, so any change to
    it counts), the adapters that send it and read the answer (all of `vision/providers.py`, which imports
    the providers' SDKs only when it calls one), and the model's id and effort."""
    from so_101.vision import prompt, providers

    def sha256(module) -> str:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()

    return {"prompt_sha256": sha256(prompt), "providers_sha256": sha256(providers), "model_id": m.model_id,
            "effort": m.effort}


GIVE_UP_AFTER = 3  # non-retryable errors in a row after which a model is asked nothing more
WAIT_S = 0.1  # how often the main thread looks up from waiting, so Ctrl+C reaches it


@dataclass
class Variant:
    cameras: tuple[str, ...] = ("top", "wrist")
    moment_hint: bool = True

    @property
    def name(self) -> str:
        return "+".join(self.cameras) + ("" if self.moment_hint else "-nomoment")


class Cache:
    """`predictions.jsonl`, read once and appended to. A line that isn't valid JSON, which a run killed mid-write
    leaves, is skipped with one warning for the file and asked about again; the file is never rewritten, and
    `put` starts a new line first when the file doesn't end with one, so the next answer never joins it."""

    def __init__(self, path: Path, log=print):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[tuple[str, str, str], dict] = {}
        skipped = 0
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    self.rows[(r["model"], r["variant"], r["sample"])] = r  # later lines win
        if skipped:
            one = skipped == 1
            log(f"Skipped {skipped} {'line' if one else 'lines'} of {path} that {'isn' if one else 'aren'}'t valid JSON, "
                f"most likely cut off by a run killed while writing. The file is left as it is, so this repeats each "
                f"time it's read, and `run` asks again for any answer no later line holds.")

    def get(self, model: str, variant: str, sample: str) -> dict | None:
        return self.rows.get((model, variant, sample))

    def put(self, row: dict) -> None:
        with self.lock:
            self.rows[(row["model"], row["variant"], row["sample"])] = row
            with open(self.path, "a+b") as f:  # writes always append; the seek only reads the last byte
                cut_off = False
                end = f.seek(0, 2)
                if end:
                    f.seek(end - 1)
                    cut_off = f.read(1) != b"\n"
                f.write((("\n" if cut_off else "") + json.dumps(row) + "\n").encode())

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
    # The selected models whose cached answers were asked under another prompt, adapter or model settings (`asked`): the
    # run asked nothing.
    stale: list[str] = field(default_factory=list)


def run(models: list[Model], samples: list[dataset.Sample], root: Path, cache: Cache, variant: Variant,
        budget: Budget, workers: int = 4, retry_errors: bool = False, log=print,
        stop: threading.Event | None = None) -> Result:
    """Ask every model about every sample not yet in the cache. Returns how it ended; never raises
    KeyboardInterrupt.

    The budget admits whole samples: the first worker to take any call of a sample reserves, at once, the
    estimates of every call still to make for it, across all models, and each call then settles its own share.
    A sample that doesn't fit isn't admitted and stops the budget; workers then drop the calls of samples not
    admitted but finish those of the ones that were, so every sample asked is asked of every model, and
    `metrics.common()` keeps what was paid for. Each provider still has its own queue and workers, so a fast
    provider can't spend the cap on samples a slow one never reaches. The share of a call that never runs (its
    model gave up, or a stop came first) is given back."""
    result = Result()
    stop = stop or threading.Event()
    now = {m.key: asked(m) for m in models}
    result.stale = sorted({r["model"] for r in cache.rows.values() if r["model"] in now and r.get("asked") != now[r["model"]]})
    if result.stale:
        log(f"Asking nothing: the cached answers of {', '.join(result.stale)} were asked under another prompt, "
            "adapter or model settings.")
        return result
    adapters = {}
    for m in models:
        try:
            adapters[m.key] = m.adapter()
        except ProviderError as e:
            result.skipped[m.key] = str(e)
            log(f"Skipping {m.key}: {e}")
    queues: dict[str, deque[tuple[Model, dataset.Sample]]] = {}
    calls: dict[str, list[Model]] = {}  # sample id -> the models still to ask about it
    for s in samples:
        for m in models:
            if m.key not in adapters:
                continue
            row = cache.get(m.key, variant.name, s.id)
            if row and not (retry_errors and row.get("error")):
                continue
            queues.setdefault(m.provider, deque()).append((m, s))
            calls.setdefault(s.id, []).append(m)
    if not queues:
        if adapters:
            log("Nothing to do: every sample already has an answer from every model.")
        return result
    lock = threading.Lock()
    answered = {m.key: [0, 0.0] for m in models}  # calls that answered with a cost, and that cost
    failing = {m.key: 0 for m in models}  # non-retryable errors in a row
    done = [0]
    todo = sum(len(q) for q in queues.values())
    # sample id -> model key -> the share of the sample's reservation its call still holds, for each call of an
    # admitted sample not yet started. Read and written under `lock`.
    admitted: dict[str, dict[str, float]] = {}

    def estimate_for(m: Model) -> float:
        """The model's running average cost of an answer, or its listed estimate before any. Under `lock`."""
        n, c = answered[m.key]
        return c / n if n else m.estimate

    def admit(s: dataset.Sample) -> bool:
        """Reserve every call still to make for `s` at once, unless the budget has stopped or it doesn't fit.
        Under `lock`."""
        if s.id in admitted:
            return True
        if budget.stopped:
            return False
        shares = {m.key: estimate_for(m) for m in calls[s.id] if m.key not in result.skipped}
        if not budget.reserve(sum(shares.values())):
            return False
        admitted[s.id] = shares
        return True

    def give_back(key: str) -> None:
        """Return the shares of `key`'s calls not yet started: its model gave up. Under `lock`."""
        for shares in admitted.values():
            if (share := shares.pop(key, None)) is not None:
                budget.settle(share, 0.0)

    def one(m: Model, s: dataset.Sample, estimate: float) -> None:
        started = time.monotonic()
        row = {"model": m.key, "variant": variant.name, "sample": s.id, "time": time.time(), "asked": now[m.key]}
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
                give_back(m.key)
                log(f"{m.key} stopped after {GIVE_UP_AFTER} errors in a row that retrying can't fix; the last: "
                    f"{row['error'][:200]}")
            done[0] += 1
            count = done[0]
        cache.put(row)
        mark = "ERROR " + row["error"][:120] if row.get("error") else f"wrong={row.get('wrong')} p={row.get('p_wrong')}"
        log(f"[{count}/{todo}] {m.key} {s.id} ({s.label}): {mark}  ${budget.spent:.3f} spent")

    def worker(queue: deque) -> None:
        while not stop.is_set():
            with lock:
                while queue:
                    m, s = queue.popleft()
                    if m.key not in result.skipped and admit(s) and m.key in admitted[s.id]:
                        estimate = admitted[s.id].pop(m.key)
                        break
                else:
                    return
            one(m, s, estimate)

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
    with lock:  # calls of admitted samples a stop left unstarted: they never billed
        for shares in admitted.values():
            for share in shares.values():
                budget.settle(share, 0.0)
            shares.clear()
    result.interrupted = stop.is_set()
    result.capped = budget.stopped
    if result.capped:
        log(f"Stopped at the spending cap of ${budget.max_cost:.2f}: some samples weren't asked. "
            "Run again with a higher --max-cost to finish; answers so far are kept.")
    elif result.interrupted:
        log("Stopped before every sample was asked; answers so far are kept, and running the same command "
            "again asks the rest.")
    return result
