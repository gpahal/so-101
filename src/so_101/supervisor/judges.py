"""The vision checks: the shared judges' adapters (`so_101.vision`), asked from the supervisor's worker pool.

- **Event check** (after a close, after a release): the decision model and the chat model get the same
  frames, as two calls the layer sends at the same time. After a close, a miss needs both sure it's
  wrong (`both_wrong`); after a release, the decision model's answer stands when it's sure, the chat
  model's when only it is (`combine`), and success needs both (`success`). `decided` says when the
  answers in so far settle a check, so a sure answer doesn't wait for the other.
- **Watch check** (when a stuck rule holds the arm): the decision model, with each camera now and
  `lookback_s` before.
- **Escalation**: a frontier chat model through OpenRouter reads a contact sheet of the attempt's closes
  and the detector's log, and picks one of `CHOICES`. It never picks another identical retry.

A judge whose key or SDK is missing is left out at the start and named in the supervisor's start event.
Every call is counted against the program's `max_cost_usd`; once a run has spent it, checks answer
with an error instead of calling. Nothing here raises: a failed call is a verdict with `error` set.

The calls run on `DaemonPool`'s threads, so a call that hangs never keeps the process from exiting, and
closing the pool cancels the calls still queued.
"""

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass

from so_101.supervisor.program import Program

JPEG_QUALITY = 90  # as so101-failbench's samples, so its scores hold for these checks
OK, WRONG, UNSURE = "ok", "wrong", "unsure"
CHOICES = {
    "teleop": "Take over with the leader: the block is in a pose or place the demos don't cover.",
    "check_gripper": "Check the gripper: it isn't following its commands, or the jaws can't close on the block.",
    "reset_failure": "Reset and label the attempt failed: the scene changed, or the block is out of reach.",
}
ESCALATION_SYSTEM = """You supervise a small robot arm (an SO-101) doing one task: "Pick up the red block and put it in
the cylindrical container". A policy drives it. It looks stuck, and it is paused now. You get a contact sheet:
each column is one moment of the attempt, with the TOP camera (a fixed view from above) over the WRIST camera
(on the gripper, its white jaws at the bottom). Each column's caption says when it was taken. You also get the
log of what the arm's readings showed. Retrying the same move would repeat it: choose what a person should do.
"""
ESCALATION_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string", "description": "One or two sentences: what went wrong, from the frames."},
        "choice": {"type": "string", "enum": list(CHOICES)},
        "reason": {"type": "string", "description": "One sentence: why this choice."},
    },
    "required": ["observation", "choice", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Verdict:
    judge: str
    p_wrong: float | None = None
    category: str | None = None
    latency_s: float | None = None
    cost_usd: float = 0.0
    error: str | None = None

    def call(self, program: Program) -> str | None:
        """ok, wrong or unsure by the program's thresholds; None without an answer."""
        if self.p_wrong is None:
            return None
        j = program.judges
        return WRONG if self.p_wrong >= j.sure_wrong_above else OK if self.p_wrong <= j.sure_ok_below else UNSURE

    def record(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items() if v is not None}


def combine(program: Program, decision: Verdict | None, chat: Verdict | None) -> str:
    """The after_release check's answer: the decision model when it's sure, else the chat model when it is."""
    for verdict in (decision, chat):
        call = verdict.call(program) if verdict is not None else None
        if call in (OK, WRONG):
            return call
    return UNSURE


def both_wrong(program: Program, decision: Verdict | None, chat: Verdict | None) -> bool:
    """A miss: both judges answered, each sure the grasp is wrong. One sure wrong alone plays nothing back."""
    return all(v is not None and v.call(program) == WRONG for v in (decision, chat))


def agree(program: Program, decision: Verdict | None, chat: Verdict | None) -> str:
    """The after_close check's answer as logged: wrong when both are sure it's wrong, ok when one is sure
    it's ok and neither sure it's wrong, else unsure."""
    calls = [v.call(program) if v is not None else None for v in (decision, chat)]
    if calls == [WRONG, WRONG]:
        return WRONG
    return OK if OK in calls and WRONG not in calls else UNSURE


def decided(outcome, program: Program, decision: Verdict | None, chat: Verdict | None) -> bool:
    """Whether `outcome(program, decision, chat)` is settled by the answers in so far: the same whatever a
    judge still out (None) answers, or fails to."""
    j = program.judges
    marks = sorted({0.0, 1.0, j.sure_ok_below, j.sure_wrong_above, j.success_p_wrong})
    points = [*marks, *((a + b) / 2 for a, b in zip(marks, marks[1:], strict=False))]
    fills = [None, *(Verdict("?", p_wrong=p) for p in points)]
    seen = {outcome(program, d, c) for d in ([decision] if decision is not None else fills)
            for c in ([chat] if chat is not None else fills)}
    return len(seen) == 1


def success(program: Program, decision: Verdict | None, chat: Verdict | None) -> bool:
    """The after_release check calls the task done only when both judges answered, both under `success_p_wrong`."""
    limit = program.judges.success_p_wrong
    return all(v is not None and v.p_wrong is not None and v.p_wrong < limit for v in (decision, chat))


def jpeg(frame) -> bytes:
    """An RGB frame from LeRobot as JPEG bytes."""
    import cv2

    ok, data = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("OpenCV couldn't encode a frame as JPEG.")
    return data.tobytes()


def contact_sheet(columns: list[tuple[str, dict]]) -> bytes:
    """Each column is a caption and {camera: RGB frame}: top over wrist, at half size."""
    import cv2
    import numpy as np

    tiles = []
    for caption, frames in columns:
        cells = []
        for camera in ("top", "wrist"):
            frame = frames.get(camera)
            cell = np.zeros((240, 320, 3), np.uint8) if frame is None else cv2.resize(frame, (320, 240))
            cells.append(cell)
        column = np.ascontiguousarray(np.vstack(cells))
        cv2.putText(column, caption[:40], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(column)
    return jpeg(np.hstack(tiles))


class DaemonPool:
    """A thread pool whose workers are daemon threads: a call that hangs never holds up the process's exit,
    as ThreadPoolExecutor's workers, which Python joins at exit, would. `shutdown` cancels what's queued."""

    def __init__(self, workers: int, name: str) -> None:
        self._workers, self._name = workers, name
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, fn, *args) -> Future:
        future = Future()
        with self._lock:
            if self._closed:
                future.cancel()
                return future
            self._queue.put((future, fn, args))
            if len(self._threads) < self._workers:
                thread = threading.Thread(target=self._work, name=f"{self._name}-{len(self._threads)}", daemon=True)
                self._threads.append(thread)
                thread.start()
        return future

    def _work(self) -> None:
        while (item := self._queue.get()) is not None:
            future, fn, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # noqa: BLE001 - the future carries it to whoever reads it
                future.set_exception(error)

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            threads = list(self._threads)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None and cancel_futures:
                item[0].cancel()
        for _ in threads:
            self._queue.put(None)
        if wait:
            for thread in threads:
                thread.join()


class Judges:
    def __init__(self, program: Program, *, adapters=None, pool=None) -> None:
        """`adapters` maps a model key to an object with the shared adapters' `predict(images, moment)` (and,
        for the escalation model, `escalate(sheet, log)`), for tests; without it, `so_101.vision`'s own
        adapters, which `so101-failbench` scores too, are made for the models whose key and SDK are there. `pool` runs the calls: a `DaemonPool` by default, with
        a worker for every call one attempt can have out at once and spares for calls that hang."""
        from so_101.vision.models import MODELS

        self.p = program
        self.models = {key: MODELS[key] for key in (program.judges.decision, program.judges.chat, program.judges.escalation)}
        self.spent = 0.0
        self._lock = threading.Lock()
        self._pool = pool or DaemonPool(8, "supervisor-judge")
        self.missing: dict[str, str] = {}  # model key -> why it can't be asked
        self._adapters = dict(adapters) if adapters is not None else self._make_adapters()

    def _make_adapters(self) -> dict:
        adapters = {}
        for key, model in self.models.items():
            if not model.has_key():
                self.missing[key] = f"{model.env} isn't set"
                continue
            if key == self.p.judges.escalation:
                adapters[key] = None  # called through `escalate`, which makes its own client
                continue
            try:
                adapters[key] = model.adapter()
            except Exception as error:  # the SDK isn't installed, or its client refused the key
                self.missing[key] = f"{type(error).__name__}: {error}"
        if self.p.judges.escalation in adapters:
            from so_101.vision.providers import sdk_missing

            if why := sdk_missing(self.models[self.p.judges.escalation].provider):
                del adapters[self.p.judges.escalation]
                self.missing[self.p.judges.escalation] = why
        return adapters

    def available(self, key: str) -> bool:
        return key in self._adapters

    def close(self) -> None:
        """Cancel the calls still queued; a call already running is left to finish, or hang, on its own."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def submit(self, fn, *args):
        """Run `fn(*args)` on the judges' pool. Returns its Future."""
        return self._pool.submit(fn, *args)

    def _budget(self) -> str | None:
        with self._lock:
            if self.spent >= self.p.judges.max_cost_usd:
                return f"the run has spent its ${self.p.judges.max_cost_usd:g} on checks (max_cost_usd)"
        return None

    def out_of_budget(self) -> str | None:
        """Why checks no longer call their models: the run has spent `max_cost_usd`. None while it hasn't."""
        return self._budget()

    def _charge(self, cost: float) -> None:
        with self._lock:
            self.spent += cost

    def ask(self, key: str, images: list[tuple[str, bytes]], moment: str) -> Verdict:
        """One model's verdict on `images` at `moment`. Never raises."""
        if key not in self._adapters:
            return Verdict(key, error=self.missing.get(key, "not available"))
        if reason := self._budget():
            return Verdict(key, error=reason)
        started = time.monotonic()
        # A call that reports no cost, or fails, may still have been billed: it's charged the model's
        # estimate, as `so101-failbench run` charges it, so the cap holds.
        estimate = self.models[key].estimate if key in self.models else 0.0
        try:
            answer = self._adapters[key].predict(images, moment)
        except Exception as error:
            self._charge(estimate)
            return Verdict(key, latency_s=time.monotonic() - started, error=f"{type(error).__name__}: {error}"[:300])
        reported = answer.get("cost_usd")
        cost = float(reported) if reported is not None else estimate
        self._charge(cost)
        return Verdict(key, p_wrong=answer.get("p_wrong"), category=answer.get("category"),
                       latency_s=answer.get("latency_s", time.monotonic() - started), cost_usd=cost,
                       error=answer.get("error"))

    def watch_check(self, now: list[tuple[str, bytes]], before: list[tuple[str, bytes]]) -> Verdict:
        """The decision model on each camera now and `lookback_s` before; the chat model if it's missing."""
        ago = f"{self.p.watch.lookback_s:g}s_ago".replace(".", "_")
        images = now + [(f"{camera}_{ago}", data) for camera, data in before]
        key = self.p.judges.decision if self.available(self.p.judges.decision) else self.p.judges.chat
        return self.ask(key, images, "stuck")

    def escalate(self, sheet: bytes, log: str) -> dict:
        """The frontier model's choice for a stuck, paused attempt: {choice, reason, observation, cost_usd,
        latency_s} or {error}. Never raises."""
        key = self.p.judges.escalation
        if key not in self._adapters:
            return {"error": self.missing.get(key, "not available")}
        if reason := self._budget():
            return {"error": reason}
        if (stand_in := self._adapters[key]) is not None:  # tests pass an object with its own `escalate`
            answer = stand_in.escalate(sheet, log)
            self._charge(float(answer.get("cost_usd") or 0.0))
            return answer
        from so_101.vision.providers import openrouter_json

        model = self.models[key]
        started = time.monotonic()
        try:
            choices = "\n".join(f"- {name}: {why}" for name, why in CHOICES.items())
            answer, cost = openrouter_json(model, ESCALATION_SYSTEM, sheet, f"Log:\n{log}\n\nChoices:\n{choices}",
                                           "escalation", ESCALATION_SCHEMA, title="so101-supervisor")
            self._charge(cost)
            if answer.get("choice") not in CHOICES:
                return {"error": f"no choice in the answer: {str(answer)[:200]}", "cost_usd": round(cost, 4)}
            return {**{k: str(answer.get(k, ""))[:300] for k in ("choice", "reason", "observation")},
                    "cost_usd": round(cost, 4), "latency_s": round(time.monotonic() - started, 2)}
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"[:300], "latency_s": round(time.monotonic() - started, 2)}
