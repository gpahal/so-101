"""The supervisor as a layer of the policy stack (docs/policy-layers.md): it watches each attempt and steps in.

It sits over the perception (and the skill sequencer, for a skills model), and each tick either passes
the tick down or drives the arm itself:

- **Watch.** It reads the perception's signals and readings in each Output that comes up. At the
  program's moments the judges look at the camera frames, on their own worker pool; their verdicts are
  collected on later ticks, so a step never waits for one.
- **Rewind and retry.** A miss (an after_close check that says wrong) plays back `miss_rewind_s[n]` of the
  model's own measured motion, newest first (`control.rewind`, the same playback as serve's Rewind),
  then hands the arm back with `Handback(LAYER, moved=True)`, so the layers below restart from where the
  arm is. A miss past that list counts as stuck.
- **Hold and check.** A stuck rule holds the arm and asks the watch check. A sure ok hands it back with
  `Handback(LAYER, moved=False)`; anything else asks for a person (`Output.asks`), and the escalation
  model's advice for them follows as a message (`maintain`).
- **Done.** A release that both judges are sure put the block in the container asks for a person too:
  the task is done, and only a person ends or labels an attempt.

In `shadow` mode every tick passes straight through: the decisions are logged with `acted: false` and
nothing else changes. Either way each decision and check is a `supervisor` event, and each playback a
`playback` event with the `start_frame` and `end_frame` of the motion it undid, which `control.analysis`
never trains on, whichever layer wrote it.

What the runner owns, it leaves alone: it never ends, labels or resets an attempt, and anything between
`pause` and `run` (a person's Teleop, a serve Rewind) is outside it. After each handback its playback
history starts again, so it never drives back past a pose something else set.
"""

from collections import deque
from dataclasses import dataclass, field, replace

from so_101.control import rewind as R
from so_101.grasp.thresholds import RULES
from so_101.policy.base import (DONE, HOLD, LAYER, PLAYBACK, STUCK, Ask, Handback, Layer, LayerContext, Output,
                                Readings, Tick)
from so_101.supervisor import judges as J

MODES = ("off", "shadow", "act")
CAMERAS = ("top", "wrist")
FRAME_EVERY_S = 0.2  # frames kept for the checks at most this often
RING_S = 4.0  # frames kept for the watch check's lookback
MAX_STEP_S = 0.5  # a longer gap between steps (a stalled loop) counts as this much of the playback history
PASS, REWINDING, HOLDING = "pass", "rewinding", "holding"  # what the layer is doing with the arm


@dataclass
class _Attempt:
    number: int
    history: R.History = field(default_factory=R.History)  # the model's motion since the last handback
    readings: Readings = field(default_factory=Readings)  # the perception's, as of the model's last tick
    frames: dict | None = None  # the newest frames
    ring: deque = field(default_factory=deque)  # (t, frames)
    closes: list = field(default_factory=list)  # (caption, frames) for the contact sheet
    log: list[str] = field(default_factory=list)
    scheduled: list = field(default_factory=list)  # (due t, moment, close seq)
    close_seq: int = 0
    misses: int = 0
    watches: int = 0
    last_watch_t: float = -1e9
    stuck: str | None = None  # the rule holding the arm, waiting for the watch check
    asked: bool = False  # it has asked for a person in this attempt
    escalated: bool = False  # the escalation model was asked: once an attempt, in either mode

    @property
    def t(self) -> float:
        return self.readings.t


class Supervisor(Layer):
    def __init__(self, inner, ctx: LayerContext, judges: J.Judges, *, mode: str, cameras=CAMERAS) -> None:
        super().__init__(inner, ctx, "supervisor")
        if mode not in ("shadow", "act"):
            raise ValueError(f"mode is {mode!r}: use shadow or act")
        self.p, self.judges, self.mode = ctx.program, judges, mode
        self.cameras = tuple(c for c in CAMERAS if c in cameras)
        self.attempt: _Attempt | None = None
        self._inflight: list = []  # (future, kind, attempt number, close seq, extra)
        self._messages: list[str] = []  # for the operator, through `maintain`
        self._asking: Ask | None = None  # for the runner to pause on, through Output.asks
        self._wants: list[tuple] = []  # what the next step starts, in order: (REWINDING, s), (HOLDING,), (PASS,)
        self.doing = PASS
        self._rewind: R.Rewind | None = None
        self._last: float | None = None
        self._pose: dict[str, float] = {}
        self._frames_at: float | None = None
        self._active = False  # between `run` and `pause`: the runner steps it

    # --- The policy's methods, as the runner calls them -------------------------------------------------

    def start(self) -> None:
        self.log("start", program=str(self.p.path), version=self.p.version,
                 judges={key: "ok" if self.judges.available(key) else self.judges.missing.get(key, "missing")
                         for key in self.judges.models})
        self.inner.start()

    def close(self) -> None:
        """Stop the judges' pool; calls still running finish on their own."""
        self.judges.close()
        self.inner.close()

    def maintain(self) -> list[str]:
        messages = self.inner.maintain()
        self.safely(self._collect)
        mine, self._messages = self._messages, []
        return [*messages, *mine]

    def begin_episode(self, number: int) -> None:
        self.inner.begin_episode(number)
        self.attempt = None
        self.safely(self._begin, number)

    def end_episode(self) -> None:
        self.safely(self._end)
        self.attempt = None
        self.inner.end_episode()

    def pause(self) -> None:
        """The runner takes the arm: no action from the model or from a playback reaches it after this."""
        self._active, self._wants, self._asking, self._last = False, [], None, None
        if self._rewind is not None:
            self.safely(self._finish_rewind, "paused")
        self.doing, self._rewind = PASS, None
        self.inner.pause()

    def run(self, handback: Handback) -> None:
        """The runner hands the arm to the stack, after something else may have moved it."""
        self.inner.run(handback)
        self._active, self._wants, self._asking, self._last, self.doing = True, [], None, None, PASS
        self._rewind = None
        if (a := self.attempt) is not None:
            a.history = R.History()
            a.stuck = None

    def step(self, robot, tick: Tick) -> Output:
        """This tick's goal: the stack below's, the next of a playback's, or the hold."""
        dt = 0.0 if self._last is None else min(max(tick.now - self._last, 0.0), MAX_STEP_S)
        self._last = tick.now
        self.safely(self._observe, tick)
        wants, self._wants = self._wants, []
        for want in wants if self.mode == "act" and self.broken is None else ():
            self._start(want)
        if self.doing == REWINDING:
            arrived = self._rewind.step(robot, self._pose, dt)
            if arrived is not None:
                self.safely(self._finish_rewind, "arrived" if arrived else "stopped_short")
                if self.broken is None:  # else `failed` left it holding for a person
                    self.doing = PASS
                    self.inner.run(Handback(LAYER, moved=True))  # a clean slate, from where the playback left the arm
            return Output(None, PLAYBACK, asks=self._take_ask())
        if self.doing == HOLDING:
            self.inner.hold(robot)
            return Output(None, HOLD, asks=self._take_ask())
        out = self.inner.step(robot, tick)
        self.safely(self._watch_model, dt, tick, out)
        return replace(out, asks=out.asks or self._take_ask())

    def _take_ask(self) -> Ask | None:
        asking, self._asking = self._asking, None
        return asking

    def _start(self, want: tuple) -> None:
        """Start what a decision asked for. The stack below's own calls are outside `safely`: what they
        raise is theirs, for the runner."""
        a = self.attempt
        if want[0] == REWINDING and a is not None and a.history.moments:
            self.inner.pause()
            self._rewind = R.Rewind(a.history, want[1], self.ctx.clock)
            self.doing = REWINDING
        elif want[0] == HOLDING and self.doing == PASS:
            self.inner.pause()
            self.doing = HOLDING
        elif want[0] == PASS and self.doing == HOLDING:
            self.doing = PASS
            self.inner.run(Handback(LAYER, moved=False))

    def failed(self) -> None:
        self._messages.append(f"Supervisor: stopped after an error ({self.broken}). The policy runs without it.")
        if self.doing != PASS:  # it was holding or playing back: a person decides what's next
            self.doing, self._rewind = HOLDING, None
            self._asking = Ask(LAYER, "Supervisor: stopped while it held the arm. Resume runs the policy without it.")

    # --- Logging -----------------------------------------------------------------------------------

    def log(self, kind: str, **fields) -> None:
        a = self.attempt
        context = {"for_attempt": a.number, "policy_t": round(a.t, 2)} if a is not None else {}
        self.ctx.event("supervisor", **{"mode": self.mode, **context, **fields, "kind": kind})

    def _note(self, line: str) -> None:
        if self.attempt is not None:
            self.attempt.log.append(f"t={self.attempt.t:.1f}s {line}")

    def _decide(self, decision: str, want: tuple | None = None, ask: str | None = None, tell: str | None = None,
                ask_kind: str = STUCK, **fields) -> bool:
        """Log `decision` and, in act mode while the stack has the arm, start `want`, ask for a person
        with `ask` (an `Ask` of `ask_kind`), and tell the operator `tell`. Returns whether it acted."""
        acting = self.mode == "act" and self._active
        self.log("decision", decision=decision, acted=acting, **fields)
        self._note(f"decision: {decision}")
        if acting:
            if want is not None:
                self._wants.append(want)
            if ask is not None:
                self._wants.append((HOLDING,))
                self._asking = Ask(ask_kind, f"Supervisor: {ask}")
                self.attempt.asked = True
            if tell is not None:
                self._messages.append(f"Supervisor: {tell}")
        return acting

    # --- Watching ----------------------------------------------------------------------------------

    def _begin(self, number: int) -> None:
        self.attempt = _Attempt(number)
        self._frames_at = None
        self.log("attempt")

    def _end(self) -> None:
        if (a := self.attempt) is None:
            return
        r = a.readings
        self.log("attempt_end", closes=r.closes, regrasps=r.regrasps, carried=r.carried, released=r.released,
                 fired=list(r.fired), misses=a.misses, watches=a.watches, asked=a.asked,
                 spent_usd=round(self.judges.spent, 4))

    def _observe(self, tick: Tick) -> None:
        self._pose = dict(tick.pose)
        if (a := self.attempt) is not None and (self._frames_at is None or tick.now - self._frames_at >= FRAME_EVERY_S):
            frames = {name: tick.observation[name] for name in self.cameras if name in tick.observation}
            if frames:
                a.frames = frames
                a.ring.append((a.t, frames))
                while a.ring and a.ring[0][0] < a.t - RING_S:
                    a.ring.popleft()
                self._frames_at = tick.now
        self._collect()

    def _watch_model(self, dt: float, tick: Tick, out: Output) -> None:
        """A tick the model drove (or waited for its next action): its motion for playback, the
        perception's signals, then any check due."""
        if (a := self.attempt) is None or out.readings is None:
            return
        a.readings = out.readings
        a.history.add(dt, dict(self._pose), tick.frame)
        for signal in out.signals:
            self._signal(a, signal)
        self._due_checks(a)
        self._watch(a)

    def _signal(self, a: _Attempt, s) -> None:
        self._note(f"{s.kind}{' ' + s.name if s.name else ''} {s.detail}")
        if s.kind == "close":
            a.close_seq += 1
            a.scheduled.append((s.t + self.p.after_close_s, "after_close", a.close_seq))
        elif s.kind == "release":
            a.scheduled.append((s.t + self.p.after_release_s, "after_release", a.close_seq))
        elif s.kind == "rule":
            self._stuck(a, s.name)

    def _images(self, frames: dict | None):
        """A function the judges' pool runs to encode `frames`, or None without any."""
        if not frames:
            return None
        return lambda: [(name, J.jpeg(frames[name])) for name in CAMERAS if name in frames]

    def _due_checks(self, a: _Attempt) -> None:
        due = [item for item in a.scheduled if item[0] <= a.t]
        a.scheduled = [item for item in a.scheduled if item[0] > a.t]
        for _, moment, seq in due:
            if (images := self._images(a.frames)) is None:
                self.log("check", moment=moment, error="no camera frames")
                continue
            future = self.judges.submit(lambda images=images, moment=moment: self.judges.event_check(images(), moment))
            self._inflight.append((future, moment, a.number, seq, {"frames": a.frames}))

    def _lookback(self, a: _Attempt) -> dict | None:
        target = a.t - self.p.watch.lookback_s
        older = [frames for t, frames in a.ring if t <= target]
        return older[-1] if older else (a.ring[0][1] if a.ring else None)

    def _watch(self, a: _Attempt, *, now: bool = False) -> None:
        """A watch check while a rule is near firing, every `period_s`, or `now` for a fired rule."""
        if not now and (not a.readings.near or a.t - a.last_watch_t < self.p.watch.period_s):
            return
        if a.watches >= self.p.watch.max_per_attempt and not now:
            return
        current, before = self._images(a.frames), self._images(self._lookback(a))
        if current is None or before is None:
            if now:  # a fired rule must still be settled: with nothing to look at, a person does
                self._inflight.append((_answered(J.Verdict("watch", error="no camera frames")), "watch",
                                       a.number, a.close_seq, {"near": a.stuck}))
            return
        a.watches += 1
        a.last_watch_t = a.t
        near = a.stuck or ",".join(a.readings.near)
        future = self.judges.submit(lambda: self.judges.watch_check(current(), before()))
        self._inflight.append((future, "watch", a.number, a.close_seq, {"near": near}))

    # --- Deciding ----------------------------------------------------------------------------------

    def _stuck(self, a: _Attempt, why: str) -> None:
        if a.stuck or a.asked:
            return
        a.stuck = why
        self._decide(f"hold: {why}", want=(HOLDING,), rule=why)
        self._watch(a, now=True)

    def _miss(self, a: _Attempt) -> None:
        a.misses += 1
        if a.misses <= len(self.p.miss_rewind_s):
            seconds = self.p.miss_rewind_s[a.misses - 1]
            self._decide(f"miss {a.misses}: rewind {seconds:g} s and retry", want=(REWINDING, seconds),
                         misses=a.misses, rewind_s=seconds,
                         tell=f"missed grasp {a.misses}: rewinding {seconds:g} s to try again.")
        else:
            self._stuck(a, f"miss {a.misses}")

    def _escalate(self, a: _Attempt, why: str) -> None:
        self._decide(f"ask a person: {why}", ask=f"{why}. Paused for you" + ("." if a.escalated else "; asking for advice."))
        if a.escalated:
            return
        a.escalated = True
        columns = list(a.closes[-5:])
        if a.frames:
            columns.append((f"now t={a.t:.1f}s", a.frames))
        if not columns:
            self.log("escalation", error="no frames")
            return
        sheet_log = "\n".join(a.log[-40:])
        future = self.judges.submit(lambda: self.judges.escalate(J.contact_sheet(columns), sheet_log))
        self._inflight.append((future, "escalation", a.number, a.close_seq, {}))

    def _finish_rewind(self, cause: str) -> None:
        running, self._rewind = self._rewind, None
        if running is not None:
            a = self.attempt
            self.ctx.event("playback", layer=self.name, **({"for_attempt": a.number} if a else {}),
                           **R.finish(running, self._pose, cause))

    def _collect(self) -> None:
        """Every verdict that has come back since the last call."""
        done = [item for item in self._inflight if item[0].done()]
        self._inflight = [item for item in self._inflight if not item[0].done()]
        for future, kind, number, seq, extra in done:
            a = self.attempt
            answer = future.result()
            stale = a is None or a.number != number or (kind == "after_close" and seq != a.close_seq)
            mark = {"stale": True} if stale else {}
            if kind == "escalation":
                self.log("escalation", **answer, **mark)
                if "choice" in answer and not stale and self.mode == "act":
                    self._messages.append(f"Supervisor's advice: {J.CHOICES[answer['choice']]} ({answer.get('reason', '')})")
                continue
            if kind == "watch":
                call = answer.call(self.p)
                self.log("check", moment="watch", near=extra["near"], call=call, decision=answer.record(), **mark)
                if stale or a.stuck is None:
                    continue
                self._note(f"watch check: {call}")
                why = a.stuck
                a.stuck = None
                if call == J.OK and why in RULES:  # a run of misses always goes to a person
                    self._decide("resume: the watch check saw nothing wrong", want=(PASS,), rule=why)
                else:
                    self._escalate(a, f"stuck ({why.replace('_', ' ')})")
                continue
            decision, chat = answer
            call = J.combine(self.p, decision, chat)
            fields = {"moment": kind, "call": call, "decision": decision.record(), "chat": chat.record()}
            if kind == "after_release":
                fields["success"] = J.success(self.p, decision, chat)
            self.log("check", **fields, **mark)
            if stale:
                continue
            self._note(f"{kind} check: {call}")
            if kind == "after_close":
                a.closes.append((f"close {seq}: {call}", extra["frames"]))
                if call == J.WRONG:
                    self._miss(a)
            elif kind == "after_release":
                if call == J.WRONG:
                    self._decide("tell the operator: the release missed the container",
                                 tell="the block may have missed the container.")
                elif fields["success"] and not a.asked:
                    self._decide("done: the block is in the container", ask_kind=DONE,
                                 ask="the block looks in the container. Reset and label the attempt.")


def _answered(value):
    from concurrent.futures import Future

    future = Future()
    future.set_result(value)
    return future
