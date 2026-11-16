"""The supervisor as a policy over the policy: `SupervisedPolicy` wraps the policy driver a runner steps.

A runner (`so101 serve`'s control loop) steps it exactly as it steps `control.policy.EngineDriver`: the
same methods, called at the same times, on the loop's own thread. Each step it either passes the tick to
the policy it wraps, or drives the arm itself:

- **Watch.** On every tick the policy drives, the detector reads the measured pose. At the program's
  moments the judges look at the camera frames, on their own worker pool; their verdicts are collected
  on later ticks, so a step never waits for one.
- **Rewind and retry.** A miss (an after_close check that says wrong) plays back `miss_rewind_s[n]` of
  the policy's own measured motion, newest first (`control.rewind`, the same playback as serve's Rewind),
  then hands the arm back to the policy with its queue cleared. A miss past that list counts as stuck.
- **Hold and check.** A stuck rule holds the arm and asks the watch check. A sure ok hands it back to the
  policy; anything else asks for a person (`asks`), and the escalation model's advice for them follows
  as a message (`maintain`).
- **Done.** A release that both judges are sure put the block in the container asks for a person too:
  the task is done, and only a person ends or labels an attempt.

In `shadow` mode every step passes straight through: the decisions are logged with `acted: false` and
nothing else changes. Either way each decision, check and playback is a `supervisor` event in the run's
event log. A playback's event has the `start_frame` and `end_frame` of the motion it undid, as serve's
`rewind` event does, so `control.analysis` never trains on that motion.

What the runner owns, the layer leaves alone: it never ends, labels or resets an attempt, and anything
the runner does between `pause` and `run` (a person's Teleop, a serve Rewind) is outside it. After each
`run` its playback history starts again, so it never drives back past a pose something else set.

Its own failures stay its own: an exception in its watching or deciding is logged once, and from then
on every step passes straight through (asking for a person first if it was holding the arm). An exception
from the arm or the wrapped policy reaches the runner unchanged, as it would without the layer.
"""

import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from so_101.control import rewind as R
from so_101.supervisor import judges as J
from so_101.supervisor.detector import Detector, Signal
from so_101.supervisor.program import RULES, Program

MODES = ("off", "shadow", "act")
CAMERAS = ("top", "wrist")
FRAME_EVERY_S = 0.2  # frames kept for the checks at most this often
RING_S = 4.0  # frames kept for the watch check's lookback
MAX_STEP_S = 0.5  # a longer gap between steps (a stalled loop) counts as this much of the policy's time
POLICY, REWIND, HOLD = "policy", "rewind", "hold"  # what drives the arm on a step


@dataclass
class _Attempt:
    number: int
    detector: Detector
    history: R.History = field(default_factory=R.History)  # the policy's motion since the last `run`
    t: float = 0.0  # policy clock: seconds the policy has driven
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


class SupervisedPolicy:
    def __init__(self, inner, program: Program, judges: J.Judges, *, mode: str, log: Callable,
                 frame: Callable[[], int], cameras=CAMERAS, clock: Callable[[], float] = time.monotonic) -> None:
        """`inner` is the policy driver it wraps. `log(type, **fields)` writes an event (the recorder's
        `event`); `frame()` is the open episode's frame number this tick, for the playback events."""
        if mode not in ("shadow", "act"):
            raise ValueError(f"mode is {mode!r}: use shadow or act")
        self.inner, self.p, self.judges, self.mode = inner, program, judges, mode
        self._log, self._frame, self.clock = log, frame, clock
        self.cameras = tuple(c for c in CAMERAS if c in cameras)
        self.attempt: _Attempt | None = None
        self._inflight: list = []  # (future, kind, attempt number, close seq, extra)
        self._messages: deque[str] = deque()  # for the operator, through `maintain`
        self._asking: str | None = None  # for the runner to pause on, through `asks`
        self._wants: list[tuple] = []  # what the next step starts, in order: (REWIND, s), (HOLD,), (POLICY,)
        self.driving = POLICY
        self._rewind: R.Rewind | None = None
        self._last: float | None = None  # the clock at the last step since `run`
        self._pose: dict[str, float] = {}
        self._frames_at: float | None = None
        self._active = False  # between `run` and `pause`: the runner steps it
        self._broken: str | None = None  # why the layer stopped supervising, after an error of its own

    # --- The driver's interface, as the runner calls it --------------------------------------------

    def start(self) -> None:
        self.log("start", program=str(self.p.path), version=self.p.version,
                 judges={key: "ok" if self.judges.available(key) else self.judges.missing.get(key, "missing")
                         for key in self.judges.models})
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def close(self) -> None:
        """Stop the judges' pool; calls still running finish on their own."""
        self.judges.close()

    def maintain(self) -> str | None:
        """Called every tick, in every state. Collects the verdicts that came back; returns the wrapped
        driver's message, else the next of the layer's own."""
        message = self.inner.maintain()
        self._safely(self._collect)
        if message is not None:
            return message
        return self._messages.popleft() if self._messages else None

    def asks(self) -> str | None:
        """Why the layer wants a person now, once: the runner pauses on it. None otherwise."""
        asking, self._asking = self._asking, None
        return asking

    def begin_episode(self, number: int) -> None:
        self.attempt = None
        self._safely(self._begin, number)

    def end_episode(self) -> None:
        self._safely(self._end)
        self.attempt = None

    def pause(self) -> None:
        """The runner takes the arm: no action from the policy or from a playback reaches it after this."""
        self._active, self._wants, self._asking, self._last = False, [], None, None
        if self._rewind is not None:
            self._safely(self._finish_rewind, "paused")
        self.driving, self._rewind = POLICY, None
        self.inner.pause()

    def run(self) -> None:
        """The runner hands the arm to the layer, after something else may have moved it."""
        self.inner.run()
        self._active, self._wants, self._asking, self._last, self.driving = True, [], None, None, POLICY
        self._rewind = None
        if (a := self.attempt) is not None:
            a.history = R.History()
            a.stuck = None
            a.detector.resumed()

    def hold(self, robot) -> None:
        self.inner.hold(robot)

    @property
    def failure(self) -> str | None:
        return self.inner.failure

    @property
    def raised(self) -> str | None:
        return self.inner.raised

    @raised.setter
    def raised(self, value: str | None) -> None:
        self.inner.raised = value

    @property
    def retrying(self) -> str | None:
        return self.inner.retrying

    def step(self, robot, observation: dict) -> dict | None:
        """Send this tick's goal: the policy's action, the next of a playback's, or the hold. Returns the
        policy's action when it reached the arm, else None, so the runner never records a playback or
        a hold as the policy's."""
        now = self.clock()
        dt = 0.0 if self._last is None else min(max(now - self._last, 0.0), MAX_STEP_S)
        self._last = now
        self._safely(self._observe, observation, now)
        wants, self._wants = self._wants, []
        for want in wants if self.mode == "act" and self._broken is None else ():
            self._start(want)
        if self.driving == REWIND:
            arrived = self._rewind.step(robot, self._pose, dt)
            if arrived is not None:
                self._safely(self._finish_rewind, "arrived" if arrived else "stopped_short")
                if self._broken is None:  # else `_safely` left it holding for a person
                    self.driving = POLICY
                    self.inner.run()  # a clean slate: nothing the policy queued before the miss
                    if (a := self.attempt) is not None:
                        a.detector.resumed()
            return None
        if self.driving == HOLD:
            self.inner.hold(robot)
            return None
        action = self.inner.step(robot, observation)
        self._safely(self._watch_policy, dt, action is not None)
        return action

    def _start(self, want: tuple) -> None:
        """Start what a decision asked for. The wrapped driver's own calls are outside `_safely`: what
        they raise is theirs, for the runner."""
        a = self.attempt
        if want[0] == REWIND and a is not None and a.history.moments:
            self.inner.pause()
            self._rewind = R.Rewind(a.history, want[1], self.clock)
            self.driving = REWIND
        elif want[0] == HOLD and self.driving == POLICY:
            self.inner.pause()
            self.driving = HOLD
        elif want[0] == POLICY and self.driving == HOLD:
            self.inner.run()
            self.driving = POLICY
            if a is not None:
                a.detector.resumed()

    # --- Failures of its own -----------------------------------------------------------------------

    def _safely(self, fn, *args) -> None:
        """Run the layer's own logic. An exception is logged once, and the layer stops supervising:
        every step passes through from then on."""
        if self._broken is not None:
            return
        try:
            fn(*args)
        except Exception as error:
            self._broken = f"{type(error).__name__}: {error}"
            self.log("error", error=self._broken, traceback=traceback.format_exc()[-2000:])
            self._messages.append(f"Supervisor: stopped after an error ({self._broken}). The policy runs without it.")
            if self.driving != POLICY:  # it was holding or rewinding the arm: a person decides what's next
                self.driving, self._rewind = HOLD, None
                self._asking = "Supervisor: stopped while it held the arm. Resume runs the policy without it."

    # --- Logging -----------------------------------------------------------------------------------

    def log(self, kind: str, **fields) -> None:
        a = self.attempt
        context = {"for_attempt": a.number, "policy_t": round(a.t, 2)} if a is not None else {}
        self._log("supervisor", **{"mode": self.mode, **context, **fields, "kind": kind})

    def _note(self, line: str) -> None:
        if self.attempt is not None:
            self.attempt.log.append(f"t={self.attempt.t:.1f}s {line}")

    def _decide(self, decision: str, want: tuple | None = None, ask: str | None = None, tell: str | None = None,
                **fields) -> bool:
        """Log `decision` and, in act mode while the layer has the arm, start `want`, ask for a person
        with `ask`, and tell the operator `tell`. Returns whether it acted."""
        acting = self.mode == "act" and self._active
        self.log("decision", decision=decision, acted=acting, **fields)
        self._note(f"decision: {decision}")
        if acting:
            if want is not None:
                self._wants.append(want)
            if ask is not None:
                self._wants.append((HOLD,))
                self._asking = f"Supervisor: {ask}"
                self.attempt.asked = True
            if tell is not None:
                self._messages.append(f"Supervisor: {tell}")
        return acting

    # --- Watching ----------------------------------------------------------------------------------

    def _begin(self, number: int) -> None:
        self.attempt = _Attempt(number, Detector(self.p))
        self._frames_at = None
        self.log("attempt")

    def _end(self) -> None:
        if (a := self.attempt) is None:
            return
        d = a.detector
        self.log("attempt_end", closes=d.closes, regrasps=d.regrasps, carried=d.carried, released=d.released,
                 fired=sorted(d.fired), misses=a.misses, watches=a.watches, asked=a.asked,
                 spent_usd=round(self.judges.spent, 4))

    def _observe(self, observation: dict, now: float) -> None:
        self._pose = {k: float(v) for k, v in observation.items() if k.endswith(".pos")}
        if (a := self.attempt) is not None and (self._frames_at is None or now - self._frames_at >= FRAME_EVERY_S):
            frames = {name: observation[name] for name in self.cameras if name in observation}
            if frames:
                a.frames = frames
                a.ring.append((a.t, frames))
                while a.ring and a.ring[0][0] < a.t - RING_S:
                    a.ring.popleft()
                self._frames_at = now
        self._collect()

    def _watch_policy(self, dt: float, drove: bool) -> None:
        """A tick the policy drove (or waited for its next action): the detector, then any check due."""
        if (a := self.attempt) is None:
            return
        a.t += dt
        a.history.add(dt, dict(self._pose), self._frame())
        for signal in a.detector.step(a.t, self._pose, drove):
            self._signal(a, signal)
        self._due_checks(a)
        self._watch(a)

    def _signal(self, a: _Attempt, s: Signal) -> None:
        self.log(s.kind, **({"name": s.name} if s.name else {}), t=round(s.t, 2), **s.detail)
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
        if not now and (not a.detector.near() or a.t - a.last_watch_t < self.p.watch.period_s):
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
        near = a.stuck or ",".join(a.detector.near())
        future = self.judges.submit(lambda: self.judges.watch_check(current(), before()))
        self._inflight.append((future, "watch", a.number, a.close_seq, {"near": near}))

    # --- Deciding ----------------------------------------------------------------------------------

    def _stuck(self, a: _Attempt, why: str) -> None:
        if a.stuck or a.asked:
            return
        a.stuck = why
        self._decide(f"hold: {why}", want=(HOLD,), rule=why)
        self._watch(a, now=True)

    def _miss(self, a: _Attempt) -> None:
        a.misses += 1
        if a.misses <= len(self.p.miss_rewind_s):
            seconds = self.p.miss_rewind_s[a.misses - 1]
            self._decide(f"miss {a.misses}: rewind {seconds:g} s and retry", want=(REWIND, seconds),
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
            self.log("rewind", **R.finish(running, self._pose, cause))

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
                    self._decide("resume: the watch check saw nothing wrong", want=(POLICY,), rule=why)
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
                    self._decide("done: the block is in the container",
                                 ask="the block looks in the container. Reset and label the attempt.")


def _answered(value):
    from concurrent.futures import Future

    future = Future()
    future.set_result(value)
    return future
