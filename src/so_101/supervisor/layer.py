"""The supervisor as a layer of the policy stack (docs/policy-layers.md): it watches each attempt and steps in.

It sits over the perception (and the skill sequencer, for a skills model), and each tick either passes
the tick down or drives the arm itself:

- **Watch.** It reads the perception's signals and readings in each Output that comes up. At the
  program's moments the judges look at the camera frames, on their own worker pool: each judge is one
  future, sent from the loop's thread, collected on later ticks, so a step never waits for one. A check
  is settled as soon as the answers in can't be overturned (`judges.decided`), and a judge that hasn't
  answered by `answer_within_s` counts as no answer. Settling cancels the check's calls still queued,
  and each request times out at `answer_within_s` too, so a stale check neither bills nor holds a worker.
- **Rewind and retry.** A miss (an after_close check that both judges call wrong) plays back
  `miss_rewind_s[n]` of the model's own measured motion, newest first (`control.rewind`, the playback a
  person's Rewind uses), on the model's clock (1/fps per tick it drove, as the perception's), so it
  reaches the close the check was for at any loop speed. Then it hands the arm back with
  `Handback(LAYER, moved=True)`, so the perception's detector forgets the close it undid. A playback
  that stops short (a joint still more than `rollout.TOLERANCE` off after `REWIND_SETTLE_S`) hands
  nothing back: it holds where it stopped and asks for a person, as a person's Rewind does. A miss past
  that list counts as stuck. While the list has rungs left, both judges answer and the budget lasts, a
  `regrasp_loop` rule is left to it (`_ladder_open`).
- **Hold and check.** A stuck rule holds the arm and asks the watch check. A sure ok hands it back with
  `Handback(LAYER, moved=False)`; anything else, or no answer within `hold_deadline_s`, asks for a person
  (`Output.asks`), and the escalation model's advice for them follows as a message (`maintain`), once an
  attempt. A `pause` ends a hold: whoever paused has the arm, so its deadline asks nobody and its verdict
  is stale. Once the runner hands the arm back (`run`), after an ask or not, the rules not yet fired, the
  miss ladder and the done check watch again; each rule still fires once an attempt.
- **Done.** A release that both judges are sure put the block in the container is logged, and the
  policy drives on until the arm is at rest (`Tick.at_rest`) or `home_within_s` of its time has passed;
  then the layer asks for a person with kind `done`: only a person, or `run`'s autopilot, ends or labels
  an attempt.

Every verdict carries the generation it was asked in, raised by `pause`, `run` and each hold: a watch
verdict settles only the hold it was asked for, and an after_close verdict is dropped after a pause or
resume, a newer close, a carry or release since it was asked, or once it is `after_close_answer_s` of
the model's time old, when the first playback could not reach back past the close. A check not
yet sent is dropped at a pause or resume too, and an after_close check not yet sent when a newer close
comes is dropped unsent; every check keeps the close it was scheduled for. Those scheduling rules are
`supervisor.schedule.Schedule`, which `failbench.candidates` shares. A handback that moved the arm
forgets a confirmed release, so nothing about the scene before a handback is acted on after it.

In `shadow` mode every tick passes straight through: the decisions are logged with `acted: false` and
nothing else changes. Either way each decision and check is a `supervisor` event, and each playback a
`playback` event with the `start_frame` and `end_frame` of the motion it undid, which `control.analysis`
never trains on, whichever layer wrote it. Each playback tick's Output names the frame it goes back to
(`goal_frame`), so the loop drops what it undid from a person's Rewind history.

Each check sends the newest frames, the tick's own, so it judges the picture `so101-failbench candidates`
scores for the same moment; only the watch check's lookback ring keeps one every `FRAME_EVERY_S`. That ring
(`FRAME_EVERY_S`, `lookback_s`), `hold_deadline_s` and `answer_within_s` are on the loop's clock instead:
they're about real time, a person's and a provider's, not the model's.

What the runner owns, it leaves alone: it never ends, labels or resets an attempt, and anything between
`pause` and `run` (a person's Teleop or Rewind) is outside it. After each handback its playback
history starts again, so it never drives back past a pose something else set.
"""

from collections import deque
from dataclasses import dataclass, field, replace

from so_101 import rollout
from so_101.control import rewind as R
from so_101.grasp.thresholds import RULES
from so_101.policy.base import (DONE, HOLD, LAYER, PLAYBACK, STUCK, Ask, Handback, Layer, LayerContext, Output,
                                Readings, Tick)
from so_101.supervisor import judges as J
from so_101.supervisor.program import RING_S
from so_101.supervisor.schedule import Schedule

MODES = ("off", "shadow", "act")
CAMERAS = ("top", "wrist")
FRAME_EVERY_S = 0.2  # frames kept in the watch check's lookback ring at most this often
FORGET_AFTER_S = 300.0  # a call still out this long after its check settled isn't waited for, even to log
PASS, REWINDING, HOLDING = "pass", "rewinding", "holding"  # what the layer is doing with the arm
EVENT_ROLES = ("decision", "chat")


@dataclass
class _Check:
    """One check out with the judges: what it was asked about, and the answers in so far."""

    moment: str  # after_close, after_release, watch or escalation
    number: int  # the attempt
    generation: int  # the layer's generation when asked
    asked_t: float  # the model's seconds when asked
    asked_at: float  # the loop's clock when asked
    futures: dict  # role -> Future
    seq: int = 0  # the close it's about
    progress: int = 0  # carries and releases seen when asked
    extra: dict = field(default_factory=dict)
    answers: dict = field(default_factory=dict)  # role -> Verdict, or the escalation's dict
    settled: bool = False
    pending: tuple = ()  # the roles still out when it settled, logged as they come in


@dataclass
class _Attempt:
    number: int
    schedule: Schedule  # the after_close and after_release checks not yet sent, and the closes so far
    history: R.History = field(default_factory=R.History)  # the model's motion since the last handback
    readings: Readings = field(default_factory=Readings)  # the perception's, as of the model's last tick
    frames: dict | None = None  # the newest frames
    ring: deque = field(default_factory=deque)  # (loop clock, frames), for the watch check's lookback
    closes: list = field(default_factory=list)  # (caption, frames) for the contact sheet
    log: list[str] = field(default_factory=list)
    progress: int = 0  # carries and releases so far
    misses: int = 0
    watches: int = 0
    stuck: str | None = None  # the rule holding the arm, waiting for the watch check
    hold_generation: int = -1  # the generation the hold began, which its watch check carries
    hold_at: float = 0.0  # the loop's clock when it began
    judges_silent: bool = False  # the last after_close check settled without both judges' answers
    released_t: float | None = None  # the model's seconds at the release
    done: bool = False  # a release the judges are sure of: ask once the arm is home
    asked: bool = False  # it has asked for a person in this attempt, at least once (`attempt_end` logs it)
    waiting: bool = False  # it has asked for a person since the runner last handed the stack the arm
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
        self.generation = 0
        self._inflight: list[_Check] = []
        self._asking: Ask | None = None  # for the runner to pause on, through Output.asks
        self._wants: list[tuple] = []  # what the next step starts, in order: (REWINDING, s), (HOLDING,), (PASS,)
        self.doing = PASS
        self._rewind: R.Rewind | None = None
        self._stepped = False  # a tick was stepped since the last handback (`run`) or `pause`
        self._pose: dict[str, float] = {}
        self._frames_at: float | None = None
        self._now = 0.0  # the loop's clock at the last tick
        self._active = False  # between `run` and `pause`: the runner steps it
        self._told_budget = False

    # --- The policy's methods, as the runner calls them -------------------------------------------------

    def start(self) -> None:
        self.safely(self._announce_judges)
        self.inner.start()

    def stop(self) -> None:
        self.judges.close()
        self.inner.stop()

    def close(self) -> None:
        """Stop the judges' pool: queued calls are cancelled, and a call still running finishes, or hangs,
        on a daemon thread that never holds up the process's exit."""
        self.judges.close()
        self.inner.close()

    def maintain(self) -> list[str]:
        messages = self.inner.maintain()
        self.safely(self._collect)
        self.safely(self._check_budget)
        return [*messages, *self.said()]

    def begin_episode(self, number: int) -> None:
        self.inner.begin_episode(number)
        self.attempt = None
        self.safely(self._begin, number)

    def end_episode(self) -> None:
        self.safely(self._end)
        self.attempt = None
        self.inner.end_episode()

    def pause(self) -> None:
        """The runner takes the arm: no action from the model or from a playback reaches it after this. A
        hold ends here: whoever paused has the arm, so its deadline asks nobody and its watch check is stale."""
        self._active, self._wants, self._asking, self._stepped = False, [], None, False
        self.generation += 1
        self.safely(self._end_hold)
        self.safely(self._drop_scheduled)
        if self._rewind is not None:
            self.safely(self._finish_rewind, "paused")
        self.doing, self._rewind = PASS, None
        self.inner.pause()

    def run(self, handback: Handback) -> None:
        """The runner hands the arm to the stack, after something else may have moved it. A person has
        answered any ask, so the layer watches and may ask again. A release the judges confirmed stays done
        only if nothing moved the arm since: a person may have undone it."""
        self.inner.run(handback)
        self._active, self._wants, self._asking, self._stepped, self.doing = True, [], None, False, PASS
        self.generation += 1
        self._rewind = None
        self.safely(self._drop_scheduled)
        if (a := self.attempt) is not None:
            a.history = R.History()
            a.stuck, a.waiting = None, False  # a person answered any ask: the rules not yet fired watch again
            if handback.moved:
                a.done, a.released_t = False, None

    def step(self, robot, tick: Tick) -> Output:
        """This tick's goal: the stack below's, the next of a playback's, or the hold. The layer's own
        logic runs in `safely`; every call to the stack below, and the playback's goals to the arm, run
        outside it, so what they raise is theirs, for the runner."""
        # The model's clock, as the perception's: 1/fps a tick, the first after a handback adding nothing.
        dt = 1 / self.ctx.fps if self._stepped else 0.0
        self._stepped = True
        self.safely(self._observe, tick)
        wants, self._wants = self._wants, []
        for want in wants if self.mode == "act" and self.broken is None else ():
            made: list = []
            if not self.safely(self._plan, want, made) or not made:
                continue
            kind, rewind = made[0]
            if kind == REWINDING:
                self.inner.pause()
                self._rewind, self.doing = rewind, REWINDING
            elif kind == HOLDING:
                self.inner.pause()
                self.doing = HOLDING
            else:
                self.doing = PASS
                self.inner.run(Handback(LAYER, moved=False))
        asking, self._asking = self._asking, None
        if self.doing == REWINDING:
            arrived = self._rewind.step(robot, self._pose, dt)
            goal_frame = self._rewind.start_frame  # read before `_finish_rewind` lets the playback go
            if arrived:
                self.safely(self._finish_rewind, "arrived")
                if self.broken is None:  # else `failed` left it holding for a person
                    self.doing = PASS
                    self.inner.run(Handback(LAYER, moved=True))  # a clean slate, from where the playback left the arm
            elif arrived is False:  # something blocks the arm: it holds where it stopped, as after a person's Rewind
                joint, off = rollout.farthest(self._pose, self._rewind.target)  # before `_finish_rewind` lets it go
                self.safely(self._finish_rewind, "stopped_short")
                if self.broken is None:  # else `failed` left it holding and asked for a person
                    self.doing = HOLDING
                    self.safely(self._stopped_short, joint.removesuffix(".pos"), off)
            later, self._asking = self._asking, None  # `failed`'s, if the playback's end broke the layer
            return Output(None, PLAYBACK, asks=asking or later, goal_frame=goal_frame)
        if self.doing == HOLDING:
            self.inner.hold(robot)
            return Output(None, HOLD, asks=asking)
        out = self.inner.step(robot, tick)
        self.safely(self._watch_model, dt, tick, out)
        later, self._asking = self._asking, None
        return replace(out, asks=out.asks or asking or later)

    def _plan(self, want: tuple, made: list) -> None:
        """What a decision asks the next step to start, as (kind, the playback): nothing when it doesn't apply."""
        a = self.attempt
        if want[0] == REWINDING and a is not None and a.history.moments:
            made.append((REWINDING, R.Rewind(a.history, want[1], self.ctx.clock)))
        elif want[0] == HOLDING and self.doing == PASS:
            made.append((HOLDING, None))
        elif want[0] == PASS and self.doing == HOLDING:
            made.append((PASS, None))

    def failed(self) -> None:
        if self.doing != PASS:  # it was holding or playing back: a person decides what's next
            self.doing, self._rewind = HOLDING, None
            self._asking = Ask(LAYER, "Supervisor: stopped while it held the arm. Resume runs the policy without it.")

    # --- Logging and telling ------------------------------------------------------------------------

    def log(self, kind: str, **fields) -> None:
        a = self.attempt
        context = {"for_attempt": a.number, "policy_t": round(a.t, 2)} if a is not None else {}
        self.ctx.event("supervisor", **{"mode": self.mode, **context, **fields, "kind": kind})

    def _note(self, line: str) -> None:
        if self.attempt is not None:
            self.attempt.log.append(f"t={self.attempt.t:.1f}s {line}")

    def _announce_judges(self) -> None:
        j = self.p.judges
        self.log("start", program=str(self.p.path), version=self.p.version,
                 judges={key: "ok" if self.judges.available(key) else self.judges.missing.get(key, "not available")
                         for key in self.judges.models})
        for key in dict.fromkeys((j.decision, j.chat, j.escalation)):
            if self.judges.available(key):
                continue
            why = self.judges.missing.get(key, "not available")
            then = ("so a stuck attempt gets no advice" if key == j.escalation else
                    f"so a missed grasp is never played back: that needs both {j.decision} and {j.chat} to call it "
                    "wrong" + (", and a hold asks for a person at once" if key == j.decision and
                               not self.judges.available(j.chat) else ""))
            self.say(f"Supervisor: runs without {key} ({why}), {then}.")

    def _check_budget(self) -> None:
        if self._told_budget or (why := self.judges.out_of_budget()) is None:
            return
        self._told_budget = True
        self.say(f"Supervisor: {why} in {self.p.path}. It no longer checks or plays back a miss, and a stuck rule "
                 "asks for a person at once.")

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
                self.attempt.asked = self.attempt.waiting = True
            if tell is not None:
                self.say(f"Supervisor: {tell}")
        return acting

    # --- Watching ----------------------------------------------------------------------------------

    def _begin(self, number: int) -> None:
        self.attempt = _Attempt(number, Schedule(self.p.after_close_s, self.p.after_release_s))
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
        self._now = tick.now
        if (a := self.attempt) is not None:
            frames = {name: tick.observation[name] for name in self.cameras if name in tick.observation}
            if frames:
                # Encoded only when a check is sent (`_images`); the ring keeps one every FRAME_EVERY_S.
                a.frames = frames
                if self._frames_at is None or tick.now - self._frames_at >= FRAME_EVERY_S:
                    a.ring.append((tick.now, frames))
                    while a.ring and a.ring[0][0] < tick.now - RING_S:
                        a.ring.popleft()
                    self._frames_at = tick.now
        self._collect()

    def _watch_model(self, dt: float, tick: Tick, out: Output) -> None:
        """A tick the model drove (or waited for its next action): its motion for playback, the
        perception's signals, then any check due, and a done ask once the arm is home."""
        if (a := self.attempt) is None or out.readings is None:
            return
        a.readings = out.readings
        a.history.add(dt, dict(self._pose), tick.frame)
        for signal in out.signals:
            self._signal(a, signal)
        self._due_checks(a)
        if a.done and not a.waiting and (tick.at_rest or a.t - a.released_t >= self.p.home_within_s):
            a.done = False
            self._decide("done: the block is in the container", ask_kind=DONE,
                         ask="the block looks in the container. Reset and label the attempt.",
                         home="at_rest" if tick.at_rest else "home_within_s")

    def _signal(self, a: _Attempt, s) -> None:
        self._note(f"{s.kind}{' ' + s.name if s.name else ''} {s.detail}")
        a.schedule.signal(s.kind, s.t)
        if s.kind == "carry":
            a.progress += 1
        elif s.kind == "release":
            a.progress += 1
            a.released_t = s.t
        elif s.kind == "rule":
            self._stuck(a, s.name)

    def _images(self, frames: dict | None):
        """A function the judges' pool runs to encode `frames` (once, however many judges call it), or None
        without any."""
        if not frames:
            return None
        encoded: list = []

        def images():
            if not encoded:
                encoded.append([(name, J.jpeg(frames[name])) for name in CAMERAS if name in frames])
            return encoded[0]

        return images

    def _send(self, moment: str, a: _Attempt, futures: dict, seq: int | None = None, **extra) -> None:
        """Put a check out, about close `seq` (by default the newest close)."""
        self._inflight.append(_Check(moment, a.number, self.generation, a.t, self.ctx.clock(), futures,
                                     seq=a.schedule.closes if seq is None else seq, progress=a.progress, extra=extra))

    def _due_checks(self, a: _Attempt) -> None:
        j = self.p.judges
        for moment, seq, stale in a.schedule.due(a.t):
            if stale:  # a newer close came: its own check is scheduled
                self.log("check", moment=moment, stale="newer_close", sent=False)
                continue
            if (images := self._images(a.frames)) is None:
                self.log("check", moment=moment, error="no camera frames")
                continue
            futures = {role: self.judges.submit(lambda key=key, images=images, moment=moment:
                                                self.judges.ask(key, images(), moment))
                       for role, key in zip(EVENT_ROLES, (j.decision, j.chat), strict=True)}
            self._send(moment, a, futures, seq=seq, frames=a.frames)

    def _lookback(self, a: _Attempt) -> dict | None:
        target = self._now - self.p.watch.lookback_s
        older = [frames for t, frames in a.ring if t <= target]
        return older[-1] if older else (a.ring[0][1] if a.ring else None)

    def _ask_watch(self, a: _Attempt) -> None:
        """The watch check for the hold just begun: each camera now and `lookback_s` before."""
        current, before = self._images(a.frames), self._images(self._lookback(a))
        if current is None or before is None:  # a held rule must still be settled: with nothing to look at, a person does
            futures = {"watch": _answered(J.Verdict("watch", error="no camera frames"))}
        else:
            a.watches += 1
            futures = {"watch": self.judges.submit(lambda: self.judges.watch_check(current(), before()))}
        self._send("watch", a, futures, near=a.stuck)

    # --- Deciding ----------------------------------------------------------------------------------

    def _ladder_open(self, a: _Attempt) -> bool:
        """The miss ladder has rungs left and both judges answer: a regrasp loop is its to handle."""
        j = self.p.judges
        return (a.misses < len(self.p.miss_rewind_s) and self.judges.available(j.decision)
                and self.judges.available(j.chat) and self.judges.out_of_budget() is None and not a.judges_silent)

    def _stuck(self, a: _Attempt, why: str) -> None:
        if a.stuck or a.waiting:
            return
        if why == "regrasp_loop" and self._ladder_open(a):
            self.log("ignored", rule=why, why="the miss ladder has rungs left and both judges answer")
            self._note(f"ignored {why}: the miss ladder handles it")
            return
        self.generation += 1
        a.stuck, a.hold_generation, a.hold_at = why, self.generation, self.ctx.clock()
        self._decide(f"hold: {why}", want=(HOLDING,), rule=why)
        self._ask_watch(a)

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
        a.stuck = None
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
        self._send("escalation", a, {"escalation": future})

    def _stopped_short(self, joint: str, off: float) -> None:
        """A playback ended more than `rollout.TOLERANCE` from the pose it went back to: the model never
        drives on from a pose it didn't reach. The arm holds there and a person is asked, as when a
        person's Rewind stops short."""
        if (a := self.attempt) is not None:
            self._escalate(a, f"the playback stopped short: {joint} is {off:.0f} away from the pose it went back to")

    def _end_hold(self) -> None:
        """A pause took the arm while a stuck rule held it: the hold is over, with no ask and no escalation.
        The rule stays spent for the attempt (the detector fires each once), so after Resume the policy
        drives and only the other rules and the miss ladder still watch: the person saw it stuck and handed
        it back."""
        if (a := self.attempt) is None or a.stuck is None:
            return
        self.log("hold_ended", rule=a.stuck, why="paused")
        self._note(f"hold on {a.stuck} ended: paused")
        a.stuck = None

    def _drop_scheduled(self) -> None:
        """A pause or resume drops the checks due but not yet sent, as it does the verdicts still out: each
        was about the scene before the handback, so it would judge, and could play back, motion since."""
        if (a := self.attempt) is None:
            return
        for moment in a.schedule.drop():
            self.log("check", moment=moment, stale="paused_or_resumed", sent=False)

    def _finish_rewind(self, cause: str) -> None:
        running, self._rewind = self._rewind, None
        if running is not None:
            a = self.attempt
            self.ctx.event("playback", layer=self.name, **({"for_attempt": a.number} if a else {}),
                           **R.finish(running, self._pose, cause))

    # --- Collecting verdicts ---------------------------------------------------------------------------

    def _collect(self) -> None:
        """Every answer that has come back since the last call, each check settled once it can be, and a
        hold past its deadline."""
        now = self.ctx.clock()
        for check in list(self._inflight):
            for role, future in check.futures.items():
                if role not in check.answers and future.done():
                    check.answers[role] = _result(future, role)
                    if check.settled and role in check.pending:
                        self.log("late_answer", moment=check.moment, role=role, **_record(check.answers[role]))
            if not check.settled:
                self._consider(check, now)
            if check.settled and (len(check.answers) == len(check.futures) or now - check.asked_at > FORGET_AFTER_S):
                self._inflight.remove(check)
        a = self.attempt
        if a is not None and a.stuck and now - a.hold_at >= self.p.hold_deadline_s:
            self._escalate(a, f"stuck ({a.stuck.replace('_', ' ')}): the watch check didn't clear it within "
                              f"{self.p.hold_deadline_s:g} s")

    def _settle(self, check: _Check, **fields) -> None:
        """Log the check as settled. A call still out that hasn't started is cancelled, so it
        neither bills nor keeps a worker; one already running finishes, and is logged as a late answer."""
        check.settled = True
        cancelled = [role for role, future in check.futures.items() if role not in check.answers and future.cancel()]
        for role in cancelled:
            check.answers[role] = _result(check.futures[role], role)
        check.pending = tuple(role for role in check.futures if role not in check.answers)
        pending = {"pending": list(check.pending)} if check.pending else {}
        dropped = {"cancelled": cancelled} if cancelled else {}
        fields = {key: value for key, value in fields.items() if value is not None}
        self.log("check" if check.moment != "escalation" else "escalation", moment=check.moment, **fields, **pending,
                 **dropped)

    def _stale(self, check: _Check, a: _Attempt | None) -> str | None:
        """Why a check's answer doesn't apply, or None."""
        if a is None or a.number != check.number:
            return "other_attempt"
        if check.moment == "watch":
            return None if a.stuck is not None and check.generation == a.hold_generation == self.generation else "hold_ended"
        if check.generation != self.generation:
            return "paused_or_resumed"
        if check.moment == "after_close":
            if check.seq != a.schedule.closes:
                return "newer_close"
            if check.progress != a.progress:
                return "carried_or_released"
            if a.t - check.asked_t > self.p.after_close_answer_s:
                return "too_old"
        return None

    def _consider(self, check: _Check, now: float) -> None:
        a = self.attempt
        timed_out = now - check.asked_at >= self.p.judges.answer_within_s
        if check.moment == "escalation":
            if "escalation" not in check.answers:
                return
            answer = check.answers["escalation"]
            stale = a is None or a.number != check.number
            self._settle(check, **answer, **({"stale": "other_attempt"} if stale else {}))
            if "choice" in answer and not stale and self.mode == "act":
                self.say(f"Supervisor's advice: {J.CHOICES[answer['choice']]} ({answer.get('reason', '')})")
            return
        if (why := self._stale(check, a)) is not None:
            if why == "too_old" and len(check.answers) < len(check.futures):
                a.judges_silent = True
            self._settle(check, stale=why, waited_s=round(a.t - check.asked_t, 2) if a and why == "too_old" else None,
                         **{role: _record(v) for role, v in check.answers.items()})
            return
        if check.moment == "watch":
            verdict = check.answers.get("watch")
            if verdict is None and not timed_out:
                return
            verdict = verdict or J.Verdict("watch", error=f"no answer within {self.p.judges.answer_within_s:g} s")
            call = verdict.call(self.p)
            self._settle(check, near=check.extra["near"], call=call, decision=verdict.record())
            self._note(f"watch check: {call}")
            why = a.stuck
            if call == J.OK and why in RULES:  # a run of misses always goes to a person
                a.stuck = None
                self._decide("resume: the watch check saw nothing wrong", want=(PASS,), rule=why)
            else:
                self._escalate(a, f"stuck ({why.replace('_', ' ')})")
            return
        decision, chat = (check.answers.get(role) for role in EVENT_ROLES)
        outcome = J.both_wrong if check.moment == "after_close" else _release_outcome
        if not (timed_out or J.decided(outcome, self.p, decision, chat)):
            return
        fields = {"decision": _record(decision), "chat": _record(chat)}
        if check.moment == "after_close":
            call = J.agree(self.p, decision, chat)
            # Silent: a judge failed, or one was still out when the check timed out.
            a.judges_silent = any(v is not None and v.p_wrong is None for v in (decision, chat)) or \
                (timed_out and None in (decision, chat))
            self._settle(check, call=call, **fields)
            self._note(f"after_close check: {call}")
            a.closes.append((f"close {check.seq}: {call}", check.extra["frames"]))
            if J.both_wrong(self.p, decision, chat):
                self._miss(a)
            return
        call, success = _release_outcome(self.p, decision, chat)
        self._settle(check, call=call, success=success, **fields)
        self._note(f"after_release check: {call}")
        if call == J.WRONG:
            self._decide("tell the operator: the release missed the container",
                         tell="the block may have missed the container.")
        elif success and not a.waiting:
            a.done = True  # asked for once the arm is home (`_watch_model`)


def _release_outcome(program, decision, chat) -> tuple[str, bool]:
    return J.combine(program, decision, chat), J.success(program, decision, chat)


def _result(future, role: str):
    """A finished future's answer: a call cancelled, or raising where the judges never do, is an error verdict."""
    if future.cancelled():
        return J.Verdict(role, error="cancelled") if role != "escalation" else {"error": "cancelled"}
    if (error := future.exception()) is not None:
        message = f"{type(error).__name__}: {error}"[:300]
        return J.Verdict(role, error=message) if role != "escalation" else {"error": message}
    return future.result()


def _record(verdict) -> dict | None:
    return verdict.record() if isinstance(verdict, J.Verdict) else verdict


def _answered(value):
    from concurrent.futures import Future

    future = Future()
    future.set_result(value)
    return future
