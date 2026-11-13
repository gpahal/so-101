"""Connect the follower without a jump to an old goal.

LeRobot's follower `connect` runs `configure`, which turns every motor's torque off, writes its
settings, and turns the torque on again on the way out (`torque_disabled`). Each servo then drives to
its Goal_Position register at full speed, outside `max_relative_target`, which caps only the goals
software sends. That register keeps the last goal until the servo loses power. So after `serve`'s
`y`, which turns the motors off with the arm raised, and the arm lowered by hand, the next command to
connect snapped it back up to where it was.

`install` makes `configure` write each motor's Goal_Position as its Present_Position just before the
torque comes back on, so the arm holds where it is. Every command that connects the follower calls
it first: `serve` (`control/main.py`), `run` (`rollout.main`) and `check --live` in their process,
and `teleop`, `record`, `record --dry-run --replay` and `calibrate` through the `so101-teleoperate`,
`so101-record`, `so101-replay` and `so101-calibrate` entry points below, which run LeRobot's scripts.

A joint can rest a few steps past its calibrated limit: on 2026-11-13 the folded elbow read 3035 against a
Max_Position_Limit of 3031, because calibration never pushed it to its stop. The servo stores any goal
past its limits as the limit, so a goal of where it is can't read back. Up to OVERSHOOT_MAX_STEPS past,
the goal is the limit instead, and the torque pulls the joint that far back in. Further past, it refuses.
"""

import sys

_INSTALLED = "_so101_holds_where_it_is"


GOAL_TRIES = 3  # a goal write gets no reply, so a garbled one goes unnoticed: read it back, and try again
# How far past its calibrated limits a joint may rest, in raw steps (20 is 1.8°), and still connect: turning
# the torque on pulls it back to the limit at full speed, outside `max_relative_target`, so only a twitch is
# allowed. Further past, a hand moves it back, or a horn or gear slipped and the arm needs calibrating.
OVERSHOOT_MAX_STEPS = 20


def past_limits(present: dict[str, int], limits: dict[str, tuple[int, int]]) -> dict[str, int]:
    """How many steps each motor of `present` rests past its (min, max) limits, for the motors past them."""
    past = {}
    for motor, steps in present.items():
        low, high = limits[motor]
        if (over := max(low - steps, steps - high)) > 0:
            past[motor] = over
    return past


def describe_past(past: dict[str, int], limits: dict[str, tuple[int, int]], present: dict[str, int]) -> str:
    return ", ".join(
        f"{m} at {present[m]}, {n} steps ({n * 360 / 4096:.1f}°) past its limit {min(max(present[m], limits[m][0]), limits[m][1])}"
        for m, n in past.items()
    )


def read_limits(bus) -> dict[str, tuple[int, int]]:
    low = bus.sync_read("Min_Position_Limit", normalize=False, num_retry=2)
    high = bus.sync_read("Max_Position_Limit", normalize=False, num_retry=2)
    return {motor: (low[motor], high[motor]) for motor in low}


def hold_where_it_is(bus) -> None:
    """Aim every motor of `bus` at where it is, in raw steps, so turning its torque on moves nothing. A
    motor up to OVERSHOOT_MAX_STEPS past its position limits is aimed at the limit, which the servo would
    store anyway; one further past raises ConnectionError before any goal is written. A sync write gets
    no reply, so each motor's goal is read back. A motor whose goal still isn't its aim after GOAL_TRIES
    raises ConnectionError, before any torque is turned on: that motor would drive to its old goal."""
    limits = read_limits(bus)
    for _ in range(GOAL_TRIES):
        present = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        if far := {m: n for m, n in past_limits(present, limits).items() if n > OVERSHOOT_MAX_STEPS}:
            raise ConnectionError(
                f"{describe_past(far, limits, present)}, more than the {OVERSHOOT_MAX_STEPS} steps connecting pulls "
                "back, so its torque stays off. Move it back inside its range by hand and try again; if it reads "
                "past its limit again, a horn or gear may have slipped: recalibrate the arm.")
        aim = {m: min(max(steps, limits[m][0]), limits[m][1]) for m, steps in present.items()}
        bus.sync_write("Goal_Position", aim, normalize=False)
        goals = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
        if not (missed := [motor for motor, steps in aim.items() if goals[motor] != steps]):
            return
    raise ConnectionError(f"Aiming {', '.join(missed)} where it is didn't take after {GOAL_TRIES} tries, so its "
                          "torque stays off rather than drive it to an old goal. Check the arm's cables and power, "
                          "then try again.")


def install() -> None:
    """Make the follower's `configure` aim each motor at where it is before its torque comes back on."""
    from lerobot.robots.so_follower.so_follower import SOFollower

    configure = SOFollower.configure
    if getattr(configure, _INSTALLED, False):
        return

    def configure_in_place(self) -> None:
        bus = self.bus
        enable_torque = bus.enable_torque

        def enable_where_it_is(*args, **kwargs) -> None:
            hold_where_it_is(bus)
            enable_torque(*args, **kwargs)

        bus.enable_torque = enable_where_it_is  # `torque_disabled` turns the torque on through it
        try:
            configure(self)
        finally:
            del bus.enable_torque

    setattr(configure_in_place, _INSTALLED, True)
    SOFollower.configure = configure_in_place


def teleoperate() -> None:
    """`lerobot-teleoperate`, connecting the follower where it is."""
    from lerobot.scripts.lerobot_teleoperate import main

    install()
    sys.exit(main())


def record() -> None:
    """`lerobot-record`, connecting the follower where it is. It records video, so the SVT-AV1 fix
    loads first, before LeRobot starts any thread."""
    from so_101 import svtshim

    svtshim.ensure()  # started by `so101`, the fix is already loaded; run directly, it restarts itself
    svtshim.require()
    from lerobot.scripts.lerobot_record import main

    install()
    sys.exit(main())


def replay() -> None:
    """`lerobot-replay`, connecting the follower where it is: a dry run's replay starts from a pose
    moved by hand, while each servo still holds the recording's last goal."""
    from lerobot.scripts.lerobot_replay import main

    install()
    sys.exit(main())


def calibrate() -> None:
    """`lerobot-calibrate`, connecting the follower where it is before calibration turns its torque off."""
    from lerobot.scripts.lerobot_calibrate import main

    install()
    sys.exit(main())
