"""Connect the follower without a jump to an old goal.

LeRobot's follower `connect` runs `configure`, which turns every motor's torque off, writes its
settings, and turns the torque on again on the way out (`torque_disabled`). Each servo then drives to
its Goal_Position register at full speed, outside `max_relative_target`, which caps only the goals
software sends. That register keeps the last goal until the servo loses power. So after `serve`'s
`y`, which turns the motors off with the arm raised, and the arm lowered by hand, the next command to
connect snapped it back up to where it was.

`install` makes `configure` write each motor's Goal_Position as its Present_Position just before the
torque comes back on, so the arm holds where it is. Every command that connects the follower calls
it first: `serve` (`control/main.py`), `run` (`rollout.main`), and `teleop` and `record` through the
`so101-teleoperate` and `so101-record` entry points below, which run LeRobot's scripts.
"""

import sys

_INSTALLED = "_so101_holds_where_it_is"


def hold_where_it_is(bus) -> None:
    """Aim every motor of `bus` at where it is, in raw steps, so turning its torque on moves nothing."""
    bus.sync_write("Goal_Position", bus.sync_read("Present_Position", normalize=False, num_retry=2), normalize=False)


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
