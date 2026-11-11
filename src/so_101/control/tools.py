"""Run checks and Check pose: read-only tools the control loop runs inside a tick, on the devices it has open.

They run inside the loop, so no serial port is ever opened twice. Run checks takes about 3 s, during
which the loop sends no goal, so it runs only while the arm holds still: Resting, Paused or Fault
(`states.tool_refusal`). Check pose compares the follower's last measured pose with the demos'
start pose, so it costs nothing.
"""

import traceback
from collections.abc import Callable

from so_101 import rollout
from so_101.control.follower import Follower
from so_101.control.leader import Leader
from so_101.control.states import LEADER_MISSING
from so_101.report import Report


def pose_report(pose: dict[str, float], start_pose: dict[str, float] | None) -> Report:
    """`so101 pose` as a report: each joint of `pose` against the demos' start pose, both in the
    arm's units and keyed like its actions. A joint more than `rollout.TOLERANCE` off is a WARN:
    Resume drives it there before the policy starts."""
    report = Report()
    if start_pose is None:
        for key, position in pose.items():
            report.ok(key.removesuffix(".pos"), f"now {position:.1f}")
        report.warn("start pose", "no build to take the demos' start pose from, so an attempt starts from wherever "
                    "the arm is", fix="make one with `uv run so101 build`")
        return report
    for key, demo in start_pose.items():
        difference = pose[key] - demo
        detail = f"now {pose[key]:.1f}, demos {demo:.1f}, diff {difference:.1f}"
        if abs(difference) <= rollout.TOLERANCE:
            report.ok(key.removesuffix(".pos"), detail)
        else:
            report.warn(key.removesuffix(".pos"), f"{detail}: more than {rollout.TOLERANCE:g} from the demos' start pose",
                        fix="Resume moves it to the start pose before the policy starts")
    return report


def run_checks(checks: Callable[[dict, dict], Report], follower: Follower, leader: Leader,
               leader_reason: str | None) -> Report:
    """Run `checks(buses, cameras)`, `so101 check`'s read-only checks, on the open devices: the
    follower's bus, the leader's when it's connected, and the cameras. A leader that isn't connected
    is a WARN with `leader_reason`, why it isn't. A check that raises is a FAIL."""
    buses = {"follower": follower.inner.bus}
    if leader.connected:
        buses["leader"] = leader.arm.bus
    try:
        report = checks(buses, follower.cameras())
    except Exception as error:
        report = Report()
        report.fail("checks", f"the checks stopped: {type(error).__name__}: {error}",
                    fix="see the terminal; run `uv run so101 check` after stopping the server")
        traceback.print_exc()
    if not leader.connected:
        report.warn("leader", leader_reason or LEADER_MISSING,
                    fix="not checked: the checks use only the leader the server has connected")
    return report
