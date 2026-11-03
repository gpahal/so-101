"""Build LeRobot arm objects from `so101.toml`, for use in your own scripts."""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from so_101 import settings


def make_follower() -> SO101Follower:
    s = settings.load()
    arm = s.follower
    return SO101Follower(
        SO101FollowerConfig(
            port=arm.resolve_port(),
            id=arm.id,
            calibration_dir=arm.calibration_dir,
            max_relative_target=s.teleop.max_relative_target or None,
        )
    )


def make_leader() -> SO101Leader:
    arm = settings.load().leader
    return SO101Leader(SO101LeaderConfig(port=arm.resolve_port(), id=arm.id, calibration_dir=arm.calibration_dir))
