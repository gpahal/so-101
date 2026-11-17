"""Grasps from the arm's own readings: the gripper events and stuck rules every run's perception sees.

Modules:
- `thresholds` the gripper and stuck-rule thresholds, from `grasp.toml` beside this file or the file
               `[perception] thresholds` in so101.toml names
- `detector`   the arm's readings into gripper events and stuck rules; pure, so the perception steps it
               live, `so101 skills` and `build --skills` cut demos with it, and `so101-failbench
               candidates` replays recorded runs through it
"""
