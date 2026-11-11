# Hardware notes

Status: reference for debugging, and plans for hardware changes nobody has built yet. What `so101` already does about each item is marked. Written in November 2026 from Feetech's STS memory table, LeRobot 0.7.0's motor code and this repo's tests. Items marked "untested" haven't been tried on this arm.

## The servos

The SO-101 uses Feetech STS3215 servos on a Waveshare bus servo board. Register addresses below are Feetech's; LeRobot and rustypot name them differently, as noted.

### No command timeout

The STS3215 has no watchdog or communication timeout: none of its registers (0 to 86) is one. If the host process dies, each servo keeps driving to the last goal it received, then holds it with torque on. Only cutting its power makes it go limp.

- **A clean exit** (Esc, Ctrl+C, SIGTERM, or SIGHUP from closing its terminal) turns torque off: `serve` first drives the arm to rest. Where it can't (after a fault, after a move to rest that stopped short, or resting away from the rest pose), only `y` at the terminal exits there, and no signal does, so with no terminal the arm holds until someone switches its power off.
- **A crash or `kill -9`** skips that. The arm most likely finishes its last move and holds there until someone switches the power off. Untested. To settle it, run `kill -9` on `so101-control` while the arm moves to the start pose, again while the policy drives, and pull the follower's USB cable mid-move, with a hand under the arm each time. Then note whether the arm stops and holds, finishes its last move and holds, or goes limp.
- **For operators away from the robot**, see [Stopping the arm](#stopping-the-arm).

### Power loss and restarts

A servo that loses power, or restarts in a brownout, comes back with torque off. Writing a goal turns its torque back on by itself (writing Goal_Position, address 42, sets Torque_Enable, 40, to 1), so the next goal drives it there at full speed.

`serve` handles this (README, "Devices and faults"):
- **While the follower is silent,** it gets no goal, whatever other error comes meanwhile.
- **Once it answers,** `serve` re-marks the motors, sets each goal to where the motor is found, then writes `Torque_Enable` = 1 itself rather than rely on the goal write turning it on.
- **Restart detection:** `serve` sets each motor's `torque_limit` (address 48, in RAM) one below its `max_torque` (address 16, in EEPROM), and keeps each `max_torque`. At power-on the servo copies `max_torque` into `torque_limit`, so a restarted motor reads back at `max_torque`. `serve` reads the marks on every tick, before any goal goes out, and faults on the first tick after a restart.
- **Only `max_torque` means a restart.** A `torque_limit` that reads anything else, neither the mark nor `max_torque`, was changed by something other than a restart, perhaps the servo's own overload protection. That is a fault of its own, `TorqueLimitChanged`, naming the motor and the value, reported once per value, in Fault too. The arm holds its last goal, as for a motor alarm, and `serve` doesn't write the mark over it, so the value stays as found.

Untested on this firmware:

- that `torque_limit` really is reloaded at power-on. To test it, switch the arm's power off and on while `serve` holds it.
- whether overload protection changes `torque_limit` at all. Feetech's table says a trip drops output to the holding torque (address 34), not whether it writes address 48. To test it, close the gripper on a rigid object for 3 s while paused, then read address 48.
- whether a held grasp sets the overload bit (32) in the error byte of the gripper's status replies. `serve` reads every motor's temperature once a second and faults on any alarm bit in those replies, so if a firm grip sets it, holding a grasp faults the attempt. To test it, grip a rigid object and then a sponge for 10 s each while paused, and see whether `serve` faults with "Motors in a protection alarm: gripper (overload)".

### The gripper's overload trips (left as is)

The follower's gripper sometimes stops answering and blinks red until its power is switched off and on (README, Troubleshooting). The likely cause is the overload threshold LeRobot sets.

- **What LeRobot writes.** On every connect, LeRobot's `configure()` writes to the gripper only:
  - `Max_Torque_Limit` 500 (50%);
  - `Protection_Current` 250 (about 1.6 A);
  - `Overload_Torque` 25, commented as "25% torque when overloaded".
- **What the register actually is.** In Feetech's table, `Overload_Torque` (address 36) is the trip threshold, not the torque after a trip. When the load stays above it for `Protection_Time` (address 35, 2 s by default), output drops to the holding torque (address 34, 20% by default). So a grip that needs more than 25% load for 2 s trips. LeRobot's open PR #4783 reads the register the same way.
- **The fix, if it's needed:** raise `Overload_Torque` to 40 or 50 after LeRobot's `configure()`. The 50% torque limit and 1.6 A current limit stay as burn protection, and `serve` already watches the motors' temperature.
- **Why it isn't done:** it isn't measured on this arm yet. LeRobot rewrites the value on every connect, so `teleop` and `record` (which run LeRobot's commands) would need a wrapper too.
- **Things to try first:**
  - Close the leader's gripper only until the follower grips.
  - Print the TPU compliant fingers from the SO-ARM100 repo. They spread a squeeze over time, so the servo spends less time near stall.
- **Clearing a trip without the power switch:** the servo protocol has a Restart instruction (0x08), which rustypot 1.13 exposes as `reboot(id)`. The servo doesn't answer it, takes about 1.8 s, and comes back limp with RAM settings at their defaults. Untested on this firmware.

### Registers worth knowing

| Register | Address | Default | Note |
| --- | --- | --- | --- |
| Max temperature | 13 | 70 °C | `serve` warns at 55 °C and `check` fails at 60 °C |
| Min voltage | 15 | 4.0 V | Cuts output only if bit 0 of the unload condition is set |
| Max torque | 16 | 1000 (0.1% units) | LeRobot sets the gripper's to 500. Copied into `torque_limit` at power-on |
| Unload condition | 19 | Model-specific bitmask | Which protections cut output: temperature, voltage, current and load each have a bit |
| LED alarm condition | 20 | Model-specific bitmask | Which protections blink the LED |
| Protection current | 28 | 511 × 6.5 mA ≈ 3.3 A | LeRobot sets the gripper's to 250 |
| Holding torque | 34 | 20% | Output after an overload trip |
| Protection time | 35 | 200 × 10 ms = 2 s | How long a load above the threshold lasts before it trips |
| Overload torque | 36 | 80% | The trip threshold. LeRobot sets the gripper's to 25 |
| Acceleration | 41 | — | 8.79 °/s² per unit. LeRobot writes 254, about 2,230 °/s², close to the maximum |
| Goal speed | 46 | 0 (no limit) | 0.732 rpm, about 4.4 °/s, per unit. Unused: the arm's speed is capped by `max_relative_target` |
| Torque limit | 48 | `max_torque` at power-on | `serve` sets it one lower to detect restarts |
| Status | 65 | — | Alarm bits: 1 voltage, 2 encoder, 4 temperature, 8 current, 32 load |

Two of LeRobot's writes don't match Feetech's table: `Return_Delay_Time` goes to address 7, which Feetech lists as reserved on the STS, and `Maximum_Acceleration` to 85, which Feetech lists as read-only. Neither has shown an effect.

Unchecked: the actual values of registers 19 and 20 on these servos, which Feetech doesn't publish. Reading them is read-only, so `so101 motors` could print them.

### Speed

Measured on the 60 cylinder-bowl demos: 99.9% of joint motion is under 92 °/s, and the fastest single frame was 227 °/s (the elbow). `max_relative_target` (10 units a tick at 30 Hz) allows up to 300 °/s. A speed cap in the servos' own memory (Goal speed, address 46) was considered and dropped: it would stay set until power-off, so it would also slow `teleop` and `record` after a `serve` session.

## Stopping the arm

Today the only stop that works when the software is dead is the arm's power switch. Pause and Reset are software. These were considered and not built:

| Option | What it stops | Cost |
| --- | --- | --- |
| A latching mushroom e-stop in series with the follower's DC power, rated above stall current (15 A or more) | Everything, with the host, network or software dead | About $15-40 and some wiring. The arm drops, so it's for emergencies |
| A watchdog relay that cuts servo power when `serve` stops sending a heartbeat (a TI TPS3823 trips after 1.6 s without one) | A crashed or frozen host, a `kill -9`, a lost network | About $10-30 in parts, a USB microcontroller (the Mac has no GPIO) or the Jetson's GPIO, and a heartbeat in `serve` |
| A Shelly smart plug on the supply, with a power-off button on the page | Remote power-off over Wi-Fi | About $20-30, no wiring. Its `toggle_after` timer might act as a heartbeat cutoff, but Shelly's docs don't say whether calling it again restarts the timer: test it first |

Power cut at rest is safe: the rest pose is one the arm keeps with no power (`so101 pose --save-rest`).

## Power

- **Stall current:**
  - ST-3215-C001 (7.4 V): 2.0 A at 6 V. Six stalled together draw about 12 A.
  - ST-3215-C018 (12 V): 2.7 A at 12 V, so about 16 A for six.
- **The kit's supplies are smaller than that.** The SO-ARM100 bill of materials lists a 5 V supply for the 7.4 V arm and "12 V 5 A+" for the 12 V one. A user's calculation (SO-ARM100 issue 132) says 12 V 5 A, with power daisy-chained through the servos' 3-pin cables, browns out on heavy moves and heats the first cables. If brownouts show up as restart faults in `serve`, use a larger supply and run power straight to the board.
- **The Waveshare board** takes 9-12.6 V matched to the servos. Waveshare doesn't publish its current rating, fuse or reverse-polarity protection.
- **Unknown here:** which follower servos this arm has (C001 or C018) and the supply's rating. The voltages `so101 motors` reads would tell the servo variant.

## USB

- **No latency timer to tune.** The board shows up as `/dev/cu.usbmodem…` on macOS, a CDC-ACM device. FTDI's latency timer doesn't apply to it.
- **LeRobot's retries:** it waits 50 ms per packet and retries a failed read twice (`num_read_retries`). A goal write gets no reply, so a lost one goes unnoticed. An alarm bit in a reply doesn't raise an error: `serve` reads the status itself once a second.

## Calibration drift

A leader and follower whose calibrations disagree train a policy with an offset: one LeRobot user found 17° on one joint, about 6 cm at the gripper (LeRobot issue 3758). On still frames of a demo, `action - observation.state` should be close to 0; a steady gap on one joint means recalibrating. Nothing checks this automatically yet.

## The Jetson (not supported yet)

`so101` runs on macOS. What a Jetson Orin needs, from NVIDIA's and the kernel's docs:

- **Python and PyTorch:**
  - The repo pins Python 3.13, but NVIDIA's Jetson PyTorch wheels (and the jetson-ai-lab index for JetPack 6) are Python 3.10 only.
  - Options: allow Python 3.10 or 3.12 on Linux, whose draccus versions work, or run NVIDIA's container.
  - Unknown: whether PyPI's CUDA 13 aarch64 torch uses the Orin's GPU under JetPack 7.2.
- **JetPack:** 6.2.x (Jetson Linux 36.4) or 7.2 (39.2). 7.0 and 7.1 drop Orin.
- **Stable ports:**
  - Use udev's `/dev/serial/by-id/…` links, or a rule matching the board's USB id and serial number, with group `dialout`.
  - Avoid `chmod 666`, which a replug undoes.
- **USB autosuspend:** the kernel leaves it off for non-hub devices by default. Force it off with `usbcore.autosuspend=-1`, or per device with `power/control=on`.
- **Cameras:**
  - Use MJPEG (`fourcc="MJPG"`), because two uncompressed streams on one USB 2.0 bus can exceed its bandwidth.
  - Use `/dev/v4l/by-id/…` for identification. That needs a camera serial number; without one, use `/dev/v4l/by-path/…`, which is per USB port.
  - Exposure and anti-flicker go through V4L2's standard controls, since `uvc-util` is macOS-only.
- **Power and heat:**
  - Fix an `nvpmodel` power mode, and use the "cool" fan profile (Orin NX and Nano default to "quiet").
  - Log temperatures with `tegrastats`.
  - Throttling starts at 99 °C.
- **Foot pedal (not built):**
  - The plan: on Linux, `serve` reads a 3-pedal PCsensor FootSwitch at `/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd`, the pedal and keys LeRobot's DAgger rollout uses. Left (`KEY_A`) acts like Space on the page, middle (`KEY_B`) takes over with the leader, right (`KEY_C`) rewinds.
  - It should grab the device (`EVIOCGRAB`) while it runs, so presses don't also type `a`, `b` or `c` into the focused window: in `serve`'s own terminal `b` also rewinds and `c` runs the checks. LeRobot's own pedal listener doesn't grab.
  - A reader that did this was removed before PR #16 merged, so it isn't in `main`'s history. The PR's commit list on GitHub still has it: the parent of the commit titled "Delete the features that were cut" holds `src/so_101/control/pedal.py` and its tests, `tests/test_pedal.py`.
  - Check the grab once it's built: with `serve` running and another terminal window focused at a shell prompt, press each pedal. Nothing is typed there, and `serve` acts on each press. After `serve` exits, a press types its letter again.

## Sources

- [Feetech STS memory table](https://github.com/ftservo/ftservo-wiki/blob/HEAD/docs/en/reference/parameter/memory-sts.md) and [FT-SCS protocol](https://github.com/ftservo/ftservo-wiki/blob/HEAD/docs/en/reference/protocol/ft-scs.md)
- ST-3215 [C001](https://github.com/ftservo/ftservo-wiki/blob/HEAD/docs/en/products/models/st-3215-c001/official-specs.json) and [C018](https://github.com/ftservo/ftservo-wiki/blob/HEAD/docs/en/products/models/st-3215-c018/official-specs.json) specs
- LeRobot 0.7.0: `lerobot/robots/so_follower/so_follower.py` (`configure`), `lerobot/motors/motors_bus.py`; [PR #1809](https://github.com/huggingface/lerobot/pull/1809), [PR #4783](https://github.com/huggingface/lerobot/pull/4783), [issue #2819](https://github.com/huggingface/lerobot/issues/2819), [issue #3758](https://github.com/huggingface/lerobot/issues/3758)
- [SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100), [issue 132](https://github.com/TheRobotStudio/SO-ARM100/issues/132)
- [Waveshare Bus Servo Adapter (A)](https://www.waveshare.com/wiki/Bus_Servo_Adapter_(A))
- [TI TPS3823](https://www.ti.com/product/TPS3823), [Shelly Gen2 Switch API](https://shelly-api-docs.shelly.cloud/gen2/ComponentsAndServices/Switch)
- [systemd 60-serial.rules](https://github.com/systemd/systemd/blob/main/rules.d/60-serial.rules), [60-persistent-v4l.rules](https://github.com/systemd/systemd/blob/main/rules.d/60-persistent-v4l.rules), [kernel USB power management](https://docs.kernel.org/driver-api/usb/power-management.html), [V4L2 camera controls](https://github.com/torvalds/linux/blob/master/Documentation/userspace-api/media/v4l/ext-ctrls-camera.rst)
- [PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html), [JetPack release notes](https://docs.nvidia.com/jetson/jetpack/release-notes/index.html), [Jetson Orin power and performance](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
