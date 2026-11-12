# so-101

SO-101 leader and follower arms driven by [LeRobot](https://huggingface.co/docs/lerobot/en/so101), over its Feetech motor driver, rustypot.

Python 3.13, managed with uv. LeRobot needs 3.12 or later, and its CLI argument parser (draccus) breaks on 3.14. `uv sync` sets up `.venv`.

## Where things live

| What | Where |
| --- | --- |
| USB ports, arm ids, cameras, teleop, run and data settings | [`so101.toml`](so101.toml) |
| Tasks: the props, cameras and demo counts of each task you train | [`tasks.toml`](tasks.toml) |
| uvc-util installer, for camera exposure and anti-flicker | `scripts/install-uvc-util.sh` |
| Calibration files | `calibration/follower/so101_follower.json`, `calibration/leader/so101_leader.json` |
| Recorded demos, edits, training datasets and `serve` runs | `data/<task>/` (git-ignored; see [Your demos on disk](#your-demos-on-disk)) |
| `so101` command | `src/so_101/cli.py` |
| `serve`, `runs`, and what `run` and `serve` share | `src/so_101/commands/` |
| `serve`'s control center: state machine, control loop, page and recorder | `src/so_101/control/` |
| `run`'s rollout, which starts each attempt from the demos' start pose | `src/so_101/rollout.py` |
| The cloud GPU policy server | `src/so_101/remote.py` |
| Config loading and port lookup | `src/so_101/settings.py` |
| Health checks (`check`, and before record, teleop, run and serve) | `src/so_101/check.py` |
| Finding cameras by USB id, camera controls | `src/so_101/devices.py` |
| Recording sessions, edits and builds | `src/so_101/data.py`, `recording.py`, `review.py`, `review_runs.py`, `build.py` |
| Dry run, dataset reading and checks, PASS/WARN/FAIL reports | `src/so_101/dryrun.py`, `dataset_io.py`, `validate.py`, `report.py` |
| The fix for SVT-AV1's macOS crash when a video encoder closes | `src/so_101/svtshim.py`, `svtsem.c` (see [The video encoder's macOS crash](#the-video-encoders-macos-crash)) |
| Downloaded policies, `run` recordings, `runs` exports, camera snapshots, check and dry-run reports | `outputs/` (git-ignored) |
| Notes too long for this README, and ideas not built yet | [`docs/`](docs/) |

`so101.toml` is the only place ports and ids are written down. Every `so101` command reads it, so when a port changes you fix it once.

## Commands

Run `uv run so101 check` first: it checks both arms, both cameras and the rest of the setup, and says how to fix anything wrong.

| Command | Does |
| --- | --- |
| `uv run so101 check [task] [--live] [--json]` | Check arms, cameras and setup. Moves nothing unless `--live`, and sets only the cameras' UVC controls. `--json` prints the results as JSON, and can't be combined with `--live`. See [Checking the setup](#checking-the-setup) |
| `uv run so101 info` | Show config, resolved ports, camera indices and calibration status |
| `uv run so101 find-port follower` | Detect a board by unplugging it, and save its port and serial number to `so101.toml` |
| `uv run so101 pose [task]` | Compare the follower's current pose with the demos' start pose (read-only) |
| `uv run so101 pose --save-rest` | Save the follower's current pose as its rest pose in `so101.toml`. Reads the motors only. See [Saving a rest pose](#saving-a-rest-pose) |
| `uv run so101 motors [follower\|leader] [--watch S]` | Show each motor's temperature, voltage, load, position within its calibrated range, and alarms. Read-only; `--watch 2` refreshes every 2 seconds |
| `uv run so101 scan [follower\|leader]` | Ping motors at every baud rate and list the ids that answer. Read-only |
| `uv run so101 setup-motors follower` | Set motor ids and baud rates, one motor at a time. Writes to motor EEPROM; only needed once per arm |
| `uv run so101 calibrate follower` | Calibrate an arm |
| `uv run so101 teleop [--cameras]` | Drive the follower with the leader; `--cameras` shows both camera feeds in a Rerun window |
| `uv run so101 cameras` | List connected cameras and save a snapshot from each to `outputs/cameras/` |
| `uv run so101 tasks` | List the tasks in `tasks.toml` with their props, camera setup and demo counts |
| `uv run so101 record [task] --dry-run [--replay] [--keep] [--no-train] [--no-display]` | Rehearse a recording session and check the result, without keeping anything. `--replay` replays an episode on the follower, `--keep` keeps the rehearsal dataset, `--no-train` skips the short training test |
| `uv run so101 record [task] [--episodes N] [--note TEXT] [--no-display]` | Record demos of your own task by teleoperating, as a new session. `--no-display` doesn't open the Rerun camera viewer. A session in which nothing was saved is removed; one with data but no readable episodes is kept, and `build` skips it |
| `uv run so101 review [task] [--episode SESSION/N] [--all] [--no-view]` | Step through recorded demos to trim idle time or delete bad ones. `--all` includes demos already reviewed, `--no-view` doesn't open the Rerun viewer |
| `uv run so101 build [task] [--with SOURCES] [--keep-labels] [--force]` | Make the training dataset from all sessions plus your edits (`train` does this itself); `--force` rebuilds even if nothing changed; `--with corrections` adds what a human did after taking over from the policy in `serve` runs (see [Training on corrections](#training-on-corrections)) |
| `uv run so101 train [task] [--policy act\|pi05] [--target FLAVOR] [--with SOURCES]` | Train a policy on your demos, ACT by default, on Hugging Face Jobs by default |
| `uv run so101 backup [task]` | Upload a task's sessions, edits and runs to a private Hub dataset |
| `uv run so101 run [task] [--episodes N] [--duration S] [--reset-time S] [--action-steps N] [--display] [--revision TAG]` | Run a task's ACT policy on the follower, on the Mac's GPU, and record it to `outputs/rollouts/` |
| `uv run so101 run [task] --policy act\|pi05 --remote [--revision TAG]` | Run a policy on the cloud GPU below, with the arm and cameras on the Mac |
| `uv run so101 serve [task] [--action-steps N] [--revision TAG] [--operator NAME]` | Supervise a task's ACT policy from a control page over Tailscale, or the keyboard: start, pause, resume, rewind, take over with the leader and end each attempt, each recorded to `data/<task>/runs/` |
| `uv run so101 serve [task] --policy act\|pi05 --remote [--revision TAG]` | Supervise a policy that runs on the cloud GPU below |
| `uv run so101 review --runs [task] [--episode RUN/N] [--all] [--no-view]` | Step through the interruptions in a task's `serve` runs to fix attempt outcomes and suspect frames |
| `uv run so101 runs [task]` | Summarise a task's `serve` runs (outcomes, how much the policy drove, interruptions per attempt) and export failure-fix pairs to `outputs/runs/<task>/` |
| `uv run so101 server start\|stop\|status [task] [--policy act\|pi05] [--revision TAG] [--action-steps N] [--target FLAVOR] [--timeout T]` | Start, stop or check the cloud GPU job that `run --remote` and `serve --remote` use. `--target` and `--timeout`, such as `30m` or `2h`, override `[remote.<policy>]` in `so101.toml` |

Every command that runs a LeRobot command prints it in full. `setup-motors`, `calibrate`, `teleop`, `cameras`, `record`, `train`, `run` and `serve` pass extra arguments straight through to LeRobot, for example `uv run so101 teleop --teleop_time_s=60`. Every other command stops with an error on an argument it doesn't know. Option names must be written in full: `--epi` isn't taken for `--episodes`, so a shortened option can't be mistaken for one meant for LeRobot.

`teleop` runs at `fps` from the `[teleop]` section of `so101.toml`. `max_relative_target`, in the same section, caps how far a follower joint can move per control step in `teleop`, `record`, `run` and `serve`: degrees, or percent of its range for the gripper. Set it to `0` to turn the cap off.

## When a USB port changes

Ports change when a board is replugged into another socket, a hub, or another computer. Two things keep commands working:

1. **Serial number fallback.** `find-port` also saves the board's USB serial number. If the saved port is gone, commands find the board by serial number instead, wherever it is plugged in. `info` shows when this happens.
2. **Re-detect.** `uv run so101 find-port follower` (or `leader`) finds the board again and rewrites `so101.toml`.

## Hardware setup from scratch

Each step talks to the arms over USB, so plug in USB and the power supply for each arm first.

1. `uv run so101 find-port follower`, then `uv run so101 find-port leader`.
2. `uv run so101 setup-motors follower` and `... leader`, for new motors only. Connect one motor at a time as prompted, starting with the gripper.
3. Assemble the arms ([guide](https://huggingface.co/docs/lerobot/en/so101#step-by-step-assembly-instructions)).
4. `uv run so101 calibrate follower` and `... leader`. Put every joint in the middle of its range, press Enter, then move each joint through its full range and press Enter again. Use the same middle pose for both arms.
5. `uv run so101 scan` should list ids 1 to 6 on both arms.
6. Save the follower's rest pose, as below.

### Saving a rest pose

The control center's Reset drives the follower to its rest pose at the end of every attempt, and the arm sits there between attempts. Choose a pose that is folded and stays put with the power off: if the motors lose power or torque, the arm should already be resting on its own stops rather than dropping onto the table or a prop.

1. Turn off the follower's torque (switch its power off, or press Ctrl+C in `so101 teleop`) and move it there by hand.
2. Power it back on without moving it, then run `uv run so101 pose --save-rest`.

It reads each motor's position and writes it as `rest_pose` under `[control]` in `so101.toml`, replacing any earlier one and leaving the rest of the file as it was. It never writes to the motors, and if any motor reads outside 0 to 4095 it saves nothing and names the motor. The pose is saved in raw motor steps (0 to 4095), not degrees, so it stays right whatever a policy's units; it's converted with the arm's calibration when used.

Beside it goes `rest_pose_calibration`, the first 12 hex digits of the SHA-256 of the follower's calibration file, so calibrate the follower first. Recalibrating changes where each step count puts a joint, so a pose saved before it may no longer fold the arm: `serve` refuses to start while the calibration file doesn't match the fingerprint, and saving the rest pose again fixes it. A rest pose without `rest_pose_calibration` is refused the same way, and `check` warns: save it again.

## Checking the setup

`uv run so101 check` reads everything it can without moving anything, and prints PASS, WARN or FAIL for each item with a fix. It exits with status 1 if anything fails. `teleop`, `record`, `run` and `serve` run the same checks first and refuse to start on a failure; warnings never block. There's no override: fix the problem, then run `uv run so101 check` again.

- **Arms:**
  - the board is found (by saved port, else by USB serial number) and isn't the other arm's board;
  - all six motors answer and are STS3215s;
  - no motor is in a protection alarm;
  - supply voltage is even along the chain;
  - motors aren't hot.
- **Calibration:**
  - a calibration file exists;
  - no joint has a tiny range;
  - the file matches the calibration stored in the motors;
  - every joint reads inside its calibrated range.
- **Cameras:**
  - found by USB id;
  - their controls set and read back;
  - open at the configured size;
  - how many new frames per second they really deliver;
  - brightness and blank or frozen pictures;
  - flicker from mains lighting.
- **Other:**
  - the Hugging Face login;
  - free disk space;
  - uvc-util;
  - macOS Accessibility permission for the recording keys;
  - the fix for the video encoder's macOS crash: `check` builds it, and `record`, `run` and `serve` fail unless it's loaded (see [The video encoder's macOS crash](#the-video-encoders-macos-crash));
  - how fast a recording step can read both arms and cameras.
- **For the task** (the default one in `tasks.toml`, or the one you name): whether new demos would match its earlier sessions (fps, cameras, calibration).
- **For `serve`,** from `so101.toml` and the follower's calibration file:
  - the rest pose is saved, names exactly the follower's six motors, and each is a raw position from 0 to 4095;
  - `rest_pose_calibration` is there and matches the follower's calibration file (see [Saving a rest pose](#saving-a-rest-pose)). Without that file there's nothing to compare, so the rest pose gets a WARN rather than a PASS, beside the follower's own FAIL for the missing file;
  - every `[control]` number is finite and in its range: `rewind_s`, `attempt_limit_s` and `human_limit_s` above 0, `blame_s` and `preview_fps` at least 0, `link_timeout_s` at least 1 s, `port` a whole number from 1 to 65535, and `teleop_from_rest`, `on_link_loss` and `bind` one of their listed values. NaN fails too: `link_timeout_s = nan` would otherwise turn off link loss without a word, because NaN compares false with every time.

  Each problem is a FAIL in `serve`'s checks, which name the fix. `check` lists them as warnings that say `serve` won't start, and `teleop`, `record` and `run` don't check them, since they don't use them. A bad value never stops another command from loading `so101.toml`, so `pose --save-rest` still runs to fix a bad rest pose.

A camera with no `usb_id`, or one that shares its `usb_id` with another camera, gets a WARN, since the check can't confirm it's the right one. `--live` adds a 10-second teleoperation test that checks every follower joint follows the leader; it moves the arm. Camera snapshots from each check go to `outputs/check/`.

`uv run so101 record --dry-run` goes further. It runs the checks, then a guided rehearsal through the real recording command with your settings:
1. two episodes, with the task's `episode_time_s` and `reset_time_s`, exercising the finish-early and redo keys and the timeout;
2. it checks the dataset that produced:
   - every frame and video frame present;
   - evenly spaced timestamps;
   - the real recording loop rate;
   - pictures;
   - each follower joint following the leader;
3. it trains ACT for 50 steps on it, to prove it's trainable.

`--replay` also replays a rehearsal episode on the follower. The rehearsal data is deleted afterwards unless you add `--keep`, and nothing is added to your demos. `--replay`, `--keep` and `--no-train` only work with `--dry-run`: without it, `record` stops and says so, rather than record real demos with the flags ignored. Do a dry run after anything changes: a recalibration, moved or replugged cameras, new lighting, or a LeRobot upgrade.

## Teleoperation

1. Put both arms in about the same pose. The follower jumps to the leader's pose when it connects.
2. `uv run so101 teleop`.
3. To stop: lower the follower to a resting pose, press **Ctrl+C** (this turns off motor torque, so the follower goes limp), then switch off power. In an emergency, pull the follower's power.

## Cameras

`so101.toml` has a `[cameras.top]` (fixed, above the workspace) and a `[cameras.wrist]` (on the gripper), each with a USB id, resolution, fps and rotation. An OpenCV device `index` is optional with a USB id, and is 0 when left out.

OpenCV numbers cameras by position in a list that changes whenever a camera is plugged in or out; an iPhone near the Mac can join it as a Continuity Camera. So on macOS every command finds each camera by its USB vendor:product id (`usb_id`) and works out its current index; `index` only matters for a camera without a `usb_id`, or when two cameras share one.

1. Plug in both cameras and run `uv run so101 cameras`. It lists the connected cameras with their USB ids, marks which is top and which is wrist, and saves one image per index to `outputs/cameras/`.
2. For a new camera, set its `usb_id` in `so101.toml` from that list. If an image is upside down or sideways, set `rotation` (`0`, `90`, `180` or `-90`).
3. Check with `uv run so101 teleop --cameras`. It opens a Rerun window with both feeds and the joint positions.

macOS asks for camera access the first time; allow it for your terminal app.

### Camera controls: exposure and anti-flicker

macOS gives OpenCV no access to UVC controls such as exposure or anti-flicker, so `so101` sets them with [uvc-util](https://github.com/jtfrey/uvc-util). Install it once with `scripts/install-uvc-util.sh` (builds a pinned commit with the Xcode command line tools into `~/.local/bin`).

Each camera's `usb_id` and `[cameras.<name>.controls]` in `so101.toml` say what to set. The checks before `teleop --cameras`, `record`, `run` and `serve`, `so101 check` itself, and the `cameras` command, apply them and read them back; settings stay until the camera loses power. To see a camera's controls and ranges: `uvc-util -V 0x32e6:0x9005 -S '*'`.

Tube lights on 50 Hz mains flicker 100 times a second, which shows up as rolling horizontal bands. A fixed exposure avoids them and keeps the picture the same from demo to demo. The scene is a matte black mat (see [Training your own task](#training-your-own-task)): it reflects nothing and sets the props apart, but a fixed exposure that keeps the white arm and wall unclipped leaves little light for the rest, so both cameras raise `gamma` to lift the mid-tones. `so101.toml` holds settings tuned on the mat:

- **Top camera:** a fixed 20 ms exposure, two whole flicker cycles, with gamma 150 and saturation 64. The mat comes out about 40/255, the wall and arm about 211/255, and 0.16% of pixels clip. 30 ms looked a little richer but blurred the arm when it moved fast. White balance is fixed at 4000 K, which turns the white wall neutral, so colours don't shift as the arm moves through the picture.
- **Wrist camera:** it ignores `power-line-frequency`, so it uses a fixed exposure (auto measured 5.3% flicker). Its exposure units run about 12% off, so the band-free values are 114, about 228 and 285; values between them band (3-8% flicker from 140 to 185). It uses 228 with gamma 120, contrast 32 and saturation 40: a black mat, a white wall and a saturated red block, with no clipping on the gripper fingers. 114 needed gamma 200 to be bright enough, which turned the mat milky grey, and higher gamma at 228 clips the fingers (25% of their pixels at gamma 180). White balance stays automatic: every fixed temperature from 3600 to 6500 K tinted its whites green or yellow. A longer exposure cuts the frame rate: 40 ms gives 14 fps.

The top camera gives 20 frames per second at 640x480 whatever the exposure, and the wrist camera about 23, so `check` warns that up to a third of recorded frames repeat the one before. That's harmless for ACT. Setting the task's `fps` to 20 in `tasks.toml` makes repeats rare; the loop and the cameras aren't in step, so a few remain.

A fixed exposure doesn't adapt to light. If you change the lighting or the mat, run `uv run so101 check` and look at the brightness it reports and the snapshots it saves, then retune `exposure-time-abs` (units of 0.1 ms), `gamma` or `brightness`. `so101.toml` lists every control it sets, so applying it also undoes anything left over from experimenting.

## Training your own task

Every task in `tasks.toml` is one you record demos for and train yourself, in degrees. `block-container` (pick up a red block and put it in a cylindrical container) is the first: copy it and change the name, prompt and props for another task. `cameras` names the cameras to record, as in `so101.toml`; it defaults to `["top", "wrist"]`.

`default` at the top of `tasks.toml` names the task commands use when you don't name one, so `uv run so101 record` means `uv run so101 record block-container`. Name a task to use another, for example `uv run so101 record <task>`.

One-time setup: create a write token at https://huggingface.co/settings/tokens, run `uv run hf auth login`, and add credit to your Hugging Face account (pay-as-you-go or PRO) for cloud training. Recording works without it.

1. **Set up the scene and keep it fixed.** Same camera positions, lighting and surface (the black mat) for recording and running; the policy only learns this scene. Demos recorded on another surface don't match, so if you change the surface, record the task's demos again.
2. **Rehearse:** `uv run so101 record --dry-run`. Fix anything it reports before recording demos you want to keep.
3. **Record:** `uv run so101 record`. Drive the follower with the leader for each demo. A Rerun window shows both cameras.
   - Each demo lasts `episode_time_s` (30 s); Right arrow or `n` finishes it early, Left arrow or `r` redoes it, Esc or `q` stops the session. Esc mid-demo keeps the demo so far, so delete it in `review` if it's incomplete.
   - **Stop a session with Esc, `q` or Ctrl+C, never with `kill -9`.** LeRobot finishes a session's files only when it stops, so a crash, a power cut or `kill -9` loses every episode of that session, not just the last one. A plain `kill` (SIGTERM) of `so101 record` is safe: it reaches LeRobot as Ctrl+C, and `record` waits for it to finish. `record` warns when a session is set to more than 20 episodes: shorter sessions risk less.
   - Between demos you get `reset_time_s` (15 s) to put the block back.
   - Record about 100 demos: 5 block positions x 20 demos each, with the container in one spot. Recording 12 at each spot and keeping the best 10, then going round the 5 spots a second time, gave an ACT policy that put the block in the container in 4 of its first 5 attempts. Start every demo from the same rest pose and do the task the same way each time; smooth, deliberate demos train better than fast ones.
   - Every run of `record` is a new session. Stopping is safe: running it again records until there are `episodes` kept demos. `--episodes 10` adds 10 more; `--note "lamp on"` saves a note with the session.
4. **Review:** `uv run so101 review`.
   - Each new demo opens in a Rerun window: both cameras plus joint plots, with a timeline in seconds.
   - It suggests cutting the idle time at the start and end. It keeps 0.2 s before the first movement and 0.7 s after the last, so the policy still learns to settle.
   - Press Enter to accept the suggestion. Or type `t START END` to keep a different part, `k` to keep it all, or `d reason` to delete it.
   - Decisions go to `data/<task>/edits.toml`; the recordings are never changed.
5. **Train:** `uv run so101 train`.
   - It builds the training dataset from every session and your edits.
   - It uploads the dataset to the private Hub dataset `<you>/so101_<task>`, replacing any earlier upload.
   - It trains ACT on Hugging Face Jobs (`[train.act]` in `so101.toml`: an RTX PRO 6000 for 2,000 steps per demo at batch 8). 100 demos take 200k steps, about 3 hours and $8, stopped at 8 hours. The first 100-demo run took 2 hours 35 minutes.
   - `--policy pi05` trains π0.5 instead (`[train.pi05]`: an RTX PRO 6000 for 120 steps per demo at batch 32). 100 demos take 12k steps, about 10 hours and $27, stopped at 25 hours. First request access to [its tokenizer](https://huggingface.co/google/paligemma-3b-pt-224).
   - Steps follow the demos kept after your edits, and the timeout follows the steps at 2.5x the expected time, so time and cost grow in step with the number of demos. The command prints all of this before it starts. Other `--target` flavors need a measured speed in `so101.toml`.
   - Checkpoints go to the model repo as they're saved. To continue a stopped run: `uv run lerobot-train --config_path=<you>/<policy>_so101_<task> --resume=true --job.target=<flavor>`.
   - Logs stream to the terminal. Ctrl+C stops watching but the job keeps running (`uv run hf jobs ps`, `uv run hf jobs logs <id>`).
   - The trained policy lands in the private repo `<you>/<policy>_so101_<task>`.
   - `--target local` trains ACT on this Mac instead, which takes days at these settings.
6. **Run:** `uv run so101 run --display`, from the same rest pose with the block somewhere you recorded.
   - It downloads the checkpoint to `outputs/policies/`, captures each camera at the resolution it was trained on, and runs it on the Mac's GPU (`[run] device`). The policy reads both camera images and the six joint positions 30 times a second, and outputs six target joint positions. Keep a hand near the follower's power switch; Ctrl+C stops the run.
   - Every run records to its own folder, `outputs/rollouts/<policy>_<task>_<time>/`: the video and joint data in `dataset/`, and the terminal output in `run.log`. It prints where at the start and at the end, including after Ctrl+C.
   - It predicts as often as the model was trained to, every 100 steps for ACT and 50 for pi0.5 (pass `--action-steps` to change it), and uses the same `max_relative_target` as recording, because a lower limit changes the arm's path.
   - It makes one attempt of 30 seconds by default (`[run]` in `so101.toml`). `--episodes 5` makes more, with `--reset-time` seconds between them to reset the scene: right arrow ends an attempt or reset early, left arrow redoes an attempt, and Esc stops.
   - Every attempt starts from the pose your built demos start from (the arm's pose at launch, for a task with no build in `data/<task>/train/`). The arm goes there after each attempt and holds it, so reset the scene around it. An attempt won't start until the arm has arrived: if it can't get there, right arrow turns the motors off so you can move it by hand, and right arrow again carries on. Support the arm first, because it goes limp.
   - Retraining replaces the model in the same repo. A tag keeps a trained model you can still run later: `uv run hf repos tag create <you>/<policy>_so101_<task> <tag>` before you retrain, then `--revision <tag>` runs that model, and leaving it out runs the newest. `run`, `serve` and `server start` all take `--revision`.

If it fails at some block positions, record 10 to 20 more demos there (`--episodes 20`), review them, and train again.

### Run on a cloud GPU

π0.5 is too heavy to run on the Mac, so `--remote` runs it on a Hugging Face Jobs GPU, paid from your Hugging Face credit, while the arm and cameras stay on the Mac. LeRobot hasn't validated robots on a hosted GPU yet, so keep a hand on the arm's power on the first runs.

1. Once, register your SSH key on Hugging Face, because the connection to the job goes through an SSH tunnel:
   ```bash
   pbcopy < ~/.ssh/id_ed25519.pub   # paste it at https://huggingface.co/settings/keys
   ssh -T git@hf.co                 # should greet you by name
   ```
2. Start the server, which takes a few minutes while it loads the model. It bills from now on (`[remote.pi05]` in `so101.toml`: an L4 at $0.80 an hour, stopped after 1 hour):
   ```bash
   uv run so101 server start --policy pi05
   ```
3. Run it as many times as you like. The arm starts moving as soon as the client connects:
   ```bash
   uv run so101 run --policy pi05 --remote
   ```
   Or supervise it, pausing and resuming as you go (see [Supervising a policy with serve](#supervising-a-policy-with-serve)):
   ```bash
   uv run so101 serve --policy pi05 --remote
   ```
4. Stop the billing when you're done. `stop` cancels every running policy-server job, and `uv run so101 server status` lists them all:
   ```bash
   uv run so101 server stop
   ```

The log prints `turnaround mean/max` every 5 seconds. π0.5 predicts 1.67 seconds of motion at a time, so motion is only continuous while turnaround stays under about 0.83 seconds. It measured about 0.65 seconds from India with frames sent at JPEG quality 30, of which 0.36 seconds is the GPU. Higher quality needs a faster upload: quality 90 ran out of actions and faulted within 10 seconds on the same connection. To try another, add `--inference.jpeg_quality=50` to the run command.

If the connection drops or actions run out, the arm finishes its buffered motion, holds for a second, returns home and goes limp, so keep something soft under it.

The SSH tunnel to the job comes back on its own after a network drop. ssh gives up after 15 seconds without an answer from the job (a keepalive every 5 seconds, 3 missed). `run` and `serve` then print `SSH tunnel to the job dropped; reconnecting.` and start it again 5 seconds after each exit until the network is back. Only the first connection, at startup, fails at once. Ctrl+C while it opens stops `run` or `serve` with exit status 130, after the command's usual summary and the reminder that the server keeps billing.

### Your demos on disk

```
data/<task>/
  sessions/<YYYYmmdd-HHMMSS>/
    dataset/        one LeRobot dataset per `record` run; never modified
    session.json    what it was recorded with: settings, camera ids and controls,
                    both calibrations, software versions, check results, your note
  edits.toml        trims and deletions from `review` (you can edit it by hand)
  train/            the training dataset `build` makes; rebuilt when sessions or edits change
                    (and, with `--with`, runs or their labels)
  build.json        which session episodes, run attempts and frames went into train/
  runs/<YYYYmmdd-HHMMSS>/
    dataset/        one LeRobot dataset per `serve` run, one episode per attempt; never modified
    run.json        what it ran with: settings, calibrations, cameras, check results,
                    the policy's repo, revision, commit and action steps, whether it
                    ran on the cloud GPU and then the checkpoint its server admitted,
                    software versions; written once, at the start
    events.jsonl    every command, state change, attempt end and outcome, as it happened,
                    ending with run_end (finished time, attempts, episodes) when the run
                    ended cleanly; a run without it crashed
    meta-copy/<n>/meta/  a copy of dataset/meta/ from the last finished save, of n episodes,
                    which `runs` and `build` read when a kill tore the live one
    labels.toml     decisions from `review --runs`: outcomes and suspect frames (you can edit it by hand)
```

- `data/` sits next to `so101.toml` and is git-ignored. Change it with `[data] dir` in `so101.toml`.
- `uv run so101 backup <task>` uploads the sessions, `edits.toml` and the runs, each with its `labels.toml`, to the private Hub dataset `<you>/so101_<task>_raw`, under the same `sessions/` and `runs/` paths as on disk. It first prints how many sessions and runs it found and their size, and at the end the command to restore them. It leaves `train/` and `build.json` out, since `build` remakes them, along with half-written temporary files, and changes nothing on disk. It stops before uploading if the Hub dataset already exists and is public. Demos and runs are your time; back them up. Back up between runs, not during one: a run being recorded is still half-written.
- Each session is a separate LeRobot dataset, so a crash or a bad session only affects that session. To drop a whole session, move its folder out of `sessions/`.
- **Building:**
  - decodes each kept episode and writes it into a fresh dataset with LeRobot's own writer, at a higher video quality than recording to limit re-encoding loss;
  - checks the result before replacing the old build;
  - stops if sessions were recorded with different fps, cameras or joints, because those can't share one dataset.
- `check` and `record` warn when the calibration changed since the last session, because joint values then no longer line up with earlier demos.
- Runs sit beside the sessions, never among them, so `build` trains on a policy's attempts only when you ask with `--with`. See [What serve records](#what-serve-records) and [Training on corrections](#training-on-corrections).

## Supervising a policy with serve

`uv run so101 serve` runs a policy the way you'd supervise it: the arm rests until you start an attempt, you can pause it at any moment and carry on, and you end each attempt yourself, which sends the arm back to rest. You drive it from the control page, on a phone or laptop over Tailscale (see [The control page](#the-control-page)), or from the keyboard of the computer the arm is plugged into. `so101 run` makes fixed-length attempts, for comparable evaluations.

Before the first time, save a rest pose with `uv run so101 pose --save-rest` (see [Saving a rest pose](#saving-a-rest-pose)). `serve` refuses to start without one, with one that leaves out a motor, with one saved without its calibration fingerprint, or after the follower was recalibrated since it was saved, as well as on any `[control]` setting out of its range (see [Checking the setup](#checking-the-setup)).

`serve` runs the checks first and stops on any failure, like `run`. The leader is optional: it's checked only when it's plugged in, and since only Teleop needs it, each of its failures is a WARN ending "Teleop won't be available" rather than a FAIL. A leader with its motor power off, for example, doesn't stop `serve`. The follower's and the cameras' failures still stop it. It then loads the policy, connects the follower and holds the arm where it is.

The control center runs as a second process, `so101-control`, which `serve` starts. It reads `so101.toml` and `tasks.toml` itself, and it inherits `serve`'s environment, so `SO101_CONFIG` and `SO101_DATA_DIR` apply to both. The control loop runs at the task's `fps` from `tasks.toml`, so `serve` refuses a forwarded `--fps`.

| Key | Does |
| --- | --- |
| Space | Pause whatever is moving, like `p`. At the terminal Space never starts or resumes, because a terminal can't tell a held Space from a fresh press: pressed at rest or paused, it says to use `r`. On the page Space still toggles (see [The control page](#the-control-page)) |
| `r` | From rest: move the arm to the demos' start pose, then start the policy as a new attempt. When paused or in Teleop: the policy carries on with the same attempt |
| `b` | Rewind: play the arm's last `rewind_s` (2.5 s by default) of motion backwards, then pause (see [Rewinding](#rewinding)) |
| `t` | Take over with the leader (see [Taking over with the leader](#taking-over-with-the-leader)) |
| `p` | Pause only, never resume: stop the policy, or the leader in Teleop, and hold the arm with the motors on. Also cuts short a move to the start pose, to rest, a handover or a rewind |
| Right arrow | End the attempt: asks how it went (`s` succeeded, `f` failed, `i` ignore it; any other key keeps it going), then stops the policy and drives the arm to the rest pose. The answer applies only to the attempt it asked about: if another attempt has started since, from the page, the answer is refused and that attempt goes on. With no attempt open (after a fault, or with the arm resting away from the rest pose) it drives the arm to rest without asking |
| Left arrow | End the attempt as one to ignore, such as a bad setup, and drive the arm to rest |
| `c` | Run checks: print `so101 check`'s report for the devices `serve` has open (see [Run checks and Check pose](#run-checks-and-check-pose)). Only at rest, paused or after a fault |
| `o` | Check pose: print how far the follower is from the demos' start pose |
| `h` or `?` | Print the keys |
| Esc or Ctrl+C | End the attempt the same way, unlabelled, then exit once the arm is at rest, with the motors off. A SIGTERM does the same (see Stopping `serve` with `kill` below). Where the arm can't get to rest, it warns and asks for `y` instead of exiting |
| `y` | Only after Esc's warning, within 30 s: exit where the arm is, with the motors off, so support it first. The one way to exit where the arm can't get to rest (see Quitting below) |

- **The page's Space toggles,** decided by the control loop against the state it has when it applies the key, not the state the page last saw: it pauses whatever is moving, and otherwise starts an attempt or resumes. A resume needs the arm stopped, and no toggle pressed, for at least 1 s, so a double press or a held key, meant to stop the arm, never starts it again. When a toggle refuses to resume for that reason, it says so; `r` resumes at once.
- **Resuming after a pause** resets the policy first, so it drops the actions it had queued and predicts again from what the cameras see now.
- **The start pose** is the median of the first frames of your built demos (`data/<task>/train/`, made by `build` or `train`), the same pose `so101 run` starts from. If the arm can't get within 4 units of it in 5 seconds of trying, the attempt pauses rather than start the policy from the wrong pose; `r` tries again. A task with no build starts from wherever the arm is.
- **Pausing on the way to rest** cancels the reset and keeps the attempt open: `r` carries on with the policy, right arrow sends it to rest again.
- **An attempt nobody ends** ends on its own after 45 s of policy driving (`attempt_limit_s` in `[control]`), half as long again as a 30 s demo. Only the time the policy drives counts: paused time, Teleop, rewinds and the moves to the start pose and to rest don't, so a long pause never ends an attempt. The arm returns to rest as it does for the right arrow, the attempt is saved with `ended_by` `time_limit`, and `serve` asks how it went: press `s`, `f` or `i` at rest, or answer the page's card. With no answer the attempt stays unlabelled, and starting the next attempt drops the question. An answer labels only the attempt it was given for: one sent for an earlier attempt, after the next one reached its limit too, is refused. Resuming after a pause on the way to rest gets no extra time: the policy has used its 45 s, so the attempt ends again at once.
- **Quitting never drops the arm mid-air.** Esc, Ctrl+C or a SIGTERM does what `lerobot-rollout` does: it stops the policy, ends the attempt, drives the arm to rest, saves the attempt, then exits and the motors turn off. The rest pose is one the arm keeps unpowered.
  - More presses on the way to rest are ignored, with a message. Space still pauses there, which cancels the quit and keeps the attempt open.
  - At rest, Esc exits at once.
  - **Where the arm can't get to rest, nothing but `y` exits.** That's after a fault, when the follower may not answer; when the move to rest stops short, as when something blocks the arm, which holds where it stopped with the attempt saved there; and when the arm rests more than 4 units from the rest pose with no attempt open, as at startup. Esc there says why, and asks you to support the arm, then press `y` at the robot's terminal within 30 s. `y` exits and the motors turn off, so lower the arm to rest by hand. Another Esc only repeats the warning and restarts the 30 s, so a held Esc can't exit. After 30 s, or once the state changes, `y` is refused and Esc starts over. The page can't send `y`, and no signal counts as one (below).
  - **A follower that doesn't answer is the one exception.** In Fault with the follower silent, as after its power is switched off or its USB cable comes out, Esc, Ctrl+C or a `kill` exits at once, with no `y`: `serve` can't turn those motors off, so exiting changes nothing about the arm. It leaves the motors as they are, even if the follower answers again as `serve` exits: an arm without power stays limp, and one whose USB cable came out keeps holding until you switch its power off. Once the follower answers again, a quit asks for `y` as above.
  - **Ctrl+Z doesn't suspend `serve`.** It prints that Esc quits instead: a suspended `serve` would leave the arm holding with the keys, the page and the policy frozen.
  - **Reset is the other way out.** Once the follower answers, the right arrow or the page's Reset drives the arm to rest from wherever it holds, with no attempt open, and Esc then exits at once.
  - A command queued behind a Quit that exits, in the same tick, is refused with "serve is exiting."
- **Stopping `serve` with `kill`** (SIGTERM), or with SIGHUP, as service managers send them, is the same as Ctrl+C: `so101 serve` sends it on to the control process as SIGINT, waits however long the move to rest takes, then exits with the control process's exit code. Closing the terminal, or signalling the whole process group, signals both processes; the second signal is refused while the first quit is under way, so the arm still goes to rest, or holds where it can't. **No signal ever confirms an exit away from rest:** a Ctrl+C or a `kill` there, however many and however far apart, only repeats the warning, like Esc, unless the follower doesn't answer (above). Only `y`, pressed at the terminal by someone beside the arm, turns the motors off there. With no terminal left, as after closing its window, the arm keeps holding after a fault or a stop short of rest. Support it, then switch the arm's power off. `so101` never escalates to SIGKILL, and `kill -9` can't be passed on, so never use it to stop `serve`. Before the arm is under the control loop, while the policy loads and the follower connects, a SIGTERM or SIGHUP stops the start at once and turns the motors off, as Ctrl+C does then. A signal during the exit interrupts only the cleanup step it lands in. The motors still turn off, unless it lands in the step that turns them off: the arm then stays stiff where it is, normally at rest, so support it and switch its power off.
- **Moves** to the start pose and to rest go at about 45 units a second, and every command, rewinds included, the policy's included, still passes through `max_relative_target` (`[teleop]`).

**On the cloud GPU:** `uv run so101 serve --policy pi05 --remote` supervises a policy running on the server `so101 server start` started (see [Run on a cloud GPU](#run-on-a-cloud-gpu)). `serve` never starts a server itself, because a server bills until it's stopped. It refuses one that serves a different task, policy, revision or action step count, as `run --remote` does, and prints the `server start` command that would match. The action steps and other policy options are set when the server starts: its client rejects `--policy.*` flags. Pause and resume clear the server's queued actions too, so after a resume the policy predicts from what the cameras see now. Unlike `run --remote`, the arm doesn't move when the client connects: it waits at rest for `r`. When `serve` exits, it reminds you that the server is still billing.

This isn't an emergency stop: Pause relies on the software. Keep a hand near the follower's power switch.

### The control page

`serve` prints the page's address when it's ready, such as `Control page: http://100.89.78.11:8101/`, and on the next line its MagicDNS address, such as `http://studio.tail1234.ts.net:8101/`. Open either in a browser on any phone or laptop signed in to the same Tailscale network. There's no password: Tailscale encrypts the connection and decides who can reach it, and over HTTPS it also names who is calling (below). The port is `port` under `[control]` in `so101.toml` (8101).

- **It listens only on this computer's Tailscale address,** found with the `tailscale` command: the one on your `PATH`, else the one inside the Mac app. When there's no such command, or it gives no address, Tailscale counts as stopped. It never listens on every interface, so nobody else on the Wi-Fi can open it. When Tailscale isn't running, it listens on 127.0.0.1 instead and says, in the terminal and on the page: "Tailscale isn't running, so the control page is on this computer only and no remote operator can reach it. Start Tailscale and restart `so101 serve` to allow remote operators." Then open `http://127.0.0.1:8101/` on this computer. With the `tailscale` command installed but the app quit, that address refuses every request until you start Tailscale (see the route checks below). `bind = "127.0.0.1"` under `[control]` always listens on 127.0.0.1.
- **On 127.0.0.1 without HTTPS the page is for this Mac's browser only.** A request that came through Tailscale Serve or Funnel is refused with a 403 saying to set `bind = "tailscale"`, and `serve` warns at startup about each such route to the port. A route that forwards raw TCP to the port (`tailscale serve` or `tailscale funnel` with `--tcp` or `--tls-terminated-tcp`) is worse. Its connections carry none of Serve's headers, so they would pass for this Mac's own browser. `serve` won't start on 127.0.0.1 while one exists, and names the route and the command that removes it, such as `tailscale serve --tcp=2222 off`.
- **Open it by an address `serve` prints.** The server answers only requests addressed, with its port, to its Tailscale IP or to this Mac's MagicDNS name, in full (`studio.tail1234.ts.net`) or short (`studio`), or to `localhost` on a 127.0.0.1 bind. It refuses any other hostname. That stops another website from reaching the page through a DNS trick. `serve` asks `tailscale status --json` for the name once, at startup; when Tailscale doesn't give one, only the IP works. A connection that sends nothing for 10 s is closed. The refusal of a wrong hostname names no address, and every response forbids framing the page.
- **HTTPS through Tailscale Serve,** set up once: `tailscale serve --bg --https=443 localhost:8101` (the port under `[control]`). It survives reboots. `serve` then finds it, listens on 127.0.0.1 behind it, and prints `https://<this Mac's name>.<tailnet>.ts.net/` as the page's address. Only an HTTPS route counts: a plain-HTTP Serve route (`--http`) is ignored. Without it the page stays on plain HTTP at the Tailscale IP, and `serve` prints the command to turn HTTPS on. `tailscale serve reset` turns it off again.
- **Over HTTPS every request needs your Tailscale login.** Tailscale Serve adds the caller's login (`Tailscale-User-Login`) to each request from a device signed in to your tailnet as a user, and strips any copy the caller sent. The status and the events name the caller by that login only for a request that came through Serve, and any other request as "this Mac". Requests from the public internet through Tailscale Funnel never carry one, so the page refuses them with a 403 that names Funnel, even if Funnel is turned on after `serve` starts. A tagged device has no login either, so it is refused too. At startup `serve` also refuses HTTPS when Funnel offers any Serve route to the port, on any path, or when any route forwards raw TCP to the port (`tailscale serve` or `tailscale funnel` with `--tcp` or `--tls-terminated-tcp`), Funnel or not, and says which. Raw TCP connections carry no login and none of Serve's headers, so they would pass for this Mac's own browser; the message gives the command that removes the route, such as `tailscale serve --tcp=2222 off`. Routes of a `tailscale serve` or `tailscale funnel` running in the foreground count too. `serve` then listens on the Tailscale IP over plain HTTP instead.
- **`serve` checks the routes again every 5 s while it runs,** on the web server's own thread, never the control loop's, because a route can be added at any time. While a raw TCP forward reaches the page on 127.0.0.1, or, behind HTTPS, Funnel offers any route to the port, every request is refused with a 403 that names the route and how to remove it. That includes requests with a Tailscale login and requests from this Mac's own browser, because a raw forward passes on any header its caller sends, so no request can be told apart. The page shows the reason in its banner, and the terminal prints it once, then "The Tailscale route is gone: the control page serves requests again." once it's removed. A check that can't read Tailscale's status, because the CLI timed out or `tailscaled` didn't answer, changes nothing: the last answer stands until a check reads the status again. The first check runs before the first request is served. On 127.0.0.1, when the `tailscale` command is installed and that first check can't read the status, the page refuses every request until a check reads it, because a raw forward might exist. So with the Tailscale app quit, the page on 127.0.0.1 stays refused until you start Tailscale, and it serves requests within 5 s of that. Without the `tailscale` command there is no route to fear, and the page serves requests at once. On the Tailscale IP over plain HTTP, a Serve route to `localhost:8101` reaches nothing, so there is nothing to watch.
- **This Mac's own browser needs no login:** open `http://127.0.0.1:8101/` or `http://localhost:8101/` on the Mac. Behind HTTPS the server listens only on 127.0.0.1, so such a request can come only from the Mac. A request that came through Serve is still told apart by the `X-Forwarded-*` headers Serve adds, because Serve passes on whatever Host its caller sent.
- **The phone's screen stays on** while the page has control, so it doesn't dim, lock and pause the policy. Browsers allow this only over HTTPS (the Screen Wake Lock API), so it needs Tailscale Serve above. The browser lets go when the page is hidden, and the page takes it again when it's back.
- **The Mac stays awake while `serve` runs:** `serve` runs `caffeinate -i -w <its process id>`, which ends with it. A closed lid still sleeps a laptop without an external display.
- **Tailscale keys expire after 180 days by default,** and the robot then drops off your tailnet. Turn key expiry off for the robot's computer in the Tailscale admin console (Machines, the computer's menu, Disable key expiry).
- **The page can't quit `serve`:** only Esc or Ctrl+C at the terminal, or a `kill`, can, and where the arm can't get to rest only `y` at the terminal.
- **`--operator NAME`** puts your name in the page's header and on every event a person's action causes in the event log: commands, labels, outcomes, taking control, Run checks and Check pose. Without it the page shows no name, and events record `"operator": null`.
- **The page decides nothing itself.** It asks the server 4 times a second what's allowed, and each button follows that. A disabled button says why, such as "Leader arm not connected". Every tap shows "accepted", or why it was refused, for 2 seconds. A tap with no answer in 5 seconds says so, and that it may still apply: watch the state badge. The events list gives up on a read after 5 seconds too, and tries again with the next status poll. The keyboard keeps working alongside the page.
- **On a laptop the page takes the terminal's keys,** listed under the controls (a phone doesn't show the list). Each key acts through its button, so it does only what the button would, and a key whose button is disabled shows the reason. Space toggles: it sends a pause-or-resume that the control loop decides. Unlike at the terminal, the page sees the key come up, so it ignores a held Space's repeats and any keydown until the key is released, and only a fresh press can resume. Right arrow presses Reset: with an attempt open it opens the Reset sheet, where `s`, `f` and `i` answer and Esc cancels; with none open it drives the arm to rest. Left arrow ends the attempt as ignored. `s`, `f` and `i` also answer the card after a time limit, `1` to `6` pick the step-in chips in order, `c` runs the checks and `o` checks the pose. Esc closes a sheet. **With the Reset sheet or a results sheet open, Space and `p` both pause,** because the policy keeps driving behind a sheet, and Space never resumes there. Every other key does nothing while a results sheet is open, and Esc never quits from the page.
- **One page controls the robot at a time** (see [Handing control between pages](#handing-control-between-pages)). Others view, and any of them can take control.

Top to bottom (on a laptop the cameras move to a left column):

- **Header:** the task, your name if you gave one, and "Connected" with the last round trip to the robot computer in ms, plus "in control" on the page that has control. "Not connected: retrying" means the page can't reach it. "Refused: see above" means the server answered but refused the page, with the reason in a banner, such as a request through Funnel without a login, or a Tailscale route that forwards raw TCP to the port.
- **Viewing banner:** on a page that doesn't have control, "Viewing: another page has control (alice@example.com)." with a Take control button. The name in brackets is the caller of the page in control: its Tailscale login over HTTPS, "this Mac" for the Mac's own browser, or its Tailscale IP, such as 100.89.78.12, over plain HTTP. The events `control_taken`, `link_lost`, `page_hidden` and `link_restored` name it the same way.
- **Device chips:** Follower, Top cam, Wrist cam and Leader. A missing follower or camera turns red, with a red banner naming the device and the fix. A missing leader is a dashed grey "Not connected": it's optional.
- **State badge:** what the arm is doing, in one large word and a sentence, with the attempt number, its elapsed time and the policy time used out of 45 s.
- **Camera previews:** the top and wrist cameras, 320 px wide at 5 frames a second (`preview_fps` under `[control]`). They're the frames the control loop already reads for the policy and the recording, so a preview never opens a camera; the web server shrinks and JPEG-encodes them, off the loop. The page fetches one JPEG at a time at that rate, and none while it's hidden. "No preview" means no frame has come yet, the last fetch failed, the camera's last read failed, or its newest frame is over 2 s old, so a camera that stopped never leaves a frozen picture that looks live; the page tries again every 3 seconds. When `serve` restarts, the page reloads itself, so nothing of the old run stays on it: its previews, events and any open sheet go, and the reloaded page views until a button press or Take control gives it control. When the page loses the server for a moment, the previews come back within 3 seconds and the events carry on where they stopped. When the loop falls behind, previews slow down or stop first, with a note under them (see [Devices and faults](#devices-and-faults)).
- **Attempt timeline:** the attempt so far, colored by phase: green for the policy, violet for a human in Teleop, blue for a rewind, amber for paused, grey for moves to the start pose, the handover and the return to rest. A ▲ marks each time someone stepped in on the policy with Pause, Rewind or Teleop. After an attempt ends it stays up until the next one starts.
- **Why did you step in?** After you step in, until the policy drives again, chips for missed grasp, about to collide, wrong object, dropped it, stuck and other. One tap records a `label` event for the latest step-in; tapping another replaces it in the page, and both stay in the log. It's optional.
- **The controls,** in the state colors:

  | Button | Does |
  | --- | --- |
  | Resume (green) | From rest: move to the demos' start pose and start the policy as a new attempt. Paused or in Teleop: the policy carries on with the same attempt, like `r` |
  | Pause (amber) | Stop the policy or the leader and hold the arm, motors on; cuts short any move, like `p` |
  | Rewind 2.5 s (blue) | Play the last 2.5 s of motion backwards, then pause, like `b`. It shows how many seconds are left to undo |
  | Teleop (violet) | Take over with the leader, like `t` |
  | Reset (full width) | End the attempt and send the arm to rest, through the Reset sheet below |

  Under Reset: "Not an emergency stop. For emergencies, use the power switch on the arm."
- **The Reset sheet:** Reset asks Succeeded, Failed, Ignore this attempt, or Cancel, like the right arrow's `s`, `f` and `i`. The first three end the attempt with that outcome and drive the arm to rest; the attempt is saved when it gets there. Cancel leaves it running. The sheet ends only the attempt it was opened for: it closes when another attempt starts, and the server refuses its answer once any attempt but that one, or the one a fault just ended, is current, so a sheet left open on a second page can't label a newer attempt. While Pause is allowed the sheet also shows a Pause button, because the policy keeps driving while you decide; it leaves the sheet open. With no attempt open (after a fault, where the attempt was already saved, or with the arm resting away from the rest pose) Reset opens no sheet and sends the arm to rest straight away.
- **Run checks and Check pose,** plain text links under the controls. Each opens a sheet listing what it found as FAIL, WARN and PASS, with the fix under each problem (see [Run checks and Check pose](#run-checks-and-check-pose)). A tool that isn't allowed now is greyed out, with the reason under it.
- **After a time limit** the arm returns to rest, and a card asks how the attempt went, with Succeeded, Failed and Ignore. Answering on the page or with `s`, `f` or `i` at the keyboard labels it; starting the next attempt without answering leaves it unlabelled.
- **This run:** attempts ended so far, how many succeeded, the share of driving time the policy drove (policy time over policy plus Teleop time, without the ticks where the policy sent no action and the arm held, as in `so101 runs`) and step-ins per attempt. Ignored attempts count as attempts but not toward the last two.
- **Recent:** the last 8 events, newest first, with their times.

### Handing control between pages

Only one page controls the robot at a time, so two people can't fight over it. Each page load makes itself a random id, kept only in that tab's memory, and sends it with every request. So a duplicated tab, a second browser or another device always starts as a viewer, and so does a reload: tap Take control again after reloading the page that had control.

- **The first page to press a button,** when no page has control, takes control. The event log records `control_taken` with the page's caller: its Tailscale login, "this Mac", or its Tailscale IP over plain HTTP.
- **Every other page views.** It shows the same status, but its buttons are disabled with "Another page has control (alice@example.com). Take control to use this page.", and a command it sends anyway is refused with that reason.
- **Take control** on a viewing page moves control to it at once, with no confirmation from the other page, and logs `control_taken` with both callers. Anyone on the Tailscale network can do this: it prevents accidents, not misuse.
- **The keyboard always works,** whichever page has control. Whoever is at the robot can always pause it.

### Link loss

The controlling page polls the status 4 times a second, and those polls are the link's heartbeat. If they stop for `link_timeout_s` (3 s, 12 missed polls, `[control]` in `so101.toml`), because the phone locked, the browser went to the background or the network dropped, the link is lost:

- **With `on_link_loss = "pause"`** (the default), a running policy pauses and the arm holds, so the policy never runs unsupervised. If the link is lost during the move to the start pose, the move stops and the arm holds there, because the policy would start unwatched on arrival; Resume carries on to the start pose. Teleop (and a move to the start pose that hands over to Teleop), rewinds and moves to rest carry on: they aren't the policy.
- **With `on_link_loss = "continue"`,** the policy keeps running and the loss is only logged.
- **Either way** `serve` logs `link_lost` and says so in the terminal. The controlling page's next poll logs `link_restored`. The policy stays paused until someone resumes it.
- **From the keyboard,** `r` after a link loss runs the policy without the page: you're at the robot, so it isn't unsupervised, and it doesn't pause again until the page is lost again after coming back.
- **Polls from viewing pages don't count.** If the controlling page is gone for good, take control on another one.
- **A hidden page is a lost link at once.** When the controlling page goes out of sight (the phone locks, or another app or tab comes to the front), it tells `serve` straight away, rather than leaving it to notice 3 s of missing polls: `serve` logs `page_hidden` and, with `on_link_loss = "pause"`, pauses a running policy. A phone's browser slows a hidden page's polls to about one a second rather than stopping them, so those polls don't restore the link: only its first poll once back in sight does, at most 250 ms later, logged as `link_restored`. The policy stays paused until someone resumes it.

A link loss isn't an interruption of the policy in `so101 runs`: the pause it causes isn't a judgement of the policy.

### Rewinding

`b` undoes the last moves: the follower plays its last `rewind_s` (2.5 s, `[control]` in `so101.toml`) of actual motion backwards, in real time, then pauses. It retraces the path the arm really took, measured at every frame, so it goes back along a path known to be clear rather than in a straight line.

- **Each `b` goes back another 2.5 s,** whether pressed again after the rewind pauses or while it's still running, which extends it. It never goes back past the start of the attempt, which for an attempt started with `r` is the rest pose: once there, `b` is refused with "Nothing left to rewind".
- **It works from the policy running, paused or Teleop.** It's refused during the move to the start pose, a handover or the move to rest; pause first.
- **The gripper is rewound too.** A gripper that closed on an object in the last 2.5 s opens again, so the object drops. Support it, or rewind less, if dropping it matters.
- **Only driven motion counts.** Paused and handover time, when the arm holds still, isn't played back, so a long pause before `b` doesn't use up the rewind. The rewind's own motion isn't added back either, so `b` after `b` keeps going further back.
- **Space stops a rewind** where it is and holds the arm there. Right arrow ends the attempt as usual.
- **At the end,** the arm gets up to 1 second to reach the pose it went back to, within 4 units on every joint. If it doesn't, `serve` says which joint is how far off, and pauses there.
- **The leader follows the follower:** if it's connected, its motors turn on and it tracks the follower's goals, catching up at about 45 units a second if it started elsewhere, so `t` afterwards hands over at once. Let go of the leader while it moves.
- **Rewinding counts toward neither time limit,** and every goal still passes through `max_relative_target`.
- **A rewind is logged** with the frames whose actions it undid (see [What serve records](#what-serve-records)), which mark the policy's suspect actions for review.

### Taking over with the leader

`t` hands the follower to you on the leader arm, from exactly where it is, so the follower doesn't jump when you take over.

- **From a running or paused attempt,** the policy stops and the follower holds. Then the leader's motors turn on and drive it to the follower's measured pose (the *handover*), not to the goal the follower was last sent, which can differ by several degrees when the arm presses on something. Once every joint is within 4 units of the follower, the leader's motors turn off and you drive. The handover moves the leader one step per control tick, so the follower keeps holding and Space cuts the handover short. Let go of the leader during the handover: it moves on its own at about 45 units a second, and if it can't line up within 5 seconds, both arms hold and the attempt pauses. Press `t` again once you've let go.
- **From rest,** `t` opens a new attempt as a human demo: the arm first drives to the demos' start pose, so your demo starts where the policy's attempts start, then hands over. With `teleop_from_rest = "rest"` under `[control]` in `so101.toml`, it hands over at the rest pose instead.
- **Leaving Teleop:** Space pauses with both arms held, the leader's motors on, so you can let go of it. `r` gives the arm back to the policy in the same attempt, and the leader stays held where you left it. Right arrow ends the attempt, as always.
- **Ending an attempt leaves the leader as it is:** a free leader, as in Teleop, stays free, and one held with its motors on, as after Space in Teleop, stays held until the next handover or until `serve` exits. So a time limit or a fault never drops a leader you were told you could let go of. Support it before quitting `serve`, which turns its motors off.
- **Teleop time has its own limit:** an attempt ends on its own after `human_limit_s` (180 s) of Teleop, the same way the 45 s policy limit ends one, and `serve` then asks how it went. Teleop time never counts toward the policy's 45 s.
- **Every goal still passes through `max_relative_target`,** the leader's included. Each goal the leader gets, in a handover or while it follows a rewind, is at most `max_relative_target` (10 degrees) from where the leader was measured that tick. So a leader you hold still during a handover isn't sent further and further ahead, and doesn't jump when you let go.

The leader is optional. `serve` connects it only while the arm rests or is paused, and only if its calibration file exists (`uv run so101 calibrate leader` makes one). Without it, `t` is refused with "Leader arm not connected" and the reason. Plugged in mid-run, it connects within 2 seconds of the arm resting or pausing, with no restart. A leader that is plugged in but fails to connect (its motor power off, or calibration that doesn't match) is tried again after 4, 8 and 16 s, then every 30 s, because each try holds up the control loop; unplugging it and plugging it back in tries again within 2 s. Unplugged, or no longer answering, during a handover or Teleop, the follower pauses and holds where it is rather than chase a stale pose. A plugged-in leader that fails `serve`'s checks at startup only warns, since only Teleop needs it.

### Run checks and Check pose

Both are read-only. They run inside `serve`, on the follower, leader and cameras it already has open, so you don't have to stop it, and no serial port is ever opened twice: two connections to one port corrupt each other's packets.

- **Run checks** (`c`, or the page) runs the checks of `so101 check`: disk space and `uvc-util`, the follower's and leader's motors, voltages, temperatures and calibration, each camera's picture and frame rate, and the task's sessions. It leaves out the Hugging Face login check, because it asks the Hub with no timeout and would hold up the control loop. It also leaves out the loop timing test, since a slow loop is logged as `loop_rate` and the page names it, in the previews' note or a warning, the live test, since it moves the arm, and the rest pose and `[control]` checks, since `serve`'s checks already passed them before it started and `so101.toml` isn't read again while it runs. The leader is checked only while `serve` has it connected; otherwise the report has a WARN saying why it isn't. The cameras are sampled for 2.5 s inside the control loop, which sends no goal for about 3 s meanwhile. The motors hold their last goal on their own, so Run checks runs only while nothing drives the arm: at rest, paused, or after a fault, to diagnose it. While the arm moves it's refused: "Run checks holds the control loop for about 3 s, so it runs only while the arm holds still: Pause first." When paused during an attempt, the attempt's recording skips those seconds.
- **Check pose** (`o`, or the page) compares the follower's pose, as the control loop last measured it, with the demos' start pose, like `so101 pose`. A joint within 4 units is a PASS; one further off is a WARN, since Resume moves it there before the policy starts. It costs the loop nothing, so it's allowed in every state but Fault.

Each run logs a `check` or `pose` event with the summary, such as "2 FAIL, 0 WARN: fix the failures first.", and every FAIL and WARN.

### Devices and faults

The follower and both cameras are required; the leader is optional. Every 2 seconds `serve` looks for the follower's and the leader's boards by their USB serial numbers and for each camera by its `usb_id` on the USB bus (with macOS's `ioreg`, which sees a replug where AVFoundation's camera list doesn't), without opening them, so the check never slows the control loop or shares the arm's serial port. Once they're open, a failed read or a camera that stops sending frames (none for 0.5 s) counts as missing too.

- **A missing follower or camera refuses `r`, `b` and `t`,** with a message naming the device and the fix. An unplugged follower or camera needs a restart of `serve` once it's back: the open connection doesn't survive a replug. The leader reconnects on its own.
- **A fault** is a follower that stops answering, a camera that stops sending frames or reports itself disconnected (the fault names it), a motor protection alarm (overload, overheat, voltage) or a policy that fails, on the cloud GPU included. An overload on the gripper alone is only a warning: LeRobot sets the gripper to trip at 25% load held for 2 s, so every firm grasp trips it, and it then grips at 20% force. `serve` says so once each time it trips, shows it under the state badge while it stands, logs a `gripper_overload` event, and the attempt carries on, as in `so101 run`. A motor alarm's message says to support the arm, or Reset it to rest if Reset is allowed, before switching the arm's power off and on to clear it, because the motors go limp at power-off. Any other unexpected error in the control loop is a fault too: the loop never stops on an error, because stopping would turn the motors off with the arm in the air. The arm holds where it is, the attempt is saved there as it is, with `fault` as what ended it, and the error is logged. A fault on the tick an attempt opens, before its first frame, leaves nothing to save: that attempt isn't counted, and a Reset with an outcome is refused. The arm isn't driven to rest, because the follower may not be answering.
- **A power cut or a restarted motor is a fault too,** and the arm is never driven back to where it was. A motor that loses power comes back with its torque off, and its first goal would turn the torque on and drive it there at full speed. So while the follower is silent `serve` sends it nothing, whatever else goes wrong meanwhile. Once it answers again, `serve` aims each motor at where it is found, such as lower than it was, then turns its torque on, and the arm holds there. A motor that restarts without the follower going silent, as in a brownout, is caught on the next tick, before any goal goes out. A `torque_limit` that something else changed, perhaps the servo's own overload protection, is a fault of its own, naming the motor and the value. How restarts are detected is in [`docs/hardware-notes.md`](docs/hardware-notes.md#power-loss-and-restarts).
- **After a fault** only Reset (right arrow) and Run checks (`c`) are allowed, and Reset only once the follower answers again: it drives the arm to rest without opening an attempt. If you pause that move, or it stops short, the arm holds where it stopped, and Reset again drives it on to rest; the same holds whenever the arm rests away from the rest pose with no attempt open. At the keyboard the right arrow resets straight away, without asking how it went. A motor alarm refuses Reset, with the alarm named, until a read shows it cleared, which takes switching the arm's power off and on. Esc or Ctrl+C asks you to support the arm and press `y` within 30 s to exit with the motors off, or exits at once while the follower doesn't answer (see Quitting above).
- **A failed local policy stays failed:** once it has failed, Resume is refused until `serve` restarts, because LeRobot's engine can't resume after a failure.
- **A failed cloud-GPU policy reconnects on its own** (`serve --remote`), such as after a network drop. `serve` connects a new session to the same server in the background, trying every 5 seconds: the server frees the lost session after about 10 seconds, so the first tries may be refused. The server admits one session at a time, so when the policy raised an error while its session stayed up (the server didn't fail), `serve` first closes that session. Nothing moves meanwhile, and Resume says how reconnecting is going. Once it's back, Resume works again. A reconnect must find the same checkpoint as the first session (its `artifact_identity`, printed at startup), so a server restarted with another model is refused.
- **A camera that fails while no attempt is open** doesn't fault: it shows as missing, which refuses `r`, `b` and `t`. Goals and holds need no camera, so they reach the arm even while a camera reports itself disconnected, which LeRobot's own follower refuses. So after a camera fault the arm keeps holding, Reset reads only the follower, and the arm can still go to rest with a camera gone. At exit the follower's motors still turn off while a camera reports itself disconnected.
- **Free disk space** is read once a second. Under 1 GB where the run is recorded, `r` and `t` refuse to start a new attempt, saying how much is left; under 5 GB `serve` warns once. These are the thresholds where `so101 check` fails and warns. LeRobot has no handling for a disk that fills up mid-attempt.
- **A failed save of an attempt,** as on a full or unplugged data disk, loses that attempt: its `attempt_end` says `saved: false`, a `recording_failed` event says why, and `serve` prints it. The page shows the attempt as not saved, the page's run summary leaves it out, and no outcome is asked for it after a time limit. The arm still goes to rest, but `r` and `t` refuse every new attempt with that reason until `serve` restarts, because the dataset may be left half-written. An attempt counts as saved only once its data file is finished on disk, so one whose file can't be finished is lost the same way. When the file is finished but the dataset can't be reopened for the next attempt, the attempt stays saved, and new attempts are refused.
- **A failed write to the event log,** as on a full or unplugged data disk, prints one line in the terminal for each type of event that fails (`command`, `state`, `loop_rate` and so on). The lines that couldn't be written stay in memory and go into `events.jsonl`, in order, once the disk takes writes again, so they land late; those still unwritten when `serve` exits are lost, with a line saying so. The arm, the control loop and the page carry on, so you can still pause or reset, and the page still lists the events. A terminal that is gone too, as after it was closed, only drops these lines.
- **Dropped video frames.** LeRobot's video encoder drops a camera frame when it falls behind, and the attempt's video then no longer lines up with its other columns. Such an attempt is still saved, with `dropped_frames` (per camera, such as `{"top": 3}`) on its `attempt_end`, and builds leave it out. The first time in a run, `serve` says "The video encoder fell behind and dropped frames in attempt 4 (top 3), …", with how to lighten the Mac's load: close other apps, or lower `preview_fps`.
- **Motor temperature** is read once a second, in the control loop. Above 55 °C `serve` warns and the status shows it, so you can let the arm cool down before it reaches 60 °C, where `so101 check` fails (it warns from 50 °C).
- **The loop rate** is measured every second. When it drops under 95% of the task's fps (28.5 Hz at 30 fps), `serve` says so and logs it, and again when it recovers. The policy keeps running. Saving an attempt (about 0.1 s, longer on a slow disk), connecting the leader (up to 1 s) and Run checks (about 3 s) hold up the loop while nothing moves, so each starts a new second of measuring instead of counting as a slow loop and shedding previews.
- **Previews are shed first,** because they're the only work the loop can skip; recording and control keep full priority. In the first slow second they drop to half rate (2.5 fps), in the next they stop, and the page says so under the cameras with the measured rate. If the loop is still slow with previews off, the page shows a warning banner with the measured rate: recorded frames are then further apart than the task's fps. Once the loop keeps up for 5 seconds, previews step back up to half rate, and after 5 more to full rate, so turning them on doesn't flip the loop between slow and keeping up every second.

### What serve records

Every `serve` session is a run in `data/<task>/runs/<YYYYmmdd-HHMMSS>/`, beside your demos and never among them (layout in [Your demos on disk](#your-demos-on-disk)). The folder and its `run.json` are made only once the page's port is bound and the policy has loaded, so a start that fails, as when the policy won't load or the port is taken, makes no folder, and nothing under `data/` is ever deleted. `run.json` is written then and never changed. With `--remote` its `policy` entry also has `artifact_identity`, the checkpoint the cloud GPU's server admitted the run to, since a revision such as `main` can move. When `serve` exits it prints where the run is and how many attempts were saved and labelled, read from `events.jsonl`.

- **Every frame of an attempt is recorded,** from the move to the start pose until the arm is back at rest, paused frames included. LeRobot needs evenly spaced frames, at the task's fps.
- **An attempt is saved as one episode when the arm arrives at rest,** or where it holds: at once after a fault, or where a quit's move to rest stopped short. Each attempt's data file is finished on disk (about 0.1 s, at rest) before the attempt counts as saved, so a crash, even a SIGKILL or a power cut, loses only the open attempt's frames and none of the run's events. LeRobot otherwise finishes a dataset's files only when the recording ends, and a run killed before then would lose every attempt in it.
- **Each finished save leaves a copy of the dataset's shared metadata** (`info.json`, `stats.json`, the tasks and the episode index) in `meta-copy/<n>/meta/` beside the dataset, where `n` is the number of episodes saved. Every episode shares `dataset/meta/`, so a kill during a later save that tears it would otherwise hide the whole run from `runs` and `build`. `runs`, `review --runs` and `build --with` fall back to the copy, so such a kill costs only the attempt being saved. The copy is built under a temporary name and renamed into place, and only the newest is kept.
- **Ignored attempts are kept.** "Ignore" only labels the attempt, so it can be left out of training and success rates later.
- **Columns** beside the cameras and `observation.state`:

  | Column | Holds |
  | --- | --- |
  | `action` | the goal actually sent to the follower, after the `max_relative_target` cap, holds included: a hold resends the last goal, capped against where the arm is when a hand has pushed it away |
  | `intervention` | true in Teleop frames after the policy had driven in that attempt, meaning at least one of its actions reached the arm: a human stepping in on the policy. False in an attempt a human drove with no policy, in Teleop before the policy first drove, and after a policy that only waited |
  | `control.phase` | 0 moving to the start pose, 1 policy driving, 2 paused, 3 handing over to the leader, 4 teleop, 5 rewinding, 6 returning to rest |
  | `control.policy_action` | the policy's own action before the cap; NaN on frames where it isn't driving, and on a *waiting frame*: an Autonomous frame where the policy sent no action that tick, as while a cloud GPU's next chunk of actions is late, so the arm held |
  | `control.leader` | the leader's pose, read every frame while it's connected; NaN without a leader. A read that fails drops the leader as unplugging does, but pauses the arm only in a handover or Teleop |

  None of them starts with `observation.`, because LeRobot feeds every `observation.*` column to a policy as an input.
- **`events.jsonl`** has one JSON object per line, written and flushed as it happens: `attempt_start`, `command` (with its result or the reason it was refused; the page's Space is logged as the pause or resume it became, with `"via": "toggle"`), `state` (each change of state; the change to returning carries `ended_by`, how the attempt was asked to end, at the moment it was asked), `handover` (both arms' poses when the leader is freed, and the largest gap between them), `rewind` (logged when a rewind ends: `start_frame` and `end_frame`, the episode frames from `start_frame` up to but not including `end_frame` whose actions it undid, with the arm back at the pose measured in `start_frame`; rewinds in a row cover adjacent ranges), `attempt_end` (written once the episode's file is finished, when the arm arrives at rest, or at once after a fault: what ended it, whether it was saved, seconds per phase, for a time limit which limit fired, and `dropped_frames` per camera when the video encoder dropped any), `outcome`, `label` (why the operator stepped in, from the page's chips), `fault` (the error, the device to blame and the traceback), `control_taken` (`who`, the page's caller, and `previous`, the one it took control from), `link_lost` (the controlling page's `who`, how long it went without polling, and `driving`, whether the policy drove or the arm moved to the start pose), `page_hidden` (the controlling page went out of sight: its `who`, `on_link_loss` and `driving`), `link_restored`, `check` and `pose` (the summary, the FAIL and WARN counts, and each FAIL and WARN), `loop_rate` (the measured rate, each time it crosses 95% of the task's fps), `gripper_load` (the gripper's load in %, its temperature and its alarm bits, once a second while the load is over 20%, to measure what a grasp does), `gripper_overload` (its overload protection tripped), `recording_failed` (why no new attempt can be recorded: a save or its file's finish failed, or the dataset couldn't be reopened after one) and `run_end` (when the run finished, the attempts it opened and the episodes it saved). A run that ends cleanly, with a quit, ends with `run_end`; one without it crashed. Each event has an `id`, numbered from 1 in the order written, which is how the page asks for the events it hasn't shown yet. During an attempt each event carries the attempt number and `frame`, the episode frame it lines up with. `command` events also say where the command came from (`source`: `terminal`, `page`, or `signal` for Ctrl+C, a `kill` or a closed terminal, with the page's caller as `who`: a Tailscale login over HTTPS, `this Mac`, or an IP address). The login is kept only in the local event log. The events a person's action causes (`command`, `label`, `outcome`, `control_taken`, `check` and `pose`) carry the `operator`; the ones the robot causes on its own, such as `fault`, `link_lost` and `loop_rate`, don't.

### Summarising runs

`uv run so101 runs [task]` reads every run of a task and prints, per run and in total:

- **attempts** and their outcomes: succeeded, failed or unlabelled, with the success rate among labelled attempts and its 95% interval, the range the policy's true rate likely lies in. Few attempts give a wide range: 14 of 20 is 48-85%, and 10 of 20 is 30-70%, so 20 attempts can't tell those two policies apart. Compare policies by their intervals;
- **how much the policy drove:** its share of the time anyone drove, the policy or a human with the leader;
- **interruptions per attempt;**
- **per attempt,** what ended it, the seconds the policy, a human and the script (moves, holds, rewinds, and waiting frames, where the policy sent no action and the arm held) drove, and each interruption with what came next and its suspect frames. When the attempt ended straight after an interruption, it says how: `ended by reset`, `ended by quit` or `ended by time_limit`.

Ignored attempts count only in the ignored total, never in outcomes, time or interruptions. Attempts that weren't saved, as after a crash, are listed but not counted. An attempt whose video dropped frames counts in all of these, but gives no training cuts, with a note naming it.

Each of a run's data files is read on its own. LeRobot writes a Parquet file's footer only when it closes the file, so a hard kill during a save can leave one that can't be read. `runs` skips such a file with one `note:` line naming it and why, and counts every attempt whose episode is in the other files. Only the attempts in the skipped file show as not saved. `build --with` and `review --runs` skip it the same way and print the same note. The Rerun viewer that `review --runs` opens can't skip it: it loads every data file of the run, so it can't open any of that run's attempts until the file is removed or repaired. `review --runs` says so in a second note and steps through the run's attempts without the viewer. `review --runs` never changes the run, so the file stays where it is.

- **An interruption** is a Pause, Rewind or Teleop while the policy drives. Reset ends an attempt rather than interrupting it, and a time limit, a fault or a lost link isn't a judgement of the policy, so none of them counts. A Pause and Resume so close together that both land in one tick of the control loop still count, though the dataset has no paused frame for them.
- **Suspect frames** are the frames that led to an interruption, and are never used as training targets:
  - when you rewound right after it (with nothing but a pause in between), they are the frames the rewinds undid;
  - otherwise they are the last `blame_s` seconds of policy frames before it, 2.5 s by default (`[control]` in `so101.toml`), counted back through pauses and waiting frames but not through a correction or a rewind. Waiting frames don't count toward the `blame_s` seconds.
  - the frames any other rewind undid are suspect too, such as a rewind from Teleop or after a lost link: you undid that motion, so it's never trained on. `labels.toml` can't change these.
- **Corrections** are the Teleop segments after the policy drove in that attempt, where a human stepped in with the leader. The policy drives from its first action that reached the arm: waiting frames before it, where it sent nothing, don't count. Teleop before the policy first drove, as in an attempt you drove alone or one where the policy only waited, corrects no policy, so it isn't one.
- **A failure-fix pair** is a rewind that undid the policy's motion, then a correction from the pose the rewind went back to. These are the most useful corrections: what the policy did wrong, and what a human did instead from the same place. `runs` writes them all to `outputs/runs/<task>/failure_fixes.jsonl`, one JSON object per pair, rewritten each time. With no pairs left, as after you mark their attempts ignored, it's rewritten empty, so it never holds pairs `runs` no longer finds. Each has the run, attempt and episode, and for the failure and the fix their frames with `observation.state` and `action` per frame. The camera frames stay in the run's dataset.

Each run's `labels.toml` overrides outcomes and suspect frames. `review --runs` writes it (see [Reviewing runs](#reviewing-runs)), and `runs` reads it but never writes it. Frames are the attempt's episode frames, and ranges are `[start, stop)`, stop not included:

```toml
[[attempt]]
attempt = 3
outcome = "failed"        # succeeded, failed, ignored or unlabelled

[[interruption]]
attempt = 3
frame = 412               # the first frame after the policy stopped, as `runs` prints it
suspect = [380, 412]      # or [] for no suspect frames
```

An entry that matches no attempt or interruption of the run, or a top-level key other than these two, changes nothing. `runs`, `review --runs` and `build --with` each print one note for it, so a typo doesn't go unseen. `review --runs` refuses to start while a run it would review has a top-level key it doesn't know, and names the key and the file, because a save rewrites the file and would drop it.

`runs` reads `data/` only. Its one output is the export under `outputs/`.

`runs`, `review --runs` and `build --with` check `blame_s` before reading any run: anything but a finite number of seconds, 0 or more, stops them with one line, "[control] blame_s in so101.toml is …: set it to a number of seconds, 0 or more, such as 2.5 (0 marks no frames as suspect)." A typo then neither crashes them nor, as a negative number would, silently marks no frame as suspect.

### Reviewing runs

`uv run so101 review --runs [task]` steps through each saved attempt of a task's runs, oldest first. For each attempt it opens the episode in the Rerun viewer, then shows each interruption in turn, then asks for the attempt's outcome. Every decision is written to that run's `labels.toml` straight away. Each save rewrites the whole file, without its comments, so keep notes somewhere else. The run's dataset, `events.jsonl` and `run.json` are never changed, so you can review again or delete `labels.toml` to start over.

For each interruption it prints:

- the attempt, the command that interrupted the policy (pause, rewind or teleop), the reason you gave on the control page, and what came next;
- the suspect frames, in frames and in seconds, and where they came from: the range a rewind undid, the last `blame_s` seconds of policy frames, or `labels.toml`;
- what the arm was doing from 3 s before the suspect frames to 3 s after the interruption, one line per phase, so you can find the same span on the viewer's timeline, which is in seconds.

Answers:

| Answer | Does |
| --- | --- |
| Enter | keep what is shown and go on; at the outcome prompt, confirm the outcome |
| `f START STOP` | set the suspect frames, stop not included: `f 380 412` in frames, or `f 12.7s 13.7s` in seconds |
| `c` | clear the suspect frames: nothing before this interruption is suspect |
| `a` | go back to the automatic suspect frames |
| `o OUTCOME` | at the outcome prompt, after the interruptions, set the attempt's outcome: `succeeded`, `failed`, `ignored` or `unlabelled`, or their first letter |
| `v` | open the attempt in the viewer again |
| `n` | skip to the next attempt without confirming its outcome |
| `q` | quit; decisions so far are saved |

- **Ignoring is reversible:** an attempt you ignored on the control page or here comes back with `o s`, `o f` or `o u`.
- **An attempt counts as reviewed** once `labels.toml` sets its outcome, by Enter or `o` at the outcome prompt. Later runs of `review --runs` skip reviewed attempts; `--all` shows them again, and `--episode RUN` or `--episode RUN/N` picks one run or attempt.
- **Suspect frames you leave alone stay automatic,** so they follow a later change to `blame_s`. Only `f` and `c` write them to `labels.toml`.
- Attempts that weren't saved have no frames and are skipped. `--no-view` skips the viewer.

Plain `uv run so101 review` reviews demos and writes `edits.toml`.

### Training on corrections

`uv run so101 build --with corrections,human-attempts` adds what humans did in `serve` runs to the demos, so the policy learns what to do where it went wrong. `uv run so101 train --with corrections,human-attempts` builds the same way, then trains. The demos are always in, and without `--with`, `build` uses only the demos.

| Source | Adds |
| --- | --- |
| `corrections` | each Teleop segment that came after the policy drove in its attempt, a human stepping in on the policy, as its own episode. From any attempt that isn't ignored, failed ones included |
| `human-attempts` | the Teleop segments from before the policy first drove, in attempts that succeeded: every segment of an attempt a human drove alone, and the human's driving before the policy took over in the others. Each Teleop segment is its own episode, so the paused and scripted frames between them are never trained on. Such Teleop in a failed or unlabelled attempt is left out |
| `policy-successes` | the policy's own frames from attempts that succeeded, minus suspect and waiting frames. Off unless you name it |

- **Never in a build:** paused, handover, moving to the start pose, rewinding and returning frames, which the script drove; waiting frames, where the policy sent no action and the arm held, so a policy is never trained to wait; Teleop frames with no leader pose, the tick the leader was lost; suspect frames; ignored attempts; and attempts whose video dropped frames, with a note. A cut whose episode the run's episode index doesn't list, as when a kill came between finishing its data file and writing its index row, is left out with a note too. A segment with suspect frames inside it becomes two episodes, one on each side, because an episode can't skip frames. Episodes under 0.5 s are left out and listed in the build's notes.
- **Labels decide it.** Set an attempt's outcome or its suspect frames in the run's `labels.toml` (see above), and the next build follows them.
- **`build.json` records the sources,** the runs it read and their `labels.toml`, `blame_s`, and the version of the rules that cut and label runs. Changing any of them, or recording another run, rebuilds; so does `train` without `--with` after a build with it. A build made with `--with` or `--keep-labels` under other rules rebuilds too.
- **Steps still follow the demos kept,** not the added episodes: corrections are a few seconds each, and `train` prints how many it adds.
- **A run must match the demos:** the same fps, robot, joints, and cameras at the same size. Otherwise `build` stops and names the run and what differs. Set its attempts to `ignored` in its `labels.toml`, or move it out of `runs/`.
- **`--keep-labels`** keeps the run columns for training methods that use them, and the built dataset has these besides the demo columns. Without it the dataset has only the demo columns. `train` uses a current `--keep-labels` build as it is, rather than rebuilding without the columns, and LeRobot's training ignores them.

  | Column | Holds |
  | --- | --- |
  | `intervention` | true only in Teleop frames after the policy had driven in that attempt: a human stepping in on the policy. False in demos and in human attempts, where no policy ran. Worked out from `control.phase`, whatever the recorded column holds |
  | `control.phase` | the run's own; 4 (teleop) in demo frames |
  | `control.label` | 1 where `intervention` is true, 0 elsewhere. Demos and human attempts are 0: they correct no policy. -1 marks a suspect frame, which no build includes |
  | `control.attempt_success` | true when the attempt's outcome is succeeded; true in demos |

Other ways to use runs, such as weighting corrections, a failure detector, a π0.5 quality tag or simulating from the states where the policy failed, are in [`docs/using-run-data.md`](docs/using-run-data.md). None of them is built yet.

## The video encoder's macOS crash

**Every command that records or builds video runs with a fix for a crash in SVT-AV1, the AV1 encoder LeRobot uses.** The fix is a small C library that `so101` builds with the Mac's C compiler, then loads by restarting itself with the library in `DYLD_INSERT_LIBRARIES`. If a check says it isn't loaded, don't record until it is.

- **The bug.** When a video encoder closes at the end of an episode, a race inside SVT-AV1 can leave one of its semaphores a count short. macOS then stops the whole process with SIGTRAP: exit code 133, `zsh: trace trap` in the terminal and no Python traceback. Without the fix it hit about 0.1 to 0.4% of encoder closes, mostly on short clips. SVT-AV1 fixed it in 4.3.0, but PyAV 15.1.0, the newest LeRobot 0.7.0 allows, bundles SVT-AV1 3.0.0.
- **What it would cost you.** `serve` records inside the control center, so a crash would end the session and lose the open attempt (saved attempts are safe; see [What serve records](#what-serve-records)). `record` would lose the session's unfinished episodes, and `build` would stop partway.
- **Checks.** `check` says whether it builds. `record` (and its dry run), `run` and `serve` fail their checks unless it's loaded, and `build`, `train` when it has to build, `so101-control` and `so101-rollout` refuse to start. It can be missing for two reasons:
  - Xcode's command line tools aren't installed, so there's no C compiler: run `xcode-select --install`.
  - Something between your shell and `so101` dropped `DYLD_INSERT_LIBRARIES`: with System Integrity Protection on (the default), macOS strips `DYLD_*` variables when it starts a system program such as `/bin/sh`. Run `uv run so101 ...` directly rather than through `sh -c`.

How the fix works, the measurements, the alternatives and how to remove the fix once PyAV bundles SVT-AV1 4.3.0 are in [`docs/svt-av1-crash.md`](docs/svt-av1-crash.md).

## Troubleshooting

- **Missing motor IDs on connect:** run `uv run so101 scan`. Nothing at any baud rate means a hardware fault: servo power off, Waveshare jumpers not on B, or a loose 3-pin cable. Ids at a baud rate other than 1000000 mean the motors need `setup-motors`. Some ids missing means a break in the daisy chain after the last one that answered.
- **Board not found:** plug it in, then run `uv run so101 info`. If the port has changed, run `find-port`.
- **`USB camera ... is not connected`:** the camera with that `usb_id` isn't plugged in (or is on a dead hub). The message lists the cameras that are connected.
- **A camera "delivers 20.0 new frames/s against the 30 recorded":** the camera can't keep up, so some recorded frames repeat the one before. With a fixed exposure this kit's cameras reach 20 fps at any setting, which is harmless for ACT; setting the task's `fps` to 20 makes repeats rare. On auto exposure, dim light slows a camera further: add light or fix the exposure.
- **Dry run: "the recording loop ran at 25 of 30 steps/s":** the Mac can't record that fast with these settings, so demos would play back faster than you moved. Record with `--no-display`, close other apps, or try `--dataset.rgb_encoder.vcodec=auto` (hardware video encoding).
- **`WARNING: shoulder_pan barely moved during calibration`:** that joint's calibrated range is tiny, so the arm can hardly move it. Recalibrate the arm and move every joint through its full range.
- **The follower's gripper blinks red** (`check` reports "gripper (overload)"; `serve` only warns that its overload protection tripped): the gripper kept pushing against something until it tripped its overload protection, and it ignores commands until the arm's power is switched off and on. It happens when the leader's gripper closes further than the follower's can, usually while the follower holds the object. Close the leader's gripper only until the follower grips, and open it before resting the arms. If it trips with nothing in the jaws, recalibrate the follower. Why it trips so easily, and what could change it, is in [`docs/hardware-notes.md`](docs/hardware-notes.md#the-grippers-overload-trips-left-as-is).
- **`Relative goal position magnitude had to be clamped to be safe`**, over and over: the follower can't reach where the leader is, so `max_relative_target` keeps limiting the step. For the gripper, it's the squeezing above.
- **`objc[...]: Class AVFFrameReceiver is implemented in both ...`:** OpenCV and PyAV each bundle FFmpeg. The warning is harmless, and `so101` filters it out of its own output and the LeRobot commands it runs.
- **A checkpoint "sets config fields this LeRobot version doesn't support":** it was saved by a newer LeRobot. `run` drops unset fields itself; set ones need a LeRobot upgrade.
- **A command that records or builds stops with `zsh: trace trap`, or exit code 133:** SVT-AV1's crash when a video encoder closes, which happens only when its fix wasn't loaded. Run `uv run so101 check`, and see [The video encoder's macOS crash](#the-video-encoders-macos-crash).
- **`TorchCodec could not load the FFmpeg libraries.`**, followed by "'torchcodec' is installed but cannot decode videos (see the error above). Falling back to 'pyav' as a default decoder.", each time LeRobot loads a dataset or decodes a video: torchcodec needs a system FFmpeg and found none, so LeRobot decodes with PyAV instead. It's harmless here, and `brew install ffmpeg` silences it.

## License

[MIT](LICENSE).
