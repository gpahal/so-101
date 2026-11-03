# so-101

SO-101 leader and follower arms driven by [LeRobot](https://huggingface.co/docs/lerobot/en/so101), with the Feetech motor SDK.

Python 3.13, managed with uv. LeRobot supports 3.12 and 3.13 only; its CLI argument parser (draccus) breaks on 3.14. `uv sync` sets up `.venv`.

## Where things live

| What | Where |
| --- | --- |
| USB ports, arm ids, cameras, teleop, run and data settings | [`so101.toml`](so101.toml) |
| Tasks: pretrained policies and your own | [`tasks.toml`](tasks.toml) |
| uvc-util installer, for camera exposure and anti-flicker | `scripts/install-uvc-util.sh` |
| Calibration files | `calibration/follower/so101_follower.json`, `calibration/leader/so101_leader.json` |
| Recorded demos, edits and training datasets | `data/<task>/` (git-ignored; see [Your demos on disk](#your-demos-on-disk)) |
| `so101` command | `src/so_101/cli.py` |
| Config loading and port lookup | `src/so_101/settings.py` |
| Health checks (`check`, and before record, teleop and run) | `src/so_101/check.py` |
| Finding cameras by USB id, camera controls | `src/so_101/devices.py` |
| Recording sessions, edits and builds | `src/so_101/data.py`, `recording.py`, `review.py`, `build.py` |
| Dry run, dataset reading and checks, PASS/WARN/FAIL reports | `src/so_101/dryrun.py`, `dataset_io.py`, `validate.py`, `report.py` |
| Python helpers for your own scripts | `src/so_101/arms.py` |
| Downloaded policies, camera snapshots, check and dry-run reports | `outputs/` (git-ignored) |

`so101.toml` is the only place ports and ids are written down. Every `so101` command, the Python helpers, and `so101 env` read it, so when a port changes you fix it once.

## Commands

Run `uv run so101 check` first: it checks both arms, both cameras and the rest of the setup, and says how to fix anything wrong.

| Command | Does |
| --- | --- |
| `uv run so101 check [task] [--live] [--save-reference]` | Check arms, cameras and setup. Read-only unless `--live`. See [Checking the setup](#checking-the-setup) |
| `uv run so101 info` | Show config, resolved ports, camera indices and calibration status |
| `uv run so101 find-port follower` | Detect a board by unplugging it, and save its port and serial number to `so101.toml` |
| `uv run so101 scan [follower\|leader]` | Ping motors at every baud rate and list the ids that answer. Read-only |
| `uv run so101 setup-motors follower` | Set motor ids and baud rates, one motor at a time. Writes to motor EEPROM; only needed once per arm |
| `uv run so101 calibrate follower` | Calibrate an arm |
| `uv run so101 teleop [--cameras]` | Drive the follower with the leader; `--cameras` shows both camera feeds in a Rerun window |
| `uv run so101 cameras` | List connected cameras and save a snapshot from each to `outputs/cameras/` |
| `uv run so101 tasks` | List the tasks in `tasks.toml` with their props, camera setup and demo counts |
| `uv run so101 record [task] --dry-run` | Rehearse a recording session and check the result, without keeping anything |
| `uv run so101 record [task] [--episodes N] [--note TEXT]` | Record demos of your own task by teleoperating, as a new session |
| `uv run so101 review [task] [--episode SESSION/N]` | Step through recorded demos to trim idle time or delete bad ones |
| `uv run so101 build [task]` | Make the training dataset from all sessions plus your edits (`train` does this itself) |
| `uv run so101 train [task] [--target FLAVOR]` | Train ACT on your demos, on Hugging Face Jobs by default |
| `uv run so101 backup [task]` | Upload a task's sessions and edits to a private Hub dataset |
| `uv run so101 run [task] [--duration S] [--display]` | Run a task's ACT policy on the follower, on the Mac's GPU |
| `uv run so101 env` | Print the settings as shell variables |

Every command prints the full LeRobot command it runs. Extra arguments go straight through to LeRobot, for example `uv run so101 teleop --teleop_time_s=60`.

`teleop`, `record` and `run` run the checks first and stop if any fail. There's no override: fix the problem, then run `uv run so101 check` again.

`teleop` applies `fps` and `max_relative_target` from the `[teleop]` section of `so101.toml`. `max_relative_target` caps how many degrees a follower joint can move per control step; set it to `0` to turn the cap off.

## When a USB port changes

Ports change when a board is replugged into another socket, a hub, or another computer. Two things keep commands working:

1. **Serial number fallback.** `find-port` also saves the board's USB serial number. If the saved port is gone, commands find the board by serial number instead, wherever it is plugged in. `info` shows when this happens.
2. **Re-detect.** `uv run so101 find-port follower` (or `leader`) finds the board again and rewrites `so101.toml`.

For a one-off override without editing the file, set `SO101_FOLLOWER_PORT` or `SO101_LEADER_PORT` (also `SO101_FOLLOWER_ID`, `SO101_LEADER_ID`).

## Using the settings elsewhere

Shell, for raw LeRobot commands or other tools:

```bash
eval "$(uv run so101 env)"
uv run lerobot-calibrate --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" \
  --robot.id="$SO101_FOLLOWER_ID" --robot.calibration_dir="$SO101_FOLLOWER_CALIBRATION_DIR"
```

Python:

```python
from so_101.arms import make_follower, make_leader

follower = make_follower()  # port, id, calibration and max_relative_target from so101.toml
leader = make_leader()
follower.connect()
leader.connect()
follower.send_action(leader.get_action())
```

## Hardware setup from scratch

Each step talks to the arms over USB, so plug in USB and the power supply for each arm first.

1. `uv run so101 find-port follower`, then `uv run so101 find-port leader`.
2. `uv run so101 setup-motors follower` and `... leader`, for new motors only. Connect one motor at a time as prompted, starting with the gripper.
3. Assemble the arms ([guide](https://huggingface.co/docs/lerobot/en/so101#step-by-step-assembly-instructions)).
4. `uv run so101 calibrate follower` and `... leader`. Put every joint in the middle of its range, press Enter, then move each joint through its full range and press Enter again. Use the same middle pose for both arms.
5. `uv run so101 scan` should list ids 1 to 6 on both arms.

## Checking the setup

`uv run so101 check` reads everything it can without moving anything, and prints PASS, WARN or FAIL for each item with a fix. It exits with status 1 if anything fails. `teleop`, `record` and `run` run the same checks first and refuse to start on a failure; warnings never block.

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
  - flicker from mains lighting;
  - a match against a saved reference image.
- **Other:**
  - the Hugging Face login;
  - free disk space;
  - uvc-util;
  - macOS Accessibility permission for the recording keys;
  - how fast a recording step can read both arms and cameras.
- **With a task:** whether new demos would match the task's earlier sessions (fps, cameras, calibration).

Once the cameras are where you want them and the follower is in its rest pose, save each camera's view with `uv run so101 check --save-reference`. Later checks compare against it to catch a camera that moved or a wrong camera. `--live` adds a 10-second teleoperation test that checks every follower joint follows the leader; it moves the arm. Camera snapshots from each check go to `outputs/check/`.

`uv run so101 record --dry-run` goes further. It runs the checks, then a guided rehearsal through the real recording command with your settings:
1. two short episodes, exercising the finish-early, redo and timeout keys;
2. it checks the dataset that produced:
   - every frame and video frame present;
   - evenly spaced timestamps;
   - the real recording loop rate;
   - pictures;
   - each follower joint following the leader;
3. it trains ACT for 50 steps on it, to prove it's trainable.

`--replay` also replays a rehearsal episode on the follower. The rehearsal data is deleted afterwards unless you add `--keep`, and nothing is added to your demos. Do a dry run after anything changes: a recalibration, moved or replugged cameras, new lighting, or a LeRobot upgrade.

## Teleoperation

1. Put both arms in about the same pose. The follower jumps to the leader's pose when it connects.
2. `uv run so101 teleop`.
3. To stop: lower the follower to a resting pose, press **Ctrl+C** (this turns off motor torque, so the follower goes limp), then switch off power. In an emergency, pull the follower's power.

## Cameras

`so101.toml` has a `[cameras.front]` (fixed, looking at the table) and a `[cameras.wrist]` (on the gripper), each with a USB id, an OpenCV device index, resolution, fps and rotation.

OpenCV numbers cameras by position in a list that changes whenever a camera is plugged in or out; an iPhone near the Mac can join it as a Continuity Camera. So on macOS every command finds each camera by its USB vendor:product id (`usb_id`) and works out its current index; `index` only matters for a camera without a `usb_id`.

1. Plug in both cameras and run `uv run so101 cameras`. It lists the connected cameras with their USB ids, marks which is front and which is wrist, and saves one image per index to `outputs/cameras/`.
2. For a new camera, set its `usb_id` in `so101.toml` from that list. If an image is upside down or sideways, set `rotation` (`0`, `90`, `180` or `-90`).
3. Check with `uv run so101 teleop --cameras`. It opens a Rerun window with both feeds and the joint positions.

macOS asks for camera access the first time; allow it for your terminal app.

### Camera controls: exposure and anti-flicker

macOS gives OpenCV no access to UVC controls such as exposure or anti-flicker, so `so101` sets them with [uvc-util](https://github.com/jtfrey/uvc-util). Install it once with `scripts/install-uvc-util.sh` (builds a pinned commit with the Xcode command line tools into `~/.local/bin`).

Each camera's `usb_id` and `[cameras.<name>.controls]` in `so101.toml` say what to set. The checks before `teleop --cameras`, `record` and `run`, and the `cameras` command, apply them and read them back; settings stay until the camera loses power. To see a camera's controls and ranges: `uvc-util -V 0x32e6:0x9005 -S '*'`.

Tube lights on 50 Hz mains flicker 100 times a second, which shows up as rolling horizontal bands. The front camera handles this with `power-line-frequency = 1` (50 Hz). The wrist camera ignores that setting, so it uses a fixed exposure instead: 28.5 ms measured 0.3% flicker against 5.3% on auto, at about 20 fps. A fixed exposure doesn't adapt to light, so if the wrist image is too dark or bright under different lighting, retune `exposure-time-abs` (units of 0.1 ms) or go back to `auto-exposure-mode = 8`.

## Running a pretrained policy

There is no general-purpose pretrained ACT model. Every ACT checkpoint was trained on one person's arm, cameras, table and lighting, and only works in a scene that looks like theirs. `tasks.toml` lists community SO-101 ACT checkpoints that use one fixed camera plus a wrist camera like this kit, with the props and camera placement each one expects. Expect these to be hit and miss; they are a way to try the full pipeline before recording your own demos.

1. `uv run so101 tasks`, pick one, and open its `look` link to watch the training episodes.
2. Set up the props and move the front camera until `uv run so101 teleop --cameras` looks like the episodes.
3. Stop teleop and rest the follower in its folded pose, as at the start of the episodes. The policy takes over from wherever the arm is.
4. `uv run so101 run cube-dish`. Add `--display` to watch what the policy sees.

What the policy uses: both camera images and the six joint positions, read 30 times a second. It outputs six target joint positions. ACT ignores the task text.

`run` downloads the checkpoint to `outputs/policies/`, names your cameras after the checkpoint's cameras, captures at the resolution it was trained on, and runs it on the Mac's GPU (`[run] device`). It stops after `[run] duration_s` seconds and moves the arm back to where it started. `[run] max_relative_target` caps each joint at 5 degrees per step. Keep a hand near the follower's power switch; Ctrl+C stops the run.

| Task | Props | Cameras |
| --- | --- | --- |
| `cube-dish` | Red ~2.5 cm cube, round ~10 cm silver dish, plain light-wood table | Front camera overhead looking down, wrist camera on the gripper |
| `lego-bowl` | Small LEGO brick, ~20 cm pale bowl, dark wood table | Front camera at the far corner looking diagonally at the arm, wrist camera |
| `lego-box` | Red 2x4 LEGO brick, clear ~15 x 10 cm box outlined in orange tape, white table | Two fixed cameras, front and side; the wrist camera has to come off the gripper |

To add a task, add a section to `tasks.toml`. Find checkpoints on the Hub by searching for `act_so101`, and check their `config.json` for camera names and image sizes.

## Training your own task

Your own tasks live at the end of `tasks.toml`, without `policy` or `dataset`. `cylinder-bowl` (pick up a cylindrical block and put it in a bowl) is yours: copy it and change the name, prompt and props for another task.

`default` at the top of `tasks.toml` names the task commands use when you don't name one, so `uv run so101 record` means `uv run so101 record cylinder-bowl`. Name a task to use another, for example `uv run so101 run cube-dish`.

One-time setup: create a write token at https://huggingface.co/settings/tokens, run `uv run hf auth login`, and add credit to your Hugging Face account (pay-as-you-go or PRO) for cloud training. Recording works without it.

1. **Set up the scene and keep it fixed.** Same camera positions, lighting and table for recording and running; the policy only learns this scene. Then `uv run so101 check --save-reference` to save each camera's view.
2. **Rehearse:** `uv run so101 record --dry-run`. Fix anything it reports before recording demos you want to keep.
3. **Record:** `uv run so101 record`. Drive the follower with the leader for each demo. A Rerun window shows both cameras.
   - Each demo lasts `episode_time_s` (20 s); Right arrow or `n` finishes it early, Left arrow or `r` redoes it, Esc or `q` stops the session. Esc mid-demo keeps the demo so far, so delete it in `review` if it's incomplete.
   - Between demos you get `reset_time_s` (10 s) to put the cylinder back.
   - Record about 50 demos: 5 cylinder positions x 10 demos each, with the bowl in one spot. Start every demo from the same rest pose and do the task the same way each time; smooth, deliberate demos train better than fast ones.
   - Every run of `record` is a new session. Stopping is safe: running it again records until there are `episodes` kept demos. `--episodes 10` adds 10 more; `--note "lamp on"` saves a note with the session.
4. **Review:** `uv run so101 review`.
   - Each new demo opens in a Rerun window: both cameras plus joint plots, with a timeline in seconds.
   - It suggests cutting the idle time at the start and end. It keeps 0.2 s before the first movement and 0.7 s after the last, so the policy still learns to settle.
   - Press Enter to accept the suggestion. Or type `t START END` to keep a different part, `k` to keep it all, or `d reason` to delete it.
   - Decisions go to `data/<task>/edits.toml`; the recordings are never changed.
5. **Train:** `uv run so101 train`.
   - It builds the training dataset from every session and your edits.
   - It uploads the dataset to the private Hub dataset `<you>/so101_<task>`, replacing any earlier upload.
   - It trains ACT on Hugging Face Jobs (`[train]` in `so101.toml`: an A100 for 30k steps at batch 64, roughly 30 minutes and a few dollars, capped at 3 hours).
   - Logs stream to the terminal. Ctrl+C stops watching but the job keeps running (`uv run hf jobs ps`, `uv run hf jobs logs <id>`).
   - The trained policy lands in the private repo `<you>/act_so101_<task>`.
   - `--target local` trains on this Mac instead.
6. **Run:** `uv run so101 run --display`, from the same rest pose with the cylinder somewhere you recorded.

If it fails at some cylinder positions, record 10 to 20 more demos there (`--episodes 20`), review them, and train again.

### Your demos on disk

```
data/<task>/
  sessions/<YYYYmmdd-HHMMSS>/
    dataset/        one LeRobot dataset per `record` run; never modified
    session.json    what it was recorded with: settings, camera ids and controls,
                    both calibrations, software versions, check results, your note
  edits.toml        trims and deletions from `review` (you can edit it by hand)
  train/            the training dataset `build` makes; rebuilt when sessions or edits change
  build.json        which session episodes and frames went into train/
```

- `data/` sits next to `so101.toml` and is git-ignored. Change it with `[data] dir` in `so101.toml`.
- `uv run so101 backup <task>` uploads the sessions and edits to the private Hub dataset `<you>/so101_<task>_raw`, and prints the command to restore them. Demos are your time; back them up.
- Each session is a separate LeRobot dataset, so a crash or a bad session only affects that session. To drop a whole session, move its folder out of `sessions/`.
- **Building:**
  - decodes each kept episode and writes it into a fresh dataset with LeRobot's own writer, at a higher video quality than recording to limit re-encoding loss;
  - checks the result before replacing the old build;
  - stops if sessions were recorded with different fps, cameras or joints, because those can't share one dataset.
- `check` and `record` warn when the calibration changed since the last session, because joint values then no longer line up with earlier demos.

## Troubleshooting

- **Missing motor IDs on connect:** run `uv run so101 scan`. Nothing at any baud rate means a hardware fault: servo power off, Waveshare jumpers not on B, or a loose 3-pin cable. Ids at a baud rate other than 1000000 mean the motors need `setup-motors`. Some ids missing means a break in the daisy chain after the last one that answered.
- **Board not found:** plug it in, then run `uv run so101 info`. If the port has changed, run `find-port`.
- **`USB camera ... is not connected`:** the camera with that `usb_id` isn't plugged in (or is on a dead hub). The message lists the cameras that are connected.
- **A camera "delivers 22 new frames/s against the 30 configured":** the camera can't keep up, so some recorded frames repeat the one before. Dim light makes auto-exposure cameras slower; add light, or set a shorter fixed exposure.
- **Dry run: "the recording loop ran at 25 of 30 steps/s":** the Mac can't record that fast with these settings, so demos would play back faster than you moved. Record with `--no-display`, close other apps, or try `--dataset.rgb_encoder.vcodec=auto` (hardware video encoding).
- **`WARNING: shoulder_pan barely moved during calibration`:** that joint's calibrated range is tiny, so the arm can hardly move it. Recalibrate the arm and move every joint through its full range.
- **`objc[...]: Class AVFFrameReceiver is implemented in both ...`:** OpenCV and PyAV each bundle FFmpeg. The warning is harmless.
- **A checkpoint "sets config fields this LeRobot version doesn't support":** it was saved by a newer LeRobot. `run` drops unset fields itself; set ones need a LeRobot upgrade.
- **`Could not load this library: ...libtorchcodec_core...`** tracebacks when LeRobot starts: torchcodec can't find a system FFmpeg, so LeRobot falls back to PyAV for video. Harmless here; `brew install ffmpeg` silences it.

## License

[MIT](LICENSE).
