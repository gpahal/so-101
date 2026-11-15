"""Run a policy on a Hugging Face Jobs GPU while the arm and cameras stay on this Mac.

The job runs LeRobot's policy server, listening only on its own loopback. An SSH tunnel makes that
port appear on this Mac, and `so101-rollout` (`run --remote`) or `so101-control` (`serve --remote`)
connects to it with `client_args`.
"""

import base64
import json
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress

from so_101 import svtshim

# LeRobot publishes no image tagged for a release, so this is what `latest` was on 2026-11-11, by
# digest: a job must not change under a working setup when `latest` moves. Move it on purpose. ACT and
# pi0.5 have both run on it against the 0.7.0 client, on 2026-11-13. `wait_until_ready` checks what
# PI05_SERVER changes in it took effect, so a new image that renames them fails at `server start`.
SERVER_IMAGE = "huggingface/lerobot-gpu@sha256:aed1a4990fedee463b545050f33a38afce4f6ec18f7ceb8be17f850737860d6c"
PORT = 7447
DEPLOYMENT = "so101"
SEMANTICS = "so101-degrees-gripper-percent-v1"
SERVER_LABEL = "policy-server"
PIN_LABEL = "pin_unused_actions"
# What the pi0.5 job prints the first time it pins the unused action columns, during warm-up.
PIN_MARKER = "so101: unused action columns pinned"
SSH_KEYS_URL = "https://huggingface.co/settings/keys"
DONE_STAGES = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
# ssh gives up after 15 s without an answer (ServerAliveInterval 5 x ServerAliveCountMax 3), and is
# started again this long after each exit, so the tunnel comes back on its own after a network drop.
RESTART_DELAY_S = 5.0

# How each policy's chunks of actions are joined. ACT's server returns plain chunks, each replacing the
# rest of the last ("chunk"). pi0.5 uses guided real-time chunking ("rtc_guided"): each request carries
# the actions still queued, and the server steers the new chunk to agree with them, so a new plan
# can't jump away from the motion under way. Plain chunks measured jumps of up to 47 degrees at the
# seams with pi0.5, because its flow-matching samples can disagree from one second to the next.
EXECUTION = {"act": "chunk", "pi05": "rtc_guided"}
# With RTC, the client asks for the next chunk once this much motion is left in its queue, or the
# turnaround plus one step if that's more. LeRobot's 0.5 s, raised to the turnaround, left 0.71 s
# against a 0.65 s turnaround from India, and one request that took over 1.7 s, likely a network
# stall since the GPU takes 0.45 s at most, ran the arm out of actions after 18 s. With 1 s, and the
# grace below, a stall of about 2.3 s holds the arm still instead. The cost is requests back to back,
# about 1.5 a second instead of 1, which RTC makes harmless to the motion.
RTC_REFILL_S = 1.0
# How long the arm holds its last target when the queue runs out before the run fails (LeRobot: 1 s).
RTC_STARVATION_GRACE_S = 2.0
# With RTC, how much of the queued motion the server steers the new chunk to match: all of it, since
# the arm plays the queue while the request is out, and every step it plays before the new chunk takes
# over must be steered. LeRobot's default of 10 steps (0.33 s) would leave the seam free. Past the
# queue, the server pads with its last action, pulling the new chunk toward a hold, so this matches
# the refill. Seams are steered while the turnaround stays under it.
RTC_HORIZON_S = RTC_REFILL_S

# The script the pi0.5 job runs in place of `lerobot-policy-server`. `setup_script` writes it after a
# line that sets PIN_UNUSED_ACTIONS. It changes four things in LeRobot before starting its server.
#
# LeRobot 0.7's server requires every input the model declares, but the client can only send cameras
# that exist. pi0.5 pads and masks a missing camera itself, which is what it saw in training, so
# accept its empty camera slots missing. This follows LeRobot's own override for SmolVLA.
#
# pi0.5 has 32 action columns and this arm uses 6. Training gives the other 26 a target of 0 and
# leaves them out of the loss, so at every denoising step the model saw them as `time * noise`, and
# nothing taught it where to move them. LeRobot's sampler moves them by the model's output anyway:
# they end the last step 1.5 times as wide as in training (0.149 against 0.100), and the 6 real
# columns, which read them, wobble. `pin_unused_actions` gives the model those columns as training
# did, at every step. On 24 demo frames with 3 samples each, on an L4, the motion the arm would play
# across two chunks went from 0.73 to 0.23 degrees per step squared (RMS second difference; the demos
# measure 0.11 and ACT's plans 0.23) and from 25.7 direction changes per 50 steps to 3.4. The error
# against the demos didn't change (3.28 to 3.26 degrees), nor the time per chunk (0.41 s).
#
# Training keeps the weights in 32-bit, and LeRobot serves a model in the precision it was trained
# in. Serve in bfloat16 instead: on an L4 a chunk took 0.36 s against 0.77 s in 32-bit, and the plans
# differed by 0.09 degrees. A model trained compiled would be served compiled too, so that is turned
# off: the pin above is plain Python that must run at every step.
#
# LeRobot's server downloads the whole model repo, and training leaves its checkpoints there, about
# 72 GB each for pi0.5. Download the model without them.
PI05_SERVER = '''\
import torch
from lerobot.configs import FeatureType
from lerobot.remote_inference import deployment
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.scripts.lerobot_policy_server import main


def validate(self, features):
    expected = self.config.input_features or {}
    supplied = {f.name: f for f in features}
    required = {n for n in expected if not n.startswith("observation.images.empty_camera_")}
    if not required <= supplied.keys() or not supplied.keys() <= expected.keys():
        raise ValueError(f"Expected {sorted(required)}, got {sorted(supplied)}")
    for name, feature in supplied.items():
        wanted = expected[name]
        if wanted.type == FeatureType.VISUAL:
            if feature.kind != "rgb" or feature.shape != (wanted.shape[1], wanted.shape[2], wanted.shape[0]):
                raise ValueError(f"RGB shape differs: {name}")
        elif feature.kind != "tensor" or feature.shape != tuple(wanted.shape):
            raise ValueError(f"Tensor shape differs: {name}")


def pin_unused_actions():
    sample, denoise = PI05Pytorch.sample_actions, PI05Pytorch.denoise_step
    start, pinned = {}, []

    def sample_actions(self, *args, **kwargs):
        start.clear()  # every chunk starts from new noise, which its first step below records
        return sample(self, *args, **kwargs)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        used = self.config.output_features["action"].shape[0]
        if "noise" not in start:  # the first step's input is the starting noise, at time 1
            start["noise"] = x_t[:, :, used:].detach().clone()
        time = timestep.reshape(timestep.shape + (1,) * (x_t.ndim - timestep.ndim)).to(x_t.dtype)
        x_t = x_t.clone()
        x_t[:, :, used:] = time * start["noise"]
        if not pinned:  # `so101 server start` waits for this line
            pinned.append(True)
            print(f"so101: unused action columns pinned (columns {used} to {x_t.shape[2] - 1})", flush=True)
        return denoise(
            self, prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t, timestep=timestep
        )

    PI05Pytorch.sample_actions = sample_actions
    PI05Pytorch.denoise_step = denoise_step


load = PI05Policy.from_pretrained.__func__


def from_pretrained(cls, *args, config=None, **kwargs):
    if config is not None:
        config.dtype = torch.bfloat16
        config.compile_model = False
    return load(cls, *args, config=config, **kwargs)


download = deployment.snapshot_download


def download_model(*args, **kwargs):
    return download(*args, ignore_patterns=["checkpoints/**"], **kwargs)


deployment.snapshot_download = download_model
PI05Policy.from_pretrained = classmethod(from_pretrained)
PI05Policy.validate_chunk_input_features = validate
if PIN_UNUSED_ACTIONS:
    pin_unused_actions()
main()
'''


def server_config(
    repo: str, policy_type: str, revision: str, image_sizes: dict[str, tuple[int, int]], action_steps: int, fps: int
) -> dict:
    """The policy server's config, describing the model's inputs and outputs. `image_sizes` is each
    camera's (width, height), from `tasks.image_sizes`. With real-time chunking, `action_steps` is
    the model's own and only bounds the steered horizon: LeRobot takes no execution slice then."""
    vector = {"dtype": "float32", "names": JOINTS, "semantics": SEMANTICS, "shape": [len(JOINTS)]}
    cameras = [
        {
            "name": f"observation.images.{camera}",
            "shape": [height, width, 3],
            "dtype": "uint8",
            "kind": "rgb",
            "semantics": f"rgb-{camera}-v1",
        }
        for camera, (width, height) in image_sizes.items()
    ]
    return {
        "deployment": DEPLOYMENT,
        "model": {"repo_or_path": repo, "revision": revision, "device": "cuda"},
        "execution": execution_config(EXECUTION[policy_type], action_steps, fps),
        "language": {"enabled": False},
        "zenoh": {"mode": "peer", "listen_endpoints": [f"tcp/127.0.0.1:{PORT}"]},
        "semantics": SEMANTICS,
        "robot_type": "so101_follower",
        "features": [{"name": "observation.state", **vector}, *cameras],
        "action_feature": {"name": "action", **vector},
    }


def execution_config(mode: str, action_steps: int, fps: int) -> dict:
    if mode == "chunk":
        return {"supported_modes": ["chunk"], "action_fps": float(fps), "n_action_steps": action_steps}
    horizon = min(action_steps, round(RTC_HORIZON_S * fps))
    return {
        "supported_modes": [mode],
        "action_fps": float(fps),
        "rtc": {"enabled": True, "mode": mode.removeprefix("rtc_"), "execution_horizon": horizon},
    }


def refuse_action_steps(policy_type: str, requested: int | None) -> None:
    """With real-time chunking the client, not a fixed step count, decides when to predict again."""
    if requested is not None and EXECUTION[policy_type] != "chunk":
        sys.exit(f"{policy_type} on the cloud GPU uses real-time chunking, which asks for the next prediction "
                 "once the measured turnaround plus one step of motion is left, so it takes no --action-steps.")


def steps_flag(policy_type: str, action_steps: int) -> str:
    """The --action-steps flag that names a server's model, or nothing with real-time chunking."""
    return f" --action-steps {action_steps}" if EXECUTION[policy_type] == "chunk" else ""


def setup_script(policy_type: str, config: dict, pin_unused_actions: bool) -> tuple[str, str]:
    """Shell lines that write the job's files, and the command that then starts the server.
    `pin_unused_actions` applies to pi0.5 only: see PI05_SERVER."""

    def write(path: str, text: str) -> str:
        return f"echo {base64.b64encode(text.encode()).decode()} | base64 -d > {path}\n"

    # JSON is valid YAML, and LeRobot picks the parser from the file's extension.
    script = write("/tmp/server.yaml", json.dumps(config))
    if policy_type == "pi05":
        serve = f"PIN_UNUSED_ACTIONS = {pin_unused_actions!r}\n{PI05_SERVER}"
        return script + write("/tmp/serve.py", serve), "python /tmp/serve.py --config_path=/tmp/server.yaml"
    return script, "lerobot-policy-server --config_path=/tmp/server.yaml"


def check_ssh_key() -> None:
    """Stop before paying for a job the tunnel couldn't reach."""
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "git@hf.co"],
        capture_output=True,
        text=True,
        env=svtshim.other_env(),
    )
    reply = result.stdout + result.stderr
    if "welcome to hugging face" not in reply.lower() or "hi anonymous" in reply.lower():
        sys.exit(
            "Your SSH key isn't registered on Hugging Face, and the tunnel to the job needs it.\n"
            f"  1. pbcopy < ~/.ssh/id_ed25519.pub  (`ssh-keygen -t ed25519` first if you have no key)\n"
            f"  2. Paste it at {SSH_KEYS_URL}\n"
            "  3. `ssh -T git@hf.co` should now greet you by name."
        )


def active_servers() -> list:
    """Every running policy-server job. There should be one, but each one bills."""
    from huggingface_hub import HfApi

    return [job for job in HfApi().list_jobs(labels={"so101": SERVER_LABEL}) if job.status.stage not in DONE_STAGES]


def active_server():
    """The running policy-server job, if there is one."""
    servers = active_servers()
    return servers[0] if servers else None


def describe(job) -> str:
    labels = job.labels or {}
    execution = labels.get("execution")
    steps = (f"{labels.get('action_steps')} action steps" if execution == "chunk"
             else "real-time chunking" if execution else "plain chunks from before real-time chunking")
    pin = {"true": ", unused action columns pinned", "false": ", unused action columns not pinned"}.get(
        labels.get(PIN_LABEL), ", started before the unused action columns setting" if labels.get("policy") == "pi05" else "")
    return (
        f"{labels.get('policy')} for {labels.get('task')} at {labels.get('revision')}, {steps}{pin}, "
        f"on {job.flavor} (job {job.id}, {job.status.stage})"
    )


def _labels(task: str, policy_type: str, revision: str, action_steps: int, pin_unused_actions: bool) -> dict[str, str]:
    """The job labels that say which model a server serves, how its chunks are joined and, for pi0.5,
    whether its unused action columns are pinned: `start_server` sets them, and `require_server`
    matches them."""
    labels = {"task": task, "policy": policy_type, "revision": revision, "action_steps": str(action_steps),
              "execution": EXECUTION[policy_type]}
    if policy_type == "pi05":
        labels[PIN_LABEL] = "true" if pin_unused_actions else "false"
    return labels


def start_server(
    *, repo: str, task: str, policy_type: str, revision: str, image_sizes: dict[str, tuple[int, int]], action_steps: int,
    fps: int, target: str, timeout: str, pin_unused_actions: bool,
) -> None:
    """Start the job and wait until the model is loaded and serving."""
    from huggingface_hub import HfApi, get_token

    check_ssh_key()
    if server := active_server():
        sys.exit(f"A policy server is already running: {describe(server)}.\nStop it first: uv run so101 server stop")
    setup, serve = setup_script(
        policy_type, server_config(repo, policy_type, revision, image_sizes, action_steps, fps), pin_unused_actions
    )
    api = HfApi()
    job = api.run_job(
        image=SERVER_IMAGE,
        command=["bash", "-c", f"{setup}exec {serve}\n"],
        env={"PYTHONUNBUFFERED": "1"},
        secrets={"HF_TOKEN": get_token()},
        flavor=target,
        timeout=timeout,
        labels={"so101": SERVER_LABEL, **_labels(task, policy_type, revision, action_steps, pin_unused_actions)},
        ssh=True,
    )
    print(f"Started job {job.id} on {target}. It bills until `uv run so101 server stop`, or {timeout} at most.")
    print(f"Logs: {job.url}\nCtrl+C stops watching but leaves it running. Loading the model takes a few minutes...\n")
    try:
        wait_until_ready(job.id, policy_type, pin_unused_actions)
    except KeyboardInterrupt:
        sys.exit("\nStopped watching. The job is still running and billing: uv run so101 server stop")
    steps = steps_flag(policy_type, action_steps)
    print(
        "\nThe policy server is ready. Next:\n"
        f"  uv run so101 run {task} --policy {policy_type} --remote --revision {revision}{steps}\n"
        f"  uv run so101 serve {task} --policy {policy_type} --remote --revision {revision}{steps}\n"
        "  uv run so101 server stop"
    )


def wait_until_ready(job_id: str, policy_type: str, pin_unused_actions: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    weights_loaded = pinned = False
    for line in api.fetch_job_logs(job_id=job_id, follow=True):
        print(line, flush=True)
        weights_loaded = weights_loaded or "All keys loaded successfully!" in line
        pinned = pinned or PIN_MARKER in line
        if "Policy server ready" in line:
            # pi0.5 only warns when its weights fail to load, and would then serve random ones.
            if policy_type == "pi05" and not weights_loaded:
                api.cancel_job(job_id=job_id)
                sys.exit("The model's weights didn't load cleanly, so the job was cancelled. See the log above.")
            # The pin wraps two of LeRobot's methods by name. If an image renamed one, the server would
            # run unpinned under a label that says pinned.
            if policy_type == "pi05" and pin_unused_actions and not pinned:
                api.cancel_job(job_id=job_id)
                sys.exit("The server warmed up without pinning the unused action columns, so the job was cancelled: "
                         "LeRobot in this image no longer matches PI05_SERVER in src/so_101/remote.py.")
            return
    status = api.inspect_job(job_id=job_id).status
    sys.exit(f"The job stopped before the server was ready: {status.stage} {status.message or ''}")


def stop_server() -> None:
    from huggingface_hub import HfApi

    servers = active_servers()
    if not servers:
        print("No policy server is running.")
        return
    api = HfApi()
    failed = []
    for server in servers:  # one that can't be cancelled must not leave the others billing
        try:
            api.cancel_job(job_id=server.id)
        except Exception as error:
            failed.append(f"{describe(server)}: {error}")
        else:
            print(f"Stopped {describe(server)}.")
    if failed:
        sys.exit("Couldn't stop these, and they may still bill. Run `uv run so101 server stop` again:\n  "
                 + "\n  ".join(failed))


def require_server(task: str, policy_type: str, revision: str, action_steps: int, pin_unused_actions: bool):
    """The running server, if it serves what this run asks for, as so101.toml sets it. Starting one
    bills, so never do it implicitly."""
    start = (f"uv run so101 server start {task} --policy {policy_type} --revision {revision}"
             f"{steps_flag(policy_type, action_steps)}")
    server = active_server()
    if server is None:
        sys.exit(f"No policy server is running. Start one first:\n  {start}")
    labels = server.labels or {}
    wanted = _labels(task, policy_type, revision, action_steps, pin_unused_actions)
    if {key: labels.get(key) for key in wanted} != wanted or server.status.stage != "RUNNING":
        sys.exit(f"The policy server isn't serving this model: {describe(server)}.\nStop it, then: {start}")
    return server


@contextmanager
def tunnel(job_id: str) -> Iterator[None]:
    """Forward the job's server port to this Mac for as long as the block runs, restarting ssh when it exits."""
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            sys.exit(f"Something on this Mac already uses port {PORT}. Close it and try again.")
    proc = _start_ssh(job_id)
    supervisor = None
    try:
        deadline = time.monotonic() + 30
        while True:
            if proc.poll() is not None:
                sys.exit(f"The SSH tunnel to the job failed. Is your key registered at {SSH_KEYS_URL}?")
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", PORT)) == 0:
                    break
            if time.monotonic() > deadline:
                sys.exit("The SSH tunnel to the job didn't open within 30 seconds.")
            time.sleep(0.2)
        supervisor = _Supervisor(job_id, proc)
        yield
    finally:
        if supervisor is not None:
            proc = supervisor.stop()
        proc.terminate()
        proc.wait()


def _start_ssh(job_id: str) -> subprocess.Popen:
    # In its own session, so Ctrl+C lets the client shut down over a tunnel that's still open.
    return subprocess.Popen(
        [
            "ssh", "-N", "-L", f"127.0.0.1:{PORT}:127.0.0.1:{PORT}",
            "-o", "ForwardAgent=no",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=3",
            "-o", "ControlMaster=no",
            "-o", "ControlPath=none",
            f"{job_id}@ssh.hf.jobs",
        ],
        start_new_session=True,
        env=svtshim.other_env(),
    )  # fmt: skip


class _Supervisor:
    """Starts ssh again `RESTART_DELAY_S` after each exit or failed start, until `stop`. Its printed
    line can't end its thread: once the terminal is closed, printing fails, and the line is dropped."""

    def __init__(self, job_id: str, proc: subprocess.Popen) -> None:
        self.job_id = job_id
        self.proc = proc
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ssh-tunnel", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stopping.wait(0.2):
            if self.proc.poll() is None:
                continue
            with suppress(OSError, ValueError, RuntimeError):  # a closed terminal (EIO), a closed stdout
                print("SSH tunnel to the job dropped; reconnecting.", flush=True)
            if self.stopping.wait(RESTART_DELAY_S):
                return
            with suppress(OSError):  # as when too many files are open: the next pass tries again
                self.proc = _start_ssh(self.job_id)

    def stop(self) -> subprocess.Popen:
        """Stop restarting, and return the ssh that's running now, or that ran last."""
        self.stopping.set()
        self.thread.join()
        return self.proc


def client_args(policy_type: str) -> list[str]:
    mode = EXECUTION[policy_type]
    return [
        "--inference.type=remote",
        f"--inference.endpoint=tcp/127.0.0.1:{PORT}",
        f"--inference.deployment={DEPLOYMENT}",
        f"--inference.semantics={SEMANTICS}",
        f"--inference.mode={mode}",
        # A plain chunk replaces the rest of the last from its own time on. LeRobot takes only "append"
        # with RTC, and its RTC queue still replaces the rest of the last chunk, from the step the arm
        # reached while the request was out, with the new chunk the server steered to agree with it.
        f"--inference.chunk_merge={'aligned' if mode == 'chunk' else 'append'}",
        *([] if mode == "chunk" else [f"--inference.refill_seconds={RTC_REFILL_S:g}",
                                      f"--inference.action_starvation_grace_s={RTC_STARVATION_GRACE_S:g}"]),
        # Measured with pi0.5 on an L4 from India: quality 90 ran out of actions and faulted within 10
        # seconds, 50 kept the arm moving 90% of the time, and 30 kept it moving 97% of the time.
        "--inference.encoding=jpeg",
        "--inference.jpeg_quality=30",
    ]
