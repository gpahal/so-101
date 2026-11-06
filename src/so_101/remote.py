"""Run a policy on a Hugging Face Jobs GPU while the arm and cameras stay on this Mac.

The job runs LeRobot's policy server, listening only on its own loopback. An SSH tunnel makes that
port appear on this Mac, and `lerobot-rollout --inference.type=remote` connects to it.
"""

import base64
import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

SERVER_IMAGE = "huggingface/lerobot-gpu:latest"
PORT = 7447
DEPLOYMENT = "so101"
SEMANTICS = "so101-degrees-gripper-percent-v1"
SERVER_LABEL = "policy-server"
SSH_KEYS_URL = "https://huggingface.co/settings/keys"
DONE_STAGES = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
IMAGE_PREFIX = "observation.images."
EMPTY_CAMERA_PREFIX = IMAGE_PREFIX + "empty_camera_"

# LeRobot 0.7's server requires every input the model declares, but the client can only send cameras
# that exist. pi0.5 pads and masks a missing camera itself, which is what it saw in training, so
# accept its empty camera slots missing. This follows LeRobot's own override for SmolVLA.
PI05_SERVER = '''\
from lerobot.configs import FeatureType
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
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


PI05Policy.validate_chunk_input_features = validate
main()
'''


def server_config(repo: str, revision: str, config: dict, action_steps: int) -> dict:
    """The policy server's config, describing the model's inputs and outputs from its config.json."""
    vector = {"dtype": "float32", "names": JOINTS, "semantics": SEMANTICS, "shape": [len(JOINTS)]}
    cameras = [
        {
            "name": key,
            "shape": [feature["shape"][1], feature["shape"][2], feature["shape"][0]],
            "dtype": "uint8",
            "kind": "rgb",
            "semantics": f"rgb-{key.removeprefix(IMAGE_PREFIX)}-v1",
        }
        for key, feature in config["input_features"].items()
        if key.startswith(IMAGE_PREFIX) and not key.startswith(EMPTY_CAMERA_PREFIX)
    ]
    return {
        "deployment": DEPLOYMENT,
        "model": {"repo_or_path": repo, "revision": revision, "device": "cuda"},
        "execution": {"supported_modes": ["chunk"], "action_fps": 30.0, "n_action_steps": action_steps},
        "language": {"enabled": False},
        "zenoh": {"mode": "peer", "listen_endpoints": [f"tcp/127.0.0.1:{PORT}"]},
        "semantics": SEMANTICS,
        "robot_type": "so101_follower",
        "features": [{"name": "observation.state", **vector}, *cameras],
        "action_feature": {"name": "action", **vector},
    }


def setup_script(policy_type: str, config: dict) -> tuple[str, str]:
    """Shell lines that write the job's files, and the command that then starts the server."""

    def write(path: str, text: str) -> str:
        return f"echo {base64.b64encode(text.encode()).decode()} | base64 -d > {path}\n"

    # JSON is valid YAML, and LeRobot picks the parser from the file's extension.
    script = write("/tmp/server.yaml", json.dumps(config))
    if policy_type == "pi05":
        return script + write("/tmp/serve.py", PI05_SERVER), "python /tmp/serve.py --config_path=/tmp/server.yaml"
    return script, "lerobot-policy-server --config_path=/tmp/server.yaml"


def check_ssh_key() -> None:
    """Stop before paying for a job the tunnel couldn't reach."""
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "git@hf.co"],
        capture_output=True,
        text=True,
    )
    reply = result.stdout + result.stderr
    if "welcome to hugging face" not in reply.lower() or "hi anonymous" in reply.lower():
        sys.exit(
            "Your SSH key isn't registered on Hugging Face, and the tunnel to the job needs it.\n"
            f"  1. pbcopy < ~/.ssh/id_ed25519.pub  (`ssh-keygen -t ed25519` first if you have no key)\n"
            f"  2. Paste it at {SSH_KEYS_URL}\n"
            "  3. `ssh -T git@hf.co` should now greet you by name."
        )


def active_server():
    """The running policy-server job, if there is one."""
    from huggingface_hub import HfApi

    servers = [
        job
        for job in HfApi().list_jobs(labels={"so101": SERVER_LABEL})
        if job.status.stage not in DONE_STAGES
    ]
    return servers[0] if servers else None


def describe(job) -> str:
    labels = job.labels or {}
    return (
        f"{labels.get('policy')} for {labels.get('task')} at {labels.get('revision')}, "
        f"{labels.get('action_steps')} action steps, "
        f"on {job.flavor} (job {job.id}, {job.status.stage})"
    )


def start_server(
    *, repo: str, task: str, policy_type: str, revision: str, config: dict, action_steps: int, target: str, timeout: str
) -> None:
    """Start the job and wait until the model is loaded and serving."""
    from huggingface_hub import HfApi, get_token

    check_ssh_key()
    if server := active_server():
        sys.exit(f"A policy server is already running: {describe(server)}.\nStop it first: uv run so101 server stop")
    setup, serve = setup_script(policy_type, server_config(repo, revision, config, action_steps))
    api = HfApi()
    job = api.run_job(
        image=SERVER_IMAGE,
        command=["bash", "-c", f"{setup}exec {serve}\n"],
        env={"PYTHONUNBUFFERED": "1"},
        secrets={"HF_TOKEN": get_token()},
        flavor=target,
        timeout=timeout,
        labels={
            "so101": SERVER_LABEL,
            "task": task,
            "policy": policy_type,
            "revision": revision,
            "action_steps": str(action_steps),
        },
        ssh=True,
    )
    print(f"Started job {job.id} on {target}. It bills until `uv run so101 server stop`, or {timeout} at most.")
    print(f"Logs: {job.url}\nCtrl+C stops watching but leaves it running. Loading the model takes a few minutes...\n")
    try:
        wait_until_ready(job.id, policy_type)
    except KeyboardInterrupt:
        sys.exit("\nStopped watching. The job is still running and billing: uv run so101 server stop")
    print(
        "\nThe policy server is ready. Next:\n"
        f"  uv run so101 run {task} --policy {policy_type} --remote --revision {revision} --action-steps {action_steps}\n"
        "  uv run so101 server stop"
    )


def wait_until_ready(job_id: str, policy_type: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    weights_loaded = False
    for line in api.fetch_job_logs(job_id=job_id, follow=True):
        print(line, flush=True)
        weights_loaded = weights_loaded or "All keys loaded successfully!" in line
        if "Policy server ready" in line:
            # pi0.5 only warns when its weights fail to load, and would then serve random ones.
            if policy_type == "pi05" and not weights_loaded:
                api.cancel_job(job_id=job_id)
                sys.exit("The model's weights didn't load cleanly, so the job was cancelled. See the log above.")
            return
    status = api.inspect_job(job_id=job_id).status
    sys.exit(f"The job stopped before the server was ready: {status.stage} {status.message or ''}")


def stop_server() -> None:
    from huggingface_hub import HfApi

    server = active_server()
    if server is None:
        print("No policy server is running.")
        return
    HfApi().cancel_job(job_id=server.id)
    print(f"Stopped {describe(server)}.")


def require_server(task: str, policy_type: str, revision: str, action_steps: int):
    """The running server, if it serves what this run asks for. Starting one bills, so never do it implicitly."""
    start = f"uv run so101 server start {task} --policy {policy_type} --revision {revision} --action-steps {action_steps}"
    server = active_server()
    if server is None:
        sys.exit(f"No policy server is running. Start one first:\n  {start}")
    labels = server.labels or {}
    wanted = {"task": task, "policy": policy_type, "revision": revision, "action_steps": str(action_steps)}
    if {key: labels.get(key) for key in wanted} != wanted or server.status.stage != "RUNNING":
        sys.exit(f"The policy server isn't serving this model: {describe(server)}.\nStop it, then: {start}")
    return server


@contextmanager
def tunnel(job_id: str) -> Iterator[None]:
    """Forward the job's server port to this Mac for as long as the block runs."""
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            sys.exit(f"Something on this Mac already uses port {PORT}. Close it and try again.")
    # In its own session, so Ctrl+C lets the client shut down over a tunnel that's still open.
    proc = subprocess.Popen(
        [
            "ssh", "-N", "-L", f"{PORT}:127.0.0.1:{PORT}",
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
    )  # fmt: skip
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
        yield
    finally:
        proc.terminate()
        proc.wait()


def client_args() -> list[str]:
    return [
        "--inference.type=remote",
        f"--inference.endpoint=tcp/127.0.0.1:{PORT}",
        f"--inference.deployment={DEPLOYMENT}",
        f"--inference.semantics={SEMANTICS}",
        "--inference.mode=chunk",
        "--inference.chunk_merge=aligned",
        # Measured with pi0.5 on an L4 from India: quality 90 ran out of actions and faulted within 10
        # seconds, 50 kept the arm moving 90% of the time, and 30 kept it moving 97% of the time.
        "--inference.encoding=jpeg",
        "--inference.jpeg_quality=30",
    ]
