"""The checks and LeRobot arguments `so101 run` and `so101 serve` share."""

import argparse
import json
import re
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from so_101 import remote, settings, tasks
from so_101.check import preflight
from so_101.console import run_lerobot
from so_101.report import Report
from so_101.settings import ARMS, PortNotFoundError, cameras_arg
from so_101.skills import STATE_KEY
from so_101.tasks import get_task


@dataclass(frozen=True)
class PolicySetup:
    """What `run` and `serve` both need to drive the follower with a task's policy, after their
    shared checks: the policy, robot and camera arguments, and the remote server, if any."""

    task: tasks.Task
    action_steps: int
    server: object | None  # the running policy server's job, with --remote
    sizes: dict[str, tuple[int, int]]  # each camera's (width, height), as the checkpoint was trained
    arms: tuple[settings.ArmName, ...]  # the arms the checks covered
    found: list  # the task's cameras, in order, as the checks found them, with their current indices
    report: Report
    checkpoint: Path | None  # the local checkpoint; None with a remote server
    args: list[str]  # the strategy, policy, robot and camera arguments, and the step cap
    # A skills run's `[gripper]` thresholds hash (`skills.gripper_hash`), as checked against the model's build;
    # run.json records it, and so101-control checks the thresholds it loads against it. None otherwise.
    gripper_sha256: str | None = None
    # The Hub commit every file of the model was read at: the server's for a remote run (`remote.served_commit`),
    # else the one the revision named when the run started. None with --checkpoint.
    commit: str | None = None

    def steps_line(self, s: settings.Settings) -> str:
        n, cap = self.action_steps, s.teleop.max_relative_target
        limit = f"limit {cap:g} per step" if cap else "no per-step limit"
        if not self.server:
            return f"Predicts every {n} steps ({n / self.task.fps:.1f} s), {limit}."
        if remote.EXECUTION[self.server.labels["policy"]] == "chunk":
            asks = "asks for the next once the turnaround plus one step is left (at least 0.5 s)"
            return f"Predicts {n} steps ({n / self.task.fps:.1f} s) at a time and {asks}, {limit}."
        return (f"Predicts {n / self.task.fps:.1f} s at a time with real-time chunking: it asks for the next "
                f"once {remote.RTC_REFILL_S:g} s is left (more if the turnaround is longer), and the server "
                f"steers each new prediction to agree with the motion under way, {limit}.")

    def run(self, command: str, args: list[str], log_path: Path | None, summary: Callable[[], None]) -> NoReturn:
        """Run `command` through the SSH tunnel to the remote server, if any, then call `summary`, say
        what still bills, and exit with the command's status: 130 when Ctrl+C lands while the tunnel opens."""
        code = 130
        try:
            with remote.tunnel(self.server.id) if self.server else nullcontext():
                code = run_lerobot(command, args, log_path=log_path)
        except KeyboardInterrupt:
            pass
        finally:
            summary()
            if self.server:
                print(f"The policy server keeps billing until: uv run so101 server stop (job {self.server.id})", flush=True)
        sys.exit(code)


def policy_task(ns: argparse.Namespace) -> tasks.Task:
    """The task `run` or `serve` drives. pi0.5 only runs on the cloud GPU, so it needs --remote. A local
    --checkpoint runs on this Mac, from no Hub revision."""
    task = get_task(ns.task)
    if ns.checkpoint and (ns.remote or ns.revision):
        sys.exit("--checkpoint runs a model saved on this Mac, so it takes neither --remote nor --revision.")
    if ns.policy == "pi05" and not ns.remote:
        sys.exit(f"pi0.5 runs on a cloud GPU. Add --remote, after: uv run so101 server start {task.name} --policy pi05")
    return task


def policy_setup(s: settings.Settings, ns: argparse.Namespace, task: tasks.Task) -> PolicySetup:
    """Check the setup and build the policy, robot and camera arguments for `run` and `serve`.

    With --remote, the server must already serve this model: starting one bills until `so101 server
    stop`, so it's never done here. The leader is checked too when it's plugged in, and its failures
    are only warnings, since only Teleop needs it, and the checks also fail on a bad rest pose or
    `[control]` setting.

    The model is read at one Hub commit (`commit`): a remote run's is the one its server loaded, so a server
    started before a newer model reached main is checked as the model it serves; a local run's is the one
    the revision names when the run starts, so a push meanwhile can't mix two models' files."""
    skills = bool(getattr(ns, "skills", False))
    if skills:
        task.require_skills()
    if ns.remote:
        remote.refuse_action_steps(ns.policy, ns.action_steps)
    dataset, revision = ("skills" if skills else "whole"), ns.revision or "main"
    local = local_checkpoint(ns.checkpoint, ns.policy) if ns.checkpoint else None
    commit = None
    if local is None:
        if ns.remote:
            running = remote.active_server_or_exit(task.name, ns.policy, revision, dataset)
            commit = remote.served_commit(running)
        else:
            commit = task.policy_commit(ns.revision, ns.policy, skills)
    config = json.loads((local / "config.json").read_text()) if local else task.policy_config(commit, ns.policy, skills)
    gripper_sha256 = check_skills_model(s, ns, task, config, local, skills, commit)
    action_steps = resolve_action_steps(ns.action_steps, config)
    server = None
    if ns.remote:
        if ns.policy not in s.remote:
            sys.exit(f"No [remote.{ns.policy}] section in {s.path}.")
        server = remote.require_server(
            task.name, ns.policy, revision, action_steps, s.remote[ns.policy].pin_unused_actions, dataset,
        )
        if (server.labels or {}).get("commit") != commit:
            sys.exit("The policy server changed while the run was being checked. Start the run again.")

    # Capture each camera at the size the checkpoint was trained on.
    sizes = tasks.image_sizes(config)
    if set(sizes) != set(task.cameras):
        model = local or task.policy_for(ns.policy, skills)
        sys.exit(f"{model} expects cameras {sorted(sizes)}, but the task records {sorted(task.cameras)}.")
    arms: tuple[settings.ArmName, ...] = ("follower",)
    try:
        s.leader.resolve_port()
        arms = ARMS
    except PortNotFoundError:
        pass  # the leader is optional: only Teleop needs it
    found, report = preflight(s, "running the policy", arms=arms, cameras=[s.camera(name) for name in task.cameras],
                              control="FAIL", teleop_only=("leader",), video=True, run_task=task.name)
    specs = [cam.lerobot_spec(width=sizes[cam.name][0], height=sizes[cam.name][1]) for cam in found]
    checkpoint = None if server else local or task.download_policy(commit, skills, folder=ns.revision)
    args = [
        "--strategy.type=base",  # LeRobot's default; so101-control runs its own loop, never a strategy
        # A remote server sets the action steps itself, and its client rejects --policy flags.
        *(
            remote.client_args(ns.policy)
            if server
            else [f"--policy.path={checkpoint}", f"--policy.n_action_steps={action_steps}"]
        ),
        *s.follower_args(),
        cameras_arg(specs),
        f"--fps={task.fps}",
        *([] if server else [f"--device={s.run.device}"]),
    ]
    return PolicySetup(task, action_steps, server, sizes, arms, found, report, checkpoint, args, gripper_sha256, commit)


def check_skills_model(s: settings.Settings, ns: argparse.Namespace, task: tasks.Task, config: dict,
                       local: Path | None, skills: bool, commit: str | None = None) -> str | None:
    """Exit unless the model fits how it's run: a model that reads the skill one-hot only with --skills,
    an ACT skills model only with it, with one value per skill in tasks.toml, and a skills model only
    with the skills, gripper thresholds, cut rules and run labelling its dataset was built with
    (`require_skills_contract`), its train_config.json read at `commit`, the Hub commit the run checked.
    Returns the hash of the `[gripper]` thresholds it checked a skills model
    against, for run.json, else None."""
    def model() -> str:
        return str(local or task.policy_for(ns.policy, skills))

    one_hot = (config.get("input_features") or {}).get(STATE_KEY)
    if one_hot and not skills:
        sys.exit(f"{model()} reads a skill one-hot ({STATE_KEY}): it's a skills model. Run it with --skills.")
    if not skills:
        return None
    if not one_hot and ns.policy == "act":
        sys.exit(f"{model()} has no {STATE_KEY} input, so it can't be told which skill it's on: train it with "
                 "`so101 train --skills`.")
    if one_hot and list(one_hot["shape"]) != [len(task.skills)]:
        sys.exit(f"{model()} reads a one-hot of {one_hot['shape'][0]} skills, and [{task.name}.skills] in tasks.toml "
                 f"lists {len(task.skills)} ({', '.join(k.name for k in task.skills)}). Train it again with --skills, "
                 "or run it with the skills it was trained on.")
    from so_101 import skills as skill_rules
    from so_101.grasp import thresholds

    build = skills_build(s, local, None if local else task.policy_for(ns.policy, skills), commit)
    gripper = thresholds.load(s.perception.thresholds).gripper
    require_skills_contract(model(), task, gripper, build)
    return skill_rules.gripper_hash(gripper)


def skills_build(s: settings.Settings, local: Path | None, repo: str | None, revision: str | None) -> dict:
    """The build.json of the skills build a model was trained on, pinned by the dataset revision that
    `train --skills` records in its train_config.json:
    - after a Hub job, the commit of the Hub dataset it uploaded and trained from: build.json at that commit;
    - after `--target local`, `data.build_revision` of the build.json it trained on: the copy `train` kept
      under that name (`data.trained_build_path`), which must still be that file. A later build doesn't
      touch it, so it doesn't retire the model.
    Exits when the model records neither, or when the build can't be read: which skills it was trained on
    would be a guess."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

    from so_101 import data

    model = str(local or repo)
    retrain = "Train it again with `so101 train --skills`"
    try:
        path = local / "train_config.json" if local else Path(hf_hub_download(repo, "train_config.json", revision=revision))
        train = json.loads(path.read_text())
    except (OSError, EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, json.JSONDecodeError) as error:
        sys.exit(f"{model} has no readable train_config.json ({type(error).__name__}), so which skills it was "
                 f"trained on is unknown. {retrain}.")
    dataset = train.get("dataset") if isinstance(train, dict) else None
    dataset = dataset if isinstance(dataset, dict) else {}
    pinned, source, root = dataset.get("revision"), dataset.get("repo_id"), dataset.get("root")
    if isinstance(pinned, str) and pinned.startswith(data.BUILD_REVISION):
        build = data.trained_build_path(s, pinned)
        if not build.is_file() or data.build_revision(build) != pinned:
            why = "changed" if build.is_file() else "missing"
            sys.exit(f"{model} was trained on build {pinned}, whose copy {build} is {why}, so which skills it was "
                     "trained on is unknown. Copy it there from the computer that trained the model, or train again "
                     "with `so101 train --skills --target local`.")
        return json.loads(build.read_text())
    if not (isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{40}", pinned)) or not source or source.startswith("local/"):
        sys.exit(f"{model} recorded no build it was trained on: its train_config.json's dataset.revision is "
                 f"{pinned!r}, not the Hub dataset's commit or the build's sha256 on this Mac that `so101 train` "
                 "records. Its dataset's build.json may describe a later build, so which skills it was trained on "
                 f"can't be checked. {retrain}"
                 f"{' --target local' if root else ''}.")
    try:
        return json.loads(Path(hf_hub_download(source, "build.json", repo_type="dataset", revision=pinned)).read_text())
    except (OSError, EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, json.JSONDecodeError) as error:
        sys.exit(f"{model} was trained on commit {pinned} of {source}, whose build.json can't be read "
                 f"({type(error).__name__}), so which skills it was trained on is unknown. {retrain}, which "
                 "uploads it with the dataset.")


_MISSING = object()  # a field a build.json's skill record leaves out


def require_skills_contract(model: str, task: tasks.Task, gripper, build: dict) -> None:
    """Exit unless the skills dataset's `build` (its build.json) recorded the same `skills.contract` as this
    checkout: every skill as `Skill.record` gives it (name, prompt, from, to and handover) in the same
    order, the same `[gripper]` thresholds, the same cut rules and, when the build took runs (`so101 build
    --with`), the same run labelling and grasp detector. Those decide where the demos were cut, how runs
    were labelled and when a skill switches, so a model trained under others would be told the wrong
    skill at the wrong time. A build.json missing any of them is refused, naming what's missing. One
    refusal names every difference."""
    from so_101 import skills

    inputs = build.get("inputs") or {}
    runs = any(source != "demos" for source in inputs.get("sources") or ())
    want = skills.contract(task.skills, gripper, runs=runs)
    differ, definitions = [], False

    def show(field: str, value) -> str:
        if value is _MISSING:
            return "nothing"
        if value is None:
            return "none"
        return repr(value) if field == "prompt" else str(value)

    trained = inputs.get("skills")
    if not (isinstance(trained, list) and trained and all(isinstance(r, dict) for r in trained)):
        differ.append("its build recorded no skills")
        definitions = True
    elif [r.get("name") for r in trained] != [r["name"] for r in want["skills"]]:
        differ.append("its skills were " + ", ".join(f"{r.get('name')} ({r.get('prompt')!r})" for r in trained)
                      + f", and [{task.name}.skills] in tasks.toml has "
                      + ", ".join(f"{r['name']} ({r['prompt']!r})" for r in want["skills"]))
        definitions = True
    else:
        for here, there in zip(want["skills"], trained, strict=True):
            for field, value in here.items():
                if field != "name" and there.get(field, _MISSING) != value:
                    differ.append(f"{here['name']}: {field} {show(field, value)} here, "
                                  f"{show(field, there.get(field, _MISSING))} in the model's build")
                    definitions = True
    if inputs.get("gripper_sha256") != want["gripper_sha256"]:
        differ.append("its demos were cut with other [gripper] thresholds than the ones this run switches skills with"
                      if inputs.get("gripper_sha256") else
                      "its build recorded no [gripper] thresholds, as builds from before they were recorded")
        definitions = True
    if "cut_rules" not in inputs:
        differ.append("its build recorded no cut_rules (skills.CUT_RULES_VERSION), as builds from before it was "
                      "recorded")
    elif inputs["cut_rules"] != want["cut_rules"]:
        differ.append(f"its demos were cut under cut rules {inputs['cut_rules']}, and this checkout cuts under "
                      f"{want['cut_rules']}")
    if runs and "run_labels" not in inputs:
        differ.append("its build took runs and recorded no run_labels (skills.RUN_LABEL_VERSION), as skills builds "
                      "with runs from before it was recorded")
    elif runs and inputs["run_labels"] != want["run_labels"]:
        differ.append(f"its runs were labelled under run labels {inputs['run_labels']}, and this checkout labels "
                      f"them under {want['run_labels']}")
    if differ:
        back = ", or put the skills or thresholds it was trained on back" if definitions else ""
        sys.exit(f"{model} doesn't match this task's skills: {'; '.join(differ)}. Train it again with "
                 f"`so101 train --skills`{back}.")


def local_checkpoint(text: str, policy: str) -> Path:
    """The model folder (`pretrained_model`, with config.json) under `text`: the folder `train
    --target local` printed, which holds the last checkpoint, or a model folder itself."""
    path = Path(text).expanduser()
    for model in (path / "checkpoints" / "last" / "pretrained_model", path):
        if (model / "config.json").is_file():
            break
    else:
        sys.exit(f"--checkpoint {text}: no model there. Pass the folder `train --target local` printed, once it "
                 "has saved a checkpoint, or a pretrained_model folder.")
    found = json.loads((model / "config.json").read_text()).get("type")
    if found != policy:
        sys.exit(f"--checkpoint {text} holds a model of type {found}, not {policy}: pass --policy {found}.")
    return model.resolve()


def resolve_action_steps(requested: int | None, config: dict) -> int:
    """How many actions to carry out per prediction: at least 1, and no more than the model predicts.

    Without a `requested` number (--action-steps), uses the value the model in `config` was trained with.
    """
    steps = requested
    if steps is None:
        steps = config.get("n_action_steps", config.get("chunk_size"))
    if steps is None:
        sys.exit("The model's config has neither n_action_steps nor chunk_size; pass --action-steps.")
    chunk = config.get("chunk_size", steps)
    if steps < 1:
        sys.exit(f"--action-steps must be at least 1, not {steps}.")
    if steps > chunk:
        print(f"This model predicts {chunk} steps at a time, so using {chunk} action steps instead of {steps}.", flush=True)
    return min(steps, chunk)
