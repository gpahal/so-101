"""`so101 sim`: build a task's simulated twin from its demos, check it, and try a policy in it."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from so_101 import data, settings
from so_101.check import preflight
from so_101.commands.policy import local_checkpoint, resolve_action_steps
from so_101.console import run_lerobot
from so_101.tasks import get_task

SCENE_DIR_ENV = "SO101_SIM_SCENE"  # `so_101.sim.env.SCENE_DIR_ENV`, which imports MuJoCo
CAMERAS = ("top", "wrist")
ACTIONS = ("build", "replay", "score", "nudge", "physics", "check", "eval")


def record_nudges(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    """Teleoperated pushes for fitting the sim's physics. They stay out of the sessions a policy trains on."""
    from so_101.recording import record_args

    task = get_task(ns.task)
    episodes = 3 if ns.episodes is None else ns.episodes
    if episodes < 1:
        sys.exit(f"--episodes is {episodes}: name at least 1 push to record.")
    cameras, _ = preflight(
        s, "recording", cameras=[s.camera(name) for name in task.cameras], task=task,
        loop_fps=task.fps, keyboard=True, video=True,
    )  # fmt: skip
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = data.task_dir(s, task) / "nudges" / stamp / "dataset"
    root.parent.mkdir(parents=True)
    print(
        "In each recording: start at the demos' start pose with the block on the mat, in the top camera's view and "
        "clear of the arm. Push it a few centimetres with the jaws, then go back to the start pose and hold still "
        "until the end. These never go into training.\n"
    )
    args = record_args(
        s, task, cameras, root=root, repo_id=f"local/so101_{task.name}_nudges_{stamp}", episodes=episodes, display=False,
    )  # fmt: skip
    code = run_lerobot("so101-record", [*args, *extra])  # lerobot-record, connecting the follower where it is
    if not any((root / "data").rglob("*.parquet")):
        # LeRobot makes the folder when it starts, so a recording that saved nothing would otherwise leave
        # an empty one for `sim physics` to trip over.
        import shutil

        shutil.rmtree(root.parent)
        print("\nNo pushes were saved, so nothing was kept.")
    else:
        print(f"\nSaved in {root}. Fit them with: uv run so101 sim physics {task.name}")
    if code != 0:
        sys.exit(code)


def cmd_sim(s: settings.Settings, ns: argparse.Namespace, extra: list[str]) -> None:
    if ns.action == "nudge":  # on the arm's computer, which needs neither MuJoCo nor the demos
        record_nudges(s, ns, extra)
        return
    if extra and ns.action != "eval":
        sys.exit(f"so101 sim {ns.action} doesn't take {' '.join(extra)}.")
    try:
        from so_101.sim import build as sim_build
    except ImportError as error:
        sys.exit(f"The simulator needs MuJoCo ({error}). Install it with: uv sync --extra sim")
    from so_101.sim import source

    task = get_task(ns.task)
    if set(task.cameras) != set(CAMERAS):
        sys.exit(f"The sim draws a top and a wrist camera, and {task.name} records {', '.join(task.cameras)}.")
    if ns.action in ("check", "eval") and ns.policy != "act":
        sys.exit(f"sim {ns.action} runs the model on this Mac, which only ACT fits: {ns.policy} runs on a cloud GPU.")
    demos, scene = data.train_root(s, task), data.sim_dir(s, task)
    if ns.action != "build" and not (scene / "scene.json").is_file():
        sys.exit(f"No scene yet. Run `uv run so101 sim build {task.name}` first.")
    if ns.action not in ("eval", "physics"):  # those only need the scene
        source.ensure_demos(demos, scene)
    limit = s.teleop.max_relative_target if ns.limit is None else ns.limit
    if ns.action == "build":
        try:
            sim_build.build(
                demos, scene, task.prompt, s.sim, s.teleop.max_relative_target, task.dataset, s.follower.calibration_file,
                refit_cameras=ns.refit_cameras, refit_motors=ns.refit_motors, report=lambda line: print(line, flush=True),
            )  # fmt: skip
        except ValueError as error:
            sys.exit(str(error))
        print(f"\nScene written to {scene}. Check it with `uv run so101 sim score {task.name}`, then commit it.")
    elif ns.action == "replay":
        video = s.outputs_dir / "sim" / f"replay_{task.name}.mp4"
        episodes = list(range(0, ns.episodes)) if ns.episodes else None
        result = sim_build.replay(demos, scene, episodes, limit, video)
        count = len(result["results"])
        print(f"Replaying real commands: {result['lifted']}/{count} lifted the block, {result['in_jar']}/{count} put it in the jar.")
        print("Joint error per joint: " + ", ".join(f"{e:.1f}" for e in result["error"]))
        missed = [str(r["episode"]) for r in result["results"] if not r["in_jar"]]
        if missed:
            print(f"Demos whose replay missed the jar: {', '.join(missed)}")
        print(f"Real and sim side by side: {video}")
    elif ns.action == "physics":
        from so_101.sim import physics

        nudges = data.task_dir(s, task) / "nudges"
        source.ensure_nudges(nudges, json.loads((scene / "scene.json").read_text())["table"].get("friction_from"))
        roots = physics.nudge_roots(data.task_dir(s, task))
        if not roots:
            sys.exit(f"No nudge recordings yet. Record them with: uv run so101 sim nudge {task.name}")
        pushes, skipped = physics.read_pushes(roots, scene)
        for reason in skipped:
            print(f"Left out {reason}.")
        if not pushes:
            sys.exit("No usable pushes.")
        print(f"Keeping the recordings with the sim: uploading them to the private Hub dataset {task.dataset}_nudges...", flush=True)
        came_from = source.push_nudges(nudges, f"{task.dataset}_nudges")
        result = physics.fit_friction(scene, pushes, s.teleop.max_relative_target, came_from)
        for friction, error in result["by_friction"].items():
            print(f"friction {friction:4.2f}: block ends {error:3.0f} mm from where it really stopped")
        print(f"Using {result['friction']:g}, from {result['pushes']} pushes that really moved it {result['moved_mm']:.0f} mm on average.")
        print(f"Check the demos still work with: uv run so101 sim replay {task.name}")
    elif ns.action == "score":
        from so_101.sim import score

        stamp = time.strftime("%Y%m%d-%H%M%S")
        sheet = s.outputs_dir / "sim" / f"score_{task.name}_{stamp}.jpg"
        print(f"Comparing the sim with held-back real frames ({s.run.device})...", flush=True)
        result = score.images_and_geometry(demos, scene, s.run.device, sheet)
        result["physics"] = score.physics(demos, scene, s.teleop.max_relative_target)
        score.show(result)
        report = sheet.with_suffix(".json")
        report.write_text(json.dumps(result, indent=1))
        print(f"\nReal and sim side by side: {sheet}\nNumbers: {report}")
    elif ns.action == "check":
        policy = _policy(ns, task)
        print(f"Comparing the model's predictions on real and sim images ({s.run.device})...", flush=True)
        result = sim_build.check(demos, scene, policy, s.run.device)
        for label, key in (("real images", "real"), ("sim images", "sim"), ("sim top, real wrist", "sim_top"), ("real top, sim wrist", "sim_wrist")):
            print(f"{label:22s} error from the demo: {result[key].mean():.2f}  shoulder pan {result[key][0]:.2f}")
        print(
            f"Extra error on sim images: {result['sim'].mean() - result['real'].mean():.2f} degrees. Compare this between sim "
            "versions, but don't tune the sim to it: that fits the sim to one model."
        )
    elif ns.action == "eval":
        output = s.outputs_dir / "sim" / f"eval_{ns.policy}_{task.name}_{time.strftime('%Y%m%d-%H%M%S')}"
        policy = _policy(ns, task)
        action_steps = resolve_action_steps(ns.action_steps, json.loads((policy / "config.json").read_text()))
        os.environ[SCENE_DIR_ENV] = str(scene.resolve())
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        length = int((s.run.duration_s or 60) * task.fps)
        args = [
            f"--policy.path={policy}",
            f"--policy.device={s.run.device}",
            f"--policy.n_action_steps={action_steps}",
            "--env.type=envhub",
            f"--env.hub_path={Path(__file__).parent.parent / 'sim'}",
            "--env.trust_remote_code=true",
            f"--env.make_env_kwargs={json.dumps({'max_relative_target': limit, 'episode_length': length})}",
            f"--eval.n_episodes={ns.episodes or 10}",
            "--eval.batch_size=1",
            f"--output_dir={output}",
        ]
        print(f"Each attempt gets {length / task.fps:g} s ([run] duration_s), and ends early once the block has lain in the jar for 0.5 s.", flush=True)
        code = run_lerobot("lerobot-eval", [*args, *extra])
        print(f"\nResults and videos: {output}")
        sys.exit(code)


def _policy(ns: argparse.Namespace, task) -> Path:
    """The model folder to run: a local --checkpoint, or the task's ACT policy from the Hub."""
    if ns.checkpoint:
        if ns.revision:
            sys.exit("--checkpoint runs a model saved on this Mac, so it doesn't take --revision.")
        return local_checkpoint(ns.checkpoint, ns.policy)
    return task.download_policy(ns.revision)
