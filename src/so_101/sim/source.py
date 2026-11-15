"""Where a sim's inputs come from, so it can be rebuilt anywhere.

A scene records the Hub revision of the demos it was built from and a fingerprint of their files. Commands
that need the demos download that revision when there are none locally, and say when local demos differ.
Nudge recordings for the physics fit go to their own private Hub dataset the same way.
"""

import hashlib
import json
import sys
from pathlib import Path

PARTS = ("data", "meta", "videos")  # what makes up a LeRobot dataset; not the Hub's README or cache


def _files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).parts[0] in PARTS}


def fingerprint(root: Path) -> str:
    """One hash of every file in a dataset and where it sits."""
    digest = hashlib.sha256()
    for name, path in _files(root).items():
        digest.update(name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def hub_revision(root: Path, repo: str) -> str | None:
    """The Hub commit holding exactly these files, or None if the Hub has something else or can't be reached."""
    from huggingface_hub import HfApi

    try:
        api = HfApi()
        revision = api.dataset_info(repo).sha
        remote = {
            f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True, expand=True)
            if hasattr(f, "size") and f.path.split("/")[0] in PARTS
        }  # fmt: skip
    except Exception:
        return None
    local = _files(root)
    if set(local) != set(remote):
        return None
    for name, path in local.items():
        data = path.read_bytes()
        entry = remote[name]
        if entry.lfs is not None:
            if hashlib.sha256(data).hexdigest() != entry.lfs.sha256:
                return None
        elif hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != entry.blob_id:
            return None
    return revision


def describe(root: Path, repo: str) -> dict:
    """What to record in a scene about the demos it's built from."""
    revision = hub_revision(root, repo)
    if revision is None:
        print(
            f"WARNING: these demos aren't on the Hub as {repo}, so only this computer can rebuild the sim from them. "
            "`so101 train` with a cloud target uploads them.", file=sys.stderr,
        )  # fmt: skip
    return {"repo": repo, "revision": revision, "fingerprint": fingerprint(root)}


def ensure_demos(root: Path, scene_dir: Path) -> None:
    """Make sure the demos a scene was built from are here: download them if missing, warn if they differ."""
    scene_file = scene_dir / "scene.json"
    recorded = json.loads(scene_file.read_text()).get("demos", {}) if scene_file.is_file() else {}
    if not (root / "meta" / "info.json").is_file():
        if not recorded.get("revision"):
            sys.exit(f"No demos at {root}. Build them with `uv run so101 build`, or download them from the Hub.")
        from huggingface_hub import snapshot_download

        print(f"Downloading the demos the sim was built from: {recorded['repo']} at {recorded['revision'][:8]}...", flush=True)
        snapshot_download(recorded["repo"], repo_type="dataset", revision=recorded["revision"], local_dir=root)
    if recorded.get("fingerprint") and fingerprint(root) != recorded["fingerprint"]:
        print(
            f"WARNING: the demos in {root} aren't the ones the sim in {scene_dir} was built from. "
            "Run `uv run so101 sim build` to rebuild it from these.", file=sys.stderr,
        )  # fmt: skip


def push_nudges(nudges_dir: Path, repo: str) -> dict:
    """Upload nudge recordings to a private Hub dataset; returns where they are."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=nudges_dir, allow_patterns=["*/dataset/**"], commit_message="so101 sim physics")
    return {"repo": repo, "revision": commit.oid, "sessions": sorted(p.parent.name for p in nudges_dir.glob("*/dataset"))}


def ensure_nudges(nudges_dir: Path, recorded: dict | None) -> None:
    """Download the nudge recordings a friction fit came from, if none are here."""
    if any(nudges_dir.glob("*/dataset/meta/info.json")) or not recorded or not recorded.get("revision"):
        return
    from huggingface_hub import snapshot_download

    print(f"Downloading the nudge recordings: {recorded['repo']} at {recorded['revision'][:8]}...", flush=True)
    snapshot_download(recorded["repo"], repo_type="dataset", revision=recorded["revision"], local_dir=nudges_dir)
