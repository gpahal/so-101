"""The benchmark's samples: a labels file, and the image folder built from it.

A labels file has one row per sample: which recorded attempt (`src`, the folder name of a `so101 serve` or
`so101 run` run under `data/<task>/runs/`, and its episode `ep`), which frame (`t`, counted from the
episode's start), the moment (a key of `prompt.MOMENTS`), the label (`ok` or `wrong`), the category (a key
of `prompt.CATEGORIES`, `none` for ok), the gripper position at that frame and a note saying how the label
was reached. Lines starting with `#` are comments, and a row with an empty label is one nobody has
labelled yet, which is left out. `so101-failbench candidates` writes such a file from the runs, with the
labels empty.

`build` decodes each sample's top and wrist frames from its run's videos into `<out>/images/<id>_top.jpg`
and `_wrist.jpg` and writes `<out>/samples.jsonl`. It reads each run through `data.Run`, so a run whose
metadata a kill tore reads from the copy the recorder kept, and decodes through
`dataset_io.episode_frames`. The runs are git-ignored and stay on the machine that recorded them, so a
dataset can only be built there.
"""

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from so_101.vision import prompt

CAMERAS = ("top", "wrist")


@dataclass(frozen=True)
class Sample:
    id: str
    src: str
    ep: int
    t: int
    moment: str
    label: str  # "ok" or "wrong"
    category: str
    gripper: float
    note: str

    @property
    def wrong(self) -> bool:
        return self.label == "wrong"


class MissingRecordings(FileNotFoundError):
    """Samples whose run, episode or frame isn't on this machine. The message names them all, on one line."""


def _rows(path: Path) -> list[dict]:
    lines = [line for line in Path(path).read_text().splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def unlabelled(path: Path) -> int:
    """How many rows of the labels file have no label yet."""
    return sum(not (row.get("label") or "").strip() for row in _rows(path))


def read_labels(path: Path) -> list[Sample]:
    """The labelled samples of the labels file at `path`, leaving out rows with an empty label."""
    samples = []
    for row in _rows(path):
        if not (row.get("label") or "").strip():
            continue
        s = Sample(id=row["id"], src=row["src"], ep=int(row["ep"]), t=int(row["t"]), moment=row["moment"],
                   label=row["label"].strip(), category=row["category"].strip(),
                   gripper=float(row["gripper"] or "nan"), note=row.get("note") or "")
        if s.moment not in prompt.MOMENTS:
            raise ValueError(f"{path}: sample {s.id} has an unknown moment {s.moment!r}")
        if s.label not in ("ok", "wrong") or s.category not in prompt.CATEGORIES:
            raise ValueError(f"{path}: sample {s.id} has label {s.label!r} and category {s.category!r}")
        if (s.label == "ok") != (s.category == "none"):
            raise ValueError(f"{path}: sample {s.id} is {s.label} but its category is {s.category}")
        samples.append(s)
    if len({s.id for s in samples}) != len(samples):
        raise ValueError(f"{path}: sample ids aren't unique")
    return samples


def _runs(samples: list[Sample], roots: list[Path]) -> dict:
    """Each sample's run's metadata, by `src`. Raises `MissingRecordings` naming every sample it can't
    build: its run isn't under any of `roots` or saved no attempt, or its episode or frame isn't in it."""
    from so_101.data import Run

    metas, missing = {}, []
    for src in dict.fromkeys(s.src for s in samples):
        folder = next((root / src for root in roots if (root / src / "dataset").is_dir()), None)
        meta = Run.at(folder).meta if folder is not None else None
        if meta is None:
            missing.append(src)
        else:
            metas[src] = meta
    gone = [s.id for s in samples if s.src in metas and not (
        s.ep < metas[s.src].total_episodes and 0 <= s.t < int(metas[s.src].episodes[s.ep]["length"]))]
    if missing or gone:
        needing = sum(s.src in missing for s in samples) + len(gone)
        parts = [f"the runs {', '.join(missing)} aren't under {' or '.join(map(str, roots))}"] if missing else []
        parts += [f"the episodes or frames of {', '.join(gone)} aren't in their run"] if gone else []
        raise MissingRecordings(f"{needing} of the {len(samples)} labelled samples can't be built on this machine: "
                                + "; ".join(parts) + ". Build where they were recorded, or delete their rows.")
    return metas


def build(out: Path, roots: list[Path], labels: Path) -> int:
    """Decode every labelled sample's frames into `out`, finding each run under one of `roots`. Returns how
    many samples were written. Raises `MissingRecordings`, before decoding anything, when any can't be."""
    from PIL import Image

    from so_101 import dataset_io

    samples = read_labels(labels)
    metas = _runs(samples, [Path(r) for r in roots])
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    for s in samples:
        for cam in CAMERAS:
            path = image_path(out, s, cam)
            if path.exists():
                continue
            _, frame = next(dataset_io.episode_frames(metas[s.src], s.ep, f"observation.images.{cam}", s.t, s.t + 1))
            Image.fromarray(frame).save(path, quality=90)
    with open(out / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
    return len(samples)


def load(root: Path) -> list[Sample]:
    path = root / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No benchmark dataset at {root}: run `so101-failbench build` first.")
    samples = [Sample(**json.loads(line)) for line in path.read_text().splitlines() if line]
    for s in samples:
        for cam in CAMERAS:
            if not image_path(root, s, cam).exists():
                raise FileNotFoundError(f"{image_path(root, s, cam)} is missing: run `so101-failbench build` again.")
    return samples


def image_path(root: Path, s: Sample, camera: str) -> Path:
    return root / "images" / f"{s.id}_{camera}.jpg"


def subset(samples: list[Sample], limit: int | None, seed: int = 0) -> list[Sample]:
    """A deterministic subset of about `limit` samples with the same mix of label and moment as the whole."""
    if not limit or limit >= len(samples):
        return list(samples)
    groups: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        groups.setdefault((s.label, s.moment), []).append(s)
    rng = random.Random(seed)
    chosen = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda s: s.id)
        rng.shuffle(group)
        chosen += group[:max(1, round(limit * len(group) / len(samples)))]
    return sorted(chosen, key=lambda s: s.id)
