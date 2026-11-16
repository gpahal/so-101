"""The benchmark's samples: `labels.csv` beside this file, and the image folder built from it.

`labels.csv` has one row per sample: which recorded attempt (`src`, a folder under `outputs/rollouts/`
or a `so101 serve` run's folder under `data/<task>/runs/`, and its episode `ep`), which frame (`t`, from the episode's start at 30 fps), the moment (a key of
`prompt.MOMENTS`), the label (`ok` or `wrong`), the category (a key of `prompt.CATEGORIES`, `none` for
ok), the gripper position at that frame and a short note. How the rows were chosen and checked is in the
file's header comment.

`build` decodes each sample's top and wrist frames from the recorded videos into
`<out>/images/<id>_top.jpg` and `_wrist.jpg` and writes `<out>/samples.jsonl`. The recordings are
git-ignored and stay on the machine that recorded them, so only `labels.csv` is in the repo. The rollouts
its rows name were deleted for a fresh start: build a new labels file from `serve` runs with
`so101-failbench candidates`, label it, and pass it to `build --labels`.
"""

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from so_101.failbench import prompt

LABELS = Path(__file__).with_name("labels.csv")
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

    @property
    def policy(self) -> str:
        return self.src.split("_block")[0]


def read_labels(path: Path = LABELS) -> list[Sample]:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    samples = []
    for row in csv.DictReader(lines):
        s = Sample(id=row["id"], src=row["src"], ep=int(row["ep"]), t=int(row["t"]), moment=row["moment"],
                   label=row["label"], category=row["category"], gripper=float(row["gripper"] or "nan"),
                   note=row.get("note", ""))
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


def recording(src: str, roots: list[Path]) -> Path | None:
    """The dataset folder of the recording `src`, from the first of `roots` that has it."""
    return next((root / src / "dataset" for root in roots if (root / src / "dataset").exists()), None)


def build(out: Path, roots: list[Path], labels: Path = LABELS) -> int:
    """Decode every sample's frames into `out`, finding each recording under one of `roots`. Returns how
    many samples were written."""
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    samples = read_labels(labels)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, LeRobotDataset] = {}
    for s in samples:
        paths = {cam: images / f"{s.id}_{cam}.jpg" for cam in CAMERAS}
        if all(p.exists() for p in paths.values()):
            continue
        if s.src not in datasets:
            root = recording(s.src, roots)
            if root is None:
                raise FileNotFoundError(f"Sample {s.id} needs the recording {s.src}, which isn't under "
                                        f"{' or '.join(map(str, roots))} on this machine.")
            datasets[s.src] = LeRobotDataset(f"local/{s.src}", root=root)
        ds = datasets[s.src]
        item = ds[int(ds.meta.episodes[s.ep]["dataset_from_index"]) + s.t]
        for cam, p in paths.items():
            frame = (item[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).save(p, quality=90)
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
