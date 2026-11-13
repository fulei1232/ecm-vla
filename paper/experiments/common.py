"""Shared loaders for the paper experiments: episodes of the fruit-picking datasets with their ground-truth
sidecar, layout and the event signal used for training (memory_management.fruit_distance)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import sys

import numpy as np

ECM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ECM))
sys.path.insert(0, str(ECM / "src"))

from fruit_picking.layout import sample_layout  # noqa: E402
from memory_management import fruit_distance  # noqa: E402

OUT = ECM / "paper/results"
TASKS = {"count": "FruitCount", "occluded_ripe": "FruitOccludedRipe"}
DATA = ECM / "data"
# name -> (root, split, episodes)
SETS = {
    "train_clean": (DATA / "fruit_picking/rm75b_eg2", "train", range(0, 100)),
    "train_dart": (DATA / "fruit_picking_dart/rm75b_eg2", "train", range(100, 200)),
    "dev": (DATA / "fruit_picking_dev/rm75b_eg2", "train", range(200, 220)),
    "val_a": (DATA / "fruit_picking/rm75b_eg2", "val", range(0, 50)),
    "val_b": (DATA / "fruit_picking/rm75b_eg2", "val", range(50, 100)),
}


@dataclass
class Episode:
    set_name: str
    task: str
    episode: int
    directory: Path

    @cached_property
    def manifest(self):
        return json.loads((self.directory / f"ep{self.episode}_manifest.json").read_text())

    @property
    def seed(self):
        return int(self.manifest["seed"])

    @cached_property
    def rows(self):
        with open(self.directory / self.manifest["gt"]) as f:
            return [json.loads(line) for line in f]

    @cached_property
    def layout(self):
        return sample_layout(self.task, self.seed)

    @cached_property
    def ripe(self):
        return [bool(f["ripe"]) for f in self.layout.fruits]

    @cached_property
    def distance(self):
        """The training event signal d_t (m) for every recorded frame."""
        return np.array([fruit_distance.from_gt_row(r, self.ripe) for r in self.rows])

    @cached_property
    def detach_times(self):
        """{fruit index: first timestep at which it is off the tree}."""
        out = {}
        for r in self.rows:
            for i, f in enumerate(r["fruits"]):
                if f["attached"] and i not in out:
                    out[i] = int(r["timestep"])
        return out

    @property
    def h5(self):
        return self.directory / self.manifest["h5"]


SKIPPED = []


def episodes(set_name, tasks=tuple(TASKS)):
    """Episodes with a readable ground-truth sidecar; others are listed in SKIPPED."""
    root, split, eps = SETS[set_name]
    for task in tasks:
        for e in eps:
            directory = root / split / task
            if not (directory / f"ep{e}_manifest.json").exists():
                continue
            ep = Episode(set_name, task, e, directory)
            with open(directory / ep.manifest["gt"], "rb") as f:
                if f.read(1) != b"{":
                    SKIPPED.append(f"{set_name}/{task}/ep{e}: unreadable gt sidecar")
                    continue
            yield ep


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "min": float(x.min()), "max": float(x.max())}


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    return path
