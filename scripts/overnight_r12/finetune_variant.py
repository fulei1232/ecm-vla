#!/usr/bin/env python3
"""Diagnostic training variants on top of finetune_event_keyframes.py (exploratory, dev only).

spec["variant"]:
  oracle_count  uniform memory (spec sampler) + prompt suffix " fruits already in the basket: k."
                where k is the demonstration's ground-truth n_in_basket at the sample's frame.
                Information ceiling for counting; never a deployable method.
  no_memory     memory holds only the current frame (no past information), same tensor shapes.

Everything else (init, data order, schedule, steps) is the official loop of finetune_event_keyframes.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ECM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ECM / "scripts"))
sys.path.insert(0, str(ECM / "src"))
import finetune_event_keyframes as base  # noqa: E402
from memory_management import event_keyframes as ek  # noqa: E402


def oracle_suffix(k):
    return f" fruits already in the basket: {int(k)}."


def basket_counts(dataset_path):
    """{global_episode: np.array n_in_basket per timestep} from the demonstrations' ground truth."""
    rows = json.loads((Path(dataset_path) / "meta/episodes.json").read_text())
    out = {}
    for r in rows:
        h5 = Path(r["h5"])
        label = h5.name.rsplit(".", 1)[0]
        gt = h5.parent.parent / "gt" / f"{label}.jsonl"
        counts = [json.loads(x)["n_in_basket"] for x in gt.read_text().splitlines() if x.strip()]
        if len(counts) != r["timesteps"]:
            raise ValueError(f"{gt}: {len(counts)} rows for {r['timesteps']} timesteps")
        out[int(r["global_episode"])] = np.asarray(counts, dtype=np.int64)
    return out


def install_oracle_prompt(dataset_path):
    from mme_vla_suite.training import dataset as _dataset
    counts = basket_counts(dataset_path)
    original = _dataset.RoboMMEDataset.__getitem__

    def getitem(self, idx):
        data = original(self, idx)
        e, t = int(np.asarray(data["epis_idx"]).item()), int(np.asarray(data["step_idx"]).item())
        data["prompt"] = data["prompt"] + oracle_suffix(counts[e][t])
        return data

    _dataset.RoboMMEDataset.__getitem__ = getitem


def install_no_memory(temporal_index):
    from mme_vla_suite.training import dataset as _dataset
    from memory_management.policy_adapter import pack_frame_memory

    def prepare_frame_sampling(self, epis_idx, step_idx):
        frames = [int(step_idx)]
        hc = self.history_config
        history = self._gather_history_feat(frames, epis_idx)
        prefix = self._gather_history_feat([0], epis_idx) if temporal_index == "order" else None
        return pack_frame_memory(self.mem_buffer, history, frames, hc.budget, hc.token_per_image,
                                 temporal_index=temporal_index, position_source=prefix)

    _dataset.RoboMMEDataset.prepare_frame_sampling = prepare_frame_sampling


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--steps", type=int)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    if a.steps:
        spec["train"]["steps"] = a.steps
    variant = spec["variant"]
    temporal = spec.get("temporal_index", "absolute")
    if variant == "oracle_count":
        base.install_sampler(ek.KeyframeConfig(**spec["sampler"]), spec["dataset"], temporal)
        install_oracle_prompt(spec["dataset"])
    elif variant == "no_memory":
        install_no_memory(temporal)
    else:
        raise ValueError(variant)
    entry = importlib.util.spec_from_file_location("official_train", Path.cwd() / "scripts/train.py")
    train = importlib.util.module_from_spec(entry)
    entry.loader.exec_module(train)
    config = base.train_config(spec)
    print(json.dumps({"arm": spec["arm"], "variant": variant, "steps": config.num_train_steps,
                      "checkpoint_dir": str(config.checkpoint_dir)}), flush=True)
    train.main(config)


if __name__ == "__main__":
    main()
