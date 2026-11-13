#!/usr/bin/env python3
"""Fine-tune MME-VLA from checkpoint 79999 with the 16-frame event-keyframe memory (event_keyframes.py).

Arms differ only in the number of event keyframes k (KeyframeConfig.events); initialisation, data order,
flow-matching noise, learning-rate schedule and step count are shared. The official training loop
(robomme_policy_learning/scripts/train.py: main) runs unchanged. This entry point only
  * builds the TrainConfig: `mme_vla_suite` with history perceptual-framesamp-modul.yaml, weights of 79999,
    norm stats of 79999 or of the spec's `assets_dir` (asset id robomme), the dataset of
    build_finetune_dataset.py / build_fruit_dataset.py and wandb off;
  * routes RoboMMEDataset.prepare_frame_sampling through event_keyframes.select on the episode's
    distance.npy: the memory of the sample at frame t is chosen from frames 0..t and packed by the
    official packer (right-padded and masked to the 32 slots of the 512-token budget).
With k = 0 the memory is the official even sampling with 16 frames (the control arm).

Run from the policy repository root (relative config paths), e.g.
    CUDA_VISIBLE_DEVICES=0,1 python finetune_event_keyframes.py --spec <run>/arms/k4.json
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ECM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECM / "src"))
from memory_management import event_keyframes as ek  # noqa: E402

HISTORY_CONFIG = "perceptual-framesamp-modul.yaml"


def install_sampler(cfg: ek.KeyframeConfig, dataset_path, temporal_index="absolute"):
    from mme_vla_suite.training import dataset as _dataset
    from memory_management.policy_adapter import pack_frame_memory
    if temporal_index not in ("absolute", "order"):
        raise ValueError("temporal_index must be absolute or order")
    features = Path(dataset_path) / "features"
    cache = {}

    def prepare_frame_sampling(self, epis_idx, step_idx):
        if epis_idx not in cache:
            cache[epis_idx] = np.load(features / f"episode_{epis_idx}/distance.npy")
        frames, _ = ek.select(cache[epis_idx], step_idx, cfg)
        hc = self.history_config
        if len(frames) * hc.token_per_image * hc.num_views > hc.budget:
            raise ValueError("event-keyframe memory exceeds the token budget")
        history = self._gather_history_feat(frames, epis_idx)
        prefix = (self._gather_history_feat(list(range(len(frames))), epis_idx)
                  if temporal_index == "order" else None)
        return pack_frame_memory(self.mem_buffer, history, frames, hc.budget, hc.token_per_image,
                                 temporal_index=temporal_index, position_source=prefix)

    _dataset.RoboMMEDataset.prepare_frame_sampling = prepare_frame_sampling


def train_config(spec):
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders
    from mme_vla_suite.training import config as _config
    t = spec["train"]
    base = _config.get_config("mme_vla_suite")
    checkpoint = Path(spec["init_checkpoint"])
    assets = spec.get("assets_dir") or str(checkpoint / "assets")
    return dataclasses.replace(
        base,
        exp_name=spec["arm"],
        model=dataclasses.replace(base.model, use_history=True, history_config=HISTORY_CONFIG),
        data=dataclasses.replace(base.data, assets=_config.AssetsConfig(assets_dir=assets, asset_id="robomme")),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint / "params")),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=t["warmup_steps"], peak_lr=t["peak_lr"],
                                                   decay_steps=t["steps"], decay_lr=t["decay_lr"]),
        num_train_steps=t["steps"], save_interval=t.get("save_interval", t["steps"]),
        keep_period=None, log_interval=t["log_interval"],
        batch_size=t["batch_size"], num_workers=t["num_workers"], fsdp_devices=t["fsdp_devices"], seed=t["seed"],
        ema_decay=t["ema_decay"], dataset_path=spec["dataset"], checkpoint_base_dir=spec["checkpoint_base_dir"],
        assets_base_dir=str(Path(spec["checkpoint_base_dir"]) / "assets"), wandb_enabled=False,
        overwrite=False, resume=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, required=True, help="arm specification written by run_finetune_kf.py")
    p.add_argument("--steps", type=int, help="override the step count (smoke runs only)")
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    if a.steps:
        spec["train"]["steps"] = a.steps
    cfg = ek.KeyframeConfig(**spec["sampler"])
    install_sampler(cfg, spec["dataset"], spec.get("temporal_index", "absolute"))
    entry = importlib.util.spec_from_file_location("official_train", Path.cwd() / "scripts/train.py")
    train = importlib.util.module_from_spec(entry)
    entry.loader.exec_module(train)
    config = train_config(spec)
    print(json.dumps({"arm": spec["arm"], "sampler": cfg.to_dict(), "steps": config.num_train_steps,
                      "checkpoint_dir": str(config.checkpoint_dir)}), flush=True)
    train.main(config)


if __name__ == "__main__":
    main()
