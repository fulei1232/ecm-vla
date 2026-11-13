#!/usr/bin/env python3
"""HarvestMem v3 S2 training entry point for SG, K0, and NM."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path
import sys

import numpy as np

ECM = Path(__file__).resolve().parents[2]
POLICY = ECM.parent / "adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning"
CHECKPOINT = ECM.parent / "adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999"
DATASET = ECM / "data/harvestmem_v3_dataset_20261110"
HISTORY_CONFIG = "perceptual-framesamp-modul.yaml"

sys.path.insert(0, str(ECM / "scripts"))
sys.path.insert(0, str(ECM / "src"))
sys.path.insert(0, str(ECM / "scripts" / "harvestmem_v3"))

import finetune_event_keyframes as base  # noqa: E402
from memory_management import event_keyframes as ek  # noqa: E402
from subgoal import compose_prompt  # noqa: E402


def install_uniform_memory(dataset: Path, temporal_index: str) -> None:
    base.install_sampler(ek.KeyframeConfig(frames=16, events=0, radius=0.03, tol=0.005),
                         dataset, temporal_index)


def install_no_memory(temporal_index: str) -> None:
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


def install_grounded_prompt() -> None:
    from mme_vla_suite.training import dataset as _dataset
    original = _dataset.RoboMMEDataset.__getitem__

    def getitem(self, idx):
        data = original(self, idx)
        grounded = data.get("grounded_subgoal")
        if grounded is None:
            raise ValueError("SG sample is missing grounded_subgoal")
        if not isinstance(grounded, str):
            grounded = grounded.item()
        data["prompt"] = compose_prompt(data["prompt"], grounded)
        return data

    _dataset.RoboMMEDataset.__getitem__ = getitem


def make_config(arm: str, run_dir: Path, steps: int, fsdp_devices: int = 4):
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders
    from mme_vla_suite.training import config as _config

    base_config = _config.get_config("mme_vla_suite")
    train = dict(steps=steps, warmup_steps=500, peak_lr=2.5e-5, decay_lr=2.5e-6,
                 batch_size=64, fsdp_devices=fsdp_devices, num_workers=12,
                 seed=42, ema_decay=0.999, log_interval=50)
    if steps <= 3:
        train.update(warmup_steps=min(2, steps), log_interval=1)
    return dataclasses.replace(
        base_config,
        exp_name=arm,
        model=dataclasses.replace(base_config.model, use_history=True, history_config=HISTORY_CONFIG),
        data=dataclasses.replace(base_config.data,
                                 assets=_config.AssetsConfig(
                                     assets_dir=str(DATASET / "assets"), asset_id="robomme")),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(CHECKPOINT / "params")),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=train["warmup_steps"],
                                                   peak_lr=train["peak_lr"],
                                                   decay_steps=steps,
                                                   decay_lr=train["decay_lr"]),
        num_train_steps=steps,
        save_interval=steps,
        keep_period=None,
        log_interval=train["log_interval"],
        batch_size=train["batch_size"],
        num_workers=train["num_workers"],
        fsdp_devices=train["fsdp_devices"],
        seed=train["seed"],
        ema_decay=train["ema_decay"],
        dataset_path=str(DATASET),
        checkpoint_base_dir=str(run_dir / "ckpt"),
        assets_base_dir=str(run_dir / "ckpt" / "assets"),
        wandb_enabled=False,
        overwrite=False,
        resume=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("sg", "k0", "nm"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    args = parser.parse_args()
    if not (DATASET / "meta/build.json").exists():
        raise FileNotFoundError(DATASET / "meta/build.json")

    temporal_index = "order"
    install_uniform_memory(DATASET, temporal_index)
    if args.arm == "sg":
        install_grounded_prompt()
    elif args.arm == "nm":
        install_no_memory(temporal_index)

    entry = importlib.util.spec_from_file_location("official_train", POLICY / "scripts/train.py")
    train = importlib.util.module_from_spec(entry)
    assert entry.loader is not None
    entry.loader.exec_module(train)
    config = make_config(args.arm, args.run_dir.resolve(), args.steps, args.fsdp_devices)
    print({"arm": args.arm, "steps": config.num_train_steps,
           "checkpoint_dir": str(config.checkpoint_dir)}, flush=True)
    train.main(config)


if __name__ == "__main__":
    main()
