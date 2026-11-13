#!/usr/bin/env python3
"""Seal an isolated 200-step K0 DAgger pilot without touching the formal study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ECM / "runs/fruit_dagger_pilot_20261030")
    p.add_argument("--dataset", type=Path, default=ECM / "data/finetune_kf/fruit_dagger_pilot_mix_20261030")
    p.add_argument("--init", type=Path, default=ECM / "runs/fruit_dart_kf_20261027/ckpt/mme_vla_suite/k0/11999")
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--arm", default="k0_dagger_pilot")
    a = p.parse_args()
    out, dataset, init = (x.resolve() for x in (a.out, a.dataset, a.init))
    if (out / "plan.json").exists():
        p.error(f"pilot plan already exists: {out}")
    mix = read_json(dataset / "meta/build.json")
    if not mix["passed"] or mix["execution_samples"] < 12102:
        p.error("unexpected or unverified pilot mixture")
    if a.steps < 1 or not a.arm or "/" in a.arm:
        p.error("invalid training steps or arm name")
    if not (init / "params/manifest.ocdbt").is_file():
        p.error("K0 initial checkpoint is incomplete")
    assets = ECM / "data/finetune_kf/fruit_dart_rm75b_train/assets"
    if sha256(init / "assets/robomme/norm_stats.json") != sha256(assets / "robomme/norm_stats.json"):
        p.error("initial K0 and base dataset normalization statistics differ")
    if len(set(a.gpus)) != len(a.gpus) or any(g < 0 or g > 7 for g in a.gpus):
        p.error("gpus must be unique indexes in 0-7")
    train = {"steps": a.steps, "warmup_steps": min(500, max(10, a.steps // 20)),
             "peak_lr": 5e-6, "decay_lr": 1e-6,
             "batch_size": 64, "fsdp_devices": len(a.gpus), "num_workers": 8, "seed": 42,
             "ema_decay": 0.999, "log_interval": 10 if a.steps <= 200 else 50,
             "save_interval": 2000 if a.steps >= 2000 else a.steps}
    spec = {"arm": a.arm, "sampler": {"frames": 16, "radius": 0.03,
                                                   "tol": 0.005, "events": 0},
            "dataset": str(dataset), "init_checkpoint": str(init),
            "checkpoint_base_dir": str(out / "ckpt"), "assets_dir": str(assets),
            "train": train, "gpus": a.gpus}
    (out / "arms").mkdir(parents=True, exist_ok=True)
    spec_path = out / f"arms/{a.arm}.json"
    atomic_json(spec_path, spec)
    plan = {"formal": False, "purpose": "short DAgger correction signal test", "created_utc": timestamp(),
            "spec_path": str(spec_path), "spec_sha256": sha256(spec_path),
            "dataset_build_sha256": sha256(dataset / "meta/build.json"),
            "dataset_stats_sha256": sha256(dataset / "meta/stats.json"),
            "init_params_manifest_sha256": sha256(init / "params/manifest.ocdbt"),
            "init_norm_stats_sha256": sha256(init / "assets/robomme/norm_stats.json"),
            "train_script_sha256": sha256(ECM / "scripts/finetune_event_keyframes.py"),
            "mixture_script_sha256": sha256(ECM / "scripts/build_fruit_dagger_pilot_mix.py"),
            "output_checkpoint": str(out / f"ckpt/mme_vla_suite/{a.arm}/{a.steps - 1}")}
    atomic_json(out / "plan.json", plan)
    print(json.dumps({"out": str(out), "dataset_samples": mix["execution_samples"],
                      "steps": train["steps"], "init": str(init), "gpus": spec["gpus"]}, indent=2))


if __name__ == "__main__":
    main()
