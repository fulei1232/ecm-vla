#!/usr/bin/env python3
"""Compute guard v1 from exactly the 400 clean+DART training demonstrations."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("guard statistics already exist; refusing overwrite")
    states, differences, sources = [], [], []
    for root, first, stop in ((ECM / "data/fruit_picking/rm75b_eg2/train", 0, 100),
                             (ECM / "data/fruit_picking_dart/rm75b_eg2/train", 100, 200)):
        for task in ("count", "occluded_ripe"):
            for e in range(first, stop):
                path = root / task / f"ep{e}_manifest.json"
                man = read_json(path)
                assert man["split"] == "train" and man["episode"] == e
                with h5py.File(path.parent / man["h5"], "r") as f:
                    ep = f[f"episode_{e}"]
                    ids = sorted(int(k.split("_")[1]) for k in ep if k.startswith("timestep_"))
                    assert ids == list(range(len(ids)))
                    q = np.stack([ep[f"timestep_{t}/obs/joint_state"][()] for t in ids]).astype(np.float64)
                assert q.ndim == 2 and q.shape[1] == 7 and np.isfinite(q).all()
                states.append(q)
                differences.append(np.abs(np.diff(q, axis=0)))
                sources.append({"manifest": str(path.relative_to(ECM)), "sha256": sha256(path),
                                "h5_sha256_from_manifest": man["h5_sha256"], "states": len(q)})
    q, dq = np.concatenate(states), np.concatenate(differences)
    lower, upper = np.quantile(q, [0.005, 0.995], axis=0)
    p999 = np.quantile(dq, 0.999, axis=0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"version": "v1", "generated_utc": timestamp(), "lower": lower.tolist(), "upper": upper.tolist(),
              "v_max": (1.5 * p999).tolist(), "expert_step_p999": p999.tolist(),
              "state_samples": len(q), "within_episode_state_differences": len(dq), "training_episodes": len(sources),
              "definition": "clean train 0-99 + DART train 100-199, both tasks; measured qpos; absolute adjacent state differences within each episode; numpy linear quantiles; radians per control step; q7 additionally clipped to [-pi,pi] at load",
              "sources": sources, "script_sha256": sha256(Path(__file__))}
    atomic_json(a.out, result)
    print({k: result[k] for k in ("lower", "upper", "v_max", "state_samples", "training_episodes")}, flush=True)


if __name__ == "__main__":
    main()
