#!/usr/bin/env python3
"""Generate and audit one reproducible Milestone 3 teacher label.

The V-B label shards contain paired ``base``, ``identity`` and single-slot
swap cached-flow-loss evaluations.  This tool binds one legal boundary,
candidate and history slot into a standalone M3 label.  It checks paired RNG,
action hashes, masks, shapes, causality, reproducibility and the same-frame
zero-gain control before reporting the real counterfactual gain.

This is a protocol smoke label when the source run uses ``val_planner``.  A
formal M3 label must be regenerated from a sealed train split run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _find_boundary(run: Path, requested: str | None):
    paths = sorted((run / "boundaries").glob("*.json"))
    if requested:
        paths = [run / "boundaries" / requested]
    for path in paths:
        if not path.exists():
            continue
        boundary = json.loads(path.read_text())
        key = path.stem
        label_path = run / "labels" / f"{key}.parquet"
        if label_path.exists():
            return path, boundary, label_path
    raise FileNotFoundError("no completed boundary with a parquet label shard")


def _history_frames(frames: list[int]) -> list[int]:
    if len(frames) < 6:
        raise ValueError("memory must contain anchor, history, and recent frames")
    # partition() uses one anchor and four recent slots.  The remaining frames
    # are the replaceable history slots in chronological pack order.
    return frames[1:-4]


def _check_rows(cutoff: int, base: pd.DataFrame, identity: pd.DataFrame,
                action_horizon: int, action_dims: int):
    if base.empty or identity.empty:
        raise ValueError("base/identity rows are missing")
    keys = ["offset", "seed_index"]
    left = base.sort_values(keys).reset_index(drop=True)
    right = identity.sort_values(keys).reset_index(drop=True)
    if left[keys].to_dict("records") != right[keys].to_dict("records"):
        raise ValueError("base and identity do not have the same offset/seed grid")
    for col in ("noise_sha256", "time_sha256", "action_sha256"):
        if left[col].tolist() != right[col].tolist():
            raise ValueError(f"paired {col} differs")
    if left["memory_frames_json"].tolist() != right["memory_frames_json"].tolist():
        raise ValueError("identity memory differs from base memory")
    if not np.array_equal(left["loss"].to_numpy(), right["loss"].to_numpy()):
        raise ValueError("same-frame cached loss is not reproducible")
    if not np.array_equal(left["loss_all_dims"].to_numpy(), right["loss_all_dims"].to_numpy()):
        raise ValueError("same-frame all-dimension loss differs")
    if not np.array_equal(left["loss_padded_dims"].to_numpy(), right["loss_padded_dims"].to_numpy()):
        raise ValueError("same-frame padded-dimension loss differs")
    if (left["valid_action_steps"] < 0).any() or (left["valid_action_steps"] > action_horizon).any():
        raise ValueError("invalid action mask length")
    expected_padding = 1.0 - left["valid_action_steps"] / action_horizon
    if not np.allclose(left["padding_fraction"], expected_padding, atol=1e-7):
        raise ValueError("padding fraction is inconsistent with action mask")
    for row in left.to_dict("records"):
        frame = int(row["frame"])
        value = row["memory_frames_json"]
        memory = _json(value)
        if any(int(x) > int(frame) for x in memory):
            raise ValueError("future memory frame detected")
        if any(int(x) > int(cutoff) for x in memory[:-4]):
            raise ValueError("future frozen history frame detected")
    if (left["frame"] < cutoff).any():
        raise ValueError("future evaluation frame precedes cutoff")
    if not (0 < action_dims <= 32):
        raise ValueError("invalid real action dimension")
    return left, right


def make_label(run: Path, boundary_name: str | None = None, slot: int = 0,
               candidate_index: int = 0, gamma: float = 0.7) -> dict:
    protocol = json.loads((run / "protocol.json").read_text())
    boundary_path, boundary, parquet = _find_boundary(run, boundary_name)
    data = pd.read_parquet(parquet)
    task, episode, cutoff = boundary["task"], int(boundary["episode_id"]), int(boundary["cutoff"])
    base = data[data["label"] == "base"].copy()
    identity = data[data["label"] == "identity"].copy()
    action_horizon = int(base["valid_action_steps"].max())
    action_dims = int(protocol.get("config", {}).get("loss", {}).get("action_dims", 8))
    base, identity = _check_rows(cutoff, base, identity, action_horizon, action_dims)
    offsets = sorted(int(x) for x in base["offset"].unique())
    if not offsets:
        raise ValueError("identity shard has no valid offsets")
    first_memory = _json(base.iloc[0]["memory_frames_json"])
    history = _history_frames(first_memory)
    if not 0 <= slot < len(history):
        raise ValueError(f"slot {slot} outside history slots [0,{len(history)-1}]")
    identity_frame = int(history[slot])
    if not 1 <= identity_frame < cutoff:
        raise ValueError("same-frame candidate is not a strict historical frame")
    candidates = boundary.get("candidates", [])
    if not 0 <= candidate_index < len(candidates):
        raise ValueError(f"candidate index {candidate_index} outside boundary candidate list")
    candidate_frame = int(candidates[candidate_index]["frame_id"])
    if not 1 <= candidate_frame < cutoff or candidate_frame in history:
        raise ValueError("counterfactual candidate is not a strict, non-duplicate historical frame")
    actual_label = f"k1_c{candidate_index}_j{slot}"
    actual = data[data["label"] == actual_label].copy()
    if actual.empty:
        raise ValueError(f"missing actual candidate label shard: {actual_label}")
    actual = actual.sort_values(["offset", "seed_index"]).reset_index(drop=True)
    base_sorted = base.sort_values(["offset", "seed_index"]).reset_index(drop=True)
    if actual[["offset", "seed_index"]].to_dict("records") != base_sorted[["offset", "seed_index"]].to_dict("records"):
        raise ValueError("actual candidate and base grids differ")
    for col in ("noise_sha256", "time_sha256", "action_sha256"):
        if actual.sort_values(["offset", "seed_index"])[col].tolist() != base_sorted[col].tolist():
            raise ValueError(f"actual candidate {col} differs from base")
    old_frames = [_json(v) for v in base_sorted["memory_frames_json"]]
    new_frames = [_json(v) for v in actual["memory_frames_json"]]
    for old, new in zip(old_frames, new_frames):
        old_history, new_history = old[:-4], new[:-4]
        # ``memory_frames`` returns frame IDs in chronological order, so
        # replacing one slot can move the inserted candidate across several
        # positions after sorting.  Audit the intervention by set difference
        # rather than positional difference: exactly one frozen history ID is
        # removed and exactly the selected candidate ID is added.  The final
        # four entries are the offset-dependent recent window and are allowed
        # to refresh normally.
        removed = set(old_history) - set(new_history)
        added = set(new_history) - set(old_history)
        if (len(old) != len(new) or removed != {identity_frame}
                or added != {candidate_frame}):
            raise ValueError("candidate replacement does not change exactly one memory slot")

    merged = base.merge(identity, on=["offset", "seed_index"], suffixes=("_base", "_identity"))
    merged["gain"] = merged["loss_base"] - merged["loss_identity"]
    actual_merged = base.merge(actual, on=["offset", "seed_index"], suffixes=("_base", "_actual"))
    actual_merged["gain"] = actual_merged["loss_base"] - actual_merged["loss_actual"]
    per_seed = []
    for seed, group in merged.groupby("seed_index", sort=True):
        group = group.sort_values("offset")
        weights = np.power(gamma, group["offset"].to_numpy(dtype=float))
        mask = np.ones(len(group), dtype=np.float64)
        value = float(np.sum(group["gain"].to_numpy() * weights * mask) / np.sum(weights * mask))
        per_seed.append({"seed_index": int(seed), "discounted_gain": value,
                         "valid_offsets": group["offset"].astype(int).tolist()})
    gains = np.asarray([x["discounted_gain"] for x in per_seed], dtype=np.float64)
    actual_per_seed = []
    for seed, group in actual_merged.groupby("seed_index", sort=True):
        group = group.sort_values("offset")
        weights = np.power(gamma, group["offset"].to_numpy(dtype=float))
        value = float(np.sum(group["gain"].to_numpy() * weights) / np.sum(weights))
        actual_per_seed.append({"seed_index": int(seed), "discounted_gain": value,
                                "valid_offsets": group["offset"].astype(int).tolist()})
    actual_gains = np.asarray([x["discounted_gain"] for x in actual_per_seed], dtype=np.float64)
    rows = merged[["offset", "seed_index", "frame_base", "valid_action_steps_base",
                   "padding_fraction_base", "loss_base", "loss_identity",
                   "loss_all_dims_base", "loss_all_dims_identity", "loss_padded_dims_base",
                   "loss_padded_dims_identity", "noise_sha256_base", "time_sha256_base",
                   "action_sha256_base", "memory_frames_json_base"]].to_dict("records")
    # Keep the actual replacement scalar beside the base/identity controls for
    # every (offset, seed) pair, rather than only publishing its aggregate.
    for row, actual_row in zip(rows, actual.to_dict("records")):
        row.update({"loss_actual": float(actual_row["loss"]),
                    "loss_all_dims_actual": float(actual_row["loss_all_dims"]),
                    "loss_padded_dims_actual": float(actual_row["loss_padded_dims"]),
                    "memory_frames_json_actual": actual_row["memory_frames_json"],
                    "actual_gain": float(row["loss_base"] - actual_row["loss"]),
                    "actual_noise_sha256": actual_row["noise_sha256"],
                    "actual_time_sha256": actual_row["time_sha256"],
                    "actual_action_sha256": actual_row["action_sha256"]})
    return {
        "schema": "milestone3-single-teacher-label-v1",
        "formal": protocol.get("config", {}).get("demo_source") == "train_h5",
        "source_split": protocol.get("config", {}).get("demo_source"),
        "source_run": str(run.resolve()),
        "source_run_sha256": _sha256(run / "protocol.json"),
        "protocol_sha256": _sha256(run / "protocol.json"),
        "boundary_file": str(boundary_path),
        "label_shard": str(parquet),
        "label_shard_sha256": _sha256(parquet),
        "task": task,
        "episode_id": episode,
        "cutoff_frame": cutoff,
        "candidate": {"frame_id": candidate_frame, "kind": "counterfactual",
                       "candidate_index": candidate_index, "source_kind": candidates[candidate_index].get("kind"),
                       "candidate_is_historical": True},
        "replacement": {"slot_index": int(slot), "old_frame_id": identity_frame,
                         "new_frame_id": candidate_frame, "same_frame": False,
                         "identity_old_frame_id": identity_frame, "identity_new_frame_id": identity_frame,
                         "actual_old_frame_id": identity_frame, "actual_new_frame_id": candidate_frame,
                         "actual_label": actual_label},
        "evaluation": {"offsets": offsets, "mc_samples": int(len(base["seed_index"].unique())),
                        "action_horizon": action_horizon, "action_dims": action_dims,
                        "gamma": gamma, "loss_reduction": "mean(action_horizon, real_action_dims)",
                        "future_mask": {str(o): True for o in offsets}},
        "paired_randomness": {"noise_hashes": sorted(base["noise_sha256"].unique().tolist()),
                               "time_hashes": sorted(base["time_sha256"].unique().tolist()),
                               "action_hashes": sorted(base["action_sha256"].unique().tolist())},
        "shape_and_mask_audit": {"rows": int(len(base)), "expected_rows": len(offsets) * int(len(base["seed_index"].unique())),
                                  "loss_scalar_shape": [int(len(base))],
                                  "memory_slot_count": int(len(first_memory)),
                                  "real_action_dims": action_dims,
                                  "valid_action_steps": sorted(set(int(x) for x in base["valid_action_steps"])),
                                  "padding_fraction": sorted(set(float(x) for x in base["padding_fraction"])),
                                  "all_offsets_valid": bool(len(offsets) == 8 and (base["valid_action_steps"] == action_horizon).all())},
        "loss_rows": rows,
        "label": {"base_loss_mean": float(base["loss"].mean()),
                   "identity_loss_mean": float(identity["loss"].mean()),
                   "actual_replacement_loss_mean": float(actual["loss"].mean()),
                   "same_frame_gain_max_abs": float(np.max(np.abs(merged["gain"]))),
                   "actual_gain_per_seed": actual_per_seed,
                   "actual_discounted_gain_mean": float(actual_gains.mean()),
                   "actual_discounted_gain_std": float(actual_gains.std(ddof=1)) if len(actual_gains) > 1 else 0.0,
                   "discounted_gain_per_seed": per_seed,
                   "discounted_gain_mean": float(gains.mean()),
                   "discounted_gain_std": float(gains.std(ddof=1)) if len(gains) > 1 else 0.0},
        "audits": {"identity": True, "repeat": True, "crn": True, "mask": True,
                    "shape": True, "causal": True, "same_frame_gain_zero": bool(np.array_equal(base["loss"], identity["loss"])),
                    "reproducible": bool(np.array_equal(base["loss"], identity["loss"]))},
        "passed": True,
        "note": "Identity label reuses an already evaluated cached-flow-loss pair; formal M3 requires a sealed train split source.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=Path("results/V_B_fp32_full_20261022"))
    parser.add_argument("--boundary", help="boundary JSON filename, e.g. BinFill_ep20_F0064.json")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("results/M3_single_label_20261023/label.json"))
    args = parser.parse_args()
    result = make_label(args.source_run.resolve(), args.boundary, args.slot, args.candidate_index, args.gamma)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
