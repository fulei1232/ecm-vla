#!/usr/bin/env python3
"""Isolate V-B prefix batching from suffix-only cache differences without changing gates.

Reads the failed run's sealed protocol and feature cache. Produces diagnostics
only: it never writes teacher labels or changes a failed run into a passed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_v_b_teacher_probe as probe
from v_a_common import ECM, atomic_json, sha256, timestamp
from verify_cached_flow_loss import error_metrics
from memory_management.validation.cached_flow_loss import common_random_conditions
from memory_management.validation.probe_design import fixed_variants, design_seed, memory_frames
from memory_management.validation.partition import partition


def scalar_losses(values):
    """Reduce full-forward per-step losses to the exact production scalar metrics."""
    return {
        "loss": values["loss"].mean(axis=-1),
        "loss_all_dims": values["loss_all_dims"].mean(axis=-1),
        "loss_padded_dims": values["loss_per_dim"][..., 8:].mean(axis=-1),
    }


def compare(actual, reference):
    return {key: error_metrics(actual[key], reference[key]) for key in reference}


def valid_prefix_errors(actual, reference):
    """Exclude masked prompt rows, whose full attention context is intentionally different."""
    import jax
    mask = np.asarray(reference["prefix_mask"], dtype=bool)
    result = []
    for a, b in zip(jax.tree.leaves(actual["kv_cache"]), jax.tree.leaves(reference["kv_cache"]), strict=True):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        a = np.broadcast_to(a, b.shape)
        valid = np.broadcast_to(mask[None, :, :, None, None], b.shape)
        x, y = a[valid], b[valid]
        result.append({**error_metrics(x, y),
            "relative_l2_error": float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)), 1e-8)),
            "dtype": str(b.dtype), "shape": list(b.shape), "valid_elements": int(valid.sum())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--samples", type=int, default=16,
                        help="Number of the failed run's original stratified seeds to check")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Check provenance and existing features without importing JAX")
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.out.resolve() if args.out else run/"cache_batch_diagnostic.json"
    protocol_path = run/"protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["code_sha256"] != probe.source_hashes():
        raise ValueError("production sources changed since this failed run was sealed")
    gate_path = Path(protocol["g0b_report"])
    if sha256(gate_path) != protocol["g0b_sha256"]:
        raise ValueError("sealed G0b report changed")
    probe.accepted_gate(gate_path, Path(protocol["checkpoint_path"]), protocol["primary_dtype"])
    if not protocol["smoke"] or len(protocol["plan"]) != 1:
        raise ValueError("diagnose a single-boundary smoke run")
    cfg, plan = protocol["config"], protocol["plan"][0]
    nseed, batch_size = cfg["loss"]["mc_samples"], cfg["runtime"]["batch_size"]
    if not 1 <= args.samples <= nseed or batch_size != 4:
        parser.error("samples must be within the sealed seed count; this diagnostic requires batch=4")
    if out.exists():
        raise ValueError(f"preserving previous diagnostics; choose a new --out: {out}")
    task, episode, cutoff = plan["task"], plan["episode_id"], plan["cutoff"]
    demo = probe.load_demo_arrays(task, episode)
    expected = protocol["manifest_provenance"][f"{task}/{episode}"]
    if (sha256(demo["manifest"]["manifest_path"]) != expected["manifest_sha256"]
            or sha256(expected["h5_path"]) != expected["h5_sha256"]):
        raise ValueError("sealed demonstration changed")
    namespace = hashlib.sha256(json.dumps({k: protocol[k] for k in (
        "checkpoint_path", "g0b_sha256", "primary_dtype", "code_sha256")}, sort_keys=True).encode()).hexdigest()[:16]
    feature_path = ECM/cfg["runtime"]["feature_cache"]/namespace/task/f"ep{episode:02d}.npz"
    if not feature_path.is_file():
        raise ValueError(f"expected the failed run's existing feature cache: {feature_path}")
    print(json.dumps({"task": task, "episode_id": episode, "cutoff": cutoff,
        "feature_cache": str(feature_path), "samples": args.samples, "out": str(out)}, indent=2), flush=True)
    if args.prepare_only:
        return 0
    policy = probe.load_policy(protocol)
    import jax
    import jax.numpy as jnp
    from memory_management.validation.cached_flow_loss import (
        jit_probe_batch, jit_modulation_prefix_cache, jit_explicit_loss_with_cache,
    )

    # Reuses the exact feature namespace of the failed protocol, including its source hashes.
    features = probe.episode_features(policy, demo, task, episode, protocol)
    specs = fixed_variants(plan["candidates"], list(partition(cutoff).history),
                           seed=design_seed(task, episode, cutoff), random_sets={})[:4]
    memories = [probe.pack_memory(policy, features, memory_frames(cutoff, cutoff, plan["candidates"], s))
                for s in specs]
    obs, actions, valid_steps = probe.prepare_observation(policy, demo, task, episode, cutoff, memories[0])
    memory_batch = {name: jnp.asarray(np.stack([m[name] for m in memories])) for name in probe.STATIC_KEYS}
    batch_obs = jax.tree.map(lambda v: jnp.repeat(v, batch_size, axis=0), obs).replace(**memory_batch)
    batch_actions = jnp.repeat(actions, batch_size, axis=0)
    prefix = jit_modulation_prefix_cache(policy._model)
    cached = jit_probe_batch(policy._model)
    full = jit_probe_batch(policy._model, cached=False)
    full_with_cache = jit_explicit_loss_with_cache(policy._model)
    print("Encoding diagnostic prefixes: batch=1 and batch=4...", flush=True)
    cache1, cache4 = prefix(obs), prefix(batch_obs)
    jax.block_until_ready((cache1, cache4))
    noise, tau = common_random_conditions(task, episode, cutoff, nseed,
        policy._model.action_horizon, policy._model.action_dim, offset=0)
    report = {
        "schema": "v-b-cache-batch-diagnostic-v1", "created": timestamp(),
        "diagnostic_only": True, "protocol_sha256": sha256(protocol_path),
        "feature_cache_path": str(feature_path), "feature_cache_sha256": sha256(feature_path),
        "diagnostic_source_sha256": sha256(Path(__file__)),
        "task": task, "episode_id": episode, "cutoff": cutoff,
        "valid_action_steps": valid_steps, "compute_dtype": protocol["primary_dtype"],
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "jax_devices": [str(d) for d in jax.devices()],
        "batch_size": batch_size, "variant_order": [s["label"] for s in specs],
        "threshold": cfg["gates"]["cached_vs_full_rel_err_max"],
        "prefix1_vs_prefix4_valid_tokens": valid_prefix_errors(cache1, cache4),
        "records": [],
        "limitations": "Diagnostic only; batch=4 prefix is an unvalidated alternative until runtime gates pass.",
    }
    for si in range(args.samples):
        eps = jnp.repeat(jnp.asarray(noise[si:si+1]), batch_size, axis=0)
        times = jnp.repeat(jnp.asarray(tau[si:si+1]), batch_size, axis=0)
        reference = jax.device_get(full(cache1, obs, memory_batch, actions, eps, times))
        original = jax.device_get(cached(cache1, obs, memory_batch, actions, eps, times))
        matched_batch = jax.device_get(cached(cache4, obs, memory_batch, actions, eps, times))
        row = {"seed_index": si, "time": float(tau[si]),
            "full": {k: np.asarray(v).tolist() for k, v in reference.items()},
            "cached_prefix1": {k: np.asarray(v).tolist() for k, v in original.items()},
            "cached_prefix4": {k: np.asarray(v).tolist() for k, v in matched_batch.items()},
            "prefix1_cached_vs_full": compare(original, reference),
            "prefix4_cached_vs_full": compare(matched_batch, reference)}
        if si == 0:
            print("Isolating suffix with the full batch's own prefix KV...", flush=True)
            complete = full_with_cache(None, batch_obs, batch_actions, eps, times)
            cache_from_full = complete["prefix_cache"]
            complete_scalar = jax.device_get(scalar_losses(complete))
            isolated = jax.device_get(cached(cache_from_full, obs, memory_batch, actions, eps, times))
            row["full_with_cache_vs_production_full"] = compare(complete_scalar, reference)
            row["suffix_with_full_own_cache_vs_same_full"] = compare(isolated, complete_scalar)
            row["prefix1_vs_full_valid_tokens"] = valid_prefix_errors(cache1, cache_from_full)
            row["prefix4_vs_full_valid_tokens"] = valid_prefix_errors(cache4, cache_from_full)
            # Use identical conditions and memory in each B=1 control to expose batch-size effects.
            single_full, single_cached = [], []
            for vi in range(batch_size):
                single_mem = {k: v[vi:vi+1] for k,v in memory_batch.items()}
                single_full.append(jax.device_get(full(cache1, obs, single_mem, actions, eps[:1], times[:1])))
                single_cached.append(jax.device_get(cached(cache1, obs, single_mem, actions, eps[:1], times[:1])))
            loop_full = {k: np.concatenate([r[k] for r in single_full]) for k in reference}
            loop_cached = {k: np.concatenate([r[k] for r in single_cached]) for k in reference}
            row["single_cached_vs_single_full"] = compare(loop_cached, loop_full)
            row["batch_full_vs_single_full"] = compare(reference, loop_full)
            row["batch_cached_vs_single_cached"] = compare(original, loop_cached)
            del complete, cache_from_full
        report["records"].append(row)
        # Persist incremental evidence even if a later compilation/run fails.
        atomic_json(out, report)
        print(f"Seed {si}: tau={float(tau[si]):.6g}, "
              f"prefix1/full={row['prefix1_cached_vs_full']['loss']['max_rel_err']:.6g}, "
              f"prefix4/full={row['prefix4_cached_vs_full']['loss']['max_rel_err']:.6g}", flush=True)
    report["max_relative_errors"] = {mode: {
        metric: max(r[mode][metric]["max_rel_err"] for r in report["records"])
        for metric in ("loss", "loss_all_dims", "loss_padded_dims")}
        for mode in ("prefix1_cached_vs_full", "prefix4_cached_vs_full")}
    atomic_json(out, report)
    print(json.dumps({"report": str(out), "samples": len(report["records"]),
                      "max_relative_errors": report["max_relative_errors"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
