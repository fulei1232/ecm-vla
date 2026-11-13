#!/usr/bin/env python3
"""Compare explicit and prefix-cached Flow Matching losses on the frozen checkpoint."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, POLICY


def error_metrics(actual, reference) -> dict:
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f"loss shape mismatch: {actual.shape} != {reference.shape}")
    if not (np.isfinite(actual).all() and np.isfinite(reference).all()):
        raise ValueError("non-finite loss; equivalence cannot pass")
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    return {
        "max_rel_err": float(np.max(error / np.maximum(np.abs(reference), 1e-8))),
        "max_abs_err": float(np.max(error)),
    }


def cache_error_metrics(actual, reference) -> list[dict]:
    """Host-side KV diagnostics; use L2 alongside near-zero-sensitive max-relative error."""
    import jax

    result = []
    for actual_leaf, reference_leaf in zip(jax.tree.leaves(actual["kv_cache"]),
                                            jax.tree.leaves(reference["kv_cache"]), strict=True):
        a = np.asarray(actual_leaf, dtype=np.float32)
        b = np.asarray(reference_leaf, dtype=np.float32)
        metrics = error_metrics(a, b)
        metrics.update({
            "relative_l2_error": float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-8)),
            "different_fraction": float(np.mean(a != b)),
            "dtype": str(reference_leaf.dtype),
            "shape": list(b.shape),
        })
        result.append(metrics)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--task", default="VideoUnmaskSwap")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-relative-error", type=float, default=1e-3)
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="Forward compute dtype for BOTH full and cached paths; weights stay frozen")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Compare eager/JIT conditions and caches, plus a seeded full forward")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or not np.isfinite(args.max_relative_error) or args.max_relative_error <= 0:
        parser.error("samples and max-relative-error must be positive and finite")
    args.server_config = args.server_config.resolve()
    if args.out:
        args.out = args.out.resolve()
    config = json.loads(args.server_config.read_text())
    # Upstream history config loading still resolves paths relative to the policy root.
    os.chdir(POLICY)
    os.environ.setdefault("OPENPI_DATA_HOME", config["openpi_data_home"])
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config["policy_gpu"]))
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(ECM / "src"))
    sys.path.insert(0, str(POLICY / "src"))

    import jax
    import jax.numpy as jnp
    if args.compute_dtype == "float32":
        # Prevent reduced-precision TF32 multiplication from obscuring the FP32 control.
        jax.config.update("jax_default_matmul_precision", "highest")
    from openpi.shared import nnx_utils
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config as training_config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from memory_management.validation.demo_source import (
        action_chunk, load_demo_arrays, read_demo_frames, read_demo_wrist_frames,
    )
    from memory_management.validation.cached_flow_loss import (
        jit_cached_loss,
        jit_explicit_loss,
        jit_explicit_loss_with_cache,
        jit_modulation_prefix_cache,
        jit_seeded_explicit_loss,
        draw_explicit_conditions,
        prepare_modulation_prefix_cache,
    )
    from memory_management.validation.partition import partition

    try:
        devices = jax.devices("cuda")
    except Exception as error:
        raise RuntimeError(
            "V-B G0b requires a CUDA JAX backend, but this policy environment cannot initialize one. "
            "Check the JAX CUDA plugin, driver compatibility, and GPU visibility before running the gate."
        ) from error
    if not devices:
        raise RuntimeError("V-B G0b requires a CUDA JAX backend; no CUDA devices are visible")

    checkpoint = Path(config["checkpoint_path"])
    train_config = training_config.get_config(config["policy_config"])
    train_config = dataclasses.replace(
        train_config, model=dataclasses.replace(train_config.model, dtype=args.compute_dtype)
    )
    policy = policy_config.create_trained_policy(
        train_config, checkpoint, seed=int(config["model_seed"])
    )
    if policy._model.integration_type != "modulation":
        raise RuntimeError(f"expected modulation checkpoint, got {policy._model.integration_type}")

    demo = load_demo_arrays(args.task, args.episode_id)
    eligible = list(range(max(64, demo["exec_start_idx"]), len(demo["joint_action"]), 16))
    if not eligible:
        raise RuntimeError("demo has no legal teacher probe boundary")
    cutoff = eligible[len(eligible) // 2]
    indices = list(partition(cutoff).uniform)
    images = read_demo_frames(args.task, args.episode_id, indices)[:, None]
    states = demo["joint_state"][indices]
    policy.add_buffer({"images": images, "state": states, "exec_start_idx": demo["exec_start_idx"]})
    memory = policy._prepare_history({})
    raw = {
        "observation/image": read_demo_frames(args.task, args.episode_id, [cutoff])[0],
        "observation/wrist_image": read_demo_wrist_frames(args.task, args.episode_id, [cutoff])[0],
        "observation/state": demo["joint_state"][cutoff],
        "prompt": demo["task_goal"],
        **memory,
    }
    transformed = policy._input_transform(raw)
    observation = HistAugObservation.from_dict(
        jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
    )
    actions_np, valid_action_steps = action_chunk(demo["joint_action"], cutoff, policy._model.action_horizon)
    # Normalize the 8 dataset action dimensions before the policy transform pads to action_dim.
    transformed_actions = policy._input_transform({**raw, "actions": actions_np})["actions"]
    actions = jnp.asarray(transformed_actions[None, ...])
    print("Encoding prefix with JIT...", flush=True)
    prefix_cache = jit_modulation_prefix_cache(policy._model)(observation)
    jax.block_until_ready(prefix_cache)
    native = nnx_utils.module_jit(policy._model.compute_loss)
    explicit = jit_explicit_loss(policy._model)
    cached = jit_cached_loss(policy._model)
    draw_conditions = jax.jit(lambda key: draw_explicit_conditions(jax, actions, key))
    diagnostics = {}
    if args.diagnostics:
        print("Checking eager versus JIT prefix cache...", flush=True)
        eager_cache = prepare_modulation_prefix_cache(policy._model, observation)
        diagnostics["prefix_eager_vs_jit"] = cache_error_metrics(eager_cache, prefix_cache)
        del eager_cache
        seeded_explicit = jit_seeded_explicit_loss(policy._model)
        explicit_with_cache = jit_explicit_loss_with_cache(policy._model)

    rng = jax.random.key(20261022)
    worst = 0.0
    records = []
    for index in range(args.samples):
        rng, sample_rng = jax.random.split(rng)
        preprocess_rng, _, _ = jax.random.split(sample_rng, 3)
        noise, time = draw_conditions(sample_rng)
        native_loss, _ = native(sample_rng, observation, actions)
        explicit_losses = explicit(preprocess_rng, observation, actions, noise, time)
        cached_losses = cached(prefix_cache, observation, actions, noise, time)
        ref = np.asarray(jax.device_get(native_loss))
        full = np.asarray(jax.device_get(explicit_losses["loss_all_dims"]))
        fast = np.asarray(jax.device_get(cached_losses["loss_all_dims"]))
        record = {"sample": index}
        for label, actual, reference in (
            ("native_vs_explicit", full, ref),
            ("cached_vs_explicit", fast, full),
            ("cached_vs_explicit_real_dims", cached_losses["loss"], explicit_losses["loss"]),
        ):
            metrics = error_metrics(actual, reference)
            record.update({f"{label}_{key}": value for key, value in metrics.items()})
            worst = max(worst, metrics["max_rel_err"])
        if index == 0:
            record.update({
                "native_loss_sample0": ref.tolist(),
                "explicit_loss_sample0": full.tolist(),
                "cached_loss_sample0": fast.tolist(),
            })
            repeated = cached(prefix_cache, observation, actions, noise, time)
            record["cached_repeat_bitwise_equal"] = all(
                np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(jax.tree.leaves(repeated), jax.tree.leaves(cached_losses), strict=True)
            )
            if args.diagnostics:
                eager_noise, eager_time = draw_explicit_conditions(jax, actions, sample_rng)
                diagnostics["conditions_eager_vs_jit"] = {
                    "noise": error_metrics(eager_noise, noise),
                    "time": error_metrics(eager_time, time),
                    "eager_time": np.asarray(eager_time).tolist(),
                    "jit_time": np.asarray(time).tolist(),
                }
                print("Checking seeded explicit forward against native...", flush=True)
                seeded = seeded_explicit(sample_rng, observation, actions)["loss_all_dims"]
                diagnostics["native_vs_seeded_explicit"] = error_metrics(seeded, ref)
                diagnostics["seeded_vs_external_conditions_explicit"] = error_metrics(full, seeded)
                print("Isolating prefix versus suffix with the full forward's own KV cache...", flush=True)
                full_with_cache = explicit_with_cache(preprocess_rng, observation, actions, noise, time)
                diagnostics["full_cache_forward_vs_explicit"] = error_metrics(
                    full_with_cache["loss_all_dims"], full
                )
                diagnostics["prefix_jit_vs_full_forward"] = cache_error_metrics(
                    prefix_cache, full_with_cache["prefix_cache"]
                )
                cache_from_full = cached(full_with_cache["prefix_cache"], observation, actions, noise, time)
                diagnostics["cached_with_full_prefix_vs_same_full_forward"] = {
                    key: error_metrics(cache_from_full[key], full_with_cache[key])
                    for key in ("loss", "loss_all_dims")
                }
        records.append(record)
        print(f"Sample {index + 1}/{args.samples}: native/full={record['native_vs_explicit_max_rel_err']:.6g}, "
              f"cache/full={record['cached_vs_explicit_max_rel_err']:.6g}, "
              f"cache/full[:8]={record['cached_vs_explicit_real_dims_max_rel_err']:.6g}", flush=True)
    result = {
        "schema": "v-b-cached-flow-loss-equivalence-v3",
        "task": args.task,
        "episode_id": args.episode_id,
        "cutoff_frame": cutoff,
        "valid_action_steps": valid_action_steps,
        "samples": len(records),
        "max_relative_error": worst,
        "threshold": args.max_relative_error,
        "jax_devices": [str(device) for device in devices],
        "conditions_mode": "jitted_noise_and_time",
        "prefix_mode": "jitted",
        "compute_dtype": args.compute_dtype,
        "checkpoint_restore_dtype": "bfloat16",
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "checkpoint_path": str(checkpoint.resolve()),
        "prefix_cache_dtypes": sorted({str(leaf.dtype) for leaf in jax.tree.leaves(prefix_cache["kv_cache"])}),
        "sample_check_passed": worst <= args.max_relative_error,
        "repeat_check_passed": records[0]["cached_repeat_bitwise_equal"],
        "passed": (len(records) >= 50 and worst <= args.max_relative_error
                   and records[0]["cached_repeat_bitwise_equal"]),
        "diagnostics": diagnostics,
        "records": records,
    }
    encoded = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")
    print(encoded)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
