#!/usr/bin/env python3
"""Seal and run the held-out V-B teacher probe after the FP32 G0b gate.

--prepare-only validates data/protocol without importing JAX. --smoke runs one
boundary with fewer variants/offsets and is never interpreted as a formal V-B.
Use --resume RUN to continue completed boundary shards without changing protocol.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from v_a_common import ECM, POLICY, CHECKPOINT, OPENPI_DATA_HOME, atomic_json, sha256, timestamp

sys.path.insert(0, str(ECM / "src"))
from memory_management.validation.demo_source import load_demo_arrays, read_demo_frames, read_demo_wrist_frames, action_chunk
from memory_management.validation.teacher_probe import decision_boundaries, select_v_b_candidates
from memory_management.validation.partition import partition
from memory_management.validation.probe_design import (
    array_hash, fixed_variants, design_seed, memory_frames, greedy_variants, discounted_gain, variant_features,
)
from memory_management.validation.cached_flow_loss import common_random_conditions

STATIC_KEYS = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")


def source_hashes():
    files = [*sorted((ECM / "src/memory_management/validation").rglob("*.py")),
             Path(__file__), ECM / "scripts/v_b_table_io.py", ECM / "scripts/analyze_v_b.py"]
    files += [POLICY / "src/mme_vla_suite/shared/mem_buffer.py",
              POLICY / "src/mme_vla_suite/models/integration/history_pi0.py",
              POLICY / "src/mme_vla_suite/models/integration/history_gemma.py"]
    return {str(p): sha256(p) for p in files}


def accepted_gate(path: Path, checkpoint: Path, dtype: str):
    gate = json.loads(path.read_text())
    if not (gate.get("passed") is True and gate.get("samples", 0) >= 50
            and 0 <= float(gate["max_relative_error"]) <= 1e-3
            and gate.get("compute_dtype") == dtype and gate.get("prefix_mode") == "jitted"
            and gate.get("matmul_precision") == "highest"
            and gate.get("repeat_check_passed") is True
            and Path(gate["checkpoint_path"]).resolve() == checkpoint.resolve()):
        raise ValueError("G0b report does not authorize this dtype/checkpoint/runtime")
    return gate


def feature_namespace(protocol):
    return hashlib.sha256(json.dumps({k: protocol[k] for k in (
        "checkpoint_path", "g0b_sha256", "primary_dtype", "code_sha256")}, sort_keys=True).encode()).hexdigest()[:16]


def seal_feature_imports(source_run, protocol):
    """Bind compatible encoded frames to a new run without importing any old loss labels."""
    source_path = source_run.resolve()/"protocol.json"
    old = json.loads(source_path.read_text())
    for key in ("checkpoint_path", "g0b_sha256", "primary_dtype", "checkpoint_restore_dtype", "matmul_precision"):
        if old[key] != protocol[key]:
            raise ValueError(f"feature reuse has incompatible {key}")
    for key in ("model_seed", "encode_batch_size"):
        if old["config"]["runtime"][key] != protocol["config"]["runtime"][key]:
            raise ValueError(f"feature reuse has incompatible {key}")
    # Only this runner changed for the prefix-batching fix; encoder/model dependencies must match.
    current, previous = protocol["code_sha256"], old["code_sha256"]
    if set(current) != set(previous) or any(current[k] != previous[k] for k in current if k != str(Path(__file__))):
        raise ValueError("feature reuse requires unchanged encoder/model dependencies")
    namespace, imports = feature_namespace(old), {}
    for key, provenance in protocol["manifest_provenance"].items():
        if key not in old["manifest_provenance"]:
            continue
        if old["manifest_provenance"][key] != provenance:
            raise ValueError(f"feature reuse has incompatible demo: {key}")
        task, episode = key.rsplit("/", 1)
        path = ECM/old["config"]["runtime"]["feature_cache"]/namespace/task/f"ep{int(episode):02d}.npz"
        if not path.exists():
            continue
        metadata = {"h5_sha256": provenance["h5_sha256"], "namespace": namespace}
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata"])) != metadata:
                raise ValueError(f"feature cache provenance mismatch: {key}")
        imports[key] = {"path": str(path), "sha256": sha256(path), "metadata": metadata,
                        "source_protocol": str(source_path), "source_protocol_sha256": sha256(source_path)}
    if not imports:
        raise ValueError("no compatible existing episode feature caches to reuse")
    return imports


def seal(args):
    cfg = yaml.safe_load(args.config.read_text())["v_b"]
    if cfg["loss"]["compute_dtype"] != "float32" or cfg["loss"]["mc_samples"] < 4:
        raise ValueError("this runner requires the validated FP32 teacher and >=4 paired seeds")
    gate_path = (ECM / cfg["runtime"]["g0b_report"]).resolve()
    accepted_gate(gate_path, args.checkpoint, cfg["loss"]["compute_dtype"])
    tasks = args.tasks or cfg["tasks"]
    demo_root = (args.demo_root or (ECM / ("data/train_demos" if cfg.get("demo_source") == "train_h5" else "data/val_demos"))).resolve()
    ids = args.episode_ids or list(range(cfg["episode_ids"]["start"], cfg["episode_ids"]["stop"]))
    if cfg.get("demo_source") != "train_h5" and not args.smoke and (tasks != cfg["tasks"] or ids != list(range(20, 30))):
        raise ValueError("subset runs must use --smoke so they cannot be mistaken for the formal study")
    if args.smoke:
        cfg["future"]["offsets"] = [0]
    run = args.out.resolve() if args.out else ECM / "results" / (
        "V_B_" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + ("_smoke" if args.smoke else ""))
    if (run / "protocol.json").exists():
        raise ValueError("existing run: use --resume to preserve the sealed protocol")
    run.mkdir(parents=True, exist_ok=True)
    plans, excluded, manifests = [], [], {}
    for task in tasks:
        for episode in ids:
            demo = load_demo_arrays(task, episode, demo_root)
            manifest = demo["manifest"]
            if cfg.get("demo_source") != "train_h5" and not args.smoke and not 20 <= episode < 30:
                raise ValueError("formal V-B uses held-out val episodes 20-29")
            if cfg.get("demo_source") == "train_h5" and demo["manifest"].get("split") != "train":
                raise ValueError("train_h5 protocol requires train-split manifests")
            manifests[f"{task}/{episode}"] = {
                "manifest_sha256": sha256(manifest["manifest_path"]),
                "h5_path": manifest["h5_path"], "h5_sha256": manifest["h5_sha256"],
            }
            cuts = decision_boundaries(len(demo["joint_action"]), demo["exec_start_idx"],
                stride=cfg["decision_stride_frames"], min_cutoff_frame=cfg["min_cutoff_frame"],
                count=cfg["boundaries_per_episode"])
            if not cuts:
                excluded.append({"task": task, "episode_id": episode,
                                 "reason": "fewer than three legal execution decisions", "n_frames": len(demo["joint_action"])})
            for cutoff in cuts:
                p = partition(cutoff)
                kemo = cfg["candidates"]["kemo"]
                candidates = select_v_b_candidates(task=task, episode_id=episode,
                    model_seed=cfg["runtime"]["model_seed"], cutoff_frame=cutoff,
                    recent_first_frame=p.recent[0], d0_frames=list(p.uniform),
                    joint_state=demo["joint_state"][:, :7], subgoal_boundaries=demo["subgoal_boundaries"].tolist(),
                    kemo_count=cfg["candidates"]["kemo_peaks"], random_count=cfg["candidates"]["random"],
                    left=kemo["peak_left"], right=kemo["peak_right"], min_gap=kemo["min_event_gap_frames"],
                    window=kemo["window"])
                plans.append({"task": task, "episode_id": episode, "cutoff": cutoff,
                              "candidates": candidates, "candidate_shortfall": max(0, 6-len(candidates))})
    if args.smoke:
        plans = plans[:1]
    if not plans:
        raise ValueError("no eligible boundaries; choose a longer demo for smoke")
    protocol = {"schema": "v-b-protocol-v1", "created": timestamp(), "smoke": args.smoke,
        "config": cfg, "checkpoint_path": str(args.checkpoint.resolve()), "policy_config": "mme_vla_suite",
        "openpi_data_home": str(OPENPI_DATA_HOME), "g0b_report": str(gate_path), "g0b_sha256": sha256(gate_path),
        "analysis_python": str(Path(args.analysis_python).resolve()), "code_sha256": source_hashes(),
        "manifest_provenance": manifests, "demo_root": str(demo_root), "plan": plans, "excluded_episodes": excluded,
        "greedy_analysis": "selected on even seeds, evaluated on odd only; excluded from fixed-pool reliability and symmetric xfit",
        "candidate_shortfalls": "use available confirmed events; report shortages; never invent event peaks",
        "prefix_batching": "repeat observation before prefix JIT to match suffix/full batch size",
        "prefix_batch_size": cfg["runtime"]["batch_size"],
        "primary_dtype": "float32", "checkpoint_restore_dtype": "bfloat16", "matmul_precision": "highest"}
    protocol["feature_cache_imports"] = seal_feature_imports(args.reuse_features_from, protocol) if args.reuse_features_from else {}
    atomic_json(run / "protocol.json", protocol)
    return run, protocol


def load_policy(protocol):
    os.environ.setdefault("OPENPI_DATA_HOME", protocol["openpi_data_home"])
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(POLICY / "src"))
    os.chdir(POLICY)
    import jax
    from mme_vla_suite.training import config
    from mme_vla_suite.policies.policy_config import create_trained_policy
    jax.config.update("jax_default_matmul_precision", "highest")
    if not jax.devices("cuda"):
        raise RuntimeError("V-B requires CUDA")
    train = config.get_config(protocol["policy_config"])
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, dtype="float32"))
    return create_trained_policy(train, Path(protocol["checkpoint_path"]),
                                 seed=protocol["config"]["runtime"]["model_seed"])


def episode_features(policy, demo, task, episode, protocol):
    """Ingest every raw frame chronologically with its true ID, preserving official position codes."""
    imported = protocol.get("feature_cache_imports", {}).get(f"{task}/{episode}")
    if imported:
        path = Path(imported["path"])
        if sha256(path) != imported["sha256"] or sha256(imported["source_protocol"]) != imported["source_protocol_sha256"]:
            raise ValueError("imported feature cache or its source protocol changed")
        with np.load(path, allow_pickle=False) as f:
            metadata = json.loads(str(f["metadata"]))
            if metadata != imported["metadata"] or metadata["h5_sha256"] != demo["manifest"]["h5_sha256"]:
                raise ValueError("imported feature cache provenance mismatch")
            features = {key: f[key] for key in ("image", "position", "state")}
        if any(len(v) != len(demo["joint_action"]) or not np.isfinite(v).all() for v in features.values()):
            raise ValueError("invalid imported frame features")
        print(f"{task} ep{episode:02d} reusing verified frame features: {path}", flush=True)
        return features
    namespace = feature_namespace(protocol)
    root = ECM / protocol["config"]["runtime"]["feature_cache"] / namespace / task
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"ep{episode:02d}.npz"
    expected = {"h5_sha256": demo["manifest"]["h5_sha256"], "namespace": namespace}
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata"])) != expected:
                raise ValueError("feature cache provenance mismatch")
            features = {key: f[key] for key in ("image", "position", "state")}
        return features
    policy.reset()
    n = len(demo["joint_action"])
    chunk = protocol["config"]["runtime"]["encode_batch_size"]
    for start in range(0, n, chunk):
        ids = list(range(start, min(start+chunk, n)))
        demo_root = Path(protocol.get("demo_root", str(ECM / "data/val_demos")))
        policy.add_buffer({"images": read_demo_frames(task, episode, ids, demo_root)[:, None],
                           "state": demo["joint_state"][ids], "exec_start_idx": demo["exec_start_idx"]})
        for frame in ids:
            row = policy.mem_buffer._history_feats[frame]
            for key in list(row):
                if key not in ("image_emb_4x4", "pos_emb_4x4", "state_emb"):
                    del row[key]
        print(f"{task} ep{episode:02d} encoding {min(start+chunk,n)}/{n}", flush=True)
    features = {key: np.stack([policy.mem_buffer._history_feats[i][name] for i in range(n)]).astype(np.float32)
                for key, name in (("image", "image_emb_4x4"), ("position", "pos_emb_4x4"), ("state", "state_emb"))}
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, metadata=json.dumps(expected), **features)
    os.replace(temp, path)
    policy.mem_buffer._history_feats.clear()
    return features


def pack_memory(policy, features, frames):
    feats = {i: {"image_emb_4x4": features["image"][i], "pos_emb_4x4": features["position"][i],
                 "state_emb": features["state"][i]} for i in frames}
    image, position, state, mask = policy.mem_buffer._prepare_frame_sampling(
        feats, frames, policy.config.budget, policy.config.token_per_image)
    return dict(zip(STATIC_KEYS, (image, position, policy._normalize_state(state), mask)))


def prepare_observation(policy, demo, task, episode, frame, memory, demo_root=None):
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    demo_root = demo_root or (ECM / "data/val_demos")
    raw = {"observation/image": read_demo_frames(task, episode, [frame], demo_root)[0],
           "observation/wrist_image": read_demo_wrist_frames(task, episode, [frame], demo_root)[0],
           "observation/state": demo["joint_state"][frame], "prompt": demo["task_goal"], **memory}
    actions, valid_steps = action_chunk(demo["joint_action"], frame, policy._model.action_horizon)
    data = policy._input_transform({**raw, "actions": actions})
    target = jnp.asarray(data.pop("actions")[None])
    obs = HistAugObservation.from_dict(jax.tree.map(lambda value: jnp.asarray(value)[None], data))
    return obs, target, valid_steps


def write_parquet(protocol, path, rows):
    subprocess.run([protocol["analysis_python"], str(ECM / "scripts/v_b_table_io.py"), "--out", str(path)],
        input="".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), text=True, check=True)


def run_boundary(run, protocol, plan, policy, demo, features, runtime):
    import jax
    import jax.numpy as jnp
    cfg = protocol["config"]
    task, episode, cutoff, candidates = plan["task"], plan["episode_id"], plan["cutoff"], plan["candidates"]
    key = f"{task}_ep{episode:02d}_F{cutoff:04d}"
    dest = run / "labels" / (key + ".parquet")
    done_path = run / "boundaries" / (key + ".json")
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if not dest.exists() or done["parquet_sha256"] != sha256(dest):
            raise ValueError("completed boundary shard was modified")
        return
    offsets = [o for o in cfg["future"]["offsets"] if cutoff+cfg["decision_stride_frames"]*o < len(demo["joint_action"])]
    history = list(partition(cutoff).history)
    single = cfg.get("m3_single")
    specs = fixed_variants(candidates, history, seed=design_seed(task, episode, cutoff),
                           random_sets={k: (0 if protocol["smoke"] or single else cfg["interventions"][f"k{k}"]["random_sets"])
                                        for k in (2,4)})
    if single:
        candidate_index = int(single["candidate_index"])
        slot = int(single["slot"])
        target = f"k1_c{candidate_index}_j{slot}"
        selected = [spec for spec in specs if spec["label"] in {"base", "d0", "hist_off", "identity", target}]
        if len(selected) != 5 or selected[-1]["label"] != target:
            raise ValueError(f"M3 single intervention is unavailable: {target}")
        specs = selected
    elif protocol["smoke"]:
        singles = specs[4:]
        specs = specs[:4] + [singles[i*len(history)+i % len(history)] for i in range(min(len(candidates), 6))]
    nseed, batch_size = cfg["loss"]["mc_samples"], cfg["runtime"]["batch_size"]
    if batch_size < 2:
        raise ValueError("batch size must be >=2 for same-batch identity gate")
    scalar_features = {i: features["image"][i].mean(axis=(0,1)) for i in range(len(features["image"]))}
    rows, contexts, audits = [], {}, {"identity": True, "repeat": True, "crn": True, "causal": True,
                                     "batch_full_rel_error_max": 0.0, "prefix_batch_size": batch_size}

    def evaluate(variants):
        result = np.empty((len(variants), len(offsets), nseed))
        descriptors = {spec["label"]: variant_features(cutoff, demo["exec_start_idx"], candidates, spec,
                                                       scalar_features) for spec in variants}
        for oi, offset in enumerate(offsets):
            frame = cutoff + cfg["decision_stride_frames"]*offset
            if offset not in contexts:
                memory = pack_memory(policy, features, memory_frames(cutoff, frame, candidates, specs[0]))
                obs, target, valid_steps = prepare_observation(
                    policy, demo, task, episode, frame, memory,
                    Path(protocol.get("demo_root", str(ECM / "data/val_demos"))))
                # Keep the repeat OUTSIDE the prefix JIT. A B=1 vision/VLM forward broadcast
                # afterward can differ from a real B=4 forward even with FP32/highest precision.
                prefix_obs = jax.tree.map(lambda value: jnp.repeat(value, batch_size, axis=0), obs)
                cache = runtime["prefix"](prefix_obs)
                if cache["prefix_mask"].shape[0] != batch_size or any(
                    value.shape[1] != batch_size for value in jax.tree.leaves(cache["kv_cache"])
                ):
                    raise RuntimeError("prefix cache batch size differs from the full/suffix batch")
                noise, tau = common_random_conditions(task, episode, cutoff, nseed, policy._model.action_horizon,
                                                       policy._model.action_dim, offset=offset)
                contexts[offset] = (obs, target, valid_steps, cache, noise, tau)
            obs, target, valid_steps, cache, noise, tau = contexts[offset]
            memories = [pack_memory(policy, features, memory_frames(cutoff, frame, candidates, spec)) for spec in variants]
            # Order by seed so references and identity use precisely the same batch context.
            jobs = [(vi, si) for si in range(nseed) for vi in range(len(variants))]
            for start in range(0, len(jobs), batch_size):
                chunk = jobs[start:start+batch_size]
                padded = chunk + [chunk[-1]]*(batch_size-len(chunk))
                mem = {name: jnp.asarray(np.stack([memories[v][name] for v,s in padded])) for name in STATIC_KEYS}
                eps = jnp.asarray(np.stack([noise[s] for v,s in padded]))
                times = jnp.asarray(np.array([tau[s] for v,s in padded]))
                out = jax.device_get(runtime["cached"](cache, obs, mem, target, eps, times))
                if not all(np.isfinite(value).all() for value in out.values()):
                    raise ValueError("non-finite teacher loss")
                if start == 0:
                    repeated = jax.device_get(runtime["cached"](cache, obs, mem, target, eps, times))
                    audits["repeat"] &= all(np.array_equal(out[k], repeated[k]) for k in out)
                    full = jax.device_get(runtime["full"](cache, obs, mem, target, eps, times))
                    if not all(np.isfinite(value).all() for value in full.values()):
                        raise ValueError("non-finite full-forward reference loss")
                    for metric in ("loss", "loss_all_dims"):
                        error = float(np.max(np.abs(out[metric]-full[metric]) / np.maximum(np.abs(full[metric]), 1e-8)))
                        audits["batch_full_rel_error_max"] = max(audits["batch_full_rel_error_max"], error)
                    if variants[0]["label"] == "base":
                        # Two identical memories and conditions occupy the SAME batch, even at batch_size=2.
                        identity_mem = {name: jnp.repeat(mem[name][:1], batch_size, axis=0) for name in STATIC_KEYS}
                        identity_eps = jnp.repeat(eps[:1], batch_size, axis=0)
                        identity_tau = jnp.repeat(times[:1], batch_size, axis=0)
                        identity = jax.device_get(runtime["cached"](cache, obs, identity_mem, target, identity_eps, identity_tau))
                        audits["identity"] &= all(np.array_equal(value, np.repeat(value[:1],batch_size)) for value in identity.values())
                if not (audits["identity"] and audits["repeat"] and audits["batch_full_rel_error_max"] <= cfg["gates"]["cached_vs_full_rel_err_max"]):
                    raise RuntimeError(f"V-B correctness gate failed: {audits}")
                for bi, (vi, si) in enumerate(chunk):
                    spec = variants[vi]
                    result[vi, oi, si] = out["loss"][bi]
                    frames = memory_frames(cutoff, frame, candidates, spec)
                    row = {"task": task, "episode_id": episode, "cutoff": cutoff, "offset": offset,
                        "frame": frame, "seed_index": si, "label": spec["label"], "k": spec["k"], "kind": spec["kind"],
                        "selection_half": spec["selection_half"], "pairs_json": json.dumps(spec["pairs"]),
                        "memory_frames_json": json.dumps(frames), "valid_action_steps": valid_steps,
                        "padding_fraction": 1-valid_steps/policy._model.action_horizon,
                        "loss": float(out["loss"][bi]), "loss_all_dims": float(out["loss_all_dims"][bi]),
                        "loss_padded_dims": float(out["loss_padded_dims"][bi]),
                        "noise_sha256": array_hash(noise[si]), "time_sha256": array_hash(np.asarray(tau[si])),
                        "action_sha256": array_hash(np.asarray(target)),
                        **descriptors[spec["label"]]}
                    rows.append(row)
            print(f"{key}: {len(variants)} variants, offset {offset}, {nseed} seeds complete", flush=True)
        return result

    losses = evaluate(specs)
    gains = discounted_gain(losses, losses[0], offsets, cfg["future"]["discount_primary"])
    greedy = [] if single else greedy_variants(specs, gains, sizes=(2,4), count=2)
    if greedy:
        evaluate(greedy)
    # G0e validates every stored row against its offset/seed reference, not just shared seed labels.
    hashes = {}
    for row in rows:
        crn_key = (row["offset"],row["seed_index"])
        value = row["noise_sha256"],row["time_sha256"]
        if crn_key in hashes and hashes[crn_key] != value:
            raise RuntimeError("G0e: unequal noise/time across interventions")
        hashes[crn_key] = value
    for si in range(nseed):
        for oi in range(len(offsets)):
            if losses[0,oi,si] != losses[3,oi,si]:
                raise RuntimeError("identity reference differed across evaluation batches")
    write_parquet(protocol, dest, rows)
    done = {**plan, "offsets": offsets, "n_rows": len(rows), "variants": specs+greedy, "audits": audits,
            "exec_start_idx": demo["exec_start_idx"], "n_frames": len(demo["joint_action"]),
            "parquet_sha256": sha256(dest), "completed": timestamp()}
    done_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(done_path, done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ECM/"configs/memory_management/validation/v_b_teacher_probe.yaml")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reuse-features-from", type=Path,
                        help="Seal checksummed compatible frame features from an earlier run; labels are not reused")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analysis-python", default=shutil.which("python3"))
    parser.add_argument("--demo-root", type=Path,
                        help="root containing <task>/epXX_manifest.json; defaults from demo_source")
    args = parser.parse_args()
    if args.resume:
        if any((args.out, args.tasks, args.episode_ids, args.smoke, args.reuse_features_from, args.demo_root)):
            parser.error("resume accepts no protocol overrides")
        run = args.resume.resolve()
        protocol = json.loads((run/"protocol.json").read_text())
        if protocol["code_sha256"] != source_hashes():
            raise ValueError("code changed since sealing: create a fresh run")
        if sha256(protocol["g0b_report"]) != protocol["g0b_sha256"]:
            raise ValueError("G0b report changed since sealing")
        accepted_gate(Path(protocol["g0b_report"]), Path(protocol["checkpoint_path"]), protocol["primary_dtype"])
    else:
        run, protocol = seal(args)
    subprocess.run([protocol["analysis_python"], "-c", "import pandas, pyarrow, scipy, sklearn, matplotlib"], check=True)
    summary = {"run": str(run), "planned_boundaries": len(protocol["plan"]),
               "excluded_episodes": protocol["excluded_episodes"], "smoke": protocol["smoke"],
               "prefix_batch_size": protocol.get("prefix_batch_size"),
               "reused_feature_episodes": list(protocol.get("feature_cache_imports", {})),
               "candidate_shortfalls": sum(p["candidate_shortfall"]>0 for p in protocol["plan"])}
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return 0
    policy = load_policy(protocol)
    from memory_management.validation.cached_flow_loss import jit_probe_batch, jit_modulation_prefix_cache
    runtime = {"cached": jit_probe_batch(policy._model), "full": jit_probe_batch(policy._model, cached=False),
               "prefix": jit_modulation_prefix_cache(policy._model)}
    previous, demo, features = None, None, None
    for plan in protocol["plan"]:
        current = plan["task"], plan["episode_id"]
        if current != previous:
            demo = load_demo_arrays(*current, Path(protocol.get("demo_root", str(ECM / "data/val_demos"))))
            expected = protocol["manifest_provenance"][f"{current[0]}/{current[1]}"]
            if sha256(demo["manifest"]["manifest_path"]) != expected["manifest_sha256"] or sha256(expected["h5_path"]) != expected["h5_sha256"]:
                raise ValueError("demo content changed since protocol sealing")
            features = episode_features(policy, demo, *current, protocol)
            previous = current
        run_boundary(run, protocol, plan, policy, demo, features, runtime)
    if not protocol["config"].get("m3_single"):
        subprocess.run([protocol["analysis_python"], str(ECM/"scripts/analyze_v_b.py"), str(run)], check=True)
    print(f"Completed V-B run: {run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
