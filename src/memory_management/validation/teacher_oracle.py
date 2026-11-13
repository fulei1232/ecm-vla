"""Closed-loop Teacher Oracle, step 3 (PatternLock pilot): the server-side memory choice.

Protocol: reports/teacher_oracle_pilot_protocol_20261024.md. At each decision the official
FrameSamp selection U (32 frames) is the base. A candidate swaps one history slot of U for one of
`pool_size` frames U dropped; an FP32 copy of the checkpoint scores every candidate by the
flow-matching loss of the expert's next 20 actions under shared noise/time samples. Memory is not
carried across decisions: every candidate differs from that decision's U by one frame.
  U0  official selection; no expert, no scoring.
  S0  official selection; the expert chunk is scored and logged only (shadow).
  TO  the lowest-loss candidate, if its mean paired gain exceeds se_mult standard errors.
  AO  the highest-loss candidate, by the symmetric rule.
  RS  a random candidate at exactly the decisions where the paired TO episode swapped.
Memory ablation (reports/memory_ablation_protocol_20261024.md), no expert and no scoring:
  NH  anchor + recent window of U (history slots removed).
  NM  the current frame only.
Memory budget calibration (reports/memory_budget_calibration_protocol_20261025.md):
  B16, B8  the official even sampling with 16 or 8 frames instead of 32.
Removed slots are right-padded and masked by the official packer. With params["capacity"] = K < 32
(reports/teacher_oracle_budget16_protocol_20261025.md) the base U and its partition are the even
sampling with K frames, and every condition above starts from that U.
Event memory (event_memory.py; needs params["event"]); the event signal is read in every condition:
  EV, EVU, EVR, EVA  closed loop on the event memory, the same sampler without event information, the
                     matched random control, the anti-event memory.
  ES  U; the expert chunk scores every event memory against U and is logged only (shadow).
  TE  the event memory with the largest paired gain over U, if it exceeds se_mult standard errors.
  With event["pool"] = "event" the single-swap pool of S0/TO/AO/RS is drawn by the event density.
Event-keyframe fine-tuning (reports/finetune_kf_protocol_20261025.md; needs params["event"] for the signal
and params["keyframes"]):
  K0, K4, K8, K12  the event-keyframe memory (memory_management.event_keyframes) with the condition's
                   KeyframeConfig, served by the checkpoint fine-tuned with that memory; the server
                   routes each condition to its checkpoint (serve_teacher_policy.py).

Only numpy is imported at module level; JAX code imports lazily inside the policy server.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .acm_copy.diagnostic_logging import PolicyLogger
from . import event_memory as em
from .partition import even_sampling_indices, partition
from .planner_branch import STICK_TASKS
from .teacher_probe import stable_seed
from .. import event_keyframes as ek

CONDITIONS = ("U0", "S0", "TO", "AO", "RS", "NH", "NM", "B16", "B8", "EV", "EVU", "EVR", "EVA", "ES", "TE",
              "K0", "K4", "K8", "K12", "C0")
SCORED = frozenset({"S0", "TO", "AO", "ES", "TE"})
SHADOWS = frozenset({"S0", "ES"})
EVENT = frozenset(em.STRATEGIES)
EVENT_SCORED = frozenset({"ES", "TE"})
# C0: K0's memory served by a reference checkpoint (the stage's "checkpoints" maps every condition to its own)
KEYFRAME = frozenset({"K0", "K4", "K8", "K12", "C0"})
BUDGETS = {"B16": 16, "B8": 8}
ABLATIONS = frozenset({"NH", "NM", *BUDGETS})
TEACHER_KEYS = ("teacher_decision", "teacher_status", "teacher_actions", "teacher_valid_steps")
STATIC_KEYS = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")
TOKENS_PER_FRAME = 16


def loss_dims(task):
    """The stick planner commands the 7 arm joints only; gripper tasks also score the gripper."""
    return 7 if task in STICK_TASKS else 8


def candidate_swaps(task, episode_id, model_seed, decision, step_idx, pool_size, capacity=32, pool=None):
    """U's partition, the candidate pool (random unless given), and every (pool frame, history slot) swap of U."""
    p = partition(step_idx, capacity)
    if not p.full or not p.history:
        return p, [], []
    uniform = set(p.uniform)
    if pool is None:
        allowed = [f for f in range(1, p.recent[0]) if f not in uniform]
        rng = np.random.default_rng(stable_seed(task, episode_id, model_seed, decision, "pool"))
        pool = (sorted(int(f) for f in rng.choice(allowed, size=min(pool_size, len(allowed)), replace=False))
                if allowed else [])
    swaps = [{"frame": f, "slot": slot, "replaced": int(old), "frames": sorted((uniform - {old}) | {f})}
             for f in pool for slot, old in enumerate(p.history)]
    return p, pool, swaps


def paired_gains(losses):
    """losses [1 + V, S], row 0 the base memory: mean and standard error of base - candidate."""
    gains = losses[:1] - losses[1:]
    return gains.mean(axis=1), gains.std(axis=1, ddof=1) / np.sqrt(gains.shape[1])


def extremes(mean, se, se_mult):
    """The TO and AO candidates (indices into the swaps) and whether each passes its rule."""
    best, worst = int(np.argmax(mean)), int(np.argmin(mean))
    return {"TO": (best, bool(mean[best] > se_mult * se[best])),
            "AO": (worst, bool(mean[worst] < -se_mult * se[worst]))}


def ablate(condition, part):
    """The memory of an ablation condition, from the partition of the official selection."""
    if condition == "NH":
        return sorted(set(part.anchor) | set(part.recent))
    if condition == "NM":
        return [int(part.uniform[-1])]
    if condition in BUDGETS:
        return [int(f) for f in even_sampling_indices(part.max_id, BUDGETS[condition])]
    raise ValueError(f"not an ablation condition: {condition!r}")


def event_groups(memories, names, uniform):
    """{memory: [names]} for the distinct named memories that differ from U, in name order."""
    groups = {}
    for name in names:
        if memories[name] != uniform:
            groups.setdefault(tuple(memories[name]), []).append(name)
    return groups


def random_swap(task, episode_id, model_seed, decision, n):
    return int(np.random.default_rng(stable_seed(task, episode_id, model_seed, decision, "rs")).integers(n))


def common_conditions(task, episode_id, model_seed, decision, samples, horizon, action_dim):
    """Shared epsilon and stratified Beta(1.5, 1) times, as cached_flow_loss.common_random_conditions."""
    from scipy.special import betaincinv
    rng = np.random.default_rng(stable_seed(task, episode_id, model_seed, decision, "epsilon"))
    noise = rng.standard_normal((samples, horizon, action_dim), dtype=np.float32)
    probs = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    return noise, betaincinv(1.5, 1.0, probs).astype(np.float32) * 0.999 + 0.001


def expert_target(actions, state):
    """[H, 7 or 8] expert actions -> [H, 8]; a missing gripper column repeats the current state's (unscored)."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[1] == 8:
        return actions.copy()
    if actions.shape[1] != 7:
        raise ValueError(f"expert actions have {actions.shape[1]} dims")
    gripper = np.full((len(actions), 1), np.asarray(state, dtype=np.float32)[7], dtype=np.float32)
    return np.concatenate([actions, gripper], axis=1)


def rs_schedule(rows):
    """Decision indices at which a TO episode swapped, from its oracle_decisions.jsonl rows."""
    return sorted(int(r["decision"]) for r in rows if r.get("swapped"))


# --------------------------------------------------------------------------- FP32 teacher (JAX)

def jit_oracle_batch(model, dims, cached=True):
    """Batch (memory, condition) pairs over one observation; loss over the first `dims` action dims."""
    import jax
    import jax.numpy as jnp
    from .cached_flow_loss import _jit_frozen, compute_flow_loss_cached, compute_flow_loss_explicit

    def apply(current_model, cache, observation, memory, actions, noise, time_):
        batch = noise.shape[0]
        obs = jax.tree.map(lambda v: jnp.broadcast_to(v, (batch, *v.shape[1:])), observation).replace(**memory)
        target = jnp.broadcast_to(actions, (batch, *actions.shape[1:]))
        if cached:
            cache = {"prefix_mask": jnp.broadcast_to(cache["prefix_mask"], (batch, cache["prefix_mask"].shape[1])),
                     "kv_cache": jax.tree.map(lambda v: jnp.broadcast_to(v, (v.shape[0], batch, *v.shape[2:])),
                                              cache["kv_cache"])}
            losses = compute_flow_loss_cached(current_model, cache, obs, target, noise, time_)
        else:
            losses = compute_flow_loss_explicit(current_model, None, obs, target, noise, time_)
        per_dim = losses["loss_per_dim"]
        return {"loss": jnp.mean(per_dim[..., :dims], axis=-1), "loss_8": jnp.mean(per_dim[..., :8], axis=-1)}

    return _jit_frozen(model, apply)


def gather_memory(image, position, state, index, pad=None):
    """Device-side _prepare_frame_sampling: index [B, K] -> [B, K*16, ...] memories, then `pad` (the
    packer's padded tail of a K-frame memory, shared by all rows) appended when K is below the budget."""
    import jax.numpy as jnp
    b, k = index.shape
    memory = {"static_image_emb": image[index].reshape(b, k * TOKENS_PER_FRAME, image.shape[-1]),
              "static_pos_emb": position[index].reshape(b, k * TOKENS_PER_FRAME, position.shape[-1]),
              "static_state_emb": jnp.repeat(state[index], TOKENS_PER_FRAME, axis=1),
              "static_mask": jnp.ones((b, k * TOKENS_PER_FRAME), dtype=bool)}
    if pad is None:
        return memory
    return {key: jnp.concatenate([v, jnp.broadcast_to(pad[key], (b, *pad[key].shape))], axis=1)
            for key, v in memory.items()}


class OracleScorer:
    """FP32 copy of the served checkpoint with its own memory buffer, mirroring every ingest."""

    def __init__(self, policy_config, checkpoint, seed, batch_size, samples, dims, rel_err_max):
        import dataclasses
        import jax
        from mme_vla_suite.policies.policy_config import create_trained_policy
        from mme_vla_suite.training import config as _config
        from .cached_flow_loss import jit_modulation_prefix_cache
        self.jax, self.batch_size, self.samples, self.rel_err_max = jax, batch_size, samples, rel_err_max
        train = _config.get_config(policy_config)
        train = dataclasses.replace(train, model=dataclasses.replace(train.model, dtype="float32"))
        with self.precision():
            self.policy = create_trained_policy(train, Path(checkpoint), seed=seed)
        model = self.policy._model
        self.horizon, self.action_dim = model.action_horizon, model.action_dim
        self.prefix = jit_modulation_prefix_cache(model)
        self.cached = {d: jit_oracle_batch(model, d) for d in dims}
        self.full = {d: jit_oracle_batch(model, d, cached=False) for d in dims}
        self.gather = jax.jit(gather_memory)

    def precision(self):
        # Scoped, so the served BF16 policy keeps its default matmul precision.
        return self.jax.default_matmul_precision("highest")

    def reset(self):
        with self.precision():
            self.policy.reset()

    def add_buffer(self, obs):
        with self.precision():
            first = self.policy.step_idx + 1
            self.policy.add_buffer(obs)
        for idx in range(first, self.policy.step_idx + 1):
            row = self.policy.mem_buffer._history_feats[idx]
            for key in [k for k in row if k not in ("image_emb_4x4", "pos_emb_4x4", "state_emb")]:
                del row[key]

    def pack(self, frames):
        """The official host-side memory for `frames` (pack_memory in run_v_b_teacher_probe.py)."""
        p = self.policy
        image, position, state, mask = p.mem_buffer._prepare_frame_sampling(
            p.mem_buffer._history_feats, frames, p.config.budget, p.config.token_per_image)
        return dict(zip(STATIC_KEYS, (image, position, p._normalize_state(state), mask)))

    def tables(self, n):
        feats, p = self.policy.mem_buffer._history_feats, self.policy
        image = np.stack([np.asarray(feats[i]["image_emb_4x4"]).reshape(TOKENS_PER_FRAME, -1) for i in range(n)])
        position = np.stack([np.asarray(feats[i]["pos_emb_4x4"]).reshape(TOKENS_PER_FRAME, -1) for i in range(n)])
        state = p._normalize_state(np.stack([np.asarray(feats[i]["state_emb"]) for i in range(n)]))
        return tuple(self.jax.numpy.asarray(x) for x in (image, position, state))

    def warmup(self, raw, dims, n=32):
        """Compile prefix, cached and full forwards for n-frame memories before serving (shapes are fixed)."""
        feats = self.policy.mem_buffer._history_feats
        saved = dict(feats)
        try:
            for i in range(n):
                feats[i] = {"image_emb_4x4": np.zeros((1, TOKENS_PER_FRAME, 2048), np.float32),
                            "pos_emb_4x4": np.zeros((1, TOKENS_PER_FRAME, 768), np.float32),
                            "state_emb": np.zeros(8, np.float32)}
            noise = np.zeros((self.samples, self.horizon, self.action_dim), np.float32)
            tau = np.full(self.samples, 0.5, np.float32)
            frames = list(range(n))
            target = np.zeros((self.horizon, 8), np.float32)
            for d in dims:
                self.score(raw, [frames, frames], target, noise, tau, d, n_frames=n)
        finally:
            feats.clear()
            feats.update(saved)

    def score(self, raw, memories, target_actions, noise, tau, dims, n_frames=None):
        """Losses [len(memories), samples]; memories[0] is the base. Audits run on the first batch."""
        jax, jnp = self.jax, self.jax.numpy
        from mme_vla_suite.models.integration.history_observation import HistAugObservation
        timing, audits = {}, {}
        start = time.perf_counter()
        with self.precision():
            n = self.policy.step_idx + 1 if n_frames is None else n_frames
            tables = self.tables(n)
            base = self.pack(memories[0])
            used = len(memories[0]) * TOKENS_PER_FRAME
            pad = ({k: jnp.asarray(base[k][used:]) for k in STATIC_KEYS}
                   if used < len(base["static_mask"]) else None)
            data = self.policy._input_transform({**raw, **base, "actions": target_actions.copy()})
            target = jnp.asarray(data.pop("actions")[None])
            obs = HistAugObservation.from_dict(jax.tree.map(lambda v: jnp.asarray(v)[None], data))
            b = self.batch_size
            cache = self.prefix(jax.tree.map(lambda v: jnp.repeat(v, b, axis=0), obs))
            if cache["prefix_mask"].shape[0] != b or any(v.shape[1] != b for v in jax.tree.leaves(cache["kv_cache"])):
                raise RuntimeError("prefix cache batch size differs from the suffix batch")
            timing["prepare_s"] = time.perf_counter() - start
            index = np.asarray(memories, dtype=np.int32)
            losses = np.full((len(memories), len(tau)), np.nan)
            jobs = [(v, s) for s in range(len(tau)) for v in range(len(memories))]
            for at in range(0, len(jobs), b):
                chunk = jobs[at:at + b]
                padded = chunk + [chunk[-1]] * (b - len(chunk))
                mem = self.gather(*tables, jnp.asarray(index[[v for v, _ in padded]]), pad)
                eps = jnp.asarray(noise[[s for _, s in padded]])
                times = jnp.asarray(tau[[s for _, s in padded]])
                out = jax.device_get(self.cached[dims](cache, obs, mem, target, eps, times))
                if not np.isfinite(out["loss"]).all():
                    raise ValueError("non-finite teacher loss")
                if at == 0:
                    audits = self.audit(cache, obs, mem, target, eps, times, out, base, dims)
                for row, (v, s) in enumerate(chunk):
                    losses[v, s] = out["loss"][row]
        timing["score_s"] = time.perf_counter() - start
        return losses, audits, timing

    def audit(self, cache, obs, mem, target, eps, times, out, base, dims):
        jax, jnp = self.jax, self.jax.numpy
        b = self.batch_size
        again = jax.device_get(self.cached[dims](cache, obs, mem, target, eps, times))
        full = jax.device_get(self.full[dims](cache, obs, mem, target, eps, times))
        first = jax.device_get({k: v[0] for k, v in mem.items()})
        same = {k: jnp.repeat(v[:1], b, axis=0) for k, v in mem.items()}
        identity = jax.device_get(self.cached[dims](cache, obs, same, target, jnp.repeat(eps[:1], b, axis=0),
                                                    jnp.repeat(times[:1], b, axis=0)))
        rel = float(np.max(np.abs(out["loss"] - full["loss"]) / np.maximum(np.abs(full["loss"]), 1e-8)))
        audits = {"repeat": all(np.array_equal(out[k], again[k]) for k in out),
                  "identity": all(np.array_equal(v, np.repeat(v[:1], b)) for v in identity.values()),
                  "cached_vs_full_rel_err": rel,
                  "gather_equals_pack": all(np.array_equal(first[k], np.asarray(jnp.asarray(base[k])))
                                            for k in STATIC_KEYS)}
        audits["passed"] = bool(audits["repeat"] and audits["identity"] and audits["gather_equals_pack"]
                                and rel <= self.rel_err_max)
        return audits


# --------------------------------------------------------------------------- server-side logger

class OracleLogger(PolicyLogger):
    """PolicyLogger whose selector returns the decision's Oracle choice; the policy's RNG is untouched."""

    def __init__(self, policy, context_path, scorer, params):
        self.scorer, self.params = scorer, params
        self.condition, self.pending, self.oracle_log, self.schedule = None, None, {}, set()
        self.event = em.EventMemoryConfig.from_params(params["event"]) if params.get("event") else None
        self.signal = em.make_signal(self.event.signal) if self.event else None
        self.signal_logged = 0
        super().__init__(policy, context_path=context_path)

    def reset(self):
        result = super().reset()
        self.pending, self.oracle_log, self.schedule, self.signal_logged = None, {}, set(), 0
        self.condition = (self.context or {}).get("condition")
        if self.context:
            if self.condition not in CONDITIONS:
                raise ValueError(f"unknown Teacher Oracle condition {self.condition!r}")
            if self.condition in EVENT | EVENT_SCORED | KEYFRAME and self.event is None:
                raise ValueError(f"condition {self.condition} needs params['event']")
            if self.condition in KEYFRAME and self.condition not in (self.params.get("keyframes") or {}):
                raise ValueError(f"condition {self.condition} needs params['keyframes'][{self.condition!r}]")
            self.schedule = set(self.context.get("rs_schedule") or [])
            if self.condition in SCORED:
                self.scorer.reset()
            if self.signal is not None:
                self.signal.reset(self.context)
        return result

    def add_buffer(self, obs):
        first = self.policy.step_idx + 1
        result = super().add_buffer(obs)
        if self.signal is not None and self.context and self.signal.needs_frames:
            images, states = np.asarray(obs["images"]), np.asarray(obs["state"])
            self.signal.observe([em.FrameInput(self.context["task"], first + k, images[k][0], states[k],
                                               first + k < self.policy.exec_start_idx) for k in range(len(images))])
        if self.condition in SCORED:
            self.scorer.add_buffer(obs)
            if self.scorer.policy.step_idx != self.policy.step_idx:
                raise RuntimeError("teacher memory buffer is out of step with the served policy")
        return result

    def infer(self, obs):
        obs = dict(obs)
        teacher = {k: obs.pop(k) for k in TEACHER_KEYS if k in obs}
        if not self.context:
            raise RuntimeError("Teacher Oracle server received a decision without an episode context")
        self.decide(obs, teacher)
        return super().infer(obs)

    def decide(self, obs, teacher):
        p, c, d = self.policy, self.condition, self.decision
        task, episode = self.context["task"], int(self.context["episode_id"])
        step_idx = int(p.step_idx)
        capacity = int(self.params.get("capacity", 32))
        official = [int(i) for i in self.selector_original(step_idx, p.config.budget, p.config.token_per_image)]
        if list(partition(step_idx).uniform) != official:
            raise RuntimeError("partition differs from the official selector")
        distances = memories = event_pool = None
        if self.signal is not None:
            cfg, base = self.event.for_task(task), partition(step_idx, capacity)
            distances = self.signal.distances(step_idx)
            self.log_signal(d, distances)
            memories = em.strategy_memories(base, distances, cfg, stable_seed(task, episode, p._seed, d, "evr"))
            if cfg.pool == "event":
                event_pool = em.event_pool(base, distances, cfg, self.params["pool_size"])
        part, pool, swaps = candidate_swaps(task, episode, p._seed, d, step_idx, self.params["pool_size"], capacity,
                                            event_pool)
        uniform = [int(i) for i in part.uniform]
        log = {"condition": c, "decision": d, "step_idx": step_idx, "exec_start_idx": int(p.exec_start_idx),
               "uniform": uniform, "pool": pool, "n_candidates": len(swaps), "swapped": False, "reason": None}
        if capacity != 32:
            log["capacity"] = capacity
        if self.event is not None:
            log["pool_kind"] = "random" if event_pool is None else "event"
        if c in EVENT_SCORED:
            groups = event_groups(memories, self.event.candidates(), uniform)
            log["n_candidates"] = len(groups)
        chosen = uniform
        if c in SCORED:
            status = str(teacher.get("teacher_status", "missing"))
            if int(teacher.get("teacher_decision", -1)) != d:
                raise RuntimeError(f"teacher payload is for decision {teacher.get('teacher_decision')}, not {d}")
            log.update(expert_status=status, valid_steps=int(teacher.get("teacher_valid_steps", 0)))
            if status != "ok":
                log["reason"] = f"no_expert:{status}"
            elif c in EVENT_SCORED:
                chosen = self.score_events(obs, teacher, task, episode, d, uniform, groups, log)
            elif not swaps:
                log["reason"] = "no_candidates"
            else:
                chosen = self.score(obs, teacher, task, episode, d, uniform, swaps, log)
        elif c == "RS":
            if d in self.schedule:
                if swaps:
                    v = random_swap(task, episode, p._seed, d, len(swaps))
                    chosen = swaps[v]["frames"]
                    log.update(swapped=True, choice={k: swaps[v][k] for k in ("frame", "slot", "replaced")}, index=v)
                else:
                    log["reason"] = "no_candidates"
        elif c in ABLATIONS:
            chosen = ablate(c, part)
            log["ablated"] = chosen
        elif c in EVENT:
            chosen = memories[c]
        elif c in KEYFRAME:
            chosen, log["keyframes"] = ek.select(distances, step_idx, ek.KeyframeConfig(**self.params["keyframes"][c]))
        if memories is not None:
            log["event"] = em.memory_stats(chosen, uniform, distances, cfg.tau)
        self.pending, self.oracle_log = (step_idx, chosen), log
        self.emit("oracle_decisions.jsonl", log)

    def log_signal(self, d, distances):
        new = distances[self.signal_logged:]
        self.emit("event_signal.jsonl", {"decision": d, "first_frame": self.signal_logged,
                                         "distance": [float(v) if np.isfinite(v) else None for v in new]})
        self.signal_logged = len(distances)

    def score_memories(self, obs, teacher, task, episode, d, memories):
        """Losses [len(memories), samples] of the expert chunk; memories[0] is the base. Saved per decision."""
        p = self.policy
        dims = loss_dims(task)
        target = expert_target(teacher["teacher_actions"], obs["observation/state"])
        noise, tau = common_conditions(task, episode, p._seed, d, self.params["mc_samples"],
                                       self.scorer.horizon, self.scorer.action_dim)
        raw = {k: obs[k] for k in ("observation/image", "observation/wrist_image", "observation/state", "prompt")}
        losses, audits, timing = self.scorer.score(raw, memories, target, noise, tau, dims)
        directory = Path(self.context["directory"]) / "oracle"
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / f"decision_{d:04d}.npz", losses=losses, target=target, noise=noise, tau=tau,
                            memories=np.asarray(memories, dtype=np.int32))
        return losses, audits, timing, dims

    def score_events(self, obs, teacher, task, episode, d, uniform, groups, log):
        """ES/TE: score each distinct event memory that differs from U; TE takes the best if it passes."""
        if not groups:
            log["reason"] = "no_candidates"
            return uniform
        keys = list(groups)
        losses, audits, timing, dims = self.score_memories(obs, teacher, task, episode, d,
                                                           [uniform] + [list(k) for k in keys])
        mean, se = paired_gains(losses)
        log.update(audits=audits, timing={k: round(v, 3) for k, v in timing.items()}, loss_dims=dims,
                   base_loss=float(losses[0].mean()),
                   gains={n: {"mean_gain": float(mean[i]), "se": float(se[i])} for i, k in enumerate(keys)
                          for n in groups[k]})
        if not audits["passed"]:
            log["reason"] = "audit_failed"
            return uniform
        if self.condition == "TE":
            v = int(np.argmax(mean))
            if mean[v] > self.params["se_mult"] * se[v]:
                log.update(swapped=True, index=v,
                           choice={"name": groups[keys[v]][0], "mean_gain": float(mean[v]), "se": float(se[v])})
                return list(keys[v])
            log["reason"] = "below_threshold"
        return uniform

    def score(self, obs, teacher, task, episode, d, uniform, swaps, log):
        losses, audits, timing, dims = self.score_memories(obs, teacher, task, episode, d,
                                                           [uniform] + [s["frames"] for s in swaps])
        mean, se = paired_gains(losses)
        ext = extremes(mean, se, self.params["se_mult"])
        log.update(audits=audits, timing={k: round(v, 3) for k, v in timing.items()}, loss_dims=dims,
                   base_loss=float(losses[0].mean()),
                   extremes={k: {"index": v, "accepted": ok, "mean_gain": float(mean[v]), "se": float(se[v]),
                                 **{f: swaps[v][f] for f in ("frame", "slot", "replaced")}}
                             for k, (v, ok) in ext.items()},
                   n_positive_2se=int((mean > self.params["se_mult"] * se).sum()),
                   n_negative_2se=int((mean < -self.params["se_mult"] * se).sum()))
        if not audits["passed"]:
            log["reason"] = "audit_failed"
            return uniform
        if self.condition in ("TO", "AO"):
            v, ok = ext[self.condition]
            if ok:
                log.update(swapped=True, index=v, choice=log["extremes"][self.condition])
                return swaps[v]["frames"]
            log["reason"] = "below_threshold"
        return uniform

    def select(self, step_idx, token_budget, token_per_image):
        if self.pending is None or self.pending[0] != step_idx:
            raise RuntimeError("selection requested without a prepared Oracle decision")
        indices = list(self.pending[1])
        self.pending = None
        if indices != sorted(set(indices)) or len(indices) * token_per_image > token_budget:
            raise ValueError("Invalid selection or token budget")
        if any(i < 0 or i > step_idx for i in indices):
            raise ValueError("Noncausal selection")
        self.selected = indices
        return indices

    def emit(self, name, row):
        if name == "policy_decisions.jsonl":
            row = {**row, "oracle": {k: self.oracle_log.get(k) for k in ("condition", "swapped", "reason", "index")}}
        super().emit(name, row)


def read_jsonl(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
