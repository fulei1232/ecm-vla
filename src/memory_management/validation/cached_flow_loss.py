"""Explicit-noise Flow Matching loss for fair V-B memory comparisons.

The first implementation deliberately uses the model's full forward path. It is
the correctness reference for the later prefix-cache optimization.
"""
from __future__ import annotations

import numpy as np


def draw_explicit_conditions(jax, actions, rng):
    """Match the random tensors produced by the upstream compute_loss split."""
    _, noise_rng, time_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    return noise, time


def compute_flow_loss_explicit(
    model,
    preprocess_rng,
    observation,
    actions,
    noise,
    time,
    *,
    action_dims: int = 8,
    train: bool = False,
    memory_tokens=None,
    memory_mask=None,
    return_prefix_cache: bool = False,
):
    """Return true-action-dimension and full-dimension losses using fixed noise/time."""
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask, preprocess_observation

    observation = preprocess_observation(preprocess_rng, observation, train=train)
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    target_velocity = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )

    if model.integration_type == "expert":
        mem_tokens, mem_mask, mem_ar_mask, mem_na_mask, _ = model.embed_memory(observation)
        input_mask = jnp.concatenate([mem_mask, prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([jnp.asarray(mem_ar_mask), prefix_ar_mask, suffix_ar_mask], axis=0)
        na_mask = jnp.concatenate([jnp.asarray(mem_na_mask), prefix_na_mask, suffix_na_mask], axis=0)
    else:
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)

    if model.use_history and model.representation_type != "symbolic":
        attention_mask = make_attn_mask(input_mask, ar_mask, na_mask)
    else:
        attention_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    if model.integration_type == "expert":
        (_, _, suffix_out), kv_cache = model.PaliGemma.llm(
            [mem_tokens, prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, None, adarms_cond],
        )
    elif model.integration_type == "modulation":
        if memory_tokens is None:
            mem_seq, mem_mask, _, _, _ = model.embed_memory(observation)
        else:
            if memory_mask is None:
                raise ValueError("memory_mask is required with precomputed memory_tokens")
            mem_seq, mem_mask = memory_tokens, memory_mask
        (_, suffix_out), kv_cache = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            mem_seq=[None, mem_seq],
            mem_mask=[None, mem_mask],
        )
    else:
        (_, suffix_out), kv_cache = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attention_mask, positions=positions,
            adarms_cond=[None, adarms_cond]
        )

    predicted_velocity = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    squared_error = jnp.square(predicted_velocity - target_velocity)
    if not 0 < action_dims <= squared_error.shape[-1]:
        raise ValueError(f"action_dims={action_dims} is incompatible with {squared_error.shape[-1]} outputs")
    result = {
        "loss": jnp.mean(squared_error[..., :action_dims], axis=-1),
        "loss_all_dims": jnp.mean(squared_error, axis=-1),
        "loss_per_dim": jnp.mean(squared_error, axis=-2),
    }
    if return_prefix_cache:
        import jax

        if model.integration_type != "modulation":
            raise ValueError("full-forward prefix cache diagnostics require modulation")
        result["prefix_cache"] = {
            "prefix_mask": prefix_mask,
            # Upstream KV layout is [layers, batch, tokens, kv_heads, head_dim].
            "kv_cache": jax.tree.map(lambda value: value[:, :, :prefix_mask.shape[1]], kv_cache),
        }
    return result


def prepare_modulation_prefix_cache(model, observation):
    """Cache the VLM prefix once; modulation memory is intentionally excluded."""
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask, preprocess_observation

    if model.integration_type != "modulation":
        raise ValueError("prefix KV caching is currently validated only for modulation models")
    observation = preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = model.embed_prefix(observation)
    if model.use_history and model.representation_type != "symbolic":
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
    else:
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return {"prefix_mask": prefix_mask, "kv_cache": kv_cache}


def _jit_frozen(model, function):
    """Compile a forward function with frozen NNX state passed as an argument."""
    import flax.nnx as nnx
    import jax

    graphdef, state = nnx.split(model)

    def apply(model_state, *args):
        current_model = nnx.merge(graphdef, model_state)
        return function(current_model, *args)

    compiled = jax.jit(apply)
    return lambda *args: compiled(state, *args)


def jit_modulation_prefix_cache(model):
    """Encode the prefix under JIT, as in the complete forward reference."""
    return _jit_frozen(model, prepare_modulation_prefix_cache)


def jit_explicit_loss(model):
    """JIT the complete explicit-noise reference, retaining all loss metrics."""
    return _jit_frozen(model, compute_flow_loss_explicit)


def jit_explicit_loss_with_cache(model):
    """Diagnostic: return the full forward's own prefix KV for suffix isolation."""
    def apply(current_model, preprocess_rng, observation, actions, noise, time):
        return compute_flow_loss_explicit(
            current_model, preprocess_rng, observation, actions, noise, time,
            return_prefix_cache=True,
        )

    return _jit_frozen(model, apply)


def jit_seeded_explicit_loss(model):
    """Diagnostic reference: draw conditions inside JIT exactly as compute_loss does."""
    import jax

    def apply(current_model, rng, observation, actions):
        preprocess_rng, _, _ = jax.random.split(rng, 3)
        noise, time = draw_explicit_conditions(jax, actions, rng)
        return compute_flow_loss_explicit(
            current_model, preprocess_rng, observation, actions, noise, time
        )

    return _jit_frozen(model, apply)


def jit_cached_loss(model):
    """JIT suffix-only loss, retaining real-dimension and padded-dimension metrics."""
    return _jit_frozen(model, compute_flow_loss_cached)


def compute_flow_loss_cached(model, prefix_cache, observation, actions, noise, time, *, action_dims=8):
    """Compute explicit Flow Matching loss using a previously encoded VLM prefix."""
    import einops
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

    if model.integration_type != "modulation":
        raise ValueError("cached loss is currently implemented only for modulation models")
    prefix_mask, kv_cache = prefix_cache["prefix_mask"], prefix_cache["kv_cache"]
    time_expanded = time[..., None, None]
    noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
    target_velocity = noise - actions
    suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = model.embed_suffix(
        observation, noisy_actions, time
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    mem_seq, mem_mask, _, _, _ = model.embed_memory(observation)
    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
        mem_seq=[None, mem_seq],
        mem_mask=[None, mem_mask],
    )
    predicted_velocity = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    squared_error = jnp.square(predicted_velocity - target_velocity)
    if not 0 < action_dims <= squared_error.shape[-1]:
        raise ValueError(f"action_dims={action_dims} is incompatible with {squared_error.shape[-1]} outputs")
    return {
        "loss": jnp.mean(squared_error[..., :action_dims], axis=-1),
        "loss_all_dims": jnp.mean(squared_error, axis=-1),
        "loss_per_dim": jnp.mean(squared_error, axis=-2),
    }


def common_random_conditions(task: str, episode_id: int, boundary: int, samples: int,
                             horizon: int, action_dim: int, *, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create repeatable shared epsilon and stratified Beta(1.5, 1) time samples."""
    if samples <= 0 or horizon <= 0 or action_dim <= 0:
        raise ValueError("sample count and action dimensions must be positive")
    from scipy.special import betaincinv
    from .teacher_probe import stable_seed

    eps_rng = np.random.default_rng(stable_seed(task, episode_id, boundary, offset, "epsilon"))
    noise = eps_rng.standard_normal((samples, horizon, action_dim), dtype=np.float32)
    probs = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    tau = betaincinv(1.5, 1.0, probs).astype(np.float32) * 0.999 + 0.001
    return noise, tau


def jit_probe_batch(model, *, cached: bool = True):
    """Batch independent memory/seed pairs, broadcasting a single observation and prefix KV."""
    import jax
    import jax.numpy as jnp

    def apply(current_model, cache, observation, memory, actions, noise, time):
        batch_size = noise.shape[0]
        obs = jax.tree.map(lambda value: jnp.broadcast_to(value, (batch_size, *value.shape[1:])), observation)
        obs = obs.replace(**memory)
        target = jnp.broadcast_to(actions, (batch_size, *actions.shape[1:]))
        if cached:
            batched_cache = {
                "prefix_mask": jnp.broadcast_to(cache["prefix_mask"], (batch_size, cache["prefix_mask"].shape[1])),
                "kv_cache": jax.tree.map(
                    lambda value: jnp.broadcast_to(value, (value.shape[0], batch_size, *value.shape[2:])),
                    cache["kv_cache"],
                ),
            }
            losses = compute_flow_loss_cached(current_model, batched_cache, obs, target, noise, time)
        else:
            losses = compute_flow_loss_explicit(current_model, None, obs, target, noise, time)
        # Only scalar metrics leave the GPU; loss_per_dim includes the unused padded dimensions.
        return {
            "loss": jnp.mean(losses["loss"], axis=-1),
            "loss_all_dims": jnp.mean(losses["loss_all_dims"], axis=-1),
            "loss_padded_dims": jnp.mean(losses["loss_per_dim"][..., 8:], axis=-1),
        }

    return _jit_frozen(model, apply)
