"""Small real-transformer regression; run in the policy venv with its src on PYTHONPATH.

This checks cache slicing, attention masking, frozen JIT wrappers and memory
modulation without loading the production checkpoint. It does not replace G0b.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
bridge = pytest.importorskip("flax.nnx.bridge")
gemma = pytest.importorskip("mme_vla_suite.models.integration.history_gemma")
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from openpi.models.gemma import Config

from src.memory_management.validation.cached_flow_loss import (
    jit_cached_loss,
    jit_explicit_loss,
    jit_explicit_loss_with_cache,
    jit_modulation_prefix_cache,
    jit_probe_batch,
)


class SmallModulationModel(nnx.Module):
    """One upstream HistoryBlock; suffix width follows MemoryAttention's contract."""
    def __init__(self):
        rngs = nnx.Rngs(17)
        self.integration_type = "modulation"
        self.representation_type = "perceptual"
        self.use_history = True
        self.action_horizon = 3
        configs = [Config(width=w, depth=1, mlp_dim=32, num_heads=4,
                          num_kv_heads=1, head_dim=256) for w in (16, 1024)]
        llm = bridge.ToNNX(gemma.Module(configs=configs, embed_dtype="float32",
                                       integration_type="modulation"))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, False], mem_mods=[False, True])
        self.PaliGemma = nnx.Dict(llm=llm)
        self.action_in_proj = nnx.Linear(32, 1024, rngs=rngs)
        self.memory_proj = nnx.Linear(16, 1024, rngs=rngs)
        self.action_out_proj = nnx.Linear(1024, 32, rngs=rngs)

    def embed_prefix(self, obs):
        # Prefix inputs stay independent of the intervention's memory, as in modulation.
        return (obs.static_pos_emb, obs.tokenized_prompt_mask, jnp.array([True, False, False, False, False]),
                jnp.array([True, True, False, False, False]), None)

    def embed_suffix(self, obs, actions, time):
        tokens = self.action_in_proj(actions) + time[:, None, None]
        return (tokens, jnp.ones(tokens.shape[:2], dtype=bool), jnp.array([True, False, False]),
                jnp.zeros(3, dtype=bool), None)

    def embed_memory(self, obs):
        return self.memory_proj(obs.static_image_emb), obs.static_mask, None, None, None


def test_full_and_cached_forward_with_masked_prefix_and_memory():
    model = SmallModulationModel()
    obs = HistAugObservation(
        images={}, image_masks={}, state=jnp.zeros((1, 8)),
        static_image_emb=jax.random.normal(jax.random.key(21), (1, 5, 16)),
        static_pos_emb=jax.random.normal(jax.random.key(24), (1, 5, 16)),
        tokenized_prompt_mask=jnp.array([[True, True, False, True, True]]),
        static_mask=jnp.array([[True, True, False, True, True]]),
    )
    actions = jax.random.normal(jax.random.key(22), (1, 3, 32))
    noise = jax.random.normal(jax.random.key(23), actions.shape)
    time = jnp.array([0.35])
    complete = jit_explicit_loss(model)(None, obs, actions, noise, time)
    complete_with_cache = jit_explicit_loss_with_cache(model)(None, obs, actions, noise, time)
    prefix = jit_modulation_prefix_cache(model)(obs)
    for leaf in jax.tree.leaves(complete_with_cache["prefix_cache"]["kv_cache"]):
        assert leaf.shape == (1, 1, 5, 1, 256)
    cached = jit_cached_loss(model)
    for cache in (prefix, complete_with_cache["prefix_cache"]):
        actual = cached(cache, obs, actions, noise, time)
        for key in ("loss", "loss_all_dims", "loss_per_dim"):
            np.testing.assert_allclose(actual[key], complete[key], rtol=1e-5, atol=1e-5)
        repeated = cached(cache, obs, actions, noise, time)
        np.testing.assert_array_equal(actual["loss"], repeated["loss"])
    memory = {"static_image_emb": jnp.repeat(obs.static_image_emb, 2, axis=0),
              "static_mask": jnp.repeat(obs.static_mask, 2, axis=0)}
    eps = jnp.repeat(noise, 2, axis=0)
    tau = jnp.repeat(time, 2, axis=0)
    batch = jit_probe_batch(model)(prefix, obs, memory, actions, eps, tau)
    full = jit_probe_batch(model, cached=False)(prefix, obs, memory, actions, eps, tau)
    for key in batch:
        np.testing.assert_allclose(batch[key],full[key],rtol=1e-5,atol=1e-5)
        np.testing.assert_array_equal(batch[key],jnp.repeat(batch[key][:1],2))
    np.testing.assert_allclose(batch["loss"],jnp.repeat(complete["loss"].mean(),2),rtol=1e-5,atol=1e-5)
    # Production prefixes must be encoded at the actual evaluation batch size.
    # Use different memory masks per row, including the hist_off-style reference.
    batch_obs = jax.tree.map(lambda value: jnp.repeat(value, 2, axis=0), obs)
    memory["static_mask"] = memory["static_mask"].at[1, 1:].set(False)
    batch_prefix = jit_modulation_prefix_cache(model)(batch_obs)
    assert batch_prefix["prefix_mask"].shape[0] == 2
    assert all(leaf.shape[1] == 2 for leaf in jax.tree.leaves(batch_prefix["kv_cache"]))
    batch = jit_probe_batch(model)(batch_prefix, obs, memory, actions, eps, tau)
    full = jit_probe_batch(model, cached=False)(batch_prefix, obs, memory, actions, eps, tau)
    for key in batch:
        np.testing.assert_allclose(batch[key], full[key], rtol=1e-5, atol=1e-5)
