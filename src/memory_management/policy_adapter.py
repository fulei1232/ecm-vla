"""Instance-level policy routing. Upstream model, sampler and baseline code stay untouched."""
from dataclasses import asdict
import numpy as np

from .config import MemoryConfig, ManagerMode
from .types import FrameRecord
from .online_controller import OnlineMemoryController


def pack_frame_memory(buffer, history_feats, frames, token_budget, token_per_image,
                      temporal_index="absolute", position_source=None, packer=None):
    """Shared training/serving packer. Order mode only replaces the position-table index.

    Offline buffers have no table; position_source supplies the already-stored rows
    0..len(frames)-1 from this episode. No new positional formula or image encoding is used.
    """
    if temporal_index not in ("absolute", "order"):
        raise ValueError("temporal_index must be absolute or order")
    packer = packer or buffer._prepare_frame_sampling
    if temporal_index == "absolute":
        return packer(history_feats, frames, token_budget, token_per_image)
    frames = [int(f) for f in frames]
    if frames != sorted(set(frames)):
        raise ValueError("order encoding requires distinct chronological frames")
    side = int(np.sqrt(token_per_image))
    key = f"pos_emb_{side}x{side}"
    table = getattr(buffer, "pos_emb_dict", None)
    if table is None and position_source is None:
        raise ValueError("order encoding needs the original position table or its stored prefix")
    remapped = {}
    for rank, frame in enumerate(frames):
        row = dict(history_feats[frame])
        position = (table[f"{side}x{side}"][rank * buffer.num_views:(rank + 1) * buffer.num_views]
                    if table is not None else position_source[rank][key])
        if np.shape(position) != np.shape(row[key]):
            raise ValueError("position table row has the wrong shape")
        row[key] = position
        remapped[frame] = row
    return packer(remapped, frames, token_budget, token_per_image)


def install_temporal_index(policy, temporal_index="absolute"):
    """Install the shared order packer on this policy and its subsequent reset buffers.

    The default is a no-op, including RNG, features, buffer methods and padding.
    """
    if temporal_index not in ("absolute", "order"):
        raise ValueError("temporal_index must be absolute or order")
    if temporal_index == "absolute":
        return None
    previous = policy._prepare_mem_buffer

    def wrap_buffer():
        buffer = policy.mem_buffer
        original = buffer._prepare_frame_sampling

        def pack(history_feats, frames, token_budget, token_per_image):
            return pack_frame_memory(buffer, history_feats, frames, token_budget, token_per_image,
                                     temporal_index="order", packer=original)
        buffer._prepare_frame_sampling = pack

    def prepare():
        result = previous()
        wrap_buffer()
        return result

    policy._prepare_mem_buffer = prepare
    wrap_buffer()
    return previous


class SourcePositionEncoder:
    """Upstream frequencies/spatial grid, with a bounded on-demand temporal computation."""
    def __init__(self, dim):
        import jax
        import jax.numpy as jnp
        from mme_vla_suite.shared.posemb_3d import PosEmb3D
        official = PosEmb3D(dim)
        temporal, spatial = official.temporal_omega, official.spatial_pe4x4

        def encode(indices):
            values = jnp.einsum("m,d->md", indices, temporal)
            times = jnp.concatenate((jnp.sin(values), jnp.cos(values)), axis=-1)
            return jnp.concatenate((jnp.broadcast_to(times[:, None], (len(indices), 16, times.shape[-1])),
                                    jnp.broadcast_to(spatial, (len(indices), *spatial.shape))), axis=-1)
        self._encode = jax.jit(encode)

    def __call__(self, indices):
        import jax.numpy as jnp
        return np.asarray(self._encode(jnp.asarray(indices, dtype=jnp.int32)))


class ManagedFeatureBuffer:
    def __init__(self, policy, config, mode, seed, candidate_generator=None):
        self.config = config
        self.controller = OnlineMemoryController(config, mode, seed, policy._normalize_state,
                                                 candidate_generator=candidate_generator)
        self.vision_enc = policy._vision_encode
        self.position_encoder = SourcePositionEncoder(config.position_dim)

    @property
    def _history_feats(self):
        # Compatibility for upstream infer's nonempty assertion; no backing full-history cache.
        frames = {f.frame_id: f for f in self.controller.rolling.snapshot()}
        frames.update({s.frame_id: s.frame for s in self.controller.bank.snapshot()})
        return {i: {"image_emb_4x4": f.image_feat, "pos_emb_4x4": f.pos_feat,
                    "state_emb": f.raw_robot_state} for i, f in frames.items()}

    def add_buffer(self, images, states, step_idx_list):
        import jax
        import jax.numpy as jnp
        import einops
        from openpi.shared import image_tools
        from mme_vla_suite.shared.data_utils import pool_tokens_to_size
        cfg = self.config
        images, states = np.asarray(images), np.asarray(states)
        if (images.ndim != 5 or images.shape[1] != cfg.num_views or images.shape[-1] != 3
                or states.shape != (len(images), cfg.state_dim) or len(step_idx_list) != len(images)):
            raise ValueError("invalid streaming image/state layout")
        if not np.isfinite(states).all():
            raise ValueError("nonfinite raw robot state")
        first = self.controller.rolling.latest_frame_id + 1
        if list(step_idx_list) != list(range(first, first + len(images))):
            raise ValueError("noncontiguous policy ingestion")
        for start in range(0, len(images), cfg.encoding_batch):
            end = min(start + cfg.encoding_batch, len(images))
            # Match official normalization, padding and mean spatial pooling.
            block = jnp.array(images[start:end].astype(np.float32) / 255.0 * 2.0 - 1.0)
            t, v = block.shape[:2]
            block = einops.rearrange(block, "t v h w c -> (t v) h w c")
            block = image_tools.resize_with_pad(block, 224, 224)
            block = einops.rearrange(block, "(t v) h w c -> t v h w c", t=t, v=v)
            features = np.asarray(jax.device_get(pool_tokens_to_size(self.vision_enc(block), 16)))
            ids = step_idx_list[start:end]
            positions = self.position_encoder(ids)
            for offset, frame_id in enumerate(ids):
                self.controller.ingest(FrameRecord(
                    self.controller.rolling.episode_id, frame_id, self.controller.decision_id, None,
                    features[offset], positions[offset][None], states[start + offset]))


class ManagedPolicyAdapter:
    def __init__(self, policy, mode, config=None, context=None, emit=None, candidate_generator=None):
        self.policy, self.mode = policy, ManagerMode(mode)
        self.config = config or MemoryConfig()
        self.context, self.emit = context or (lambda: {}), emit or (lambda _: None)
        self.candidate_generator = candidate_generator
        self.original = {name: getattr(policy, name) for name in
                         ("reset", "add_buffer", "_prepare_history", "_prepare_mem_buffer")}
        self.episode_number = 0
        cfg = policy.config
        if (cfg.representation_type != "perceptual" or cfg.perceptual_memory.type != "frame_sampling"
                or cfg.num_views != self.config.num_views or cfg.budget != self.config.token_budget
                or cfg.token_per_image != self.config.tokens_per_frame or self.config.num_views != 1
                or self.config.tokens_per_frame != 16):
            raise ValueError("adapter supports the official single-view 4x4 perceptual frame sampler")
        policy._prepare_mem_buffer = self.make_buffer
        policy.reset, policy.add_buffer, policy._prepare_history = self.reset, self.add_buffer, self.prepare_history
        self.make_buffer()  # Release the already-created upstream full-history buffer and position tables.

    def make_buffer(self):
        self.policy.mem_buffer = ManagedFeatureBuffer(self.policy, self.config, self.mode, self.policy._seed,
                                                      self.candidate_generator)
        self.initialized = False

    def reset(self):
        result = self.original["reset"]()
        self.episode_number += 1
        self.episode_context = self.context()
        return result

    def add_buffer(self, obs):
        policy = self.policy
        # The official websocket client sends exec_start_idx only on the first
        # demonstration packet; later packets may carry zero. Zero means
        # "unchanged/default" once the episode has been initialized, not a new
        # execution boundary.
        supplied_boundary = int(obs.get("exec_start_idx", 0))
        boundary = supplied_boundary if supplied_boundary > 0 else int(policy.exec_start_idx)
        controller = policy.mem_buffer.controller
        if not self.initialized:
            context = self.episode_context if hasattr(self, "episode_context") else self.context()
            episode = str(context.get("episode_key", f"episode-{self.episode_number}"))
            controller.reset(episode, boundary)
            policy.exec_start_idx = boundary
            self.initialized = True
        elif supplied_boundary > 0 and boundary != policy.exec_start_idx:
            raise ValueError("execution boundary changed within the episode")
        return self.original["add_buffer"](obs)

    def prepare_history(self, inputs):
        controller = self.policy.mem_buffer.controller
        packed = controller.prepare()
        inputs.update(packed)
        stats = controller.stats()
        self.emit({"schema": "m1-memory-decision-v1", "mode": self.mode.value,
                   "decision_idx": controller.decision_id - 1, "available_max_frame_id": self.policy.step_idx,
                   "exec_start_idx": self.policy.exec_start_idx,
                   "config": asdict(self.config), **stats,
                   "tensor_shapes": {k: list(v.shape) for k, v in packed.items()},
                   "tensor_dtypes": {k: str(v.dtype) for k, v in packed.items()},
                   "valid_tokens": int(packed["static_mask"].sum()), "source_position_frame_ids": stats["selected_frame_ids"]})
        return inputs


def install_memory(policy, mode=ManagerMode.BASELINE, config=None, context=None, emit=None,
                   candidate_generator=None):
    """The baseline flag performs no mutation and consumes no RNG."""
    mode = ManagerMode(mode)
    if mode == ManagerMode.BASELINE:
        return None
    return ManagedPolicyAdapter(policy, mode, config, context, emit, candidate_generator)
