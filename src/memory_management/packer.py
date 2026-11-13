"""Official frame-major token layout; normalization follows zero padding exactly once."""
import numpy as np
from .config import MemoryConfig


class ManagedMemoryPacker:
    def __init__(self, config: MemoryConfig, normalize_state=None):
        self.config = config
        self.normalize_state = normalize_state

    def pack(self, slots, *, max_frame_id, decision_id):
        cfg = self.config
        slots = sorted(slots, key=lambda s: s.frame_id)
        ids = [s.frame_id for s in slots]
        if not 1 <= len(slots) <= cfg.capacity or len(ids) != len(set(ids)):
            raise ValueError("packer requires 1..capacity unique slots")
        if len({s.frame.episode_id for s in slots}) != 1:
            raise ValueError("mixed episodes")
        if any(s.frame_id > max_frame_id or s.frame.decision_id_available > decision_id for s in slots):
            raise ValueError("future frame or availability boundary")
        image = np.zeros((cfg.token_budget, cfg.image_dim), np.float32)
        position = np.zeros((cfg.token_budget, cfg.position_dim), np.float32)
        state = np.zeros((cfg.token_budget, cfg.state_dim), np.float32)
        mask = np.zeros(cfg.token_budget, bool)
        count = cfg.num_views * cfg.tokens_per_frame
        for index, slot in enumerate(slots):
            f = slot.frame
            if (f.image_feat.shape != (cfg.num_views, cfg.tokens_per_frame, cfg.image_dim)
                    or f.pos_feat.shape != (cfg.num_views, cfg.tokens_per_frame, cfg.position_dim)
                    or f.raw_robot_state.shape != (cfg.state_dim,)):
                raise ValueError("frame does not match the configured tensor contract")
            sl = slice(index * count, (index + 1) * count)
            image[sl] = f.image_feat.reshape(count, cfg.image_dim)
            position[sl] = f.pos_feat.reshape(count, cfg.position_dim)
            state[sl] = f.raw_robot_state
            mask[sl] = True
        if self.normalize_state is not None:
            # Official _prepare_history normalizes padded zeros too. Do not zero them again.
            state = np.asarray(self.normalize_state(state), np.float32)
        return dict(static_image_emb=image, static_pos_emb=position, static_state_emb=state, static_mask=mask)
