"""R12 replay additions: front-camera hashes and two noisy pick cycles for data."""
import hashlib
import json

import numpy as np

from v_a_common import atomic_json
from fruit_r11_runtime import Revision11Replay


class Revision12Replay(Revision11Replay):
    last_instance = None

    def __init__(self, *args, **kwargs):
        env = args[0]
        directory = args[2]
        self.capture_front_hashes = bool(getattr(env, 'r12_capture_front_hashes', False))
        self.front_hash_log = (directory/'front_rgb_sha256.jsonl').open('x') if self.capture_front_hashes else None
        self.r12_two_cycle_dart = bool(getattr(env, 'r12_two_cycle_dart', False))
        self.r12_noise_seed = getattr(env, 'r12_noise_seed', None)
        self.r12_layout_seed = getattr(env, 'r12_layout_seed', None)
        self.r12_dart_perturbation_cycles = []
        if self.r12_two_cycle_dart:
            n = int(env.unwrapped.layout.n_pick)
            if n < 2 or self.r12_noise_seed is None:
                raise ValueError('R12 DART needs N>=2 and an attempt-specific noise seed')
            self.r12_dart_perturbation_cycles = sorted(int(x) for x in
                np.random.default_rng([int(self.r12_noise_seed), 12]).choice(n, size=2, replace=False))
        Revision12Replay.last_instance = self
        super().__init__(*args, **kwargs)
        from fruit_shield_v2_runtime import scene_manifest
        manifest = scene_manifest(self.u)
        manifest.update(instruction_revision=12,
            count_observation_definition='fixed_v2 plus opaque visual-only false bottom at z=.310 m',
            false_bottom=getattr(self.u, 'count_shield_false_bottom_box', None),
            false_bottom_spec=getattr(self.u, 'count_shield_false_bottom_spec', None),
            false_bottom_collision_shapes=(0 if hasattr(self.u, 'count_shield_false_bottom_actor') else None))
        atomic_json(directory/'scene_manifest.json', manifest)

    def record(self, phase):
        super().record(phase)
        if self.front_hash_log is not None:
            rgb = self.obs['sensor_data']['front_camera']['rgb'][0].cpu().numpy()
            row = {'step': int(self.u.control_steps), 'shape': list(rgb.shape),
                   'dtype': str(rgb.dtype), 'sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
            self.front_hash_log.write(json.dumps(row, separators=(',', ':'))+'\n')
            self.front_hash_log.flush()

    def step(self, action, labels):
        if not self.r12_two_cycle_dart:
            return super().step(action, labels)
        cycle = labels.get('pick_index')
        perturbed = cycle in self.r12_dart_perturbation_cycles
        before_step = self.u.control_steps
        if not perturbed and self.noise is not None:
            noise = self.noise
            self.noise = None
            try:
                return super().step(action, labels)
            finally:
                self.noise = noise
                noise[:] = 0.
                self._annotate_action(before_step, cycle, perturbed)
        try:
            return super().step(action, labels)
        finally:
            self._annotate_action(before_step, cycle, perturbed)

    def _annotate_action(self, step, cycle, perturbed):
        # SeamReplay records the post-step control count, one greater than
        # the pre-step count captured above.
        if self.actions and self.actions[-1].get('step') == step + 1:
            self.actions[-1]['r12_pick_index'] = cycle
            self.actions[-1]['r12_dart_perturbed_cycle'] = bool(perturbed)

    def close_r12_logs(self):
        if self.front_hash_log is not None and not self.front_hash_log.closed:
            self.front_hash_log.close()
