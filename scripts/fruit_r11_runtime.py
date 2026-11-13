"""Revision 11 replay policy, preserving the accepted fixed_v2 constructor."""
import numpy as np

from v_a_common import atomic_json
from fruit_r9_episode_support import R9Replay, EpisodeEnd
from fruit_shield_v2_runtime import SeamReplay, save_forensics
from fruit_picking.count_shield_spec import sphere_intersections
from fruit_r11_metrics import FrameMetrics, GripperDistance
from fruit_r11_rules import hidden_content_visible


class Revision11Replay(SeamReplay):
    last_instance = None

    def __init__(self, *args, **kwargs):
        env, _, directory = args[:3]
        self.is_data = any(p in ('train_clean', 'train_dart') for p in directory.parts)
        self.metrics = FrameMetrics(directory, GripperDistance(
            getattr(env.unwrapped, 'count_shield_boxes', []), env.unwrapped.agent.urdf_path))
        self.intersection_failure = None
        Revision11Replay.last_instance = self
        super().__init__(*args, **kwargs)
        from fruit_shield_v2_runtime import scene_manifest
        manifest = scene_manifest(self.u)
        manifest.update(instruction_revision=11, count_observation_definition='fixed_v2 required for R0/data/train/eval',
                        basket_body_visibility='report_only', basket_content_visibility='zero_pixels_fatal')
        atomic_json(self.directory/'scene_manifest.json', manifest)

    def visibility(self):
        u = self.u
        inside, physical = u.in_basket(), u.present
        return {camera: {'basket': counts.get(int(u.basket.per_scene_id[0]), 0),
                         'basket_fruit': sum(counts.get(int(u.fruits[int(i)].per_scene_id[0]), 0) for i in np.flatnonzero(inside)),
                         'inactive_reserve': sum(counts.get(int(u.fruits[int(i)].per_scene_id[0]), 0) for i in np.flatnonzero(~physical))}
                for camera, counts in self.counts.items()}

    def fail(self, reason, details=None):
        if reason == 'BASKET_OR_CONTENT_VISIBLE' and not hidden_content_visible(details):
            return  # Body pixels remain in raw traces and are reported separately.
        self.metrics.observe(self, self.visibility())
        save_forensics(self.directory, self.obs, self.u, 'check_failure')
        return R9Replay.fail(self, reason, details)

    def record(self, phase):
        R9Replay.record(self, phase)
        self.metrics.observe(self, self.visibility())
        boxes = getattr(self.u, 'count_shield_audit_boxes', [])
        if not boxes:
            return
        ids = np.flatnonzero(self.u.present)
        hit = sphere_intersections(self.u.fruit_positions()[ids],
            [self.u.layout.fruits[int(i)]['radius'] for i in ids], boxes)
        self.shield_intersections_checked += 1
        if not np.any(hit):
            return
        details = {'actors': ids[hit].tolist(), 'step': self.u.control_steps, 'phase': phase}
        if not self.is_data:
            self.fail('FRUIT_INTERSECTS_FIXED_SHIELD', details)
        self.intersection_failure = details
        self.capture('shield_intersection')
        save_forensics(self.directory, self.obs, self.u, 'shield_intersection')
        atomic_json(self.directory/'attempt_failure.json', {'outcome': 'shield_intersection', **details})
        self.ended, self.success, self.reason = True, False, 'shield_intersection'
        raise EpisodeEnd()
