"""Shared native-render shield and forensic recording; no policy image masks."""
from pathlib import Path

import numpy as np

from v_a_common import atomic_json
from fruit_r9_episode_support import R9Replay
from fruit_r9_data import H5Recorder
from fruit_picking.count_shield_spec import sphere_intersections


def scene_manifest(u):
    import sapien
    actors = {}
    for name, actor in u.scene.actors.items():
        entity = actor._objs[0]
        body = entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        if body is None:
            body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        actors[name] = {'segmentation_id': int(actor.per_scene_id[0]),
                        'collision_shapes': len(body.collision_shapes) if body else 0}
    return {'actors': actors, 'shield_spec': u.count_shield_spec.to_dict() if hasattr(u, 'count_shield_spec') else {'mode':'off'},
            'shield_boxes': getattr(u, 'count_shield_boxes', []),
            'shield_physics': 'Optical geometry only; zero collision shapes; original basket, solver and actions unchanged.'}


def save_forensics(directory, obs, u, stage):
    directory = Path(directory)
    payload = {}; coordinates = {}
    basket = int(u.basket.per_scene_id[0])
    fruit_ids = np.asarray([int(a.per_scene_id[0]) for a in u.fruits])
    for camera, data in obs['sensor_data'].items():
        seg = data['segmentation'][0].cpu().numpy()
        payload[camera+'_segmentation'] = seg
        payload[camera+'_depth'] = data['depth'][0].cpu().numpy()
        mask = seg[..., 0]
        coordinates[camera] = {
            'basket_pixels_yx': np.argwhere(mask == basket).tolist(),
            'basket_fruit_pixels_yx': np.argwhere(np.isin(mask, fruit_ids[u.in_basket()])).tolist(),
            'inactive_reserve_pixels_yx': np.argwhere(np.isin(mask, fruit_ids[~u.present])).tolist()}
    np.savez_compressed(directory/(stage+'_raw_sensors.npz'), **payload)
    atomic_json(directory/(stage+'_pixel_coordinates.json'),
                {'step':u.control_steps, 'coordinates_order':'row(y), column(x)', 'coordinates':coordinates,
                 **scene_manifest(u)})


class ShieldReplay(R9Replay):
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.shield_intersections_checked = 0
        super().__init__(*args, **kwargs)
        atomic_json(self.directory/'scene_manifest.json', scene_manifest(self.u))
        ShieldReplay.last_instance = self

    def fail(self, reason, details=None):
        save_forensics(self.directory, self.obs, self.u, 'check_failure')
        super().fail(reason, details)

    def record(self, phase):
        super().record(phase)
        boxes = getattr(self.u, 'count_shield_boxes', [])
        if boxes:
            ids = np.flatnonzero(self.u.present)
            hit = sphere_intersections(self.u.fruit_positions()[ids],
                [self.u.layout.fruits[int(i)]['radius'] for i in ids], boxes)
            self.shield_intersections_checked += 1
            if np.any(hit):
                self.fail('FRUIT_INTERSECTS_FIXED_SHIELD', {'actors':ids[hit].tolist()})


def install_environment():
    """Route Count only to the explicit new env. Occluded retains its original env."""
    import gymnasium as gym
    import fruit_picking.count_shield_env  # noqa: F401
    original = gym.make

    def make(env_id, *args, **kwargs):
        if env_id == 'FruitPickTaskV2-v0' and kwargs.get('task') == 'count':
            env_id = 'FruitPickCountShield-v0'
            kwargs['count_shield'] = 'fixed_v1'
        return original(env_id, *args, **kwargs)
    gym.make = make


class ShieldRecorder(H5Recorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import json
        self.shield_mode = getattr(self.u, 'count_shield_spec', None)
        self.shield_mode = self.shield_mode.mode if self.shield_mode else 'off'
        self.group['setup'].create_dataset('count_shield_mode', data=self.shield_mode)
        self.group['setup'].create_dataset('count_shield_geometry_json',
                                           data=json.dumps(getattr(self.u, 'count_shield_boxes', [])))

    def summary(self):
        return {**super().summary(), 'count_shield_mode':self.shield_mode}
