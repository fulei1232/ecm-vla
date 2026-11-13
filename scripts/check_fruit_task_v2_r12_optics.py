#!/usr/bin/env python3
"""CPU-only ray check for the R12 false bottom against the archived R11 leak."""
import json
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, atomic_json, sha256, timestamp
sys.path.insert(0, str(ECM))
from fruit_picking.count_shield_r12_spec import CountShieldR12Spec, false_bottom_box, segment_box_interval


SOURCE = ECM/'results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3'
OUT = ECM/'results/fruit_task_v2_R0_r12_20261109/cpu_optics'


def world_points_from_cv_pixels(mask, depth_mm, intrinsic, world_to_camera):
    R = np.asarray(world_to_camera[:, :3], dtype=float)
    t = np.asarray(world_to_camera[:, 3], dtype=float)
    K = np.asarray(intrinsic, dtype=float)
    origin = -R.T @ t
    ys, xs = np.where(mask)
    z = np.asarray(depth_mm[ys, xs], dtype=float) * .001
    camera = np.stack(((xs-K[0, 2]) / K[0, 0] * z,
                       (ys-K[1, 2]) / K[1, 1] * z,
                       z), axis=1)
    targets = (R.T @ (camera - t).T).T
    return origin, targets, ys, xs, z


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    pixel_record = json.loads((SOURCE/'check_failure_pixel_coordinates.json').read_text())
    manifest = json.loads((SOURCE/'scene_manifest.json').read_text())
    camera = np.load(SOURCE/'check_failure_camera_parameters.npz')
    raw = np.load(SOURCE/'check_failure_raw_sensors.npz')

    fruit_name = 'reserve_slot3_generation1'
    fruit_id = manifest['actors'][fruit_name]['segmentation_id']
    hand_seg = raw['hand_camera_segmentation'][..., 0]
    front_seg = raw['front_camera_segmentation'][..., 0]
    hand_depth = raw['hand_camera_depth'][..., 0]
    front_fruit_ids = [manifest['actors'][name]['segmentation_id'] for name in
                       ('fruit_0', 'fruit_1', 'fruit_2', 'fruit_3')]
    pixels = int(np.count_nonzero(hand_seg == fruit_id))
    assert pixels == 127
    assert len(pixel_record['coordinates']['hand_camera']['basket_fruit_pixels_yx']) == pixels
    assert int(np.count_nonzero(front_seg == fruit_id)) == 0

    # Read the frozen basket center from the episode HDF5. This is evidence only;
    # no GPU, simulation reset, or new layout sampling occurs in this check.
    import h5py
    with h5py.File(SOURCE/'demo.h5', 'r') as h5:
        layout_raw = h5['episode_113/setup/layout_json'][()]
        layout_json = layout_raw.decode() if isinstance(layout_raw, bytes) else str(layout_raw)
        layout = json.loads(layout_json)
    center = layout['basket']['center']
    assert np.allclose(center, [-.06, -.34, .15], atol=1e-9, rtol=0)

    old_boxes = manifest['shield_boxes']
    spec = CountShieldR12Spec(mode='fixed_v2_r12')
    plate = false_bottom_box(center, spec)
    assert plate['center'][2] == .310
    overlap = [plate['half'][0]-spec.liner_inner_half_m,
               plate['half'][1]-spec.liner_inner_half_m]
    assert min(overlap) >= .002 - 1e-12

    origin, targets, ys, xs, depths = world_points_from_cv_pixels(
        hand_seg == fruit_id, hand_depth,
        camera['hand_camera_intrinsic_cv'], camera['hand_camera_extrinsic_cv'])
    new_intervals = [segment_box_interval(origin, target, plate) for target in targets]
    old_intervals = [[segment_box_interval(origin, target, box) for box in old_boxes]
                     for target in targets]
    old_blocked = [any(interval is not None and interval[0] < 1-1e-9
                       for interval in intervals) for intervals in old_intervals]
    new_blocked = [interval is not None and interval[0] < 1-1e-9
                   for interval in new_intervals]
    assert not any(old_blocked), f'Old fixed_v2 unexpectedly blocks {sum(old_blocked)} observed leak rays'
    assert all(new_blocked), f'R12 false bottom misses {sum(not x for x in new_blocked)} of {pixels} leak rays'
    entry = np.asarray([interval[0] for interval in new_intervals], dtype=float)
    assert np.all(entry > 0) and np.all(entry < 1)

    summary = {
        'instruction_revision': 12,
        'check': 'CPU-only line-of-sight against the archived ep113 leak frame',
        'source': str(SOURCE),
        'source_sha256': {name: sha256(SOURCE/name) for name in
                          ('check_failure.json', 'check_failure_pixel_coordinates.json',
                           'check_failure_raw_sensors.npz', 'check_failure_camera_parameters.npz',
                           'scene_manifest.json', 'demo.h5')},
        'leaked_actor': fruit_name,
        'hand_camera_visible_fruit_pixels': pixels,
        'depth_m': {'min': float(depths.min()), 'mean': float(depths.mean()), 'max': float(depths.max())},
        'front_camera_visible_pixels_for_actor': int(np.count_nonzero(front_seg == fruit_id)),
        'basket_center': center,
        'old_scheme': {'boxes': len(old_boxes), 'blocked_observed_rays': int(sum(old_blocked)),
                       'observed_rays': pixels, 'leak_reproduced_geometrically': not any(old_blocked)},
        'new_false_bottom': {'box': plate, 'z_m': plate['center'][2],
                             'edge_overlap_m': overlap, 'minimum_edge_overlap_m': min(overlap),
                             'collision_shapes': 0, 'fruit_pass_through': True},
        'new_scheme': {'blocked_observed_rays': int(sum(new_blocked)), 'observed_rays': pixels,
                       'all_leak_rays_occluded': all(new_blocked),
                       'plate_entry_fraction': {'min': float(entry.min()), 'mean': float(entry.mean()),
                                                'max': float(entry.max())}},
        'passed': True,
        'created_utc': timestamp()
    }
    atomic_json(OUT/'result.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
