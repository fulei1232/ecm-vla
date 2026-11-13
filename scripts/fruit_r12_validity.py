"""R11 audit invariants plus R12 visual-floor and two-cycle-DART checks."""
from fruit_r9_r0_support import np, read_json, read_jsonl, sha256
import fruit_r11_validity as r11
from fruit_r12_common import OUT

r11.OUT = OUT


def audit_case(directory):
    base = r11.audit_unchanged_fields(directory)
    r = read_json(directory/'result.json')
    metrics = read_json(directory/'case_metrics.json')
    frames = read_jsonl(directory/'frame_metrics.jsonl')
    trace = read_jsonl(directory/'trace.jsonl')
    assert r['instruction_revision'] == 12
    assert r['r12_shield_scheme'] == 'new'
    assert r['count_shield_mode'] == ('fixed_v2' if r['task']=='count' else 'off')
    assert metrics['observations'] == len(frames) == r['steps']+1
    assert r['case_metrics_sha256'] == sha256(directory/'case_metrics.json')
    assert r['frame_metrics_sha256'] == sha256(directory/'frame_metrics.jsonl')
    for i, (frame, original) in enumerate(zip(frames, trace)):
        assert frame['step'] == original['step'] == i
        for camera, values in frame['visibility'].items():
            assert values == {k: original['visibility'][camera][k] for k in values}
            assert values['basket_fruit'] == values['inactive_reserve'] == 0
    assert not any(v[k] for v in r['max_visibility'].values()
                   for k in ('basket_fruit', 'inactive_reserve'))

    scene = read_json(directory/'scene_manifest.json')
    if r['task'] == 'count':
        floor = scene['false_bottom']
        assert len(scene['shield_boxes']) == 15
        assert len(scene['shield_new_obstacle_audit_boxes']) == 13
        assert scene['actors']['count_fixed_optical_shield']['collision_shapes'] == 0
        assert scene['actors']['count_fixed_false_bottom']['collision_shapes'] == 0
        assert scene['basket_build_records_audit']['collision_before'] == scene['basket_build_records_audit']['collision_after']
        assert floor['center'] == [-.06, -.34, .31]
        assert floor['half'] == [.092, .092, .001]
        assert r['shield_geometry_checks'] == r['steps']+1
    else:
        assert scene['false_bottom'] is None
        assert not scene['shield_boxes']

    if r['purpose'] == 'train_dart' and r['r12_dart_two_cycle_recipe']:
        selected = set(r['r12_dart_perturbation_cycles'])
        assert len(selected) == 2
        nonzero = set()
        for action in read_json(directory/'actions.json'):
            offset = np.asarray(action['noise_offset'])
            cycle = action['r12_pick_index']
            assert action['r12_dart_perturbed_cycle'] == (cycle in selected)
            if np.any(offset):
                assert cycle in selected
                nonzero.add(cycle)
            else:
                assert not np.any(offset)
        assert nonzero <= selected
        if r['success']:
            assert nonzero == selected
    elif r['purpose'] == 'train_dart':
        assert not r['r12_dart_perturbation_cycles']

    frame_hashes = directory/'front_rgb_sha256.jsonl'
    if r['r12_front_frame_hashes']:
        rows = read_jsonl(frame_hashes)
        assert len(rows) == r['steps']+1
        assert all(x['step'] == i and x['dtype'] == 'uint8' for i,x in enumerate(rows))
    for name in ('case_metrics.json','frame_metrics.jsonl','scene_manifest.json'):
        base['source_sha256'][name] = sha256(directory/name)
    base.update(instruction_revision=12, false_bottom_collision_shapes=0,
                hidden_content_zero=True)
    if frame_hashes.exists():base['front_rgb_sha256_jsonl'] = sha256(frame_hashes)
    return base
