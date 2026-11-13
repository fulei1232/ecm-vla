"""Original validity audit, with only the R11 basket-body predicate removed.

The remaining guard, information restriction, action, label and HDF5 checks are
copied verbatim from the frozen R9 audit. R11 metrics receive additional checks.
"""
from fruit_r9_r0_support import (np, read_json, read_jsonl, sha256, GUARD,
                                guard_v2, estimate_onset, choose_count)
from fruit_r11_common import OUT


def audit_unchanged_fields(directory):
    r = read_json(directory / 'result.json');trace = read_jsonl(directory / 'trace.jsonl')
    actions = read_json(directory / 'actions.json');decisions = read_json(directory / 'decisions.json')
    assert len(trace) == len(actions) + 1 == r['steps'] + 1
    assert 30 <= r['steps'] <= r['budget']
    assert r['task_revision'] == 9 and r['distance_signal'] == 'any_hanging'
    if r['task']=='count':
        assert r['fruit_actors']==r['M']*(r['N']+1) and r['reserve_actors']==r['M']*r['N']
    assert not any(v[k] for v in r['max_visibility'].values() for k in ('basket_fruit', 'inactive_reserve'))
    guard = read_json(GUARD)
    for i, a in enumerate(actions):
        assert a['step'] == i + 1
        expected, details = guard_v2(a['proposed'], a['measured_arm_qpos'], guard)
        assert np.array_equal(expected, a['guarded_clean'])
        actual = expected.copy();actual[:7] += a['noise_offset']
        assert np.array_equal(actual, a['executed'])
        assert details['guard_clipped'] == a['guard_clipped']
        if r['purpose'] != 'train_dart':assert not any(a['noise_offset'])
        if i < 30:assert a['phase'] == 'observe' and a['executed'][7] == 1
        if a['phase'] == 'wait_for_green':assert a['executed'] == actions[0]['executed']
    assert r['guard_checked_steps'] == len(actions) - 30
    assert r['guard_clipped_steps'] == sum(a['guard_clipped'] for a in actions[30:])
    if r['purpose'] == 'early_start_probe':
        assert r['outcome'] == 'early_start' and not r['success'] and r['early_start_steps']
        assert actions[30]['phase'] == 'early_start_probe' and 30 < r['steps'] < r['green_step']
    elif r['task'] == 'count':assert not r['early_start_steps']
    any_distances = [f['signals']['any_hanging'] for f in trace]
    lamps = [f['visible']['lamp_green'] for f in trace]
    fits = read_json(OUT / 'clock_fit.json') if r['level'].startswith('clock') else None
    if fits:assert r['clock_fit_sha256'] == sha256(OUT / 'clock_fit.json')
    for decision in decisions:
        p = decision['provided'];t = decision['step'];view = trace[t]['visible'];level = r['level']
        keys = {'tree','ground','lamp_green','N','visible_basket_count','frame0','release_events','t','green_step'}
        if level == 'clock_onset_k0':keys.add('lamp_bracket')
        assert set(p) == keys
        assert p['ground'] == view['ground'] and p['lamp_green'] == view['lamp_green'] and p['visible_basket_count'] == 0
        if not (r['task'] == 'occluded_ripe' and level == 'history'):assert p['tree'] == view['tree']
        assert p['frame0'] == (None if level == 'current' else trace[0]['visible'])
        assert p['release_events'] == (None if level != 'history' else [e for e in r['own_release_events'] if e['step'] <= t])
        assert p['t'] == (t if level.startswith('clock') else None)
        if level == 'clock_onset_k0':
            onset, bracket = estimate_onset(any_distances, lamps, t)
            assert p['green_step'] == onset and p['lamp_bracket'] == bracket
        else:assert p['green_step'] == (r['green_step'] if level == 'clock_onset' else None)
        if r['task'] == 'count':
            clock = fits['absolute' if level == 'clock' else 'relative'] if fits else None
            actual_level = 'nonstop' if level == 'clock_onset_k0' and p['green_step'] is None else level
            expected = choose_count(actual_level, p['N'], p['tree'], p['frame0']['tree'] if p['frame0'] else None,
                len(p['release_events']) if p['release_events'] is not None else None,
                p['visible_basket_count'], len(p['ground']), p['t'], clock, p['green_step'])
            assert expected == decision['chosen_slot']
            assert all(0 <= f['slot'] < r['M'] and f['id'] == f['slot'] for f in p['tree'])
    assert all(c['passed'] and c['mask_iou'] >= .98 and c['mean_abs_color_diff_255'] <= 2 for c in r['appearance_checks'])
    if r['success'] and r['task'] == 'count':
        assert r['outcome'] == 'finish_signal' and r['steps'] < 2800
        assert r['final_robot_in_basket'] == r['N'] and r['appearance_final_hold_available']
    if r['training_data']:
        import h5py
        assert read_json(OUT / 'R0_gate.json')['passed'] and r['data_frames'] == r['steps'] + 1
        assert sha256(directory / r['h5']) == r['h5_sha256'] and sha256(directory / r['gt']) == r['gt_sha256']
        with h5py.File(directory / r['h5']) as f:
            ep = f[f"episode_{r['episode']}"]
            assert int(ep['setup/n_timesteps'][()]) == r['data_frames']
            assert [bool(ep[f'timestep_{i}/info/is_video_demo'][()]) for i in range(r['data_frames'])] == [True]*30+[False]*(r['data_frames']-30)
    return {'passed': True, 'decisions': len(decisions),
        'source_sha256': {name: sha256(directory / name) for name in ('result.json','trace.jsonl','actions.json','decisions.json')},
        'image_sha256': {name: sha256(directory / name) for item in r['snapshots'].values() for name in item['files'].values()}}


def audit_case(directory):
    result = audit_unchanged_fields(directory)
    r = read_json(directory/'result.json')
    metrics = read_json(directory/'case_metrics.json')
    frames = read_jsonl(directory/'frame_metrics.jsonl')
    trace = read_jsonl(directory/'trace.jsonl')
    assert r['instruction_revision'] == 11
    assert r['count_shield_mode'] == ('fixed_v2' if r['task']=='count' else 'off')
    assert metrics['observations'] == len(frames) == r['steps']+1
    assert r['case_metrics_sha256'] == sha256(directory/'case_metrics.json')
    assert r['frame_metrics_sha256'] == sha256(directory/'frame_metrics.jsonl')
    for i, (frame, original) in enumerate(zip(frames, trace)):
        assert frame['step'] == original['step'] == i
        for camera, v in frame['visibility'].items():
            assert v == {k: original['visibility'][camera][k] for k in v}
            assert v['basket_fruit'] == v['inactive_reserve'] == 0
    body = metrics['basket_body']
    assert body['visible_frames_either_camera'] == sum(any(v['basket'] for v in f['visibility'].values()) for f in frames)
    assert body['maximum_single_camera_pixels'] == max(v['basket'] for f in frames for v in f['visibility'].values())
    assert body['total_pixels_both_cameras'] == sum(v['basket'] for f in frames for v in f['visibility'].values())
    gripper = metrics['gripper_shield']
    assert gripper['entered_frames'] == sum(f['gripper']['entered'] for f in frames)
    assert gripper['finger_entered_frames'] == sum(f['gripper']['finger_entered'] for f in frames)
    assert gripper['tcp_entered_frames'] == sum(f['gripper']['tcp_entered'] for f in frames)
    distances = [f['gripper']['minimum_distance_m'] for f in frames if f['gripper']['minimum_distance_m'] is not None]
    assert gripper['minimum_distance_m'] == (min(distances) if distances else None)
    scene = read_json(directory/'scene_manifest.json')
    if r['task']=='count':
        assert len(scene['shield_boxes']) == 14 and len(scene['shield_new_obstacle_audit_boxes']) == 13
        assert scene['actors']['count_fixed_optical_shield']['collision_shapes'] == 0
        assert scene['basket_build_records_audit']['collision_before'] == scene['basket_build_records_audit']['collision_after']
        assert r['shield_geometry_checks'] == r['steps']+1
    else:
        assert not scene['shield_boxes'] and not distances and not gripper['entered_frames']
        assert r['shield_geometry_checks'] == 0
    if r['outcome']=='shield_intersection':
        assert r['training_data'] and not r['success']
        failure = read_json(directory/'attempt_failure.json')
        assert failure['outcome']=='shield_intersection' and failure['step']==r['steps']
        assert r['shield_intersection_failure']['actors'] == failure['actors']
    else:
        assert r['shield_intersection_failure'] is None
    for name in ('case_metrics.json','frame_metrics.jsonl','scene_manifest.json'):
        result['source_sha256'][name] = sha256(directory/name)
    result.update(instruction_revision=11, basket_body_gate=False, hidden_content_zero=True)
    return result
