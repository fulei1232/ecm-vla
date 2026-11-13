#!/usr/bin/env python3
"""Read-only cross-check of terminal R8 records; never launches or retries a replay."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).absolute().parents[1]
OUT = BASE / 'results/fruit_task_v2_R0_r8_20261105'
TRACKED = BASE / 'reports/fruit_task_v2_R0_r8_20261105'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    state = read(OUT / 'status.json')
    assert state['state'] in {
        'awaiting_R0_review', 'R8_check_failed', 'R8_budget_failed',
        'R8_audit_failed', 'R8_engineering_error'}, 'Wait until the runner has stopped.'
    plan = read(OUT / 'plan.json')
    plan_hash = digest(OUT / 'plan.json')
    for name, expected in plan['source_sha256'].items():
        assert digest(name) == expected, name
    fit = read(OUT / 'clock_fit.json') if (OUT / 'clock_fit.json').exists() else None
    if fit:
        assert digest(OUT / 'clock_fit.json') == state['clock_fit_sha256']
        assert len(fit['source_sha256']) == 10
        for name, expected in fit['source_sha256'].items():
            assert '/calibration/count/history/ep25' in name
            assert digest(name) == expected

    counts = defaultdict(Counter)
    cases = []
    frames = 0
    decisions = 0
    signal_max_error = 0.0
    for validity_path in sorted((OUT / 'replays').glob('*/*/*/ep*/validity.json')):
        directory = validity_path.parent
        proof = read(validity_path)
        assert proof['passed'] and proof['plan_sha256'] == plan_hash
        for name, expected in {**proof['source_sha256'], **proof['image_sha256']}.items():
            assert digest(directory / name) == expected, str(directory / name)
        result = read(directory / 'result.json')
        started = read(directory / 'started.json')
        assert started['plan_sha256'] == plan_hash
        purpose, task, level, episode_dir = directory.relative_to(OUT / 'replays').parts
        assert (result['purpose'], result['task'], result['level']) == (purpose, task, level)
        assert episode_dir == f"ep{result['episode']}"
        assert result['seed'] == 4200000 + 16 * result['episode']
        assert 200 <= result['episode'] < 250 if purpose == 'audit' else 250 <= result['episode'] < 260
        assert result['training_data'] is False
        assert result['budget'] == (1800 if task == 'count' else 1300)
        assert result['guard_sha256'] == plan['guard_sha256']
        if level in ('clock', 'clock_onset'):
            assert fit and result['clock_fit_sha256'] == digest(OUT / 'clock_fit.json')
            assert datetime.fromisoformat(started['utc']) > datetime.fromisoformat(fit['created_utc'])
        actions = read(directory / 'actions.json')
        assert len(actions) == result['steps']
        assert sum(a['guard_clipped'] for a in actions[30:]) == result['guard_clipped_steps']
        assert len(actions) - 30 == result['guard_checked_steps']
        maxima = {cam: dict.fromkeys(('basket', 'basket_fruit', 'inactive_reserve', 'lamp'), 0)
                  for cam in ('front_camera', 'hand_camera')}
        first_entries = []
        seen = set()
        unequal_distance_frames = 0
        replenishments = {e['step']: e for e in result['replenish_events']}
        with (directory / 'trace.jsonl').open() as stream:
            for step, line in enumerate(stream):
                frame = json.loads(line)
                assert frame['step'] == step
                positions = np.asarray(frame['positions'], dtype=float)
                relative = positions - np.array([-.06, -.34, .15])
                inside = (np.abs(relative[:, 0]) < .09) & (np.abs(relative[:, 1]) < .09)
                inside &= (relative[:, 2] > 0) & (relative[:, 2] < .12)
                inside &= np.asarray(frame['present'], dtype=bool)
                assert np.array_equal(inside, frame['inside'])
                hanging = np.asarray(frame['present']) & ~np.asarray(frame['detached'])
                assert np.array_equal(hanging, frame['hanging'])
                for actor in np.flatnonzero(inside & ~np.asarray(frame['wind_detached'])):
                    actor = int(actor)
                    if actor not in seen:
                        first_entries.append({'step': step, 'actor': actor, 'slot': result['slot_of'][actor]})
                        seen.add(actor)
                for camera, values in frame['visibility'].items():
                    for name, value in values.items():
                        maxima[camera][name] = max(maxima[camera][name], value)
                        if name != 'lamp':
                            assert value == 0
                if task == 'count':
                    assert frame['lamp_green'] == (step >= result['green_step'])
                    if step < result['green_step']:
                        assert frame['finish_streak'] == 0
                    if step == 0:
                        assert frame['visibility']['front_camera']['lamp'] >= 16
                if step in replenishments:
                    event = replenishments[step]
                    target = np.asarray(result['slot_geometry'][event['slot']]['pos'])
                    assert np.linalg.norm(np.asarray(frame['tcp_position']) - target) >= .15
                    assert event['delay_steps'] == step - event['ready_step']
                    assert not frame['wind_detached'][event['old_actor']]
                tcp = np.asarray(frame['tcp_position'])
                mature = np.array([result['slot_geometry'][slot]['ripe'] for slot in result['slot_of']])
                for version, allowed in [('mature', hanging & mature), ('any_hanging', hanging)]:
                    expected = min((float(np.linalg.norm(tcp - positions[i]))
                                    for i in np.flatnonzero(allowed)), default=10.0)
                    recorded = frame['signals'][version]
                    error = abs(expected - recorded)
                    signal_max_error = max(signal_max_error, error)
                    assert error < 1e-12
                unequal_distance_frames += abs(frame['signals']['mature'] - frame['signals']['any_hanging']) > 1e-12
                frames += 1
        assert step == result['steps']
        assert maxima == result['max_visibility']
        assert first_entries == result['basket_entries']
        assert all(c['passed'] and c['mask_iou'] >= .98 and c['mean_abs_color_diff_255'] <= 2
                   for c in result['appearance_checks'])
        if task == 'count' and result['success']:
            assert result['outcome'] == 'finish_signal'
            assert result['steps'] < 1800 and result['final_robot_in_basket'] == result['N']
            assert result['appearance_final_hold_available']
        decision_count = len(read(directory / 'decisions.json'))
        decisions += decision_count
        group = counts[purpose, task, level]
        group.update(episodes=1, successes=int(result['success']), decisions=decision_count,
                     checked=result['guard_checked_steps'], clipped=result['guard_clipped_steps'],
                     unequal_distance_frames=unequal_distance_frames, trace_frames=result['steps'] + 1)
        cases.append({'purpose': purpose, 'task': task, 'level': level, 'episode': result['episode'],
                      'success': result['success'], 'outcome': result['outcome'],
                      'result_sha256': digest(directory / 'result.json')})

    summary = read(OUT / 'summary.json')
    for key, cohort in summary['cohorts'].items():
        task, level = key.split('/')
        group = counts['audit', task, level]
        assert group['episodes'] == cohort['episodes']
        assert group['successes'] == cohort['successes']
        assert group['checked'] == cohort['guard_checked_steps']
        assert group['clipped'] == cohort['guard_clipped_steps']
        if group['episodes'] == 50 and level != 'clock_onset':
            high = level == 'history' or task == 'occluded_ripe' and level == 'initial_current'
            passed = group['successes'] >= 45 if high else group['successes'] <= 15
            if level == 'history':
                passed &= group['checked'] > 0 and group['clipped'] / group['checked'] < .005
            assert bool(passed) == cohort['passed']
        else:
            assert cohort['passed'] is None
    assert summary['formal_replays'] == sum(v['episodes'] for k, v in counts.items() if k[0] == 'audit')
    # This preservation check only reads D5/D6/R7 trees and frozen files.
    from fruit_r8_r0_support import preserve
    preserve()
    event_comparison = {}
    for task, item in summary['event_signal_comparison']['tasks'].items():
        mature = {r['episode']: r for r in item['versions']['mature']['records']}
        any_hanging = {r['episode']: r for r in item['versions']['any_hanging']['records']}
        assert mature.keys() == any_hanging.keys()
        event_comparison[task] = {
            'expert_episodes': len(mature),
            'different_event_intervals_episodes': sum(
                mature[e]['event_intervals'] != any_hanging[e]['event_intervals'] for e in mature),
            'unequal_distance_frames': counts['audit', task, 'history']['unequal_distance_frames'],
            'expert_trace_frames': counts['audit', task, 'history']['trace_frames'],
            'interpretation': 'Expert event agreement does not remove the maturity privilege in the original function.'}
    review = {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'terminal_state': state['state'], 'plan_sha256': plan_hash,
              'clock_fit_sha256': digest(OUT / 'clock_fit.json') if fit else None,
              'pinned_source_files_unchanged': len(plan['source_sha256']),
              'valid_cases_checked': len(cases), 'frames_checked': frames,
              'attempts_without_validity': [str(p.parent.relative_to(OUT))
                  for p in sorted((OUT / 'replays').glob('*/*/*/ep*/started.json'))
                  if not (p.parent / 'validity.json').exists()],
              'check_failure_records': {str(p.relative_to(OUT)): digest(p)
                  for p in sorted((OUT / 'replays').glob('*/*/*/ep*/check_failure.json'))},
              'decisions_in_checked_cases': decisions, 'distance_signal_max_abs_error': signal_max_error,
              'expert_signal_comparison': event_comparison,
              'D5_D6_R7_preservation_passed': True, 'groups': {'/'.join(k): dict(v) for k, v in counts.items()},
              'scope': 'Independent record/geometry/signal/hash cross-check; pixels were measured during simulation. '
                       'No new rendering, episode, retry, definition change or reinterpretation of outcomes.',
              'verifier_sha256': digest(__file__), 'cases': cases}
    for directory in (OUT, TRACKED):
        (directory / 'independent_validation.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in review.items() if k not in ('cases', 'groups')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
