"""Predeclared R9 information audit; no outcome-dependent selection or fitting."""
import numpy as np

from memory_management.event_keyframes import KeyframeConfig, detect_events, select


MEMORIES = {
    'K0': KeyframeConfig(frames=16, events=0),
    'K8': KeyframeConfig(frames=16, events=8),
    'KA': KeyframeConfig(frames=16, events=12, per_event='release'),
}
BANDS = {'all': (2, 10), '2-4': (2, 4), '5-7': (5, 7), '8-10': (8, 10)}


def memory_frames(distances, step, condition):
    if condition == 'recent16':return list(range(max(0, step - 15), step + 1))
    return select(distances, step, MEMORIES[condition])[0]


def estimate_onset(distances, lamp_observations, step):
    """Only K0's currently selected, visually readable lamps; no onset truth or cache."""
    frames = memory_frames(distances, step, 'K0')
    red = [t for t in frames if lamp_observations[t] is False]
    green = [t for t in frames if lamp_observations[t] is True]
    last_red, first_green = (max(red) if red else None), (min(green) if green else None)
    midpoint = (last_red + first_green) / 2 if last_red is not None and first_green is not None else None
    return midpoint, {'frame_ids': frames, 'visible_lamps': [lamp_observations[t] for t in frames],
                      'last_red': last_red, 'first_green': first_green}


def episode_information(result, trace, decisions):
    """Every attempt remains in denominators; a missing end decision cannot pass coverage."""
    N = result['N']
    finish = [d['step'] for d in decisions if d['purpose'] == 'pick_or_finish'
              and d['provided']['release_events'] is not None
              and len(d['provided']['release_events']) == N]
    end = min(finish) if finish else None
    distances = [f['signals']['any_hanging'] for f in trace]
    events = detect_events(distances[:end + 1], .03, .005) if end is not None else []
    intervals = []
    if end is not None:
        for replenish in result['replenish_events']:
            if replenish['step'] > end:continue
            actor = replenish['old_actor']
            detached = [f['step'] for f in trace[:end + 1]
                        if f['present'][actor] and f['detached'][actor] and not f['wind_detached'][actor]]
            detach = min(detached) if detached else None
            # The detach changes hanging membership, so its frame is usually the first far/release frame.
            containing = [e for e in events if e['release'] is not None
                          and detach is not None and e['start'] <= detach <= e['release']]
            event = containing[-1] if containing else None
            intervals.append({'actor': actor, 'slot': replenish['slot'], 'detach_step': detach,
                              'replenish_step': replenish['step'],
                              'near_start': event['start'] if event else None,
                              'release_frame': event['release'] if event else None})
    information = {}
    for condition in (*MEMORIES, 'recent16'):
        selected = memory_frames(distances, end, condition) if end is not None else []
        broad = [i['near_start'] is not None and any(i['near_start'] <= t <= i['replenish_step'] for t in selected)
                 for i in intervals]
        strict = [i['detach_step'] is not None and any(i['detach_step'] <= t <= i['replenish_step'] for t in selected)
                  for i in intervals]
        information[condition] = {'frames': selected, 'broad_covered': broad, 'strict_covered': strict,
            'all_broad': bool(len(intervals) == N and all(broad)),
            'all_strict': bool(len(intervals) == N and all(strict))}
    return {'episode': result['episode'], 'N': N, 'expert_success': result['success'],
            'end_decision_step': end, 'evidence_intervals': intervals, 'conditions': information}


def information_summary(episodes, gate=True):
    stats = {}
    for condition in (*MEMORIES, 'recent16'):
        stats[condition] = {}
        for band, (lower, upper) in BANDS.items():
            rows = [r for r in episodes if lower <= r['N'] <= upper]
            n = len(rows)
            broad = sum(r['conditions'][condition]['all_broad'] for r in rows)
            strict = sum(r['conditions'][condition]['all_strict'] for r in rows)
            stats[condition][band] = {'episodes': n, 'all_broad': broad, 'all_strict': strict,
                'broad_fraction': broad / n if n else None, 'strict_fraction': strict / n if n else None}
    criteria = {}
    if gate:
        for band, threshold in [('all', .95), ('2-4', .90), ('5-7', .90), ('8-10', .90)]:
            value = stats['KA'][band]['broad_fraction']
            criteria['KA/' + band] = value is not None and value >= threshold
        value = stats['K0']['8-10']['broad_fraction']
        criteria['K0/8-10'] = value is not None and value <= .50
        value = stats['recent16']['all']['broad_fraction']
        criteria['recent16/all'] = value is not None and value <= .05
    return {'conditions': stats, 'criteria': criteria, 'passed': all(criteria.values()) if gate else None,
            'gate_definition': 'near interval start through replenishment, inclusive; detach may equal event release',
            'strict_definition_report_only': 'detach through replenishment, inclusive',
            'missing_end_decisions_count_as_not_covered': True,
            'interpretation': 'Information upper bound on expert trajectories, not proof that learned policies use it.',
            'episodes': episodes}
