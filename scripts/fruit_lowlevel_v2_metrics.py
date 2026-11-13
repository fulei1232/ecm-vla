"""Read-only event validity and preregistered, mutually exclusive failure classes."""
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, read_json, read_jsonl

sys.path.insert(0, str(ECM/'src'))
from memory_management.event_keyframes import KeyframeConfig, detect_events, select

STOP_WINDOW = 200
ARM_MOTION_RAD = .005
TCP_STILL_M = .05


def fraction(numerator, denominator):
    return {'numerator': int(numerator), 'denominator': int(denominator),
            'fraction': numerator/denominator if denominator else None}


def classify_failure(result, raw, controls, near_miss=False):
    """Memory errors take priority; invalid runs never receive a mechanism label.

    Count premature-stop is an observable open-gripper inactivity proxy, not a
    claim about intent: the complete final200 controls are still, with >=200
    budget steps left at their start, quota unmet, and a ripe fruit still attached.
    """
    if result.get('error') or result.get('client_exit_code', 0) or not raw:
        return {'class': 'invalid', 'memory_errors': [], 'operation_tags': []}
    gt = [r['gt'] for r in raw]
    distances = [g['fruit_distance'] for g in gt]
    events = detect_events(distances, .03, .005)
    memory = []
    if result['task'] == 'FruitCount':
        quota = int(result['n_pick'])
        baskets = [int(g['n_in_basket']) for g in gt]
        reached = next((i for i, n in enumerate(baskets) if n >= quota), None)
        if max(baskets) > quota:
            memory.append('count_over_quota')
        if reached is not None:
            new_detach = any(any(bool(b) and not bool(a) for a, b in zip(gt[i-1]['detached'], gt[i]['detached']))
                             for i in range(max(1, reached+1), len(gt)))
            if new_detach or any(e['start'] > reached for e in events):
                memory.append('count_continued_picking_after_quota')
        n = len(controls)
        if result['outcome'] == 'timeout' and n >= STOP_WINDOW+1 and baskets[-1] < quota:
            start = n-STOP_WINDOW
            budget = result.get('total_step_budget', 1300)
            a = np.asarray([c['action'] for c in controls[start-1:]], dtype=float)
            tcp = np.asarray([g['tcp_position'] for g in gt[start:]], dtype=float)
            still = (np.all(np.max(np.abs(np.diff(a[:, :7], axis=0)), axis=1) <= ARM_MOTION_RAD)
                     and np.all(a[1:, 7] > 0)
                     and np.all(np.linalg.norm(tcp-tcp[0], axis=1) <= TCP_STILL_M))
            unpicked = any(bool(r) and not bool(d) for r, d in zip(gt[start]['ripe'], gt[start]['detached']))
            no_events = all(gt[i]['detached'] == gt[start]['detached']
                            and gt[i]['in_basket'] == gt[start]['in_basket'] for i in range(start, len(gt)))
            if budget-start >= STOP_WINDOW and still and unpicked and no_events:
                memory.append('count_premature_stop_proxy')
    else:
        if any(any(bool(b) and i not in set(g['targets']) for i, b in enumerate(g['in_basket'])) for g in gt):
            memory.append('occluded_wrong_in_basket')
        if any(any(bool(d) and not bool(r) for d, r in zip(g['detached'], g['ripe'])) for g in gt):
            memory.append('occluded_unripe_detached')
    tags = []
    if not any(gt[-1]['detached']):
        tags.append('never_detached')
    if near_miss:
        tags.append('near_miss')
    if result['outcome'] == 'fail':
        tags.append('dropped_fruit_outcome_fail')
    if result['outcome'] == 'timeout':
        # Separate the inactivity proxy from the other incomplete manipulations.
        tags.append('timeout_stopped' if 'count_premature_stop_proxy' in memory else 'timeout_operating_or_stalled')
    category = 'memory' if memory else ('success' if result['outcome'] == 'success' else 'operation')
    return {'class': category, 'memory_errors': memory, 'operation_tags': tags,
            'premature_stop_proxy_window': STOP_WINDOW}


def event_statistics(directory, config):
    """Recompute causal event keys; report both distance-release proxy and true detach."""
    directory = Path(directory)
    result = read_json(directory/'result.json')
    raw = read_jsonl(directory/'raw_frames.jsonl')
    decisions = read_jsonl(directory/'policy_decisions.jsonl')
    oracle = read_jsonl(directory/'oracle_decisions.jsonl')
    cfg = KeyframeConfig(**config)
    distances = np.asarray([r['gt']['fruit_distance'] for r in raw], dtype=float)
    errors = []
    if not len(raw) or not len(decisions) or len(oracle) != len(decisions):
        raise ValueError('missing physical frames or decision records')
    if not np.isfinite(distances).all():
        errors.append('nonfinite_raw_distance')
    full = detect_events(distances, cfg.radius, cfg.tol)
    last_available = int(decisions[-1]['available_max_frame_id'])
    served = detect_events(distances[:last_available+1], cfg.radius, cfg.tol)
    designated = selected_event = slots = 0
    by_decision = []
    for i, (d, o) in enumerate(zip(decisions, oracle)):
        t = int(d['available_max_frame_id'])
        selected, info = select(distances, t, cfg)
        if d['selected_frame_ids'] != selected or o.get('keyframes') != info:
            errors.append(f'causal_event_selection/{i}')
        events = detect_events(distances[:t+1], cfg.radius, cfg.tol)
        possible = {e[k] for e in events for k in ('contact', 'release') if e[k] is not None}
        chosen = len(set(info['keyframes']) & set(d['selected_frame_ids']))
        incidental = len(possible & set(d['selected_frame_ids']))
        designated += chosen
        selected_event += incidental
        slots += len(d['selected_frame_ids'])
        by_decision.append({'decision': i, 'physical_frame': t, 'detected_events': len(events),
            'designated_event_frames': info['keyframes'], 'valid_memory_slots': len(d['selected_frame_ids']),
            'selected_causal_event_frames': incidental})
    true_detach = [{'frame': i, 'fruit': j} for i in range(1, len(raw))
                   for j, (a, b) in enumerate(zip(raw[i-1]['gt']['detached'], raw[i]['gt']['detached']))
                   if bool(b) and not bool(a)]
    releases = [e['release'] for e in full if e['release'] is not None]
    explained = None
    if designated == 0:
        if cfg.events == 0:
            explained = 'K0 designates no event slots by definition; incidental event selections reported separately.'
        elif not served:
            explained = ('No distance<=0.03m event was observed before a policy decision; the policy did not '
                         'reach the near region, or its first contact occurred only after the final decision.')
        else:
            errors.append('event_seen_but_no_event_slots')
    return {'task': result['task'], 'episode_id': result['episode_id'], 'condition': result['condition'],
        'contact_events': len(full), 'detach_event_keyframes': len(releases), 'events': full,
        'server_observed_contact_events': len(served),
        'server_observed_detach_event_keyframes': sum(e['release'] is not None for e in served),
        'physical_detach_events': len(true_detach), 'physical_detaches': true_detach,
        'detach_proxy_matches_within_one_frame': sum(any(abs(e['frame']-t) <= 1 for e in true_detach) for t in releases),
        'designated_event_slot_fraction': fraction(designated, slots),
        'selected_causal_event_slot_fraction': fraction(selected_event, slots),
        'min_distance_before_last_decision': float(distances[:last_available+1].min()),
        'zero_event_slot_explanation': explained, 'event_validity_passed': not errors,
        'errors': errors, 'decisions': by_decision,
        'definition': 'detach_event_keyframe is the unchanged selector release (first far frame), a distance proxy; physical detaches are separate GT transitions. Fractions count valid frame slots, excluding padding.'}


def aggregate_event_statistics(rows):
    def group(values):
        keys = ('contact_events', 'detach_event_keyframes', 'physical_detach_events',
                'server_observed_contact_events', 'server_observed_detach_event_keyframes')
        return {'episodes': len(values), **{k: sum(v[k] for v in values) for k in keys},
            **{k: fraction(sum(v[k]['numerator'] for v in values), sum(v[k]['denominator'] for v in values))
               for k in ('designated_event_slot_fraction', 'selected_causal_event_slot_fraction')},
            'zero_event_slot_episodes': [v['episode_id'] for v in values if v['designated_event_slot_fraction']['numerator'] == 0],
            'all_zero_event_slots_explained': all(v['zero_event_slot_explanation'] for v in values if v['designated_event_slot_fraction']['numerator'] == 0),
            'event_validity_passed': all(v['event_validity_passed'] for v in values)}
    return {'all': group(rows), 'by_task': {t: group([r for r in rows if r['task'] == t])
                                          for t in sorted({r['task'] for r in rows})},
            'per_episode': rows}
