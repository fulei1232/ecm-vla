"""Exploratory memory-use diagnosis. Does not change any D5 artifact/definition."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, read_json, read_jsonl

sys.path.insert(0, str(ECM))
from fruit_picking.layout import sample_layout

LABEL = '探索性、非预登记，不改变D5结论，不写入论文'
OUT = ECM/'results/fruit_memory_usage_diag_20261104'
D5 = ECM/'results/fruit_lowlevel_v2_eval_20261104'
G1 = ECM/'results/fruit_k0v2_gate_20261102'
MODES = ('K0_normal', 'K0_recent16', 'K0_no_pregust')
ABLATIONS = {'K0_normal': None, 'K0_recent16': 'recent16', 'K0_no_pregust': 'no_pregust'}
TASKS = ('FruitCount', 'FruitOccludedRipe')


def ratio(n, d):
    return {'numerator': int(n), 'denominator': int(d), 'fraction': n/d if d else None}


def select_frames(step_idx, ablation=None):
    """Physical frame ids equal physical control steps; never use policy-relative time."""
    t = int(step_idx)
    if ablation not in (None, 'recent16', 'no_pregust'):
        raise ValueError(f'Unknown memory ablation: {ablation}')
    if ablation == 'recent16':
        return list(range(max(0, t-15), t+1))
    candidates = np.arange(20 if ablation == 'no_pregust' else 0, t+1, dtype=np.int64)
    if not len(candidates):
        return []
    n = min(16, len(candidates))
    return candidates[np.linspace(0, len(candidates)-1, n, dtype=np.int64)].tolist()


@lru_cache(maxsize=1000)
def layout_from_record(task, environment_seed):
    name = 'count' if task == 'FruitCount' else 'occluded_ripe'
    return json.loads(sample_layout(name, int(environment_seed)).to_json())


def checked_layout(result, raw):
    """Reconstruct labels from the recorded seed, not val manifests or demonstrations."""
    layout = layout_from_record(result['task'], result['environment_seed'])
    first = raw[0]['gt']
    assert np.allclose([f['pos'] for f in layout['fruits']], first['fruit_positions'], rtol=0, atol=1e-6)
    assert [f['ripe'] for f in layout['fruits']] == first['ripe']
    assert layout['targets'] == first['targets'] and layout['n_pick'] == first['n_pick']
    return layout


def hidden_metrics(layout, raw):
    gt = [r['gt'] for r in raw]
    tcp = np.array([g['tcp_position'] for g in gt])
    positions = np.array([g['fruit_positions'] for g in gt])
    detached = np.array([g['detached'] for g in gt], dtype=bool)
    inside = np.array([g['in_basket'] for g in gt], dtype=bool)
    steps = np.array([r['control_step'] for r in raw])
    hidden = {o['fruit'] for o in layout['occluders']}
    by_fruit = []
    for j, fruit in enumerate(layout['fruits']):
        det = np.flatnonzero(detached[:, j])
        stop = int(det[0]) if len(det) else len(raw)-1
        distances = np.linalg.norm(tcp[:stop+1]-positions[:stop+1, j], axis=1)
        approached = np.flatnonzero(distances < .15)
        first = int(approached[0]) if len(approached) else None
        ripe = bool(fruit['ripe'])
        wrong = not ripe and (bool(detached[:, j].any()) or bool(inside[:, j].any()))
        item = {'fruit': j, 'hidden': j in hidden, 'ripe': ripe,
            'approached_15cm': first is not None, 'first_approach_step': int(steps[first]) if first is not None else None,
            'detached': bool(len(det)), 'first_detachment_step': int(steps[stop]) if len(det) else None,
            'approached_then_detached': first is not None and bool(len(det)) and first <= stop,
            'wrong_fruit': wrong, 'wrong_in_basket': not ripe and bool(inside[:, j].any()),
            'min_tcp_distance_before_detachment_m': float(distances.min())}
        if j in hidden:
            occ = next(o for o in layout['occluders'] if o['fruit'] == j)
            alpha = np.clip((steps[:stop+1]-8)/12, 0, 1)
            alpha = alpha*alpha*(3-2*alpha)
            center = (1-alpha[:, None])*np.asarray(occ['open'])+alpha[:, None]*np.asarray(occ['closed'])
            # Actual PUSH_START is distance to the moving cluster's base center.
            revealed = np.flatnonzero(np.linalg.norm(tcp[:stop+1]-center, axis=1) < .15)
            item['actual_leaf_push_started'] = bool(len(revealed))
            item['first_leaf_push_step'] = int(steps[revealed[0]]) if len(revealed) else None
        if j in hidden and not ripe and first is not None:
            candidates = [k for k in layout['targets'] if not detached[first, k]]
            item['simultaneously_near_attached_ripe_target'] = any(np.linalg.norm(tcp[first]-positions[first, k]) < .15 for k in candidates)
        by_fruit.append(item)
    hr = [f for f in by_fruit if f['hidden'] and f['ripe']]
    hu = [f for f in by_fruit if f['hidden'] and not f['ripe']]
    omission = any(not f['approached_15cm'] for f in hr)
    commission = any(f['wrong_fruit'] for f in by_fruit)
    return {'omission_hidden_ripe': omission,
        'checked_hidden_unripe': any(f['approached_15cm'] and not f['detached'] for f in hu),
        'eligible_hidden_unripe': bool(hu), 'wrong_fruit': commission,
        'wrong_in_basket': any(f['wrong_in_basket'] for f in by_fruit),
        'wrong_hidden_unripe': any(f['wrong_fruit'] and f['hidden'] for f in by_fruit),
        'wrong_visible_unripe': any(f['wrong_fruit'] and not f['hidden'] for f in by_fruit),
        'omission_plus_wrong_fruit': int(omission)+int(commission),
        'omission_or_wrong_fruit': omission or commission,
        'omitted_hidden_ripe_fruits': sum(not f['approached_15cm'] for f in hr),
        'hidden_ripe_fruits': len(hr), 'hidden_unripe_fruits': len(hu), 'fruits': by_fruit}


def retained_metrics(raw, decisions):
    counts = {p: [0, 0] for p in ('before_target_1', 'before_target_2', 'before_target_3', 'after_all_targets')}
    rows = []
    includes_zero = 0
    for d in decisions:
        t = int(d['available_max_frame_id']);gt = raw[t]['gt']
        k = sum(bool(gt['detached'][j]) for j in gt['targets'])
        stage = 'after_all_targets' if k >= len(gt['targets']) else f'before_target_{k+1}'
        ids = d['selected_frame_ids']
        retained = any(raw[i]['control_step'] < 8 for i in ids)
        includes_zero += 0 in ids
        counts[stage][0] += int(retained);counts[stage][1] += 1
        rows.append({'decision': d['decision_idx'], 'physical_control_step': raw[t]['control_step'],
            'phase': stage, 'pre_gust_retained': retained, 'frame_zero_selected': 0 in ids,
            'selected_physical_steps': [raw[i]['control_step'] for i in ids]})
    return {'all': ratio(sum(v[0] for v in counts.values()), len(rows)),
        'by_phase': {p: ratio(*v) for p, v in counts.items()}, 'frame_zero_selected': ratio(includes_zero, len(rows)),
        'decisions': rows}


def count_failure_metrics(raw, controls, outcome):
    gt = [r['gt'] for r in raw]
    tcp = np.asarray([g['tcp_position'] for g in gt]);positions = np.asarray([g['fruit_positions'] for g in gt])
    detached = np.asarray([g['detached'] for g in gt], dtype=bool)
    inside = np.asarray([g['in_basket'] for g in gt], dtype=bool)
    drops = []
    if outcome == 'fail':
        for j in range(positions.shape[1]):
            fell = np.flatnonzero(detached[:, j] & ~inside[:, j] & (positions[:, j, 2] < .06))
            if not len(fell):
                continue
            ground = int(fell[0]);start = int(np.flatnonzero(detached[:, j])[0])
            separation = np.linalg.norm(positions[start:ground+1, j]-tcp[start:ground+1], axis=1)
            lost = np.flatnonzero(separation > .08)
            loss = start+int(lost[0]) if len(lost) else ground
            opened = [t for t in range(max(1, start), loss+1) if controls[t-1]['action'][7] >= 0]
            if opened:
                phase = 'release'
            elif loss <= start+2:
                phase = 'grasp'
            elif np.linalg.norm(tcp[max(start, loss-1)]-tcp[start]) <= .05:
                phase = 'pull'
            else:
                phase = 'carry'
            drops.append({'fruit': j, 'phase_proxy': phase, 'detach_step': start,
                'first_separation_over_8cm_step': loss, 'first_open_after_detach': opened[0] if opened else None,
                'ground_failure_step': ground, 'phase_is_observational_proxy': True})
    tail = None
    if outcome == 'timeout':
        n = min(200, len(controls));a = np.array([c['action'] for c in controls[-n:]])
        change = np.max(np.abs(np.diff(a[:, :7], axis=0)), axis=1) if n > 1 else np.array([])
        moving = float((change > .005).mean()) if len(change) else 0.
        holding = [any(d and not b and np.linalg.norm(np.asarray(p)-np.asarray(g['tcp_position'])) < .04
                       for d, b, p in zip(g['detached'], g['in_basket'], g['fruit_positions'])) for g in gt[-n:]]
        label = 'ongoing_motion' if moving >= .05 else ('holding_stall' if any(holding) else
                 'after_basket_stall' if gt[-1]['n_in_basket'] else 'empty_gripper_stall')
        tail = {'label': label, 'window': n, 'arm_target_motion_fraction': moving,
            'open_gripper_fraction': float((a[:, 7] >= 0).mean()), 'holding_fruit_fraction': float(np.mean(holding)),
            'tcp_path_length_m': float(np.linalg.norm(np.diff(tcp[-n:], axis=0), axis=1).sum()),
            'tcp_net_displacement_m': float(np.linalg.norm(tcp[-1]-tcp[-n])),
            'final_in_basket': gt[-1]['n_in_basket'], 'n_pick': gt[-1]['n_pick']}
    return {'drops': drops, 'timeout_tail': tail}


def hidden_summary(rows):
    keys = ('omission_hidden_ripe', 'checked_hidden_unripe', 'wrong_fruit', 'wrong_in_basket',
            'wrong_hidden_unripe', 'wrong_visible_unripe', 'omission_or_wrong_fruit')
    own = [r['hidden'] for r in rows]
    eligible = [r for r in own if r['eligible_hidden_unripe']]
    return {'episodes': len(rows), **{k: ratio(sum(r[k] for r in own), len(own)) for k in keys},
        'checked_hidden_unripe_among_eligible': ratio(sum(r['checked_hidden_unripe'] for r in eligible), len(eligible)),
        'omission_plus_wrong_fruit': {'score_sum': sum(r['omission_plus_wrong_fruit'] for r in own),
            'episodes': len(own), 'mean_score': np.mean([r['omission_plus_wrong_fruit'] for r in own]).item() if own else None},
        'omitted_hidden_ripe_fruits': ratio(sum(r['omitted_hidden_ripe_fruits'] for r in own), sum(r['hidden_ripe_fruits'] for r in own))}


def paired_ci(rows, a, b, metric, task, seeds, name):
    selected = [r for r in rows if r['task'] == task and r['condition'] in (a, b)]
    layouts = sorted({r['episode'] for r in selected})
    lookup = {(r['condition'], r['episode'], r['model_seed']): r for r in selected}
    def value(c, e, s):
        r = lookup[c, e, s]
        return float(r['outcome'] == 'success') if metric == 'success' else float(r['hidden'][metric])
    delta = np.array([[value(a, e, s)-value(b, e, s) for s in seeds] for e in layouts]).mean(axis=1)
    seed = int.from_bytes(hashlib.sha256(f'D6|20261104|{name}'.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {'difference': float(delta.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
        'layouts': len(layouts), 'seeds_per_layout': len(seeds), 'bootstrap_draws': 10000,
        'bootstrap_seed': seed, 'exploratory': True}


def diagnose_rule(rows):
    normal = [r for r in rows if r['task'] == 'FruitOccludedRipe' and r['condition'] == 'K0_normal']
    removed = [r for r in rows if r['task'] == 'FruitOccludedRipe' and r['condition'] == 'K0_no_pregust']
    assert len(normal) == len(removed) == 80
    drop = sum(r['outcome'] == 'success' for r in normal)-sum(r['outcome'] == 'success' for r in removed)
    rise = sum(r['hidden']['omission_plus_wrong_fruit'] for r in removed)-sum(r['hidden']['omission_plus_wrong_fruit'] for r in normal)
    verdict = 'A' if drop*100 >= 10*80 or rise*100 >= 10*80 else 'B' if drop*100 < 5*80 and rise*100 < 5*80 else 'C'
    return {'verdict': verdict, 'success_drop_pp': drop/80*100, 'omission_plus_commission_rise_pp': rise/80*100,
        'denominator': 80, 'rule': '§14.5 signed directional losses, pooled seeds7/8; threshold applied to point estimates, no p-value gate.',
        'compound_definition': 'I(any omitted hidden ripe)+I(any unripe detached or wrong fruit in basket); one episode may contribute2.',
        'label': LABEL}
