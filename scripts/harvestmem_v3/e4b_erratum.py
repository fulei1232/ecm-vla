#!/usr/bin/env python3
"""Reconstruct E4b handoffs and paired trajectories from retained CPU traces."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
E4B = ROOT / 'results/harvestmem_v3_E4b'
E4A = ROOT / 'results/harvestmem_v3_E4a'
DEV = ROOT / 'results/harvestmem_v3_S1/replays/dev_expert'
OFFSETS = (0, 50, 100, 200)
SPECS = [('DP1R','count',E4B), ('DPaR','count',E4B),
         ('EP1R','count',E4B), ('EPaR','count',E4B),
         ('D','occluded_ripe',E4B), ('DPa','occluded_ripe',E4B),
         ('E','occluded_ripe',E4B), ('EPa','occluded_ripe',E4B),
         ('DR','count',E4A), ('ER','count',E4A)]


def load(path):
    return json.loads(path.read_text())


def jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def distance(a, b):
    return math.dist(a, b)


def hanging(f):
    # The environment's attached flag means a fruit has been detached.
    return f['present'] and not f['attached'] and not f['wind_detached']


def trajectory(trace, by_step, start, slot, slot_of, cutoff):
    row = by_step.get(start)
    if row is None:
        return None
    actors = [i for i, f in enumerate(row['fruits']) if hanging(f) and slot_of[i] == slot]
    if len(actors) != 1:
        return None
    actor = actors[0]
    points = {}
    for offset in OFFSETS:
        t = start + offset
        if t >= cutoff or t not in by_step:
            continue
        # Only measure this fruit while still hanging and still the selected target.
        interval = (r for r in trace if start < r['timestep'] <= t)
        if any(not hanging(r['fruits'][actor]) or
               ('target_selected' in r['oracle_events'] and r['oracle_target_slot'] != slot)
               for r in interval):
            continue
        r = by_step[t]
        points[str(offset)] = distance(r['tcp_pos'], r['fruits'][actor]['pos'])
    return {'start': start, 'slot': slot, 'actor': actor, 'distances': points}


def process(arm, task, root, path):
    result = load(path / 'result.json')
    episode = result['episode']
    dev_path = DEV / task / f'ep{episode}_attempt0/result.json'
    dev = load(dev_path) if dev_path.exists() else None
    nslots = len(result['slot_xy'])
    slot_of = list(range(nslots))
    if task == 'count':
        slot_of += [slot for slot in range(nslots) for _ in range(result['N'])]
    if dev:
        assert slot_of == dev['slot_of'], (task, episode)
    ripe = [x['ripe'] for x in dev['slot_geometry']] if dev else None
    trace = jsonl(path / 'trace.jsonl')
    assert all(row['timestep'] == i for i, row in enumerate(trace)), path
    if dev:
        assert dev['seed'] == result['seed']
        assert all(distance(trace[0]['fruits'][i]['pos'], geom['pos']) < 1e-5
                   for i, geom in enumerate(dev['slot_geometry']))
    by_step = {r['timestep']: r for r in trace}
    actions = jsonl(path / 'actions.jsonl')
    action_by_step = {a['step']: a for a in actions}
    assert len(by_step) == len(trace) and len(action_by_step) == len(actions)
    detaches = []
    for prev, row in zip(trace, trace[1:]):
        for actor, (before, after) in enumerate(zip(prev['fruits'], row['fruits'])):
            if not before['attached'] and after['attached'] and not after['wind_detached']:
                detaches.append({'step': row['timestep'], 'actor': actor,
                                 'slot': slot_of[actor], 'ripe': bool(ripe[slot_of[actor]]) if ripe else None})
    success = result['outcome'] == 'success'
    reason = result['reason']
    start_fail = task == 'count' and not success and reason == 'finish_signal' and result['n_in_basket'] == 0
    category = ('success' if success else 'start_failure' if start_fail else
                'robot_drop' if reason == 'robot_drop' else 'budget' if reason == 'budget' else
                result['outcome'] if reason is None else str(reason))
    occluded_failure = None
    if task == 'occluded_ripe' and result['outcome'] == 'fail':
        final = trace[-1]['fruits']
        dropped = any(f['attached'] and not f['in_basket'] and f['pos'][2] < 0.06 for f in final)
        green = (any(f['in_basket'] and not ripe[slot_of[i]] for i, f in enumerate(final))
                 if ripe is not None else (True if not dropped else None))
        assert dropped or green is True, (path, dropped, green)
        occluded_failure = ('drop_and_green' if dropped and green is True else
                            'drop_unknown_green' if dropped and green is None else
                            'drop_only' if dropped else 'green_only')
    max_basket = max(r['n_in_basket'] for r in trace)
    reached = task == 'count' and max_basket >= result['N']
    decision_failure = task == 'count' and not success and (start_fail or
        (reached and reason in ('overpick', 'budget')) or
        (reason == 'finish_signal' and 0 < result['n_in_basket'] < result['N']))
    execution_failure = task == 'count' and not success and not decision_failure and (
        reason in ('robot_drop', 'budget', 'tcp_stuck_300') or
        (not reached and reason == 'finish_signal') or not detaches)
    target_correct = 0
    in_basket_after = 0
    for d in detaches:
        latest = next((r for r in reversed(trace[:d['step'] + 1])
                       if 'target_selected' in r['oracle_events']), None)
        target_correct += bool(latest and latest['oracle_target_slot'] == d['slot'])
        in_basket_after += any(r['fruits'][d['actor']]['in_basket']
                               for r in trace[d['step']:d['step'] + 300])
    record = {'arm': arm, 'task': task, 'episode': episode, 'outcome': result['outcome'],
              'reason': reason, 'category': category, 'steps': result['steps'],
              'n_in_basket': result['n_in_basket'], 'N': result['N'],
              'detaches': len(detaches), 'ripe_detaches': sum(d['ripe'] is True for d in detaches),
              'ripe_known_detaches': sum(d['ripe'] is not None for d in detaches),
              'target_correct': target_correct, 'detach_in_basket_300': in_basket_after,
              'decision_failure': decision_failure if task == 'count' else None,
              'execution_failure': execution_failure if task == 'count' else None,
              'occluded_failure': occluded_failure,
              'path': str(path.relative_to(ROOT)), 'first_target': None, 'first_trajectory': None}
    selected = [r for r in trace if 'target_selected' in r['oracle_events']]
    if selected:
        r = selected[0]
        record['first_target'] = r['oracle_target_slot']
        record['first_trajectory'] = trajectory(trace, by_step, r['timestep'], r['oracle_target_slot'], slot_of, float('inf'))
    handoffs = []
    starts = [h['timestep'] for h in result.get('handoffs', [])]
    for index, h in enumerate(result.get('handoffs', [])):
        start = h['timestep']
        next_start = starts[index + 1] if index + 1 < len(starts) else float('inf')
        end = start
        while end < next_start and action_by_step.get(end, {}).get('source') == 'planner':
            end += 1
        # Next planner action ends attribution even if that call fails.
        next_planner = min((a['step'] for a in actions if a['source'] == 'planner' and a['step'] >= end), default=float('inf'))
        boundary = min(next_start, next_planner)
        row = by_step.get(end)
        entry = {'arm': arm, 'task': task, 'episode': episode, 'index': index,
                 'recorded_start_step': start, 'end_step': end, 'planner_steps': end-start,
                 'target_slot': h['target_slot'], 'next_planner_start': None if math.isinf(boundary) else boundary,
                 'policy_at_end': action_by_step.get(end, {}).get('source') == 'policy',
                 'category': None, 'actual_detached_slot': None, 'trajectory': None}
        if row is None or end == start:
            entry['category'] = 'unresolved_boundary'
            handoffs.append(entry)
            continue
        entry['tcp_pos'] = row['tcp_pos']
        candidates = [(distance(row['tcp_pos'], f['pos']), i, slot_of[i])
                      for i, f in enumerate(row['fruits']) if hanging(f)]
        candidates.sort()
        entry['nearest_actor'] = candidates[0][1] if candidates else None
        entry['nearest_slot'] = candidates[0][2] if candidates else None
        entry['trajectory'] = trajectory(trace, by_step, end, h['target_slot'], slot_of, boundary)
        eligible = [d for d in detaches if end < d['step'] <= boundary]
        if not eligible:
            entry['category'] = 'no_detach'
        else:
            simultaneous = [d for d in eligible if d['step'] == eligible[0]['step']]
            if len(simultaneous) != 1:
                entry['category'] = 'simultaneous_detach'
            else:
                d = simultaneous[0]
                entry.update(detach_step=d['step'], actual_detached_slot=d['slot'], actual_detached_actor=d['actor'])
                entry['category'] = ('target' if d['slot'] == h['target_slot'] else
                                     'nearest' if d['actor'] == entry['nearest_actor'] else 'other')
        handoffs.append(entry)
    return record, handoffs


def main():
    records, handoffs = [], []
    for arm, task, root in SPECS:
        paths = sorted((root / arm / task).glob('ep*/result.json'))
        assert len(paths) == 60, (arm, task, len(paths))
        for p in paths:
            record, hs = process(arm, task, root, p.parent)
            records.append(record)
            handoffs.extend(hs)
        print(arm, task, len(paths), flush=True)
    summaries = {}
    for arm, task, _ in SPECS:
        rr = [r for r in records if r['arm'] == arm and r['task'] == task]
        hh = [h for h in handoffs if h['arm'] == arm and h['task'] == task]
        summaries[arm] = {'categories': dict(Counter(r['category'] for r in rr)),
                          'mean_steps': mean(r['steps'] for r in rr),
                          'detaches': sum(r['detaches'] for r in rr),
                          'ripe_detaches': sum(r['ripe_detaches'] for r in rr),
                          'ripe_known_detaches': sum(r['ripe_known_detaches'] for r in rr),
                          'target_correct': sum(r['target_correct'] for r in rr),
                          'detach_in_basket_300': sum(r['detach_in_basket_300'] for r in rr),
                          'decision_failure': sum(r['decision_failure'] is True for r in rr) if task == 'count' else None,
                          'execution_failure': sum(r['execution_failure'] is True for r in rr) if task == 'count' else None,
                          'occluded_failure': dict(Counter(r['occluded_failure'] for r in rr if r['occluded_failure'])),
                          'handoffs': dict(Counter(h['category'] for h in hh)),
                          'policy_at_end': sum(h['policy_at_end'] for h in hh)}
    pairs = {}
    for p, c in [('DP1R','DR'),('DPaR','DR'),('EP1R','ER'),('EPaR','ER'),('DPa','D'),('EPa','E')]:
        pr = {r['episode']: r for r in records if r['arm'] == p}
        cr = {r['episode']: r for r in records if r['arm'] == c}
        assert set(pr) == set(cr)
        pairs[f'{p} vs {c}'] = dict(Counter(('both' if pr[e]['outcome'] == cr[e]['outcome'] == 'success' else
                  'planner_only' if pr[e]['outcome'] == 'success' else
                  'control_only' if cr[e]['outcome'] == 'success' else 'neither') for e in pr))
    trajectories = {}
    for p,c in [('DP1R','DR'),('DPaR','DR'),('EP1R','ER'),('EPaR','ER')]:
        cc = {r['episode']:r for r in records if r['arm']==c}
        hh = [h for h in handoffs if h['arm']==p and h['index']==0 and h['policy_at_end']]
        detail = []
        for h in hh:
            control = cc[h['episode']]
            a, b = h['trajectory'], control['first_trajectory']
            if not a or not b or a['slot'] != b['slot']:
                continue
            for offset in OFFSETS:
                k = str(offset)
                if k in a['distances'] and k in b['distances']:
                    detail.append({'episode': h['episode'], 'slot': a['slot'], 'offset': offset,
                                   'planner': a['distances'][k], 'control': b['distances'][k],
                                   'difference': a['distances'][k]-b['distances'][k],
                                   'planner_change': a['distances'][k]-a['distances']['0'],
                                   'control_change': b['distances'][k]-b['distances']['0']})
        summary = {}
        for offset in OFFSETS:
            d = [x for x in detail if x['offset']==offset]
            if d:
                summary[str(offset)] = {'n':len(d), 'planner_mean_m':mean(x['planner'] for x in d),
                    'control_mean_m':mean(x['control'] for x in d),
                    'mean_paired_difference_m':mean(x['difference'] for x in d),
                    'median_paired_difference_m':median(x['difference'] for x in d),
                    'planner_mean_change_m':mean(x['planner_change'] for x in d),
                    'control_mean_change_m':mean(x['control_change'] for x in d),
                    'planner_moved_away':sum(x['planner_change']>0.01 for x in d)}
        trajectories[f'{p} vs {c}'] = {'summary':summary, 'pairs':detail}
    out = {'records': records, 'handoffs': handoffs, 'summaries': summaries, 'pairs': pairs, 'trajectories': trajectories}
    path = E4B / 'erratum_analysis.json'
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
