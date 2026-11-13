#!/usr/bin/env python3
"""Audit smoke-run handoff gates without treating missing data as a pass."""
import argparse
import json
import math
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(path):
    result = json.loads((path / 'result.json').read_text())
    actions = {a['step']: a for a in read_jsonl(path / 'actions.jsonl')}
    trace = {x['timestep']: x for x in read_jsonl(path / 'trace.jsonl')}
    nslots = len(result['slot_xy'])
    slots = list(range(nslots))
    if result['task'] == 'count':
        slots += [s for s in range(nslots) for _ in range(result['N'])]
    checks = []
    for index, handoff in enumerate(result['handoffs']):
        end = handoff['timestep']
        prev, next_action = actions.get(end - 1), actions.get(end)
        row = {'index': index, 'step': end, 'jump_rad': None, 'jump_pass': False,
               'policy_at_end':bool(next_action and next_action['source'] == 'policy')}
        if prev and next_action and prev['source'] == 'planner' and next_action['source'] == 'policy':
            row['jump_rad'] = max(abs(a-b) for a,b in zip(prev['action'][:7], next_action['action'][:7]))
            row['jump_pass'] = row['jump_rad'] <= .3
        if index == 0:
            start = trace.get(end)
            after = trace.get(end + 8)
            slot = handoff['target_slot']
            actors = [i for i, f in enumerate(start['fruits'])
                      if slots[i] == slot and f['present'] and not f['attached'] and not f['wind_detached']] if start else []
            row['distance8_m'] = None
            row['distance8_pass'] = False
            # A new planner/stop phase before +8 does not verify policy continuity.
            continuous = all(actions.get(t, {}).get('source') == 'policy' for t in range(end, end + 8))
            if after and len(actors) == 1 and continuous:
                row['distance8_m'] = math.dist(after['tcp_pos'], after['fruits'][actors[0]]['pos'])
                row['distance8_pass'] = row['distance8_m'] <= .20
        checks.append(row)
    calls = result['planner_calls']
    count_ok = (result['task'] != 'count' or
                (result.get('reason') != 'overpick' and not result.get('stop_primitive_timeout') and
                 (result.get('stop_trigger_step') is None or
                  (result.get('stop_trigger_n') == result['N'] and result.get('reason') == 'finish_signal' and
                   result.get('stop_end_step') is not None and result.get('stop_arrive_step') is not None and
                   result['stop_end_step'] - result['stop_arrive_step'] <= 300))))
    policy_checks = [c for c in checks if actions.get(c['step'], {}).get('source') == 'policy']
    first = checks[0] if checks else None
    return {'path':str(path), 'outcome':result['outcome'], 'checks':checks,
            'planner_failure_rate':len(result['planner_failures']) / calls if calls else None,
            'pass':bool(not result.get('error') and calls and len(result['planner_failures']) / calls <= .05 and
                        policy_checks and sum(c['jump_pass'] for c in policy_checks) / len(policy_checks) >= .95 and
                        first and first.get('distance8_pass') and count_ok),
            'count_stop_pass': count_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('episode_dirs', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    results = [audit(p) for p in args.episode_dirs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'episodes':results, 'all_pass':all(r['pass'] for r in results)}, indent=2)+'\n')
    print(json.dumps({'all_pass':all(r['pass'] for r in results), 'episodes':len(results)}))
    return 0 if all(r['pass'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
