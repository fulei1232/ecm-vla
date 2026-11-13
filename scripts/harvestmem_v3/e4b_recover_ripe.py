#!/usr/bin/env python3
"""Recover seeded Occluded layout metadata and verify against saved traces."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fruit_picking.task_v2_spec import TaskV2Spec, task_layout

OLD = ROOT / 'results/harvestmem_v3_E4b'
DEV = ROOT / 'results/harvestmem_v3_S1/replays/dev_expert/occluded_ripe'
OUT = ROOT / 'results/harvestmem_v3_E4b_r1/seeded_ripe.json'


def main():
    episodes = {}
    counts = {arm: {'ripe': 0, 'all': 0} for arm in ('D', 'DPa', 'E', 'EPa')}
    for ep in range(200, 260):
        paths = [OLD / arm / 'occluded_ripe' / f'ep{ep}' for arm in counts]
        records = [json.loads((p / 'result.json').read_text()) for p in paths]
        seed = 4200000 + ep * 16
        assert all(r['seed'] == seed for r in records)
        layout = task_layout('occluded_ripe', seed, 'v2', TaskV2Spec(version='v2'))
        ripe = [bool(f['ripe']) for f in layout.fruits]
        assert all(len(r['slot_xy']) == len(ripe) and r['N'] == layout.n_pick for r in records)
        dev = DEV / f'ep{ep}_attempt0/result.json'
        if dev.exists():
            archived = json.loads(dev.read_text())
            assert archived['seed'] == seed
            assert ripe == [bool(f['ripe']) for f in archived['slot_geometry']]
            assert all(math.dist(a['pos'], b['pos']) < 1e-9
                       for a, b in zip(layout.fruits, archived['slot_geometry']))
        for arm, path in zip(counts, paths):
            with (path / 'trace.jsonl').open() as f:
                prev = json.loads(next(f))
                assert len(prev['fruits']) == len(ripe)
                assert all(math.dist(fruit['pos'], geom['pos']) < 1e-5
                           for fruit, geom in zip(prev['fruits'], layout.fruits))
                for line in f:
                    row = json.loads(line)
                    for i, (before, after) in enumerate(zip(prev['fruits'], row['fruits'])):
                        if not before['attached'] and after['attached'] and not after['wind_detached']:
                            counts[arm]['all'] += 1
                            counts[arm]['ripe'] += int(ripe[i])
                    prev = row
        episodes[str(ep)] = {'seed': seed, 'ripe': ripe, 'expert_metadata_verified': dev.exists()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'method': 'task_layout seed reconstruction, not simulator reset',
                               'episodes': episodes, 'counts': counts}, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
