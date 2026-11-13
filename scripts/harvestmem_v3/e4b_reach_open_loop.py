#!/usr/bin/env python3
"""Evaluate model/idle action MAE on expert reach-to-approach boundaries."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from open_loop import _load_replay
from openpi_client import websocket_client_policy
from utils import pack_buffer

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / 'results/harvestmem_v3_S1/replays/dev_expert'


def boundaries(path):
    with h5py.File(path / 'demo.h5', 'r') as h5:
        ep = h5[next(k for k in h5 if k.startswith('episode_'))]
        ts = sorted(int(k.split('_')[1]) for k in ep if k.startswith('timestep_'))
        phases = {t: ep[f'timestep_{t}']['info']['phase'][()].decode()
                  for t in ts if 'phase' in ep[f'timestep_{t}']['info']}
        return [t for t in ts if t >= 30 and phases.get(t - 1) == 'reach' and phases.get(t) == 'approach']


def select_replay(task, episode):
    candidates = sorted((DEV / task).glob(f'ep{episode}_attempt*/result.json'),
                        key=lambda p: int(p.parent.name.rsplit('attempt', 1)[1]))
    for result in candidates:
        metadata = json.loads(result.read_text())
        if metadata.get('success') and (result.parent / 'demo.h5').is_file():
            return result.parent, metadata
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['k0', 'nm'])
    ap.add_argument('--port', required=True, type=int)
    ap.add_argument('--context', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.context.parent.mkdir(parents=True, exist_ok=True)
    args.context.write_text(json.dumps({'memory': 'none' if args.arm == 'nm' else 'uniform'}))
    client = websocket_client_policy.MMEVLAWebsocketClientPolicy('127.0.0.1', args.port)
    details, excluded, completed = [], [], []
    checkpoint_path = args.out.with_suffix('.partial.json')
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint['arm'] != args.arm:
            raise ValueError('partial output belongs to a different model')
        details, excluded, completed = checkpoint['details'], checkpoint['excluded'], checkpoint['completed']
    def save_partial():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'arm':args.arm, 'details':details, 'excluded':excluded,
                                    'completed':completed}, indent=2) + '\n')
        temp.replace(checkpoint_path)
    for task in ('count', 'occluded_ripe'):
        for ep in range(200, 260):
            if [task, ep] in completed:
                continue
            path, metadata = select_replay(task, ep)
            if path is None:
                excluded.append({'task': task, 'episode': ep, 'reason': 'no successful expert replay'})
                completed.append([task, ep]); save_partial()
                continue
            replay = _load_replay(path)
            frames, actions = replay['frames'], replay['actions']
            assert client.reset().get('reset_finished')
            last = -1
            for t in boundaries(path):
                expert = np.stack([actions[x] for x in range(t, t + 16) if x in actions]) if t in actions else None
                if expert is None:
                    excluded.append({'task':task, 'episode':ep, 'timestep':t, 'reason':'missing expert action'})
                    continue
                assert t < len(frames) and t > last
                # Feed every recorded frame exactly once, preserving E2's step-30 boundary.
                while last < t:
                    end = min(t, 30 if last < 0 else last + 16)
                    batch = frames[last + 1:end + 1]
                    client.add_buffer(pack_buffer([x['front'] for x in batch],
                                                  [x['state'] for x in batch], 30 if last < 0 else 0))
                    last = end
                frame = frames[t]
                prediction = np.asarray(client.infer({'observation/image':frame['front'],
                    'observation/wrist_image':frame['wrist'], 'observation/state':frame['state'],
                    'prompt':replay['instruction'].lower()})['actions'], dtype=np.float32)[:len(expert)]
                assert len(prediction) == len(expert)
                model_sum = float(np.abs(prediction[:, :7] - expert[:, :7]).sum())
                idle_sum = float(np.abs(frame['state'][:7][None, :] - expert[:, :7]).sum())
                details.append({'task':task, 'episode':ep, 'attempt':metadata['attempt'],
                                'seed':metadata['seed'], 'timestep':t, 'steps':len(expert),
                                'model_abs_sum':model_sum, 'idle_abs_sum':idle_sum,
                                'elements':int(expert[:, :7].size)})
            completed.append([task, ep]); save_partial()
            print(task, ep, 'boundaries', sum(d['task'] == task and d['episode'] == ep for d in details), flush=True)
    model_sum = sum(d['model_abs_sum'] for d in details)
    idle_sum = sum(d['idle_abs_sum'] for d in details)
    result = {'arm':args.arm, 'boundary_definition':'previous phase reach, current phase approach',
              'details':details, 'excluded':excluded,
              'aggregate_ratio':model_sum / idle_sum if idle_sum else None,
              'e2_train_combined_ratio':json.loads((ROOT / f'results/harvestmem_v3_E2/open_loop_{args.arm}.json').read_text())['train_combined_ratio']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
