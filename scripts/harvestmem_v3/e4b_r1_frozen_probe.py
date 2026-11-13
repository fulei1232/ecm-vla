#!/usr/bin/env python3
"""Compare identical inputs across three fresh servers, without closed-loop runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('E4B_PREFLIGHT', 'preflight_frozen_probe')
import e4b_r1_preflight as preflight


def client(args):
    import numpy as np
    from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
    from open_loop import _load_replay
    from utils import pack_buffer

    payload = preflight.PREFLIGHT / 'frozen_inputs.npz'
    if not payload.exists():
        replay = _load_replay(preflight.ROOT / 'results/harvestmem_v3_S1/replays/dev_expert/count/ep210_attempt0')
        frames = replay['frames'][:31]
        assert len(frames) == 31
        np.savez(payload, images=np.stack([f['front'] for f in frames]),
                 states=np.stack([f['state'] for f in frames]), wrist=frames[-1]['wrist'],
                 prompt=np.asarray(replay['instruction'].lower()))
    with np.load(payload, allow_pickle=False) as saved:
        images, states, wrist, prompt = (saved[k] for k in ('images','states','wrist','prompt'))
        context = Path(args.context)
        context.write_text(json.dumps({'memory':'uniform'}))
        connection = MMEVLAWebsocketClientPolicy('127.0.0.1', preflight.PORT)
        for repeat in range(2):
            if not connection.reset().get('reset_finished'):
                raise RuntimeError('reset failed')
            connection.add_buffer(pack_buffer(list(images), list(states), 30))
            prediction = connection.infer({'observation/image':images[-1],
                'observation/wrist_image':wrist, 'observation/state':states[-1], 'prompt':str(prompt)})
            np.save(Path(args.output) / f'client_actions_{repeat}.npy', prediction['actions'])
        print('Two frozen-input predictions completed; no environment steps.', flush=True)


def summarize(root):
    import numpy as np
    probes = [json.loads((root / f'process_{i}/infer_000.json').read_text()) for i in range(3)]
    base = probes[0]
    a = np.load(root / 'process_0/actions_000.npy')[:, :7]
    comparisons = []
    for i, probe in enumerate(probes):
        b = np.load(root / f'process_{i}/actions_000.npy')[:, :7]
        second = json.loads((root / f'process_{i}/infer_001.json').read_text())
        comparisons.append({'process':i, 'observation_equal_to_process0':probe['observation']==base['observation'],
            'rng_equal_to_process0':probe['rng']==base['rng'],
            'history_inputs_equal_to_process0':probe['history_inputs']==base['history_inputs'],
            'actions_equal_to_process0':probe['actions']==base['actions'],
            'within_process_repeat_equal':all(probe[k]==second[k] for k in
                ('observation','rng','history_inputs','actions')),
            'first_action_max_delta_rad':float(np.max(np.abs(a[0]-b[0]))),
            'chunk_max_delta_rad':float(np.max(np.abs(a-b)))})
    result = {'comparisons':comparisons, 'preflight_gate':'failed; unchanged',
              'input_source':'fixed expert Count ep210 attempt0 prefix frames 0–30; not a live episode'}
    (root / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--client', action='store_true')
    ap.add_argument('--context')
    ap.add_argument('--output')
    ap.add_argument('--summarize', action='store_true')
    args = ap.parse_args()
    if args.client:
        return client(args)
    if args.summarize:
        return summarize(preflight.PREFLIGHT)
    subprocess.run(['nvidia-smi', '-L'], check=True)
    root = preflight.PREFLIGHT
    root.mkdir(parents=True, exist_ok=False)
    try:
        for index in range(3):
            probe_dir = root / f'process_{index}'
            preflight.SERVER_ENV['E4B_PROBE_DIR'] = str(probe_dir)
            with preflight.server(f'frozen_{index}', preflight.config.CK['DP1R'],
                                  preflight.GPUS[0], preflight.PORT) as context:
                preflight.run_logged(f'probe_{index}', [preflight.SIM_PY, '-u', str(Path(__file__).resolve()),
                    '--client', '--context', str(context), '--output', str(probe_dir)], timeout=600)
        preflight.run_logged('comparison', [preflight.SIM_PY, '-u', str(Path(__file__).resolve()), '--summarize'], timeout=60)
        print((root / 'comparison.log').read_text(), flush=True)
        print('No closed-loop runs started; preflight gate remains failed.', flush=True)
    except BaseException as exc:
        (root / 'STOPPED.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise


if __name__ == '__main__':
    main()
