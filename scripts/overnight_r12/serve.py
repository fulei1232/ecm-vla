#!/usr/bin/env python3
"""Policy server for the overnight R12 diagnosis: the official MME-VLA server with the memory selector
chosen per episode from a context file written by the client before reset.

  memory=uniform  official even sampling (16 frames), order temporal index
  memory=ka       event_keyframes.select(frames=16, events=12, per_event=release) on the any_hanging distance,
                  read from the client's per-frame trace (simulator geometry, same signal as training)
  memory=none     only the current frame

Usage: serve.py --checkpoint <dir> --port <p> --context <file>
"""
import argparse
import json
import os
from pathlib import Path
import importlib.util
import sys

import numpy as np

ECM = Path('/media/fulei/fuleivla/ecm_vla')
POLICY = Path('/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning')
sys.path[:0] = [str(ECM / 'src')]
from memory_management import event_keyframes as ek  # noqa: E402
from memory_management.policy_adapter import install_temporal_index  # noqa: E402

KA = ek.KeyframeConfig(frames=16, events=12, radius=0.03, tol=0.005, per_event='release')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--context', type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('official_server_entry', POLICY / 'scripts/serve_policy.py')
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    factory = entry.create_policy

    def instrument(args):
        policy = factory(args)
        state = {'memory': 'uniform', 'trace': None}
        original_reset = policy.reset

        def read_distances(n):
            rows = [json.loads(x) for x in Path(state['trace']).read_text().splitlines() if x.strip()]
            if len(rows) < n:
                raise RuntimeError(f'trace has {len(rows)} rows, need {n}')
            return np.asarray([r['distance'] for r in rows[:n]], dtype=np.float64)

        def install_selector():
            buffer = policy.mem_buffer

            def indices(step_idx, token_budget, token_per_image):
                t = int(step_idx)
                if state['memory'] == 'none':
                    return [t]
                if state['memory'] == 'ka':
                    frames, _ = ek.select(read_distances(t + 1), t, KA)
                    return frames
                frames, _ = ek.select(np.full(t + 1, 10.0), t, ek.KeyframeConfig(frames=16, events=0))
                return frames

            buffer.get_frame_sampling_indices = indices

        def reset():
            if a.context.exists():
                state.update(json.loads(a.context.read_text()))
            result = original_reset()  # new buffer; the order wrapper below is re-applied by _prepare_mem_buffer
            install_selector()
            return result

        policy.reset = reset
        install_temporal_index(policy, 'order')
        install_selector()
        if os.environ.get('E4B_PROBE_DIR'):
            from harvestmem_v3.e4b_inference_probe import install_probe
            install_probe(policy, os.environ['E4B_PROBE_DIR'])
        return policy

    entry.create_policy = instrument
    entry.main(entry.Args(seed=7, port=a.port,
                          policy=entry.Checkpoint(config='mme_vla_suite', dir=a.checkpoint)))


if __name__ == '__main__':
    main()
