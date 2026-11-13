"""Optional observation/history/RNG digests for isolated inference diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest_tree(value):
    if isinstance(value, dict):
        return {str(k):digest_tree(v) for k,v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [digest_tree(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        raise TypeError(f'Cannot hash object array: {type(value)}')
    return {'shape':list(arr.shape), 'dtype':str(arr.dtype),
            'sha256':hashlib.sha256(arr.tobytes(order='C')).hexdigest()}


def install_probe(policy, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original_infer, original_history = policy.infer, policy._prepare_history
    state = {'index':0}

    def history(inputs):
        result = original_history(inputs)
        state['history'] = digest_tree(result)
        return result

    def infer(obs):
        import jax
        index = state['index']
        state['index'] += 1
        row = {'index':index, 'observation':digest_tree(obs),
               'rng':digest_tree(jax.random.key_data(policy._rng)),
               'selected_step':int(policy.step_idx), 'exec_start_idx':int(policy.exec_start_idx)}
        state.pop('history', None)
        result = original_infer(obs)
        row['history_inputs'] = state.pop('history', None)
        row['actions'] = digest_tree(result['actions'])
        (directory / f'infer_{index:03d}.json').write_text(json.dumps(row, indent=2)+'\n')
        np.save(directory / f'actions_{index:03d}.npy', np.asarray(result['actions']))
        return result

    policy._prepare_history = history
    policy.infer = infer
