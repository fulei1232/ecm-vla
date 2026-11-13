#!/usr/bin/env python3
"""Repeat the failed DR control check; never changes the preflight gate."""
import json
import os
from pathlib import Path
import socket
import subprocess

os.environ.setdefault('E4B_PREFLIGHT', 'preflight_control_diagnostic')
import e4b_r1_preflight as preflight

ROOT = preflight.ROOT
OUT = preflight.PREFLIGHT


def actions(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    subprocess.run(['nvidia-smi', '-L'], check=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', preflight.PORT))
    OUT.mkdir(parents=True, exist_ok=False)
    baseline = ROOT / 'results/harvestmem_v3_E4a/DR/count/ep210'
    old_result = json.loads((baseline / 'result.json').read_text())
    old_action = actions(baseline / 'actions.jsonl')[0]
    records = []
    try:
        with preflight.server('control_diagnostic', preflight.config.CK['DP1R'],
                              preflight.GPUS[0], preflight.PORT) as context:
            for attempt in range(3):
                path = OUT / f'repeat_{attempt}'
                path.mkdir()
                command = [preflight.SIM_PY, '-u', str(ROOT / 'scripts/harvestmem_v3/eval_episode.py'),
                    '--port', str(preflight.PORT), '--task', 'count', '--episode', '210', '--arm', 'DR',
                    '--checkpoint', preflight.config.CK['DP1R'], '--memory', 'uniform', '--context', str(context),
                    '--out', str(path), '--planner-reach', 'off', '--enforce-stop']
                preflight.run_logged(f'control_210_repeat_{attempt}', command, timeout=1800)
                result = json.loads((path / 'result.json').read_text())
                first = actions(path / 'actions.jsonl')[0]
                diff = max(abs(a - b) for a, b in zip(first['action'][:7], old_action['action'][:7]))
                entry = {'attempt':attempt, 'outcome':result['outcome'], 'reason':result['reason'],
                         'steps':result['steps'], 'first_action_max_abs_delta_from_e4a_rad':diff,
                         'matches_e4a':(result['outcome'], result['reason']) ==
                                       (old_result['outcome'], old_result['reason'])}
                records.append(entry)
                (OUT / 'diagnostic.json').write_text(json.dumps({'baseline': {'outcome':old_result['outcome'],
                    'reason':old_result['reason'], 'steps':old_result['steps']}, 'runs':records,
                    'preflight_gate': 'failed; unchanged'}, indent=2) + '\n')
                print(json.dumps(entry), flush=True)
        print('Diagnostic complete. Preflight remains failed; no formal episodes were started.', flush=True)
    except BaseException as exc:
        (OUT / 'STOPPED.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise


if __name__ == '__main__':
    main()
