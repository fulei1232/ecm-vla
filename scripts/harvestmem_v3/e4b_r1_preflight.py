#!/usr/bin/env python3
"""Host-side E4b r1 open-loop and smoke checks. Never starts formal episodes."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from e4b_r1_gates import audit
import run_e4b as config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/harvestmem_v3_E4b_r1'
PREFLIGHT_NAME = os.environ.get('E4B_PREFLIGHT', 'preflight')
if not PREFLIGHT_NAME.startswith('preflight') or Path(PREFLIGHT_NAME).name != PREFLIGHT_NAME:
    raise ValueError('E4B_PREFLIGHT must be a directory name beginning with preflight')
PREFLIGHT = OUT / PREFLIGHT_NAME
PORT = 29700
GPUS = os.environ.get('E4B_GPUS', '0,4').split(',')
if len(GPUS) != 2 or any(not gpu.isdigit() for gpu in GPUS) or GPUS[0] == GPUS[1]:
    raise ValueError('E4B_GPUS must contain two distinct GPU indices, e.g. 0,4')
SIM_PY = config.SIM_PY
POLICY_PY = config.POLICY_PY
POLICY = config.POLICY
BASE_ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': GPUS[0], 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
            'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json', 'OMP_NUM_THREADS':'2',
            'OPENBLAS_NUM_THREADS':'2', 'PYTHONDONTWRITEBYTECODE':'1'}
SIM_ENV = {**BASE_ENV, 'PYTHONPATH': f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{ROOT}:{ROOT}/src:{ROOT}/scripts:{ROOT}/scripts/harvestmem_v3'}
SERVER_ENV = {**BASE_ENV, 'PYTHONPATH': f'{POLICY}/src:{POLICY}/packages/openpi-client/src:{ROOT}/src:{ROOT}/scripts'}


def run_logged(label, cmd, env=SIM_ENV, timeout=None):
    path = PREFLIGHT / f'{label}.log'
    print(f'[{label}] started; log: {path}', flush=True)
    with path.open('x') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)
    print(f'[{label}] complete', flush=True)


@contextmanager
def server(label, checkpoint, gpu, port):
    sd = PREFLIGHT / f'server_{label}'
    sd.mkdir()
    context = sd / 'context.json'
    command = [POLICY_PY, '-u', str(ROOT / 'scripts/overnight_r12/serve.py'),
               '--checkpoint', checkpoint, '--port', str(port), '--context', str(context)]
    with (sd / 'server.log').open('x') as log:
        proc = subprocess.Popen(command, cwd=POLICY, env={**SERVER_ENV, 'CUDA_VISIBLE_DEVICES':gpu}, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'{label} server exited {proc.returncode}; see {sd}/server.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{label} server did not become healthy')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if response.read() == b'OK\n':
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(2)
            print(f'[{label}] server ready', flush=True)
            yield context
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def smoke(context, arm, task, ep, planner, stop):
    path = PREFLIGHT / 'smoke' / arm / task / f'ep{ep}'
    path.mkdir(parents=True)
    cmd = [SIM_PY, '-u', str(ROOT / 'scripts/harvestmem_v3/eval_episode.py'),
           '--port', str(PORT), '--task', task, '--episode', str(ep), '--arm', arm,
           '--checkpoint', config.CK['DP1R'], '--memory', 'uniform', '--context', str(context),
           '--out', str(path), '--planner-reach', planner]
    if stop:
        cmd.append('--enforce-stop')
    if planner != 'off':
        cmd.append('--capture-handoff-frames')
    run_logged(f'smoke_{arm}_{task}_{ep}', cmd, timeout=1800)
    return path


def main():
    subprocess.run(['nvidia-smi', '-L'], check=True)
    for port in (PORT, PORT + 1):
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'port {port} is in use; stop and choose a free port in this script') from exc
    PREFLIGHT.mkdir(parents=True, exist_ok=False)
    try:
        verified = OUT / 'reset_ripe_verification.json'
        if verified.exists():
            rows = json.loads(verified.read_text())
            expected = json.loads((OUT / 'seeded_ripe.json').read_text())['episodes']
            assert len(rows) == 10 and {r['episode'] for r in rows} == set(range(250, 260))
            assert all(r['seed'] == expected[str(r['episode'])]['seed'] and
                       r['ripe'] == expected[str(r['episode'])]['ripe'] for r in rows)
            print('[reset_ripe] reusing verified resets for ep250–259', flush=True)
        else:
            run_logged('reset_ripe', [SIM_PY, '-u', str(ROOT / 'scripts/harvestmem_v3/e4b_r1_reset_ripe.py')], timeout=600)
        def open_loop(arm, gpu, port):
            checkpoint = config.CK['DP1R' if arm == 'k0' else 'EP1R']
            with server(f'open_loop_{arm}', checkpoint, gpu, port) as context:
                run_logged(f'open_loop_{arm}', [SIM_PY, '-u', str(ROOT / 'scripts/harvestmem_v3/e4b_reach_open_loop.py'),
                    '--arm', arm, '--port', str(port), '--context', str(context),
                    '--out', str(PREFLIGHT / f'open_loop_{arm}.json')],
                    env={**SIM_ENV, 'CUDA_VISIBLE_DEVICES':gpu})
                data = json.loads((PREFLIGHT / f'open_loop_{arm}.json').read_text())
                if not data['details'] or data['aggregate_ratio'] is None:
                    raise RuntimeError(f'{arm} open-loop result has no usable boundaries')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(open_loop, 'k0', GPUS[0], PORT),
                       pool.submit(open_loop, 'nm', GPUS[1], PORT + 1)]
            for future in futures:
                future.result()
        with server('smoke_k0', config.CK['DP1R'], GPUS[0], PORT) as context:
            gates = []
            specs = [('DP1R', 'count', 202, 'first', True), ('DP1R', 'count', 207, 'first', True),
                     ('DPa', 'occluded_ripe', 200, 'all', False), ('DPa', 'occluded_ripe', 201, 'all', False)]
            for spec in specs:
                path = smoke(context, *spec)
                result = audit(path)
                screenshots = [str(path / f'handoff_000_plus_{offset:02d}.png') for offset in (0, 8, 16)]
                result['first_handoff_screenshots_present'] = all(Path(p).is_file() for p in screenshots)
                gates.append(result)
                print(f'[{path.parent.parent.name} {path.name}] jump/distance gate: {result["pass"]}; screenshots: {result["first_handoff_screenshots_present"]}', flush=True)
                if not result['pass'] or not result['first_handoff_screenshots_present']:
                    (PREFLIGHT / 'gates.json').write_text(json.dumps({'episodes':gates, 'all_pass':False}, indent=2)+'\n')
                    raise RuntimeError('smoke gate failed; no formal runs started')
            controls = []
            for ep in (201, 206, 210):
                path = smoke(context, 'DR', 'count', ep, 'off', True)
                actual = json.loads((path / 'result.json').read_text())
                baseline = json.loads((ROOT / f'results/harvestmem_v3_E4a/DR/count/ep{ep}/result.json').read_text())
                same = (actual['outcome'], actual['reason']) == (baseline['outcome'], baseline['reason'])
                controls.append({'episode':ep, 'same_outcome_and_reason':same,
                                 'actual':[actual['outcome'],actual['reason']],
                                 'baseline':[baseline['outcome'],baseline['reason']]})
                print(f'[DR ep{ep}] matches E4a: {same}', flush=True)
                if not same:
                    raise RuntimeError(f'DR ep{ep} does not match E4a; no formal runs started')
        (PREFLIGHT / 'gates.json').write_text(json.dumps({'episodes':gates, 'controls':controls, 'all_pass':True}, indent=2)+'\n')
        print('PREFLIGHT PASSED. No formal episodes were started. Share gates.json and open-loop results.', flush=True)
    except BaseException as exc:
        (PREFLIGHT / 'STOPPED.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        raise


if __name__ == '__main__':
    main()
