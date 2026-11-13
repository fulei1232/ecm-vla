#!/usr/bin/env python3
"""Overnight R12 diagnosis queue (runs unattended; resumable).

Phase 1 (already launched separately): K0 and KA train in parallel on GPUs 0-3 / 4-7.
Phase 2: when both finish -> OR (oracle count) on 0-3 and NM (no memory) on 4-7, in parallel.
Evaluation: as soon as a checkpoint exists, evaluate it on dev episodes with policy servers
on whatever GPUs are not training; results -> results/overnight_r12_20261109/<arm>/..., SUMMARY.md rebuilt after
every episode.

Usage: nohup python3 queue.py > queue.log 2>&1 &
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ECM = Path('/media/fulei/fuleivla/ecm_vla')
POLICY = Path('/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning')
POLICY_PY = '/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python'
SIM_PY = '/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python'
RUN = ECM / 'runs/overnight_r12_20261109'
OUT = ECM / 'results/overnight_r12_20261109'
HERE = Path(__file__).resolve().parent
DATASET = ECM / 'data/finetune_kf/fruit_task_v2_r12_20261109'

ARMS = {  # arm: (memory at eval, oracle prompt, training gpus)
    'k0': ('uniform', False, [0, 1, 2, 3]),
    'ka': ('ka', False, [4, 5, 6, 7]),
    'or': ('uniform', True, [0, 1, 2, 3]),
    'nm': ('none', False, [4, 5, 6, 7]),
}
PHASE2 = ('or', 'nm')
# Dev episodes, interleaved so that partial results are balanced across tasks.
EPISODES = [(t, e) for e in range(200, 240) for t in ('count', 'occluded_ripe')]
SERVERS_PER_GPU = 2  # one client per server; 3 exhausted Vulkan device memory alongside training
lock = threading.Lock()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def ckpt(arm):
    return RUN / f'ckpt/mme_vla_suite/{arm}/11999'


def trained(arm):
    return (ckpt(arm) / 'params').exists() and (ckpt(arm) / '_CHECKPOINT_METADATA').exists()


def write_specs():
    base = json.loads((RUN / 'arms/k0.json').read_text())
    for arm, variant in (('or', 'oracle_count'), ('nm', 'no_memory')):
        path = RUN / f'arms/{arm}.json'
        if not path.exists():
            spec = {**base, 'arm': arm, 'variant': variant, 'gpus': ARMS[arm][2]}
            path.write_text(json.dumps(spec, indent=2))


def train(arm):
    spec = RUN / f'arms/{arm}.json'
    gpus = ','.join(map(str, ARMS[arm][2]))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpus, 'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.92',
           'OPENPI_DATA_HOME': '/media/fulei/fuleivla/acm_vla/cache/openpi', 'WANDB_MODE': 'disabled',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': f'{POLICY}/src:{POLICY}/packages/openpi-client/src'}
    script = HERE / 'finetune_variant.py'
    for attempt in range(2):
        log('train start', arm, 'attempt', attempt)
        stale = RUN / f'ckpt/mme_vla_suite/{arm}'
        if stale.exists() and not trained(arm):
            stale.rename(RUN / f'ckpt/mme_vla_suite/{arm}_failed_{int(time.time())}')
        with (RUN / f'logs/{arm}.log').open('a') as f:
            code = subprocess.call([POLICY_PY, '-u', str(script), '--spec', str(spec)], cwd=POLICY, env=env,
                                   stdout=f, stderr=subprocess.STDOUT)
        if trained(arm):
            log('train done', arm)
            return True
        log('train FAILED', arm, 'exit', code)
    return False


def start_server(arm, gpu, port):
    d = OUT / arm / 'servers' / f'gpu{gpu}_port{port}'
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'OPENPI_DATA_HOME': '/media/fulei/fuleivla/acm_vla/cache/openpi', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': f'{POLICY}/src:{POLICY}/packages/openpi-client/src'}
    log_file = (d / 'server.log').open('a')
    child = subprocess.Popen([POLICY_PY, '-u', str(HERE / 'serve.py'), '--checkpoint', str(ckpt(arm)),
                              '--port', str(port), '--context', str(d / 'context.json')],
                             cwd=POLICY, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + 1800
    while time.time() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f'server {arm} gpu{gpu} died; see {d}/server.log')
        try:
            with opener.open(f'http://127.0.0.1:{port}/healthz', timeout=2) as r:
                if r.read() == b'OK\n':
                    return child, d
        except OSError:
            pass
        time.sleep(5)
    child.kill()
    raise RuntimeError('server timeout')


def run_episode(arm, port, sim_gpu, task, episode, server_dir):
    memory, oracle, _ = ARMS[arm]
    d = OUT / arm / task / f'ep{episode}'
    if (d / 'result.json').exists():
        r = json.loads((d / 'result.json').read_text())
        if r.get('outcome') != 'error':
            return
    d.mkdir(parents=True, exist_ok=True)
    context = server_dir / 'context.json'
    cmd = [SIM_PY, '-u', str(HERE / 'eval_episode.py'), '--port', str(port), '--task', task,
           '--episode', str(episode), '--memory', memory, '--context', str(context), '--out', str(d)]
    if oracle:
        cmd.append('--oracle-count')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(sim_gpu), 'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:'
                         f'{POLICY}/third_party/robomme_benchmark/src'}
    with (d / 'client.log').open('w') as f:
        try:
            subprocess.call(cmd, cwd=ECM, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            (d / 'result.json').write_text(json.dumps({'task': task, 'episode': episode, 'outcome': 'error',
                                                         'error': 'client timeout 1800 s'}))


def evaluate(arm, gpus):
    """One server per GPU; each server takes CLIENTS_PER_SERVER episodes at a time.

    The server's memory selector is per episode (context file), but the policy itself is a single
    stream, so clients sharing a server are serialized by a per-server lock around each episode.
    """
    todo = collections.deque(EPISODES)
    qlock = threading.Lock()

    def lane(gpu, port):
        try:
            child, d = start_server(arm, gpu, port)
        except Exception as e:
            log('server failed', arm, gpu, e)
            return
        try:
            while True:
                with qlock:
                    if not todo:
                        return
                    task, episode = todo.popleft()
                for _ in range(3):
                    run_episode(arm, port, gpu, task, episode, d)
                    r = OUT / arm / task / f'ep{episode}' / 'result.json'
                    if r.exists() and json.loads(r.read_text()).get('outcome') != 'error':
                        break
                    time.sleep(20)
                summarize()
        finally:
            child.kill()

    threads = []
    for i, gpu in enumerate(gpus):
        for j in range(SERVERS_PER_GPU):
            port = 29800 + 10 * gpu + j + 100 * list(ARMS).index(arm)
            t = threading.Thread(target=lane, args=(gpu, port), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    log('eval done', arm)


def summarize():
    with lock:
        lines = ['# Overnight R12 diagnosis (dev 200–239, exploratory)', '',
                 f'Updated {time.strftime("%Y-%m-%d %H:%M")}. Arms: K0 uniform | KA event memory | '
                 'OR uniform + told the basket count (ceiling, diagnostic only) | NM no memory.', '',
                 '| arm | task | done | success | rate | fail reasons |', '|---|---|---|---|---|---|']
        strata = []
        for arm in ARMS:
            for task in ('count', 'occluded_ripe'):
                rows = []
                for p in (OUT / arm / task).glob('ep*/result.json'):
                    r = json.loads(p.read_text())
                    if r.get('outcome') != 'error':
                        rows.append(r)
                if not rows:
                    continue
                s = sum(r['outcome'] == 'success' for r in rows)
                reasons = collections.Counter(r.get('reason') or r['outcome'] for r in rows if r['outcome'] != 'success')
                lines.append(f'| {arm.upper()} | {task} | {len(rows)} | {s} | {s/len(rows):.0%} | '
                             + ', '.join(f'{k}:{v}' for k, v in reasons.most_common()) + ' |')
                if task == 'count':
                    by = collections.defaultdict(lambda: [0, 0])
                    for r in rows:
                        band = 'N2-4' if r['N'] <= 4 else 'N5-7' if r['N'] <= 7 else 'N8-10'
                        by[band][0] += r['outcome'] == 'success'
                        by[band][1] += 1
                        if r['outcome'] != 'success' and r.get('reason') in ('overpick', 'finish_signal'):
                            by['memory_fail'][0] += 1
                        by['memory_fail'][1] += 1
                    strata.append(f'| {arm.upper()} | ' + ' | '.join(
                        f'{by[b][0]}/{by[b][1]}' for b in ('N2-4', 'N5-7', 'N8-10', 'memory_fail')) + ' |')
        lines += ['', '## Count by N (success/episodes); memory_fail = overpick or stopped short', '',
                  '| arm | N2-4 | N5-7 | N8-10 | memory_fail |', '|---|---|---|---|---|', *strata, '',
                  'Errors (engineering) are excluded and retried once on resume. Logs: runs/overnight_r12_20261109/logs, '
                  'per-episode: results/overnight_r12_20261109/<arm>/<task>/ep*/']
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def wait_for(arms):
    while not all(trained(a) for a in arms):
        time.sleep(120)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_specs()
    summarize()
    # Phase 1 training (k0, ka) was launched outside this queue; wait for it.
    # Run 2: OR/NM training was started by run 1 and keeps running; evaluate all arms once it finishes.
    log('waiting for or/nm training to finish')
    while any(subprocess.run(['pgrep', '-f', f'arms/{a}.json'], capture_output=True).stdout for a in PHASE2):
        time.sleep(120)
    log('training finished; evaluating')
    ev = [threading.Thread(target=evaluate, args=(arm, gpus)) for arm, gpus in
          (('k0', [0, 1]), ('ka', [2, 3]), ('or', [4, 5]), ('nm', [6, 7])) if trained(arm)]
    for t in ev:
        t.start()
    for t in ev:
        t.join()
    summarize()
    log('ALL DONE')


if __name__ == '__main__':
    main()
