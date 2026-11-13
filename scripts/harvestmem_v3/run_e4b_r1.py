#!/usr/bin/env python3
"""Eight-lane, six-arm E4b r1 queue with auditable admission and stopping gates."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request

import run_e4b as config
from e4b_r1_gates import audit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/harvestmem_v3_E4b_r1'
PREFLIGHT = OUT / 'preflight_r2'
ARMS = ('DP1R', 'DPaR', 'DPa', 'EP1R', 'EPaR', 'EPa')
GROUPS = (ARMS[:3], ARMS[3:])
PORT_BASE = 29800
SOURCES = ('scripts/harvestmem_v3/eval_episode.py', 'scripts/harvestmem_v3/oracle.py',
           'scripts/harvestmem_v3/subgoal.py', 'scripts/harvestmem_v3/e4b_r1_gates.py',
           'scripts/harvestmem_v3/run_e4b_r1.py', 'scripts/harvestmem_v3/run_e4b.py',
           'scripts/overnight_r12/serve.py', 'fruit_picking/task_v2_env.py',
           'src/memory_management/fruit_rescue_platform.py',
           'src/memory_management/policy_adapter.py', 'src/memory_management/event_keyframes.py')
LOCK = threading.Lock()
STOP = threading.Event()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {name:sha(ROOT / name) for name in SOURCES}


def admit():
    specs = [('DP1R','count',202), ('DP1R','count',207),
             ('DPa','occluded_ripe',200), ('DPa','occluded_ripe',201)]
    smoke, controls, hashes = [], [], {}
    for arm, task, ep in specs:
        path = PREFLIGHT / 'smoke' / arm / task / f'ep{ep}'
        result = audit(path)
        if not result['pass']:
            raise RuntimeError(f'smoke engineering gate failed: {path}')
        for i, _ in enumerate(read(path / 'result.json')['handoffs']):
            for offset in (0, 8, 16):
                if not (path / f'handoff_{i:03d}_plus_{offset:02d}.png').is_file():
                    raise RuntimeError(f'missing handoff screenshot: {path} {i}/{offset}')
        smoke.append(result)
        for name in ('result.json','actions.jsonl','trace.jsonl'):
            p = path / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    for ep in (201,206,210):
        path = PREFLIGHT / 'smoke/DR/count' / f'ep{ep}/result.json'
        actual = read(path)
        baseline = read(ROOT / f'results/harvestmem_v3_E4a/DR/count/ep{ep}/result.json')
        if actual.get('error') or actual.get('planner_reach') != 'off':
            raise RuntimeError(f'invalid control result: {path}')
        if not audit(path.parent)['count_stop_pass']:
            raise RuntimeError(f'Count stopping gate failed in control: {path}')
        controls.append({'episode':ep, 'actual':[actual['outcome'],actual['reason']],
                         'baseline':[baseline['outcome'],baseline['reason']],
                         'matches':(actual['outcome'],actual['reason']) == (baseline['outcome'],baseline['reason'])})
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for arm in ('k0','nm'):
        path = PREFLIGHT / f'open_loop_{arm}.json'
        data = read(path)
        if not data['details'] or data['aggregate_ratio'] is None:
            raise RuntimeError(f'missing open-loop result: {path}')
        hashes[str(path.relative_to(ROOT))] = sha(path)
    verified = OUT / 'reset_ripe_verification.json'
    rows = read(verified)
    expected = read(OUT / 'seeded_ripe.json')['episodes']
    if len(rows) != 10 or {r['episode'] for r in rows} != set(range(250,260)):
        raise RuntimeError('incomplete seed reset verification')
    if not all(r['ripe'] == expected[str(r['episode'])]['ripe'] and
               r['seed'] == expected[str(r['episode'])]['seed'] for r in rows):
        raise RuntimeError('seed reset verification mismatch')
    hashes[str(verified.relative_to(ROOT))] = sha(verified)
    for name in ('preflight_frozen_probe/comparison.json',
                 'preflight_control_diagnostic/diagnostic.json'):
        path = OUT / name
        hashes[str(path.relative_to(ROOT))] = sha(path)
    return {'smoke':smoke, 'controls':controls, 'artifact_hashes':hashes,
            'source_hashes':source_hashes(), 'arms':list(ARMS), 'episodes':list(range(200,260)),
            'control_mismatch_policy': 'User explicitly authorized recording DR ep210 mismatch as nonblocking after fixed-input probe; reuse existing controls.',
            'controls_reused':{'DR':'results/harvestmem_v3_E4a/DR/count',
                               'ER':'results/harvestmem_v3_E4a/ER/count',
                               'D':'results/harvestmem_v3_E4b/D/occluded_ripe',
                               'E':'results/harvestmem_v3_E4b/E/occluded_ripe'},
            'preregistration':'reports/harvestmem_v3_codex_tasks_20261112_E4b.md',
            'new_thresholds':{'handoff_jump_rad':.3,'handoff_jump_pass_rate':.95,
                              'first_handoff_distance8_m':.20,'first_handoff_distance8_pass_rate':.90},
            'autostop':'Per arm, before 60 episodes use failure-count allowances with planner denominator max(calls,60), jump denominator max(jump_n,60), and distance8 denominator 60. At 60 episodes use actual denominators. Count stop violations and finish before first completed reach stop immediately. No new jobs start after failure; in-flight episodes finish and are retained.'}


def engineering(records, audits):
    problems = []
    totals = {}
    for arm in ARMS:
        rows = [r for r in records if r['arm'] == arm]
        aa = [a for a in audits if a['arm'] == arm]
        calls = sum(r['planner_calls'] for r in rows)
        failures = sum(len(r['planner_failures']) for r in rows)
        jumps = [c for a in aa for c in a['checks'] if c['policy_at_end']]
        first = [a['checks'][0] for a in aa if a['checks']]
        counts = {'episodes':len(rows), 'planner_calls':calls, 'planner_failures':failures,
                  'jump_good':sum(c['jump_pass'] for c in jumps), 'jump_n':len(jumps),
                  'distance8_good':sum(c.get('distance8_pass', False) for c in first), 'distance8_n':len(first)}
        totals[arm] = counts
        n = counts['episodes']
        final = n == 60
        jump_bad = counts['jump_n'] - counts['jump_good']
        distance_bad = counts['distance8_n'] - counts['distance8_good']
        # Use the 60-episode allowances until the arm is complete; then actual denominators.
        if failures > .05 * (calls if final else max(calls, 60)):
            problems.append(f'{arm}: planner failure rate {failures}/{calls} > 5%')
        if jump_bad > .05 * (counts['jump_n'] if final else max(counts['jump_n'], 60)):
            problems.append(f'{arm}: handoff action continuity below 95% ({counts["jump_good"]}/{counts["jump_n"]})')
        if distance_bad > .10 * (counts['distance8_n'] if final else 60):
            problems.append(f'{arm}: first handoff distance8 pass rate below 90% ({counts["distance8_good"]}/{counts["distance8_n"]})')
        for r, a in zip(rows, aa):
            if not a['count_stop_pass']:
                problems.append(f'{arm} ep{r["episode"]}: Count stop gate failed')
            if r.get('reason') == 'finish_signal' and not r.get('handoffs'):
                problems.append(f'{arm} ep{r["episode"]}: finish before first completed reach')
    return totals, problems


def summarize(recover=False):
    records, audits = [], []
    for arm in ARMS:
        for path in sorted((OUT / arm / config.TASKS[arm]).glob('ep*/result.json')):
            ap = path.parent / 'engineering.json'
            if not ap.exists() and not recover:
                # Other lanes can be writing a result before its audit is committed.
                continue
            row = read(path)
            if row.get('error') or row.get('outcome') not in ('success','fail','timeout'):
                raise RuntimeError(f'incomplete/error result retained for inspection: {path}')
            if not ap.exists():
                a = audit(path.parent)
                write(ap, {**a, 'arm':arm, 'episode':row['episode']})
            records.append(row)
            audits.append(read(ap))
    totals, problems = engineering(records, audits)
    write(OUT / 'rows.json', records)
    write(OUT / 'engineering_totals.json', {'arms':totals,'problems':problems})
    lines = ['# E4b r1 live summary', '', 'Only six planner arms; existing controls reused.', '',
             '| Arm | Done | Success | Planner failures/calls | Jump good/n | Distance8 good/n |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        t = totals[arm]
        nsuccess = sum(r['arm'] == arm and r['outcome'] == 'success' for r in records)
        lines.append(f'| {arm} | {t["episodes"]}/60 | {nsuccess} | {t["planner_failures"]}/{t["planner_calls"]} | {t["jump_good"]}/{t["jump_n"]} | {t["distance8_good"]}/{t["distance8_n"]} |')
    if problems:
        lines += ['', 'Engineering stop:', *[f'- {p}' for p in problems]]
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return records, problems


def env_for(gpu, server=False):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'PYTHONDONTWRITEBYTECODE':'1',
           'XLA_PYTHON_CLIENT_PREALLOCATE':'false','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
           'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json'}
    env.pop('E4B_PROBE_DIR', None)
    policy = config.POLICY
    env['PYTHONPATH'] = (f'{policy}/src:{policy}/packages/openpi-client/src:{ROOT}/src:{ROOT}/scripts' if server else
                        f'{policy}/examples/robomme:{policy}/packages/openpi-client/src:{ROOT}:{ROOT}/src:{ROOT}/scripts:{ROOT}/scripts/harvestmem_v3')
    return env


@contextmanager
def server_for(gpu, model, run_dir):
    sd = run_dir / f'{model}_gpu{gpu}'
    sd.mkdir()
    port = PORT_BASE + gpu
    checkpoint = config.CK['DP1R' if model == 'k0' else 'EP1R']
    with (sd / 'server.log').open('x') as log:
        proc = subprocess.Popen([config.POLICY_PY,'-u',str(ROOT/'scripts/overnight_r12/serve.py'),
            '--checkpoint',checkpoint,'--port',str(port),'--context',str(sd/'context.json')],
            cwd=config.POLICY, env=env_for(gpu, server=True), stdout=log, stderr=subprocess.STDOUT)
        write(sd/'provenance.json', {'pid':proc.pid,'gpu':gpu,'model':model,'port':port,
              'checkpoint':checkpoint,'environment':{k:v for k,v in env_for(gpu, server=True).items()
                  if k.startswith(('CUDA','XLA','JAX','CUBLAS')) or k == 'PYTHONHASHSEED'}})
        try:
            deadline = time.monotonic() + 900
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                if STOP.is_set():
                    raise RuntimeError('another lane stopped during server startup')
                if proc.poll() is not None:
                    raise RuntimeError(f'{model} GPU{gpu} server exited; inspect {sd}/server.log')
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{model} GPU{gpu} server startup timeout')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:
                        if response.read() == b'OK\n':
                            break
                except OSError:
                    time.sleep(2)
            yield sd, port
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()


def lane(gpu, model, jobs, run_dir, hashes):
    try:
        with server_for(gpu, model, run_dir) as (sd, port):
            while not STOP.is_set():
                with LOCK:
                    if STOP.is_set() or not jobs:
                        return
                    arm, ep = jobs.popleft()
                if source_hashes() != hashes:
                    raise RuntimeError('evaluation code changed while queue was active')
                path = OUT / arm / config.TASKS[arm] / f'ep{ep}'
                path.mkdir(parents=True, exist_ok=False)
                command = [config.SIM_PY,'-u',str(config.SCRIPT),'--port',str(port),'--task',config.TASKS[arm],
                           '--episode',str(ep),'--arm',arm,'--checkpoint',config.CK[arm],
                           '--memory',config.MEM[arm],'--context',str(sd/'context.json'),'--out',str(path),
                           '--planner-reach',config.PLAN[arm]]
                if arm.endswith('R'):
                    command.append('--enforce-stop')
                write(path/'launch.json', {'gpu':gpu, 'port':port, 'command':command,
                                          'server':str(sd.relative_to(ROOT))})
                with (path/'client.log').open('x') as log:
                    subprocess.run(command,cwd=ROOT,env=env_for(gpu),stdout=log,stderr=subprocess.STDOUT,
                                   timeout=1800,check=True)
                result = read(path/'result.json')
                if result.get('error') or result.get('outcome') not in ('success','fail','timeout'):
                    raise RuntimeError(f'episode error: {path}')
                checked = audit(path)
                with LOCK:
                    write(path/'engineering.json', {**checked, 'arm':arm,'episode':ep})
                    rows, problems = summarize()
                    print(f'{len(rows)}/360 {arm} ep{ep} {result["outcome"]}/{result["reason"]}',flush=True)
                    if problems:
                        STOP.set()
                        write(run_dir/'STOPPED.json', {'problems':problems, 'completed':len(rows)})
    except BaseException as exc:
        STOP.set()
        with LOCK:
            write(run_dir/f'ERROR_gpu{gpu}.json', {'type':type(exc).__name__,'error':str(exc)})
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-only',action='store_true')
    args = ap.parse_args()
    admission = admit()
    approval = OUT/'formal_admission.json'
    if approval.exists() and read(approval) != admission:
        raise RuntimeError('formal admission inputs or code changed; inspect before resuming')
    if not approval.exists():
        write(approval, admission)
    if args.prepare_only:
        print('Admission checks passed. Control mismatch recorded as user-authorized nonblocking. No GPU jobs started.')
        return
    subprocess.run(['nvidia-smi','-L'],check=True)
    busy = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name',
                           '--format=csv,noheader'],capture_output=True,text=True,check=True)
    if busy.stdout.strip():
        raise RuntimeError('GPU compute processes already present; inspect before starting eight dedicated lanes:\n' + busy.stdout)
    with (OUT/'.formal_queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous, problems = summarize(recover=True)
        if problems:
            raise RuntimeError('retained engineering results fail gates; inspect before resuming')
        complete = {(r['arm'],r['episode']) for r in previous}
        if len(complete) == 360:
            print('All 360 formal results already exist. Stop for review.'); return
        for gpu in range(8):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',PORT_BASE+gpu))
        run_dir = OUT/'formal_servers'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        run_dir.mkdir(parents=True)
        with (run_dir/'gpu_inventory.csv').open('x') as inventory:
            subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total',
                            '--format=csv'],stdout=inventory,check=True)
        signals = {s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        def stop(signum, frame):
            STOP.set()
            print('Stop requested: no new episodes; retaining in-flight results.',flush=True)
        for sig in signals:
            signal.signal(sig,stop)
        jobs = [deque((arm,ep) for ep in range(200,260) for arm in group if (arm,ep) not in complete)
                for group in GROUPS]
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(lane,gpu,'k0' if gpu<4 else 'nm',jobs[0 if gpu<4 else 1],
                                       run_dir,admission['source_hashes']) for gpu in range(8)]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:
                        errors.append(str(exc))
            rows, problems = summarize()
            status = {'completed':len(rows),'expected':360,'problems':problems,'errors':errors,
                      'stopped':STOP.is_set(),'review_required':True}
            write(OUT/'formal_status.json',status)
            print(json.dumps(status,ensure_ascii=False),flush=True)
            if errors or problems or len(rows) != 360:
                raise RuntimeError('queue stopped before successful completion; inspect logs')
            print('360 episodes complete. Stop for review; no C arm, training, or further runs started.',flush=True)
        finally:
            for sig, handler in signals.items():
                signal.signal(sig,handler)


if __name__ == '__main__':
    main()
