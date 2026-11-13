#!/usr/bin/env python3
"""Sealed revision5 evaluation: main900, sensitivity300, then stop B.

Retry only an engineering abort without any task outcome, at most twice.
All attempts live under superseded; accepted episodes are immutable symlinks.
Any completed-outcome validity failure stops new work immediately.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
import uuid

from audit_fruit_rescue_memory import audit
from fruit_lowlevel_v2_eval_common import (OUT, DOC, SEAL, G1, VAL, TASKS, CONDITIONS, VALID,
                                          verify_seal, terminal_outcome, retry_allowed)
from fruit_lowlevel_v2_metrics_r5 import analyze_directory, event_statistics
from v_a_common import (ECM, POLICY, POLICY_PYTHON, SIM, SIM_PYTHON, CHECKPOINT, OPENPI_DATA_HOME,
                        atomic_json, append_jsonl, read_json, read_jsonl, sha256, timestamp)


class ValidityFailure(RuntimeError):
    pass


def publish(record, **changes):
    state = record
    state.update(changes, updated_utc=timestamp())
    atomic_json(OUT/'status.json', state)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase=5, instruction_revision=5, state=state['state'], updated_utc=state['updated_utc'],
        decision='Revision5 sealed; run main900 and sensitivity300, stop B or on validity failure.',
        next='Complete sealed evaluation and deliver D5 at B; no settings changes.')
    if state['state'] == 'awaiting_review_B':
        root.update(decision='Stop B: sealed main900 and sensitivity300 complete; await user review.', next='Await user review of D5; do not modify paper.')
    elif state['state'] == 'validity_failed_stopped':
        root.update(decision='Validity failure: stopped as required by revision5; engineering diagnostics only.', next='Report failure and await user decision; no automatic remediation.')
    root['links'].update(D5_eval=str(OUT), D5_eval_status=str(OUT/'status.json'), D5_seal=str(SEAL))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)


def start_server(seed, lane, seal):
    session = f'seed{seed}_{uuid.uuid4().hex}'
    d = OUT/'servers'/session
    d.mkdir(parents=True)
    params = read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']
    cfg = {'model_seed': seed, 'replica': 0, 'port': 29621+lane, 'policy_gpu': lane,
        'checkpoint_path': str(CHECKPOINT), 'policy_config': 'mme_vla_suite',
        'openpi_data_home': str(OPENPI_DATA_HOME), 'tasks': list(TASKS),
        'conditions': list(CONDITIONS), 'oracle': params['oracle'], 'checkpoints': seal['checkpoints']}
    atomic_json(d/'config.json', cfg)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(lane), 'JAX_PLATFORMS': 'cuda',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
        'OPENPI_DATA_HOME': str(OPENPI_DATA_HOME), 'MPLCONFIGDIR': str(ECM/'cache/matplotlib'),
        'PYTHONPATH': f'{POLICY}/src:{POLICY}/packages/openpi-client/src'}
    with (d/'server.log').open('x') as log:
        child = subprocess.Popen([str(POLICY_PYTHON), '-u', str(ECM/'scripts/serve_teacher_policy.py'),
            '--config', str(d/'config.json'), '--context', str(d/'context.json')], cwd=POLICY,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    from run_fruit_diagnosis import wait_ready
    try:
        wait_ready([(child, cfg)])
    except BaseException:
        stop_child(child)
        raise
    return {'child': child, 'config': cfg, 'directory': d, 'session': session}


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run_client(command, env, directory, timeout=1800):
    with (directory/'console.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 124


def validate_episode(d, seal, stage, task, episode, seed, condition, server):
    row = read_json(d/'result.json')
    if (row.get('task'), row.get('episode_id'), row.get('model_seed'), row.get('condition')) != (task, episode, seed, condition):
        raise ValidityFailure(f'Condition/layout/seed mismatch: {d}')
    if row.get('error') or row.get('client_exit_code') != 0 or row.get('outcome') not in VALID:
        raise ValidityFailure(f'Client engineering error after task outcome: {d}')
    registry_task = 'count' if task == 'FruitCount' else 'occluded_ripe'
    registered = next(r for r in seal['val_registry'] if r['task'] == registry_task and r['episode'] == episode)
    manifest = ECM/registered['manifest']
    assert sha256(manifest) == registered['sha256']
    if row['environment_seed'] != registered['seed']:
        raise ValidityFailure(f'Environment seed mismatch: {d}')
    if row['joint_guard'] != ('v2' if stage == 'main' else 'off') or row['exec_horizon'] != 16 or row['observe_hold_steps'] != 30 or row['total_step_budget'] != 1300:
        raise ValidityFailure(f'Platform mismatch: {d}')
    cfg = server['config']
    if cfg['model_seed'] != seed or cfg['checkpoints'] != seal['checkpoints']:
        raise ValidityFailure(f'Server seed/checkpoint mismatch: {d}')
    evidence = audit(d, cfg['oracle'])
    atomic_json(d/'memory_audit.json', evidence)
    if not evidence['passed']:
        raise ValidityFailure(f'Memory audit failed after outcome: {d}')
    lifecycle = read_jsonl(d/'policy_lifecycle.jsonl')
    resets = [r for r in lifecycle if r.get('kind') == 'reset']
    if len(resets) != 1 or resets[0].get('model_seed') != seed:
        raise ValidityFailure(f'Policy RNG reset audit failed: {d}')
    _, metrics, classification = analyze_directory(d, seal['q7_center'])
    events = event_statistics(d, cfg['oracle']['keyframes'][condition])
    atomic_json(d/'classification.json', classification)
    atomic_json(d/'event_statistics.json', events)
    if metrics['engineering_error'] or classification['class'] == 'invalid' or not events['event_validity_passed']:
        raise ValidityFailure(f'Unified/classification/event validity failed: {d}')
    raw, controls = read_jsonl(d/'raw_frames.jsonl'), read_jsonl(d/'control_actions.jsonl')
    if len(raw) != len(controls)+1 or len(controls) != row['executed_steps'] or len(controls) > 1300:
        raise ValidityFailure(f'Physical clock mismatch: {d}')
    valid = {'passed': True, 'validated_utc': timestamp(), 'stage': stage, 'task': task, 'episode': episode,
        'condition': condition, 'model_seed': seed, 'server_session': server['session'],
        'seal_sha256': sha256(SEAL), 'memory_decisions': evidence['decisions'],
        'inputs_sha256': {n: sha256(d/n) for n in ('result.json', 'raw_frames.jsonl', 'control_actions.jsonl',
            'memory_audit.json', 'policy_decisions.jsonl', 'oracle_decisions.jsonl', 'event_signal.jsonl',
            'classification.json', 'event_statistics.json')}}
    atomic_json(d/'validity.json', valid)
    return row


def run_group(group, server, lane, seal, stop):
    stage, task, episode, seed = (group[k] for k in ('stage', 'task', 'episode', 'seed'))
    group_id = f'{stage}_{task}_ep{episode}_seed{seed}'
    record_path = OUT/'groups'/f'{group_id}.json'
    if record_path.exists():
        prior = read_json(record_path)
        if prior.get('complete') and prior['seal_sha256'] == sha256(SEAL):
            return server
        raise ValidityFailure(f'Partial group retained for diagnosis, cannot replay completed outcomes: {record_path}')
    record = {'group': group, 'started_utc': timestamp(), 'seal_sha256': sha256(SEAL),
              'complete': False, 'accepted': []}
    atomic_json(record_path, record)
    for condition in group['condition_order']:
        if stop.is_set():
            return server
        accepted = OUT/stage/condition/task/f'ep{episode}_seed{seed}'
        if accepted.exists() or accepted.is_symlink():
            raise ValidityFailure(f'Unexpected pre-existing accepted episode: {accepted}')
        for attempt in range(3):
            if stop.is_set():
                return server
            if server['child'].poll() is not None:
                if record['accepted']:
                    raise ValidityFailure('Server lost mid-group; finished conditions cannot be rerun to restore pairing')
                replacement = start_server(seed, lane, seal)
                server.clear()
                server.update(replacement)
            d = OUT/'superseded'/stage/condition/task/f'ep{episode}_seed{seed}'/f'attempt{attempt}'
            if d.exists():
                raise ValidityFailure(f'Attempt already exists; inspect before further execution: {d}')
            d.mkdir(parents=True)
            command = [str(SIM_PYTHON), '-u', str(ECM/'scripts/evaluate_fruit_lowlevel_v2_episode.py'),
                '--port', str(server['config']['port']), '--context', str(server['directory']/'context.json'),
                '--root', str(VAL), '--task', task, '--episode-id', str(episode), '--condition', condition,
                '--model-seed', str(seed), '--attempt-dir', str(d)]
            if stage == 'main':
                command += ['--guard-stats', str(G1/'guard_v2.json')]
            atomic_json(d/'started.json', {'command': command, 'task': task, 'episode': episode,
                'condition': condition, 'model_seed': seed, 'attempt': attempt, 'utc': timestamp(),
                'group_id': group_id, 'server_session': server['session'], 'seal_sha256': sha256(SEAL),
                'server_config_sha256': sha256(server['directory']/'config.json')})
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(3+lane),
                'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json',
                'MPLCONFIGDIR': str(ECM/'cache/matplotlib'), 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'PYTHONPATH': f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src'}
            code = run_client(command, env, d)
            produced = (d/'result.json').exists()
            row = read_json(d/'result.json') if produced else {'task': task, 'episode_id': episode,
                'condition': condition, 'model_seed': seed, 'outcome': 'error', 'error': f'client exit={code}; result absent'}
            row.update(client_exit_code=code, server_session=server['session'], retry_attempt=attempt)
            atomic_json(d/'result.json', row)
            terminal = terminal_outcome(d)
            ledger = {'utc': timestamp(), 'group': group_id, 'stage': stage, 'task': task,
                'episode': episode, 'condition': condition, 'model_seed': seed, 'attempt': attempt,
                'directory': str(d), 'client_exit_code': code, 'outcome': row.get('outcome'),
                'terminal_task_outcome_found': terminal, 'server_session': server['session']}
            append_jsonl(OUT/f'attempts_seed{seed}.jsonl', ledger)
            if terminal is None:
                engineering_abort = bool(row.get('error')) or code != 0 or row.get('outcome') == 'error' or not produced
                if not engineering_abort or not retry_allowed(d, attempt):
                    raise ValidityFailure(f'No valid task outcome after allowed attempts: {d}')
                print({'retry': ledger, 'reason': 'engineering abort without task outcome'}, flush=True)
                continue
            row = validate_episode(d, seal, stage, task, episode, seed, condition, server)
            accepted.parent.mkdir(parents=True, exist_ok=True)
            accepted.symlink_to(os.path.relpath(d, accepted.parent), target_is_directory=True)
            record['accepted'].append({'condition': condition, 'directory': str(accepted),
                'attempt_directory': str(d), 'outcome': row['outcome'], 'server_session': server['session']})
            atomic_json(record_path, record)
            print({'stage': stage, 'task': task, 'episode': episode, 'seed': seed,
                   'condition': condition, 'outcome': row['outcome'], 'attempt': attempt}, flush=True)
            break
    if not stop.is_set():
        if len(record['accepted']) != 3 or len({r['server_session'] for r in record['accepted']}) != 1:
            raise ValidityFailure(f'Incomplete or mismatched group session: {record_path}')
        record.update(complete=True, completed_utc=timestamp())
        atomic_json(record_path, record)
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--analyze-only', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'groups').mkdir(exist_ok=True)
    lock = (OUT/'eval.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('evaluation already running')
    seal = verify_seal()
    if args.analyze_only:
        from report_fruit_lowlevel_v2_eval import report
        report(seal)
        lock.close()
        return
    state = read_json(OUT/'status.json') if (OUT/'status.json').exists() else {'phase': 5, 'revision': 5}
    if state.get('state') in ('awaiting_review_B', 'validity_failed_stopped'):
        print(state, flush=True)
        lock.close()
        return
    if not args.run:
        publish(state, state='sealed_ready_for_gpu_eval', error='')
        print({'sealed': True, 'main': 900, 'sensitivity': 300, 'stop': 'B'}, flush=True)
        lock.close()
        return
    state.setdefault('started_utc', timestamp())
    servers = {}
    stop = threading.Event()
    try:
        schedule = read_json(OUT/'schedule.json')
        publish(state, state='loading_three_seed_servers')
        for lane, seed in enumerate((7, 8, 9)):
            servers[seed] = start_server(seed, lane, seal)
        for stage in ('main', 'sensitivity'):
            verify_seal()
            publish(state, state=f'evaluating_{stage}')
            seeds = (7, 8, 9) if stage == 'main' else (7,)

            def work(seed):
                lane = seed-7
                try:
                    for group in schedule:
                        if group['stage'] == stage and group['seed'] == seed:
                            if stop.is_set():
                                return
                            servers[seed] = run_group(group, servers[seed], lane, seal, stop)
                except BaseException:
                    stop.set()
                    raise
            with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
                for future in as_completed([pool.submit(work, seed) for seed in seeds]):
                    future.result()
            complete = [read_json(f) for f in (OUT/'groups').glob(stage+'_*.json')]
            expected = 300 if stage == 'main' else 100
            if len(complete) != expected or not all(r['complete'] for r in complete):
                raise ValidityFailure(f'{stage}: incomplete groups')
            publish(state, **{f'{stage}_complete_episodes': expected*3})
        verify_seal()
        from report_fruit_lowlevel_v2_eval import report
        report(seal)
        publish(state, state='awaiting_review_B', error='', completed_utc=timestamp())
    except Exception as exc:
        stop.set()
        publish(state, state='validity_failed_stopped', error=str(exc), traceback=traceback.format_exc())
        from report_fruit_lowlevel_v2_eval import failure_report
        failure_report(seal, state)
        raise
    finally:
        for server in servers.values():
            stop_child(server['child'])
        lock.close()


if __name__ == '__main__':
    main()
