#!/usr/bin/env python3
"""Revision6 exploratory diagnosis only. No training or new val rollouts.

Prepare CPU posthoc evidence, reproduce five G1 cases, then dev220–259 seeds7/8.
All sealed sources/results remain untouched; inference switches default off.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import uuid

import numpy as np

from audit_fruit_memory_diag import audit
from fruit_memory_diag_core import OUT, G1, LABEL, MODES, ABLATIONS, TASKS
from posthoc_fruit_memory_usage import begin, posthoc, verify_preservation
from v_a_common import (ECM, POLICY, SIM, POLICY_PYTHON, SIM_PYTHON, CHECKPOINT, OPENPI_DATA_HOME,
    atomic_json, read_json, read_jsonl, sha256, timestamp)

DEV = ECM/'data/fruit_picking_dev/rm75b_eg2/train'
DOC = ECM/'reports/fruit_memory_usage_diag_plan_20261104.md'


def publish(record, **changes):
    record.update(changes, updated_utc=timestamp())
    atomic_json(OUT/'status.json', record)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase=6, instruction_revision=6, state=record['state'], updated_utc=record['updated_utc'],
        decision='Exploratory memory-use diagnosis only; D5 confirmatory conclusions unchanged.',
        next='Complete §14.3/14.4 and deliver D6 with §14.5 verdict; stop for review.')
    if record['state'] == 'awaiting_D6_review':
        root.update(decision='D6 exploratory diagnosis complete; stop for user review. D5 conclusions unchanged.', next='Await user review of D6; no task redesign or paper edits.')
    elif record['state'] == 'diagnosis_stopped':
        root.update(decision='D6 stopped; report preserved exploratory evidence and reason.', next='Await user review; do not infer an A/B/C verdict from incomplete data.')
    root['links'].update(D6_status=str(OUT/'status.json'), D6_plan=str(DOC), D6_results=str(OUT))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)


def remaining(plan):
    return (datetime.fromisoformat(plan['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()


def check_deadline(plan):
    if remaining(plan) <= 0:
        raise TimeoutError('Revision6 one-day deadline reached; stop and deliver partial evidence')


def verify_sources(plan):
    for path, digest in plan['source_sha256'].items():
        assert sha256(path) == digest, path
    assert sha256(DOC) == plan['plan_doc_sha256']
    assert sha256(ECM/'reports/split_registry_20261102.json') == plan['registry_sha256']
    assert sha256(Path(plan['checkpoint'])/'params/manifest.ocdbt') == plan['checkpoint_manifest_sha256']


def prepare():
    state = begin()
    post = posthoc()
    files = [ECM/'scripts'/n for n in ('fruit_memory_diag_core.py', 'posthoc_fruit_memory_usage.py',
        'serve_fruit_memory_diag.py', 'evaluate_fruit_memory_diag.py', 'audit_fruit_memory_diag.py',
        'run_fruit_memory_usage_diag.py', 'report_fruit_memory_usage.py')]
    files += [ECM/'src/memory_management/fruit_memory_diag.py']
    files += [Path(p) for p in read_json(OUT/'preservation/sealed_files_sha256.json') if Path(p).suffix == '.py']
    g1 = read_json(G1/'G1_continuation_plan.json')
    rng = np.random.default_rng(20261104)
    pilot = []
    for task, n in (('FruitCount', 2), ('FruitOccludedRipe', 3)):
        pilot += [{'task': task, 'episode': int(e), 'seed': 7} for e in sorted(rng.choice(np.arange(220, 260), n, replace=False).tolist())]
    schedule = [{'task': task, 'episode': e, 'seed': seed,
                 'conditions': rng.permutation(list(MODES)).tolist()}
                for seed in (7, 8) for e in range(220, 260) for task in TASKS]
    plan = {'phase': 6, 'revision': 6, 'label': LABEL, 'started_utc': state['started_utc'],
        'deadline_utc': state['deadline_utc'], 'checkpoint': g1['checkpoint'],
        'checkpoint_manifest_sha256': g1['checkpoint_manifest_sha256'],
        'guard_sha256': g1['guard_stats_sha256'], 'registry_sha256': g1['registry_sha256'],
        'source_sha256': {str(p): sha256(p) for p in dict.fromkeys(files)},
        'plan_doc_sha256': sha256(DOC), 'pilot_cases': pilot, 'schedule': schedule,
        'conditions': dict(ABLATIONS), 'tasks': list(TASKS), 'episodes': list(range(220, 260)), 'model_seeds': [7, 8],
        'platform': {'guard': 'v2', 'exec_horizon': 16, 'observe_hold_steps': 30, 'total_step_budget': 1300},
        'all_switches_default_off': True, 'new_val_rollouts_allowed': False,
        'repro_rule': 'All5 physical/model bitwise comparisons match -> reuse80 G1 seed7 cases; else run all80 normal seed7 anew (pilot cases may count).',
        'pilot_bitwise_fields': ['proposed/executed/measured actions', 'all GT and image/state hashes', 'policy output hashes', 'memory selection/mask/tensor hashes'],
        'metadata_excluded_from_bitwise': ['timestamps', 'file paths', 'process/session ids', 'video container bytes'],
        'primary_rule_metric': 'I(any hidden ripe never within15cm)+I(any unripe detached or wrong fruit in basket)',
        'thresholds': {'A_loss_or_error_rise_pp': 10, 'B_both_signed_changes_strictly_below_pp': 5},
        'D5_conclusions_unchanged': True}
    path = OUT/'plan.json'
    if path.exists():
        assert read_json(path) == plan, 'Diagnostic definition/source changed; do not mix runs'
    else:
        atomic_json(path, plan)
    verify_sources(plan)
    return state, plan


def start_server(seed, lane, plan, pilot=False):
    check_deadline(plan)
    d = OUT/'servers'/f'{"pilot" if pilot else "diagnostic"}_seed{seed}_lane{lane}_{uuid.uuid4().hex}'
    d.mkdir(parents=True)
    params = read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']
    cfg = {'model_seed': seed, 'replica': 0, 'port': 29721+lane, 'policy_gpu': lane,
        'checkpoint_path': str(CHECKPOINT), 'policy_config': 'mme_vla_suite',
        'openpi_data_home': str(OPENPI_DATA_HOME), 'tasks': list(TASKS), 'conditions': ['K0'],
        'oracle': params['oracle'], 'checkpoints': {'K0': plan['checkpoint']}}
    atomic_json(d/'config.json', cfg)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(lane), 'JAX_PLATFORMS': 'cuda',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
        'OPENPI_DATA_HOME': str(OPENPI_DATA_HOME), 'MPLCONFIGDIR': str(ECM/'cache/matplotlib'),
        'PYTHONPATH': f'{POLICY}/src:{POLICY}/packages/openpi-client/src'}
    command = [str(POLICY_PYTHON), '-u', str(ECM/'scripts'/('serve_teacher_policy.py' if pilot else 'serve_fruit_memory_diag.py')),
        '--config', str(d/'config.json'), '--context', str(d/'context.json')]
    if not pilot:
        command.append('--enable-memory-diagnostics')
    with (d/'server.log').open('x') as log:
        child = subprocess.Popen(command, cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    from run_fruit_diagnosis import wait_ready
    try:
        wait_ready([(child, cfg)], timeout=max(1, min(1800, remaining(plan))))
    except BaseException:
        stop_server(child);raise
    return {'child': child, 'config': cfg, 'directory': d}


def stop_server(child):
    if child.poll() is None:
        child.terminate()
        try: child.wait(timeout=30)
        except subprocess.TimeoutExpired: child.kill();child.wait()


def raw_case(task, episode, seed, condition, server, lane, plan, pilot=False):
    check_deadline(plan)
    d = OUT/('pilot' if pilot else f'replays/{condition}')/task/f'ep{episode}_seed{seed}'
    meta_dir = OUT/'audits'/('pilot' if pilot else condition)/task/f'ep{episode}_seed{seed}'
    meta_dir.mkdir(parents=True, exist_ok=True)
    if (meta_dir/'summary.json').exists():
        evidence = read_json(meta_dir/'summary.json')
        assert evidence['passed'] and evidence['plan_sha256'] == sha256(OUT/'plan.json')
        return d
    if d.exists() and any(d.iterdir()):
        raise RuntimeError(f'Partial diagnostic case retained for inspection: {d}')
    d.mkdir(parents=True, exist_ok=True)
    mode = ABLATIONS[condition]
    command = [str(SIM_PYTHON), '-u', str(ECM/'scripts'/('evaluate_fruit_rescue_episode.py' if pilot else 'evaluate_fruit_memory_diag.py')),
        '--port', str(server['config']['port']), '--context', str(server['directory']/'context.json'),
        '--root', str(DEV), '--task', task, '--episode-id', str(episode), '--attempt-dir', str(d),
        '--guard-stats', str(G1/'guard_v2.json')]
    if not pilot:
        command += ['--condition', 'K0', '--model-seed', str(seed)]
        if mode is not None: command += ['--memory-ablation', mode]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(lane if pilot else 4+lane),
        'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json', 'MPLCONFIGDIR': str(ECM/'cache/matplotlib'),
        'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'PYTHONPATH': f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src'}
    atomic_json(d/'diagnostic_started.json', {'command': command, 'task': task, 'episode': episode,
        'model_seed': seed, 'condition': condition, 'ablation': mode, 'pilot': pilot, 'utc': timestamp(),
        'plan_sha256': sha256(OUT/'plan.json'), 'server_config_sha256': sha256(server['directory']/'config.json')})
    with (d/'console.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=max(1, min(1800, remaining(plan))))
        except subprocess.TimeoutExpired:
            import signal
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL);child.wait()
            raise TimeoutError(f'Diagnostic client timed out: {d}')
    if code:
        raise RuntimeError(f'Diagnostic client failed exit={code}: {d}')
    save_audit(d, meta_dir, mode, server['config']['oracle'], plan, task, episode, seed, condition, reused=False)
    print({'diagnostic': condition, 'task': task, 'episode': episode, 'seed': seed, 'passed': True, 'pilot': pilot}, flush=True)
    return d


def save_audit(d, meta_dir, mode, oracle, plan, task, episode, seed, condition, reused):
    r = read_json(d/'result.json')
    assert (r['task'], r['episode_id'], r['model_seed'], r['condition']) == (task, episode, seed, 'K0')
    assert not r.get('error') and r['outcome'] in ('success', 'fail', 'timeout')
    assert r['joint_guard'] == 'v2' and r['exec_horizon'] == 16 and r['observe_hold_steps'] == 30 and r['total_step_budget'] == 1300
    man = DEV/('count' if task == 'FruitCount' else 'occluded_ripe')/f'ep{episode}_manifest.json'
    assert r['environment_seed'] == read_json(man)['seed']
    proof = audit(d, mode, oracle)
    atomic_json(meta_dir/'memory_audit.json', proof)
    if not proof['passed']:
        raise RuntimeError(f'Diagnostic memory audit failed: {meta_dir}')
    lifecycle = [v for v in read_jsonl(d/'policy_lifecycle.jsonl') if v.get('kind') == 'reset']
    assert len(lifecycle) == 1 and lifecycle[0]['model_seed'] == seed
    atomic_json(meta_dir/'summary.json', {'passed': True, 'condition': condition, 'task': task, 'episode': episode,
        'model_seed': seed, 'ablation': mode, 'source': str(d), 'reused_G1': reused, 'outcome': r['outcome'],
        'decisions': proof['decisions'], 'plan_sha256': sha256(OUT/'plan.json'), 'label': LABEL,
        'source_sha256': {n: sha256(d/n) for n in ('result.json', 'raw_frames.jsonl', 'control_actions.jsonl',
            'policy_decisions.jsonl', 'policy_actions.jsonl', 'oracle_decisions.jsonl')}, 'completed_utc': timestamp()})


def bitwise_compare(old, new):
    errors = []
    a, b = read_json(old/'result.json'), read_json(new/'result.json')
    for k in ('outcome', 'executed_steps', 'environment_seed', 'n_pick', 'final_in_basket', 'final_detached'):
        if a.get(k) != b.get(k): errors.append(f'result/{k}')
    def compare_rows(name, fields):
        aa, bb = read_jsonl(old/name), read_jsonl(new/name)
        if len(aa) != len(bb): errors.append(f'{name}/length:{len(aa)}!={len(bb)}')
        for i, (x, y) in enumerate(zip(aa, bb)):
            for key in fields:
                if json.dumps(x.get(key), sort_keys=True, separators=(',', ':')) != json.dumps(y.get(key), sort_keys=True, separators=(',', ':')):
                    errors.append(f'{name}/{i}/{key}')
                    return
    compare_rows('raw_frames.jsonl', ('frame_id', 'control_step', 'front_sha256', 'wrist_sha256', 'state_sha256', 'gt'))
    compare_rows('control_actions.jsonl', ('action', 'proposed_action', 'measured_arm_qpos', 'guard_clipped_joints', 'outcome'))
    compare_rows('policy_actions.jsonl', ('input_sha256', 'actions_sha256', 'action_shape'))
    compare_rows('policy_decisions.jsonl', ('selected_frame_ids', 'source_position_frame_ids', 'mask', 'tensor_sha256'))
    return {'bitwise_equal': not errors, 'first_differences': errors, 'old': str(old), 'new': str(new), 'label': LABEL}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--run', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: p.error('D6 diagnosis already running')
        state, plan = prepare()
        if state['state'] in ('awaiting_D6_review', 'diagnosis_stopped'):
            print(state, flush=True);return
        if not args.run:
            publish(state, state='ready_for_gpu_diagnostic', error='')
            print({'ready': True, 'label': LABEL, 'deadline': plan['deadline_utc']}, flush=True);return
        stop = threading.Event();servers = {}
        from report_fruit_memory_usage import deliver
        try:
            check_deadline(plan);verify_sources(plan);verify_preservation()
            pilot_file = OUT/'pilot_comparison.json'
            if not pilot_file.exists():
                publish(state, state='five_G1_bitwise_checks')
                for lane in range(4):servers[lane] = start_server(7, lane, plan, pilot=True)
                comparisons = []
                for case in plan['pilot_cases']:
                    task, e = case['task'], case['episode']
                    lane = ((e-220) % 2)*2+TASKS.index(task)
                    d = raw_case(task, e, 7, 'K0_normal', servers[lane], lane, plan, pilot=True)
                    comparisons.append(bitwise_compare(G1/f'G1/{task}/ep{e}', d))
                reuse = all(c['bitwise_equal'] for c in comparisons)
                atomic_json(pilot_file, {'all_five_bitwise_equal': reuse, 'reuse_G1_seed7': reuse,
                    'comparisons': comparisons, 'label': LABEL, 'plan_sha256': sha256(OUT/'plan.json')})
                for s in servers.values():stop_server(s['child'])
                servers.clear()
            pilot = read_json(pilot_file);reuse = pilot['reuse_G1_seed7']
            publish(state, state='diagnostic_dev220_259', reuse_G1_seed7=reuse)
            for lane in range(4):servers[lane] = start_server(7 if lane < 2 else 8, lane, plan)
            def lane_work(lane):
                try:
                    for group in plan['schedule']:
                        seed, e, task = group['seed'], group['episode'], group['task']
                        assigned = ((e-220) % 2)+(0 if seed == 7 else 2)
                        if assigned != lane:continue
                        for condition in group['conditions']:
                            if stop.is_set():return
                            check_deadline(plan);verify_sources(plan)
                            canonical = OUT/'episodes'/condition/task/f'ep{e}_seed{seed}'
                            meta = OUT/'audits'/condition/task/f'ep{e}_seed{seed}'
                            meta.mkdir(parents=True, exist_ok=True)
                            if canonical.exists() and (meta/'summary.json').exists():continue
                            if condition == 'K0_normal' and seed == 7 and reuse:
                                d = G1/f'G1/{task}/ep{e}'
                                save_audit(d, meta, None, servers[lane]['config']['oracle'], plan, task, e, seed, condition, reused=True)
                            elif condition == 'K0_normal' and seed == 7 and any(c['task'] == task and c['episode'] == e for c in plan['pilot_cases']):
                                d = OUT/'pilot'/task/f'ep{e}_seed7'
                                save_audit(d, meta, None, servers[lane]['config']['oracle'], plan, task, e, seed, condition, reused=False)
                            else:
                                d = raw_case(task, e, seed, condition, servers[lane], lane, plan)
                            canonical.parent.mkdir(parents=True, exist_ok=True)
                            canonical.symlink_to(os.path.relpath(d, canonical.parent), target_is_directory=True)
                except BaseException:
                    stop.set();raise
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(lane_work, i) for i in range(4)]):future.result()
            verify_preservation();verify_sources(plan)
            deliver(state, complete=True)
            publish(state, state='awaiting_D6_review', error='', completed_utc=timestamp())
        except Exception as exc:
            stop.set();publish(state, state='diagnosis_stopped', error=str(exc), traceback=traceback.format_exc())
            deliver(state, complete=False)
            raise
        finally:
            for s in servers.values():stop_server(s['child'])


if __name__ == '__main__':main()
