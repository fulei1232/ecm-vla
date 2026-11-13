#!/usr/bin/env python3
"""Revision4: sequential K4/K8 training and dev validity, then STOP A.

This program has no formal-evaluation or sealing command. It never opens val
layouts; the evaluation protocol remains a draft for the user's review.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from v_a_common import (ECM, POLICY, POLICY_PYTHON, SIM, SIM_PYTHON, OPENPI_DATA_HOME,
                        atomic_json, read_json, sha256, timestamp, git_head)

OUT = ECM/'results/fruit_lowlevel_v2_stage5_20261103'
RUN = ECM/'runs/fruit_lowlevel_v2_k4k8_20261103'
K0 = ECM/'runs/fruit_k0v2_lowlevel_rescue_20261102'
G1 = ECM/'results/fruit_k0v2_gate_20261102'
DATA = ECM/'data/finetune_kf/fruit_dart_v2_rm75b_train'
DEV = ECM/'data/fruit_picking_dev/rm75b_eg2/train'
TRAIN_DOC = ECM/'reports/fruit_lowlevel_v2_training_20261103.md'
DRAFT = ECM/'reports/fruit_lowlevel_v2_protocol_20261103.md'
TASKS = ('FruitCount', 'FruitOccludedRipe')
ANALYZER = ECM/'scripts/analyze_fruit_failure_modes.py'
CLIENT = ECM/'scripts/evaluate_fruit_lowlevel_v2_episode.py'


class ValidityStop(RuntimeError):
    pass


def training_specs():
    base = read_json(K0/'arms/k0.json')
    result = {}
    for arm, k in (('k4', 4), ('k8', 8)):
        spec = deepcopy(base)
        spec.update(arm=arm, checkpoint_base_dir=str(RUN/'ckpt'))
        spec['sampler']['events'] = k
        restored = deepcopy(spec)
        restored.update(arm=base['arm'], checkpoint_base_dir=base['checkpoint_base_dir'])
        restored['sampler']['events'] = base['sampler']['events']
        assert restored == base, 'Only memory selection and output naming may differ'
        assert spec['train']['steps'] == 12000 and spec['train']['seed'] == 42
        assert spec['train']['fsdp_devices'] == 8 and spec['gpus'] == list(range(8))
        result[arm] = spec
    return result


def prepare():
    from run_finetune_kf import CODE, verify
    verify(K0)
    assert read_json(K0/'gate_t1_k0.json')['passed']
    assert read_json(G1/'summary.json')['G1_passed']
    baseline = read_json(G1/'G1_continuation_plan.json')
    for path, digest in baseline['source_sha256'].items():
        assert sha256(path) == digest, path
    assert sha256(G1/'guard_v2.json') == baseline['guard_stats_sha256']
    assert sha256(G1/'guard_choice.json') == baseline['guard_choice_sha256']
    assert read_json(G1/'guard_choice.json')['guard'] == 'v2'
    assert sha256(ECM/'reports/split_registry_20261102.json') == baseline['registry_sha256']
    build = read_json(DATA/'meta/build.json')
    assert build['passed'] and build['total_samples'] == 164639 and build['execution_samples'] == 152639
    episodes = read_json(DATA/'meta/episodes.json')
    assert len(episodes) == 400 and all(e['exec_start_idx'] == 30 for e in episodes)
    norm = DATA/'assets/robomme/norm_stats.json'
    assert sha256(norm) == '6da229fba3c9c2564ddacdc2f7093cb69c672a0ddd651cb6b18665ec57e08568'
    assert sha256(norm) == sha256(K0/'ckpt/mme_vla_suite/k0/11999/assets/robomme/norm_stats.json')
    specs = training_specs()
    init_manifest = Path(specs['k4']['init_checkpoint'])/'params/manifest.ocdbt'
    assert sha256(init_manifest) == read_json(K0/'protocol.json')['init_params_manifest_sha256']
    (RUN/'arms').mkdir(parents=True, exist_ok=True)
    for arm, spec in specs.items():
        path = RUN/f'arms/{arm}.json'
        if path.exists():
            assert read_json(path) == spec, path
        else:
            atomic_json(path, spec)
    OUT.mkdir(parents=True, exist_ok=True)
    platform_files = [G1/'guard_v2.json', G1/'guard_choice.json', G1/'expert_profile.json',
        ECM/'src/memory_management/fruit_rescue_platform.py', ECM/'scripts/evaluate_fruit_rescue_episode.py',
        CLIENT, ECM/'scripts/audit_fruit_rescue_memory.py', ECM/'scripts/run_fruit_diagnosis.py',
        ECM/'scripts/serve_teacher_policy.py', ECM/'src/memory_management/fruit_eval.py',
        ECM/'src/memory_management/fruit_distance.py', ECM/'src/memory_management/event_keyframes.py',
        ECM/'src/memory_management/validation/teacher_oracle.py',
        POLICY/'examples/robomme/eval.py', POLICY/'examples/robomme/subgoal_predictor.py',
        POLICY/'scripts/serve_policy.py']
    identity = {'G1_reference_plan_sha256': sha256(G1/'G1_continuation_plan.json'),
        'base_condition': 'A0', 'joint_guard': 'v2', 'exec_horizon': 16, 'obs_horizon': 16,
        'observe_hold_steps': 30, 'total_step_budget': 1300, 'selected_expert': 'canonical_v3.1_20261102',
        'files_sha256': {str(p): sha256(p) for p in platform_files},
        'client_extension': 'Only condition/model-seed routing generalized; same G1 evaluator/prefix/recorder/guard.'}
    platform = OUT/'platform_identity.json'
    if platform.exists():
        assert read_json(platform) == identity
    else:
        atomic_json(platform, identity)
    immutable = {str(p): sha256(p) for p in (DATA/'meta/build.json', DATA/'meta/episodes.json', norm,
        K0/'protocol.json', K0/'arms/k0.json', K0/'gate_t1_k0.json',
        K0/'ckpt/mme_vla_suite/k0/11999/params/manifest.ocdbt',
        ECM/'reports/split_registry_20261102.json', platform, TRAIN_DOC)}
    # These are the same existing distance arrays used by K0, not a rebuilt dataset.
    distance_files = sorted((DATA/'features').glob('episode_*/distance.npy'))
    assert len(distance_files) == 400
    immutable.update({str(p): sha256(p) for p in distance_files})
    immutable[str(init_manifest)] = sha256(init_manifest)
    sources = list(dict.fromkeys(CODE + platform_files + [Path(__file__).resolve(), ANALYZER,
        ECM/'scripts/fruit_lowlevel_v2_metrics.py', ECM/'scripts/fruit_lowlevel_v2_delivery.py']))
    protocol = {'name': 'fruit_lowlevel_v2_k4k8_training_only', 'study': 'fruit_lowlevel_v2',
        'formal': True, 'created_utc': timestamp(), 'protocol_doc': str(TRAIN_DOC),
        'protocol_doc_sha256': sha256(TRAIN_DOC), 'dataset': str(DATA),
        'dataset_build_sha256': sha256(DATA/'meta/build.json'),
        'dataset_episodes_sha256': sha256(DATA/'meta/episodes.json'), 'dataset_copy_check': None,
        'assets_dir': str(DATA/'assets'), 'assets_norm_stats_sha256': sha256(norm),
        'init_checkpoint': specs['k4']['init_checkpoint'],
        'init_params_manifest_sha256': sha256(Path(specs['k4']['init_checkpoint'])/'params/manifest.ocdbt'),
        'arms': {arm: {'events': spec['sampler']['events'], 'spec_sha256': sha256(RUN/f'arms/{arm}.json')}
                 for arm, spec in specs.items()},
        'code_identity': [{'path': str(p), 'sha256': sha256(p)} for p in sources],
        'ecm_sha_at_prepare': git_head(ECM), 'evaluation_protocol_sealed': False,
        'training_order': ['k4', 'k8'], 'restart_policy': 'No automatic retry after a failed training/validity check.'}
    if (RUN/'protocol.json').exists():
        verify(RUN)
        protocol = read_json(RUN/'protocol.json')
    else:
        atomic_json(RUN/'protocol.json', protocol)
    plan = {'phase': 5, 'revision': 4, 'training_run': str(RUN),
        'K0_reused_without_retraining': str(K0/'ckpt/mme_vla_suite/k0/11999'),
        'K0_preservation_and_dataset_sha256': immutable,
        'source_sha256': {str(p): sha256(p) for p in sources},
        'training_protocol_sha256': sha256(RUN/'protocol.json'),
        'training_specs': specs, 'only_effective_training_difference': 'sampler.events',
        'normalization_recomputed': False, 'dataset_rebuilt': False,
        'validity': {'episodes': list(range(200, 220)), 'tasks': list(TASKS), 'model_seed': 7,
            'guard': 'v2', 'engineering_error_target': 0, 'memory_audit_target': 'all pass',
            'event_target': 'causal events occupy memory slots; all zero-event episodes individually explained',
            'performance_adjustments_allowed': False},
        'smoke': {'episodes': [200], 'tasks': list(TASKS), 'conditions': ['K0', 'K4', 'K8'], 'model_seed': 7},
        'evaluation_protocol_draft': str(DRAFT), 'stop_at': 'A before protocol sealing and formal evaluation',
        'val_layouts_accessed': False, 'formal_evaluation_authorized': False}
    if (OUT/'plan.json').exists():
        assert read_json(OUT/'plan.json') == plan, 'Prepared stage5 inputs changed'
    else:
        atomic_json(OUT/'plan.json', plan)
        atomic_json(RUN/'plan.json', plan)
        text = DRAFT.read_text()
        begin, end = '<!-- STOP_A_RESULTS_BEGIN -->', '<!-- STOP_A_RESULTS_END -->'
        assert text.count(begin) == text.count(end) == 1
        table = ['尚待K4/K8训练、T1、有效性及dev200冒烟；未使用val布局。', '',
            f'平台清单SHA256：`{sha256(platform)}`。', '', '| 平台文件 | SHA256 |', '|---|---|']
        table += [f'| {path} | {digest} |' for path, digest in identity['files_sha256'].items()]
        DRAFT.write_text(text[:text.index(begin)+len(begin)]+'\n\n'+'\n'.join(table)+'\n'+text[text.index(end):])
    verify_inputs(plan)
    return plan


def verify_inputs(plan):
    from run_finetune_kf import verify
    verify(K0)
    verify(RUN)
    for key in ('K0_preservation_and_dataset_sha256', 'source_sha256'):
        for path, digest in plan[key].items():
            assert sha256(path) == digest, path
    assert sha256(RUN/'protocol.json') == plan['training_protocol_sha256']
    assert plan['validity']['episodes'] == list(range(200, 220)) and plan['smoke']['episodes'] == [200]


def update_status(record, **changes):
    record.update(changes, updated_utc=timestamp())
    atomic_json(OUT/'status.json', record)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase=5, instruction_revision=4, state=record['state'], updated_utc=record['updated_utc'],
        decision='G1 approved. Execute revision4 training/dev only; stop A before sealing or formal evaluation.',
        next=record.get('next', 'Complete K4/K8 T1 and validity; deliver protocol draft at stop A.'))
    root['links'].update(D5_training=str(RUN), D5_status=str(OUT/'status.json'), D5_plan=str(OUT/'plan.json'),
        D5_platform=str(OUT/'platform_identity.json'), D5_draft=str(DRAFT))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)


def check_group(condition, smoke=False):
    from audit_fruit_rescue_memory import audit
    from run_fruit_diagnosis import start_server, wait_ready
    from fruit_lowlevel_v2_metrics import event_statistics, aggregate_event_statistics
    params = read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']
    conditions = ('K0', 'K4', 'K8') if smoke else (condition,)
    params = deepcopy(params)
    params['checkpoints'] = {c: str((K0 if c == 'K0' else RUN)/f'ckpt/mme_vla_suite/{c.lower()}/11999')
                             for c in conditions}
    label = 'smoke' if smoke else f'validity/{condition}'
    server_root = OUT/f'servers/{label}'
    lanes = 2 if smoke else 4
    jobs = ([(t, 200, c) for t in TASKS for c in conditions] if smoke else
            [(t, e, condition) for e in range(200, 220) for t in TASKS])
    assert all(200 <= e <= (200 if smoke else 219) for _, e, _ in jobs)
    servers = []
    lock = threading.Lock()
    stop = threading.Event()
    rows = []
    try:
        for i in range(lanes):
            servers.append(start_server(server_root, i, i, params, conditions=conditions,
                                        base_port=29421 if not smoke else 29521))
        wait_ready(servers)
        session = timestamp()

        def episode(task, e, c, lane):
            directory = OUT/(f'smoke/{c}/{task}/ep{e}' if smoke else f'validity/{c}/{task}/ep{e}')
            assert DEV == ECM/'data/fruit_picking_dev/rm75b_eg2/train'
            if not (directory/'result.json').exists():
                if directory.exists() and any(directory.iterdir()):
                    raise ValidityStop(f'Partial episode requires inspection: {directory}')
                directory.mkdir(parents=True, exist_ok=True)
                cfg = servers[lane][1]
                command = [str(SIM_PYTHON), '-u', str(CLIENT), '--port', str(cfg['port']),
                    '--context', str(server_root/f'server{lane}/context.json'), '--root', str(DEV),
                    '--task', task, '--episode-id', str(e), '--condition', c, '--model-seed', '7',
                    '--attempt-dir', str(directory), '--guard-stats', str(G1/'guard_v2.json')]
                manifest = DEV/('count' if task == 'FruitCount' else 'occluded_ripe')/f'ep{e}_manifest.json'
                atomic_json(directory/'started.json', {'command': command, 'utc': timestamp(),
                    'manifest_sha256': sha256(manifest), 'plan_sha256': sha256(OUT/'plan.json'),
                    'server_config_sha256': sha256(server_root/f'server{lane}/config.json')})
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(lane),
                    'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json',
                    'MPLCONFIGDIR': str(ECM/'cache/matplotlib'), 'PYTHONDONTWRITEBYTECODE': '1',
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'PYTHONPATH': f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src'}
                with (directory/'console.log').open('a') as log:
                    try:
                        code = subprocess.run(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              timeout=1800).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                row = read_json(directory/'result.json') if (directory/'result.json').exists() else {
                    'task': task, 'episode_id': e, 'condition': c, 'model_seed': 7,
                    'outcome': 'error', 'error': f'client exit {code}, no result'}
                row.update(client_exit_code=code, server_session=f'{session}/server{lane}')
                atomic_json(directory/'result.json', row)
            row = read_json(directory/'result.json')
            if (row.get('task'), row.get('episode_id'), row.get('condition'), row.get('model_seed')) != (task, e, c, 7):
                raise ValidityStop(f'Condition/layout/seed routing mismatch: {directory}')
            try:
                evidence = audit(directory, params['oracle'])
            except Exception:
                evidence = {'passed': False, 'errors': [traceback.format_exc()], 'decisions': 0}
            atomic_json(directory/'memory_audit.json', evidence)
            try:
                event = event_statistics(directory, params['oracle']['keyframes'][c])
            except Exception:
                event = {'event_validity_passed': False, 'errors': [traceback.format_exc()]}
            atomic_json(directory/'event_statistics.json', event)
            valid = not row.get('error') and row.get('client_exit_code') == 0 and evidence['passed'] and event['event_validity_passed']
            with lock:
                rows.append({'task': task, 'episode': e, 'condition': c, 'valid': valid,
                             'outcome': row['outcome'], 'directory': str(directory)})
            print(rows[-1], flush=True)
            if not valid:
                raise ValidityStop(f'Engineering/memory/event validity failed: {directory}')

        def lane(i):
            try:
                assigned = [job for job in jobs if job[0] == TASKS[i]] if smoke else jobs[i::lanes]
                for task, e, c in assigned:
                    if stop.is_set():
                        return
                    episode(task, e, c, i)
            except BaseException:
                stop.set()
                raise
        with ThreadPoolExecutor(max_workers=lanes) as pool:
            for future in as_completed([pool.submit(lane, i) for i in range(lanes)]):
                future.result()
    finally:
        for child, _ in servers:
            child.terminate()
        for child, _ in servers:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    for c in conditions:
        directory = OUT/(f'smoke/{c}' if smoke else f'validity/{c}')
        analysis = OUT/(f'analysis/smoke/{c}' if smoke else f'analysis/validity/{c}')
        command = [sys.executable, str(ANALYZER), '--episodes-root', str(directory), '--pattern', 'ep{e}',
            '--episodes', '200' if smoke else '200-219', '--q7-center', str(read_json(G1/'expert_profile.json')['q7_center']),
            '--out', str(analysis)]
        subprocess.run(command, cwd=ECM, check=True)
        metrics = read_json(analysis/'summary.json')
        expected = 2 if smoke else 40
        assert metrics['all']['episodes'] == expected and not metrics['missing_expected_episodes']
        if metrics['all']['engineering_error']['numerator']:
            raise ValidityStop(f'Unified analyzer reports engineering errors: {analysis}')
        events = [read_json(directory/t/f'ep{e}/event_statistics.json') for t in TASKS
                  for e in ([200] if smoke else range(200, 220))]
        event_summary = aggregate_event_statistics(events)
        atomic_json(directory/'event_summary.json', event_summary)
        audits = [read_json(directory/t/f'ep{e}/memory_audit.json') for t in TASKS
                  for e in ([200] if smoke else range(200, 220))]
        evidence = {'passed': True, 'condition': c, 'layouts': expected, 'model_seed': 7,
            'memory_audits_passed': sum(a['passed'] for a in audits),
            'audited_decisions': sum(a['decisions'] for a in audits),
            'engineering_errors': metrics['all']['engineering_error'],
            'performance_is_descriptive_only': True, 'performance': metrics,
            'event_statistics': event_summary, 'completed_utc': timestamp()}
        atomic_json(directory/'validity.json', evidence)


def deliver(state):
    """Write stop-A evidence and fill only the marked results area of the draft."""
    from fruit_lowlevel_v2_delivery import deliver_stage5
    deliver_stage5(OUT, RUN, K0, G1, DRAFT, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', help='Train and check dev only; always stop before sealing')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('stage5 is already running')
    plan = prepare()
    state = read_json(OUT/'status.json') if (OUT/'status.json').exists() else {
        'phase': 5, 'revision': 4, 'started_utc': timestamp()}
    if state.get('state') == 'awaiting_protocol_review_A':
        print('Stop A already reached. No sealing or formal evaluation is implemented here.', flush=True)
        return
    update_status(state, state='ready_for_stage5_gpu_launch')
    if not a.run:
        print({'prepared': True, 'training_order': ['k4', 'k8'], 'stop_at': 'A', 'val_access': False}, flush=True)
        return
    try:
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENPI_DATA_HOME': str(OPENPI_DATA_HOME),
            'MPLCONFIGDIR': str(ECM/'cache/matplotlib'), 'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json'}
        from run_finetune_kf import arm_complete
        for arm in ('k4', 'k8'):
            verify_inputs(plan)
            update_status(state, state=f'training_{arm}_12000_steps')
            spec = plan['training_specs'][arm]
            if not arm_complete(RUN, spec):
                if list((RUN/'logs').glob(f'{arm}_*.log')):
                    raise ValidityStop(f'Incomplete previous {arm} training; no automatic retry. Inspect preserved logs.')
                command = [sys.executable, str(ECM/'scripts/run_finetune_kf.py'), 'train', '--out', str(RUN), '--arms', arm]
                with (OUT/f'train_{arm}.log').open('a') as log:
                    code = subprocess.run(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if code:
                    raise ValidityStop(f'{arm} training failed, exit={code}')
            command = [sys.executable, str(ECM/'scripts/run_finetune_kf.py'), 'gate', '--out', str(RUN), '--arms', arm]
            with (OUT/f'T1_{arm}.log').open('a') as log:
                code = subprocess.run(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                raise ValidityStop(f'{arm} T1 failed')
            saved = RUN/f'ckpt/mme_vla_suite/{arm}/11999/assets/robomme/norm_stats.json'
            assert sha256(saved) == sha256(DATA/'assets/robomme/norm_stats.json')
            update_status(state, state=f'validity_{arm}_dev200_219')
            check_group(arm.upper())
        verify_inputs(plan)
        update_status(state, state='smoke_dev200_only')
        check_group(None, smoke=True)
        verify_inputs(plan)
        update_status(state, state='awaiting_protocol_review_A', error='',
            next='Stop A: await user review of T1, dev validity, smoke and draft; do not seal or evaluate val.')
        deliver(state)
    except Exception as exc:
        update_status(state, state='stage5_stopped_for_validity_or_engineering', error=str(exc),
            traceback=traceback.format_exc(), next='Report preserved training/dev evidence; await user review.')
        deliver(state)
        raise


if __name__ == '__main__':
    main()
