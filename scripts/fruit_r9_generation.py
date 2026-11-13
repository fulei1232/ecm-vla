"""Authorized post-R0 data-only continuation; all attempts retained, no trainer."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import sys

from v_a_common import ECM, POLICY, POLICY_PYTHON, CHECKPOINT, read_json, read_jsonl, atomic_json, sha256, timestamp
from fruit_r9_common import OUT, DATASET, RAW_CLEAN, RAW_DART, publish, proof
from fruit_r9_r0_support import information, descriptive


def generate_data(state, execute, run_command, check_sources, abort):
    assert read_json(OUT / 'R0_gate.json')['passed']
    publish(state, state='r9_generating_clean_DART', train_data_generated=True)
    accepted = [];attempts = []
    from threading import Lock
    mutex = Lock()

    def one(task, episode, gpu):
        dart = episode >= 100;purpose = 'train_dart' if dart else 'train_clean'
        root = (RAW_DART if dart else RAW_CLEAN) / task
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'ep{episode}_manifest.json'
        if path.exists():
            manifest = read_json(path)
            for name in ('h5','gt'):assert sha256(root / manifest[name]) == manifest[name+'_sha256']
            with mutex:accepted.append(manifest)
            return
        failures = []
        for attempt in range(16):
            result = execute(task, 'history', episode, gpu, purpose, attempt)
            with mutex:attempts.append({k:result[k] for k in ('task','episode','purpose','seed','success','outcome','source')})
            if not result['success']:
                failures.append({k:result[k] for k in ('seed','outcome','source','steps')})
                continue
            directory = Path(result['source']);label = 'FruitCount' if task == 'count' else 'FruitOccludedRipe'
            relative = {}
            for key, subdirectory, extension in [('h5','hdf5_files','h5'), ('gt','gt','jsonl')]:
                link = root / subdirectory / f"{label}_ep{episode}_seed{result['seed']}.{extension}"
                link.parent.mkdir(exist_ok=True)
                target = directory / result[key]
                link.symlink_to(os.path.relpath(target, link.parent))
                assert sha256(link) == result[key+'_sha256']
                relative[key] = str(link.relative_to(root))
            manifest = {**result, **relative, 'robot':'rm75b_eg2', 'split':'train', 'failures':failures,
                        'noise': {'sigma_rad':.04,'tau_steps':10.,'phase_scale':{'reach':1.,'approach':.5,'pull':1.,'carry':1.}}
                                 if dart else None,
                        'verification': {'passed':True,'record_audit':str(directory/'validity.json'),
                                         'source_sha256':sha256(directory/'validity.json')}}
            atomic_json(path, manifest)
            with mutex:accepted.append(manifest)
            return
        raise RuntimeError(f'All 16 predeclared data-generation attempts failed: {task}/ep{episode}; no new seeds')

    jobs = [(task,e) for task in ('count','occluded_ripe') for e in range(200)]
    def lane(gpu):
        try:
            for task,e in jobs[gpu::8]:one(task,e,gpu)
        except Exception:
            abort();raise
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(lane,gpu) for gpu in range(8)]):future.result()
    assert len(accepted) == 400
    check_sources()
    publish(state, state='r9_building_shared_data')
    builder = ECM / 'scripts/build_fruit_dataset.py'
    if not (DATASET / 'meta/episodes.json').exists():
        DATASET.mkdir(parents=True, exist_ok=True)
        run_command('data_plan', [str(POLICY_PYTHON), str(builder), 'plan', '--out', str(DATASET),
            '--part', str(RAW_CLEAN), '0', '100', '--part', str(RAW_DART), '100', '200',
            '--distance-signal', 'any_hanging'], cpu=True, cwd=POLICY)
    episodes = read_json(DATASET / 'meta/episodes.json')
    assert len(episodes) == 400 and all(r['exec_start_idx'] == 30 and r['distance_signal'] == 'any_hanging' for r in episodes)
    def shard(gpu):
        run_command(f'data_shard_{gpu}', [str(POLICY_PYTHON), '-u', str(builder), 'shard', '--out', str(DATASET),
                    '--index', str(gpu), '--of', '8'], gpu=gpu, cwd=POLICY)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(shard,gpu) for gpu in range(8)]):future.result()
    run_command('data_finish', [str(POLICY_PYTHON), str(builder), 'finish', '--out', str(DATASET)], cpu=True, cwd=POLICY)
    assert read_json(DATASET / 'meta/build.json')['passed']
    publish(state, state='r9_computing_one_shared_normalization')
    if not (DATASET / 'meta/norm_stats.json').exists():
        run_command('data_norm_stats', [str(POLICY_PYTHON), str(builder), 'norm-stats', '--out', str(DATASET)], cpu=True, cwd=POLICY)
    run_command('data_encoder_check', [str(POLICY_PYTHON), str(builder), 'check-encoder', '--out', str(DATASET),
                '--checkpoint', str(CHECKPOINT), '--frames','8'], gpu=0, cwd=POLICY)
    encoder = read_json(DATASET / 'meta/encoder_check.json');assert encoder['shapes'][0] == encoder['shapes'][1]
    count_rows = [r for r in accepted if r['task'] == 'count']
    availability = information(count_rows, 'train_information_audit.json', gate_required=False)
    by_task = {}
    for task in ('count','occluded_ripe'):
        rows = [r for r in accepted if r['task'] == task]
        by_task[task] = {'episodes':len(rows), 'clean_episodes':sum(r['episode']<100 for r in rows),
            'DART_episodes':sum(r['episode']>=100 for r in rows), 'total_frames':sum(r['data_frames'] for r in rows),
            'execution_samples':sum(r['data_frames']-30 for r in rows),
            'waiting_frames':sum(r['wait_supervised_frames'] for r in rows),
            'waiting_fraction':sum(r['wait_supervised_frames'] for r in rows)/sum(r['data_frames'] for r in rows),
            'DART_perturbations':sum(r['dart_perturbations'] for r in rows),
            'preserved_failed_attempts':sum(len(r['failures']) for r in rows),
            'mean_frames_by_N':{str(N):descriptive([r['data_frames'] for r in rows if r['N']==N]) for N in sorted({r['N'] for r in rows})}}
    from fruit_picking.task_v2_audit_r9 import MEMORIES, memory_frames
    retention = {condition:{'decisions':0,'retained':0} for condition in MEMORIES}
    for r in accepted:
        if r['task'] != 'occluded_ripe':continue
        d = Path(r['source']);trace = read_jsonl(d/'trace.jsonl');distance = [f['signals']['any_hanging'] for f in trace]
        for decision in read_json(d/'decisions.json'):
            for condition in MEMORIES:
                retention[condition]['decisions'] += 1
                retention[condition]['retained'] += 0 in memory_frames(distance,decision['step'],condition)
    value = {'revision':9,'created_utc':timestamp(),'passed':True,'episodes':400,'by_task':by_task,
        'dataset':str(DATASET),'distance_signal':'any_hanging','history_prefix_only':30,
        'wait_and_completion_supervised':True,'arms':['K0','K8','KA'],'temporal_index':'order',
        'norm_stats_sha256':sha256(DATASET/'assets/robomme/norm_stats.json'),
        'normalization_computed_once_and_shared':True,'engineering_errors':0,'training':False,
        'train_information_report_only':{'path':str(OUT/'train_information_audit.json'),'conditions':availability['conditions']},
        'occluded_train_frame0_report_only':retention,'encoder_check':encoder,
        'build_sha256':sha256(DATASET/'meta/build.json'), 'attempts':attempts,
        'accepted_manifests':[str((RAW_DART if r['episode']>=100 else RAW_CLEAN)/r['task']/f"ep{r['episode']}_manifest.json") for r in accepted]}
    proof('data_delivery.json',value)
    return value
