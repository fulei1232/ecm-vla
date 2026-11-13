"""Fresh clean plus matched-layout two-cycle-DART data and one shared build."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import time

from v_a_common import (ECM, POLICY, POLICY_PYTHON, CHECKPOINT, read_json, read_jsonl,
                        atomic_json, sha256, timestamp)
from fruit_r12_common import OUT, DATASET, RAW_CLEAN, RAW_DART, proof, publish
from fruit_r11_rules import AttemptLedger
from fruit_r12_validity import audit_case


def _load_existing_manifests():
    """Load only verified R12 manifests so a stopped run can resume in place."""
    rows = []
    seen = set()
    for dart, root in ((False, RAW_CLEAN), (True, RAW_DART)):
        purpose = 'train_dart' if dart else 'train_clean'
        for task in ('count', 'occluded_ripe'):
            task_root = root/task
            for path in sorted(task_root.glob('ep*_manifest.json')):
                manifest = read_json(path)
                expected_episode = manifest['episode']
                expected_range = range(100, 200) if dart else range(100)
                assert expected_episode in expected_range, str(path)
                assert manifest['task'] == task and manifest['purpose'] == purpose, str(path)
                assert manifest['success'] and manifest['verification']['passed'], str(path)
                assert manifest['instruction_revision'] == 12, str(path)
                assert manifest['r12_layout_seed'] is not None, str(path)
                for field in ('h5', 'gt'):
                    assert sha256(task_root/manifest[field]) == manifest[field+'_sha256'], str(path)
                source = Path(manifest['source'])
                assert not source.is_relative_to(ECM/'results/fruit_task_v2_R0_r11_20261106')
                assert sha256(source/'validity.json') == manifest['verification']['audit_sha256']
                key = (task, purpose, expected_episode)
                assert key not in seen, key
                seen.add(key)
                rows.append(manifest)
    return rows


def _attempt_report(ledger_rows):
    fields = ('task', 'episode', 'purpose', 'attempt', 'seed', 'r12_layout_seed', 'N',
              'success', 'outcome', 'steps', 'r12_dart_perturbation_cycles', 'source',
              'validated_case_wall_seconds')
    report_rows = []
    for row in ledger_rows:
        item = dict(row)
        path = Path(row['key'])/'result.json'
        if path.exists():
            result = read_json(path)
            if 'outcome' not in row:
                item['ledger_state'] = 'incomplete_or_invalidated'
                item['replay_result'] = {name: result.get(name) for name in
                    ('success','outcome','r12_layout_seed','N','steps')}
            else:
                item['ledger_state'] = 'completed'
                for name in fields:
                    if name in result:
                        item.setdefault(name, result[name])
        else:
            item['ledger_state'] = 'incomplete_or_invalidated' if 'outcome' not in row else 'completed'
        report_rows.append(item)
    return report_rows


def generate(state, execute, run_command, abort):
    assert read_json(OUT/'gpu_validation.json')['passed']
    assert read_json(OUT/'R0_gate.json')['passed']
    started = time.monotonic()
    started_utc = timestamp()
    accepted = _load_existing_manifests()
    attempt_rows = []
    previous_ledger = read_json(OUT/'data_attempt_ledger.json') if (OUT/'data_attempt_ledger.json').exists() else None
    ledger = AttemptLedger(lambda value: proof('data_attempt_ledger.json', value), previous=previous_ledger)

    def one(task, episode, gpu, dart):
        purpose = 'train_dart' if dart else 'train_clean'
        root = (RAW_DART if dart else RAW_CLEAN)/task
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root/f'ep{episode}_manifest.json'
        if manifest_path.exists():
            existing = read_json(manifest_path)
            assert existing['task'] == task and existing['episode'] == episode
            assert existing['purpose'] == purpose and existing['success']
            assert any(x['task'] == task and x['purpose'] == purpose and x['episode'] == episode
                       for x in accepted)
            return existing
        paired_clean = None
        if dart:
            clean_episode = episode-100
            clean_path = RAW_CLEAN/task/f'ep{clean_episode}_manifest.json'
            paired_clean = read_json(clean_path)
            assert paired_clean['success'] and paired_clean['episode'] == clean_episode
        failures = []
        prefix = str(OUT/'replays'/purpose/task/'history'/f'ep{episode}_attempt')
        with ledger.lock:
            prior_rows = list(ledger.rows.values())
        for prior in prior_rows:
            if not prior.get('key','').startswith(prefix):
                continue
            if prior.get('success') is True:
                continue
            previous_result_path = Path(prior['key'])/'result.json'
            previous_result = read_json(previous_result_path) if previous_result_path.exists() else {}
            failures.append({
                'seed': previous_result.get('seed', prior.get('noise_seed')),
                'r12_layout_seed': previous_result.get('r12_layout_seed', prior.get('layout_seed')),
                'outcome': prior.get('outcome', 'interrupted_or_invalidated'),
                'success': previous_result.get('success', prior.get('success')),
                'source': prior['key'],
                'steps': previous_result.get('steps'),
                'validated_case_wall_seconds': previous_result.get('validated_case_wall_seconds'),
                'preserved_from_prior_attempt': True,
                'ledger_completed': 'outcome' in prior})
        for attempt in range(16):
            attempt_seed = 4100000 + 16*episode + attempt
            layout_seed = paired_clean['r12_layout_seed'] if dart else attempt_seed
            key = f'{task}/{episode}/attempt{attempt}'
            out = OUT/'replays'/purpose/task/'history'/f'ep{episode}_attempt{attempt}'
            detail = {'task':task,'purpose':purpose,'episode':episode,'attempt':attempt,
                      'noise_seed':attempt_seed,'layout_seed':layout_seed,'started_utc':timestamp()}
            if str(out) in ledger.rows:
                # Preserve every previous attempt directory and ledger row. A missing
                # manifest means this attempt was failed, interrupted, or invalidated.
                continue
            if out.exists() and any(out.iterdir()):
                raise RuntimeError(f'Unledgered R12 attempt output exists; refusing to reuse: {out}')
            ledger.start(str(out), detail)
            try:
                result = execute(task, 'history', episode, gpu, purpose, attempt,
                    r12_layout_seed=layout_seed, dart_two_cycles=dart, data_key=key)
            except Exception:
                abort()
                raise
            outcome = result['outcome']
            attempt_record = {k:result.get(k) for k in
                ('task','episode','purpose','attempt','seed','r12_layout_seed','N','success','outcome',
                 'steps','r12_dart_perturbation_cycles','source','validated_case_wall_seconds')}
            attempt_rows.append(attempt_record)
            if ledger.finish(str(out), outcome, result['success']):
                raise RuntimeError('R12 shield-intersection stop: count>=3 and fraction>2%')
            if not result['success']:
                failures.append({k:result.get(k) for k in
                    ('seed','r12_layout_seed','outcome','source','steps','validated_case_wall_seconds')})
                continue
            assert result['task'] == task and result['episode'] == episode
            assert result['purpose'] == purpose and result['success']
            if dart:
                assert result['r12_layout_seed'] == paired_clean['r12_layout_seed']
                assert result['N'] == paired_clean['N']
                assert result['r12_layout_sha256'] == paired_clean['r12_layout_sha256']
                assert result['r12_dart_two_cycle_recipe']
                assert len(result['r12_dart_perturbation_cycles']) == 2
            else:
                assert not result['r12_dart_two_cycle_recipe']
            source = Path(result['source'])
            assert not source.is_relative_to(ECM/'results/fruit_task_v2_R0_r11_20261106')
            label = 'FruitCount' if task == 'count' else 'FruitOccludedRipe'
            relative = {}
            for field, folder, ext in (('h5','hdf5_files','h5'),('gt','gt','jsonl')):
                link = root/folder/f'{label}_ep{episode}_seed{result["seed"]}.{ext}'
                link.parent.mkdir(parents=True, exist_ok=True)
                target = source/result[field]
                link.symlink_to(os.path.relpath(target, link.parent))
                assert sha256(link) == result[field+'_sha256']
                relative[field] = str(link.relative_to(root))
            manifest = {**result, **relative, 'robot':'rm75b_eg2','split':'train',
                'failures':failures,
                'paired_clean_episode':episode-100 if dart else None,
                'paired_clean_N':paired_clean['N'] if dart else result['N'],
                'DART_recipe':result.get('DART_recipe') if dart else None,
                'verification':{'passed':True,'audit':str(source/'validity.json'),
                                'audit_sha256':sha256(source/'validity.json')}}
            atomic_json(manifest_path, manifest)
            accepted.append(manifest)
            return manifest
        raise RuntimeError(f'All 16 predeclared attempts failed: {task}/ep{episode}')

    def phase(jobs, dart):
        def lane(gpu):
            try:
                for task, episode in jobs[gpu::8]:
                    one(task, episode, gpu, dart)
            except Exception:
                abort()
                raise
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(lane, gpu) for gpu in range(8)]):
                future.result()

    publish(state, state='r12_generating_clean')
    clean_jobs = [(task,e) for task in ('count','occluded_ripe') for e in range(100)]
    phase(clean_jobs, dart=False)
    publish(state, state='r12_generating_matched_layout_DART')
    dart_jobs = [(task,e) for task in ('count','occluded_ripe') for e in range(100,200)]
    phase(dart_jobs, dart=True)
    accepted.sort(key=lambda x: (x['task'], x['purpose'], x['episode']))
    identities = [(x['task'], x['purpose'], x['episode']) for x in accepted]
    assert len(identities) == len(set(identities))
    assert len(accepted) == 400

    by_task = {}
    for task in ('count','occluded_ripe'):
        clean = [r for r in accepted if r['task']==task and r['episode']<100]
        dart = [r for r in accepted if r['task']==task and r['episode']>=100]
        clean_dist = Counter(str(r['N']) for r in clean)
        dart_dist = Counter(str(r['N']) for r in dart)
        assert len(clean)==len(dart)==100 and clean_dist==dart_dist
        for item in dart:
            pair = next(x for x in clean if x['episode']==item['paired_clean_episode'])
            assert item['N']==pair['N'] and item['r12_layout_sha256']==pair['r12_layout_sha256']
        by_task[task] = {'episodes':200,'clean_successful':100,'DART_successful':100,
            'clean_N_counts':dict(sorted(clean_dist.items(),key=lambda kv:int(kv[0]))),
            'DART_N_counts':dict(sorted(dart_dist.items(),key=lambda kv:int(kv[0]))),
            'N_distributions_identical':True,
            'DART_noisy_cycles_per_episode':{str(k):sum(len(x['r12_dart_perturbation_cycles'])==k for x in dart)
                                             for k in (2,)},
            'preserved_failed_attempts':sum(len(x['failures']) for x in clean+dart),
            'mean_frames_by_group_and_N':{
                group:{str(n):float(sum(x['data_frames'] for x in rows if x['N']==n)/sum(x['N']==n for x in rows))
                       for n in sorted({x['N'] for x in rows})}
                for group,rows in (('clean',clean),('DART',dart))}}
    all_attempt_rows = _attempt_report(ledger.snapshot()['attempts'])
    proof('data_attempts.json', {'instruction_revision':12,'attempts':all_attempt_rows,
        'started_attempts':len(all_attempt_rows),'outcomes':dict(Counter(x.get('outcome','incomplete') for x in all_attempt_rows)),
        'ledger':read_json(OUT/'data_attempt_ledger.json')})

    # Build one dataset and compute its shared normalization exactly once.
    publish(state, state='r12_building_shared_data')
    build_started = time.monotonic()
    builder = ECM/'scripts/build_fruit_dataset.py'
    DATASET.mkdir(parents=True, exist_ok=False)
    run_command('data_plan',[str(POLICY_PYTHON),str(builder),'plan','--out',str(DATASET),
        '--part',str(RAW_CLEAN),'0','100','--part',str(RAW_DART),'100','200',
        '--distance-signal','any_hanging'],cpu=True,cwd=POLICY)
    eps = read_json(DATASET/'meta/episodes.json')
    assert len(eps)==400 and all(x['exec_start_idx']==30 and x['distance_signal']=='any_hanging' for x in eps)
    def shard(gpu):
        run_command(f'data_shard_{gpu}',[str(POLICY_PYTHON),'-u',str(builder),'shard','--out',str(DATASET),
            '--index',str(gpu),'--of','8'],gpu=gpu,cwd=POLICY)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(shard,g) for g in range(8)]):future.result()
    run_command('data_finish',[str(POLICY_PYTHON),str(builder),'finish','--out',str(DATASET)],cpu=True,cwd=POLICY)
    build_seconds = time.monotonic()-build_started
    assert read_json(DATASET/'meta/build.json')['passed']
    publish(state, state='r12_computing_shared_normalization')
    norm_started=time.monotonic()
    run_command('data_norm_stats',[str(POLICY_PYTHON),str(builder),'norm-stats','--out',str(DATASET)],cpu=True,cwd=POLICY)
    norm_seconds=time.monotonic()-norm_started
    norm=DATASET/'assets/robomme/norm_stats.json'
    assert norm.exists()

    delivery={'instruction_revision':12,'passed':True,'episodes':400,'by_task':by_task,
        'dataset':str(DATASET),'history_prefix_only':30,'distance_signal':'any_hanging',
        'normalization_computed_once_and_shared':True,'norm_stats_sha256':sha256(norm),
        'build_sha256':sha256(DATASET/'meta/build.json'),'training':False,'val_used':False,
        'R11_109_demonstrations_mixed':False,'attempt_ledger':str(OUT/'data_attempt_ledger.json'),
        'elapsed_seconds':time.monotonic()-started,
        'elapsed_minutes':(time.monotonic()-started)/60,
        'build_seconds':build_seconds,'normalization_seconds':norm_seconds,
        'stopped_after_build_and_normalization':True,
        'attempts':all_attempt_rows,
        'accepted_manifests':[str((RAW_DART if x['episode']>=100 else RAW_CLEAN)/x['task']/f"ep{x['episode']}_manifest.json") for x in accepted]}
    proof('data_delivery.json',delivery)
    proof('data_phase_timing.json',{'started_utc':started_utc,'finished_utc':timestamp(),
        'elapsed_seconds':delivery['elapsed_seconds'],'build_seconds':build_seconds,
        'normalization_seconds':norm_seconds,'completed':True})
    return delivery
