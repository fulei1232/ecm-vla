#!/usr/bin/env python3
"""R12: validate false bottom, regenerate 400 demos, build/norm once, then stop."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

from v_a_common import ECM, POLICY, POLICY_PYTHON, SIM_PYTHON, OPENPI_DATA_HOME
from v_a_common import read_json, read_jsonl, atomic_json, sha256, timestamp
from fruit_r12_common import (OUT, TRACKED, REPORT, DOC, R11_OUT, R11_TRACKED,
                              RAW_CLEAN, RAW_DART, DATASET, proof, publish)
from fruit_r12_reporting import report
from fruit_r12_validity import audit_case

SOURCE_NAMES = (
    'fruit_picking/count_shield_r12_spec.py', 'fruit_picking/count_shield_r12_visual.py',
    'scripts/check_fruit_task_v2_r12_optics.py', 'scripts/fruit_r12_common.py',
    'scripts/fruit_r12_runtime.py', 'scripts/fruit_r12_validity.py',
    'scripts/fruit_r12_generation.py', 'scripts/fruit_r12_reporting.py',
    'scripts/audit_fruit_task_v2_r12_episode.py', 'scripts/run_fruit_task_v2_r12.py')
R11_PROOFS = ('R0_gate.json','summary.json','information_audit.json','clock_fit.json',
              'data_attempt_ledger.json','visibility_stop_review.json')


def r11_manifest_inventory():
    import fruit_r11_common
    rows=[]
    for purpose, root in (('clean',fruit_r11_common.RAW_CLEAN),('dart',fruit_r11_common.RAW_DART)):
        for path in sorted((root/'count').glob('ep*_manifest.json')):
            m=read_json(path)
            for name in ('h5','gt'):
                assert sha256(path.parent/m[name])==m[name+'_sha256'],str(path)
            rows.append({'purpose':purpose,'path':str(path),'sha256':sha256(path),
                         'episode':m['episode'],'N':m['N'],'seed':m['seed'],
                         'h5_sha256':m['h5_sha256'],'gt_sha256':m['gt_sha256']})
    return rows


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);TRACKED.mkdir(parents=True,exist_ok=True)
    cpu=read_json(OUT/'cpu_optics/result.json')
    assert cpu['passed'] and cpu['old_scheme']['blocked_observed_rays']==0
    assert cpu['new_scheme']['blocked_observed_rays']==127
    assert cpu['new_false_bottom']['minimum_edge_overlap_m']>=.002
    old_gate=read_json(R11_OUT/'R0_gate.json');assert old_gate['passed']
    old_summary=read_json(R11_TRACKED/'summary.json')
    assert old_summary['count_R0_replays']==old_summary['count_R0_target']==350
    assert old_summary['checks']['information_audit.json']['passed']
    inventory=r11_manifest_inventory()
    clean_n=sum(x['purpose']=='clean' for x in inventory)
    dart_n=sum(x['purpose']=='dart' for x in inventory)
    assert clean_n==100 and dart_n==9 and clean_n+dart_n==109
    reuse={'instruction_revision':12,'created_utc':timestamp(),'R11_R0_reused':True,
        'R11_count_R0_replays':350,'R11_information_gate_passed':True,
        'R11_successful_count_demonstrations_preserved':109,
        'R11_clean_manifests':clean_n,'R11_DART_manifests':dart_n,
        'R11_data_used_in_R12':False,'R10_83_data_used':False,
        'source_sha256':{str(R11_OUT/name):sha256(R11_OUT/name) for name in R11_PROOFS},
        'manifest_inventory':inventory}
    if (OUT/'approved_r11_reuse.json').exists():
        previous_reuse=read_json(OUT/'approved_r11_reuse.json')
        for key,value in reuse.items():
            if key!='created_utc':assert previous_reuse.get(key)==value,key
        reuse=previous_reuse
    else:proof('approved_r11_reuse.json',reuse)
    # The legacy replay recorder requires an R0 pass to write diagnostic HDF5.
    # This scoped gate permits only named R12 validation replays, never dataset generation.
    diagnostic_gate={'passed':True,'instruction_revision':12,'scope':'GPU validation artifacts only',
        'R11_R0_reused':True,'data_generation_authorized':False,
        'reason':'R12 data generation is locked until gpu_validation.json passes.'}
    if (OUT/'R0_gate.json').exists():
        assert read_json(OUT/'R0_gate.json')==diagnostic_gate or read_json(OUT/'R0_gate.json').get('data_generation_authorized') is True
    else:proof('R0_gate.json',diagnostic_gate)
    prior_plan=read_json(R11_OUT/'plan.json')
    source_hashes={}
    for name in SOURCE_NAMES:source_hashes[str(ECM/name)]=sha256(ECM/name)
    for name,h in prior_plan['source_sha256'].items():
        assert sha256(name)==h,name
        source_hashes[name]=h
    source_hashes[str(DOC)]=sha256(DOC)
    source_hashes[str(OUT/'cpu_optics/result.json')]=sha256(OUT/'cpu_optics/result.json')
    plan={'instruction_revision':12,'task_definition_revision':9,
        'R11_plan_sha256':sha256(R11_OUT/'plan.json'),
        'R11_R0_reuse_sha256':sha256(OUT/'approved_r11_reuse.json'),
        'source_sha256':source_hashes,'doc_sha256':sha256(DOC),
        'CPU_optics_sha256':sha256(OUT/'cpu_optics/result.json'),
        'observation_condition':'count_fixed_v2_r12_false_bottom/occluded_original',
        'false_bottom':cpu['new_false_bottom']['box'],
        'DART_recipe':'paired accepted clean layout; attempt-specific OU seed; exactly two randomly selected pick cycles at original strength',
        'DART_layout_pair':'each DART episode 100+i reuses its own task clean episode i accepted layout seed and N',
        'R11_350_R0_reused_not_repeated':True,'R11_109_successes_not_mixed':True,
        'old_83_not_mixed':True,'gpu_validation_before_generation':True,
        'gpu_ids':list(range(8)),'train_successes_target':400,'build_once':True,
        'normalization_once':True,'training':False,'val_used':False}
    if (OUT/'plan.json').exists():
        previous_plan=read_json(OUT/'plan.json')
        recovery=previous_plan.get('post_gpu_recovery')
        if recovery is not None:
            plan['post_gpu_recovery']=recovery
        if previous_plan!=plan:
            gpu_path=OUT/'gpu_validation.json'
            assert gpu_path.exists() and read_json(gpu_path).get('passed'), \
                'The frozen plan only permits a narrowly scoped code recovery after passed GPU validation'
            old_sources=previous_plan.get('source_sha256',{})
            new_sources=plan.get('source_sha256',{})
            changed_sources={name for name in set(old_sources)|set(new_sources)
                             if old_sources.get(name)!=new_sources.get(name)}
            allowed={str(ECM/name) for name in (
                'scripts/audit_fruit_task_v2_r12_episode.py',
                'scripts/fruit_r12_generation.py',
                'scripts/fruit_r12_runtime.py',
                'scripts/run_fruit_task_v2_r12.py')}
            assert changed_sources and changed_sources <= allowed, \
                f'Unexpected post-validation source changes: {sorted(changed_sources-allowed)}'
            old_base={k:v for k,v in previous_plan.items() if k not in ('source_sha256','post_gpu_recovery')}
            new_base={k:v for k,v in plan.items() if k not in ('source_sha256','post_gpu_recovery')}
            assert old_base==new_base, 'Recovery may update only the allowlisted metadata/restart code'
            if recovery is None:
                if not (OUT/'plan_pre_resume.json').exists():proof('plan_pre_resume.json',previous_plan)
                recovery={'kind':'resume_after_layout_seed_metadata_assertion',
                    'previous_plan_sha256':sha256(OUT/'plan.json'),
                    'gpu_validation_sha256':sha256(gpu_path),
                    'changed_sources':sorted(changed_sources),
                    'amendments':[],
                    'scope':'record occluded task layout seeds for both tasks and resume from preserved R12 attempts; no scene, replay, task, or validation changes'}
            else:
                recovery=dict(recovery)
                amendments=list(recovery.get('amendments',[]))
                newly_changed=sorted(changed_sources)
                if newly_changed:
                    ledger_race_fix=('dictionary changed size during iteration' in
                        (read_json(OUT/'status.json').get('reason','') if (OUT/'status.json').exists() else ''))
                    amendments.append({'previous_plan_sha256':sha256(OUT/'plan.json'),
                        'changed_sources':newly_changed,
                        'reason':('Snapshot the shared attempt ledger under its lock before scanning prior attempts; '
                                  'concurrent GPU lanes mutated it during iteration. No task, platform, replay or noise changes.')
                                 if ledger_race_fix else
                                 'Assign a distinct sequential index to each DART pick so exactly the two selected cycles receive noise; no task or platform definition changes.'})
                recovery['amendments']=amendments
                recovery['changed_sources']=sorted(set(recovery.get('changed_sources',[]))|changed_sources)
            plan['post_gpu_recovery']=recovery
            proof('plan.json',plan)
    else:proof('plan.json',plan)
    state=read_json(OUT/'status.json') if (OUT/'status.json').exists() else {
        'stage':'R12 false-bottom Count/DART data only','state':'r12_cpu_optics_passed',
        'cpu_optics_passed':True,'gpu_validated':False,'data_generation_started':False,
        'training':False}
    if state.get('state')=='R12_check_failed':
        trace=state.get('traceback','')
        gpu_path=OUT/'gpu_validation.json'
        gpu_passed=gpu_path.exists() and read_json(gpu_path).get('passed')
        recoverable_layout=("assert result['r12_layout_seed']" in trace and gpu_passed)
        recoverable_dart=("KeyError: 'r12_pick_index'" in trace and gpu_passed)
        recoverable_dart_cycles=('assert nonzero == selected' in trace and gpu_passed)
        recoverable_ledger_race=('RuntimeError: dictionary changed size during iteration' in trace
                                 and 'for prior in ledger.rows.values()' in trace and gpu_passed)
        if recoverable_ledger_race:
            # Must be checked first: the race was raised alongside a pre-fix audit trace.
            recoverable_layout=recoverable_dart=recoverable_dart_cycles=False
        if recoverable_layout:
            original_status=OUT/'status_failed_before_resume.json'
            if not original_status.exists():atomic_json(original_status,state)
            ledger_path=OUT/'data_attempt_ledger.json'
            assert ledger_path.exists()
            ledger=read_json(ledger_path)
            manifest_paths=sorted(RAW_CLEAN.glob('**/ep*_manifest.json'))
            assert len(manifest_paths)==77
            manifest_rows=[]
            for path in manifest_paths:
                manifest=read_json(path)
                assert manifest['task']=='count' and manifest['purpose']=='train_clean'
                assert manifest['success'] and manifest['r12_layout_seed'] is not None
                for field in ('h5','gt'):assert sha256(path.parent/manifest[field])==manifest[field+'_sha256']
                manifest_rows.append({'path':str(path),'sha256':sha256(path),
                    'episode':manifest['episode'],'N':manifest['N'],'seed':manifest['seed']})
            incomplete=[row for row in ledger['attempts'] if 'outcome' not in row]
            invalid_replay=None
            for row in incomplete:
                result_path=Path(row['key'])/'result.json'
                if row.get('task')=='occluded_ripe' and result_path.exists():
                    result=read_json(result_path)
                    invalid_replay={'attempt':row['key'],'result_sha256':sha256(result_path),
                        'success':result.get('success'),'outcome':result.get('outcome'),
                        'r12_layout_seed':result.get('r12_layout_seed'),
                        'retained_without_manifest':True}
                    assert result.get('success') and result.get('r12_layout_seed') is None
                    break
            assert len(incomplete)==8 and invalid_replay is not None
            recovery={'instruction_revision':12,'created_utc':timestamp(),
                'cause':'Occluded task reset did not attach R12 layout-seed metadata; replay succeeded but failed the seed assertion.',
                'original_status_sha256':sha256(original_status),
                'original_status':str(original_status),
                'gpu_validation_sha256':sha256(gpu_path),'gpu_validation_passed':True,
                'accepted_r12_clean_count_manifests':len(manifest_rows),
                'accepted_manifest_inventory':manifest_rows,
                'preserved_incomplete_attempts':len(incomplete),
                'incomplete_attempt_keys':[row['key'] for row in incomplete],
                'successful_replay_excluded_for_missing_metadata':invalid_replay,
                'resume_policy':'Skip all existing ledger keys and accepted manifests; use the next unused attempt index; preserve every old result and ledger row; continue R12 generation without repeating GPU validation.'}
            recovery_path=OUT/'resume_recovery.json'
            if recovery_path.exists():
                prior=read_json(recovery_path)
                assert prior['original_status_sha256']==recovery['original_status_sha256']
                assert prior['gpu_validation_sha256']==recovery['gpu_validation_sha256']
            else:proof('resume_recovery.json',recovery)
            state.update(state='r12_resume_ready',reason='Recovered only the occluded-task layout-seed recording assertion; existing GPU validation and 77 accepted clean demonstrations are preserved.',
                resume_recovery=str(recovery_path),data_generation_started=True,
                data_generation_completed=False,gpu_validated=True,training=False,
                updated_utc=timestamp())
            state.pop('traceback',None)
        elif recoverable_dart:
            original_status=OUT/'status_failed_before_dart_recovery.json'
            if not original_status.exists():atomic_json(original_status,state)
            ledger_path=OUT/'data_attempt_ledger.json'
            ledger=read_json(ledger_path)
            clean_manifests=sorted(RAW_CLEAN.glob('**/ep*_manifest.json'))
            dart_manifests=sorted(RAW_DART.glob('**/ep*_manifest.json'))
            assert len(clean_manifests)==200 and len(dart_manifests)==0
            counts=Counter(read_json(p)['task'] for p in clean_manifests)
            assert counts==Counter({'count':100,'occluded_ripe':100})
            manifest_inventory=[]
            for path in clean_manifests:
                manifest=read_json(path)
                assert manifest['success'] and manifest['verification']['passed']
                for field in ('h5','gt'):assert sha256(path.parent/manifest[field])==manifest[field+'_sha256']
                manifest_inventory.append({'path':str(path),'sha256':sha256(path),
                    'task':manifest['task'],'episode':manifest['episode'],'N':manifest['N'],
                    'layout_seed':manifest['r12_layout_seed']})
            incomplete_dart=[row for row in ledger['attempts']
                if row.get('purpose')=='train_dart' and 'outcome' not in row]
            assert len(incomplete_dart)==8
            audit_failure=None
            for row in incomplete_dart:
                attempt=Path(row['key'])
                result_path=attempt/'result.json'
                actions_path=attempt/'actions.json'
                if result_path.exists() and actions_path.exists():
                    result=read_json(result_path)
                    actions=read_json(actions_path)
                    missing=sum('r12_pick_index' not in action for action in actions)
                    audit_failure={'attempt':row['key'],'result_sha256':sha256(result_path),
                        'actions_sha256':sha256(actions_path),'result_success':result.get('success'),
                        'result_outcome':result.get('outcome'),'actions':len(actions),
                        'actions_missing_cycle_annotation':missing,
                        'retained_without_manifest':True}
                    assert missing>0
                    break
            assert audit_failure is not None
            recovery={'instruction_revision':12,'created_utc':timestamp(),
                'cause':'DART validity audit found unannotated actions because the recorded action step is post-step while the annotation compared against the pre-step control count.',
                'original_status_sha256':sha256(original_status),
                'original_status':str(original_status),
                'gpu_validation_sha256':sha256(gpu_path),'gpu_validation_passed':True,
                'accepted_clean_manifests':len(clean_manifests),
                'clean_manifest_inventory':manifest_inventory,
                'incomplete_dart_attempts':len(incomplete_dart),
                'incomplete_dart_attempt_keys':[row['key'] for row in incomplete_dart],
                'audit_failure_attempt':audit_failure,
                'resume_policy':'Keep all 200 clean manifests and every DART attempt; retry the eight unfinished DART episodes with the same paired clean N/layout and the next unused attempt seed; do not repeat GPU validation.'}
            recovery_path=OUT/'resume_recovery_dart_audit.json'
            if recovery_path.exists():
                prior=read_json(recovery_path)
                assert prior['original_status_sha256']==recovery['original_status_sha256']
                assert prior['gpu_validation_sha256']==recovery['gpu_validation_sha256']
            else:proof('resume_recovery_dart_audit.json',recovery)
            state.update(state='r12_resume_ready',reason='DART action-cycle annotation was off by one because replay actions store the post-step index; annotation was corrected without changing the physical action or noise.',
                resume_recovery_dart_audit=str(recovery_path),data_generation_started=True,
                data_generation_completed=False,gpu_validated=True,training=False,
                updated_utc=timestamp())
            state.pop('traceback',None)
        elif recoverable_dart_cycles:
            original_status=OUT/'status_failed_before_dart_cycle_selection_recovery.json'
            if not original_status.exists():atomic_json(original_status,state)
            ledger=read_json(OUT/'data_attempt_ledger.json')
            clean_manifests=sorted(RAW_CLEAN.glob('**/ep*_manifest.json'))
            dart_manifests=sorted(RAW_DART.glob('**/ep*_manifest.json'))
            assert len(clean_manifests)==200 and len(dart_manifests)==0
            counts=Counter(read_json(p)['task'] for p in clean_manifests)
            assert counts==Counter({'count':100,'occluded_ripe':100})
            for path in clean_manifests:
                manifest=read_json(path)
                assert manifest['success'] and manifest['verification']['passed']
                for field in ('h5','gt'):assert sha256(path.parent/manifest[field])==manifest[field+'_sha256']
            incomplete_dart=[row for row in ledger['attempts']
                if row.get('purpose')=='train_dart' and 'outcome' not in row]
            assert incomplete_dart
            audit_failure=None
            for row in reversed(incomplete_dart):
                attempt=Path(row['key'])
                result_path=attempt/'result.json'
                actions_path=attempt/'actions.json'
                if not (result_path.exists() and actions_path.exists()):continue
                result=read_json(result_path)
                if not (result.get('success') and result.get('r12_dart_two_cycle_recipe')):continue
                selected=set(result['r12_dart_perturbation_cycles'])
                actions=read_json(actions_path)
                nonzero=set()
                for action in actions:
                    if any(abs(float(value))>0 for value in action.get('noise_offset',[])):
                        nonzero.add(action.get('r12_pick_index'))
                if nonzero!=selected:
                    audit_failure={'attempt':row['key'],'result_sha256':sha256(result_path),
                        'actions_sha256':sha256(actions_path),'success':result['success'],
                        'outcome':result['outcome'],'N':result['N'],
                        'selected_cycles':sorted(selected),
                        'observed_noisy_cycles':sorted(nonzero,key=lambda x:-1 if x is None else x),
                        'actions':len(actions),'actions_missing_cycle_annotation':sum(
                            'r12_pick_index' not in action for action in actions),
                        'retained_without_manifest':True}
                    break
            assert audit_failure is not None
            recovery={'instruction_revision':12,'created_utc':timestamp(),
                'cause':'The episode wrapper passed pick_index=0 on every solver.pick call, so later pick cycles were mislabeled and the two selected DART cycles did not both receive noise.',
                'original_status_sha256':sha256(original_status),
                'original_status':str(original_status),
                'gpu_validation_sha256':sha256(OUT/'gpu_validation.json'),'gpu_validation_passed':True,
                'accepted_clean_count':100,'accepted_clean_occluded_ripe':100,
                'accepted_DART_manifests':0,'incomplete_DART_attempts':len(incomplete_dart),
                'incomplete_DART_attempt_keys':[row['key'] for row in incomplete_dart],
                'audit_failure_attempt':audit_failure,
                'resume_policy':'Preserve all clean manifests and every prior DART attempt; retry each unfinished DART episode at the next unused attempt with its paired clean N/layout and a new noise seed; do not rerun GPU validation.'}
            recovery_path=OUT/'resume_recovery_dart_cycle_selection.json'
            if recovery_path.exists():
                prior=read_json(recovery_path)
                assert prior['original_status_sha256']==recovery['original_status_sha256']
                assert prior['gpu_validation_sha256']==recovery['gpu_validation_sha256']
            else:proof('resume_recovery_dart_cycle_selection.json',recovery)
            state.update(state='r12_resume_ready',reason='DART wrapper reused pick index 0 for every pick; it now assigns sequential cycle indices so the two selected cycles receive the planned perturbations.',
                resume_recovery_dart_cycle_selection=str(recovery_path),data_generation_started=True,
                data_generation_completed=False,gpu_validated=True,training=False,
                updated_utc=timestamp())
            state.pop('traceback',None)
        elif recoverable_ledger_race:
            original_status=OUT/'status_failed_before_ledger_race_recovery.json'
            if not original_status.exists():atomic_json(original_status,state)
            ledger=read_json(OUT/'data_attempt_ledger.json')
            clean_manifests=sorted(RAW_CLEAN.glob('**/ep*_manifest.json'))
            dart_manifests=sorted(RAW_DART.glob('**/ep*_manifest.json'))
            assert len(clean_manifests)==200
            counts=Counter(read_json(p)['task'] for p in clean_manifests)
            assert counts==Counter({'count':100,'occluded_ripe':100})
            for path in clean_manifests+dart_manifests:
                manifest=read_json(path)
                assert manifest['success'] and manifest['verification']['passed']
                for field in ('h5','gt'):assert sha256(path.parent/manifest[field])==manifest[field+'_sha256']
            incomplete=[row['key'] for row in ledger['attempts'] if 'outcome' not in row]
            recovery={'instruction_revision':12,'created_utc':timestamp(),
                'cause':'Eight GPU lanes share one AttemptLedger; one lane iterated ledger.rows without the lock while another inserted a row (RuntimeError: dictionary changed size during iteration). Fixed by copying the rows under ledger.lock before scanning.',
                'original_status_sha256':sha256(original_status),
                'original_status':str(original_status),
                'gpu_validation_sha256':sha256(OUT/'gpu_validation.json'),'gpu_validation_passed':True,
                'accepted_clean_manifests':len(clean_manifests),
                'accepted_DART_manifests':len(dart_manifests),
                'incomplete_attempts':len(incomplete),'incomplete_attempt_keys':incomplete,
                'resume_policy':'Preserve all manifests and every ledger row; unfinished attempts keep their slots and each DART episode resumes at its next unused attempt with the paired clean N/layout; no GPU validation rerun; physical actions, noise and task definitions unchanged.'}
            recovery_path=OUT/'resume_recovery_ledger_race.json'
            if recovery_path.exists():
                prior=read_json(recovery_path)
                assert prior['original_status_sha256']==recovery['original_status_sha256']
            else:proof('resume_recovery_ledger_race.json',recovery)
            state.update(state='r12_resume_ready',reason='Shared attempt ledger is now snapshotted under its lock before scanning; resume DART generation.',
                resume_recovery_ledger_race=str(recovery_path),data_generation_started=True,
                data_generation_completed=False,gpu_validated=True,training=False,
                updated_utc=timestamp())
            state.pop('traceback',None)
    state.update(cpu_optics_passed=True,plan_sha256=sha256(OUT/'plan.json'),
                 R11_R0_reused=True,R11_109_successes_mixed=False,training=False)
    atomic_json(OUT/'status.json',state)
    return state,plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    state,plan=prepare()
    if not args.run:
        report(state)
        print({'prepared':True,'state':state['state'],'first_gpu_validation':'ep113 old/new','training':False})
        return
    terminal=('R12_GPU_validation_failed','R12_check_failed','awaiting_data_review')
    if state['state'] in terminal:
        print(state);return
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not Path('/dev/nvidiactl').exists():raise RuntimeError('Use the authorized GPU runtime; no NVIDIA device here')
        children={};mutex=threading.Lock();stop=threading.Event()
        def abort():
            stop.set()
            with mutex:
                for child in children.values():
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
        def check_sources():
            for name,h in plan['source_sha256'].items():assert sha256(name)==h,name
            assert sha256(DOC)==plan['doc_sha256']
            assert sha256(OUT/'cpu_optics/result.json')==plan['CPU_optics_sha256']
            assert sha256(R11_OUT/'plan.json')==plan['R11_plan_sha256']
            if 'post_gpu_recovery' in plan:
                assert sha256(OUT/'gpu_validation.json')==plan['post_gpu_recovery']['gpu_validation_sha256']
                assert sha256(OUT/'resume_recovery.json')==sha256(TRACKED/'resume_recovery.json')
            for name,h in read_json(OUT/'approved_r11_reuse.json')['source_sha256'].items():assert sha256(name)==h,name
        def child_run(command,d,gpu=None,cpu=False,cwd=ECM):
            if stop.is_set():raise RuntimeError('R12 stopped; no new process starts')
            check_sources();d.mkdir(parents=True,exist_ok=True)
            if any(d.iterdir()):raise RuntimeError(f'Refusing to overwrite R12 output {d}')
            env={**os.environ,'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json',
                'MPLCONFIGDIR':str(ECM/'cache/matplotlib'),'PYTHONDONTWRITEBYTECODE':'1',
                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'OPENPI_DATA_HOME':str(OPENPI_DATA_HOME),'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
                'PYTHONPATH':str(POLICY/'src')+':'+str(POLICY/'packages/openpi-client/src')}
            if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
            if cpu:env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
            atomic_json(d/'command.json',{'command':command,'cwd':str(cwd),'utc':timestamp()})
            with (d/'console.log').open('x') as log:
                started=time.monotonic();started_utc=timestamp()
                child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with mutex:children[child.pid]=child
                try:code=child.wait(timeout=43200)
                finally:
                    if child.poll() is not None:
                        with mutex:children.pop(child.pid,None)
            timing={'started_utc':started_utc,'finished_utc':timestamp(),
                    'process_wall_seconds':time.monotonic()-started,'exit_code':code}
            atomic_json(d/'process_exit.json',timing)
            if code:raise RuntimeError(f'R12 child failed exit{code}: {d}/console.log')
            return timing

        def command_episode(task,level,episode,purpose,attempt,out,scheme='new',gpu=0,
                            layout_seed=None,dart_two_cycles=False,capture=False,expect_fail=False):
            cmd=[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_task_v2_r12_episode.py'),
                '--task',task,'--level',level,'--episode',str(episode),'--purpose',purpose,
                '--out',str(out),'--task-version','v2','--attempt',str(attempt),
                '--r12-shield-scheme',scheme]
            if layout_seed is not None:cmd+=['--r12-layout-seed',str(layout_seed)]
            if dart_two_cycles:cmd+=['--r12-dart-two-cycles']
            if capture:cmd+=['--r12-capture-front-hashes']
            if expect_fail:cmd+=['--r12-expect-visibility-failure']
            return child_run(cmd,out,gpu=gpu)

        def run_attempt(task,level,episode,gpu,purpose,attempt,*,layout_seed=None,dart_two_cycles=False):
            start=time.monotonic()
            out=OUT/'replays'/purpose/task/level/f'ep{episode}_attempt{attempt}'
            command_episode(task,level,episode,purpose,attempt,out,scheme='new',gpu=gpu,
                            layout_seed=layout_seed,dart_two_cycles=dart_two_cycles)
            result=read_json(out/'result.json')
            assert result['task']==task and result['level']==level and result['episode']==episode
            assert result['purpose']==purpose and result['attempt']==attempt
            expected_seed=(4100000 if purpose.startswith('train_') else 4200000)+16*episode+attempt
            assert result['seed']==expected_seed
            assert result['r12_layout_seed']==(layout_seed if layout_seed is not None else expected_seed)
            validity=audit_case(out)
            atomic_json(out/'validity.json',validity)
            atomic_json(out/'process_timing.json',{'validated_case_wall_seconds':time.monotonic()-start,
                'task':task,'purpose':purpose,'episode':episode,'attempt':attempt,
                'layout_seed':result['r12_layout_seed'],'N':result['N'],'success':result['success'],
                'outcome':result['outcome'],'physical_steps':result['steps']})
            return {**result,'source':str(out),
                    'validated_case_wall_seconds':time.monotonic()-start,
                    'result_sha256':sha256(out/'result.json')}

        try:
            gpu_proof=OUT/'gpu_validation.json'
            if gpu_proof.exists() and read_json(gpu_proof).get('passed'):
                gate=read_json(OUT/'R0_gate.json')
                assert gate.get('gpu_validation_sha256')==sha256(gpu_proof)
                gate.update(data_generation_authorized=True,
                            gate_reason='R12 GPU validations were already passed and are reused after the scoped metadata recovery.')
                proof('R0_gate.json',gate)
                state.update(state='r12_gpu_validations_passed',gpu_validated=True,
                             data_generation_started=True,training=False,updated_utc=timestamp())
                report(state)
            else:
                nvidia=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,check=True)
                state.update(state='r12_gpu_ep113_validation',gpu_list=nvidia.stdout,updated_utc=timestamp())
                report(state)
                # Validation 1: the original ep113 attempt3 command, first old, then new.
                old113=OUT/'replays/gpu_validation/ep113_old'
                new113=OUT/'replays/gpu_validation/ep113_new'
                command_episode('count','history',113,'train_dart',3,old113,scheme='old',gpu=0,
                                capture=True,expect_fail=True)
                old_stop=read_json(old113/'r12_expected_stop.json')['failure']
                original_stop=read_json(R11_OUT/'replays/train_dart/count/history/ep113_attempt3/check_failure.json')
                assert old_stop==original_stop
                assert old_stop['step']==1230
                assert old_stop['details']['hand_camera']['basket_fruit']==127
                command_episode('count','history',113,'train_dart',3,new113,scheme='new',gpu=0,capture=True)
                new113_result=read_json(new113/'result.json')
                new113_frames=read_jsonl(new113/'frame_metrics.jsonl')
                assert new113_result['max_visibility']
                assert all(v['basket_fruit']==0 and v['inactive_reserve']==0
                           for frame in new113_frames for v in frame['visibility'].values())
                assert all(v[k]==0 for v in new113_result['max_visibility'].values()
                           for k in ('basket_fruit','inactive_reserve'))
                new113_validity=audit_case(new113)
                atomic_json(new113/'validity.json',new113_validity)
                # The old/new front-camera prefix is also recorded as a diagnostic.
                old_hash=read_jsonl(old113/'front_rgb_sha256.jsonl')
                new_hash=read_jsonl(new113/'front_rgb_sha256.jsonl')
                common_prefix=old_hash==new_hash[:len(old_hash)]

                # Validation 2: replay the same 50 expert seeds in the old and new
                # modes. The old replay recreates per-frame RGB missing from R11.
                state.update(state='r12_gpu_expert50_validation',updated_utc=timestamp())
                report(state)
                episodes=list(range(200,250))
                def old_lane(gpu):
                    rows=[]
                    for episode in episodes[gpu::8]:
                        out=OUT/'replays/gpu_validation/expert_old/count/history'/f'ep{episode}'
                        command_episode('count','history',episode,'audit',0,out,scheme='old',gpu=gpu,capture=True)
                        rows.append(episode)
                    return rows
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for future in as_completed([pool.submit(old_lane,g) for g in range(8)]):future.result()
                def compare_old_to_r11(episode):
                    current=OUT/'replays/gpu_validation/expert_old/count/history'/f'ep{episode}'
                    previous=R11_OUT/'replays/audit/count/history'/f'ep{episode}'
                    r=read_json(current/'result.json');old=read_json(previous/'result.json')
                    fields=('success','outcome','steps','N','seed','green_step','final_robot_picked','final_robot_in_basket')
                    assert {k:r[k] for k in fields}=={k:old[k] for k in fields},f'Old replay differs from accepted R11 ep{episode}'
                    assert sha256(current/'actions.json')==sha256(previous/'actions.json')
                    assert sha256(current/'trace.jsonl')==sha256(previous/'trace.jsonl')
                    for stage,item in old['snapshots'].items():
                        current_file=current/item['files']['front_camera']
                        previous_file=previous/item['files']['front_camera']
                        assert sha256(current_file)==sha256(previous_file)
                    return {'episode':episode,'success':r['success'],'outcome':r['outcome'],
                            'steps':r['steps'],'N':r['N'],'old_R11_trace_and_actions_reproduced':True}
                old_reproduction=[]
                for e in episodes:old_reproduction.append(compare_old_to_r11(e))

                def new_lane(gpu):
                    rows=[]
                    for episode in episodes[gpu::8]:
                        out=OUT/'replays/gpu_validation/expert_new/count/history'/f'ep{episode}'
                        command_episode('count','history',episode,'audit',0,out,scheme='new',gpu=gpu,capture=True)
                        new_result=read_json(out/'result.json')
                        old_dir=OUT/'replays/gpu_validation/expert_old/count/history'/f'ep{episode}'
                        old_result=read_json(old_dir/'result.json')
                        fields=('success','outcome','steps','N','seed','green_step','final_robot_picked','final_robot_in_basket')
                        assert {k:new_result[k] for k in fields}=={k:old_result[k] for k in fields}
                        new_hash=read_jsonl(out/'front_rgb_sha256.jsonl')
                        old_hash=read_jsonl(old_dir/'front_rgb_sha256.jsonl')
                        assert new_hash==old_hash,f'Front-camera frame mismatch ep{episode}'
                        assert len(new_hash)==new_result['steps']+1
                        validity=audit_case(out)
                        atomic_json(out/'validity.json',validity)
                        vis=read_jsonl(out/'frame_metrics.jsonl')
                        assert all(v['basket_fruit']==0 and v['inactive_reserve']==0
                                   for f in vis for v in f['visibility'].values())
                        rows.append({'episode':episode,'N':new_result['N'],'success':new_result['success'],
                            'outcome':new_result['outcome'],'steps':new_result['steps'],
                            'observations':len(new_hash),'front_rgb_identical_all_frames':True,
                            'hidden_content_zero_all_frames':True})
                    return rows
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures=[pool.submit(new_lane,g) for g in range(8)]
                    new_replays=[row for future in as_completed(futures) for row in future.result()]
                new_replays.sort(key=lambda x:x['episode'])
                assert len(old_reproduction)==len(new_replays)==50
                gpu_value={'instruction_revision':12,'passed':True,'created_utc':timestamp(),
                    'ep113':{'old_scheme_reproduced_previous_failure':True,'old_failure':old_stop,
                        'new_scheme_max_content_pixels':{c:{k:v[k] for k in ('basket_fruit','inactive_reserve')}
                            for c,v in new113_result['max_visibility'].items()},
                        'new_scheme_content_zero_all_frames':True,
                        'old_new_front_rgb_common_prefix_identical':common_prefix,
                        'old_replay':str(old113),'new_replay':str(new113)},
                    'expert50':{'old_R11_outcomes_and_full_traces_reproduced':True,
                        'new_outcomes_match_old':True,'front_camera_rgb_identical_per_frame':True,
                        'new_basket_fruit_and_reserve_zero_all_frames':True,
                        'old_reproduction':old_reproduction,'new_replays':new_replays},
                    'R11_350_R0_reused_not_repeated':True,'training':False}
                assert all(gpu_value['ep113']['new_scheme_max_content_pixels'][c][k]==0
                           for c in gpu_value['ep113']['new_scheme_max_content_pixels']
                           for k in ('basket_fruit','inactive_reserve'))
                proof('gpu_validation.json',gpu_value)
                gate=read_json(OUT/'R0_gate.json')
                gate.update(data_generation_authorized=True,gpu_validation_sha256=sha256(OUT/'gpu_validation.json'),
                            gate_reason='R12 CPU optics and both required GPU validations passed.')
                proof('R0_gate.json',gate)
                state.update(state='r12_gpu_validations_passed',gpu_validated=True,
                             data_generation_started=False,training=False,updated_utc=timestamp())
                report(state)

            from fruit_r12_generation import generate
            state['data_generation_started']=True
            def execute_data(task,level,episode,gpu,purpose,attempt,*,r12_layout_seed=None,
                             dart_two_cycles=False,data_key=None):
                return run_attempt(task,level,episode,gpu,purpose,attempt,
                                   layout_seed=r12_layout_seed,dart_two_cycles=dart_two_cycles)
            delivery=generate(state,execute_data,lambda label,cmd,gpu=None,cpu=False,cwd=ECM:
                child_run(cmd,OUT/'commands'/label,gpu,cpu,cwd),abort)
            state.update(state='awaiting_data_review',data_generation_completed=True,
                train_data_generated=True,training=False,accepted_demonstrations=400,
                dataset=str(DATASET),data_delivery_sha256=sha256(OUT/'data_delivery.json'),
                reason='Both GPU validations passed; 400 new demonstrations built and one shared normalization computed; stop before training.',
                updated_utc=timestamp())
            report(state,delivery)
            print({'state':state['state'],'demonstrations':400,'training':False,'next':'await review'})
        except Exception as exc:
            abort()
            state.update(state='R12_check_failed',reason=str(exc),traceback=traceback.format_exc(),
                data_generation_completed=False,training=False,updated_utc=timestamp())
            report(state)
            raise
        finally:
            abort()


if __name__=='__main__':main()
