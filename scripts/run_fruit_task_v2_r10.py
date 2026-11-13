#!/usr/bin/env python3
"""R10 resumes at clock calibration; accepted R9 checks are never rerun. No training."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

from v_a_common import ECM,POLICY,POLICY_PYTHON,SIM_PYTHON,OPENPI_DATA_HOME,read_json,atomic_json,sha256,timestamp
from fruit_r10_common import OUT,TRACKED,DOC,R9,activate,publish,proof,preserve
from prepare_fruit_task_v2_r10 import prepare_evidence
from fruit_r10_reporting import report


def prepare():
    support=activate();reuse=prepare_evidence();preserve()
    assert read_json(OUT/'continuation_tests.json')['passed']
    old=read_json(R9/'plan.json')
    files=[ECM/name for name in ('scripts/fruit_r10_common.py','scripts/prepare_fruit_task_v2_r10.py',
        'scripts/audit_fruit_task_v2_r10_episode.py','scripts/fruit_r10_reporting.py',
        'scripts/fruit_r10_data_timing.py','scripts/run_fruit_task_v2_r10.py',
        'tests/fruit_picking/test_r10_continuation.py')]
    sources={**old['source_sha256'],**{str(p):sha256(p) for p in files}}
    plan={'revision':10,'task_definition_revision':9,'R9_plan_sha256':sha256(R9/'plan.json'),
        'source_sha256':sources,'doc_sha256':sha256(DOC),'approved_reuse_sha256':sha256(OUT/'approved_reuse.json'),
        'registry_sha256':sha256(ECM/'reports/split_registry_fruit_task_v2_r10_20261106.json'),
        'continuation_tests_sha256':sha256(OUT/'continuation_tests.json'),
        'performance_reassessment_sha256':sha256(OUT/'performance_reassessment_r10.json'),
        'budget_check_sha256':sha256(OUT/'budget_check.json'),
        'dev_audit':old['dev_audit'],'dev_diagnostic':old['dev_diagnostic'],'cohorts':old['cohorts'],
        'audit_seed':old['audit_seed'],'train_seed':old['train_seed'],'budget':2800,'gpus':list(range(8)),
        'memory':old['memory'],'temporal_index':'order','distance_signal':'any_hanging',
        'reuse_completed_checks_without_replay':True,'information_gate_after_350_main_cases':True,
        'data_only_after_all_gates':True,'report_generation_total_and_per_N_wall_times':True,
        'training':False,'val_use':False}
    if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json')==plan,'Sources/definitions changed; refuse mixed run'
    else:proof('plan.json',plan)
    state=read_json(OUT/'status.json') if (OUT/'status.json').exists() else {
        'revision':10,'stage':'R0','state':'r10_prepared_continuation','training':False,'train_data_generated':False,
        'approved_R9_reuse':str(OUT/'approved_reuse.json'),'old_R9_failure_retained':True}
    return support,state,plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True);TRACKED.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        support,state,plan=prepare()
        if state['state'].startswith('R10_') or state['state']=='awaiting_data_review':print(state);return
        if not a.run:
            publish(state,state='ready_for_r10_clock_calibration');report(state,support)
            print({'ready':True,'first_new_work':'clock calibration','repeated_accepted_checks':0,'training':False});return
        if not Path('/dev/nvidiactl').exists():raise RuntimeError('Use the authorized GPU terminal; no NVIDIA device here')
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
            assert sha256(ECM/'reports/split_registry_fruit_task_v2_r10_20261106.json')==plan['registry_sha256']
        def child_run(command,d,gpu=None,cpu=False,cwd=ECM):
            if stop.is_set():raise RuntimeError('R10 stopped; no new work')
            check_sources();d.mkdir(parents=True,exist_ok=True)
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
            if code:raise RuntimeError(f'R10 process/check failed exit{code}: {d}/console.log')
            return timing
        def run_command(label,command,gpu=None,cpu=False,cwd=ECM):
            try:return child_run(command,OUT/'commands'/label,gpu,cpu,cwd)
            except Exception:abort();raise
        def execute(task,level,episode,gpu,purpose='audit',attempt=0):
            try:
                if stop.is_set():raise RuntimeError('R10 stopped; no new episode')
                check_sources();case_started=time.monotonic()
                suffix=f'ep{episode}' if not purpose.startswith('train_') else f'ep{episode}_attempt{attempt}'
                d=OUT/'replays'/purpose/task/level/suffix
                if (d/'validity.json').exists():
                    v=read_json(d/'validity.json');assert v['plan_sha256']==sha256(OUT/'plan.json')
                    for name,h in v['source_sha256'].items():assert sha256(d/name)==h
                    return {**read_json(d/'result.json'),'source':str(d)}
                if d.exists() and any(d.iterdir()):raise RuntimeError(f'Partial attempt retained; no retry: {d}')
                d.mkdir(parents=True,exist_ok=True)
                command=[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_task_v2_r10_episode.py'),
                    '--task',task,'--level',level,'--episode',str(episode),'--purpose',purpose,
                    '--out',str(d),'--task-version','v2','--attempt',str(attempt)]
                if level.startswith('clock'):
                    assert sha256(OUT/'clock_fit.json')==state['clock_fit_sha256']
                    command+=['--clock-fit',str(OUT/'clock_fit.json')]
                atomic_json(d/'started.json',{'command':command,'utc':timestamp(),
                    'plan_sha256':sha256(OUT/'plan.json'),'instruction_revision':10,'task_definition_revision':9})
                timing=child_run(command,d,gpu)
                try:validity=support.audit_case(d)
                except Exception as exc:
                    atomic_json(d/'validity_failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
                    raise RuntimeError(f'R10 validity failed: {d}/validity_failure.json') from exc
                r=read_json(d/'result.json');seed=(4100000 if purpose.startswith('train_') else 4200000)+16*episode+attempt
                assert (r['task'],r['level'],r['episode'],r['purpose'],r['seed'])==(task,level,episode,purpose,seed)
                atomic_json(d/'validity.json',{**validity,'plan_sha256':sha256(OUT/'plan.json'),'instruction_revision':10})
                atomic_json(d/'process_timing.json',{**timing,'validated_case_wall_seconds':time.monotonic()-case_started,
                    'task':task,'purpose':purpose,'episode':episode,'attempt':attempt,'N':r['N'],
                    'success':r['success'],'physical_steps':r['steps'],'result_available':True})
                print({k:r[k] for k in ('task','level','purpose','episode','N','success','outcome','steps')},flush=True)
                return {**r,'source':str(d)}
            except Exception:abort();raise
        def cohort(level,purpose='audit'):
            episodes=plan['dev_diagnostic'] if purpose=='calibration' else plan['dev_audit']
            def lane(gpu):
                for e in episodes[gpu::8]:execute('count',level,e,gpu,purpose)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in as_completed([pool.submit(lane,gpu) for gpu in range(8)]):future.result()
        try:
            # §18 explicitly reuses all successful unit/pressure evidence. Start here.
            publish(state,state='r10_clock_calibration')
            cohort('history','calibration')
            if not (OUT/'clock_fit.json').exists():support.fit_calibration()
            publish(state,clock_fit_sha256=sha256(OUT/'clock_fit.json'))
            for level in plan['cohorts']:
                publish(state,state='r10_count_audit',level=level)
                cohort(level);rows=support.collect('audit',level);assert len(rows)==50
                if not support.gate(level,rows):
                    publish(state,state='R10_shortcut_gate_failed',reason=f'{level}: {sum(r["success"] for r in rows)}/50')
                    report(state,support);return
                report(state,support)
            publish(state,state='r10_information_gate')
            info=support.information(support.collect('audit','history'))
            if not info['passed']:
                publish(state,state='R10_information_gate_failed',reason=str(info['criteria']));report(state,support);return
            rows=support.collect();assert len(rows)==350
            other={'passed':all(not r['early_start_steps'] and all(c['passed'] for c in r['appearance_checks'])
                and not any(v[k] for v in r['max_visibility'].values() for k in ('basket','basket_fruit','inactive_reserve'))
                for r in rows),'cases':350,'source':'Existing per-episode validity proofs and runtime checks; no extra rollout.'}
            proof('other_checks.json',other);assert other['passed']
            check_sources();preserve()
            names=('approved_reuse.json','performance_reassessment_r10.json','budget_check.json','information_audit.json','other_checks.json')
            proof('R0_gate.json',{'passed':True,'revision':10,'task_definition_revision':9,'created_utc':timestamp(),
                'proof_sha256':{name:sha256(OUT/name) for name in names},
                'count_main_cases':350,'training':False,'data_only_authorized':True})
            from fruit_r10_data_timing import generate_data_with_timing
            generate_data_with_timing(state,execute,run_command,check_sources,abort)
            check_sources();preserve()
            publish(state,state='awaiting_data_review',reason='R0 passed; data generation, build, shared normalization and timing reports completed; stop before training.')
            report(state,support)
        except Exception as exc:
            abort();publish(state,state='R10_check_failed',reason=str(exc),traceback=traceback.format_exc())
            report(state,support);raise
        finally:abort()


if __name__=='__main__':main()
