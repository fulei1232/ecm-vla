#!/usr/bin/env python3
"""Approved Count-v2 shield: ep112 diagnosis, new Count R0, new data, then stop."""
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
from fruit_shield_buildfix_common import OUT,TRACKED,DOC,PREVIOUS,activate,publish,proof,preserve,report
from prepare_fruit_count_shield_buildfix import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True);TRACKED.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        support,state,plan=prepare()
        if state['state'].startswith('SHIELD_') or state['state']=='awaiting_data_review':print(state);return
        if not a.run:
            publish(state,state='ready_for_ep112_shield_diagnostic');report(state,support)
            print({'ready':True,'first_new_work':'ep112 shield diagnostic','repeated_accepted_checks':0,'training':False});return
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
            for name,h in plan['reused_copies_sha256'].items():assert sha256(name)==h,name
            assert sha256(DOC)==plan['doc_sha256']
            assert sha256(OUT/'new_tests.json')==plan['tests_sha256']
            assert sha256(OUT/'builder_tests.json')==plan['builder_tests_sha256']
            assert sha256(OUT/'optical_geometry_check.json')==plan['optical_geometry_check_sha256']
            assert sha256(ECM/'reports/split_registry_fruit_count_shield_buildfix_20261106.json')==plan['registry_sha256']
        def child_run(command,d,gpu=None,cpu=False,cwd=ECM):
            if stop.is_set():raise RuntimeError('SHIELD stopped; no new work')
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
            if code:raise RuntimeError(f'SHIELD process/check failed exit{code}: {d}/console.log')
            return timing
        def run_command(label,command,gpu=None,cpu=False,cwd=ECM):
            try:return child_run(command,OUT/'commands'/label,gpu,cpu,cwd)
            except Exception:abort();raise
        def execute(task,level,episode,gpu,purpose='audit',attempt=0):
            try:
                if stop.is_set():raise RuntimeError('SHIELD stopped; no new episode')
                check_sources();case_started=time.monotonic()
                suffix=f'ep{episode}' if not purpose.startswith('train_') else f'ep{episode}_attempt{attempt}'
                d=OUT/'replays'/purpose/task/level/suffix
                if (d/'validity.json').exists():
                    v=read_json(d/'validity.json');assert v['plan_sha256']==sha256(OUT/'plan.json')
                    for name,h in v['source_sha256'].items():assert sha256(d/name)==h
                    return {**read_json(d/'result.json'),'source':str(d)}
                if d.exists() and any(d.iterdir()):raise RuntimeError(f'Partial attempt retained; no retry: {d}')
                d.mkdir(parents=True,exist_ok=True)
                command=[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_count_shield_buildfix_episode.py'),
                    '--task',task,'--level',level,'--episode',str(episode),'--purpose',purpose,
                    '--out',str(d),'--task-version','v2','--attempt',str(attempt)]
                if level.startswith('clock'):
                    assert sha256(OUT/'clock_fit.json')==state['clock_fit_sha256']
                    command+=['--clock-fit',str(OUT/'clock_fit.json')]
                atomic_json(d/'started.json',{'command':command,'utc':timestamp(),
                    'plan_sha256':sha256(OUT/'plan.json'),'instruction_revision':'10+approved_shield_v2_buildfix','task_definition_revision':9})
                timing=child_run(command,d,gpu)
                try:validity=support.audit_case(d)
                except Exception as exc:
                    atomic_json(d/'validity_failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
                    raise RuntimeError(f'SHIELD validity failed: {d}/validity_failure.json') from exc
                r=read_json(d/'result.json');assert r['count_shield_mode']==('fixed_v2' if task=='count' else 'off')
                assert r['shield_geometry_checks']==(r['steps']+1 if task=='count' else 0)
                seed=(4100000 if purpose.startswith('train_') else 4200000)+16*episode+attempt
                assert (r['task'],r['level'],r['episode'],r['purpose'],r['seed'])==(task,level,episode,purpose,seed)
                atomic_json(d/'validity.json',{**validity,'plan_sha256':sha256(OUT/'plan.json'),'instruction_revision':'10+approved_shield_v2_buildfix'})
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
            # Approved repair: validate the failed original episode before new R0 work.
            publish(state,state='shield_ep112_diagnostic')
            if not (OUT/'ep112_diagnostic.json').exists():
                run_command('ep112_diagnostic',[str(SIM_PYTHON),'-u',str(ECM/'scripts/diagnose_fruit_count_shield_buildfix_ep112.py')],gpu=0)
            diagnostic=read_json(OUT/'ep112_diagnostic.json')
            assert diagnostic['passed'] and diagnostic['physics_collision_shapes_unchanged']
            publish(state,state='shield_clock_calibration')
            cohort('history','calibration')
            if not (OUT/'clock_fit.json').exists():support.fit_calibration()
            publish(state,clock_fit_sha256=sha256(OUT/'clock_fit.json'))
            for level in plan['cohorts']:
                publish(state,state='shield_count_audit',level=level)
                cohort(level);rows=support.collect('audit',level);assert len(rows)==50
                if not support.gate(level,rows):
                    publish(state,state='SHIELD_shortcut_gate_failed',reason=f'{level}: {sum(r["success"] for r in rows)}/50')
                    report(state,support);return
                report(state,support)
            publish(state,state='shield_information_gate')
            info=support.information(support.collect('audit','history'))
            if not info['passed']:
                publish(state,state='SHIELD_information_gate_failed',reason=str(info['criteria']));report(state,support);return
            rows=support.collect();assert len(rows)==350
            other={'passed':all(not r['early_start_steps'] and all(c['passed'] for c in r['appearance_checks'])
                and not any(v[k] for v in r['max_visibility'].values() for k in ('basket','basket_fruit','inactive_reserve'))
                for r in rows),'cases':350,'source':'Existing per-episode validity proofs and runtime checks; no extra rollout.'}
            proof('other_checks.json',other);assert other['passed']
            check_sources();preserve()
            names=('ep112_diagnostic.json','approved_reuse.json','performance_reassessment_r10.json','budget_check.json','information_audit.json','other_checks.json')
            proof('R0_gate.json',{'passed':True,'instruction':'10+approved_shield_v2_buildfix','task_definition_revision':9,'created_utc':timestamp(),
                'proof_sha256':{name:sha256(OUT/name) for name in names},
                'count_main_cases':350,'training':False,'data_only_authorized':True})
            from fruit_r10_data_timing import generate_data_with_timing
            delivery=generate_data_with_timing(state,execute,run_command,check_sources,abort)
            delivery.update(instruction='10+approved_shield_v2_buildfix',observation_condition='count_fixed_shield_v2/occluded_original')
            proof('data_delivery.json',delivery)
            check_sources();preserve()
            publish(state,state='awaiting_data_review',reason='R0 passed; data generation, build, shared normalization and timing reports completed; stop before training.')
            report(state,support)
        except Exception as exc:
            abort();publish(state,state='SHIELD_check_failed',reason=str(exc),traceback=traceback.format_exc())
            report(state,support);raise
        finally:abort()


if __name__=='__main__':main()
