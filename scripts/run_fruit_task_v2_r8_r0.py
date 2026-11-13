#!/usr/bin/env python3
"""Revision8 R0 only. All failures stop; no data generation or training continuation."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback

from v_a_common import ECM,SIM_PYTHON,atomic_json,read_json,sha256,timestamp
from fruit_r8_r0_support import (OUT,TRACKED,REPORT,DOC,REGISTRY,COHORTS,preserve,audit_case,
    collect,gate,fit_calibration,budget_check,report)


def publish(record,**changes):
    record.update(changes,updated_utc=timestamp());atomic_json(OUT/'status.json',record)
    root=read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase='R0',instruction_revision=8,state=record['state'],updated_utc=record['updated_utc'],
        decision='Revision8 only: stop on any gate/check failure; no training or train-data generation.',
        next='Finish fresh R8 R0 and stop for user review, or report first failed check.')
    root['links'].update(R0_r8_report=str(REPORT),R0_r8_status=str(OUT/'status.json'),
        R0_r8_summary=str(TRACKED/'summary.json'),R0_r8_registry=str(REGISTRY))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)


def prepare():
    preserve()
    assert read_json(OUT/'geometry_unit_checks.json')['passed']
    files=[*list((ECM/'fruit_picking').glob('*canonical*.py')),
        ECM/'fruit_picking/solver.py',ECM/'fruit_picking/joint_planning.py',ECM/'fruit_picking/env.py',
        ECM/'fruit_picking/agent.py',ECM/'fruit_picking/layout.py',ECM/'fruit_picking/task_v2_env.py',
        ECM/'fruit_picking/task_v2_spec.py',ECM/'fruit_picking/task_v2_audit_r8.py',
        ECM/'scripts/fruit_r8_episode_support.py',ECM/'scripts/audit_fruit_task_v2_r8_episode.py',
        ECM/'scripts/fruit_r8_r0_support.py',Path(__file__),ECM/'scripts/preflight_fruit_task_v2_r8.py',
        ECM/'scripts/replay_fruit_r8_ep245.py',ECM/'tests/fruit_picking/test_task_v2_r0.py',
        ECM/'src/memory_management/fruit_v2_distance.py',ECM/'src/memory_management/fruit_distance.py',
        ECM/'src/memory_management/event_keyframes.py',ECM/'src/memory_management/fruit_rescue_platform.py']
    plan={'revision':8,'stage':'R0','dev_audit':list(range(200,250)),'calibration_dev':list(range(250,260)),
        'seed_rule':'4200000+16*episode; geometry seeds unchanged; no retries',
        'source_sha256':{str(p.resolve()):sha256(p) for p in sorted(set(files))},
        'plan_doc_sha256':sha256(DOC),'registry_sha256':sha256(REGISTRY),
        'guard_sha256':sha256(ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json'),
        'geometry_unit_sha256':sha256(OUT/'geometry_unit_checks.json'),
        'D5_readonly_audit_sha256':sha256(OUT/'D5_readonly_basket_audit.json'),
        'r7_regression_sources_sha256':sha256(OUT/'r7_regression_sources/manifest.json'),
        'cohorts':[list(c) for c in COHORTS],'gpus':list(range(8)),
        'low_max_success':15,'high_min_success':45,'expert_clip_strict_max':.005,
        'count_budget':1800,'occluded_budget':1300,'image_iou_min':.98,'image_mean_color_max_255':2.,
        'calibration_and_stress_cases':30,'formal_audit_cases':450,'train_data_generation':False,'training':False,
        'val_use':False,'signal_version_selection':'await user review; both diagnostics only',
        'extra_early_start_clock_queue':False,'user_chose_to_record_early_start_risk_without_extra_queue':True}
    if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json')==plan,'R8 sources/definitions changed; refuse mixed replay'
    else:atomic_json(OUT/'plan.json',plan)
    return read_json(OUT/'status.json'),plan


def check_sources(plan):
    for p,h in plan['source_sha256'].items():assert sha256(p)==h,p
    for f,k in [(DOC,'plan_doc_sha256'),(REGISTRY,'registry_sha256'),(OUT/'geometry_unit_checks.json','geometry_unit_sha256'),
            (OUT/'D5_readonly_basket_audit.json','D5_readonly_audit_sha256')]:assert sha256(f)==plan[k]
    assert sha256(ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json')==plan['guard_sha256']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state,plan=prepare()
        terminal=('awaiting_R0_review','R8_check_failed','R8_budget_failed','R8_audit_failed','R8_engineering_error')
        if state['state'] in terminal:print(state,flush=True);return
        if not a.run:
            publish(state,state='ready_for_r8_gpu_checks');report(state);print({'ready':True,'revision':8},flush=True);return
        children={};mutex=threading.Lock();stop=threading.Event()
        def child_run(command,d,gpu):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json',
                'MPLCONFIGDIR':str(ECM/'cache/matplotlib'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            with (d/'console.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ECM,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with mutex:children[child.pid]=child
                try:code=child.wait(timeout=3600)
                finally:
                    if child.poll() is not None:
                        with mutex:children.pop(child.pid,None)
            if code:raise RuntimeError(f'R8 check/process failed exit{code}: {d}/console.log')
        def execute(task,level,e,gpu,purpose='audit'):
            if stop.is_set():return
            check_sources(plan)
            d=OUT/'replays'/purpose/task/level/f'ep{e}'
            if (d/'validity.json').exists():
                proof=read_json(d/'validity.json');assert proof['plan_sha256']==sha256(OUT/'plan.json')
                for name,h in proof['source_sha256'].items():assert sha256(d/name)==h
                return
            if d.exists() and any(d.iterdir()):raise RuntimeError(f'Partial attempt retained; no retry: {d}')
            d.mkdir(parents=True,exist_ok=True)
            cmd=[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_task_v2_r8_episode.py'),
                '--task',task,'--level',level,'--episode',str(e),'--purpose',purpose,'--out',str(d),'--task-version','v2']
            if level in ('clock','clock_onset'):
                assert sha256(OUT/'clock_fit.json')==state['clock_fit_sha256']
                cmd+=['--clock-fit',str(OUT/'clock_fit.json')]
            atomic_json(d/'started.json',{'command':cmd,'utc':timestamp(),'plan_sha256':sha256(OUT/'plan.json')})
            child_run(cmd,d,gpu)
            proof=audit_case(d);r=read_json(d/'result.json')
            assert (r['task'],r['level'],r['episode'],r['purpose'],r['seed'])==(task,level,e,purpose,4200000+16*e)
            assert r['guard_sha256']==plan['guard_sha256']
            atomic_json(d/'validity.json',{**proof,'plan_sha256':sha256(OUT/'plan.json')})
            print({k:r[k] for k in ('task','level','purpose','episode','N','success','outcome','steps')},flush=True)
        def cohort(task,level,episodes,purpose):
            def lane(gpu):
                try:
                    for e in episodes[gpu::8]:execute(task,level,e,gpu,purpose)
                except Exception:
                    stop.set()
                    with mutex:
                        for child in children.values():
                            try:os.killpg(child.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                    raise
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in as_completed([pool.submit(lane,i) for i in range(8)]):future.result()
        try:
            if not (OUT/'ep245_native_replay.json').exists():
                publish(state,state='r8_ep245_native_regression')
                d=OUT/'ep245_native';d.mkdir(exist_ok=True)
                if (d/'console.log').exists():raise RuntimeError('Partial ep245 native replay retained; no retry')
                child_run([str(SIM_PYTHON),'-u',str(ECM/'scripts/replay_fruit_r8_ep245.py')],d,0)
            assert read_json(OUT/'ep245_native_replay.json')['passed']
            publish(state,state='r8_initial_render_and_reserve_checks')
            execute('count','history',250,0,'budget_expert')
            execute('occluded_ripe','history',200,1,'audit')
            for purpose,level in [('budget_expert','history'),('budget_nonstop','nonstop')]:
                publish(state,state='r8_budget_stress',purpose=purpose)
                cohort('count',level,plan['calibration_dev'],purpose)
            budget=budget_check()
            if not budget['passed']:
                publish(state,state='R8_budget_failed',reason=str(budget));report(state);return
            publish(state,state='r8_clock_calibration')
            cohort('count','history',plan['calibration_dev'],'calibration');fit_calibration()
            publish(state,clock_fit_sha256=sha256(OUT/'clock_fit.json'))
            for task,level in COHORTS:
                publish(state,state='r8_audit_cohort',task=task,level=level)
                cohort(task,level,plan['dev_audit'],'audit')
                rows=collect('audit',task,level);assert len(rows)==50
                summary=report(state)
                if not gate(task,level,rows):
                    publish(state,state='R8_audit_failed',reason=f'{task}/{level}: {sum(r["success"] for r in rows)}/50')
                    report(state);return
            check_sources(plan);preserve()
            publish(state,state='awaiting_R0_review',reason='All prescribed R8 checks and cohorts completed; stop before training/data generation.')
            report(state)
        except Exception as exc:
            stop.set()
            with mutex:
                for child in children.values():
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
            publish(state,state='R8_check_failed',reason=str(exc),traceback=traceback.format_exc())
            report(state);raise
        finally:
            with mutex:
                for child in children.values():
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass


if __name__=='__main__':main()
