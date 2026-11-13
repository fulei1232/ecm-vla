#!/usr/bin/env python3
"""Prepare or execute R0 only. No training/data-generation continuation exists."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import traceback

from v_a_common import ECM, SIM_PYTHON, atomic_json, read_json, sha256, timestamp
from fruit_task_v2_r0_support import OUT, REPORT, PLAN_DOC, REGISTRY, preserve, audit_information, report


def publish(record, **updates):
    record.update(updates, updated_utc=timestamp());atomic_json(OUT/'status.json', record)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase='R0', instruction_revision=7, state=record['state'], updated_utc=record['updated_utc'],
        decision='Task redesign only; preserve D5/D6; no training or training data generation.',
        next='Stop at R0, or at first audit/semantic failure under §15.6, for user review.')
    root['links'].update(R0_report=str(REPORT), R0_status=str(OUT/'status.json'), R0_registry=str(REGISTRY),
        R0_summary=str(ECM/'reports/fruit_task_v2_R0_20261105/summary.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)


def prepare():
    preserve()
    sources = ['fruit_picking/task_v2_spec.py','fruit_picking/task_v2_env.py',
        'scripts/audit_fruit_task_v2_episode.py','scripts/fruit_task_v2_r0_support.py','scripts/run_fruit_task_v2_r0.py',
        'scripts/preflight_fruit_task_v2.py','tests/fruit_picking/test_task_v2_r0.py',
        'fruit_picking/solver.py','fruit_picking/canonical_solver.py','fruit_picking/canonical_v31_solver.py',
        'fruit_picking/canonical_v31_planning.py','src/memory_management/fruit_rescue_platform.py']
    assert read_json(OUT/'native_preflight.json')['passed']
    plan = {'revision':7, 'stage':'R0', 'spec':__import__('fruit_picking.task_v2_spec',fromlist=['TaskV2Spec']).TaskV2Spec(version='v2').to_dict(),
        'source_sha256':{str(ECM/p):sha256(ECM/p) for p in sources}, 'plan_doc_sha256':sha256(PLAN_DOC),
        'registry_sha256':sha256(REGISTRY), 'guard_sha256':sha256(ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json'),
        'native_preflight_sha256':sha256(OUT/'native_preflight.json'),'logical_screen_sha256':sha256(OUT/'logical_screen.json'),
        'dev_episodes':list(range(200,250)), 'seed_base':4200000, 'seed_stride':16,
        'smokes':[{'task':t,'level':'history','episode':200} for t in ('count','occluded_ripe')],
        'cohort_order':[['count','initial_current'],['count','current'],['count','history'],
            ['occluded_ripe','current'],['occluded_ripe','initial_current'],['occluded_ripe','history']],
        'low_information_max_success':15, 'expert_min_success':45, 'expert_clip_rate_strict_max':.005,
        'gpus':list(range(8)), 'stop_on_first_failed_cohort':True, 'retries':0,
        'train_generation':False,'training':False,'val_generation_or_rollouts':False}
    plan = json.loads(json.dumps(plan))
    if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json')==plan, 'R0 definition changed; do not mix results'
    else:atomic_json(OUT/'plan.json',plan)
    return read_json(OUT/'status.json'), plan


def check_sources(plan):
    for p,h in plan['source_sha256'].items():assert sha256(p)==h,p
    assert sha256(PLAN_DOC)==plan['plan_doc_sha256'] and sha256(REGISTRY)==plan['registry_sha256']
    assert sha256(OUT/'native_preflight.json')==plan['native_preflight_sha256']
    assert sha256(OUT/'logical_screen.json')==plan['logical_screen_sha256']
    assert sha256(ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json')==plan['guard_sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state,plan=prepare()
        if state['state'] in ('awaiting_R0_review','R0_audit_failed','R0_engineering_error'):
            print(state,flush=True);return
        if not args.run:
            publish(state,state='ready_for_R0_gpu_audits');report(state);print({'ready':True},flush=True);return
        children={};mutex=threading.Lock();stop=threading.Event()
        def execute(task,level,e,lane):
            if stop.is_set():return
            check_sources(plan)
            d=OUT/'replays'/task/level/f'ep{e}'
            if (d/'validity.json').exists():
                assert read_json(d/'validity.json')['plan_sha256']==sha256(OUT/'plan.json');return
            if d.exists() and any(d.iterdir()):raise RuntimeError(f'Partial attempt retained; no automatic retry: {d}')
            d.mkdir(parents=True,exist_ok=True)
            command=[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_task_v2_episode.py'),'--task',task,'--level',level,
                '--episode',str(e),'--seed',str(plan['seed_base']+16*e),'--out',str(d),'--task-version','v2']
            atomic_json(d/'started.json',{'command':command,'plan_sha256':sha256(OUT/'plan.json'),'utc':timestamp()})
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(lane),'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json',
                'MPLCONFIGDIR':str(ECM/'cache/matplotlib'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            with (d/'console.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ECM,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with mutex:children[child.pid]=child
                try:code=child.wait(timeout=1800)
                finally:
                    if child.poll() is not None:
                        with mutex:children.pop(child.pid,None)
            if code:raise RuntimeError(f'R0 engineering/semantics error exit{code}: {d}/console.log')
            proof=audit_information(d)
            r=read_json(d/'result.json')
            assert (r['task'],r['level'],r['episode'],r['seed'],r['task_version'])==(task,level,e,plan['seed_base']+16*e,'v2')
            assert r['guard_v2_sha256']==plan['guard_sha256'] and not r['engineering_error']
            if r['outcome']=='wind_basket_anomaly':raise RuntimeError(f'SEMANTICS_MISMATCH: windfall in basket: {d}')
            if task=='count' and r['success']:
                if sum(len(event['fruits']) for event in r['wind_events'])!=r['wind_plan']['count']:
                    raise RuntimeError(f'SEMANTICS_MISMATCH: success without required wind losses: {d}')
            atomic_json(d/'validity.json',{**proof,'plan_sha256':sha256(OUT/'plan.json')})
            print({'task':task,'level':level,'episode':e,'success':r['success'],'valid':True},flush=True)
        try:
            publish(state,state='R0_semantics_and_camera_smokes')
            for i,case in enumerate(plan['smokes']):execute(case['task'],case['level'],case['episode'],i)
            for task,level in plan['cohort_order']:
                publish(state,state='R0_cohort_running',task=task,level=level)
                def lane_work(lane):
                    try:
                        for e in plan['dev_episodes'][lane::8]:execute(task,level,e,lane)
                    except Exception:
                        stop.set()
                        with mutex:
                            for child in children.values():
                                try:os.killpg(child.pid,signal.SIGTERM)
                                except ProcessLookupError:pass
                        raise
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for future in as_completed([pool.submit(lane_work,i) for i in range(8)]):future.result()
                summary=report(state);cohort=summary['cohorts'][task][level]
                assert cohort['episodes']==50
                if not cohort['passed']:
                    publish(state,state='R0_audit_failed',reason=f'{task}/{level}: {cohort}')
                    report(state);return
            check_sources(plan);preserve()
            publish(state,state='awaiting_R0_review',reason='All six cohorts passed; stop before training-data generation.')
            report(state)
        except Exception as exc:
            stop.set();publish(state,state='R0_engineering_error',reason=str(exc),traceback=traceback.format_exc(),
                engineering_errors=state.get('engineering_errors',[])+[str(exc)])
            report(state);raise
        finally:
            with mutex:
                for child in children.values():
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass


if __name__=='__main__':main()
