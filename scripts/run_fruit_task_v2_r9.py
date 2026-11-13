#!/usr/bin/env python3
"""R9: gated count R0, then authorized data generation; never starts training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import traceback
import sys

from v_a_common import ECM, POLICY, POLICY_PYTHON, SIM_PYTHON, OPENPI_DATA_HOME, read_json, atomic_json, sha256, timestamp
from fruit_r9_common import OUT, TRACKED, DOC, GUARD, COHORTS, REPORT_ONLY, preserve, publish, proof
from fruit_r9_r0_support import audit_case, collect, cost_comparison, budget_check, fit_calibration, gate, information, report


def prepare():
    OUT.mkdir(exist_ok=True);TRACKED.mkdir(exist_ok=True)
    preserve();assert read_json(OUT/'cpu_preflight.json')['passed']
    assert read_json(OUT/'R8_information_definition_crosscheck.json')['passed']
    files = list((ECM/'fruit_picking').glob('*canonical*.py'))
    files += [ECM/name for name in (
        'fruit_picking/solver.py','fruit_picking/joint_planning.py','fruit_picking/env.py','fruit_picking/agent.py',
        'fruit_picking/layout.py','fruit_picking/task_v2_env.py','fruit_picking/task_v2_spec.py',
        'fruit_picking/task_v2_audit_r8.py','fruit_picking/task_v2_audit_r9.py','fruit_picking/generate.py',
        'scripts/generate_fruit_dart.py','scripts/fruit_r9_episode_support.py','scripts/audit_fruit_task_v2_r9_episode.py',
        'scripts/fruit_r9_common.py','scripts/fruit_r9_r0_support.py','scripts/fruit_r9_generation.py',
        'scripts/fruit_r9_data.py','scripts/run_fruit_task_v2_r9.py','scripts/preflight_fruit_task_v2_r9.py',
        'scripts/check_fruit_r9_memory_tensors.py','scripts/finetune_event_keyframes.py','scripts/build_fruit_dataset.py',
        'scripts/build_finetune_dataset.py','scripts/serve_fruit_task_v2_r9.py',
        'src/memory_management/event_keyframes.py','src/memory_management/policy_adapter.py',
        'src/memory_management/fruit_v2_distance.py','src/memory_management/fruit_rescue_platform.py',
        'src/memory_management/validation/teacher_oracle.py','src/memory_management/fruit_distance.py',
        'scripts/serve_teacher_policy.py')]
    files += [POLICY/'src/mme_vla_suite'/name for name in (
        'shared/mem_buffer.py','shared/posemb_3d.py','policies/policy.py','training/dataset.py',
        'dataset_builder/build_robomme_dataset.py','dataset_builder/robomme_h5_utils.py')]
    registry = ECM/'reports/split_registry_fruit_task_v2_r9_20261106.json'
    sources = {str(p):sha256(p) for p in sorted(set(files))}
    plan = {'revision':9,'created_from_R8_commit':'15401b6','source_sha256':sources,
        'doc_sha256':sha256(DOC),'registry_sha256':sha256(registry),'guard_sha256':sha256(GUARD),
        'CPU_preflight_sha256':sha256(OUT/'cpu_preflight.json'),
        'information_definition_crosscheck_sha256':sha256(OUT/'R8_information_definition_crosscheck.json'),
        'dev_audit':list(range(200,250)),'dev_diagnostic':list(range(250,260)),
        'audit_seed':'4200000+16*episode; no retries or seed changes',
        'train_seed':'4100000+16*episode+attempt; original 16-attempt data-generation recipe; every failure retained',
        'cohorts':COHORTS,'formal_count_cases':350,'accepted_occluded_R0_not_rerun':True,
        'budget':2800,'stress_N':10,'stress_D':600,'minimum_budget_reserve':150,
        'timing_gate':'matched R8/R9 maximum-work stress cases, same geometry/role/D600; whole process-to-result wall time, no step normalization; each <=2x',
        'early_start_joint_rad':.10,'early_start_TCP_m':.03,
        'memory':{'K0':{'frames':16,'events':0,'per_event':'both'},'K8':{'frames':16,'events':8,'per_event':'both'},
                  'KA':{'frames':16,'events':12,'per_event':'release'}},
        'temporal_index':'order','distance_signal':'any_hanging','gpus':list(range(8)),
        'data_only_after_all_R0_gates':True,'stop_after_data':True,'training':False,'val_use':False}
    if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json') == plan,'Source or definitions changed; no mixed run'
    else:proof('plan.json',plan)
    state = read_json(OUT/'status.json')
    return state,plan,registry


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state,plan,registry=prepare()
        if state['state'].startswith('R9_') or state['state']=='awaiting_data_review':print(state);return
        if not args.run:
            publish(state,state='ready_for_r9_GPU_checks');report(state)
            print({'ready':True,'training':False,'run_requires_GPU_terminal':True});return
        if not Path('/dev/nvidiactl').exists():
            raise RuntimeError('No NVIDIA device in this terminal; run the prepared command in the authorized GPU terminal')
        children={};mutex=threading.Lock();stop=threading.Event()
        def abort():
            stop.set()
            with mutex:
                for child in children.values():
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
        def check_sources():
            for p,h in plan['source_sha256'].items():assert sha256(p)==h,p
            assert sha256(DOC)==plan['doc_sha256'] and sha256(registry)==plan['registry_sha256']
            assert sha256(GUARD)==plan['guard_sha256']
        def child_run(command,d,gpu=None,cpu=False,cwd=ECM):
            if stop.is_set():raise RuntimeError('R9 stop is set; no new work')
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
                child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with mutex:children[child.pid]=child
                try:code=child.wait(timeout=43200)
                finally:
                    if child.poll() is not None:
                        with mutex:children.pop(child.pid,None)
            if code:raise RuntimeError(f'R9 process/check failed exit{code}: {d}/console.log')
        def run_command(label,command,gpu=None,cpu=False,cwd=ECM):
            try:child_run(command,OUT/'commands'/label,gpu,cpu,cwd)
            except Exception:abort();raise
        def execute(task,level,episode,gpu,purpose='audit',attempt=0):
            try:
                if stop.is_set():raise RuntimeError('R9 stopped; do not launch another case')
                suffix=f'ep{episode}' if not purpose.startswith('train_') else f'ep{episode}_attempt{attempt}'
                d=OUT/'replays'/purpose/task/level/suffix
                if (d/'validity.json').exists():
                    validity=read_json(d/'validity.json');assert validity['plan_sha256']==sha256(OUT/'plan.json')
                    for name,h in validity['source_sha256'].items():assert sha256(d/name)==h
                    return {**read_json(d/'result.json'),'source':str(d)}
                if d.exists() and any(d.iterdir()):raise RuntimeError(f'Partial case retained; no automatic retry: {d}')
                d.mkdir(parents=True,exist_ok=True)
                command=[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_task_v2_r9_episode.py'),
                    '--task',task,'--level',level,'--episode',str(episode),'--purpose',purpose,
                    '--out',str(d),'--task-version','v2','--attempt',str(attempt)]
                if level.startswith('clock'):
                    assert sha256(OUT/'clock_fit.json')==state['clock_fit_sha256']
                    command+=['--clock-fit',str(OUT/'clock_fit.json')]
                atomic_json(d/'started.json',{'command':command,'utc':timestamp(),'plan_sha256':sha256(OUT/'plan.json')})
                child_run(command,d,gpu)
                try:validity=audit_case(d)
                except Exception as exc:
                    atomic_json(d/'validity_failure.json',{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
                    raise RuntimeError(f'R9 validity check failed: {d}/validity_failure.json') from exc
                r=read_json(d/'result.json')
                seed=(4100000 if purpose.startswith('train_') else 4200000)+16*episode+attempt
                assert (r['task'],r['level'],r['episode'],r['purpose'],r['seed'])==(task,level,episode,purpose,seed)
                atomic_json(d/'validity.json',{**validity,'plan_sha256':sha256(OUT/'plan.json')})
                cost=cost_comparison(r)
                if cost is not None:
                    atomic_json(d/'cost_comparison.json',cost)
                    if not cost['passed']:raise RuntimeError(f'R9 simulation cost >2x R8: {d}/cost_comparison.json')
                print({k:r[k] for k in ('task','level','purpose','episode','N','success','outcome','steps')},flush=True)
                return {**r,'source':str(d)}
            except Exception:abort();raise
        def cohort(level,episodes,purpose='audit'):
            def lane(gpu):
                for episode in episodes[gpu::8]:execute('count',level,episode,gpu,purpose)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in as_completed([pool.submit(lane,gpu) for gpu in range(8)]):future.result()
        try:
            if not (OUT/'memory_tensor_checks.json').exists():
                publish(state,state='r9_memory_tensor_checks')
                run_command('memory_tensor_checks',[str(POLICY_PYTHON),'-u',str(ECM/'scripts/check_fruit_r9_memory_tensors.py')],gpu=0,cwd=POLICY)
            assert read_json(OUT/'memory_tensor_checks.json')['passed']
            tensor_proof=read_json(OUT/'memory_tensor_checks.json')
            assert tensor_proof['shared_packer_sha256']==sha256(ECM/'src/memory_management/policy_adapter.py')
            assert tensor_proof['train_sampler_sha256']==sha256(ECM/'scripts/finetune_event_keyframes.py')
            from fruit_picking.task_v2_spec import TaskV2Spec,start_delay
            eligible=[e for e in plan['dev_diagnostic'] if start_delay(4200000+16*e,TaskV2Spec())>=16]
            publish(state,state='r9_early_start_negative_checks')
            cohort('nonstop',eligible,'early_start_probe')
            probes=collect('early_start_probe')
            early={'passed':len(probes)==len(eligible)>0 and all(r['outcome']=='early_start' for r in probes),
                   'eligible_episodes':eligible,'attempts':[{'episode':r['episode'],'green_step':r['green_step'],
                       'failed_step':r['steps'],'outcome':r['outcome']} for r in probes]}
            proof('early_start_check.json',early);assert early['passed'],'Early-start check failed'
            publish(state,state='r9_first_max_work_render_and_cost_check')
            execute('count','history',250,0,'budget_expert')
            for purpose,level in [('budget_expert','history'),('budget_nonstop','nonstop')]:
                publish(state,state='r9_budget_stress',purpose=purpose)
                cohort(level,plan['dev_diagnostic'],purpose)
            budget=budget_check()
            if not budget['passed']:
                publish(state,state='R9_budget_failed',reason=str(budget));report(state);return
            publish(state,state='r9_clock_calibration')
            cohort('history',plan['dev_diagnostic'],'calibration')
            if not (OUT/'clock_fit.json').exists():fit_calibration()
            publish(state,clock_fit_sha256=sha256(OUT/'clock_fit.json'))
            for level in COHORTS:
                publish(state,state='r9_count_audit',level=level)
                cohort(level,plan['dev_audit']);rows=collect('audit',level)
                assert len(rows)==50
                if not gate(level,rows):
                    publish(state,state='R9_shortcut_gate_failed',reason=f'{level}: {sum(r["success"] for r in rows)}/50')
                    report(state);return
                if level=='history':
                    availability=information(rows)
                    if not availability['passed']:
                        publish(state,state='R9_information_gate_failed',reason=str(availability['criteria']))
                        report(state);return
                report(state)
            check_sources();preserve()
            unit_names=['cpu_preflight.json','R8_information_definition_crosscheck.json','memory_tensor_checks.json','early_start_check.json','budget_check.json','information_audit.json']
            assert all(read_json(OUT/name)['passed'] for name in unit_names)
            proof('R0_gate.json',{'passed':True,'revision':9,'created_utc':timestamp(),
                'proof_sha256':{name:sha256(OUT/name) for name in unit_names},
                'count_shortcut_cohorts':{level:{'episodes':50,'successes':sum(r['success'] for r in collect('audit',level))}
                                         for level in COHORTS},'data_only_authorized':True,'training':False})
            from fruit_r9_generation import generate_data
            generate_data(state,execute,run_command,check_sources,abort)
            check_sources();preserve()
            publish(state,state='awaiting_data_review',reason='All R0 checks passed; data generated, built and normalized. Stop before any training.')
            report(state)
        except Exception as exc:
            abort();publish(state,state='R9_check_failed',reason=str(exc),traceback=traceback.format_exc())
            report(state);raise
        finally:abort()


if __name__=='__main__':
    sys.path[:0]=[str(ECM),str(ECM/'src')]
    main()
