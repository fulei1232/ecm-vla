#!/usr/bin/env python3
"""Resume after a read-only prefix analysis fix; reuse training/dev, run G1 once."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

from v_a_common import ECM,POLICY,SIM,SIM_PYTHON,atomic_json,read_json,sha256,timestamp
from run_fruit_rescue_stage3 import OUT,RUN,DOC

ANALYZER=ECM/'scripts/analyze_fruit_failure_modes.py'
SELECTION=ECM/'results/fruit_expert_v31_audit_20261102/selection.json'


def prepare():
    from run_finetune_kf import verify
    training=verify(RUN);t1=read_json(RUN/'gate_t1_k0.json');assert t1['passed']
    selection=read_json(SELECTION);assert training['selected_expert']==selection
    workflow=read_json(SELECTION.parent/'plan.json')
    old_source=OUT/'physical_prefix_fix/analyzer_before.py'
    assert sha256(old_source)==workflow['source_sha256'][str(ANALYZER)]
    for path,digest in workflow['source_sha256'].items():
        if Path(path)!=ANALYZER: assert sha256(Path(path))==digest,path
    assert sha256(DOC)==workflow['stage3_protocol_sha256']
    assert sha256(ECM/'reports/split_registry_20261102.json')==workflow['registry_sha256']
    original_inputs=read_json(OUT/'physical_prefix_fix/input_sha256.json')
    for path,digest in original_inputs.items(): assert sha256(ECM/path)==digest,path
    profile=read_json(OUT/'expert_profile.json');dev={}
    for condition in ('A0','v2'):
        dest=OUT/f'analysis/dev_{condition}'
        summary=read_json(dest/'summary.json')
        if summary['script_sha256']!=sha256(ANALYZER):
            subprocess.run([sys.executable,str(ANALYZER),'--episodes-root',str(OUT/f'dev/{condition}'),
                '--pattern','ep{e}','--episodes','200-219','--q7-center',str(profile['q7_center']),
                '--out',str(dest)],cwd=ECM,check=True)
            summary=read_json(dest/'summary.json')
        assert summary['all']['episodes']==40 and not summary['missing_expected_episodes']
        assert summary['all']['engineering_error']['numerator']==0
        dev[condition]=summary
    decisions=0
    for e in range(200,220):
        for task in ('FruitCount','FruitOccludedRipe'):
            sessions=set()
            for condition in ('A0','v2'):
                directory=OUT/f'dev/{condition}/{task}/ep{e}'
                result=read_json(directory/'result.json');assert not result['error'] and result['client_exit_code']==0
                sessions.add(result['server_session'])
                evidence=read_json(directory/'memory_audit.json');assert evidence['passed']
                decisions+=evidence['decisions']
            assert len(sessions)==1
    off,on=dev['A0']['all'],dev['v2']['all']
    adopt=bool(profile['expert_clip_validation']['passed'] and
        on['first_target_in_basket']['numerator']>=off['first_target_in_basket']['numerator'] and
        on['wrist_deviation_executed']['numerator']<off['wrist_deviation_executed']['numerator'])
    choice={'guard':'v2' if adopt else 'off','expert_clip_validation':profile['expert_clip_validation'],
        'guard_memory_audits_passed':True,'fixed_before_G1_utc':timestamp(),
        'rule':'expert clip<.5%, first basket no lower, wrist deviation strictly lower; no engineering/memory errors',
        'dev_comparison':{name:{key:s['all'][key] for key in ('success','first_target_in_basket','wrist_deviation_executed','engineering_error')}
                          for name,s in dev.items()},
        'dev_summary_sha256':{name:sha256(OUT/f'analysis/dev_{name}/summary.json') for name in dev},
        'note':'Total-task success is reported but is not an additional guard-adoption criterion.'}
    if (OUT/'guard_choice.json').exists():
        prior=read_json(OUT/'guard_choice.json');assert prior['guard']==choice['guard'];choice=prior
    else: atomic_json(OUT/'guard_choice.json',choice)
    for name,digest in choice['dev_summary_sha256'].items():
        assert sha256(OUT/f'analysis/dev_{name}/summary.json')==digest,name
    fix={'generated_utc':timestamp(),'cause':'Analyzer excluded physical hold frames because they were history-only, while the controls correctly included them.',
        'fix':'Keep explicitly declared30-step physical prefix in physical time; validate all clocks, flags and fixed holds. Legacy video-demo exclusion is unchanged.',
        'old_analyzer_sha256':sha256(old_source),'new_analyzer_sha256':sha256(ANALYZER),
        'original_rollout_files_verified_unchanged':len(original_inputs),'dev_memory_audits_passed':80,
        'dev_audited_policy_decisions':decisions,'training_protocol_verified_unchanged':True,
        'original_failed_analyses':str(OUT/'analysis_before_physical_prefix_fix_20261103'),
        'no_policy_or_simulator_reruns_for_dev':True,'no_retraining':True}
    atomic_json(OUT/'physical_prefix_fix/validation.json',fix)
    sources={str(path):sha256(path) for path in [Path(__file__).resolve(),ANALYZER,
        ECM/'scripts/evaluate_fruit_rescue_episode.py',ECM/'scripts/audit_fruit_rescue_memory.py',
        ECM/'scripts/fruit_rescue_stage3_support.py',ECM/'src/memory_management/fruit_rescue_platform.py']}
    plan={'checkpoint':str(RUN/'ckpt/mme_vla_suite/k0/11999'),
        'checkpoint_manifest_sha256':sha256(RUN/'ckpt/mme_vla_suite/k0/11999/params/manifest.ocdbt'),
        'training_protocol_sha256':sha256(RUN/'protocol.json'),'guard_choice_sha256':sha256(OUT/'guard_choice.json'),
        'guard_stats_sha256':sha256(OUT/'guard_v2.json'),'expert_profile_sha256':sha256(OUT/'expert_profile.json'),
        'serving_protocol_sha256':sha256(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json'),
        'registry_sha256':workflow['registry_sha256'],'source_sha256':sources,'episodes':list(range(220,260)),
        'tasks':['FruitCount','FruitOccludedRipe'],'model_seed':7,'observe_hold_steps':30,'total_budget':1300,
        'postprocessing_only_change':fix,'selected_expert':selection}
    path=OUT/'G1_continuation_plan.json'
    if path.exists():
        old=read_json(path)
        for key in plan:
            if key!='postprocessing_only_change': assert old[key]==plan[key],key
        plan=old
    else: atomic_json(path,plan)
    return selection,dev,choice,plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    lock=(OUT/'run.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('stage3 or G1 already running')
    selection,dev,choice,plan=prepare()
    state=read_json(OUT/'status.json')
    if state.get('state')=='awaiting_G1_confirmation': print('G1 already completed',flush=True);return
    mutex=threading.Lock()
    def status(**changes):
        with mutex:
            state.update(changes,updated_utc=timestamp());atomic_json(OUT/'status.json',state)
            root=read_json(ECM/'reports/lowlevel_rescue_status.json')
            root.update(phase=3,instruction_revision=3,state=state['state'],updated_utc=timestamp(),selected_expert=selection)
            root['decision']='Continue authorized stage3; prefix postprocessing fixed, training/dev reused; G1 still single-use.'
            root['links'].update(D3_guard_choice=str(OUT/'guard_choice.json'),D3_prefix_fix=str(OUT/'physical_prefix_fix/validation.json'),
                D3_G1_continuation_plan=str(OUT/'G1_continuation_plan.json'))
            atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
    status(state='ready_for_G1_gpu_launch',error='',traceback='',guard=choice['guard'])
    if not a.run:
        print({'ready_for_G1':True,'guard':choice['guard'],'training_complete':True,'dev_unchanged':True,
               'G1_started':(OUT/'G1_started.json').exists()},flush=True);return
    from run_fruit_diagnosis import start_server,wait_ready
    from audit_fruit_rescue_memory import audit
    from fruit_rescue_stage3_support import report_stage3
    formal=read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']
    params={**formal,'checkpoints':{'K0':plan['checkpoint']}}
    servers=[]
    try:
        status(state='G1_starting_servers')
        for i in range(4): servers.append(start_server(OUT/'G1_servers',i,i,params,base_port=29321))
        wait_ready(servers)
        started=OUT/'G1_started.json'
        if started.exists(): assert read_json(started)==plan,'G1 was started under another plan'
        else: atomic_json(started,plan)
        status(state='G1_single_evaluation',guard=choice['guard'])
        def episode(task,e,lane):
            directory=OUT/f'G1/{task}/ep{e}';result=directory/'result.json'
            if result.exists() and (directory/'memory_audit.json').exists(): return
            if not result.exists():
                if directory.exists() and any(directory.iterdir()):
                    archive=OUT/'superseded'/timestamp().replace(':','_')/'G1'/task/f'ep{e}'
                    archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(directory),archive)
                directory.mkdir(parents=True,exist_ok=True)
                cfg=servers[lane][1]
                command=[str(SIM_PYTHON),'-u',str(ECM/'scripts/evaluate_fruit_rescue_episode.py'),
                    '--port',str(cfg['port']),'--context',str(OUT/f'G1_servers/server{lane}/context.json'),
                    '--root',str(ECM/'data/fruit_picking_dev/rm75b_eg2/train'),'--task',task,'--episode-id',str(e),
                    '--attempt-dir',str(directory)]
                if choice['guard']=='v2': command+=['--guard-stats',str(OUT/'guard_v2.json')]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(lane),'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json',
                    'MPLCONFIGDIR':str(ECM/'cache/matplotlib'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                    'PYTHONPATH':f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src'}
                atomic_json(directory/'started.json',{'command':command,'plan_sha256':sha256(OUT/'G1_continuation_plan.json'),'utc':timestamp()})
                with (directory/'console.log').open('a') as log:
                    try: code=subprocess.run(command,cwd=ECM,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
                    except subprocess.TimeoutExpired: code=124
                row=read_json(result) if result.exists() else {'task':task,'episode_id':e,'condition':'K0','model_seed':7,
                    'outcome':'error','error':f'client exit {code}; no result'}
                row.update(client_exit_code=code,server_session=f"{state['started_utc']}/G1_server{lane}")
                atomic_json(result,row)
            row=read_json(result)
            try: evidence=audit(directory,formal['oracle'])
            except Exception as exc: evidence={'passed':False,'errors':[str(exc)],'mismatches':1,'decisions':0}
            atomic_json(directory/'memory_audit.json',evidence)
            if not evidence['passed']:
                row['memory_audit_error']=evidence['errors'];row['error']=row.get('error','')+'; memory audit failed'
                atomic_json(result,row)
            with mutex:
                state['G1_recorded_results']=len(list((OUT/'G1').glob('*/ep*/result.json')))
                state['updated_utc']=timestamp();atomic_json(OUT/'status.json',state)
            print({'task':task,'episode':e,'outcome':row['outcome'],'memory_audit':evidence['passed']},flush=True)
        jobs=[(task,e) for e in range(220,260) for task in ('FruitCount','FruitOccludedRipe')]
        def lane(i):
            for task,e in jobs[i::4]: episode(task,e,i)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(lane,i) for i in range(4)]): future.result()
        profile=read_json(OUT/'expert_profile.json')
        subprocess.run([sys.executable,str(ANALYZER),'--episodes-root',str(OUT/'G1'),'--pattern','ep{e}',
            '--episodes','220-259','--q7-center',str(profile['q7_center']),'--out',str(OUT/'analysis/G1')],cwd=ECM)
        g1=read_json(OUT/'analysis/G1/summary.json');assert g1['all']['episodes']==80 and not g1['missing_expected_episodes']
        report_stage3(OUT,RUN,selection,dev,choice,g1)
        summary=read_json(OUT/'summary.json');summary['postprocessing_fix']=read_json(OUT/'physical_prefix_fix/validation.json')
        atomic_json(OUT/'summary.json',summary);atomic_json(ECM/'reports/lowlevel_rescue_D3_20261102/summary.json',summary)
        for path in (OUT/'report.md',ECM/'reports/lowlevel_rescue_D3_20261102.md'):
            with path.open('a') as f: f.write('\n分析器物理前缀对齐修复及原始日志未改动的哈希证据见physical_prefix_fix/validation.json；训练和开发集物理回放均复用原产物。\n')
        status(state='awaiting_G1_confirmation',error='',traceback='')
    except Exception as exc:
        status(state='G1_engineering_error',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        for child,_ in servers: child.terminate()
        for child,_ in servers:
            try: child.wait(timeout=60)
            except subprocess.TimeoutExpired: child.kill();child.wait()


if __name__=='__main__': main()
