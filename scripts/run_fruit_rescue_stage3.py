#!/usr/bin/env python3
"""Selected expert ->400 new demos ->build/norm ->K0/T1 ->dev ->one frozen G1."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from v_a_common import ECM,POLICY,POLICY_PYTHON,SIM_PYTHON,OPENPI_DATA_HOME,CHECKPOINT,atomic_json,read_json,sha256,timestamp
from fruit_rescue_stage3_support import prepare_training,expert_profile,report_stage3

OUT=ECM/'results/fruit_k0v2_gate_20261102'
RUN=ECM/'runs/fruit_k0v2_lowlevel_rescue_20261102'
DATASET=ECM/'data/finetune_kf/fruit_dart_v2_rm75b_train'
CLEAN=ECM/'data/fruit_picking_v2/rm75b_eg2/train'
DART=ECM/'data/fruit_picking_dart_v2/rm75b_eg2/train'
DOC=ECM/'reports/fruit_lowlevel_rescue_stage3_protocol_20261102.md'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True)
    a=p.parse_args();selection=read_json(a.selection);assert selection['continue_phase3_without_confirmation']
    workflow=read_json(a.selection.parent/'plan.json')
    def verify_sources():
        for path,digest in workflow['source_sha256'].items(): assert sha256(Path(path))==digest,path
        assert sha256(ECM/'reports/split_registry_20261102.json')==workflow['registry_sha256']
        assert sha256(DOC)==workflow['stage3_protocol_sha256']
    verify_sources()
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('stage3 already running')
    prior=read_json(OUT/'status.json') if (OUT/'status.json').exists() else {}
    if prior.get('state')=='awaiting_G1_confirmation': print(prior,flush=True);return
    state={'phase':3,'state':'preparing_data','started_utc':prior.get('started_utc',timestamp()),
           'selected_expert':selection,'attempts':prior.get('attempts',[])}
    mutex=threading.Lock()
    def status(**changes):
        with mutex:
            state.update(changes,updated_utc=timestamp());atomic_json(OUT/'status.json',state)
            root=read_json(ECM/'reports/lowlevel_rescue_status.json');root.update(phase=3,instruction_revision=3,
                state=state['state'],updated_utc=timestamp(),selected_expert=selection)
            root['links'].update(D3_status=str(OUT/'status.json'),D3_training=str(RUN),D3_results=str(OUT))
            atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
    def execute(label,command,gpus=None,policy=False,cpu=False,allow_failure=False):
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(ECM/'cache/matplotlib'),
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENPI_DATA_HOME':str(OPENPI_DATA_HOME),
            'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json'}
        if gpus is not None: env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
        if policy:
            env.update(PYTHONPATH=f'{POLICY}/src:{POLICY}/packages/openpi-client/src',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        if cpu: env.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
        with (OUT/f'{label}.log').open('a') as log:
            result=subprocess.run(command,cwd=POLICY if policy else ECM,env=env,stdout=log,stderr=subprocess.STDOUT)
        row={'label':label,'command':command,'exit_code':result.returncode,'finished_utc':timestamp()}
        with mutex:
            state['attempts'].append(row);state.update(updated_utc=timestamp());atomic_json(OUT/'status.json',state)
        print(row,flush=True)
        if result.returncode and not allow_failure: raise RuntimeError(f'{label} failed: {OUT}/{label}.log')
        return result.returncode
    def parallel(jobs,worker,lanes=8):
        def lane(i):
            for job in jobs[i::lanes]: worker(job,i)
        with ThreadPoolExecutor(max_workers=lanes) as pool:
            for future in as_completed([pool.submit(lane,i) for i in range(lanes)]): future.result()
    status()
    try:
        deadline=(datetime.now(timezone.utc)+timedelta(hours=36)).isoformat()
        def generate(job,gpu):
            task,first,stop,dart=job
            base=ECM/('data/fruit_picking_dart_v2' if dart else 'data/fruit_picking_v2')
            command=[str(SIM_PYTHON),'-u',str(ECM/'scripts/generate_fruit_rescue_expert.py'),'--out',str(base),
                '--task',task,'--first',str(first),'--stop',str(stop),'--ik-mode',selection['ik_mode'],
                '--history-prefix','--deadline-utc',deadline]
            if dart: command.append('--dart')
            execute(f'generate_{task}_{first}_{stop-1}',command,[gpu])
        jobs=[(task,first,first+5,dart) for task in ('count','occluded_ripe') for dart in (False,True)
              for first in range(100 if dart else 0,200 if dart else 100,5)]
        parallel(jobs,generate)
        for root,first,stop in ((CLEAN,0,100),(DART,100,200)):
            for task in ('count','occluded_ripe'):
                for e in range(first,stop):
                    man=read_json(root/task/f'ep{e}_manifest.json')
                    assert man['solver_version']==selection['solver_version'] and man['history_prefix_steps']==30
        status(state='building_training_dataset')
        build=str(ECM/'scripts/build_fruit_dataset.py')
        DATASET.mkdir(parents=True,exist_ok=True)
        if not (DATASET/'meta/episodes.json').exists():
            execute('build_plan',[str(POLICY_PYTHON),'-u',build,'plan','--out',str(DATASET),
                '--part',str(CLEAN),'0','100','--part',str(DART),'100','200'],policy=True,cpu=True)
        rows=read_json(DATASET/'meta/episodes.json')
        assert len(rows)==400 and all(row['exec_start_idx']==30 for row in rows)
        def shard(i,gpu):
            execute(f'build_shard{i}',[str(POLICY_PYTHON),'-u',build,'shard','--out',str(DATASET),
                                      '--index',str(i),'--of','8'],[gpu],policy=True)
        parallel(list(range(8)),shard)
        execute('build_finish',[str(POLICY_PYTHON),'-u',build,'finish','--out',str(DATASET)],policy=True,cpu=True)
        if not (DATASET/'meta/norm_stats.json').exists():
            execute('norm_stats',[str(POLICY_PYTHON),'-u',build,'norm-stats','--out',str(DATASET)],policy=True,cpu=True)
        if not (DATASET/'meta/encoder_check.json').exists():
            execute('encoder_check',[str(POLICY_PYTHON),'-u',build,'check-encoder','--out',str(DATASET),
                                    '--checkpoint',str(CHECKPOINT),'--frames','8'],[0],policy=True)
        verify_sources();prepare_training(RUN,DATASET,selection,DOC)
        profile_path=OUT/'expert_profile.json'
        profile=read_json(profile_path) if profile_path.exists() else expert_profile(CLEAN,DART,profile_path)
        atomic_json(OUT/'guard_v2.json',profile['guard'])
        if not (OUT/'data_failure_coverage.json').exists():
            from report_fruit_expert_v31 import coverage
            atomic_json(OUT/'data_failure_coverage.json',{'clean':coverage(CLEAN),'DART':coverage(DART)})
        status(state='training_K0_12000_steps')
        command=[sys.executable,str(ECM/'scripts/run_finetune_kf.py'),'train','--out',str(RUN),'--arms','k0']
        if execute('train_k0',command,allow_failure=True): execute('train_k0_restart_once',command)
        execute('T1',[sys.executable,str(ECM/'scripts/run_finetune_kf.py'),'gate','--out',str(RUN),'--arms','k0'])
        status(state='dev_200_219')
        from run_fruit_diagnosis import start_server,wait_ready
        from audit_fruit_rescue_memory import audit
        formal=read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']
        params={**formal,'checkpoints':{'K0':str(RUN/'ckpt/mme_vla_suite/k0/11999')}}
        servers=[start_server(OUT/'servers',i,i,params,base_port=29221) for i in range(4)]
        try:
            wait_ready(servers)
            def episode(job,lane):
                label,task,e,guard=job;directory=OUT/label/task/f'ep{e}'
                if (directory/'result.json').exists() and (directory/'memory_audit.json').exists(): return
                directory.mkdir(parents=True,exist_ok=True)
                cfg=servers[lane][1]
                cmd=[str(SIM_PYTHON),'-u',str(ECM/'scripts/evaluate_fruit_rescue_episode.py'),'--port',str(cfg['port']),
                    '--context',str(OUT/f'servers/server{lane}/context.json'),'--root',str(ECM/'data/fruit_picking_dev/rm75b_eg2/train'),
                    '--task',task,'--episode-id',str(e),'--attempt-dir',str(directory)]
                if guard: cmd+=['--guard-stats',str(OUT/'guard_v2.json')]
                # A recorded result is never rerun, including an engineering failure.
                code=execute(f"eval_{label.replace('/','_')}_{task}_{e}",cmd,[lane],allow_failure=True) if not (directory/'result.json').exists() else 0
                row=read_json(directory/'result.json') if (directory/'result.json').exists() else {
                    'task':task,'episode_id':e,'condition':'K0','model_seed':7,'outcome':'error','error':f'client exit {code}; no result'}
                row.update(server_session=f"{state['started_utc']}/server{lane}",client_exit_code=code)
                atomic_json(directory/'result.json',row)
                try: evidence=audit(directory,formal['oracle'])
                except BaseException as exc: evidence={'passed':False,'errors':[str(exc)],'mismatches':1,'decisions':0}
                atomic_json(directory/'memory_audit.json',evidence)
                if not evidence['passed']:
                    row['memory_audit_error']=evidence['errors'];row['error']=row.get('error','')+'; memory audit failed'
                    atomic_json(directory/'result.json',row)
                if label=='dev/A0' and not evidence['passed']: raise RuntimeError(f'dev baseline memory audit failed: {directory}')
            # Conditions of each layout are consecutive in the same policy process.
            guard_ok=profile['expert_clip_validation']['passed']
            layouts=[(task,e) for e in range(200,220) for task in ('FruitCount','FruitOccludedRipe')]
            def dev(job,lane):
                task,e=job;episode(('dev/A0',task,e,False),lane)
                if guard_ok: episode(('dev/v2',task,e,True),lane)
            parallel(layouts,dev,lanes=4)
            def analyze(label,first,last):
                dest=OUT/'analysis'/label.replace('/','_')
                execute('analyze_'+label.replace('/','_'),[sys.executable,str(ECM/'scripts/analyze_fruit_failure_modes.py'),
                    '--episodes-root',str(OUT/label),'--pattern','ep{e}','--episodes',f'{first}-{last}',
                    '--q7-center',str(profile['q7_center']),'--out',str(dest)],allow_failure=True)
                summary=read_json(dest/'summary.json');assert not summary['missing_expected_episodes'];return summary
            dev_results={'A0':analyze('dev/A0',200,219)}
            if guard_ok: dev_results['v2']=analyze('dev/v2',200,219)
            baseline=dev_results['A0']['all'];assert baseline['engineering_error']['numerator']==0
            guarded=dev_results.get('v2',{}).get('all')
            guard_audits_ok=guard_ok and all(read_json(OUT/f'dev/v2/{task}/ep{e}/memory_audit.json')['passed'] for task,e in layouts)
            adopt=bool(guarded and guard_audits_ok and guarded['engineering_error']['numerator']==0 and
                guarded['first_target_in_basket']['numerator']>=baseline['first_target_in_basket']['numerator'] and
                guarded['wrist_deviation_executed']['numerator']<baseline['wrist_deviation_executed']['numerator'])
            choice={'guard':'v2' if adopt else 'off','expert_clip_validation':profile['expert_clip_validation'],
                'guard_memory_audits_passed':bool(guard_audits_ok),'fixed_before_G1_utc':timestamp(),
                'rule':'expert clips<.5%, first basket no lower, wrist deviation strictly lower; no engineering/memory error'}
            if (OUT/'guard_choice.json').exists():
                prior_choice=read_json(OUT/'guard_choice.json');assert prior_choice['guard']==choice['guard'];choice=prior_choice
            else: atomic_json(OUT/'guard_choice.json',choice)
            verify_sources()
            gate_plan={'episodes':list(range(220,260)),'tasks':['FruitCount','FruitOccludedRipe'],'seed':7,
                'checkpoint':params['checkpoints']['K0'],'guard_choice_sha256':sha256(OUT/'guard_choice.json'),
                'checkpoint_manifest_sha256':sha256(Path(params['checkpoints']['K0'])/'params/manifest.ocdbt'),
                'observe_hold_steps':30,'budget':1300,'registry_sha256':sha256(ECM/'reports/split_registry_20261102.json')}
            if (OUT/'G1_started.json').exists(): assert read_json(OUT/'G1_started.json')==gate_plan
            else: atomic_json(OUT/'G1_started.json',gate_plan)
            status(state='G1_single_evaluation',guard=choice['guard'])
            parallel([('G1',task,e,adopt) for e in range(220,260) for task in ('FruitCount','FruitOccludedRipe')],episode,lanes=4)
            g1=analyze('G1',220,259)
            report_stage3(OUT,RUN,selection,dev_results,choice,g1)
            status(state='awaiting_G1_confirmation')
        finally:
            for child,_ in servers: child.terminate()
            for child,_ in servers:
                try: child.wait(timeout=60)
                except subprocess.TimeoutExpired: child.kill();child.wait()
    except BaseException as exc:
        status(state='stage3_engineering_error',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__': main()
