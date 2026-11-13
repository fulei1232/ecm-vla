#!/usr/bin/env python3
"""Revision3 v3.1 audit and authorized automatic continuation into stage3."""
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

from v_a_common import ECM,SIM_PYTHON,atomic_json,read_json,sha256,timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args();out=a.out.resolve()
    if out!=ECM/'results/fruit_expert_v31_audit_20261102': p.error('use the independent v3.1 directory')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('rescue workflow already running')
    code=[ECM/path for path in ('fruit_picking/solver.py','fruit_picking/generate.py','fruit_picking/canonical_solver.py',
        'fruit_picking/canonical_planning.py','fruit_picking/canonical_v31_solver.py','fruit_picking/canonical_v31_planning.py',
        'fruit_picking/rescue_generation.py','scripts/diagnose_fruit_v3_collisions.py','scripts/replan_fruit_v3_failures_v31.py',
        'scripts/generate_fruit_rescue_expert.py','scripts/audit_fruit_v31_planning.py','scripts/report_fruit_expert_v31.py',
        'scripts/run_fruit_v31_rescue.py','scripts/run_fruit_rescue_stage3.py','scripts/fruit_rescue_stage3_support.py',
        'scripts/evaluate_fruit_rescue_episode.py','scripts/audit_fruit_rescue_memory.py',
        'src/memory_management/fruit_rescue_platform.py','scripts/generate_fruit_dart.py','scripts/build_fruit_dataset.py',
        'scripts/build_finetune_dataset.py','scripts/finetune_event_keyframes.py','scripts/run_finetune_kf.py',
        'fruit_picking/env.py','fruit_picking/agent.py','fruit_picking/layout.py','scripts/preflight_fruit_v31.py',
        'scripts/serve_teacher_policy.py','scripts/run_fruit_diagnosis.py','scripts/analyze_fruit_failure_modes.py',
        'src/memory_management/fruit_eval.py','src/memory_management/fruit_distance.py',
        'src/memory_management/event_keyframes.py','src/memory_management/validation/teacher_oracle.py',
        'src/memory_management/validation/acm_copy/diagnostic_logging.py')]
    plan={'phase':2,'revision':3,'solver_version':'canonical_v3.1_20261102','audit_episodes':list(range(100)),
        'tasks':['count','occluded_ripe'],'gpus':list(range(8)),'time_limit_hours':12,
        'started_utc':'2026-11-02T14:39:53+00:00','deadline_utc':'2026-11-03T02:39:53+00:00',
        'source_sha256':{str(path):sha256(path) for path in code},
        'diagnosis_sha256':sha256(out/'v3_failure_diagnosis/summary.json'),
        'handoff_sha256':sha256(ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md'),
        'stage3_protocol_sha256':sha256(ECM/'reports/fruit_lowlevel_rescue_stage3_protocol_20261102.md'),
        'dart_protocol_sha256':sha256(ECM/'reports/fruit_dart_protocol_20261027.md'),
        'registry_sha256':sha256(ECM/'reports/split_registry_20261102.json'),
        'continuation':'select by §11.5 and automatically run stage3; no further expert iteration'}
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=plan: p.error('source/plan changed; refusing mixed versions')
    atomic_json(out/'plan.json',plan)
    saved={}
    for name in ('fruit_picking/solver.py','fruit_picking/generate.py','fruit_picking/canonical_solver.py',
                 'fruit_picking/canonical_planning.py','fruit_picking/joint_planning.py'):
        path=out/'preserved_v3_sources'/name;path.parent.mkdir(parents=True,exist_ok=True)
        original=subprocess.run(['git','show','9d1082f:'+name],cwd=ECM,capture_output=True,check=True).stdout
        if path.exists(): assert path.read_bytes()==original
        else: path.write_bytes(original)
        saved[str(path)]=sha256(path)
    v3=ECM/'results/fruit_expert_v3_audit_20261102'
    protected=[v3/name for name in ('audit.json','summary.json','report.md','independent_validation.json','canonical_stats.json')]
    protected+=list((v3/'demos_settle2/rm75b_eg2/train').glob('*/ep*_manifest.json'))
    protected+=list((v3/'demos_settle2/rm75b_eg2/train').glob('*/attempts/*.json'))
    v2_preservation=read_json(v3/'v2_preservation.json')
    preservation={'v3_source_snapshots':saved,'v3_artifacts':{str(path):sha256(path) for path in protected},
                  'v2_artifacts':v2_preservation['v2_artifact_sha256']}
    if (out/'preservation.json').exists(): assert read_json(out/'preservation.json')==preservation
    else: atomic_json(out/'preservation.json',preservation)
    prior=read_json(out/'status.json') if (out/'status.json').exists() else {}
    state={'phase':2,'revision':3,'state':'prepared_v31','started_utc':plan['started_utc'],
        'deadline_utc':plan['deadline_utc'],'attempts':prior.get('attempts',[])}
    mutex=threading.Lock()
    def status(**changes):
        with mutex:
            state.update(changes,updated_utc=timestamp());atomic_json(out/'status.json',state)
            root=read_json(ECM/'reports/lowlevel_rescue_status.json');root.update(phase=state['phase'],instruction_revision=3,
                state=state['state'],updated_utc=timestamp());root['links'].update(D2_v31_status=str(out/'status.json'),D2_v31_plan=str(out/'plan.json'))
            atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
    def execute(label,command,gpu=None,limited=True):
        remaining=(datetime.fromisoformat(plan['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
        if limited and remaining<=0: raise TimeoutError('v3.1 half-day limit reached')
        env={**os.environ,'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json','MPLCONFIGDIR':str(ECM/'cache/matplotlib'),
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
        if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (out/f'{label}.log').open('a') as log:
            result=subprocess.run(command,cwd=ECM,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining if limited else None)
        row={'label':label,'command':command,'exit_code':result.returncode,'finished_utc':timestamp()}
        with mutex: state['attempts'].append(row);atomic_json(out/'status.json',state)
        print(row,flush=True)
        if result.returncode: raise RuntimeError(f'{label} failed: {out}/{label}.log')
    def generate(job,gpu):
        task,first,stop=job
        execute(f'generate_{task}_{first}_{stop-1}',[str(SIM_PYTHON),'-u',str(ECM/'scripts/generate_fruit_rescue_expert.py'),
            '--out',str(out/'demos'),'--task',task,'--first',str(first),'--stop',str(stop),
            '--ik-mode','canonical_v31','--snapshots','--deadline-utc',plan['deadline_utc']],gpu)
    status()
    if a.preflight_only: print({'prepared':True,'deadline_utc':plan['deadline_utc']},flush=True);return
    if not (out/'selection.json').exists():
        try:
            status(state='v31_smoke')
            for i,task in enumerate(plan['tasks']): generate((task,0,1),i)
            status(state='v31_generating_200')
            jobs=[(task,e,e+5) for task in plan['tasks'] for e in range(0,100,5)]
            def lane(i):
                for job in jobs[i::8]: generate(job,i)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in as_completed([pool.submit(lane,i) for i in range(8)]): future.result()
            status(state='v31_saved_state_audit')
            execute('planning',[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_v31_planning.py'),
                '--data',str(out/'demos/rm75b_eg2/train'),'--out',str(out/'planning')],0)
            for path,digest in plan['source_sha256'].items(): assert sha256(Path(path))==digest,path
            for section in ('v3_artifacts','v2_artifacts'):
                for path,digest in preservation[section].items(): assert sha256(Path(path))==digest,path
            for field,path in (('registry_sha256',ECM/'reports/split_registry_20261102.json'),
                ('handoff_sha256',ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md'),
                ('dart_protocol_sha256',ECM/'reports/fruit_dart_protocol_20261027.md'),
                ('stage3_protocol_sha256',ECM/'reports/fruit_lowlevel_rescue_stage3_protocol_20261102.md')):
                assert sha256(path)==plan[field],path
            status(state='v31_summarizing_and_selecting')
            execute('report',[str(SIM_PYTHON),'-u',str(ECM/'scripts/report_fruit_expert_v31.py'),'--out',str(out)],0)
        except Exception as exc:
            # v3 is explicitly accepted. Preserve this failed/partial audit and
            # use that usable baseline rather than inventing another expert round.
            failure={'error':str(exc),'traceback':traceback.format_exc(),'generated_utc':timestamp()}
            atomic_json(out/'interruption.json',failure)
            atomic_json(out/'selection.json',{'revision':3,'ik_mode':'canonical','solver_version':'canonical_v3_20261102',
                'continue_phase3_without_confirmation':True,'reason':'v3.1 audit incomplete/engineering failure; use accepted usable v3 baseline.',
                'v31_interruption':failure})
            (out/'interruption_report.md').write_text('阶段2 v3.1审计中断；按§11.1/11.5回退已采纳的v3进入阶段3。\n'+str(failure)+'\n')
            selection=read_json(out/'selection.json')
            incomplete={'phase':2,'revision':3,'audit_completed':False,'selected_expert':selection,
                'interruption':failure,'completed_commands':state['attempts'],
                'diagnosis':read_json(out/'v3_failure_diagnosis/summary.json'),
                'v3_failure_replanning':read_json(out/'v3_failure_replanning/summary.json'),
                'D1_revision1_metrics':read_json(ECM/'results/fruit_expert_v2_audit_20261102/d1_new_metrics/summary.json')}
            atomic_json(out/'summary.json',incomplete)
            text='\n'.join(['阶段2 v3.1审计未完成，保留实际中断结果。',
                '按§11.1/11.5使用已采纳且可用的v3进入阶段3，不再迭代专家。',
                f'中断原因：{exc}。','不宣称未完成的v3.1审计通过，全部已有产物和日志保留。',
                '决策：直接进入阶段3，不新增确认点；只有两版均无法使用时停止。','',
                '## 做了什么与证据','',f'24次失败诊断及重规划：{out}/v3_failure_diagnosis、v3_failure_replanning。',
                f'所有已完成命令和错误见{out}/status.json、interruption.json，机器摘要见summary.json。','',
                '## 下一步','','用v3重新生成400条干净/DART数据，observe前30帧作历史，按新协议训练并评测。'])+'\n'
            (out/'report.md').write_text(text)
            tracked=ECM/'reports/lowlevel_rescue_D2_v31_20261102';tracked.mkdir(exist_ok=True)
            atomic_json(tracked/'summary.json',incomplete);(ECM/'reports/lowlevel_rescue_D2_v31_20261102.md').write_text(text)
    selection=read_json(out/'selection.json')
    status(phase=3,state='continuing_stage3',selected_expert=selection)
    try:
        execute('stage3',[sys.executable,'-u',str(ECM/'scripts/run_fruit_rescue_stage3.py'),
                          '--selection',str(out/'selection.json')],limited=False)
    except BaseException as exc:
        status(phase=3,state='stage3_engineering_error',error=str(exc));raise
    status(phase=3,state='awaiting_G1_confirmation')


if __name__=='__main__': main()
