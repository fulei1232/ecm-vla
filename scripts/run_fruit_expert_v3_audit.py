#!/usr/bin/env python3
"""One-day expert-v3 audit. No training before all five revision-2 core gates pass."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
import fcntl
import os
from pathlib import Path
import subprocess
import threading
import traceback

from v_a_common import ECM,SIM_PYTHON,atomic_json,read_json,sha256,timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();out=a.out.resolve()
    if out!=ECM/'results/fruit_expert_v3_audit_20261102': p.error('use the registered independent v3 output directory')
    if not a.gpus or len(set(a.gpus))!=len(a.gpus) or any(g<0 or g>7 for g in a.gpus): p.error('invalid GPUs')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('v3 audit already running')
    code=[ECM/path for path in ('fruit_picking/canonical_planning.py','fruit_picking/canonical_solver.py',
        'fruit_picking/joint_planning.py','fruit_picking/solver.py','fruit_picking/generate.py','fruit_picking/env.py',
        'fruit_picking/agent.py','fruit_picking/layout.py','scripts/generate_fruit_canonical.py',
        'scripts/audit_fruit_canonical_planning.py','scripts/replay_fruit_d2_physics.py',
        'scripts/analyze_fruit_physics_divergence.py','scripts/report_fruit_expert_v3.py',
        'scripts/fruit_expert_audit_stats.py','scripts/report_fruit_expert_v2.py',
        'scripts/preflight_fruit_canonical.py','scripts/run_fruit_expert_v3_audit.py')]
    plan={'phase':2,'revision':2,'solver_version':'canonical_v3_20261102','tasks':['count','occluded_ripe'],
        'episodes':list(range(100)),'ik_seed':0,'ik_candidates':64,'initial_settle_steps':2,
        'settle_fallback':'If either task or pooled basket_miss/all_attempts >3%, generate a separate settle12 dataset once.',
        'gpus':a.gpus,'time_limit_hours':24,'platform':{'condition':'A0','joint_guard':'off','exec_horizon':16},
        'core_gate_policy':'Only the five starred §10.4 items block phase3; all must pass on full200-demo audit.',
        'source_sha256':{str(path):sha256(path) for path in code},
        'registry_sha256':sha256(ECM/'reports/split_registry_20261102.json'),
        'handoff_sha256':sha256(ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md')}
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=plan: p.error('source/plan changed; refusing mixed versions')
    atomic_json(out/'plan.json',plan)
    # Full source at f992da7 is recoverable even where mode dispatch is extended.
    sealed={}
    for name in ('fruit_picking/solver.py','fruit_picking/generate.py','fruit_picking/joint_planning.py',
                 'scripts/run_fruit_expert_v2_audit.py','scripts/repeat_fruit_expert.py',
                 'scripts/fruit_expert_audit_stats.py','scripts/report_fruit_expert_v2.py'):
        path=out/'preserved_v2_sources'/name;path.parent.mkdir(parents=True,exist_ok=True)
        content=subprocess.run(['git','show','f992da7:'+name],cwd=ECM,capture_output=True,check=True).stdout
        if path.exists(): assert path.read_bytes()==content
        else: path.write_bytes(content)
        sealed[str(path)]=sha256(path)
    old=ECM/'results/fruit_expert_v2_audit_20261102'
    old_files=[old/name for name in ('audit.json','summary.json','report.md','independent_validation.json','legacy_stats.json','new_stats.json')]
    old_files+=list((old/'demos/rm75b_eg2/train').glob('*/ep*_manifest.json'))
    old_files+=list((old/'determinism').glob('*/ep*/repeat*/result.json'))
    preservation={'commit':'f992da7','source_snapshots':sealed,'v2_artifact_sha256':{str(path):sha256(path) for path in old_files}}
    if (out/'v2_preservation.json').exists(): assert preservation==read_json(out/'v2_preservation.json')
    else: atomic_json(out/'v2_preservation.json',preservation)
    prior=read_json(out/'status.json') if (out/'status.json').exists() else {}
    if prior.get('state') in ('phase3_authorized_by_core_pass','awaiting_phase2_v3_review'):
        print({'already_finished':True,'state':prior['state'],'audit':str(out/'audit.json')},flush=True)
        return
    # User's one-day limit includes preparation, rather than resetting on restart.
    started=prior.get('started_utc',timestamp())
    deadline=(datetime.fromisoformat(started)+timedelta(hours=24)).isoformat()
    state={'phase':2,'revision':2,'state':'prepared','started_utc':started,'deadline_utc':deadline,
           'attempts':prior.get('attempts',[]),'updated_utc':timestamp()}
    mutex=threading.Lock();stop=threading.Event()
    def status(**changes):
        with mutex:
            state.update(changes,updated_utc=timestamp());atomic_json(out/'status.json',state)
            root=read_json(ECM/'reports/lowlevel_rescue_status.json')
            root.update(phase=2,instruction_revision=2,state=state['state'],updated_utc=timestamp())
            root['links'].update(D2_v3_status=str(out/'status.json'),D2_v3_plan=str(out/'plan.json'),
                                 D2_v3_report=str(out/'report.md'),D2_v3_audit=str(out/'audit.json'))
            atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
    def execute(label,cmd,gpu,allow_failure=False):
        remaining=(datetime.fromisoformat(deadline)-datetime.now(timezone.utc)).total_seconds()
        if remaining<=0: raise TimeoutError('v3 one-day deadline reached')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json',
            'MPLCONFIGDIR':str(ECM/'cache/matplotlib'),'PYTHONDONTWRITEBYTECODE':'1',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        log=out/f'{label}.log'
        with log.open('a') as stream:
            result=subprocess.run(cmd,cwd=ECM,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
        row={'label':label,'gpu':gpu,'exit_code':result.returncode,'command':cmd,'log':str(log),'finished_utc':timestamp()}
        with mutex:
            state['attempts'].append(row);state.update(updated_utc=timestamp());atomic_json(out/'status.json',state)
        print(row,flush=True)
        if result.returncode and not allow_failure: raise RuntimeError(f'{label} failed: {log}')
        return result.returncode
    def lanes(jobs,worker):
        def lane(i):
            for job in jobs[i::len(a.gpus)]:
                if stop.is_set(): return
                try: worker(job,a.gpus[i])
                except BaseException: stop.set();raise
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            for future in as_completed([pool.submit(lane,i) for i in range(len(a.gpus))]): future.result()
    def generation(job,gpu):
        task,first,last,settle=job
        execute(f'generate_s{settle}_{task}_{first}_{last-1}',[str(SIM_PYTHON),'-u',str(ECM/'scripts/generate_fruit_canonical.py'),
            '--out',str(out/f'demos_settle{settle}'),'--task',task,'--first',str(first),'--stop',str(last),
            '--settle-steps',str(settle),'--deadline-utc',deadline],gpu)
    status(state='prepared' if a.preflight_only else 'expert_smoke')
    if a.preflight_only: print({'prepared':True,'deadline_utc':deadline,'out':str(out)},flush=True);return
    try:
        for i,task in enumerate(plan['tasks']): generation((task,0,1,2),a.gpus[i%len(a.gpus)])
        status(state='generating_canonical_settle2')
        jobs=[(t,e,min(e+5,100),2) for t in plan['tasks'] for e in range(0,100,5)]
        lanes(jobs,generation)
        from fruit_expert_audit_stats import summarize_demonstrations
        from report_fruit_expert_v3 import basket_miss_exceeds_three_percent,failure_breakdown
        stats=summarize_demonstrations(out/'demos_settle2/rm75b_eg2/train')
        atomic_json(out/'settle2_stats.json',stats)
        settle=2
        if basket_miss_exceeds_three_percent(stats):
            settle=12;status(state='mandated_settle12_fallback')
            lanes([(t,e,min(e+5,100),12) for t in plan['tasks'] for e in range(0,100,5)],generation)
        atomic_json(out/'selection.json',{'settle_steps':settle,'data_root':str(out/f'demos_settle{settle}/rm75b_eg2/train'),
            'fallback_triggered':settle==12,'settle2_stats':str(out/'settle2_stats.json'),
            'settle2_failure_breakdown':failure_breakdown(stats),
            'fallback_rule':plan['settle_fallback']})
        status(state='saved_state_planning_audit')
        execute('planning',[str(SIM_PYTHON),'-u',str(ECM/'scripts/audit_fruit_canonical_planning.py'),
            '--data',str(out/f'demos_settle{settle}/rm75b_eg2/train'),'--out',str(out/'planning')],a.gpus[0])
        status(state='fixed_action_physics_replay')
        def replay(job,gpu):
            e,j=job;directory=out/f'physics/ep{e}/repeat{j}'
            if not (directory/'trace.json').exists():
                execute(f'physics_ep{e}_repeat{j}',[str(SIM_PYTHON),'-u',str(ECM/'scripts/replay_fruit_d2_physics.py'),
                    '--episode',str(e),'--repeat',str(j),'--out',str(directory)],gpu)
        lanes([(e,j) for e in (0,1,3) for j in range(3)],replay)
        execute('physics_analysis',[str(SIM_PYTHON),'-u',str(ECM/'scripts/analyze_fruit_physics_divergence.py'),
            '--out',str(out/'physics')],a.gpus[0])
        status(state='summarizing_v3')
        for path,digest in plan['source_sha256'].items(): assert sha256(Path(path))==digest,path
        for path,digest in preservation['v2_artifact_sha256'].items(): assert sha256(Path(path))==digest,path
        assert sha256(ECM/'reports/split_registry_20261102.json')==plan['registry_sha256']
        assert sha256(ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md')==plan['handoff_sha256']
        execute('report',[str(SIM_PYTHON),'-u',str(ECM/'scripts/report_fruit_expert_v3.py'),'--out',str(out)],a.gpus[0])
        audit=read_json(out/'audit.json')
        status(state='phase3_authorized_by_core_pass' if audit['all_core_gates_passed'] else 'awaiting_phase2_v3_review',
               all_core_gates_passed=audit['all_core_gates_passed'],core_gates=audit['core_gates'])
    except BaseException as exc:
        status(state='time_limit_reached' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'blocked',
               error=str(exc),traceback=traceback.format_exc())
        interruption={'phase':2,'revision':2,'state':state['state'],'error':str(exc),
            'started_utc':started,'deadline_utc':deadline,'updated_utc':timestamp(),
            'status':str(out/'status.json'),'all_core_gates_passed':None,
            'note':'Run incomplete; no passing audit or phase3 training is claimed.'}
        atomic_json(out/'interruption_summary.json',interruption)
        (out/'interruption_report.md').write_text('\n'.join([
            '阶段2专家v3：执行中断，尚未完成全部审计。',
            f'状态：{state["state"]}；原因：{exc}。',
            '已有示范、失败尝试、规划输入及日志均保留在独立v3目录。',
            '未宣称核心项通过；没有启动阶段3训练。',
            '下一步依status.json中的实际错误处理；一天限时到点后停止。','',
            f'起始时间：{started}；截止时间：{deadline}。',
            f'机器摘要：{out}/interruption_summary.json；运行日志及命令：{out}/status.json。'])+'\n')
        raise


if __name__=='__main__': main()
