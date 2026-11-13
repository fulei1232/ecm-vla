#!/usr/bin/env python3
"""Stage 2, revision 1: temporary train demos and 10 layouts x 3 fixed-seed repeats.

Use nohup setsid in a GPU-capable terminal. No training or held-out layout use.
The inference platform stays A0. Expert failures/retries are retained; audit
gates use successful attempts/all attempts rather than accepted episodes alone.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from v_a_common import ECM, SIM_PYTHON, atomic_json, read_json, sha256, timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    a = p.parse_args()
    out = a.out.resolve()
    if out != ECM/'results/fruit_expert_v2_audit_20261102': p.error('use registered D2 output directory')
    if not a.gpus or len(set(a.gpus)) != len(a.gpus) or any(g<0 or g>7 for g in a.gpus): p.error('invalid GPUs')
    out.mkdir(parents=True,exist_ok=True)
    lock_file=(out/'run.lock').open('a')
    try: fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('expert audit is already running')
    code = [Path(__file__), ECM/'fruit_picking/joint_planning.py',ECM/'fruit_picking/solver.py',
            ECM/'fruit_picking/generate.py',ECM/'fruit_picking/env.py',ECM/'fruit_picking/agent.py',ECM/'fruit_picking/layout.py',
            ECM/'scripts/repeat_fruit_expert.py',ECM/'scripts/fruit_expert_audit_stats.py',ECM/'scripts/report_fruit_expert_v2.py']
    plan = {'phase':2,'revision':1,'platform':{'joint_guard':'off','exec_horizon':16},'solver_mode':'nearest',
            'ik_seed':0,'ik_candidates':64,'gpus':a.gpus,'episodes':list(range(100)),'tasks':['count','occluded_ripe'],
            'success_gate':'successful attempts/all attempts >=0.95 for each task; failures in layout retry are included',
            'determinism':'first5 episode ids per task, accepted layout seed, 3 independent processes each, no retry',
            'source_sha256':{str(path):sha256(path) for path in code},
            'frozen_registry_sha256':sha256(ECM/'reports/split_registry_20261102.json'),
            'handoff_sha256':sha256(ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md')}
    if (out/'plan.json').exists() and read_json(out/'plan.json') != plan: p.error('audit plan/source differs; refusing mixed versions')
    atomic_json(out/'plan.json',plan)
    prior=read_json(out/'status.json') if (out/'status.json').exists() else {}
    attempts=prior.get('attempts',[])
    mutex=threading.Lock()
    stop=threading.Event()
    state={'phase':2,'state':'starting','started_utc':prior.get('started_utc',timestamp()),'attempts':attempts,'updated_utc':timestamp()}
    def status(**changes):
        with mutex:
            state.update(changes,updated_utc=timestamp());atomic_json(out/'status.json',state)
            root=read_json(ECM/'reports/lowlevel_rescue_status.json')
            root.update(phase=2,state=state['state'],updated_utc=timestamp());atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
    def execute(label,cmd,gpu):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json',
             'MPLCONFIGDIR':str(ECM/'cache/matplotlib'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        log=out/f'{label}.log'
        with log.open('a') as f:
            result=subprocess.run(cmd,cwd=ECM,env=env,stdout=f,stderr=subprocess.STDOUT)
        record={'label':label,'command':cmd,'gpu':gpu,'exit_code':result.returncode,'log':str(log),'finished_utc':timestamp()}
        with mutex:
            attempts.append(record);state.update(updated_utc=timestamp());atomic_json(out/'status.json',state)
        if result.returncode: raise RuntimeError(f'{label} failed; inspect {log}')
        print(record,flush=True)
    data=out/'demos'
    def generation(task,episodes,gpu):
        execute(f'generate_{task}_{episodes[0]}_{episodes[-1]}',
                [str(SIM_PYTHON),'-u','-m','fruit_picking.generate','--task',task,'--split','train',
                 '--episodes',','.join(map(str,episodes)),'--out',str(data),'--robot','rm75b_eg2',
                 '--ik-mode','nearest','--ik-seed','0','--ik-candidates','64'],gpu)
    status(state='expert_smoke')
    try:
        for i,task in enumerate(plan['tasks']):
            generation(task,[0],a.gpus[i%len(a.gpus)])
        if not all((data/'rm75b_eg2/train'/task/'ep0_manifest.json').exists() for task in plan['tasks']):
            status(state='smoke_expert_failure')
        else:
            status(state='generating_audit_demos')
            jobs=[(task,list(range(start,min(start+5,100)))) for task in plan['tasks'] for start in range(0,100,5)]
            def lane(i):
                for task,episodes in jobs[i::len(a.gpus)]:
                    if stop.is_set(): return
                    try: generation(task,episodes,a.gpus[i])
                    except BaseException:
                        stop.set()
                        raise
            with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
                for f in as_completed([pool.submit(lane,i) for i in range(len(a.gpus))]): f.result()
            status(state='determinism_repeats')
            repeats=[(task,e,j) for task in plan['tasks'] for e in range(5) for j in range(3)]
            def repeat_lane(i):
                for task,e,j in repeats[i::len(a.gpus)]:
                    if stop.is_set(): return
                    manifest=data/'rm75b_eg2/train'/task/f'ep{e}_manifest.json'
                    if not manifest.exists(): continue
                    seed=read_json(manifest)['seed'];directory=out/'determinism'/task/f'ep{e}'/f'repeat{j}'
                    if (directory/'result.json').exists(): continue
                    try:
                        execute(f'repeat_{task}_{e}_{j}',[str(SIM_PYTHON),'-u',str(ECM/'scripts/repeat_fruit_expert.py'),
                                '--task',task,'--episode',str(e),'--seed',str(seed),'--out',str(directory)],a.gpus[i])
                    except BaseException:
                        stop.set()
                        raise
            with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
                for f in as_completed([pool.submit(repeat_lane,i) for i in range(len(a.gpus))]): f.result()
        status(state='summarizing')
        for path,digest in plan['source_sha256'].items():
            if sha256(Path(path))!=digest: raise RuntimeError(f'source changed during expert audit: {path}')
        if sha256(ECM/'reports/split_registry_20261102.json')!=plan['frozen_registry_sha256']:
            raise RuntimeError('frozen registry changed')
        execute('report',[str(SIM_PYTHON),'-u',str(ECM/'scripts/report_fruit_expert_v2.py'),'--out',str(out)],a.gpus[0])
        audit=read_json(out/'audit.json')
        status(state='passed_phase2' if audit['all_gates_passed'] else 'awaiting_phase2_review')
    except BaseException as exc:
        status(state='blocked',error=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
