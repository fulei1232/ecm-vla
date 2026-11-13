#!/usr/bin/env python3
"""Generate v3 audit demos, preserving every failed attempt and planning state."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from v_a_common import ECM, atomic_json, read_json, sha256
sys.path.insert(0,str(ECM))
from fruit_picking.generate import MAX_ATTEMPTS, SPLIT_BASE, run_episode, verify


def serializable(row, root):
    return {k:str(v.relative_to(root)) if isinstance(v,Path) else v for k,v in row.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--task',choices=['count','occluded_ripe'],required=True)
    p.add_argument('--first',type=int,required=True);p.add_argument('--stop',type=int,required=True)
    p.add_argument('--settle-steps',type=int,choices=[2,12],default=2)
    p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();root=a.out.resolve()/'rm75b_eg2/train'/a.task
    if not 0 <= a.first < a.stop <= 100: p.error('audit uses train0..99 only')
    for sub in ('hdf5_files','gt','attempts','planning_states'): (root/sub).mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(a.deadline_utc)
    for e in range(a.first,a.stop):
        manifest=root/f'ep{e}_manifest.json'
        if manifest.exists():
            man=read_json(manifest)
            assert man['solver_version']=='canonical_v3_20261102' and man['settle_steps']==a.settle_steps
            for key in ('h5','gt'): assert sha256(root/man[key])==man[key+'_sha256']
            continue
        failures=[]
        for attempt in range(MAX_ATTEMPTS):
            if datetime.now(timezone.utc)>=deadline: raise TimeoutError('v3 one-day deadline reached')
            seed=SPLIT_BASE['train']+16*e+attempt
            attempt_path=root/'attempts'/f'ep{e}_seed{seed}.json'
            if attempt_path.exists():
                row=read_json(attempt_path)
            else:
                started=time.monotonic()
                result=run_episode(a.task,e,seed,root,'rm75b_eg2',ik_mode='canonical',ik_seed=0,ik_candidates=64,
                    canonical_options={'settle_steps':a.settle_steps,'preserve_failures':True,
                                       'snapshot_dir':root/'planning_states'/f'ep{e}_seed{seed}'})
                row={**serializable(result,root),'seed':seed,'episode':e,'task':a.task,
                     'attempt':attempt,'seconds':time.monotonic()-started}
                for key in ('h5','gt','failed_h5','failed_gt'):
                    if key in row: row[key+'_sha256']=sha256(root/row[key])
                atomic_json(attempt_path,row)
            if not row['ok']:
                failures.append(row)
                print({'task':a.task,'episode':e,'seed':seed,'failure':row['reason']},flush=True)
                continue
            record={**row,'robot':'rm75b_eg2','split':'train','failures':failures,
                    'verification':verify(root/row['h5'],root/row['gt'],e)}
            atomic_json(manifest,record)
            print({'task':a.task,'episode':e,'seed':seed,'steps':row['steps'],'settle':a.settle_steps},flush=True)
            break
        else:
            atomic_json(root/f'ep{e}_FAILED.json',{'failures':failures})


if __name__=='__main__': main()
