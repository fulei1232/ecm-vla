#!/usr/bin/env python3
"""Generate independent v3.1 audit or stage3 clean/DART data, retaining failures."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import time

from v_a_common import ECM,atomic_json,read_json,sha256
sys.path.insert(0,str(ECM))
from fruit_picking.generate import MAX_ATTEMPTS,SPLIT_BASE,verify
from fruit_picking.rescue_generation import run_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--task',choices=['count','occluded_ripe'],required=True)
    p.add_argument('--first',type=int,required=True);p.add_argument('--stop',type=int,required=True)
    p.add_argument('--ik-mode',choices=['canonical','canonical_v31'],required=True)
    p.add_argument('--dart',action='store_true');p.add_argument('--history-prefix',action='store_true')
    p.add_argument('--snapshots',action='store_true');p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();assert 0<=a.first<a.stop<=200
    if a.dart: assert 100<=a.first<a.stop<=200
    else: assert a.stop<=100
    root=a.out.resolve()/'rm75b_eg2/train'/a.task
    for folder in ('hdf5_files','gt','attempts','planning_states'): (root/folder).mkdir(parents=True,exist_ok=True)
    expected='canonical_v3.1_20261102' if a.ik_mode=='canonical_v31' else 'canonical_v3_20261102'
    deadline=datetime.fromisoformat(a.deadline_utc)
    for episode in range(a.first,a.stop):
        manifest=root/f'ep{episode}_manifest.json'
        if manifest.exists():
            man=read_json(manifest);assert man['solver_version']==expected and bool(man.get('noise'))==a.dart
            assert man['history_prefix_steps']==(30 if a.history_prefix else 0)
            for key in ('h5','gt'): assert sha256(root/man[key])==man[key+'_sha256']
            continue
        failures=[]
        for attempt in range(MAX_ATTEMPTS):
            if datetime.now(timezone.utc)>=deadline: raise TimeoutError('generation deadline reached')
            seed=SPLIT_BASE['train']+16*episode+attempt;record_path=root/'attempts'/f'ep{episode}_seed{seed}.json'
            if record_path.exists(): row=read_json(record_path)
            else:
                start=time.monotonic()
                res=run_episode(a.task,episode,seed,root,a.ik_mode,a.dart,a.history_prefix,
                    root/'planning_states'/f'ep{episode}_seed{seed}' if a.snapshots else None)
                row={**{key:str(value.relative_to(root)) if isinstance(value,Path) else value for key,value in res.items()},
                    'task':a.task,'episode':episode,'seed':seed,'attempt':attempt,'seconds':time.monotonic()-start}
                for key in ('h5','gt','failed_h5','failed_gt'):
                    if key in row: row[key+'_sha256']=sha256(root/row[key])
                atomic_json(record_path,row)
            if not row['ok']:
                failures.append(row);print({'task':a.task,'episode':episode,'seed':seed,'failure':row['reason']},flush=True);continue
            man={**row,'robot':'rm75b_eg2','split':'train','failures':failures,
                 'verification':verify(root/row['h5'],root/row['gt'],episode)}
            atomic_json(manifest,man)
            print({'task':a.task,'episode':episode,'seed':seed,'steps':row['steps'],'version':expected,'dart':a.dart},flush=True)
            break
        else: atomic_json(root/f'ep{episode}_FAILED.json',{'failures':failures})


if __name__=='__main__': main()
