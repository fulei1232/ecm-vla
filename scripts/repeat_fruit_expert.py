#!/usr/bin/env python3
"""One fixed-seed nearest-expert repetition, without layout retry or replacement."""
import argparse
from pathlib import Path
import sys
import traceback

from v_a_common import ECM, atomic_json, timestamp

sys.path.insert(0, str(ECM))
from fruit_picking.generate import run_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['count','occluded_ripe'], required=True)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    for sub in ('hdf5_files','gt'): (a.out/sub).mkdir(parents=True, exist_ok=True)
    row = {'task':a.task,'episode':a.episode,'seed':a.seed,'started_utc':timestamp()}
    try:
        result = run_episode(a.task,a.episode,a.seed,a.out,'rm75b_eg2',ik_mode='nearest',ik_seed=0,ik_candidates=64)
        row.update({k:str(v.relative_to(a.out)) if isinstance(v,Path) else v for k,v in result.items()})
    except BaseException:
        row.update(ok=False,error=traceback.format_exc())
    row['finished_utc'] = timestamp()
    atomic_json(a.out/'result.json',row)
    print({k:row.get(k) for k in ('task','episode','seed','ok','reason','error')},flush=True)
    return int(bool(row.get('error')))


if __name__=='__main__':
    raise SystemExit(main())
