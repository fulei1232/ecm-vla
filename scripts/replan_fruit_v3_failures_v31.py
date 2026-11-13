#!/usr/bin/env python3
"""One v3.1 planning query for each of v3's24 failed states; report only."""
import argparse
from pathlib import Path
import sys

from v_a_common import ECM,atomic_json,read_json,sha256,timestamp
sys.path.insert(0,str(ECM))
from audit_fruit_canonical_planning import load_snapshot,make_planner
from fruit_picking.canonical_v31_planning import canonical_v31_plan
from fruit_picking.canonical_planning import result_digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    diagnosis=read_json(a.out.parent/'v3_failure_diagnosis/summary.json');cases=[]
    for old in diagnosis['cases']:
        dest=a.out/f"{old['task']}_seed{old['seed']}.json"
        if dest.exists(): cases.append(read_json(dest));continue
        snapshot=Path(old['snapshot']);data,meta=load_snapshot(snapshot);planner=make_planner(meta)
        result,trace=canonical_v31_plan(planner,data['goal_pose'],data['current'],data['reference'],meta['dt'],
            meta['attach'],meta['phase'],data['point_cloud'],meta['ik_seed'],meta['ik_candidates'],old['point_cloud_partitions'])
        row={'task':old['task'],'seed':old['seed'],'phase':meta['phase'],'snapshot':str(snapshot),
            'snapshot_sha256':sha256(snapshot),'success':result is not None,'trace':trace,'result_sha256':result_digest(result)}
        atomic_json(dest,row);cases.append(row)
        print({'task':row['task'],'seed':row['seed'],'success':row['success'],'method':trace.get('method'),
               'waypoint':trace.get('waypoint')},flush=True)
    assert len(cases)==24
    atomic_json(a.out/'summary.json',{'generated_utc':timestamp(),'report_only':True,'successes':sum(r['success'] for r in cases),
        'instances':len(cases),'scope':'Planning from saved failure state; not a rerun of the full physical episode.','cases':cases})


if __name__=='__main__': main()
