#!/usr/bin/env python3
"""CPU-only decision/execution metrics for HarvestMem result directories."""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path

def load(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def detaches(rs):
    out=[]
    for i in range(1,len(rs)):
        for a,f in enumerate(rs[i].get('fruits',[])):
            if f.get('attached') and not rs[i-1].get('fruits',[])[a].get('attached') and not f.get('wind_detached'):
                out.append((i,a))
    return out
def one(arm,task,p):
    r=load(p); tr=rows(p.parent/'trace.jsonl') if (p.parent/'trace.jsonl').exists() else []
    d=detaches(tr); n=r.get('N'); reason=r.get('reason'); basket=max((x.get('n_in_basket',0) for x in tr),default=0)
    start_fail=task=='count' and reason=='finish_signal' and r.get('n_in_basket',0)==0 and r.get('steps',999)>=0 and r.get('steps',999)<200
    reached=task=='count' and n is not None and basket>=n
    first_target=None; target_total=target_ok=0; in_basket=0; ripe_detach=0; green_basket=False
    dev=load(Path('/media/fulei/fuleivla/ecm_vla/results/harvestmem_v3_S1/replays/dev_expert')/task/f"ep{r.get('episode')}_attempt0/result.json",{})
    so=dev.get('slot_of',[]); ripe=[bool(x.get('ripe')) for x in dev.get('slot_geometry',[])]
    for i,a in d:
        for j in range(i,-1,-1):
            if 'target_selected' in tr[j].get('oracle_events',[]): first_target=tr[j].get('oracle_target_slot'); break
        actual=so[a] if a<len(so) else None
        target_total+=1; target_ok+=int(first_target==actual)
        if task=='occluded_ripe':
            ripe_detach+=int(actual is not None and actual<len(ripe) and ripe[actual])
        for x in tr[i:min(len(tr),i+300)]:
            if a<len(x.get('fruits',[])) and x['fruits'][a].get('in_basket'): in_basket+=1; break
    for x in tr:
        if task=='occluded_ripe' and any(a<len(x.get('fruits',[])) and x['fruits'][a].get('in_basket') and not ripe[so[a]] for a in range(len(x.get('fruits',[]))) if a<len(so)):
            green_basket=True
    decision=False
    if task=='count':
        decision = start_fail or (reached and reason in ('overpick','budget')) or (reason=='finish_signal' and 0<r.get('n_in_basket',0)<(n or 0))
    execution=not decision and (reason in ('robot_drop','budget','tcp_stuck_300') or (not reached and reason=='finish_signal') or (not d and not start_fail))
    return {'arm':arm,'task':task,'episode':r.get('episode'),'outcome':r.get('outcome'),'reason':reason,'N':n,'steps':r.get('steps'),'start_failure':start_fail,'reached_N':reached,'decision_failure':decision,'execution_failure':execution,'target_total':target_total,'target_correct':target_ok,'target_accuracy':target_ok/target_total if target_total else None,'detach_in_basket':in_basket,'detach_in_basket_rate':in_basket/target_total if target_total else None,'ripe_detaches':ripe_detach,'green_fruit_in_basket':green_basket}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); out=[]
    for arm in ('A','D','E'):
      for task in ('count','occluded_ripe'):
       for p in sorted((a.root/arm/task).glob('ep*/result.json')): out.append(one(arm,task,p))
    a.out.write_text(json.dumps({'rows':out},indent=2)+'\n'); print(json.dumps({'rows':len(out)}))
if __name__=='__main__': main()
