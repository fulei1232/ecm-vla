#!/usr/bin/env python3
"""Resumable E4a Count queue for regular and forced-stop arms."""
from __future__ import annotations
from collections import deque
import json, os, subprocess, threading, time, urllib.request
from pathlib import Path

ECM=Path('/media/fulei/fuleivla/ecm_vla'); POLICY=Path('/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning')
POLICY_PY='/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python'; SIM_PY='/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python'
SCRIPT=ECM/'scripts/harvestmem_v3/eval_episode.py'; OUT=ECM/'results/harvestmem_v3_E4a'
CK={'A':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/sg/ckpt/mme_vla_suite/sg/11999','AR':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/sg/ckpt/mme_vla_suite/sg/11999','D':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999','DR':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999','E':'/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999','ER':'/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999'}
MEM={a:'uniform' for a in ('A','AR','D','DR')}; MEM.update(E='none',ER='none')
GPUS={'A':[0],'AR':[1],'D':[2],'DR':[3,4],'E':[5],'ER':[6,7]}; ARMS=tuple(CK); TASK='count'; EPS=range(200,260); LOCK=threading.Lock()
def j(p,d=None): return json.loads(p.read_text()) if p.exists() else d
def summary():
 rows=[]
 for arm in ARMS:
  for p in sorted((OUT/arm/TASK).glob('ep*/result.json')):
   r=j(p,{}); rows.append(r)
 lines=['# HarvestMem v3 E4a Summary','',f'Updated {time.strftime("%Y-%m-%d %H:%M:%S")}','', '| Arm | Done | Success | Rate | Forced stop triggers | Overpick |','|---|---:|---:|---:|---:|---:|']
 for arm in ARMS:
  s=[r for r in rows if r.get('arm')==arm]; lines.append(f"| {arm} | {len(s)} | {sum(r.get('outcome')=='success' for r in s)} | {(sum(r.get('outcome')=='success' for r in s)/len(s)) if s else 0:.1%} | {sum(r.get('stop_trigger_step') is not None for r in s)} | {sum(r.get('reason')=='overpick' for r in s)} |")
 (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n'); (OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
def run(arm,gpu,q):
 sd=OUT/'servers'/f'{arm}_{gpu}'; sd.mkdir(parents=True,exist_ok=True); port=29200+gpu; env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'XLA_PYTHON_CLIENT_PREALLOCATE':'false','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':f'{POLICY}/src:{POLICY}/packages/openpi-client/src:{ECM}/src:{ECM}/scripts'}; log=(sd/'server.log').open('w')
 proc=subprocess.Popen([POLICY_PY,'-u',str(ECM/'scripts/overnight_r12/serve.py'),'--checkpoint',CK[arm],'--port',str(port),'--context',str(sd/'context.json')],cwd=POLICY,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  op=urllib.request.build_opener(urllib.request.ProxyHandler({})); deadline=time.time()+1800
  while time.time()<deadline:
   try:
    with op.open(f'http://127.0.0.1:{port}/healthz',timeout=2) as x:
     if x.read()==b'OK\n': break
   except OSError: time.sleep(3)
  while True:
   with LOCK:
    if not q: return
    ep=q.popleft()
   out=OUT/arm/TASK/f'ep{ep}'; rp=out/'result.json'; out.mkdir(parents=True,exist_ok=True)
   if rp.exists() and j(rp,{}).get('outcome')!='error': continue
   env2={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{ECM}:{ECM}/src:{ECM}/scripts:{ECM}/scripts/harvestmem_v3'}
   cmd=[SIM_PY,'-u',str(SCRIPT),'--port',str(port),'--task',TASK,'--episode',str(ep),'--arm',arm,'--checkpoint',CK[arm],'--memory',MEM[arm],'--context',str(sd/'context.json'),'--out',str(out)]
   if arm in ('AR','DR','ER'): cmd.append('--enforce-stop')
   with (out/'client.log').open('w') as f:
    for _ in range(3):
     subprocess.run(cmd,cwd=ECM,env=env2,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
     if rp.exists() and j(rp,{}).get('outcome')!='error': break
   with LOCK: summary()
 finally: proc.kill(); log.close()
def main():
 OUT.mkdir(parents=True,exist_ok=True); ts=[]
 for arm in ARMS:
  q=deque(EPS); ts += [threading.Thread(target=run,args=(arm,g,q)) for g in GPUS[arm]]
 for t in ts:t.start()
 for t in ts:t.join()
 summary()
if __name__=='__main__':main()
