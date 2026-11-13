#!/usr/bin/env python3
"""Resumable E4b planner-reach Count/Occluded queue."""
from __future__ import annotations
from collections import deque
import json, os, subprocess, threading, time, urllib.request
from pathlib import Path

ECM=Path('/media/fulei/fuleivla/ecm_vla'); POLICY=Path('/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning')
POLICY_PY='/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python'; SIM_PY='/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python'
SCRIPT=ECM/'scripts/harvestmem_v3/eval_episode.py'; OUT=ECM/'results/harvestmem_v3_E4b'
CK={'DP1R':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999','DPaR':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999','D':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999','DPa':'/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999','EP1R':'/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999','EPaR':'/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999','E':'/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999','EPa':'/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999'}
MEM={a:'uniform' for a in ('DP1R','DPaR','D','DPa')}; MEM.update({a:'none' for a in ('EP1R','EPaR','E','EPa')})
PLAN={'DP1R':'first','DPaR':'all','D':'off','DPa':'all','EP1R':'first','EPaR':'all','E':'off','EPa':'all'}
STOP={a: a.endswith('R') for a in CK}
GPU={'DP1R':[0],'DPaR':[1],'D':[2],'DPa':[3],'EP1R':[4],'EPaR':[5],'E':[6],'EPa':[7]}
TASKS={a:'count' for a in ('DP1R','DPaR','EP1R','EPaR')}; TASKS.update({a:'occluded_ripe' for a in ('D','DPa','E','EPa')})
EPS=range(200,260); LOCK=threading.Lock()

def load(p,d=None): return json.loads(p.read_text()) if p.exists() else d
def summarize():
 rows=[]
 for arm,task in TASKS.items():
  for p in sorted((OUT/arm/task).glob('ep*/result.json')):
   rows.append(load(p,{}))
 lines=['# HarvestMem v3 E4b Summary','',f'Updated {time.strftime("%Y-%m-%d %H:%M:%S")}','', '| Arm | Task | Done | Success | Rate | Planner calls | Failures |','|---|---|---:|---:|---:|---:|---:|']
 for arm,task in TASKS.items():
  s=[r for r in rows if r.get('arm')==arm]; lines.append(f"| {arm} | {task} | {len(s)} | {sum(r.get('outcome')=='success' for r in s)} | {(sum(r.get('outcome')=='success' for r in s)/len(s)) if s else 0:.1%} | {sum(r.get('planner_calls',0) for r in s)} | {sum(len(r.get('planner_failures',[])) for r in s)} |")
 (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n'); (OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
def run_one(arm,gpu,ep,server_dir):
 task=TASKS[arm]; out=OUT/arm/task/f'ep{ep}'; out.mkdir(parents=True,exist_ok=True); rp=out/'result.json'
 if rp.exists() and load(rp,{}).get('outcome')!='error': return
 port=29400+gpu; env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'VK_ICD_FILENAMES':'/etc/vulkan/icd.d/nvidia_icd.json','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':f'{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{ECM}:{ECM}/src:{ECM}/scripts:{ECM}/scripts/harvestmem_v3'}
 cmd=[SIM_PY,'-u',str(SCRIPT),'--port',str(port),'--task',task,'--episode',str(ep),'--arm',arm,'--checkpoint',CK[arm],'--memory',MEM[arm],'--context',str(server_dir/'context.json'),'--out',str(out),'--planner-reach',PLAN[arm]]
 if STOP[arm]: cmd.append('--enforce-stop')
 with (out/'client.log').open('w') as f:
  try: subprocess.run(cmd,cwd=ECM,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
  except subprocess.TimeoutExpired: rp.write_text(json.dumps({'task':task,'episode':ep,'arm':arm,'outcome':'error','error':'timeout'})+'\n')
def lane(arm,gpu,q):
 sd=OUT/'servers'/f'{arm}_{gpu}'; sd.mkdir(parents=True,exist_ok=True); port=29400+gpu; env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'XLA_PYTHON_CLIENT_PREALLOCATE':'false','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':f'{POLICY}/src:{POLICY}/packages/openpi-client/src:{ECM}/src:{ECM}/scripts'}; log=(sd/'server.log').open('w')
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
   for _ in range(3):
    run_one(arm,gpu,ep,sd); r=load(OUT/arm/TASKS[arm]/f'ep{ep}/result.json',{})
    if r.get('outcome')!='error': break
   with LOCK: summarize()
 finally: proc.kill(); log.close()
def main():
 OUT.mkdir(parents=True,exist_ok=True); threads=[]
 for arm,gpus in GPU.items():
  q=deque(EPS)
  for gpu in gpus:
   t=threading.Thread(target=lane,args=(arm,gpu,q)); t.start(); threads.append(t)
 for t in threads: t.join()
 summarize()
if __name__=='__main__': main()
