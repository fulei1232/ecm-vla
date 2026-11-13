"""CPU-only trajectory and supervision diagnostics for the stage-2 expert audit."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

from v_a_common import sha256

PHASES = ("observe", "grasp", "release", "settle", "done", "other")


def fraction(n, d):
    return {"numerator": int(n), "denominator": int(d), "fraction": float(n/d) if d else None}


def delayed_supervision(states, actions, phases, supervised):
    n = len(states)
    early, late = np.zeros(n), np.zeros(n)
    for k in range(16):
        displacement = np.max(np.abs(actions[np.minimum(np.arange(n)+k, n-1), :7]-states), axis=1)
        if k < 8:
            early = np.maximum(early, displacement)
        else:
            late = np.maximum(late, displacement)
    delayed = (early < .01) & (late >= .05)
    names = np.asarray([p if p in PHASES else "other" for p in phases])
    return {p: fraction(np.sum(delayed & supervised & (names == p)), np.sum(supervised & (names == p)))
            for p in PHASES}, fraction(np.sum(delayed & supervised), np.sum(supervised))


def load_episode(manifest_path):
    man = json.loads(manifest_path.read_text())
    with h5py.File(manifest_path.parent / man["h5"], "r") as f:
        ep = f[f"episode_{man['episode']}"]
        ids = sorted(int(k.split('_')[1]) for k in ep if k.startswith('timestep_'))
        assert ids == list(range(len(ids)))
        states = np.stack([ep[f"timestep_{i}/obs/joint_state"][()] for i in ids]).astype(np.float64)
        actions = np.stack([ep[f"timestep_{i}/action/joint_action"][()] for i in ids]).astype(np.float64)
        supervised = np.asarray([not bool(ep[f"timestep_{i}/info/is_video_demo"][()]) for i in ids])
    gt = [json.loads(line) for line in (manifest_path.parent/man['gt']).read_text().splitlines() if line.strip()]
    assert len(gt) == len(states) == len(actions)
    assert [r['timestep'] for r in gt] == ids
    return man, states, actions, supervised, gt


def summarize_demonstrations(root, first=0, stop=100):
    states_all, offsets, records, grasp, lengths = [], [], [], [], []
    swings = {p: {"q1": [], "q7": []} for p in ("reach", "carry")}
    static = {p: [0, 0] for p in PHASES}
    total_static = [0, 0]
    attempts = {t: {"success": 0, "failed": 0, "first_attempt_success": 0, "missing": []}
                for t in ("count", "occluded_ripe")}
    methods = Counter()
    phase_lengths = {p: [] for p in ("observe", "grasp", "release", "settle", "done")}
    failures = []
    for task in attempts:
        for episode in range(first, stop):
            path = root / task / f'ep{episode}_manifest.json'
            if not path.exists():
                attempts[task]['missing'].append(episode)
                failed_path = root/task/f'ep{episode}_FAILED.json'
                if failed_path.exists():
                    failed = json.loads(failed_path.read_text())['failures']
                    attempts[task]['failed'] += len(failed)
                    failures.append({'manifest':str(failed_path),'failures':failed})
                continue
            man, q, a, supervised, gt = load_episode(path)
            states_all.append(q)
            offsets.append(np.abs(a[:, :7]-q))
            lengths.append(len(q))
            attempts[task]['success'] += 1
            attempts[task]['failed'] += len(man['failures'])
            attempts[task]['first_attempt_success'] += int(not man['failures'])
            if man['failures']:
                failures.append({'manifest':str(path),'failures':man['failures']})
            phases = [r['phase'] for r in gt]
            stage, overall = delayed_supervision(q, a, phases, supervised)
            for phase in PHASES:
                static[phase][0] += stage[phase]['numerator']; static[phase][1] += stage[phase]['denominator']
            total_static[0] += overall['numerator']; total_static[1] += overall['denominator']
            starts = [0]+[i for i in range(1,len(q)) if phases[i]!=phases[i-1]]
            for start, end in zip(starts, starts[1:]+[len(q)]):
                phase = phases[start]
                if phase in swings:
                    swings[phase]['q1'].append(float(np.ptp(q[start:end, 0])))
                    swings[phase]['q7'].append(float(np.ptp(q[start:end, 6])))
                if phase == 'approach': grasp.append(float(q[end-1, 6]))
                if phase in phase_lengths: phase_lengths[phase].append(end-start)
            for trace in man.get('planning_audit', []):
                if trace.get('status') == 'Success': methods[trace.get('method','unknown')] += 1
            records.append({'manifest':str(path),'manifest_sha256':sha256(path),'seed':man['seed'],
                            'task':task,'episode':episode,'steps':len(q),'attempt':man['attempt'],
                            'solver_version':man.get('solver_version','legacy')})
    q = np.concatenate(states_all) if states_all else np.empty((0,7))
    error = np.concatenate(offsets) if offsets else np.empty((0,7))
    levels = [0,.005,.01,.5,.99,.995,1]
    center = float(np.median(grasp)) if grasp else None
    return {'records':records,'episodes':len(records),'state_samples':len(q),'failed_attempt_records':failures,
            'by_task':{t:{**v,'attempt_success_rate':fraction(v['success'],v['success']+v['failed']),
                          'accepted_episode_rate':fraction(v['success'],stop-first),
                          'first_attempt_success_rate':fraction(v['first_attempt_success'],stop-first)} for t,v in attempts.items()},
            'motion':{p:{'segments':len(v['q1']),'q1_swing_p90':float(np.quantile(v['q1'],.9)) if v['q1'] else None,
                         'q7_swing_over_pi':fraction(sum(x>np.pi for x in v['q7']),len(v['q7'])),
                         'q1_swings':v['q1'],'q7_swings':v['q7']} for p,v in swings.items()},
            'approach_end_q7':{'median':center,'count':len(grasp),'values':grasp,
                               'deviation_over_1rad':fraction(sum(abs(x-center)>1 for x in grasp),len(grasp)) if grasp else fraction(0,0)},
            'joint_state_quantiles':{'levels':levels,'values':np.quantile(q,levels,axis=0).tolist() if len(q) else []},
            'action_minus_state_abs_quantiles':{'levels':[.99,.999],
                                               'values':np.quantile(error,[.99,.999],axis=0).tolist() if len(q) else []},
            'q7_state_range':[float(q[:,6].min()),float(q[:,6].max())] if len(q) else None,
            'q7_states_outside_pi':fraction(np.sum(np.abs(q[:,6])>np.pi),len(q)),
            'static_delayed_supervision':{'by_phase':{p:fraction(*v) for p,v in static.items()},'all':fraction(*total_static),
                'definition':'16 recorded absolute actions, last-action padding; displacement relative to current measured joint_state; early max over first8 <0.01rad and late max over last8 >=0.05rad; only is_video_demo=False anchors'},
            'episode_lengths':lengths,'hold_segment_lengths':phase_lengths,'planning_methods':dict(methods)}
