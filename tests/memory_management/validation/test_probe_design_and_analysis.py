import json

import h5py
import numpy as np
import pytest

from src.memory_management.validation.teacher_probe import decision_boundaries
from src.memory_management.validation.demo_source import load_demo_arrays
from src.memory_management.validation.partition import partition
from src.memory_management.validation.cached_flow_loss import common_random_conditions
from src.memory_management.validation.probe_design import (
    fixed_variants, memory_frames, discounted_gain, greedy_variants, variant,
)
from src.memory_management.validation.probe_analysis import crossfit_best, clustered_ci, bh_significant


def test_decisions_stay_on_execution_clock_when_video_is_short():
    cuts=decision_boundaries(150,35)
    assert len(cuts)==3 and all((c-35)%16==0 and c>=64 for c in cuts)
    assert decision_boundaries(70,35)==[]


def test_future_refresh_only_changes_recent_and_rejects_leakage():
    p=partition(128)
    allowed=[f for f in range(1,p.recent[0]) if f not in p.uniform]
    candidates=[{"frame_id":f} for f in allowed[:6]]
    spec=variant("swap",[(0,3),(1,5)])
    before=memory_frames(128,128,candidates,spec)
    after=memory_frames(128,160,candidates,spec)
    assert set(before)-set(p.recent)==set(after)-set(partition(160).recent)
    assert list(partition(160).recent)==after[-4:]
    assert len(after)==32 and after==sorted(set(after))
    assert len(memory_frames(128,160,candidates,variant("off",kind="hist_off")))==5
    with pytest.raises(ValueError):
        memory_frames(128,160,[{"frame_id":150}],variant("future",[(0,1)]))


def test_random_combinations_are_unique_and_greedy_uses_selecting_half_only():
    candidates=[{"frame_id":i} for i in range(6)]
    specs=fixed_variants(candidates,list(range(27)),seed=7,random_sets={2:12,4:12})
    assert len(specs)==4+6*27+24
    assert len(fixed_variants(candidates,list(range(27)),seed=7,random_sets={2:0,4:0}))==166
    for k in (2,4):
        combos=[s for s in specs if s["k"]==k]
        assert len({tuple(map(tuple,s["pairs"])) for s in combos})==12
        assert all(len({p[0] for p in s["pairs"]})==k and len({p[1] for p in s["pairs"]})==k for s in combos)
    gains=np.random.default_rng(1).normal(size=(len(specs),16))
    a=greedy_variants(specs,gains)
    gains[:,1::2]+=1000*np.random.default_rng(3).normal(size=gains[:,1::2].shape)
    assert greedy_variants(specs,gains)==a


def test_crossfit_does_not_report_selecting_half_maximum():
    gains=np.array([[10.,-10.,10.,-10.],[-10.,10.,-10.,10.]])
    result=crossfit_best(gains)
    assert result["gain"]==-10.
    assert result["selected_on_even"]==0 and result["selected_on_odd"]==1
    base=np.array([[3.,4.],[5.,6.]])
    loss=np.stack([base,base-2])
    np.testing.assert_allclose(discounted_gain(loss,base,[0,2],.7),[[0,0],[2,2]])


def test_crn_differs_across_future_offsets_but_repeats_exactly():
    a=common_random_conditions("x",20,128,16,20,32,offset=0)
    b=common_random_conditions("x",20,128,16,20,32,offset=1)
    assert not np.array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])
    np.testing.assert_array_equal(a[0],common_random_conditions("x",20,128,16,20,32,offset=0)[0])


def test_episode_bootstrap_keeps_boundary_groups_and_bh_controls_family():
    ci=clustered_ci([1.,1.,3.,3.],[20,20,21,21],draws=500)
    assert ci["mean"]==2 and ci["ci95"]==[1.,3.]
    assert clustered_ci([1.,2.],[20,20])["ci95"] is None
    np.testing.assert_array_equal(bh_significant([.001,.02,.9]),[True,True,False])


@pytest.mark.parametrize("video_frames",[0,2])
def test_demo_reader_accepts_tasks_with_and_without_video(tmp_path,video_frames):
    path=tmp_path/"demo.h5"
    with h5py.File(path,"w") as f:
        ep=f.create_group("episode_20")
        for i in range(4):
            g=ep.create_group(f"timestep_{i}")
            obs=g.create_group("obs"); obs["joint_state"]=np.zeros(7); obs["gripper_state"]=np.zeros(2)
            g.create_group("action")["joint_action"]=np.zeros(8)
            info=g.create_group("info"); info["is_subgoal_boundary"]=False; info["is_video_demo"]=i<video_frames
    (tmp_path/"Task").mkdir()
    (tmp_path/"Task/ep20_manifest.json").write_text(json.dumps({"task":"Task","episode":20,"split":"val",
        "h5_path":str(path),"verification":{"action_finite":True,"timesteps":4,"task_goal":"goal"}}))
    assert load_demo_arrays("Task",20,tmp_path)["exec_start_idx"]==video_frames
