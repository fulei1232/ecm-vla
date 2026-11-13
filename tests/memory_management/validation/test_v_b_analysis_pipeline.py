"""Exercise raw Parquet -> gate audit -> statistics -> report with synthetic paired losses."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("pyarrow")

from src.memory_management.validation.probe_design import fixed_variants, memory_frames
from src.memory_management.validation.partition import partition

ECM=Path(__file__).resolve().parents[3]


def test_analysis_writes_artifacts_and_rejects_unpaired_randomness(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ECM/"scripts"))
    spec=importlib.util.spec_from_file_location("analyze_v_b_test",ECM/"scripts/analyze_v_b.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg=yaml.safe_load((ECM/"configs/memory_management/validation/v_b_teacher_probe.yaml").read_text())["v_b"]
    candidates=[{"frame_id":1,"saliency":.5,"kind":"random"}]
    variants=fixed_variants(candidates,list(partition(128).history),seed=7,random_sets={})
    plan={"task":"VideoUnmaskSwap","episode_id":20,"cutoff":128,"candidates":candidates,"candidate_shortfall":5}
    protocol={"config":cfg,"plan":[plan],"smoke":True,"excluded_episodes":[],"primary_dtype":"float32",
              "greedy_analysis":"held-out only"}
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    (tmp_path/"labels").mkdir(); (tmp_path/"boundaries").mkdir()
    rows=[]
    for vi,v in enumerate(variants):
        for offset in (0,1):
            for seed in range(16):
                loss=10+offset+seed/100
                if v["kind"]=="hist_off": loss+=2
                if v["k"]: loss-=.1+vi/100+(-1)**seed/1000
                row={**{k:plan[k] for k in ("task","episode_id","cutoff")},
                    "label":v["label"],"kind":v["kind"],"k":v["k"],"offset":offset,"seed_index":seed,
                    "frame":128+16*offset,"memory_frames_json":json.dumps(memory_frames(128,128+16*offset,candidates,v)),
                    "loss":loss,"loss_all_dims":loss,"loss_padded_dims":loss,
                    "noise_sha256":f"noise-{offset}-{seed}","time_sha256":str(seed),"action_sha256":str(offset),
                    **{name:float(vi) for name in module.FEATURES}}
                rows.append(row)
    df=pd.DataFrame(rows)
    key="VideoUnmaskSwap_ep20_F0128"
    raw=tmp_path/"labels"/(key+".parquet")
    df.to_parquet(raw,index=False)
    meta={**plan,"variants":variants,"offsets":[0,1],"exec_start_idx":100,
          "audits":{"identity":True,"repeat":True,"crn":True,"causal":True,"batch_full_rel_error_max":0.},
          "parquet_sha256":hashlib.sha256(raw.read_bytes()).hexdigest()}
    metadata=tmp_path/"boundaries"/(key+".json")
    metadata.write_text(json.dumps(meta))
    result=module.analyze(tmp_path,100)
    assert result["formal"] is False and result["completed_boundaries"]==1
    assert not any(v["P2_passed"] or v["P3_passed"] for v in result["combined"].values())
    for filename in ("REPORT.md","analysis.json","label_summary.csv","boundary_summary.csv","figures/crossfit_gain.png"):
        assert (tmp_path/filename).is_file()
    summary=pd.read_csv(tmp_path/"boundary_summary.csv")
    assert np.isfinite(summary.gain_xfit).all()
    df.loc[df.label=="k1_c0_j0","noise_sha256"]="unpaired"
    df.to_parquet(raw,index=False)
    meta["parquet_sha256"]=hashlib.sha256(raw.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="unequal paired conditions"):
        module.analyze(tmp_path,100)
