#!/usr/bin/env python3
"""Analyze completed V-B Parquet shards with the system Python (no policy/JAX import).

Raw losses remain immutable. Recomputes discount sensitivity, split reliability,
cross-fitted gains, episode-clustered CIs, held-out greedy performance, additivity,
event correlations, and episode-grouped heuristic CV. Smoke/partial runs never
receive a formal P2/P3 conclusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

from v_a_common import ECM, atomic_json, sha256, timestamp
sys.path.insert(0,str(ECM/"src"))
from memory_management.validation.probe_design import discounted_gain
from memory_management.validation.partition import partition
from memory_management.validation.probe_analysis import (
    standard_error, split_reliability, crossfit_best, positive_pvalues,
    bh_significant, clustered_ci, additivity_fit,
)

FEATURES = ["gap_change", "frame_distance", "slot_age", "candidate_age", "visual_cosine",
            "history_max_cosine", "saliency", "planner_event", "cross_phase"]


def clean(value):
    if isinstance(value,dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value,np.generic):
        return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def safe_median(values):
    v=[float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def correlation(x,y):
    return float(stats.spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None


def validate_shard(df, meta, cfg):
    identity = ["label","offset","seed_index"]
    if df.duplicated(identity).any() or not np.isfinite(df[["loss","loss_all_dims","loss_padded_dims"]]).all().all():
        raise ValueError("duplicate or non-finite raw teacher labels")
    expected = len(meta["variants"])*len(meta["offsets"])*cfg["loss"]["mc_samples"]
    if len(df)!=expected or set(df.label)!={s["label"] for s in meta["variants"]}:
        raise ValueError("missing label/offset/seed cells")
    sizes=df.groupby("label").size()
    if not (sizes==len(meta["offsets"])*cfg["loss"]["mc_samples"]).all():
        raise ValueError("ragged variant seed/offset coverage")
    for key in ("noise_sha256","time_sha256","action_sha256"):
        if df.groupby(["offset","seed_index"])[key].nunique().max()!=1:
            raise ValueError(f"unequal paired conditions: {key}")
    for row in df[["memory_frames_json","frame","cutoff","kind"]].drop_duplicates().itertuples(index=False):
        frames=json.loads(row.memory_frames_json)
        if frames!=sorted(set(frames)) or min(frames)<0 or max(frames)>row.frame:
            raise ValueError("noncausal or duplicate memory frame")
    base=df[df.label=="base"].sort_values(["offset","seed_index"])
    identity=df[df.label=="identity"].sort_values(["offset","seed_index"])
    if not np.array_equal(base.loss.to_numpy(),identity.loss.to_numpy()):
        raise ValueError("identity gain is not bitwise zero")
    audits=meta["audits"]
    if not (all(audits[k] for k in ("identity","repeat","crn","causal"))
            and audits["batch_full_rel_error_max"]<=cfg["gates"]["cached_vs_full_rel_err_max"]):
        raise ValueError("failed runtime correctness audit")


def heuristic_cv(labels):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if not labels:
        return {}
    df=pd.DataFrame(labels)
    df=df[(df.k==1)&(df.kind=="swap")].dropna(subset=FEATURES)
    result={}
    for task,g in df.groupby("task"):
        if g.episode_id.nunique()<3 or len(g)<30:
            result[task]={"status":"insufficient episodes for CV"}
            continue
        X,y=g[FEATURES].to_numpy(float),g.gain.to_numpy(float)
        cv=GroupKFold(n_splits=min(5,g.episode_id.nunique()))
        denom=float(np.sum((y-y.mean())**2))
        models={"ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.)),
                "gbm":HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=10.,random_state=7)}
        result[task]={}
        for name,model in models.items():
            prediction=cross_val_predict(model,X,y,groups=g.episode_id,cv=cv,n_jobs=1)
            result[task][name+"_r2_cv"]=1-float(np.sum((prediction-y)**2))/denom if denom>0 else None
    return result


def analyze(run: Path, draws: int):
    protocol=json.loads((run/"protocol.json").read_text())
    cfg=protocol["config"]
    gammas=sorted(set([cfg["future"]["discount_primary"],*cfg["future"]["discount_sensitivity"]]))
    primary=cfg["future"]["discount_primary"]
    records, labels, rankings, events, temporal, sensitivity, greedy, sums, strata = [],[],[],[],[],[],[],[],[]
    missing=[]
    for plan in protocol["plan"]:
        task,episode,cutoff=plan["task"],plan["episode_id"],plan["cutoff"]
        key=f"{task}_ep{episode:02d}_F{cutoff:04d}"
        metadata=run/"boundaries"/(key+".json")
        if not metadata.exists():
            missing.append(key)
            continue
        meta=json.loads(metadata.read_text())
        path=run/"labels"/(key+".parquet")
        if sha256(path)!=meta["parquet_sha256"]:
            raise ValueError(f"changed raw shard: {path}")
        df=pd.read_parquet(path)
        validate_shard(df,meta,cfg)
        specs=meta["variants"]
        names=[s["label"] for s in specs]
        offsets=meta["offsets"]
        ns=cfg["loss"]["mc_samples"]
        index=pd.MultiIndex.from_product([names,offsets,range(ns)],names=["label","offset","seed_index"])
        tensor=df.set_index(["label","offset","seed_index"]).loss.reindex(index).to_numpy().reshape(len(names),len(offsets),ns)
        if not np.isfinite(tensor).all():
            raise ValueError("missing offset/seed cell")
        base=tensor[names.index("base")]
        gains=discounted_gain(tensor,base,offsets,primary)
        hist=-gains[names.index("hist_off")]
        scale=float(hist.mean())
        sensitive=bool(scale>2*float(standard_error(hist)) and scale>1e-8)
        ids={s["label"]:i for i,s in enumerate(specs)}
        common={"task":task,"episode_id":episode,"cutoff":cutoff,"history_scale":scale,"memory_sensitive":sensitive}
        meta_features=df.drop_duplicates("label").set_index("label")
        ps_pos,ps_neg=positive_pvalues(gains),positive_pvalues(-gains)
        for i,spec in enumerate(specs):
            if spec["k"]<1:
                continue
            source=meta_features.loc[spec["label"]]
            record={**common,"label":spec["label"],"kind":spec["kind"],"k":spec["k"],
                    "gain":float(gains[i].mean()),"se":float(standard_error(gains[i])),
                    "gain_rel":float(gains[i].mean()/scale) if sensitive else None,
                    "p_positive":float(ps_pos[i]),"p_negative":float(ps_neg[i]),
                    **{name:float(source[name]) for name in FEATURES}}
            if spec["kind"]=="greedy":
                greedy.append({**record,"heldout_gain":float(gains[i,1::2].mean()),
                               "heldout_se":float(standard_error(gains[i,1::2]))})
            else:
                labels.append(record)
        # Every candidate's slot ranking is a separate S1/S3 observation.
        for ci,candidate in enumerate(meta["candidates"]):
            indices=[i for i,s in enumerate(specs) if s["k"]==1 and s["pairs"][0][0]==ci]
            if len(indices)<2:
                continue
            ranking=split_reliability(gains[indices])
            rankings.append({**common,"k":1,"candidate":ci,**ranking})
            cf=crossfit_best(gains[indices])
            events.append({**common,"candidate":ci,"candidate_kind":candidate["kind"],
                           "saliency":candidate["saliency"],"gain_xfit":cf["gain"]})
        for k in (1,2,4):
            fixed=[i for i,s in enumerate(specs) if s["k"]==k and s["kind"]=="swap"]
            if not fixed:
                continue
            cf=crossfit_best(gains[fixed])
            # o=0 chooses independently from o=0 scores on each selecting half.
            o0=base[0][None]-tensor[fixed,0]
            cf0=crossfit_best(o0)
            record={**common,"k":k,"gain_xfit":cf["gain"],"gain_xfit_se":cf["heldout_se"],
                    "gain_xfit_rel":cf["gain"]/scale if sensitive else None,"gain_xfit_o0":cf0["gain"],
                    "greater_than_2se":cf["gain"]>2*cf["heldout_se"],"fixed_variants":len(fixed)}
            records.append(record)
            if k>1:
                rankings.append({**common,"k":k,"candidate":None,**split_reliability(gains[fixed])})
                for i in fixed:
                    single_indices=[ids[f"k1_c{c}_j{j}"] for c,j in specs[i]["pairs"]]
                    sums.append({**common,"k":k,
                        "sum_singles_A":float(gains[single_indices,::2].mean(axis=1).sum()),
                        "combo_B":float(gains[i,1::2].mean()),
                        "sum_singles_B":float(gains[single_indices,1::2].mean(axis=1).sum()),
                        "combo_A":float(gains[i,::2].mean())})
            # Crossfit strata use fixed causal variants; selecting half never sees evaluation gains.
            categories={"all":fixed,"cross_phase":[i for i in fixed if meta_features.loc[names[i],"cross_phase"]>0],
                        "same_phase":[i for i in fixed if meta_features.loc[names[i],"cross_phase"]==0]}
            if k==1:
                for kind in sorted({c["kind"] for c in meta["candidates"]}):
                    categories[kind]=[i for i in fixed if meta["candidates"][specs[i]["pairs"][0][0]]["kind"]==kind]
                for phase in ("demo","exec"):
                    history=list(partition(cutoff).history)
                    categories["slot_"+phase]=[i for i in fixed if (history[specs[i]["pairs"][0][1]]<meta["exec_start_idx"])==(phase=="demo")]
            for name,indices in categories.items():
                if indices:
                    strata.append({**common,"k":k,"stratum":name,**crossfit_best(gains[indices])})
            for oi,offset in enumerate(offsets):
                per=base[oi][None]-tensor[fixed,oi]
                temporal.append({**common,"k":k,"offset":offset,"mean_abs_gain":float(np.mean(np.abs(per.mean(axis=1)))),
                    "reliability":split_reliability(per)["spearman_brown"],
                    "stale_cost":float((base[oi]-tensor[names.index("d0"),oi]).mean())})
            for gamma in gammas:
                g=discounted_gain(tensor[fixed],base,offsets,gamma)
                sensitivity.append({**common,"k":k,"gamma":gamma,**crossfit_best(g)})
    formal=not protocol["smoke"] and not missing
    task_results={}
    for task in cfg["tasks"]:
        task_results[task]={}
        for k in (1,2,4):
            rr=[r for r in records if r["task"]==task and r["k"]==k]
            ll=[r for r in labels if r["task"]==task and r["k"]==k]
            rk=[r for r in rankings if r["task"]==task and r["k"]==k]
            if not rr or not ll:
                task_results[task][k]={"status":"no evaluable fixed interventions"}
                continue
            means=np.array([r["gain"] for r in ll]); ses=np.array([r["se"] for r in ll])
            noise=float(np.mean(ses**2))
            snr=(float(np.var(means,ddof=1))-noise)/noise if noise>0 and len(means)>1 else None
            sb=safe_median([r["spearman_brown"] for r in rk])
            available=[r["top1_agreement"] for r in rk if r["top1_agreement"] is not None]
            top=float(np.mean(available)) if available else None
            fraction=float(np.mean([r["memory_sensitive"] for r in rr]))
            sensitive=[r for r in rr if r["memory_sensitive"]]
            ci=clustered_ci([r["gain_xfit_rel"] for r in sensitive],[r["episode_id"] for r in sensitive],draws=draws)
            ci0=clustered_ci([r["gain_xfit_o0"] for r in sensitive],[r["episode_id"] for r in sensitive],draws=draws)
            positive_fraction=float(np.mean([r["greater_than_2se"] for r in sensitive])) if sensitive else 0.
            add=[r for r in sums if r["task"]==task and r["k"]==k]
            fit=additivity_fit([r["sum_singles_A"] for r in add]+[r["sum_singles_B"] for r in add],
                               [r["combo_B"] for r in add]+[r["combo_A"] for r in add]) if k>1 else None
            thresholds=cfg["reliability_thresholds"]
            # Adaptive greedy sets never enter the fixed-pool ranking reliability calculation.
            # For 12 fixed k>1 sets use 3/12, stricter than the planned 3/14 threshold.
            top_threshold=max(thresholds["top1_agreement"][f"k{k}"],
                              3/float(np.median([r["fixed_variants"] for r in rr]))) if k>1 else thresholds["top1_agreement"]["k1"]
            reliable=(sb is not None and sb>=thresholds["spearman_brown_median"] and snr is not None
                      and snr>=thresholds["snr"] and top is not None and top>=top_threshold
                      and fraction>=thresholds["memory_sensitive_fraction"])
            if k>1:
                lo,hi=thresholds["additivity_slope"]
                reliable=bool(reliable and fit["slope"] is not None and lo<=fit["slope"]<=hi)
            positive=bool(reliable and ci["ci95"] and ci["ci95"][0]>0 and ci0["ci95"] and ci0["ci95"][0]>0
                          and positive_fraction>=cfg["positive_gain_thresholds"]["frac_boundaries_xfit_gt_2se"])
            pp=bh_significant([r["p_positive"] for r in ll],cfg["positive_gain_thresholds"]["bh_q"])
            pn=bh_significant([r["p_negative"] for r in ll],cfg["positive_gain_thresholds"]["bh_q"])
            for row,p,n in zip(ll,pp,pn):
                row["bh_positive"],row["bh_negative"]=bool(p),bool(n)
            relative=[abs(r["gain_rel"]) for r in ll if r["gain_rel"] is not None]
            task_results[task][k]={"boundaries":len(rr),"episodes":len({r["episode_id"] for r in rr}),
                "S1_spearman_brown_median":sb,"S2_snr":snr,"S3_top1_agreement":top,
                "S3_top1_threshold":top_threshold,"S4_memory_sensitive_fraction":fraction,
                "S4_abs_relative_gain_median":safe_median(relative),
                "S4_abs_relative_gain_p90":float(np.quantile(relative,.9)) if relative else None,
                "S5_additivity":fit,"S9_gain_xfit_relative":ci,"S9_gain_xfit_o0":ci0,
                "S9_fraction_gt_2se":positive_fraction,"bh_positive_fraction":float(pp.mean()),
                "bh_negative_fraction":float(pn.mean()),"meets_P2_thresholds":bool(reliable),
                "meets_P3_thresholds":bool(positive),"formal_P2_passed":bool(formal and reliable),
                "formal_P3_passed":bool(formal and positive)}
    event_metrics={}
    for task in cfg["tasks"]:
        es=[e for e in events if e["task"]==task]
        grouped={}
        for event in es:
            key=(event["episode_id"],event["cutoff"])
            kind=event["candidate_kind"]
            if kind=="kemo_peak_fallback": kind="kemo_peak"
            grouped.setdefault(key,{}).setdefault(kind,[]).append(event["gain_xfit"])
        pairs={}
        for a,b in (("kemo_peak","random"),("planner_keyframe","random"),("kemo_peak","planner_keyframe")):
            diffs=np.array([np.mean(g[a])-np.mean(g[b]) for g in grouped.values() if a in g and b in g])
            pairs[a+"_vs_"+b]={"n":len(diffs),"mean_difference":float(diffs.mean()) if len(diffs) else None,
                "wilcoxon_p":float(stats.wilcoxon(diffs).pvalue) if len(diffs)>1 and np.any(diffs!=0) else None}
        event_metrics[task]={"S6_saliency_vs_xfit_gain_spearman":correlation([e["saliency"] for e in es],[e["gain_xfit"] for e in es]),
                             "paired_candidate_comparisons":pairs}
    cv=heuristic_cv(labels)
    for task,results in cv.items():
        bound=task_results.get(task,{}).get(1,{}).get("S1_spearman_brown_median")
        r2=results.get("gbm_r2_cv")
        results["gbm_r2_over_S1"]=r2/bound if r2 is not None and bound is not None and bound>0 else None
    combined={}
    for k in (1,2,4):
        p2=[task for task in task_results if task_results[task][k].get("formal_P2_passed")]
        p3=[task for task in task_results if task_results[task][k].get("formal_P3_passed")]
        combined[k]={"reliable_tasks":p2,"positive_gain_tasks":p3,
                     "P2_passed":len(p2)>=cfg["reliability_thresholds"]["min_tasks"],
                     "P3_passed":len(p3)>=cfg["positive_gain_thresholds"]["min_tasks"]}
    output={"schema":"v-b-analysis-v1","created":timestamp(),"protocol_sha256":sha256(run/"protocol.json"),
            "formal":formal,"smoke":protocol["smoke"],"completed_boundaries":len(protocol["plan"])-len(missing),
            "planned_boundaries":len(protocol["plan"]),"missing_boundaries":missing,"excluded_episodes":protocol["excluded_episodes"],
            "candidate_shortfall_boundaries":sum(p["candidate_shortfall"]>0 for p in protocol["plan"]),
            "compute_dtype":protocol["primary_dtype"],"tasks":task_results,"combined":combined,
            "S6_events":event_metrics,"S7_heuristic_cv":cv,
            "limitations":["FP32 teacher only; BF16 cache equivalence failed.",
                protocol["greedy_analysis"],"Smoke and incomplete runs never receive formal P2/P3 conclusions.",
                "No value predictor or full manager is authorized by V-B alone; V-A remains necessary."]}
    for name,rows in (("label_summary",labels),("boundary_summary",records),("ranking_reliability",rankings),
                      ("candidate_events",events),("temporal_profile",temporal),("gamma_sensitivity",sensitivity),
                      ("greedy_heldout",greedy),("additivity",sums),("crossfit_strata",strata)):
        pd.DataFrame(clean(rows)).to_csv(run/(name+".csv"),index=False)
    atomic_json(run/"analysis.json",clean(output))
    figures=run/"figures"
    figures.mkdir(exist_ok=True)
    if records:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        table=pd.DataFrame(records)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for task,g in table[table.k==1].groupby("task"):
            axes[0].hist(g.gain_xfit,bins=15,alpha=.5,label=task)
            sensitive=g[g.memory_sensitive]
            if len(sensitive): axes[1].hist(sensitive.gain_xfit_rel,bins=15,alpha=.5,label=task)
        axes[0].set_xlabel("Cross-fitted gain (k=1)")
        axes[1].set_xlabel("Cross-fitted gain / history value")
        for ax in axes:
            ax.axvline(0,color="black",linewidth=1)
            handles,_=ax.get_legend_handles_labels()
            if handles: ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figures/"crossfit_gain.png",dpi=160); plt.close(fig)
    report=["# V-B teacher probe", "",f"Status: {'formal analysis' if formal else 'smoke / partial; no formal conclusion'}.",
            f"Completed boundaries: {output['completed_boundaries']}/{output['planned_boundaries']}. Teacher: FP32.","",
            "| Task | k | S1 | SNR | Top-1 | Sensitive | P2 | P3 |","|---|---:|---:|---:|---:|---:|---|---|"]
    for task,result in task_results.items():
        for k,r in result.items():
            if "boundaries" not in r: continue
            fmt=lambda x: f"{x:.3g}" if x is not None else "NA"
            report.append(f"| {task} | {k} | {fmt(r['S1_spearman_brown_median'])} | {fmt(r['S2_snr'])} | "
                          f"{fmt(r['S3_top1_agreement'])} | {fmt(r['S4_memory_sensitive_fraction'])} | "
                          f"{r['formal_P2_passed']} | {r['formal_P3_passed']} |")
    report += ["", "## Limits", "", *["- "+s for s in output["limitations"]],"",
               "Per-offset/seed losses remain in labels/*.parquet. Re-run this script to recompute all discount sensitivities.",
               "If every k fails reliability, the plan requires the registered 20-boundary S=64 repeat before rejecting the approach."]
    (run/"REPORT.md").write_text("\n".join(report)+"\n")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--bootstrap-draws",type=int,default=2000)
    args=p.parse_args()
    if args.bootstrap_draws<1: p.error("bootstrap draws must be positive")
    result=analyze(args.run.resolve(),args.bootstrap_draws)
    print(json.dumps(clean({k:result[k] for k in ("formal","completed_boundaries","planned_boundaries","combined")}),indent=2))


if __name__=="__main__":
    main()
