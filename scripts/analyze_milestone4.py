#!/usr/bin/env python3
"""Summarize a completed train-split teacher batch for Milestone 4.

The immutable V-B parquet shards remain the source of truth.  ``analyze_v_b``
computes the registered split-half, gain-sign and cross-fitted statistics; this
wrapper adds batch-level coverage, positive/negative gain rates and a small set
of human-review cases with their candidate/slot metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_mean(values):
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return float(x.mean()) if len(x) else None


def summarize(run: Path, top_k: int = 5) -> dict:
    protocol = json.loads((run / "protocol.json").read_text())
    analysis_path = run / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError("run analyze_v_b.py first; analysis.json is missing")
    analysis = json.loads(analysis_path.read_text())
    if protocol["config"].get("demo_source") != "train_h5":
        raise ValueError("Milestone 4 requires a sealed train_h5 protocol")
    boundary = pd.read_csv(run / "boundary_summary.csv")
    labels = pd.read_csv(run / "label_summary.csv")
    events = pd.read_csv(run / "candidate_events.csv")
    if boundary.empty or labels.empty:
        raise ValueError("completed run has no teacher gain rows")
    k1 = boundary[boundary["k"] == 1].copy()
    if k1.empty:
        raise ValueError("completed run has no k=1 fixed intervention summaries")
    gain_col = "gain_xfit"
    positive = k1[k1[gain_col] > 0]
    negative = k1[k1[gain_col] < 0]
    sensitive = k1[k1["memory_sensitive"] == True]  # noqa: E712
    case_rows = k1.sort_values(gain_col, ascending=False).head(top_k).copy()
    case_rows = pd.concat([case_rows, k1.sort_values(gain_col).head(top_k)], ignore_index=True)
    cases = []
    for row in case_rows.drop_duplicates(subset=["task", "episode_id", "cutoff", "k"]).to_dict("records"):
        matches = events[(events.task == row["task"]) &
                         (events.episode_id == row["episode_id"]) &
                         (events.cutoff == row["cutoff"])]
        best = matches.iloc[(matches["gain_xfit"] - row[gain_col]).abs().argsort()[:1]] if len(matches) else None
        candidate = best.iloc[0].to_dict() if best is not None and len(best) else {}
        cases.append({"task": row["task"], "episode_id": int(row["episode_id"]),
                      "cutoff": int(row["cutoff"]), "k": int(row["k"]),
                      "crossfit_gain": float(row[gain_col]),
                      "crossfit_gain_se": float(row["gain_xfit_se"]),
                      "memory_sensitive": bool(row["memory_sensitive"]),
                      "candidate": candidate,
                      "interpretation": "positive teacher gain candidate" if row[gain_col] > 0
                      else "negative teacher gain candidate"})
    task_summary = {}
    for task, group in k1.groupby("task"):
        task_summary[str(task)] = {
            "boundaries": int(len(group)),
            "episodes": int(group.episode_id.nunique()),
            "positive_fraction": float((group[gain_col] > 0).mean()),
            "negative_fraction": float((group[gain_col] < 0).mean()),
            "positive_gt_2se_fraction": float(group["greater_than_2se"].mean()),
            "mean_crossfit_gain": _finite_mean(group[gain_col]),
            "median_crossfit_gain": float(group[gain_col].median()),
            "split_half_reliability_median": analysis["tasks"].get(str(task), {}).get("1", {}).get("S1_spearman_brown_median"),
            "snr": analysis["tasks"].get(str(task), {}).get("1", {}).get("S2_snr"),
        }
    output = {
        "schema": "milestone4-train-batch-diagnostic-v1",
        "formal": bool(analysis.get("formal") and not analysis.get("missing_boundaries")),
        "source_run": str(run.resolve()),
        "source_split": protocol["config"].get("demo_source"),
        "planned_boundaries": int(analysis["planned_boundaries"]),
        "completed_boundaries": int(analysis["completed_boundaries"]),
        "episodes": int(boundary.episode_id.nunique()),
        "tasks": sorted(boundary.task.unique().tolist()),
        "k1_interventions": int(len(labels[labels["k"] == 1])),
        "gain_distribution": {
            "positive_fraction": float((k1[gain_col] > 0).mean()),
            "negative_fraction": float((k1[gain_col] < 0).mean()),
            "zero_fraction": float((k1[gain_col] == 0).mean()),
            "memory_sensitive_fraction": float(sensitive.shape[0] / len(k1)),
            "mean": _finite_mean(k1[gain_col]),
            "median": float(k1[gain_col].median()),
        },
        "split_half": {task: values["split_half_reliability_median"] for task, values in task_summary.items()},
        "task_summary": task_summary,
        "interpretable_cases": cases,
        "analysis_combined": analysis.get("combined", {}),
        "limits": ["A k=1 gain is a frozen-teacher action-loss signal, not closed-loop success.",
                   "Case selection is descriptive and must not be used to tune the predictor.",
                   "Train labels must remain separate from val/test evaluation."],
    }
    (run / "milestone4_diagnostic.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    lines = ["# Milestone 4 train teacher diagnostics", "",
             f"Formal: `{output['formal']}`; boundaries: {output['completed_boundaries']}/{output['planned_boundaries']}", "",
             "| Task | boundaries | positive | negative | >2SE | split-half | SNR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task, values in task_summary.items():
        fmt = lambda x: "NA" if x is None else f"{x:.4g}"
        lines.append(f"| {task} | {values['boundaries']} | {values['positive_fraction']:.3f} | "
                     f"{values['negative_fraction']:.3f} | {values['positive_gt_2se_fraction']:.3f} | "
                     f"{fmt(values['split_half_reliability_median'])} | {fmt(values['snr'])} |")
    lines += ["", "## Review cases", "", "Cases are descriptive; inspect the stored parquet row and candidate frame before interpretation.", ""]
    for case in cases:
        lines.append(f"- {case['interpretation']}: {case['task']} ep{case['episode_id']} F{case['cutoff']}, gain={case['crossfit_gain']:.6g}, candidate={case['candidate'].get('candidate')}")
    (run / "MILESTONE4_REPORT.md").write_text("\n".join(lines) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run.resolve(), args.top_k), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
