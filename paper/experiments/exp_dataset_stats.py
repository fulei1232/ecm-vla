#!/usr/bin/env python3
"""Benchmark statistics (paper Table: datasets): episode length, scene composition, picks, phase durations,
occlusion severity of FruitOccludedRipe, generator acceptance rate, and the execution noise of the DART set.

    robomme_benchmark/.venv/bin/python ecm_vla/paper/experiments/exp_dataset_stats.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json

import numpy as np

from common import DATA, SETS, SKIPPED, TASKS, describe, episodes, save_json

PHASES = ("observe", "reach", "approach", "grasp", "pull", "carry", "release", "settle", "done")


def occlusion(ep):
    """Front-camera visibility (rendered pixels) of the leaf-hidden fruits before and after the gust."""
    L = ep.layout
    rows = ep.rows
    observe = [r for r in rows if r["phase"] == "observe"]
    before, after = observe[0], observe[-1]
    out = []
    for occ in L.occluders:
        i = occ["fruit"]
        b, a = before["fruits"][i]["front_pixels"], after["fruits"][i]["front_pixels"]
        on_tree = [r for r in rows if not r["fruits"][i]["attached"]]
        hidden_frames = sum(r["fruits"][i]["front_pixels"] == 0 for r in on_tree)
        out.append({"ripe": bool(L.fruits[i]["ripe"]), "pixels_before": b, "pixels_after": a,
                    "visible_fraction_after": a / b if b else float("nan"),
                    "invisible_frames_on_tree_fraction": hidden_frames / max(len(on_tree), 1)})
    others = [after["fruits"][i]["front_pixels"] / max(before["fruits"][i]["front_pixels"], 1)
              for i in range(len(L.fruits)) if i not in {o["fruit"] for o in L.occluders}]
    return out, others


def main():
    report = {}
    for set_name in SETS:
        for task in TASKS:
            eps = list(episodes(set_name, (task,)))
            if not eps:
                continue
            steps, fruits, picks, attempts, seconds = [], [], [], [], []
            fail_reasons = Counter()
            phase_steps = defaultdict(list)
            near_frames = []
            n_ripe, n_unripe = [], []
            occ_rows, other_vis = [], []
            noise_tcp, noise_by_phase = [], defaultdict(list)
            for ep in eps:
                m = ep.manifest
                steps.append(m["steps"])
                fruits.append(m["n_fruits"])
                picks.append(m["n_pick"])
                attempts.append(m["attempt"] + 1)
                seconds.append(m.get("seconds", np.nan))
                for f in m.get("failures", []):
                    fail_reasons[f["reason"].split(" ", 2)[-1] if f["reason"].startswith("fruit") else f["reason"]] += 1
                c = Counter(r["phase"] for r in ep.rows)
                for p in PHASES:
                    phase_steps[p].append(c.get(p, 0))
                near_frames.append(int((ep.distance <= 0.03).sum()))
                if task == "occluded_ripe":
                    n_ripe.append(sum(ep.ripe))
                    n_unripe.append(len(ep.ripe) - sum(ep.ripe))
                    o, others = occlusion(ep)
                    occ_rows += o
                    other_vis += others
                if set_name == "train_dart":
                    for r in ep.rows:
                        if r.get("noise_tcp_m") is not None and r["phase"] in ("reach", "approach", "pull", "carry"):
                            noise_tcp.append(r["noise_tcp_m"])
                            noise_by_phase[r["phase"]].append(r["noise_tcp_m"])
            entry = {
                "episodes": len(eps), "steps": describe(steps), "fruits": describe(fruits),
                "picks": describe(picks), "picks_hist": dict(sorted(Counter(picks).items())),
                "fruits_hist": dict(sorted(Counter(fruits).items())),
                "attempts_total": int(sum(attempts)), "acceptance_rate": len(eps) / sum(attempts),
                "failure_reasons": dict(fail_reasons.most_common()),
                "generation_seconds": describe([s for s in seconds if np.isfinite(s)]),
                "phase_steps": {p: describe(v) for p, v in phase_steps.items()},
                "near_frames_r3cm": describe(near_frames),
            }
            if task == "occluded_ripe":
                entry["ripe"] = describe(n_ripe)
                entry["unripe"] = describe(n_unripe)
                entry["occluders"] = {
                    "n": len(occ_rows),
                    "hidden_ripe_fraction": float(np.mean([o["ripe"] for o in occ_rows])),
                    "visible_fraction_after_gust": describe([o["visible_fraction_after"] for o in occ_rows]),
                    "fully_hidden_after_gust": float(np.mean([o["pixels_after"] == 0 for o in occ_rows])),
                    "invisible_frames_while_on_tree": describe([o["invisible_frames_on_tree_fraction"] for o in occ_rows]),
                    "unoccluded_visible_fraction_after_gust": describe(other_vis),
                }
            if set_name == "train_dart":
                entry["dart_tcp_displacement_cm"] = describe(np.array(noise_tcp) * 100)
                entry["dart_tcp_displacement_cm_by_phase"] = {p: describe(np.array(v) * 100) for p, v in noise_by_phase.items()}
                entry["dart_noise"] = eps[0].manifest.get("noise")
            report[f"{set_name}/{task}"] = entry
    builds = {}
    for name in ("fruit_rm75b_train", "fruit_dart_rm75b_train"):
        b = json.loads((DATA / f"finetune_kf/{name}/meta/build.json").read_text())
        builds[name] = {k: b[k] for k in ("episodes", "execution_samples", "per_task") if k in b}
    report["feature_builds"] = builds
    report["skipped"] = SKIPPED
    path = save_json("dataset_stats.json", report)
    print(path)
    for key, e in report.items():
        if key in ("feature_builds", "skipped"):
            print(key, e)
            continue
        line = (f"{key:32s} n={e['episodes']:3d} steps={e['steps']['mean']:.0f}±{e['steps']['std']:.0f} "
                f"[{e['steps']['min']:.0f},{e['steps']['max']:.0f}] fruits={e['fruits']['mean']:.2f} "
                f"picks={e['picks']['mean']:.2f} accept={e['acceptance_rate']:.2f}")
        if "occluders" in e:
            o = e["occluders"]
            line += (f" occ_vis_after={o['visible_fraction_after_gust']['median']:.2f} "
                     f"fully_hidden={o['fully_hidden_after_gust']:.2f}")
        if "dart_tcp_displacement_cm" in e:
            d = e["dart_tcp_displacement_cm"]
            line += f" dart_tcp_cm med={d['median']:.2f} p90={d['p90']:.2f}"
        print(line)


if __name__ == "__main__":
    main()
