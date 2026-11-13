#!/usr/bin/env python3
"""From pixels to memory: the SAM-based fruit distance module (SAM-FDM) replaces the simulator distance, and the
picking events and event-keyframe memories it produces are compared with those of the ground truth.

Front camera RGB-D of the recorded val demonstrations (h5), tcp position from proprioception (forward kinematics).

SAM-FDM
  anchors   SAM-FDM instances on one front frame, each back-projected to a sphere centre (perception.fruit_center)
            with its ripeness from the mask colour: the spatial memory of where the fruits hang
  vacancy   every frame, the depth at each anchor's pixel: more than VACANT_M behind the expected fruit surface means
            the fruit is gone (VACANT_FRAMES in a row: picked); more than OCCLUDED_M in front means something
            (gripper, leaves) hides it, which says nothing
  distance  d_t = tcp to the nearest pickable anchor not yet picked (all anchors in FruitCount, ripe ones in
            FruitOccludedRipe); 10 m when none is left
Variants
  GT              simulator distance (reference)
  ANCHOR-0        anchors from frame 0 (before the gust), ideal depth
  ANCHOR-0-NOISY  same with depth noise sigma(z) = 1 mm + 2.5 mm * z^2 and 2 % missing pixels
  ANCHOR-29       anchors from the last observation frame (after the gust: no memory of what the leaves hide)
  PERFRAME        no anchors: every frame, the colour detections (HSV, as SAM-FDM's prompts) back-projected; d_t to
                  the nearest detected pickable fruit, unknown when none is visible
Metrics: anchor recall/precision/localisation error/ripeness accuracy; vacancy delay; distance error; event
P/R/F1 (plain and hysteresis detector) against the detachments; K4/K8 memory Jaccard and keyframe recall against the
memories of the ground-truth signal.

    CUDA_VISIBLE_DEVICES=0 robomme_benchmark/.venv/bin/python ecm_vla/paper/experiments/exp_perception_distance.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import argparse
import pickle

import h5py
import numpy as np

from common import OUT, SKIPPED, describe, episodes, save_json
import perception as P
from exp_event_detection import DETECTORS, R, TOL, W, match, memory_at, scores
from memory_management.event_keyframes import keyframes

VACANT_M, OCCLUDED_M, VACANT_FRAMES, GRASP_FRAMES = 0.05, 0.02, 3, 2
NO_TARGET = 10.0
EVERY = 16
ANCHOR_CACHE = OUT / "perception_anchors.pkl"
# name -> (anchor set, noisy depth, grasp cue)
VARIANTS = {
    "ANCHOR-0/DEPTH-ONLY": ("ANCHOR-0", False, False),
    "ANCHOR-0": ("ANCHOR-0", False, True),
    "ANCHOR-0-NOISY": ("ANCHOR-0-NOISY", True, True),
    "ANCHOR-29": ("ANCHOR-29", False, True),
}
ORDER = ("GT", *VARIANTS, "PERFRAME")


def noisy_depth(depth_mm, rng):
    z = depth_mm.astype(np.float64) / 1000.0
    sigma = 0.001 + 0.0025 * z ** 2
    z = z + rng.normal(size=z.shape) * sigma
    z[rng.random(z.shape) < 0.02] = 0.0
    return np.clip(z * 1000.0, 0, 32767).astype(np.int16)


def load_front(ep, frames=None):
    with h5py.File(ep.h5) as f:
        g = f[f"episode_{ep.episode}"]
        n = int(g["setup/n_timesteps"][()])
        K = g["setup/front_camera_intrinsic"][()].astype(np.float64)
        E = g["timestep_0/obs/front_camera_extrinsic"][()].astype(np.float64)
        ts = range(n) if frames is None else frames
        rgb = np.stack([g[f"timestep_{t}/obs/front_rgb"][()] for t in ts])
        depth = np.stack([g[f"timestep_{t}/obs/front_depth"][()][..., 0] for t in ts])
    return rgb, depth, K, E


def anchors_from(sam, rgb, depth, K, E, thr, prompt):
    sam.set_image(rgb)
    out = []
    for inst in sam.fdm(rgb, thr, 10, **prompt):
        c = P.fruit_center(inst["mask"], depth, K, E)
        if c is None:
            continue
        p, r = c
        if all(np.linalg.norm(p - a["p"]) > 0.03 for a in out):
            out.append({"p": p, "r": r, "ripe": inst["cls"] == P.RIPE})
    return out


def stage_anchors(eps, model):
    sam = P.SamFruit(model)
    thr, prompt = P.load_thresholds(), P.load_prompt()
    cache = {}
    for ep in eps:
        rgb, depth, K, E = load_front(ep, [0, 29])
        rng = np.random.default_rng([ep.episode, 0 if ep.task == "count" else 1, 7])
        cache[(ep.task, ep.episode)] = {
            "ANCHOR-0": anchors_from(sam, rgb[0], depth[0], K, E, thr, prompt),
            "ANCHOR-0-NOISY": anchors_from(sam, rgb[0], noisy_depth(depth[0], rng), K, E, thr, prompt),
            "ANCHOR-29": anchors_from(sam, rgb[1], depth[1], K, E, thr, prompt),
        }
    ANCHOR_CACHE.write_bytes(pickle.dumps(cache))
    return cache


def holding(rows):
    """Gripper commanded closed and the fingers stopped between empty-closed and open: an object is in the hand."""
    return np.array([r["gripper_action"] < 0 and 0.01 < sum(r["gripper_qpos"]) < 0.075 for r in rows])


def anchor_distance(anchors, depth_seq, tcp, K, E, pickable_all, hold=None):
    """(d_t, picked time per anchor) with causal vacancy detection; with `hold`, a grasp held for GRASP_FRAMES
    within R of an anchor also marks it picked."""
    n = len(tcp)
    picked = [None] * len(anchors)
    streak = [0] * len(anchors)
    hold_run = 0
    proj = []
    for a in anchors:
        uv, zc = P.project(a["p"], K, E)
        proj.append((None if uv is None else (int(round(uv[0])), int(round(uv[1]))), zc - a["r"]))
    d = np.full(n, NO_TARGET)
    for t in range(n):
        dep = depth_seq[t]
        for i, ((uv, zs), a) in enumerate(zip(proj, anchors)):
            if picked[i] is not None or uv is None:
                continue
            u, v = uv
            win = dep[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3].astype(np.float64)
            win = win[win > 0] / 1000.0
            if not win.size:
                continue
            zo = float(np.median(win))
            if zo > zs + VACANT_M:
                streak[i] += 1
                if streak[i] >= VACANT_FRAMES:
                    picked[i] = t
            elif zo < zs - OCCLUDED_M:
                pass  # hidden: keep the streak as it is
            else:
                streak[i] = 0
        if hold is not None:
            hold_run = hold_run + 1 if hold[t] else 0
            if hold_run >= GRASP_FRAMES:
                open_ = [i for i in range(len(anchors)) if picked[i] is None]
                if open_:
                    i = min(open_, key=lambda i: np.linalg.norm(tcp[t] - anchors[i]["p"]))
                    if np.linalg.norm(tcp[t] - anchors[i]["p"]) <= R:
                        picked[i] = t
        cands = [a["p"] for i, a in enumerate(anchors) if picked[i] is None and (pickable_all or a["ripe"])]
        if cands:
            d[t] = min(np.linalg.norm(tcp[t] - p) for p in cands)
    return d, picked


def perframe_distance(rgb_seq, depth_seq, tcp, K, E, thr, pickable_all):
    d = np.full(len(tcp), np.nan)
    for t in range(len(tcp)):
        pts = []
        for inst in P.hsv_instances(rgb_seq[t], thr, 10):
            if not (pickable_all or inst["cls"] == P.RIPE):
                continue
            c = P.fruit_center(inst["mask"], depth_seq[t], K, E)
            if c is not None:
                pts.append(c[0])
        if pts:
            d[t] = min(np.linalg.norm(tcp[t] - p) for p in pts)
    return d


def evaluate_signal(d_hat, d_gt, detaches):
    out = {}
    near = d_gt < 0.15
    err = np.abs(d_hat - d_gt)
    out["abs_err_near"] = err[near & np.isfinite(d_hat)].tolist()
    out["missing_near"] = float(np.mean(~np.isfinite(d_hat[near]))) if near.any() else 0.0
    for name in ("plain", "hysteresis"):
        ev = DETECTORS[name](d_hat)
        tp, leads, rel = match(ev, detaches)
        out[name] = {"tp": tp, "detected": len(ev), "gt": len(detaches), "leads": leads, "rel_err": rel}
    out["jaccard"] = {det: {4: [], 8: []} for det in ("plain", "hysteresis")}
    out["key_recall"] = {det: {4: [], 8: []} for det in ("plain", "hysteresis")}
    for t in range(EVERY, len(d_gt), EVERY):
        for k in (4, 8):
            ref = memory_at(d_gt, t, k)
            keys = keyframes(DETECTORS["plain"](d_gt[:t + 1]), k)
            for det in ("plain", "hysteresis"):
                got = memory_at(d_hat, t, k, None if det == "plain" else DETECTORS[det])
                out["jaccard"][det][k].append(len(ref & got) / len(ref | got))
                if keys:
                    g = np.array(sorted(got))
                    out["key_recall"][det][k].append(float(np.mean([np.abs(g - f).min() <= W for f in keys])))
    return out


def run_episode(args):
    task, episode, set_name, anchors = args
    ep = next(e for e in episodes(set_name, (task,)) if e.episode == episode)
    thr = P.load_thresholds()
    rgb, depth, K, E = load_front(ep)
    n = min(len(rgb), len(ep.rows))
    rgb, depth = rgb[:n], depth[:n]
    tcp = np.array([r["tcp_pos"] for r in ep.rows[:n]])
    d_gt = ep.distance[:n]
    detaches = [t for t in ep.detach_times.values() if t < n]
    pick_all = task == "count"
    res = {"task": task, "episode": episode, "variants": {}, "anchors": {}}
    rng = np.random.default_rng([episode, 0 if pick_all else 1, 11])
    noisy = np.stack([noisy_depth(x, rng) for x in depth])
    hold = holding(ep.rows[:n])
    for name, (anchor_set, use_noisy, grasp) in VARIANTS.items():
        an = anchors[anchor_set]
        d_hat, picked = anchor_distance(an, noisy if use_noisy else depth, tcp, K, E, pick_all,
                                        hold if grasp else None)
        res["variants"][name] = evaluate_signal(d_hat, d_gt, detaches)
        # anchors against the fruits at frame 0
        fruits = [np.array(f["pos"]) for f in ep.rows[0]["fruits"]]
        used, errs, ripe_ok = set(), [], []
        assign = {}
        for i, a in enumerate(an):
            dists = [np.linalg.norm(a["p"] - f) if j not in used else np.inf for j, f in enumerate(fruits)]
            j = int(np.argmin(dists))
            if dists[j] <= 0.05:
                used.add(j)
                assign[i] = j
                errs.append(dists[j])
                ripe_ok.append(a["ripe"] == ep.ripe[j])
        delays, false_vac = [], 0
        for i, j in assign.items():
            if picked[i] is None:
                continue
            if j in ep.detach_times and picked[i] >= ep.detach_times[j]:
                delays.append(picked[i] - ep.detach_times[j])
            else:
                false_vac += 1
        hidden = {o["fruit"] for o in ep.layout.occluders}
        res["anchors"][name] = {"n_anchor": len(an), "n_fruit": len(fruits), "matched": len(assign),
                                "err_m": errs, "ripe_ok": ripe_ok, "vacancy_delay": delays, "false_vacancy": false_vac,
                                "hidden_ripe_found": [j in used for j in hidden if ep.ripe[j]]}
    d_pf = perframe_distance(rgb, depth, tcp, K, E, thr, pick_all)
    res["variants"]["PERFRAME"] = evaluate_signal(d_pf, d_gt, detaches)
    res["variants"]["GT"] = evaluate_signal(d_gt, d_gt, detaches)
    return res


def aggregate(results):
    report = {}
    for scope in ("all", "count", "occluded_ripe"):
        rs = [r for r in results if scope == "all" or r["task"] == scope]
        if not rs:
            continue
        for v in ORDER:
            items = [r["variants"][v] for r in rs]
            e = {"abs_err_near_cm": describe(np.array(sum((x["abs_err_near"] for x in items), [])) * 100),
                 "missing_near": float(np.mean([x["missing_near"] for x in items]))}
            for det in ("plain", "hysteresis"):
                e[det] = scores([x[det] for x in items])
                e[det]["memory_jaccard"] = {k: float(np.mean(sum((x["jaccard"][det][k] for x in items), [])))
                                            for k in (4, 8)}
                e[det]["keyframe_recall_w3"] = {k: float(np.mean(sum((x["key_recall"][det][k] for x in items), [])))
                                                for k in (4, 8)}
            report[f"{scope}/{v}"] = e
        for v in VARIANTS:
            a = [r["anchors"][v] for r in rs]
            n_fruit = sum(x["n_fruit"] for x in a)
            matched = sum(x["matched"] for x in a)
            hidden = sum((x["hidden_ripe_found"] for x in a), [])
            report[f"{scope}/{v}/anchors"] = {
                "recall": matched / max(n_fruit, 1), "precision": matched / max(sum(x["n_anchor"] for x in a), 1),
                "loc_err_cm": describe(np.array(sum((x["err_m"] for x in a), [])) * 100),
                "ripeness_acc": float(np.mean(sum((x["ripe_ok"] for x in a), []))),
                "vacancy_delay_frames": describe(sum((x["vacancy_delay"] for x in a), [])),
                "false_vacancy": int(sum(x["false_vacancy"] for x in a)),
                "hidden_ripe_found": float(np.mean(hidden)) if hidden else None, "hidden_ripe_n": len(hidden)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse-anchors", action="store_true")
    args = ap.parse_args()
    eps = [e for e in (*episodes("val_a"), *episodes("val_b")) if args.limit is None or e.episode < args.limit]
    if args.reuse_anchors and ANCHOR_CACHE.exists():
        cache = pickle.loads(ANCHOR_CACHE.read_bytes())
    else:
        cache = stage_anchors(eps, args.model)
    print(f"anchors ready for {len(cache)} episodes", flush=True)
    jobs = [(ep.task, ep.episode, ep.set_name, cache[(ep.task, ep.episode)]) for ep in eps]
    with ProcessPoolExecutor(min(64, len(jobs))) as ex:
        results = list(ex.map(run_episode, jobs))
    report = aggregate(results)
    report["config"] = {"model": args.model, "vacant_m": VACANT_M, "occluded_m": OCCLUDED_M,
                        "vacant_frames": VACANT_FRAMES, "grasp_frames": GRASP_FRAMES, "episodes": len(results)}
    report["skipped"] = SKIPPED
    print(save_json("perception_distance.json" if args.limit is None else "perception_distance_smoke.json", report))
    for k, e in report.items():
        if not k.startswith("all/"):
            continue
        if k.endswith("/anchors"):
            print(f"{k:28s} recall={e['recall']:.3f} prec={e['precision']:.3f} loc_cm={e['loc_err_cm'].get('median', 0):.2f} "
                  f"ripe={e['ripeness_acc']:.3f} delay={e['vacancy_delay_frames'].get('median')} false_vac={e['false_vacancy']} "
                  f"hidden_found={e['hidden_ripe_found']}")
        else:
            print(f"{k:28s} err_cm={e['abs_err_near_cm'].get('median', float('nan')):.2f} miss={e['missing_near']:.3f}")
            for det in ("plain", "hysteresis"):
                s = e[det]
                print(f"    {det:10s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} exact={s['count_exact']:.2f} "
                      f"lead={s['contact_lead_frames'].get('median')} rel_err={s['release_error_frames'].get('median')} "
                      f"J4={s['memory_jaccard'][4]:.3f} J8={s['memory_jaccard'][8]:.3f} "
                      f"KR4={s['keyframe_recall_w3'][4]:.3f} KR8={s['keyframe_recall_w3'][8]:.3f}")


if __name__ == "__main__":
    main()
