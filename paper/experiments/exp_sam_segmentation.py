#!/usr/bin/env python3
"""Fruit instance segmentation and ripeness on rendered FruitPick frames (front and wrist camera).

    calibrate   HSV class thresholds that maximise pixel F1 on the training renders (train_clean ep 0-9)
    run         every method on a shard of the val renders (val_a, every 10th frame), one SAM model per job
    summarize   P/R/F1 at IoU 0.5, IoU, ripeness accuracy, recall by visible fraction, latency

Methods: HSV (colour components), SAM-FDM (colour-prompted SAM; ViT-B/L/H), SAM-AUTO (automatic masks filtered by
colour and compactness; ViT-B/H on every 5th frame), SAM-GTBOX (SAM prompted with the ground-truth box; upper bound
of mask quality). Ground truth: visible pixels of each fruit actor; fruits with fewer than MIN_GT pixels are
"ignore" regions (a prediction on them is neither a true nor a false positive).

    PY=robomme_benchmark/.venv/bin/python; cd ecm_vla/paper/experiments
    $PY exp_sam_segmentation.py calibrate
    CUDA_VISIBLE_DEVICES=0 $PY exp_sam_segmentation.py run --model vit_b --shard 0 --of 8
    $PY exp_sam_segmentation.py summarize
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from common import ECM, OUT, save_json
import perception as P

SEG = ECM / "data/perception_fruit/seg"
RAW = OUT / "sam_segmentation_raw"
MIN_GT = {"front": 20, "wrist": 60}
MIN_PRED = {"front": 10, "wrist": 30}


def frames(npz_path):
    z = np.load(npz_path)
    ids = json.loads(str(z["ids"]))
    for i, t in enumerate(z["frame"]):
        for cam in ("front", "wrist"):
            yield {"t": int(t), "cam": cam, "rgb": z[f"{cam}_rgb"][i], "seg": z[f"{cam}_seg"][i],
                   "depth": z[f"{cam}_depth"][i], "E": z[f"{cam}_extrinsic"][i], "K": z[f"{cam}_intrinsic"],
                   "fruit_ids": ids["fruit"], "ripe": z["ripe"], "radius": z["radius"], "pos": z["fruit_pos"][i],
                   "occluded": set(z["occluded"].tolist()), "index": i}


def calibrate():
    hist = {c: np.zeros((180, 32, 32), np.int64) for c in ("ripe", "unripe", "other")}
    files = sorted((SEG / "train_clean").glob("*.npz"))
    for f in files:
        for fr in frames(f):
            h = P.hsv(fr["rgb"]).reshape(-1, 3).astype(np.int64)
            idx = h[:, 0] * 1024 + (h[:, 1] // 8) * 32 + h[:, 2] // 8
            seg = fr["seg"].reshape(-1)
            lab = np.full(seg.shape, 2)
            for k, fid in enumerate(fr["fruit_ids"]):
                lab[seg == fid] = 0 if fr["ripe"][k] else 1
            for c, name in enumerate(("ripe", "unripe", "other")):
                hist[name] += np.bincount(idx[lab == c], minlength=180 * 1024).reshape(180, 32, 32)
    suf = {k: v[:, ::-1, ::-1].cumsum(1).cumsum(2)[:, ::-1, ::-1] for k, v in hist.items()}  # S >= s, V >= v
    tot = {k: int(v.sum()) for k, v in hist.items()}

    def best(pos, region_counts):
        tp = region_counts[pos]
        fp = sum(region_counts[k] for k in region_counts if k != pos)
        f1 = 2 * tp / np.maximum(2 * tp + fp + (tot[pos] - tp), 1)
        return f1

    cum = {k: np.concatenate([np.zeros((1, 32, 32), np.int64), v.cumsum(0)]) for k, v in suf.items()}  # over hue
    # ripe: H <= a or H >= b
    a = np.arange(0, 26)
    b = np.arange(150, 180)
    counts = {k: cum[k][a + 1][:, None] + (cum[k][180] - cum[k][b])[None] for k in cum}
    f1 = best("ripe", counts)
    ia, ib, s, v = np.unravel_index(np.argmax(f1), f1.shape)
    ripe = {"h_hi": int(a[ia]), "h_wrap": int(b[ib]), "s_min": int(s * 8), "v_min": int(v * 8),
            "pixel_f1": float(f1[ia, ib, s, v])}
    lo = np.arange(10, 51)
    hi = np.arange(20, 76)
    counts = {k: (cum[k][hi + 1][None] - cum[k][lo][:, None]) for k in cum}
    f1 = best("unripe", counts)
    f1[lo[:, None] > hi[None, :]] = -1
    il, ih, s, v = np.unravel_index(np.argmax(f1), f1.shape)
    unripe = {"h_lo": int(lo[il]), "h_hi": int(hi[ih]), "s_min": int(s * 8), "v_min": int(v * 8),
              "pixel_f1": float(f1[il, ih, s, v])}
    out = {"ripe": ripe, "unripe": unripe, "calibration_files": len(files), "pixels": tot}
    print(save_json("hsv_thresholds.json", out))
    print(json.dumps(out, indent=1))


def gt_instances(fr):
    out = []
    for k, fid in enumerate(fr["fruit_ids"]):
        m = fr["seg"] == fid
        px = int(m.sum())
        if px == 0:
            continue
        uv, z = P.project(fr["pos"][k], fr["K"], fr["E"])
        full = np.pi * (fr["K"][0, 0] * fr["radius"][k] / z) ** 2 if z and z > 0 else np.nan
        out.append({"k": k, "mask": m, "px": px, "vf": float(min(px / full, 1.0)) if full == full else None,
                    "ripe": bool(fr["ripe"][k]), "occluder": k in fr["occluded"], "count": px >= MIN_GT[fr["cam"]]})
    return out


def evaluate(preds, gts):
    """Greedy one-to-one matching at IoU >= 0.5; returns (per-GT rows, n false positives)."""
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            v = P.iou(p["mask"], g["mask"])
            if v > 0:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    pm, gm, best = {}, {}, defaultdict(float)
    for v, i, j in pairs:
        best[j] = max(best[j], v)
        if v >= 0.5 and i not in pm and j not in gm:
            pm[i], gm[j] = j, v
    fp = sum(1 for i in range(len(preds)) if i not in pm)  # a match to an ignore region is not counted either way
    rows = []
    for j, g in enumerate(gts):
        if not g["count"]:
            continue
        i = next((i for i, jj in pm.items() if jj == j), None)
        rows.append({"k": g["k"], "px": g["px"], "vf": g["vf"], "ripe": g["ripe"], "occ": g["occluder"],
                     "tp": i is not None, "iou": float(gm.get(j, 0.0)), "best_iou": float(best[j]),
                     "cls_ok": None if i is None else bool(preds[i]["cls"] == (P.RIPE if g["ripe"] else P.UNRIPE))})
    return rows, fp


MAX_FRAC = {"front": 0.08, "wrist": 0.6}
LIGHTING = ("dark", "bright", "warm")


def tune():
    """SAM-FDM prompt (box padding, centre point) chosen by F1 then IoU on the training renders (ViT-B)."""
    thr = P.load_thresholds()
    sam = P.SamFruit("vit_b")
    grid = [(pad, point) for pad in (0, 2, 4) for point in (True, False)]
    acc = {g: [] for g in grid}
    for f in sorted((SEG / "train_clean").glob("*.npz")):
        for fr in frames(f):
            if fr["index"] % 2:
                continue
            gts = gt_instances(fr)
            sam.set_image(fr["rgb"])
            for pad, point in grid:
                preds = sam.fdm(fr["rgb"], thr, MIN_PRED[fr["cam"]], pad=pad, point=point)
                rows, fp = evaluate(preds, gts)
                acc[(pad, point)].append((rows, fp))
    table = {}
    for (pad, point), items in acc.items():
        rows = [r for rs, _ in items for r in rs]
        fp = sum(x for _, x in items)
        tp = sum(r["tp"] for r in rows)
        p, rc = tp / max(tp + fp, 1), tp / max(len(rows), 1)
        table[f"pad={pad},point={point}"] = {"pad": pad, "point": point, "f1": 2 * p * rc / max(p + rc, 1e-9),
                                             "iou": float(np.mean([r["iou"] for r in rows if r["tp"]]))}
    best = max(table.values(), key=lambda e: (round(e["f1"], 3), e["iou"]))
    print(save_json("fdm_prompt.json", {"grid": table, "best": {"pad": best["pad"], "point": best["point"]}}))
    print(json.dumps(table, indent=1))


def run(model, shard, of):
    import torch
    thr = P.load_thresholds()
    prompt = P.load_prompt()
    sam = P.SamFruit(model)
    files = sorted((SEG / "val_a").glob("*.npz"))[shard::of]
    RAW.mkdir(parents=True, exist_ok=True)
    out = open(RAW / f"{model}_{shard}of{of}.jsonl", "w")
    for f in files:
        task, ep = f.stem.split("_ep")
        for fr in frames(f):
            cam = fr["cam"]
            gts = gt_instances(fr)
            counted = [g for g in gts if g["count"]]
            boxes = []
            for g in counted:
                ys, xs = np.nonzero(g["mask"])
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            base = {"task": task, "ep": int(ep), "t": fr["t"], "cam": cam}
            # lighting variants on every 3rd sampled frame; colour thresholds stay as calibrated
            for light in ("normal", *(LIGHTING if fr["index"] % 3 == 0 else ())):
                rgb = fr["rgb"] if light == "normal" else P.photometric(fr["rgb"], light)
                tag = "" if light == "normal" else f"@{light}"
                results = {}
                if model == "vit_b":
                    t0 = time.perf_counter()
                    results["HSV"] = (P.hsv_instances(rgb, thr, MIN_PRED[cam]), time.perf_counter() - t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                sam.set_image(rgb)
                torch.cuda.synchronize()
                t_enc = time.perf_counter() - t0
                t0 = time.perf_counter()
                fdm = sam.fdm(rgb, thr, MIN_PRED[cam], **prompt)
                torch.cuda.synchronize()
                results[f"SAM-FDM/{model}"] = (fdm, t_enc + time.perf_counter() - t0)
                masks = sam.boxes(np.array(boxes, float))[0] if counted else []
                results[f"SAM-GTBOX/{model}"] = ([{"mask": m, "cls": P.RIPE if g["ripe"] else P.UNRIPE, "score": 1.0}
                                                  for m, g in zip(masks, counted)], None)
                if model in ("vit_b", "vit_h") and fr["index"] % 5 == 0 and light == "normal":
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    auto = sam.auto(rgb, thr, MIN_PRED[cam], MAX_FRAC[cam])
                    torch.cuda.synchronize()
                    results[f"SAM-AUTO/{model}"] = (auto, time.perf_counter() - t0)
                for method, (preds, dt) in results.items():
                    rows, fp = evaluate(preds, gts)
                    out.write(json.dumps({**base, "m": method + tag, "fp": fp, "n_pred": len(preds),
                                          "sec": dt, "enc": t_enc if method.startswith("SAM-FDM") else None,
                                          "gt": rows}) + "\n")
        out.flush()
        print(f"{model} {f.stem} done", flush=True)
    out.close()


def summarize():
    acc = defaultdict(lambda: {"gt": [], "fp": 0, "pred": 0, "sec": [], "enc": []})
    for f in sorted(RAW.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            for key in ((r["m"], r["cam"]), (r["m"], "all")):
                a = acc[key]
                a["gt"] += [dict(g, task=r["task"]) for g in r["gt"]]
                a["fp"] += r["fp"]
                a["pred"] += r["n_pred"]
                if r["sec"] is not None and key[1] != "all":
                    a["sec"].append(r["sec"])
                if r.get("enc") is not None and key[1] != "all":
                    a["enc"].append(r["enc"])
    report = {}
    for (m, cam), a in sorted(acc.items()):
        g = a["gt"]
        tp = sum(x["tp"] for x in g)
        prec = tp / max(tp + a["fp"], 1)
        rec = tp / max(len(g), 1)
        bins = {"vf>=0.75": lambda x: x["vf"] is not None and x["vf"] >= 0.75,
                "0.25<=vf<0.75": lambda x: x["vf"] is not None and 0.25 <= x["vf"] < 0.75,
                "vf<0.25": lambda x: x["vf"] is not None and x["vf"] < 0.25,
                "leaf_occluded_fruit": lambda x: x["occ"]}
        cls = [x["cls_ok"] for x in g if x["cls_ok"] is not None]
        report[f"{m}/{cam}"] = {
            "gt_instances": len(g), "tp": tp, "fp": a["fp"], "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "mean_iou_tp": float(np.mean([x["iou"] for x in g if x["tp"]])) if tp else None,
            "instance_miou": float(np.mean([x["best_iou"] for x in g])) if g else None,
            "ripeness_acc": float(np.mean(cls)) if cls else None,
            "recall_by_visibility": {k: (float(np.mean([x["tp"] for x in g if f(x)])) if any(f(x) for x in g) else None,
                                         sum(1 for x in g if f(x))) for k, f in bins.items()},
            "ms_per_image": float(np.median(a["sec"]) * 1000) if a["sec"] else None,
            "encoder_ms": float(np.median(a["enc"]) * 1000) if a["enc"] else None,
        }
    print(save_json("sam_segmentation.json", report))
    for k, e in report.items():
        if k.endswith("/all"):
            continue
        rb = e["recall_by_visibility"]
        print(f"{k:24s} P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f} "
              f"IoU={e['mean_iou_tp'] or 0:.3f} ripe={e['ripeness_acc'] or 0:.3f} "
              f"R(hi/mid/lo)={rb['vf>=0.75'][0] or 0:.2f}/{rb['0.25<=vf<0.75'][0] or 0:.2f}/{rb['vf<0.25'][0] or 0:.2f} "
              f"ms={e['ms_per_image'] or 0:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("calibrate", "tune", "run", "summarize"))
    ap.add_argument("--model", default="vit_b")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    {"calibrate": calibrate, "tune": tune, "summarize": summarize}.get(
        args.cmd, lambda: run(args.model, args.shard, args.of))()


if __name__ == "__main__":
    main()
