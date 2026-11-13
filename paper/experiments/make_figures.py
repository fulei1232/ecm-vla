#!/usr/bin/env python3
"""Paper figures from paper/results (+ the recorded val demonstrations). A figure whose results are missing is
skipped, so the script can be rerun as experiments finish.

    robomme_benchmark/.venv/bin/python ecm_vla/paper/experiments/make_figures.py [--only fig7 ...] [--sam]
--sam draws the SAM qualitative panel (needs a GPU and the SAM checkpoints).
"""
from __future__ import annotations

import argparse
import json
import pickle

import h5py
import matplotlib
import matplotlib.lines
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import ECM, OUT, episodes  # noqa: E402
from memory_management.event_keyframes import KeyframeConfig, detect_events, select  # noqa: E402
from memory_management.validation.event_memory import event_selection, proximity  # noqa: E402
from memory_management.validation.partition import partition  # noqa: E402

FIG = ECM / "paper/figures"
R, TOL = 0.03, 0.005
COLORS = {"U": "#7f7f7f", "RANDOM": "#bcbd22", "RECENT": "#8c564b", "PEKM-2": "#9ecae1", "PEKM-4": "#3182bd",
          "PEKM-8": "#08519c", "PEKM-S-0.5": "#fd8d3c", "PEKM-S-1": "#d94801"}
LABELS = {"U": "Uniform", "RANDOM": "Random", "RECENT": "Recent", "PEKM-2": "PEKM k=2", "PEKM-4": "PEKM k=4",
          "PEKM-8": "PEKM k=8", "PEKM-S-0.5": "PEKM-S α=0.5", "PEKM-S-1": "PEKM-S α=1"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(FIG / f"{name}.png")


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def h5_frames(ep, ts, cam="front"):
    with h5py.File(ep.h5) as f:
        g = f[f"episode_{ep.episode}"]
        return [g[f"timestep_{t}/obs/{cam}_rgb"][()] for t in ts]


def fig2_scene():
    """The two tasks from the front and wrist cameras, before and after the gust."""
    count = next(e for e in episodes("val_a", ("count",)) if e.episode == 0)
    occ = next(e for e in episodes("val_a", ("occluded_ripe",)) if e.episode == 0)
    c_first = min(count.detach_times.values()) - 4
    o_first = min(occ.detach_times.values()) - 4
    panels = [
        (h5_frames(count, [0])[0], "(a) Count · front · t=0"),
        (h5_frames(count, [c_first + 60])[0], f"(b) Count · front · t={c_first + 60}"),
        (h5_frames(count, [c_first], "wrist")[0], f"(c) Count · wrist · contact"),
        (h5_frames(occ, [0])[0], "(d) OccludedRipe · t=0 (before gust)"),
        (h5_frames(occ, [29])[0], "(e) OccludedRipe · t=29 (after gust)"),
        (h5_frames(occ, [o_first], "wrist")[0], f"(f) OccludedRipe · wrist · contact"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9))
    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    save(fig, "fig2_scene")


TASK_NAMES = {"count": "FruitCount", "occluded_ripe": "FruitOccludedRipe"}


def fig4_timeline(task="count", episode=0, out="fig4_memory_timeline", letters=("(a) ", "(b) ")):
    """Distance signal of one episode and the 16 frames each memory keeps at the last decision."""
    ep = next(e for e in episodes("val_a", (task,)) if e.episode == episode)
    d = ep.distance
    t = (len(d) - 1) // 16 * 16
    mems = {
        "U": select(d, t, KeyframeConfig(frames=16, events=0))[0],
        "PEKM-4": select(d, t, KeyframeConfig(frames=16, events=4, radius=R, tol=TOL))[0],
        "PEKM-8": select(d, t, KeyframeConfig(frames=16, events=8, radius=R, tol=TOL))[0],
        "PEKM-S-1": event_selection(partition(t, capacity=16, recent_slots=4), proximity(d[:t + 1], 0.05), 1.0),
    }
    events = detect_events(d[:t + 1], R, TOL)
    done = [e for e in events if e["release"] is not None]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.2]})
    ax.plot(np.minimum(d[:t + 1], 0.35), color="k", lw=0.9)
    ax.axhline(R, color="tab:red", ls="--", lw=0.7)
    ax.text(t, R + 0.01, "r = 3 cm", color="tab:red", ha="right", fontsize=6)
    for i, e in enumerate(events):
        end = e["release"] if e["release"] is not None else e["end"]
        for a in (ax, bx):
            a.axvspan(e["start"] - 3, end + 3, color="tab:red", alpha=0.12, lw=0)
        ax.plot(e["contact"], d[e["contact"]], "v", color="tab:red", ms=4)
        ax.text((e["start"] + end) / 2, 0.375, f"pick {i + 1}", ha="center", fontsize=6.5, color="tab:red")
    for tt in ep.detach_times.values():
        ax.axvline(tt, color="tab:green", lw=0.6, ls=":")
    ax.set_ylim(-0.01, 0.40)
    ax.set_ylabel("$d_t$ (m, clipped)")
    ax.set_title(f"{letters[0]}{TASK_NAMES[task]} validation episode {ep.episode}: distance to the nearest pickable fruit",
                 loc="left")
    for row, (name, frames) in enumerate(mems.items()):
        frames = np.asarray(frames)
        key = np.zeros(len(frames), bool)
        hits = []
        for e in done:
            near = (np.abs(frames - e["contact"]) <= 3) | (np.abs(frames - e["release"]) <= 3)
            key |= near
            hits.append(bool(near.any()))
        bx.scatter(frames[~key], np.full((~key).sum(), row), marker="|", s=70, color=COLORS[name], alpha=0.35)
        bx.scatter(frames[key], np.full(key.sum(), row), marker="|", s=110, lw=2.0, color=COLORS[name])
        for e, hit in zip(done, hits):
            bx.text((e["start"] + e["release"]) / 2, row - 0.36, "✓" if hit else "✗", ha="center", va="center",
                    fontsize=7.5, color="tab:green" if hit else "tab:red", fontweight="bold")
        bx.text(t + 18, row, f"{sum(hits)}/{len(done)} picks" + ("" if all(hits) else ", count lost"),
                va="center", fontsize=6.5, color="k" if all(hits) else "tab:red")
    bx.set_yticks(range(len(mems)), [LABELS[n] for n in mems])
    bx.set_ylim(-0.8, len(mems) - 0.4)
    bx.set_xlim(-10, t + 120)
    bx.set_xlabel("frame")
    bx.set_title(f"{letters[1]}the 16 frames each memory holds at decision time t = {t} "
                 "(bold: within ±3 frames of a pick keyframe)", loc="left")
    bx.invert_yaxis()
    save(fig, out)


def fig_s_timelines():
    """Further timelines for the supplement: a long FruitCount episode and two FruitOccludedRipe episodes."""
    for task, episode, tag in (("count", 5, "a"), ("occluded_ripe", 3, "b"), ("occluded_ripe", 1, "c")):
        fig4_timeline(task, episode, f"figS_timeline_{tag}", (f"({tag}1) ", f"({tag}2) "))


def fig_anchor_timing():
    """Anchoring before vs after the gust and how the loss propagates to events and memory (E6)."""
    res = load("perception_distance.json")
    if res is None:
        return
    metrics = [("anchor recall", lambda a, c: a["recall"]),
               ("hidden ripe\nanchored", lambda a, c: a["hidden_ripe_found"]),
               ("event recall", lambda a, c: c["plain"]["recall"]),
               ("count exact", lambda a, c: c["plain"]["count_exact"]),
               ("K4 keyframe\nrecall", lambda a, c: c["plain"]["keyframe_recall_w3"]["4"]),
               ("K4 memory\nJaccard", lambda a, c: c["plain"]["memory_jaccard"]["4"])]
    conds = (("ANCHOR-0", "anchors at frame 0 (before gust)", "#08519c"),
             ("ANCHOR-29", "anchors at frame 29 (after gust)", "#d94801"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [6, 5]})
    for ax, (scope, title) in zip(axes, (("occluded_ripe", "(a) FruitOccludedRipe (foliage occlusion)"),
                                         ("count", "(b) FruitCount (no foliage occlusion, control)"))):
        ms = metrics if scope == "occluded_ripe" else [m for m in metrics if not m[0].startswith("hidden")]
        x = np.arange(len(ms))
        for j, (src, label, col) in enumerate(conds):
            a, c = res[f"{scope}/{src}/anchors"], res[f"{scope}/{src}"]
            ys = [f(a, c) for _, f in ms]
            bars = ax.bar(x + (j - 0.5) * 0.38, ys, 0.36, color=col, label=label)
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y + 0.02, f"{y:.2f}", ha="center", fontsize=5.5)
        ax.set_xticks(x, [m[0] for m in ms], fontsize=6)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("score")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_anchor_timing")


def fig6_events():
    res = load("event_detection.json")
    if res is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    ax = axes[0]
    for split, ls in (("train", "-"), ("val", "--")):
        rs = sorted((float(k.split("=")[1]), v) for k, v in res["radius"].items() if k.startswith(split))
        ax.plot([r * 100 for r, _ in rs], [v["f1"] for _, v in rs], ls, marker="o", ms=2.5, color="k", label=f"{split} F1")
        ax.plot([r * 100 for r, _ in rs], [v["count_exact"] for _, v in rs], ls, marker="s", ms=2.5, color="tab:blue",
                label=f"{split} count exact")
    ax.axvline(3, color="tab:red", lw=0.6, ls=":")
    ax.set(xlabel="radius r (cm)", ylabel="score", title="(a) radius sensitivity", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=5.5)
    c = res["corruption"]
    det_style = {"plain": ("tab:gray", "o"), "median5": ("tab:orange", "s"), "hysteresis": ("tab:blue", "^")}

    def curve(ax, kind, scale, key="f1", sub=None):
        for det, (col, m) in det_style.items():
            pts = sorted((float(k.split("/")[2]), v) for k, v in c.items() if k.startswith(f"{det}/{kind}/"))
            pts = [(0.0, c[f"{det}/clean/0"])] + pts
            ys = [v[key] if sub is None else v[key][sub] for _, v in pts]
            ax.plot([x * scale for x, _ in pts], ys, marker=m, ms=2.5, color=col, label=det)

    curve(axes[1], "gauss", 100)
    axes[1].set(xlabel="white noise σ (cm)", title="(b) white noise", ylim=(0, 1.05))
    axes[1].legend(frameon=False, fontsize=5.5)
    for kind, ls in (("dropout", "-"), ("burst", "--")):
        for det, (col, m) in det_style.items():
            pts = sorted((float(k.split("/")[2]), v) for k, v in c.items() if k.startswith(f"{det}/{kind}/"))
            pts = [(0.0, c[f"{det}/clean/0"])] + pts
            axes[2].plot([x * 100 for x, _ in pts], [v["f1"] for _, v in pts], ls, marker=m, ms=2.5, color=col)
    axes[2].set(xlabel="missing frames (%)", title="(c) dropout (—), burst (--)", ylim=(0, 1.05))
    curve(axes[3], "gauss", 100, "keyframe_recall_w3", "4")
    axes[3].set(xlabel="white noise σ (cm)", title="(d) keyframe recall (k=4)", ylim=(0, 1.05))
    fig.tight_layout()
    save(fig, "fig6_event_detection")


def fig7_coverage():
    res = load("memory_coverage.json")
    if res is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    ax = axes[0]
    for name in ("U", "RANDOM", "PEKM-4", "PEKM-8", "PEKM-S-0.5"):
        pts = [(B, res[f"all/B={B}/{name}"]["key"]) for B in (8, 16, 32) if f"all/B={B}/{name}" in res]
        ax.plot(*zip(*pts), marker="o", ms=3, color=COLORS[name], label=LABELS[name])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xticks=[8, 16, 32], xticklabels=["8", "16", "32"], xlabel="memory budget B (frames)",
           ylabel="event coverage", title="(a) coverage vs budget", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=5.5)
    ax = axes[1]
    names = ("U", "RANDOM", "PEKM-4", "PEKM-8", "PEKM-S-1")
    width = 0.16
    for i, name in enumerate(names):
        e = res[f"all/B=16/{name}"]
        ax.bar(np.arange(3) + (i - 2) * width, [e["key_back0"], e["key_back1"], e["key_back2"]], width,
               color=COLORS[name], label=LABELS[name])
    ax.set(xticks=range(3), xticklabels=["last pick", "2nd last", "3rd last"], ylabel="covered",
           title="(b) coverage by recency (B=16)", ylim=(0, 1.08))
    ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax = axes[2]
    offsets = {"PEKM-4": (4, -7), "PEKM-S-0.5": (-6, 5), "PEKM-S-1": (4, 2)}
    for name in ("U", "RANDOM", "RECENT", "PEKM-2", "PEKM-4", "PEKM-8", "PEKM-S-0.5", "PEKM-S-1"):
        e = res[f"all/B=16/{name}"]
        ax.scatter(e["gap"]["median"], e["key"], color=COLORS[name], s=18, zorder=3)
        ax.annotate(LABELS[name], (e["gap"]["median"], e["key"]), fontsize=5.5, xytext=offsets.get(name, (4, -2)),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xticks=[10, 20, 50, 100, 200], xticklabels=["10", "20", "50", "100", "200"],
           xlabel="median gap between memory frames", ylabel="event coverage",
           title="(c) coverage vs temporal spread (B=16)", ylim=(0, 1.05))
    fig.tight_layout()
    save(fig, "fig7_memory_coverage")


def fig_theory():
    """Coverage of completed picks vs decision time: uniform memory (measured and predicted) and PEKM."""
    res = load("coverage_theory.json")
    if res is None:
        return
    bins = list(res["B=16"]["by_decision_time"])
    labels = ["<200", "200–400", "400–600", "≥600"][:len(bins)]
    x = np.arange(len(bins))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    e = res["B=16"]["by_decision_time"]
    ax = axes[0]
    ax.plot(x, [e[b]["U_measured"] for b in bins], "o-", color=COLORS["U"], ms=3, label="Uniform (measured)")
    ax.plot(x, [e[b]["U_predicted"] for b in bins], "--", color="k", lw=0.9, label="Uniform (predicted)")
    for name in ("PEKM-4", "PEKM-8"):
        ax.plot(x, [e[b][name] for b in bins], "o-", color=COLORS[name], ms=3, label=LABELS[name])
    ax.set(xticks=x, xticklabels=labels, xlabel="decision time t (frames)", ylabel="pick coverage",
           title="(a) all completed picks (B=16)", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax = axes[1]
    ax.plot(x, [e[b]["U_last_pick"] for b in bins], "o-", color=COLORS["U"], ms=3, label="Uniform")
    ax.plot(x, [e[b]["PEKM-4_last_pick"] for b in bins], "o-", color=COLORS["PEKM-4"], ms=3, label="PEKM k=4, 8")
    ax.set(xticks=x, xticklabels=labels, xlabel="decision time t (frames)", ylabel="coverage of the last pick",
           title="(b) most recent pick (B=16)", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax = axes[2]
    markers = {"B=8": "s", "B=16": "o", "B=32": "^"}
    for key, mk in markers.items():
        pts = res[key]["by_decision_time"]
        ax.scatter([v["U_predicted"] for v in pts.values()], [v["U_measured"] for v in pts.values()], marker=mk,
                   s=16, color=COLORS["U"], label=key, zorder=3)
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.7)
    ax.set(xlabel="predicted  min(1, (B−1)|S|/t)", ylabel="measured", title="(c) uniform memory: model vs data",
           xlim=(0, 1.05), ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_coverage_theory")


SAM_METHODS = {"HSV": ("HSV", "#7f7f7f", "s"), "SAM-AUTO/vit_b": ("SAM-AUTO B", "#bcbd22", "^"),
               "SAM-AUTO/vit_h": ("SAM-AUTO H", "#8c564b", "^"), "SAM-FDM/vit_b": ("SAM-FDM B", "#9ecae1", "o"),
               "SAM-FDM/vit_l": ("SAM-FDM L", "#3182bd", "o"), "SAM-FDM/vit_h": ("SAM-FDM H", "#08519c", "o"),
               "SAM-GTBOX/vit_h": ("GT box H", "#d94801", "*")}


def fig8_sam(qualitative):
    res = load("sam_segmentation.json")
    if res is not None and "SAM-FDM/vit_h/front" in res:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1, 1.3, 1]})
        ax = axes[0]
        for m, (label, col, mk) in SAM_METHODS.items():
            e = res[f"{m}/front"]
            if e["ms_per_image"] is None:
                continue
            ax.scatter(e["ms_per_image"], e["mean_iou_tp"], marker=mk, color=col, s=22, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set(xticks=[1, 10, 100, 1000], xticklabels=["1", "10", "100", "1000"], xlabel="time per image (ms, L20)",
               ylabel="mask IoU (matched)", title="(a) mask quality vs time (front)", ylim=(0.94, 0.985))
        ax = axes[1]
        bins = ("vf>=0.75", "0.25<=vf<0.75", "vf<0.25")
        width = 0.12
        for i, (m, (label, col, _)) in enumerate(SAM_METHODS.items()):
            rb = res[f"{m}/front"]["recall_by_visibility"]
            ax.bar(np.arange(3) + (i - 3) * width, [rb[b][0] for b in bins], width, color=col)
        ax.set(xticks=range(3), xticklabels=["≥75%", "25–75%", "<25%"], xlabel="visible fraction of the fruit",
               ylabel="recall", title="(b) recall by visibility (front)", ylim=(0, 1.08))
        ax = axes[2]
        conds = ("", "@bright", "@dark", "@warm")
        for i, m in enumerate(("HSV", "SAM-FDM/vit_h", "SAM-GTBOX/vit_h")):
            ax.bar(np.arange(4) + (i - 1) * 0.26, [res[f"{m}{c}/all"]["f1"] for c in conds], 0.26,
                   color=SAM_METHODS[m][1])
        ax.set(xticks=range(4), xticklabels=["nominal", "sun", "dusk", "warm"], ylabel="instance F1@0.5",
               title="(c) illumination shifts", ylim=(0, 1.08))
        handles = [matplotlib.lines.Line2D([], [], color=col, marker=mk, ls="", ms=5, label=label)
                   for label, col, mk in SAM_METHODS.values()]
        fig.tight_layout()
        fig.legend(handles=handles, frameon=False, fontsize=6, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        save(fig, "fig8b_sam_quantitative")
    if not qualitative:
        return
    import perception as P
    from exp_sam_segmentation import frames
    sam = P.SamFruit("vit_h")
    thr, prompt = P.load_thresholds(), P.load_prompt()
    path = ECM / "data/perception_fruit/seg/val_a/occluded_ripe_ep0.npz"
    picks = [fr for fr in frames(path) if (fr["cam"] == "front" and fr["t"] in (0, 40)) or
             (fr["cam"] == "wrist" and fr["t"] == 80)]
    cache_path = OUT / "perception_anchors.pkl"
    anchors = pickle.loads(cache_path.read_bytes())[("occluded_ripe", 0)]["ANCHOR-0"] if cache_path.exists() else []
    fig, axes = plt.subplots(len(picks), 4, figsize=(7.0, 1.8 * len(picks)))
    for row, fr in zip(np.atleast_2d(axes), picks):
        rgb = fr["rgb"]
        gt = [{"mask": fr["seg"] == fid, "cls": P.RIPE if fr["ripe"][k] else P.UNRIPE}
              for k, fid in enumerate(fr["fruit_ids"]) if (fr["seg"] == fid).any()]
        sam.set_image(rgb)
        layers = {"RGB": None, "ground truth": gt,
                  "HSV": P.hsv_instances(rgb, thr, 10), "SAM-FDM (ViT-H)": sam.fdm(rgb, thr, 10, **prompt)}
        for ax, (title, layer) in zip(row, layers.items()):
            ax.imshow(rgb)
            for inst in layer or []:
                col = (1, 0, 0) if inst["cls"] == P.RIPE else (1, 1, 0)
                ax.contour(inst["mask"], levels=[0.5], colors=[col], linewidths=0.8)
            if title.startswith("SAM-FDM") and fr["cam"] == "front" and fr["t"] > 0:
                for a in anchors:
                    uv, z = P.project(a["p"], fr["K"], fr["E"])
                    if uv is not None:
                        ax.add_patch(plt.Circle(uv, fr["K"][0, 0] * a["r"] / z, fill=False, ls="--", lw=0.8,
                                                color="tab:red" if a["ripe"] else "gold"))
                title = "SAM-FDM + frame-0 anchors"
            ax.set_title(f"{title} · {fr['cam']} t={fr['t']}", fontsize=6)
            ax.axis("off")
    fig.tight_layout()
    save(fig, "fig8a_sam_qualitative")


def fig_e6():
    """GT vs perceived distance on one FruitOccludedRipe episode."""
    cache_path = OUT / "perception_anchors.pkl"
    if not cache_path.exists():
        return
    import exp_perception_distance as E
    import perception as P
    cache = pickle.loads(cache_path.read_bytes())
    ep = next(e for e in episodes("val_a", ("occluded_ripe",)) if e.episode == 0)
    anchors = cache[(ep.task, ep.episode)]
    rgb, depth, K, Ex = E.load_front(ep)
    n = min(len(rgb), len(ep.rows))
    tcp = np.array([r["tcp_pos"] for r in ep.rows[:n]])
    hold = E.holding(ep.rows[:n])
    curves = {
        "ground truth": ep.distance[:n],
        "SAM-FDM (anchors t=0, depth + grasp)": E.anchor_distance(anchors["ANCHOR-0"], depth[:n], tcp, K, Ex, False, hold)[0],
        "depth-only vacancy": E.anchor_distance(anchors["ANCHOR-0"], depth[:n], tcp, K, Ex, False)[0],
        "anchors after gust (t=29)": E.anchor_distance(anchors["ANCHOR-29"], depth[:n], tcp, K, Ex, False, hold)[0],
        "per-frame detection (no anchors)": E.perframe_distance(rgb[:n], depth[:n], tcp, K, Ex, P.load_thresholds(), False),
    }
    styles = [("k", "-", 1.6), ("tab:blue", "-", 0.9), ("tab:orange", "--", 0.9), ("tab:purple", ":", 1.1),
              ("tab:gray", "-", 0.6)]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    for (name, d), (col, ls, lw) in zip(curves.items(), styles):
        ax.plot(np.minimum(d, 0.4), ls, color=col, lw=lw, label=name)
    for tt in ep.detach_times.values():
        ax.axvline(tt, color="tab:green", lw=0.6, ls=":")
    ax.axhline(R, color="tab:red", lw=0.6, ls="--")
    ax.set(xlabel="frame", ylabel="d_t (m, clipped at 0.4)",
           title=f"FruitOccludedRipe val ep{ep.episode}: simulator vs perceived distance (green: detachments)")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "fig_e6_perceived_distance")


FIGS = {"fig2": fig2_scene, "fig4": fig4_timeline, "fig6": fig6_events, "fig7": fig7_coverage, "fig_e6": fig_e6,
        "fig_theory": fig_theory, "fig_anchor": fig_anchor_timing, "figS_timeline": fig_s_timelines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--sam", action="store_true")
    args = ap.parse_args()
    for name, fn in FIGS.items():
        if args.only is None or name in args.only:
            fn()
    if args.only is None or "fig8" in args.only:
        fig8_sam(args.sam)


if __name__ == "__main__":
    main()
