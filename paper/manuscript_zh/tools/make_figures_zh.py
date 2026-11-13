#!/usr/bin/env python3
"""Figures of the Chinese manuscript. Real panels come from paper/results/*.json and the recorded v1 validation
demonstrations; panels drawn from values that are still `ideal` in data/values.yaml carry a watermark.

    robomme_benchmark/.venv/bin/python ecm_vla/paper/manuscript_zh/tools/make_figures_zh.py [--only NAME ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "experiments"))

import make_figures as mf  # noqa: E402  (sets the shared rcParams and paths)
import matplotlib  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from memory_management.event_keyframes import KeyframeConfig, detect_events, select  # noqa: E402

FONT_DIR = Path("/media/fulei/fuleivla/tools/.TinyTeX/texmf-dist/fonts/opentype/public/fandol")
for name in ("FandolHei-Regular.otf", "FandolHei-Bold.otf", "FandolSong-Regular.otf"):
    font_manager.fontManager.addfont(str(FONT_DIR / name))
plt.rcParams.update({"font.family": ["FandolHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "pdf.fonttype": 3, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7})

FIG = ROOT / "figures"
RAW = yaml.safe_load((ROOT / "data/values.yaml").read_text(encoding="utf-8"))
C = {"K0": "#8c8c8c", "K4": "#6baed6", "K8": "#08519c", "OR": "#31a354", "FDM": "#fd8d3c", "GRIP": "#bcbd22",
     "PEDET": "#08519c", "GEO": "#9e9ac8"}
WATERMARK = "示意数据·待替换为实测"


def val(key):
    return RAW[key][0]


def is_ideal(*keys):
    return any(RAW[k][1] == "ideal" for k in keys)


def mark(ax, keys):
    if is_ideal(*keys):
        ax.text(0.5, 0.5, WATERMARK, transform=ax.transAxes, ha="center", va="center", rotation=22, fontsize=11,
                color="red", alpha=0.22, zorder=10, fontweight="bold")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(FIG / f"{name}.pdf")


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def fig_scene():
    count = next(e for e in mf.episodes("val_a", ("count",)) if e.episode == 0)
    occ = next(e for e in mf.episodes("val_a", ("occluded_ripe",)) if e.episode == 0)
    c_first = min(count.detach_times.values()) - 4
    o_first = min(occ.detach_times.values()) - 4
    panels = [
        (mf.h5_frames(count, [0])[0], "(a) 计数任务·前置相机·t=0"),
        (mf.h5_frames(count, [c_first + 60])[0], f"(b) 计数任务·前置相机·t={c_first + 60}"),
        (mf.h5_frames(count, [c_first], "wrist")[0], "(c) 计数任务·腕部相机·接触"),
        (mf.h5_frames(occ, [0])[0], "(d) 遮挡任务·t=0（阵风前）"),
        (mf.h5_frames(occ, [29])[0], "(e) 遮挡任务·t=29（阵风后）"),
        (mf.h5_frames(occ, [o_first], "wrist")[0], "(f) 遮挡任务·腕部相机·接触"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.5))
    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=7.5)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    save(fig, "zh_scene")


def fig_theory():
    res = mf.load("coverage_theory.json")
    bins = list(res["B=16"]["by_decision_time"])
    labels = ["<200", "200–400", "400–600", "≥600"][:len(bins)]
    x = np.arange(len(bins))
    e = res["B=16"]["by_decision_time"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    ax = axes[0]
    ax.plot(x, [e[b]["U_measured"] for b in bins], "o-", color=C["K0"], ms=3, label="均匀记忆（实测）")
    ax.plot(x, [e[b]["U_predicted"] for b in bins], "--", color="k", lw=0.9, label="均匀记忆（式(3)预测）")
    ax.plot(x, [e[b]["PEKM-4"] for b in bins], "o-", color=C["K4"], ms=3, label="PEKM k=4")
    ax.plot(x, [e[b]["PEKM-8"] for b in bins], "o-", color=C["K8"], ms=3, label="PEKM k=8")
    ax.set(xticks=x, xticklabels=labels, xlabel="决策时刻 t（帧）", ylabel="已完成采摘的覆盖率",
           title="(a) 全部已完成采摘（B=16）", ylim=(0, 1.05))
    ax.legend(frameon=False, loc="lower left")
    ax = axes[1]
    ax.plot(x, [e[b]["U_last_pick"] for b in bins], "o-", color=C["K0"], ms=3, label="均匀记忆")
    ax.plot(x, [e[b]["PEKM-4_last_pick"] for b in bins], "o-", color=C["K8"], ms=3, label="PEKM k=4 / 8")
    ax.set(xticks=x, xticklabels=labels, xlabel="决策时刻 t（帧）", ylabel="最近一次采摘的覆盖率",
           title="(b) 最近一次采摘（B=16）", ylim=(0, 1.05))
    ax.legend(frameon=False, loc="lower left")
    ax = axes[2]
    for key, mk in {"B=8": "s", "B=16": "o", "B=32": "^"}.items():
        pts = res[key]["by_decision_time"]
        ax.scatter([v["U_predicted"] for v in pts.values()], [v["U_measured"] for v in pts.values()], marker=mk,
                   s=16, color=C["K0"], label=key, zorder=3)
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.7)
    ax.set(xlabel="预测值 min(1,(B−1)|S|/t)", ylabel="实测值", title="(c) 均匀记忆：理论与实测",
           xlim=(0, 1.05), ylim=(0, 1.05))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "zh_coverage_theory")


def fig_timeline():
    ep = next(e for e in mf.episodes("val_a", ("count",)) if e.episode == 0)
    d = ep.distance
    t = (len(d) - 1) // 16 * 16
    r, tol = mf.R, mf.TOL
    mems = {"均匀 K0": select(d, t, KeyframeConfig(frames=16, events=0))[0],
            "PEKM k=4": select(d, t, KeyframeConfig(frames=16, events=4, radius=r, tol=tol))[0],
            "PEKM k=8": select(d, t, KeyframeConfig(frames=16, events=8, radius=r, tol=tol))[0]}
    colors = [C["K0"], C["K4"], C["K8"]]
    events = detect_events(d[:t + 1], r, tol)
    done = [e for e in events if e["release"] is not None]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.0, 3.1), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]})
    ax.plot(np.minimum(d[:t + 1], 0.35), color="k", lw=0.9)
    ax.axhline(r, color="tab:red", ls="--", lw=0.7)
    ax.text(t, r + 0.012, "r = 3 cm", color="tab:red", ha="right", fontsize=6.5)
    for i, e in enumerate(events):
        end = e["release"] if e["release"] is not None else e["end"]
        for a in (ax, bx):
            a.axvspan(e["start"] - 3, end + 3, color="tab:red", alpha=0.12, lw=0)
        ax.plot(e["contact"], d[e["contact"]], "v", color="tab:red", ms=4)
        ax.text((e["start"] + end) / 2, 0.375, f"第{i + 1}次采摘", ha="center", fontsize=6.5, color="tab:red")
    for tt in ep.detach_times.values():
        ax.axvline(tt, color="tab:green", lw=0.6, ls=":")
    ax.set_ylim(-0.01, 0.42)
    ax.set_ylabel("$d_t$ / m（截断）")
    ax.set_title("(a) 计数任务验证回合 0：TCP 到最近可采果实的距离", loc="left")
    for row, (name, frames) in enumerate(mems.items()):
        frames = np.asarray(frames)
        key = np.zeros(len(frames), bool)
        hits = []
        for e in done:
            near = (np.abs(frames - e["contact"]) <= 3) | (np.abs(frames - e["release"]) <= 3)
            key |= near
            hits.append(bool(near.any()))
        bx.scatter(frames[~key], np.full((~key).sum(), row), marker="|", s=70, color=colors[row], alpha=0.35)
        bx.scatter(frames[key], np.full(key.sum(), row), marker="|", s=110, lw=2.0, color=colors[row])
        for e, hit in zip(done, hits):
            bx.text((e["start"] + e["release"]) / 2, row - 0.38, "✓" if hit else "✗", ha="center", va="center",
                    fontsize=7.5, color="tab:green" if hit else "tab:red", fontweight="bold", family="DejaVu Sans")
        bx.text(t + 18, row, f"保留 {sum(hits)}/{len(done)} 次" + ("" if all(hits) else "，计数丢失"),
                va="center", fontsize=6.5, color="k" if all(hits) else "tab:red")
    bx.set_yticks(range(len(mems)), list(mems))
    bx.set_ylim(-0.8, len(mems) - 0.4)
    bx.set_xlim(-10, t + 135)
    bx.set_xlabel("帧")
    bx.set_title(f"(b) 决策时刻 t={t} 各记忆保存的 16 帧（粗线：位于采摘关键帧 ±3 帧内）", loc="left")
    bx.invert_yaxis()
    fig.tight_layout()
    save(fig, "zh_timeline")


def fig_events():
    res = mf.load("event_detection.json")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95))
    ax = axes[0]
    for split, ls, lab in (("train", "-", "训练集"), ("val", "--", "验证集")):
        rs = sorted((float(k.split("=")[1]), v) for k, v in res["radius"].items() if k.startswith(split))
        ax.plot([q * 100 for q, _ in rs], [v["f1"] for _, v in rs], ls, marker="o", ms=2.5, color="k", label=f"{lab} F1")
        ax.plot([q * 100 for q, _ in rs], [v["count_exact"] for _, v in rs], ls, marker="s", ms=2.5,
                color="tab:blue", label=f"{lab} 计数正确")
    ax.axvline(3, color="tab:red", lw=0.6, ls=":")
    ax.set(xlabel="半径 r（cm）", ylabel="得分", title="(a) 半径敏感性（几何事件）", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=5.5)
    c = res["corruption"]
    for det, col, mk, lab in (("plain", "tab:gray", "o", "阈值"), ("median5", "tab:orange", "s", "中值滤波"),
                              ("hysteresis", "tab:blue", "^", "迟滞")):
        pts = sorted((float(k.split("/")[2]), v) for k, v in c.items() if k.startswith(f"{det}/gauss/"))
        pts = [(0.0, c[f"{det}/clean/0"])] + pts
        axes[1].plot([q * 100 for q, _ in pts], [v["f1"] for _, v in pts], marker=mk, ms=2.5, color=col, label=lab)
    axes[1].set(xlabel="距离白噪声 σ（cm）", title="(b) 距离噪声（几何事件）", ylim=(0, 1.05))
    axes[1].legend(frameon=False, fontsize=5.5)
    ax = axes[2]
    xs = val("fig_light_x")
    ax.plot(xs, val("fig_light_pedet"), "o-", ms=3, color=C["PEDET"], label="PEDet（本文）")
    ax.plot(xs, val("fig_light_fdm"), "s-", ms=3, color=C["FDM"], label="SAM-FDM")
    ax.plot(xs, val("fig_light_grip"), "^--", ms=3, color=C["GRIP"], label="夹爪启发式")
    ax.set(xlabel="图像亮度增益", ylabel="事件 F1", title="(c) 光照扰动（仿真）", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=5.5, loc="lower center")
    mark(ax, ["fig_light_pedet", "fig_light_fdm"])
    ax = axes[3]
    xs = val("fig_hsl_x")
    ax.plot(xs, val("fig_hsl_pedet"), "o-", ms=3, color=C["PEDET"], label="PEDet+事后确认自训练")
    ax.plot(xs, val("fig_sup_ref"), "-.", color="k", lw=0.8, label="20 回合人工标注（参照）")
    ax.plot(xs, val("fig_hsl_fdm"), "--", color=C["FDM"], label="SAM-FDM")
    ax.plot(xs, val("fig_hsl_grip"), ":", color=C["GRIP"], label="夹爪启发式")
    ax.set(xlabel="无标注真机回合数", title="(d) 真机检测（人工标注真值）", ylim=(0.7, 1.0))
    ax.legend(frameon=False, fontsize=5, loc="lower right")
    mark(ax, ["fig_hsl_pedet"])
    fig.tight_layout()
    save(fig, "zh_event_detection")


def _uniform_from(pool, m):
    pool = np.asarray(sorted(pool))
    if m <= 0:
        return []
    if m == 1:
        return [pool[-1]]
    return list(pool[np.floor(np.arange(m) * (len(pool) - 1) / (m - 1)).astype(int)])


def _covered(mem, events, w=3):
    mem = np.asarray(mem)
    return np.array([(np.abs(mem - c) <= w).any() or (np.abs(mem - d) <= w).any() for c, d in events])


def _capacity_trial(n, rng, budget=16, mw=2, cap=4, smax=3):
    """Picks retained at the last decision of a synthetic episode with n picks (gaps 110-190 frames)."""
    gaps = rng.uniform(110, 190, size=n)
    starts = 100 + np.cumsum(gaps) - gaps[0]
    events = [(int(x), int(x) + 4) for x in starts]
    t = events[-1][1] + 60
    frames = set(range(t + 1))
    uniform = _uniform_from(frames, budget)
    flat = []
    for c, d in reversed(events):
        flat += [f for f in (c, d) if len(flat) < 8]
    flat += _uniform_from(frames - set(flat), budget - len(flat))
    key = {0} | {f for c, d in events[-mw:] for f in (c, d)}
    older = events[:-mw][::-1][:cap * smax]
    slots = int(np.ceil(len(older) / cap))
    hem = list(key) + _uniform_from(frames - key, budget - len(key) - slots)
    hem_cov = _covered(hem, events) | np.array([e in older for e in events])
    return _covered(uniform, events).sum(), _covered(flat, events).sum(), hem_cov.sum()


def fig_capacity():
    rng = np.random.default_rng(0)
    ns = np.arange(1, 21)
    res = np.array([np.mean([_capacity_trial(n, rng) for _ in range(300)], axis=0) for n in ns])
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.plot(ns, ns, ":", color="k", lw=0.8, label="全部记住")
    ax.plot(ns, res[:, 0], "o-", ms=2.5, color=C["K0"], label="均匀记忆")
    ax.plot(ns, res[:, 1], "s-", ms=2.5, color=C["K4"], label="平坦关键帧 PEKM k=8")
    ax.plot(ns, res[:, 2], "^-", ms=2.5, color=C["K8"], label="分层事件记忆 HEM")
    ax.set(xlabel="已完成采摘次数 n", ylabel="记忆中保留的采摘次数", xlim=(0.5, 20.5), ylim=(0, 16),
           xticks=[1, 4, 8, 12, 16, 20], yticks=[0, 4, 8, 12, 16], title="16 帧预算下的事件容量（合成事件序列）")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "zh_capacity")
    for n, r in zip(ns, res):
        print(f"  n={n:2d} uniform={r[0]:.2f} flat8={r[1]:.2f} hem={r[2]:.2f}")


def fig_sim_main():
    n = 150
    conds = ("K0", "K4", "K8", "HEM")
    cols = {"K0": C["K0"], "K4": "#9ecae1", "K8": C["K4"], "HEM": C["K8"]}
    names = {"K0": "K0 均匀", "K4": "PEKM k=4", "K8": "PEKM k=8", "HEM": "HEM（本文）"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    ax = axes[0]
    groups = [("计数 v2\n（事件记忆）", "c"), ("遮挡 v2\n（场景记忆）", "o")]
    width = 0.2
    for g, (label, pre) in enumerate(groups):
        for i, cond in enumerate(conds):
            k = val(f"{pre}_{cond.lower()}")[0]
            lo, hi = wilson(k, n)
            p = k / n
            x = g + (i - 1.5) * width
            ax.bar(x, p * 100, width, color=cols[cond], label=names[cond] if g == 0 else None)
            ax.errorbar(x, p * 100, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="k", lw=0.7, capsize=1.5)
    ax.set(xticks=[0, 1], xticklabels=[g[0] for g in groups], ylabel="成功率（%）", ylim=(0, 90),
           title="(a) 闭环成功率（95% Wilson 区间）")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=5.5)
    mark(ax, ["c_k0", "o_k0"])
    ax = axes[1]
    cats = ["成功", "多摘", "少摘后停止", "掉果", "操作失败"]
    ccols = ["#31a354", "#de2d26", "#fc9272", "#9e9ac8", "#bdbdbd"]
    data = np.array([val(f"c_{c.lower()}") for c in conds], float) / n * 100
    bottoms = np.zeros(len(conds))
    for j, (cat, col) in enumerate(zip(cats, ccols)):
        ax.bar(range(len(conds)), data[:, j], 0.6, bottom=bottoms, color=col, label=cat)
        bottoms += data[:, j]
    ax.set(xticks=range(len(conds)), xticklabels=["K0", "k=4", "k=8", "HEM"], ylabel="占比（%）", ylim=(0, 100),
           title="(b) 计数 v2 结果分解")
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    mark(ax, ["c_k0"])
    ax = axes[2]
    xs = np.arange(3)
    for key, cond in (("lN_k0", "K0"), ("lN_k8", "K8"), ("lN_hem", "HEM")):
        ax.plot(xs, val(key), "o-", ms=3, color=cols[cond], label=names[cond])
    ax.set(xticks=xs, xticklabels=["N=6–7", "N=8–9", "N=10–12"], ylabel="成功率（%）", ylim=(0, 75),
           title="(c) 多株长时配额任务")
    ax.legend(frameon=False, fontsize=5.5)
    mark(ax, ["lN_k0"])
    fig.tight_layout()
    save(fig, "zh_sim_main")


def fig_real():
    n = 30
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    bars = [("计数\nK0", "rc_k0_s", C["K0"]), ("计数\nk=8", "rc_k8_s", C["K4"]), ("计数\nHEM", "rc_hem_s", C["K8"]),
            ("遮挡\nK0", "ro_k0_s", C["K0"]), ("遮挡\nHEM", "ro_hem_s", C["K8"]),
            ("暖光\nHEM+PEDet", "rl_pedet_s", C["K8"]), ("暖光\nHEM+FDM", "rl_fdm_s", C["FDM"])]
    for i, (label, key, col) in enumerate(bars):
        s = float(val(key))
        k = round(s / 100 * n)
        lo, hi = wilson(k, n)
        ax.bar(i, s, 0.65, color=col)
        ax.errorbar(i, s, yerr=[[s - 100 * lo], [100 * hi - s]], color="k", lw=0.7, capsize=1.5)
        ax.text(i, 2, f"{k}/{n}", ha="center", fontsize=5.5, color="white")
    ax.set(xticks=range(len(bars)), ylabel="成功率（%）", ylim=(0, 100), title="(a) 真机闭环成功率（每条件 30 次）")
    ax.set_xticklabels([b[0] for b in bars], fontsize=5.5)
    mark(ax, ["rc_k0_s"])
    ax = axes[1]
    cats = ["成功", "多摘", "少摘后停止", "掉果", "操作失败"]
    cols = ["#31a354", "#de2d26", "#fc9272", "#9e9ac8", "#bdbdbd"]
    data = np.array([val("rc_k0"), val("rc_k8"), val("rc_hem")], float) / n * 100
    bottoms = np.zeros(3)
    for j, (cat, col) in enumerate(zip(cats, cols)):
        ax.bar(range(3), data[:, j], 0.6, bottom=bottoms, color=col, label=cat)
        bottoms += data[:, j]
    ax.set(xticks=range(3), xticklabels=["K0", "PEKM k=8", "HEM"], ylabel="占比（%）", ylim=(0, 100),
           title="(b) 真机计数任务结果分解")
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    mark(ax, ["rc_k0"])
    fig.tight_layout()
    save(fig, "zh_real")


FIGURES = {"scene": fig_scene, "theory": fig_theory, "timeline": fig_timeline, "capacity": fig_capacity, "events": fig_events, "sim_main": fig_sim_main, "real": fig_real}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(FIGURES))
    args = ap.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
