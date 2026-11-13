#!/usr/bin/env python3
"""Offline preview of the event memories on a recorded episode (no model, no outcomes).

For every decision of the attempt directory, the ground-truth distance stand-in (event_memory.gt_distance
on raw_frames.jsonl) gives d_t for frames 0..F; the official U and the event memories EV / EVU / EVR / EVA
are recomputed exactly as the server would. Prints per-decision medians and plots one decision.

    python3 scripts/preview_event_memory.py results/memory_ablation_20261024/episodes/U0/BinFill/ep00_seed7 \
        --out results/event_memory_preview/BinFill_ep00_seed7.png
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ECM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECM / "src"))
from memory_management.validation import event_memory as em  # noqa: E402
from memory_management.validation.partition import partition  # noqa: E402
from memory_management.validation.teacher_probe import stable_seed  # noqa: E402


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt", type=Path, help="episode attempt directory with raw_frames.jsonl")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--d0", type=float, default=0.0)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--decision", type=int, default=-1, help="decision to plot (default: the last)")
    parser.add_argument("--out", type=Path, help="PNG path for the plot")
    opts = parser.parse_args()
    attempt = next(p.parent for p in sorted(opts.attempt.rglob("raw_frames.jsonl")))
    result = json.loads((attempt / "result.json").read_text()) if (attempt / "result.json").exists() else {}
    rows = read_jsonl(attempt / "raw_frames.jsonl")
    task = result.get("task") or rows[0].get("task") or attempt.parts[-2]
    decisions = [int(r["available_max_frame_id"]) for r in read_jsonl(attempt / "policy_decisions.jsonl")]
    cfg = em.EventMemoryConfig(alpha=opts.alpha, tau=opts.tau, d0=opts.d0).for_task(task)
    distances = np.asarray([em.gt_distance(task, r.get("gt") or {}) for r in rows])
    per = []
    for k, max_id in enumerate(decisions):
        part = partition(max_id, opts.capacity)
        memories = em.strategy_memories(part, distances[:max_id + 1], cfg,
                                        stable_seed(task, result.get("episode_id"), result.get("model_seed"), k, "evr"))
        per.append((max_id, list(part.uniform), memories))
    print(f"{task} {attempt}: {len(rows)} frames, exec_start {rows[0]['exec_start_idx']}, {len(decisions)} decisions, "
          f"distance known on {np.isfinite(distances).mean():.0%} of frames, median {np.nanmedian(distances):.3f} m")
    print("per-decision medians (frames changed vs U / frames within tau of the target):")
    print(f"  U    near {statistics.median(em.memory_stats(u, u, distances, cfg.tau)['near'] for _, u, _ in per)}")
    for name in em.STRATEGIES:
        stats = [em.memory_stats(m[name], u, distances, cfg.tau) for _, u, m in per]
        print(f"  {name:4s} changed {statistics.median(s['changed'] for s in stats)}  "
              f"near {statistics.median(s['near'] for s in stats)}")
    if opts.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        max_id, uniform, memories = per[opts.decision]
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(np.arange(max_id + 1), distances[:max_id + 1], color="0.35", lw=1, label="GT distance to target (m)")
        ax.axhline(cfg.tau, color="0.6", ls=":", lw=1, label=f"tau = {cfg.tau} m")
        top = np.nanmax(distances[:max_id + 1])
        for level, (name, frames, color) in enumerate((("U (official)", uniform, "tab:blue"),
                                                       (f"EV (alpha={cfg.alpha})", memories["EV"], "tab:red"),
                                                       ("EVU (same sampler, no event)", memories["EVU"], "tab:purple"),
                                                       ("EVR (matched random)", memories["EVR"], "tab:gray"),
                                                       ("EVA (anti-event)", memories["EVA"], "tab:green"))):
            ax.scatter(frames, np.full(len(frames), top * (1.08 + 0.08 * level)), marker="|", s=120,
                       color=color, label=name)
        if rows[0]["exec_start_idx"]:
            ax.axvline(rows[0]["exec_start_idx"], color="k", lw=0.8, ls="--", label="execution starts")
        ax.set_xlabel("frame")
        ax.set_ylabel("distance (m)")
        ax.set_title(f"{task}: memories at decision {opts.decision % len(per)} (F = {max_id})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        fig.tight_layout()
        opts.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(opts.out, dpi=120)
        print(f"plot: {opts.out}")


if __name__ == "__main__":
    main()
