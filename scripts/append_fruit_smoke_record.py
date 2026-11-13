#!/usr/bin/env python3
"""Check the fruit training and closed-loop smoke runs and append their record to section 9 of the protocol.

Appends nothing and exits non-zero unless gate T1 of the training smoke passed and the closed-loop smoke
completed every block with every validity gate passed.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics

GATES = {"G3_errors": "G3", "G5_same_process": "G5", "GA_selection": "GA"}


def peaks(path):
    out = collections.defaultdict(int)
    for line in path.read_text().splitlines():
        try:
            gpu, mib = (int(x) for x in line.split(","))
        except ValueError:
            continue
        out[gpu] = max(out[gpu], mib)
    return dict(sorted(out.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--doc", type=Path, required=True)
    p.add_argument("--train-run", type=Path, required=True)
    p.add_argument("--eval-run", type=Path, required=True)
    p.add_argument("--gpu-log", type=Path, required=True)
    p.add_argument("--train-seconds", type=float, required=True)
    p.add_argument("--eval-seconds", type=float, required=True)
    p.add_argument("--formal-servers", type=int, default=3)
    a = p.parse_args()
    gate = json.loads((a.train_run / "gate_t1.json").read_text())
    analysis = json.loads((a.eval_run / "analysis.json").read_text())
    problems = []
    if not gate["passed"]:
        problems.append(f"training smoke gate T1 failed: {gate['arms']}")
    if not analysis["gates"] or not all(analysis["gates"].values()):
        problems.append(f"closed-loop smoke gates: {analysis['gates']}")
    if analysis["blocks"]["complete"] != analysis["blocks"]["planned"]:
        problems.append(f"closed-loop smoke blocks: {analysis['blocks']}")
    if problems:
        print("\n".join(problems))
        return 1
    losses = "，".join(f"{arm} {r['first_loss']:.4f} → {r['last_loss']:.4f}" for arm, r in gate["arms"].items())
    audit = analysis["diagnostics"]["selection_audit"]
    outcomes = collections.Counter()
    for conditions in analysis["per_condition"].values():
        for row in conditions.values():
            outcomes.update(row["outcomes"])
    seconds = [json.loads(r.read_text())["elapsed_seconds"] for r in (a.eval_run / "episodes").rglob("result.json")]
    median = statistics.median(seconds)
    hours = 900 * median / a.formal_servers / 3600
    memory = "，".join(f"GPU {g} {m / 1024:.1f} GB" for g, m in peaks(a.gpu_log).items())
    gates = "、".join(GATES.get(k, k) for k in analysis["gates"])
    items = [
        f"4. **训练冒烟**（`{a.train_run}`，100 步，5 步预热，三臂）：T1 通过，三个最终 checkpoint 带的归一化统计与采摘统计一致。"
        f"损失（第 0 步 → 最后一个区间均值）：{losses}。总用时 {a.train_seconds / 60:.0f} 分钟（k0、k4 并行，k8 随后，含装载和编译）。",
        f"5. **闭环冒烟**（`{a.eval_run}`，冒烟 checkpoint step 99，val episode 0、种子 7，两个任务 × 三个条件）：{gates} 全部通过；"
        f"选帧审计 {audit['decisions']} 个决策，不一致 {audit['mismatches']} 个。结局："
        f"{'，'.join(f'{k} {v}' for k, v in sorted(outcomes.items()))}（冒烟 checkpoint，不解读）。"
        f"各卡显存峰值（含同机其他进程）：{memory}。用时 {a.eval_seconds / 60:.0f} 分钟（含装载）。",
        f"6. **正式评测耗时估计**：冒烟里每条 episode 耗时中位数 {median:.0f} s（冒烟 checkpoint 多数跑满步数，接近上限）；"
        f"900 条、{a.formal_servers} 个服务进程并行，约 {hours:.1f} 小时以内。",
    ]
    text = a.doc.read_text().rstrip("\n") + "\n" + "\n".join(items) + "\n"
    a.doc.write_text(text)
    print("\n".join(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
