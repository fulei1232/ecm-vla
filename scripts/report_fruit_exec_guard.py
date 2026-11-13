#!/usr/bin/env python3
"""Stage-1 report from the common failure analyzer; stop at G0 without running a new gate."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from run_fruit_exec_guard import CONDITIONS, TASKS
from v_a_common import ECM, SIM_PYTHON, atomic_json, read_json, read_jsonl, timestamp


def frames(video, out):
    import cv2
    from PIL import Image, ImageDraw
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sheet = None
    for j in range(6):
        t = round((n-1)*j/5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {t}: {video}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((512, 440))
        if sheet is None:
            sheet = Image.new("RGB", (3*512, 2*470), "white")
        x, y = j % 3 * 512, j // 3 * 470
        sheet.paste(image, (x, y+25))
        ImageDraw.Draw(sheet).text((x+5, y+5), f"video frame {t}", fill="black")
    cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def score(summary, order):
    s = summary["all"]
    first, success = s["first_target_in_basket"]["numerator"], s["success"]["numerator"]
    return (first >= 20 and success >= 10, first, success,
            -s["wrist_deviation_executed"]["numerator"], -order)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    plan = read_json(out / "plan.json")
    summaries = {}
    for condition in CONDITIONS:
        analyzed = subprocess.run([sys.executable, str(ECM / "scripts/analyze_fruit_failure_modes.py"),
                        "--episodes-root", str(out / condition), "--pattern", "ep{e}",
                        "--episodes", "200-219", "--out", str(out / "analysis" / condition)])
        summaries[condition] = read_json(out / "analysis" / condition / "summary.json")
        if analyzed.returncode:
            if not plan.get("continue_guard_infeasible") or summaries[condition]["missing_expected_episodes"]:
                raise RuntimeError(f"incomplete or invalid analysis: {condition}")
            for row in read_jsonl(out / "analysis" / condition / "episodes.jsonl"):
                if row["engineering_error"]:
                    raw = read_json(Path(row["directory"]) / "result.json")
                    if not raw.get("joint_guard_infeasible") or len(row["errors"]) != 1:
                        raise RuntimeError(f"unexpected engineering error: {row['directory']}")
    best = max(CONDITIONS, key=lambda c: score(summaries[c], list(CONDITIONS).index(c)))
    s = summaries[best]["all"]
    g0 = s["first_target_in_basket"]["numerator"] >= 20 and s["success"]["numerator"] >= 10
    smokes = [read_json(out / label / task / "ep200/memory_audit.json")
              for label in ("smoke_A2", "smoke_A3") for task in TASKS]
    assert all(r["passed"] for r in smokes)
    # Every layout must have all five arms in one server process/session.
    sessions = []
    for task in TASKS:
        for episode in range(200, 220):
            values = {read_json(out / c / task / f"ep{episode}/result.json")["server_session"] for c in CONDITIONS}
            if len(values) != 1:
                raise RuntimeError(f"layout crosses server sessions: {task} ep{episode}")
            sessions.append({"task": task, "episode": episode, "server_session": next(iter(values))})
    archived = read_json(ECM / "reports/lowlevel_rescue_D0_20261102/summary.json")["analyses"]["dev_baseline"]
    before = {(r["task"], r["episode_id"]): r for r in read_jsonl(
        ECM / "results/fruit_lowlevel_rescue_D0_20261102/dev_baseline/episodes.jsonl")}
    a0 = read_jsonl(out / "analysis/A0/episodes.jsonl")
    changes = [{"task": r["task"], "episode_id": r["episode_id"],
                "archived_outcome": before[r["task"], r["episode_id"]]["outcome"], "A0_outcome": r["outcome"]}
               for r in a0 if r["outcome"] != before[r["task"], r["episode_id"]]["outcome"]]
    failures = []
    for path in out.rglob("result.json"):
        r = read_json(path)
        if r.get("error") or r.get("outcome") not in ("success", "fail", "timeout"):
            failures.append({"path": str(path), "result": r})
    result = {"phase": 1, "state": "awaiting_G0_confirmation", "updated_utc": timestamp(),
              "conditions": summaries, "best_condition": best, "G0_thresholds_met": g0,
              "G0": {"first_target_in_basket": s["first_target_in_basket"], "success": s["success"],
                     "required_first": {"numerator": 20, "denominator": 40},
                     "required_success": {"numerator": 10, "denominator": 40},
                     "requires_user_confirmation": True,
                     "next_if_confirmed": "one G1 on dev 220-259 with existing K0" if g0 else "phase 2 deterministic expert"},
              "memory_smokes": smokes, "paired_sessions": sessions, "failed_runs_including_superseded": failures,
              "A0_reproducibility": {"archived_summary": archived, "changed_outcomes": changes,
                                     "A0_summary": summaries["A0"]},
              "guard_stats": read_json(ECM / "reports/fruit_joint_guard_v1_20261102.json"),
              "guard_infeasible_rule": plan.get("guard_abort_rule"),
              "prior_failed_run": str(ECM / "results/fruit_exec_guard_dev_20261102") if plan.get("continue_guard_infeasible") else None}
    atomic_json(out / "summary.json", result)
    # Choose one identical layout, preferring a gain on the first target. Ties keep input order.
    baseline = {(r["task"], r["episode_id"]): r for r in a0}
    candidates = read_jsonl(out / "analysis" / best / "episodes.jsonl")
    sample = max(candidates, key=lambda r: (
        int(r["first_target_in_basket"]) - int(baseline[r["task"], r["episode_id"]]["first_target_in_basket"]),
        int(r["outcome"] == "success") - int(baseline[r["task"], r["episode_id"]]["outcome"] == "success")))
    video_paths = {}
    for c, filename in (("A0", "A0.png"), (best, "best.png")):
        directory = out / c / sample["task"] / f"ep{sample['episode_id']}"
        videos = sorted(directory.glob("*.mp4"))
        if not videos:
            raise RuntimeError(f"video missing: {directory}")
        target = out / "frames" / filename
        if sys.executable == str(SIM_PYTHON):
            frames(videos[0], target)
        else:
            # Host Python need not contain OpenCV/Pillow; the simulation venv does.
            subprocess.run([str(SIM_PYTHON), "-c",
                            "import sys; sys.path.insert(0,sys.argv[1]); from report_fruit_exec_guard import frames; from pathlib import Path; frames(Path(sys.argv[2]),Path(sys.argv[3]))",
                            str(ECM / "scripts"), str(videos[0]), str(target)], check=True)
        video_paths[filename] = str(videos[0])
    result["video_examples"] = {"task": sample["task"], "episode_id": sample["episode_id"], "videos": video_paths}
    atomic_json(out / "summary.json", result)
    def ratio(v):
        return f"{v['numerator']}/{v['denominator']}"
    lines = ["阶段 1：完成，停在 G0，等待用户确认。",
             f"已确认：最好条件 {best}，首颗目标入篮 {ratio(s['first_target_in_basket'])}，整任务成功 {ratio(s['success'])}。",
             f"G0 数值门槛：{'通过' if g0 else '未通过'}（要求首颗目标 ≥20/40 且整任务 ≥10/40）。",
             "已确认：执行步长 8/4 各两条记忆冒烟审计通过；所有布局五条件使用同一服务会话。",
             "决策：等待确认后才运行 G1 或进入阶段 2；本脚本未使用 dev 220–259 或 val 100–149。", "",
             "## 做了什么", "",
             "安全包络默认关闭；v1 对七个手臂关节应用训练状态 0.5%–99.5% 分位包络，q7 再约束至 [−π,π]，再与实测状态±v_max 的区间求交。夹爪保持策略原始输出。若交集为空即记录工程错误，不隐式放宽。日志同时记录 proposed_action、action 和逐关节裁剪标记。", "",
             "执行步长通过客户端适配器覆盖官方 eval.py 传给 get_action_chunk 的执行切片长度，obs_horizon 仍为 16；官方循环、观测缓冲和逐帧写入保持原实现。未修改 vendor 文件。", "",
             "## 关键数字与目标对照", "",
             "下列每个条件的所有数字来源为 `analysis/<条件>/summary.json`，逐回合来源为相邻 episodes.jsonl。", ""]
    metrics = [(k, k) for k in ("success", "fail", "timeout", "first_target_in_basket", "never_detached", "near_miss",
               "wrist_over_limit", "wrist_deviation_executed", "wrist_deviation_proposed", "freeze_200",
               "freeze_200_basket_unsuccessful", "count_exact", "engineering_error")]
    for scope in ("all", *TASKS):
        lines += [f"### {'合计' if scope == 'all' else scope}", "", "| 指标 | A0 | A1 | A2 | A3 | A4 |", "|---|---:|---:|---:|---:|---:|"]
        values = {c: summaries[c]["all"] if scope == "all" else summaries[c]["by_task"][scope] for c in CONDITIONS}
        for name, key in metrics:
            lines.append("| "+name+" | "+" | ".join(ratio(values[c][key]) for c in CONDITIONS)+" |")
        for key in ("correct_in_basket", "wrong_in_basket"):
            lines.append("| "+key+"（果实总数） | "+" | ".join(str(values[c][key]["total_fruits"]) for c in CONDITIONS)+" |")
        lines.append("| start_idle 中位数（控制步） | "+" | ".join(str(values[c]["start_idle"]["median"]) for c in CONDITIONS)+" |")
        lines.append("| guard_clip_rate（控制步） | "+" | ".join(ratio(values[c]["guard_clip_rate"]["any_joint"]) for c in CONDITIONS)+" |")
        for j in range(7):
            lines.append(f"| 关节 {j+1} 裁剪 | "+" | ".join(ratio(values[c]["guard_clip_rate"]["per_joint"][j]) for c in CONDITIONS)+" |")
        for context in ("carrying_fruit", "closed_empty", "open_after_basket", "open_before_first_detach"):
            lines.append("| 偏离起始 "+context+" | "+" | ".join(ratio(values[c]["deviation_onset_context"][context]) for c in CONDITIONS)+" |")
        lines.append("")
    lines += ["## 安全包络数值", "", "来源：`reports/fruit_joint_guard_v1_20261102.json`；400 条干净+DART 训练示范，173791 个状态。v_max 是每回合内相邻实测状态差绝对值的 99.9 分位×1.5，单位 rad/控制步。", "",
              "| 关节 | 状态下界 | 状态上界 | v_max |", "|---|---:|---:|---:|"]
    guard = result["guard_stats"]
    for j in range(7):
        lines.append(f"| {j+1} | {guard['lower'][j]:.9f} | {guard['upper'][j]:.9f} | {guard['v_max'][j]:.9f} |")
    lines += ["", "q7 执行包络实际还与 [−π,π] 求交。", "", "## A0 与存档基线", "",
              f"来源：summary.json 的 A0_reproducibility。存档成功 {ratio(archived['all']['success'])}，本次 {ratio(summaries['A0']['all']['success'])}；结局改变 {len(changes)}/40 场，逐条差异保存在 summary.json。相同布局跨服务仍可能产生不同轨迹，本次各条件尽量在同一服务会话内配对。", "",
              "## 记忆一致性与失败记录", "",
              "smoke_A2、smoke_A3 各运行两个任务的 ep200。复用 run_teacher_oracle.selection_audit 复算逐决策选帧，并检查全历史帧号、原始图像/状态哈希、WebSocket 输入输出哈希及位置编码来源。每条结果见对应 memory_audit.json；全实验也逐回合执行同样审计。模型返回 20×8 动作块，基线执行前 16 步、干预执行前 8/4 步。旧 policy_actions.execution_horizon 字段硬编码为 16，真实执行长度见 result.json 的 exec_horizon 和决策帧间隔。", "",
              "若显式开启 continue-guard-infeasible，仅对包络交集为空的回合停止执行，保留 outcome=error 与工程错误计数，然后继续剩余比较；不放宽任何安全约束，不把拒绝执行算作任务 fail 或成功，记忆审计仍须通过。原失败运行完整保留在 summary.json 的 prior_failed_run 所指目录。", "",
              f"工程失败及 superseded 记录共 {len(failures)} 条，全部保存在 summary.json 的 failed_runs_including_superseded；outcome=fail 的任务失败完整保留在各回合结果和汇总中。", "",
              "## 同一布局的视频抽帧", "",
              f"布局：{sample['task']} ep{sample['episode_id']}；原始视频路径见 summary.json/video_examples。", "",
              "![A0](frames/A0.png)", "", f"![最好条件 {best}](frames/best.png)", "",
              "## 按规则做出的决策与下一步", "",
              f"G0 门槛{'通过' if g0 else '未通过'}。请用户确认后，"+("用当前 K0 和该设置在 dev 220–259 做一次 G1；通过才走捷径 A。" if g0 else "进入阶段 2。执行设置是否带入后续阶段，以降低腕部偏离率为依据，结果与 A0 对照见上表。"),
              "不得自动越过 G0。本阶段没有训练，没有运行新门槛集或封存验证集。"]
    (out / "report.md").write_text("\n".join(lines)+"\n")
    status = read_json(ECM / "reports/lowlevel_rescue_status.json")
    status.update(phase=1, state="awaiting_G0_confirmation", updated_utc=timestamp())
    status["links"].update(D1_report=str(out / "report.md"), D1_summary=str(out / "summary.json"), D1_status=str(out / "status.json"))
    atomic_json(ECM / "reports/lowlevel_rescue_status.json", status)
    print({"best": best, "G0_thresholds_met": g0, "state": "awaiting_G0_confirmation"}, flush=True)


if __name__ == "__main__":
    main()
