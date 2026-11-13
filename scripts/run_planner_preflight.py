#!/usr/bin/env python3
"""Closed-loop Teacher Oracle step 2: the RoboMME planner inside R branches of D0a rollouts.

prepare  seal protocol.json (episodes, decision points, jobs, code and reference hashes)
run      one fresh simulator process per job, same environment as V-A sim children; resumable
analyze  gate verdicts per task -> checks.jsonl, summary.json, REPORT.md
all      prepare (if needed), run, analyze
Protocol documents: reports/planner_branch_preflight_protocol_20261023.md (--expert original, step 2),
reports/phase_expert_preflight_protocol_20261024.md (--expert phase, step 2b).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from v_a_common import ECM, POLICY, SIM, SIM_PYTHON, atomic_json, read_json, sha256, timestamp
import run_branch_preflight as step1

sys.path.insert(0, str(ECM / "src"))
from memory_management.validation import planner_branch as pb  # noqa: E402
from memory_management.validation import sim_branch as sb  # noqa: E402

SCRIPT = Path(__file__).resolve()
REFERENCE_RUN = step1.REFERENCE_RUN
CONDITION = "D0a"
TASKS = ("BinFill", "VideoUnmaskSwap", "PatternLock")
EXPERTS = {"original": (ECM / "reports/planner_branch_preflight_protocol_20261023.md", TASKS,
                        "planner_branch_preflight", "分支内 planner 可用性测试结果（Teacher Oracle 第 2 步）"),
           "phase": (ECM / "reports/phase_expert_preflight_protocol_20261024.md", ("BinFill", "VideoUnmaskSwap"),
                     "phase_expert_preflight", "按阶段续接专家的分支测试结果（Teacher Oracle 第 2b 步）")}
MANISKILL = SIM_PYTHON.parent.parent / "lib/python3.11/site-packages/mani_skill"
ROBOMME = SIM / "src/robomme"
CODE = (SCRIPT, ECM / "scripts/run_branch_preflight.py", ECM / "scripts/v_a_common.py",
        ECM / "src/memory_management/validation/sim_branch.py",
        ECM / "src/memory_management/validation/planner_branch.py",
        ECM / "src/memory_management/validation/phase_expert.py",
        ECM / "src/memory_management/validation/acm_copy/diagnostic_logging.py",
        POLICY / "examples/robomme/env_runner.py",
        ROBOMME / "env_record_wrapper/DemonstrationWrapper.py",
        ROBOMME / "env_record_wrapper/episode_config_resolver.py",
        ROBOMME / "robomme_env/utils/subgoal_planner_func.py",
        ROBOMME / "robomme_env/utils/subgoal_evaluate_func.py",
        ROBOMME / "robomme_env/utils/planner_fail_safe.py",
        ROBOMME / "robomme_env/BinFill.py", ROBOMME / "robomme_env/VideoUnmaskSwap.py",
        ROBOMME / "robomme_env/PatternLock.py",
        MANISKILL / "examples/motionplanning/base_motionplanner/motionplanner.py",
        MANISKILL / "examples/motionplanning/two_finger_gripper/motionplanner.py",
        MANISKILL / "examples/motionplanning/panda/motionplanner.py",
        MANISKILL / "examples/motionplanning/panda/motionplanner_stick.py")
PARAMS = {"model_seed": 7, "per_outcome": 4, "smoke_points": 3, "points_per_job": 6,
          "chunk_len": pb.CHUNK_LEN, "replan_every": pb.REPLAN_EVERY, "max_steps": pb.MAX_STEPS,
          "stall_steps": pb.STALL_STEPS, "stall_solves": pb.STALL_SOLVES, "start_steps": pb.START_STEPS,
          "p1_min": 0.9, "p2_min": 0.9, "p3_min": 0.9, "job_timeout_s": 5400}
REFERENCE_FILES = step1.REFERENCE_FILES


def select_episodes(rows, task, seed, per_outcome):
    """First `per_outcome` successes and failures of one task and model seed, by episode id."""
    rows = sorted((r for r in rows if r["condition"] == CONDITION and r["task"] == task
                   and int(r["model_seed"]) == seed), key=lambda r: int(r["episode_id"]))
    success = [int(r["episode_id"]) for r in rows if r["outcome"] == "success"][:per_outcome]
    fail = [int(r["episode_id"]) for r in rows if r["outcome"] != "success"][:per_outcome]
    if len(success) < per_outcome or len(fail) < per_outcome:
        raise SystemExit(f"{task} seed {seed}: only {len(success)} successes / {len(fail)} failures")
    return success, fail


def repeat_points(decisions):
    return sorted({decisions[0], decisions[len(decisions) // 2], decisions[-1]})


def plan_jobs(decisions, params):
    size = params["points_per_job"]
    jobs = {f"main_{j:02d}": {"kind": "points", "points": decisions[i:i + size], "modes": list(pb.MODES),
                              "repeat": False}
            for j, i in enumerate(range(0, len(decisions), size))}
    jobs["repeat"] = {"kind": "points", "points": repeat_points(decisions), "modes": list(pb.MODES), "repeat": True}
    return jobs


def actions_digest(directory):
    files = sorted((Path(directory) / "actions").glob("decision_*.npz"))
    return sb._sha("".join(f"{p.name}:{sha256(p)}\n" for p in files))


def prepare(out, smoke, expert):
    if (out / "protocol.json").exists():
        raise SystemExit(f"{out} is already prepared; use a new directory")
    doc, tasks, name, _ = EXPERTS[expert]
    params = {**PARAMS, "expert": expert}
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((REFERENCE_RUN / "per_episode.csv").open()))
    episodes = {}
    for task in tasks:
        success, fail = select_episodes(rows, task, PARAMS["model_seed"], PARAMS["per_outcome"])
        for episode_id in (success[:1] if smoke else success + fail):
            directory = REFERENCE_RUN / "episodes" / CONDITION / task / f"ep{episode_id:02d}_seed{PARAMS['model_seed']}"
            ref = sb.load_reference(directory)
            server = read_json(REFERENCE_RUN / f"server{ref.server}/config.json")
            decisions = ref.decision_points()[:PARAMS["smoke_points"]] if smoke else ref.decision_points()
            key = step1.episode_key(task, episode_id, PARAMS["model_seed"])
            episodes[key] = {
                "task": task, "episode_id": episode_id, "model_seed": PARAMS["model_seed"], "directory": str(directory),
                "server": ref.server, "sim_gpu": server["sim_gpu"], "environment_seed": ref.environment_seed,
                "outcome": ref.outcome, "executed_steps": ref.n_steps, "exec_start_idx": ref.exec_start,
                "points": decisions, "fresh_points": [t for t in pb.fresh_points(ref) if t in decisions],
                "jobs": plan_jobs(decisions, params),
                "reference_sha256": {name: sha256(directory / name) for name in REFERENCE_FILES},
                "actions_sha256": actions_digest(directory)}
    atomic_json(out / "protocol.json", {
        "name": name, "formal": not smoke, "created_utc": timestamp(),
        "protocol_doc": str(doc), "protocol_doc_sha256": sha256(doc), "reference_run": str(REFERENCE_RUN),
        "condition": CONDITION, "split": "val", "params": params, "episodes": episodes, "python": str(SIM_PYTHON),
        "code_sha256": {str(p): sha256(p) for p in CODE}})


def verify_sealed(protocol):
    step1.verify_sealed(protocol)
    changed = [key for key, info in protocol["episodes"].items() if actions_digest(info["directory"]) != info["actions_sha256"]]
    if changed:
        raise SystemExit("reference action chunks changed since prepare:\n  " + "\n  ".join(changed))


def job_result(out, key, name):
    return step1.job_result(out, key, name)


def run_job(out, protocol, key, name, lock):
    info = protocol["episodes"][key]
    path = out / "workers" / key / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.with_suffix(".log")
    command = [str(SIM_PYTHON), "-u", str(SCRIPT), "worker", "--out", str(out), "--episode", key, "--job", name]
    start = time.monotonic()
    with log.open("w") as f:
        try:
            code = subprocess.run(command, cwd=ECM, env=step1.worker_env(info["sim_gpu"], "points"), stdout=f,
                                  stderr=subprocess.STDOUT, timeout=protocol["params"]["job_timeout_s"]).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    if not (job_result(out, key, name) or {}).get("completed"):
        atomic_json(path, {"job": name, "episode": key, "completed": False, "exit": code,
                           "log_tail": log.read_text(errors="replace")[-4000:]})
    with lock:
        print(f"[{timestamp()}] {key} {name}: exit={code} {time.monotonic() - start:.0f}s", flush=True)


def run(out, parallel):
    protocol = read_json(out / "protocol.json")
    verify_sealed(protocol)
    pending = [(key, name) for key, info in protocol["episodes"].items() for name in info["jobs"]
               if not (job_result(out, key, name) or {}).get("completed")]
    # Longest jobs first: BinFill decision points replay the most steps.
    pending.sort(key=lambda kn: -max(protocol["episodes"][kn[0]]["jobs"][kn[1]]["points"]))
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for future in [pool.submit(run_job, out, protocol, key, name, lock) for key, name in pending]:
            future.result()


def worker(out, key, name):
    protocol = read_json(out / "protocol.json")
    info, params = protocol["episodes"][key], protocol["params"]
    spec = info["jobs"][name]
    sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"), str(SIM / "src")]
    import robomme
    assert Path(robomme.__file__).resolve() == (SIM / "src/robomme/__init__.py").resolve(), robomme.__file__
    import mani_skill
    assert Path(mani_skill.__file__).resolve().parent == MANISKILL.resolve(), mani_skill.__file__
    ref = sb.load_reference(info["directory"])
    scratch = out / "scratch" / key / name
    shutil.rmtree(scratch, ignore_errors=True)
    start = time.monotonic()
    result = pb.run_job(ref, spec, scratch, params)
    result.update(job=name, episode=key, kind=spec["kind"], completed=True, finished_utc=timestamp(),
                  seconds=round(time.monotonic() - start, 3), runtime=step1.runtime_info())
    atomic_json(out / "workers" / key / f"{name}.json", result)
    shutil.rmtree(scratch, ignore_errors=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Same as step 1: skip renderer teardown once the result is durable.
    os._exit(0)


# --------------------------------------------------------------------------- analysis

GATE_LABELS = (("P0_branch_start", "P0 分支起点"), ("P1_chunk_available", "P1 动作块可用"),
               ("P2_full_competent", "P2 T-full 胜任"), ("P3_rh_competent", "P3 T-rh 胜任"),
               ("P4_deterministic", "P4 确定性"))
METRIC_LABELS = (("l2_expert_policy", "专家-策略 L2"), ("l2_hold_policy", "不动-策略 L2"),
                 ("direction_cosine", "方向余弦"), ("start_speed_ratio", "起步速度比"),
                 ("hold_steps", "夹爪开合步数"), ("gripper_agreement", "夹爪一致率"))
OUTCOME_ORDER = ("success", "fail", "timeout", "stuck", "exception")


def yes(value):
    return "是" if value else "否"


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def num(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def outcome_text(counter):
    keys = list(OUTCOME_ORDER) + sorted(k for k in counter if k not in OUTCOME_ORDER)
    return "，".join(f"{k} {counter[k]}" for k in keys if counter.get(k)) or "无样本"


def write_report(out, protocol, per_task, overall):
    expert = protocol["params"].get("expert", "original")
    lines = [f"# {EXPERTS[expert][3]}", "",
             f"状态：{'正式' if protocol['formal'] else '冒烟（非正式）'}运行；协议 `{protocol['protocol_doc']}`；"
             f"生成于 {timestamp()}。", "",
             "可进入第 3 步的任务：" + ("、".join(overall["usable_tasks"]) if overall["usable_tasks"] else "无") + "。", "",
             "| 任务 | " + " | ".join(label for _, label in GATE_LABELS) + " | 可用 |",
             "|---" * (len(GATE_LABELS) + 2) + "|"]
    for task, r in per_task.items():
        lines.append(f"| {task} | " + " | ".join(yes(r["gates"][k]) for k, _ in GATE_LABELS) + f" | {yes(r['usable'])} |")
    lines += ["", "| 任务 | 决策点 | 动作块可用率 | 成功 episode 上 T-full 成功率 | 成功 episode 上 T-rh 成功率 | 可比重复对 |",
              "|---|---|---|---|---|---|"]
    for task, r in per_task.items():
        rates, c = r["rates"], r["counts"]
        lines.append(f"| {task} | {c['decisions']} | {pct(rates['P1_chunk_available'])} | "
                     f"{pct(rates['P2_full_success_on_success_episodes'])} | {pct(rates['P3_rh_success_on_success_episodes'])} | "
                     f"{c['comparable_pairs']}/{c['repeat_pairs']} |")
    for task, r in per_task.items():
        d, c = r["diagnostics"], r["counts"]
        lines += ["", f"## {task}", ""]
        if c["missing_jobs"]:
            lines.append(f"- 未完成的 job：{', '.join(c['missing_jobs'])}")
        if c["branches"] != c["expected_branches"]:
            lines.append(f"- 分支数 {c['branches']}，应为 {c['expected_branches']}")
        lines.append("- 结局分布：")
        for group, counter in d["outcomes"].items():
            lines.append(f"  - {group}：{outcome_text(counter)}")
        p = d["planner"]
        lines.append(f"- planner：用到 RRT* 的分支 {pct(p['branches_with_rrt'])}，动作块内用到 RRT* {pct(p['chunks_with_rrt'])}，"
                     f"出现 exhausted 的分支 {pct(p['branches_with_exhausted'])}，每次运动规划 {num(p['plan_seconds_per_move'])} 秒，"
                     f"每次专家运行 {num(p['expert_seconds_mean'], 1)} 秒")
        lines.append(f"- 夹爪初始化 {d['gripper_inits']} 次，与上一步动作符号不一致 {d['gripper_init_mismatches']} 次；"
                     f"目标已移出场景的决策点 {d['targets_removed']} 个")
        ph = d.get("phase")
        if ph:
            top = sorted(ph["primitives"].items(), key=lambda kv: -kv[1])
            lines.append(f"- 阶段专家：退化移动 {ph['degenerate_moves']} 次，换目标 {ph['retargets']} 次，"
                         f"回退原版 solve {ph['fallbacks']} 次，触及原语上限 {ph['primitive_limit']} 次；阶段判断："
                         + "，".join(f"{k} {v}" for k, v in top))
        det = d["determinism"]
        lines.append(f"- 重复性：不一致的可比对 {len(det['mismatched_pairs'])} 个"
                     + (f"（{det['mismatched_pairs'][:5]}）" if det["mismatched_pairs"] else "")
                     + f"；用到 RRT* 的重复对 {det['rrt_pairs']} 个，其中逐位相同 {det['rrt_pairs_equal']} 个")
        if d["chunk_metrics"]:
            lines += ["", "动作块与策略动作块对比（中位数，括号内为样本数）：", "",
                      "| 分组 | " + " | ".join(label for _, label in METRIC_LABELS) + " |",
                      "|---" * (len(METRIC_LABELS) + 1) + "|"]
            for group, m in d["chunk_metrics"].items():
                lines.append(f"| {group} | " + " | ".join(f"{num(m[k]['median'])} ({m[k]['n']})" for k, _ in METRIC_LABELS) + " |")
        if d["recoverability"]:
            lines += ["", "失败 episode 的可挽回性（每个决策点的 T-full / T-rh 结局）：", ""]
            for key, e in d["recoverability"].items():
                seq = " ".join(f"{t}:{(a or '-')[:4]}/{(b or '-')[:4]}" for t, a, b in e["points"])
                lines.append(f"- {key}（{e['n_steps']} 步）：T-full 可挽回 {pct(e['full_recoverable'])}，最后可挽回 t="
                             f"{e['full_last_recoverable_t']}；T-rh 可挽回 {pct(e['rh_recoverable'])}，最后可挽回 t="
                             f"{e['rh_last_recoverable_t']}")
                lines.append(f"  - {seq}")
        for kind, items in (("分支异常", d["errors"]), ("专家异常", d["exceptions"])):
            for x in items:
                last = x["error"].strip().splitlines()[-1] if x["error"].strip() else ""
                lines.append(f"- {kind} t={x['t']} {x['mode']}：`{last}`")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def analyze(out):
    protocol = read_json(out / "protocol.json")
    per_task, rows = {}, []
    for task in TASKS:
        episodes = [{"key": key, "info": info, "outputs": {name: job_result(out, key, name) for name in info["jobs"]}}
                    for key, info in protocol["episodes"].items() if info["task"] == task]
        if not episodes:
            continue
        per_task[task] = pb.evaluate_task(episodes, protocol["params"])
        rows += [{"task": task, "check": k, "value": v} for k, v in per_task[task]["gates"].items()]
        rows += [{"task": task, "check": k, "value": v} for k, v in per_task[task]["rates"].items()]
    overall = pb.evaluate_overall(per_task)
    (out / "checks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    atomic_json(out / "summary.json", {"formal": protocol["formal"], "analyzed_utc": timestamp(), "overall": overall,
                                       "per_task": per_task})
    write_report(out, protocol, per_task, overall)
    print(json.dumps({"overall": overall, "gates": {t: r["gates"] for t, r in per_task.items()},
                      "rates": {t: r["rates"] for t, r in per_task.items()}}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name in ("prepare", "all"):
            p.add_argument("--smoke", action="store_true", help="first selected episode per task, 3 points; not formal")
            p.add_argument("--expert", choices=sorted(EXPERTS), default="original",
                           help="original: the tasks' solve() (step 2); phase: phase_expert.PhaseSolver (step 2b)")
        if name in ("run", "all"):
            p.add_argument("--parallel", type=int, default=16, help="jobs run concurrently")
    w = sub.add_parser("worker")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--episode", required=True)
    w.add_argument("--job", required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "worker":
        return worker(out, args.episode, args.job)
    if args.command == "prepare" or (args.command == "all" and not (out / "protocol.json").exists()):
        prepare(out, args.smoke, args.expert)
    elif args.command == "all":
        existing = read_json(out / "protocol.json")
        if existing["formal"] == args.smoke or existing["params"].get("expert", "original") != args.expert:
            raise SystemExit("--smoke/--expert do not match the existing protocol.json")
    if args.command in ("run", "all"):
        run(out, args.parallel)
    if args.command in ("analyze", "all"):
        analyze(out)


if __name__ == "__main__":
    main()
