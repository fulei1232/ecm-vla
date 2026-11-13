#!/usr/bin/env python3
"""Closed-loop Teacher Oracle step 1: simulator branch replay preflight (methods S/R/F).

prepare  seal protocol.json (episodes, branch points, jobs, code and reference hashes)
run      one fresh simulator process per job, same environment as V-A sim children; resumable
analyze  gate verdicts -> checks.jsonl, summary.json, REPORT.md
all      prepare (if needed), run, analyze
Protocol document: reports/branch_replay_preflight_protocol_20261023.md
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from v_a_common import ECM, POLICY, SIM, SIM_PYTHON, atomic_json, read_json, sha256, timestamp

sys.path.insert(0, str(ECM / "src"))
from memory_management.validation import sim_branch as sb  # noqa: E402

SCRIPT = Path(__file__).resolve()
DOC = ECM / "reports/branch_replay_preflight_protocol_20261023.md"
REFERENCE_RUN = ECM / "results/V_A_formal_20261023"
EPISODES = (("BinFill", 5, 7), ("BinFill", 2, 7), ("VideoUnmaskSwap", 16, 7), ("VideoUnmaskSwap", 3, 7))
SMOKE_EPISODES = (("VideoUnmaskSwap", 16, 7),)
CODE = (SCRIPT, ECM / "scripts/v_a_common.py", ECM / "src/memory_management/validation/sim_branch.py",
        ECM / "src/memory_management/validation/acm_copy/diagnostic_logging.py",
        POLICY / "examples/robomme/env_runner.py",
        SIM / "src/robomme/env_record_wrapper/DemonstrationWrapper.py",
        SIM / "src/robomme/env_record_wrapper/episode_config_resolver.py")
PARAMS = {"max_points": 8, "smoke_max_points": 3, "max_transition_points": 3, "load_steps": 32,
          "load_delta": 0.05, "perturb_joint": 0, "perturb_delta": 1e-5, "fork_timeout_s": 120,
          "job_timeout_s": 3600, "f0_steps": 16}
FORK_THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
REFERENCE_FILES = ("result.json", "control_actions.jsonl", "raw_frames.jsonl")


def episode_key(task, episode_id, seed):
    return f"{task}_ep{episode_id:02d}_seed{seed}"


def plan_jobs(ref, params, smoke):
    points = sb.select_branch_points(ref.decision_points(), ref.subgoal_transitions(),
                                     params["smoke_max_points" if smoke else "max_points"],
                                     params["max_transition_points"])
    later = [t for t in points if t >= sb.EXEC_HORIZON]
    middle = points[len(points) // 2]
    jobs = {"g0_A": {"kind": "g0", "dump": True}, "g0_B": {"kind": "g0", "dump": True},
            "r": {"kind": "r", "points": points, "load": "L1", "repeats": 2, "off_by_one": later[0]}}
    for t in sorted({points[0], middle}):
        jobs[f"r_proc_t{t:04d}"] = {"kind": "r_point", "t": t, "load": "L1"}
    jobs.update({
        "f": {"kind": "f", "double_points": points, "load": "L1"},
        "f_l0": {"kind": "f", "double_points": [], "load": "L0"},
        "f0": {"kind": "f0", "steps": params["f0_steps"]},
        "s_L0": {"kind": "s", "points": points, "load": "L0", "python": True},
        "s_L1": {"kind": "s", "points": points, "load": "L1", "python": True},
        "s_nopy": {"kind": "s", "points": [later[0]], "load": "L1", "python": False},
        "perturb": {"kind": "perturb", "t": middle, "joint": params["perturb_joint"],
                    "delta": params["perturb_delta"]},
    })
    return points, jobs


def prepare(out, smoke):
    if (out / "protocol.json").exists():
        raise SystemExit(f"{out} is already prepared; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    episodes = {}
    for task, episode_id, seed in (SMOKE_EPISODES if smoke else EPISODES):
        directory = REFERENCE_RUN / "episodes/D0a" / task / f"ep{episode_id:02d}_seed{seed}"
        ref = sb.load_reference(directory)
        server = read_json(REFERENCE_RUN / f"server{ref.server}/config.json")
        points, jobs = plan_jobs(ref, PARAMS, smoke)
        episodes[episode_key(task, episode_id, seed)] = {
            "task": task, "episode_id": episode_id, "model_seed": seed, "directory": str(directory),
            "server": ref.server, "sim_gpu": server["sim_gpu"], "environment_seed": ref.environment_seed,
            "outcome": ref.outcome, "executed_steps": ref.n_steps, "exec_start_idx": ref.exec_start,
            "decision_points": len(ref.decision_points()), "subgoal_transitions": ref.subgoal_transitions(),
            "branch_points": points, "jobs": jobs,
            "reference_sha256": {name: sha256(directory / name) for name in REFERENCE_FILES}}
    atomic_json(out / "protocol.json", {
        "name": "branch_replay_preflight", "formal": not smoke, "created_utc": timestamp(),
        "protocol_doc": str(DOC), "protocol_doc_sha256": sha256(DOC), "reference_run": str(REFERENCE_RUN),
        "split": "val", "params": PARAMS, "episodes": episodes, "python": str(SIM_PYTHON),
        "fork_thread_env": FORK_THREADS, "code_sha256": {str(p): sha256(p) for p in CODE}})


def verify_sealed(protocol):
    changed = [p for p, h in protocol["code_sha256"].items() if sha256(p) != h]
    changed += [f"{info['directory']}/{name}" for info in protocol["episodes"].values()
                for name, h in info["reference_sha256"].items() if sha256(Path(info["directory"]) / name) != h]
    if changed:
        raise SystemExit("sealed inputs changed since prepare:\n  " + "\n  ".join(changed))


def worker_env(sim_gpu, kind):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(sim_gpu), "PYTHONDONTWRITEBYTECODE": "1",
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
    if kind == "f":
        env.update(FORK_THREADS)
    return env


def job_result(out, key, name):
    path = out / "workers" / key / f"{name}.json"
    return read_json(path) if path.exists() else None


def run_episode(out, protocol, key, lock):
    info = protocol["episodes"][key]
    for name, spec in info["jobs"].items():
        if (job_result(out, key, name) or {}).get("completed"):
            continue
        path = out / "workers" / key / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.with_suffix(".log")
        command = [str(SIM_PYTHON), "-u", str(SCRIPT), "worker", "--out", str(out), "--episode", key, "--job", name]
        start = time.monotonic()
        with log.open("w") as f:
            try:
                code = subprocess.run(command, cwd=ECM, env=worker_env(info["sim_gpu"], spec["kind"]), stdout=f,
                                      stderr=subprocess.STDOUT, timeout=protocol["params"]["job_timeout_s"]).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        if not (job_result(out, key, name) or {}).get("completed"):
            atomic_json(path, {"job": name, "episode": key, "kind": spec["kind"], "completed": False, "exit": code,
                               "log_tail": log.read_text(errors="replace")[-4000:]})
        with lock:
            print(f"[{timestamp()}] {key} {name}: exit={code} {time.monotonic() - start:.0f}s", flush=True)


def run(out, parallel):
    protocol = read_json(out / "protocol.json")
    verify_sealed(protocol)
    lock, gate = threading.Lock(), threading.Semaphore(parallel)

    def target(key):
        with gate:
            run_episode(out, protocol, key, lock)

    threads = [threading.Thread(target=target, args=(key,)) for key in protocol["episodes"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def runtime_info():
    versions = {}
    for name in ("mani_skill", "sapien", "mplib", "torch", "numpy", "gymnasium"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "executable": sys.executable, "pid": os.getpid(), "versions": versions,
            **{k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VK_ICD_FILENAMES")}}


def worker(out, key, name):
    protocol = read_json(out / "protocol.json")
    info, params = protocol["episodes"][key], protocol["params"]
    spec = info["jobs"][name]
    sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"), str(SIM / "src")]
    import robomme
    assert Path(robomme.__file__).resolve() == (SIM / "src/robomme/__init__.py").resolve(), robomme.__file__
    ref = sb.load_reference(info["directory"])
    scratch = out / "scratch" / key / name
    shutil.rmtree(scratch, ignore_errors=True)
    dump = out / "reference_states" / key / f"{name}.npz" if spec.get("dump") else None
    start = time.monotonic()
    result = sb.run_job(ref, spec, scratch, params, dump)
    result.update(job=name, episode=key, kind=spec["kind"], completed=True, finished_utc=timestamp(),
                  seconds=round(time.monotonic() - start, 3), runtime=runtime_info())
    atomic_json(out / "workers" / key / f"{name}.json", result)
    shutil.rmtree(scratch, ignore_errors=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Renderer teardown after a lost Vulkan device aborts the process; the result is already durable.
    os._exit(0)


# --------------------------------------------------------------------------- analysis

def episode_details(outputs):
    d = {"seconds": {n: (o or {}).get("seconds") for n, o in outputs.items()},
         "failed_jobs": {n: {"exit": o.get("exit"), "log_tail": o.get("log_tail", "")[-800:]}
                         for n, o in outputs.items() if o and not o.get("completed")}}
    for name in ("g0_A", "g0_B", "f", "f_l0", "s_L0", "s_L1", "s_nopy", "perturb"):
        o = outputs.get(name) or {}
        if o.get("completed"):
            d[name] = {k: o["frames"].get(k) for k in ("frames", "physics_mismatches", "render_mismatches",
                                                       "first_physics_mismatch", "first_render_mismatch")}
            if "controls" in o:
                d[name]["controls"] = {k: o["controls"][k] for k in ("mismatches", "first_mismatch")}
            if o.get("error"):
                d[name]["error"] = o["error"][-800:]
    for name in ("s_L0", "s_L1", "s_nopy"):
        o = outputs.get(name) or {}
        bad = [e for e in o.get("events", []) if not (e["sim_equal"] and e["python_equal"] and e["rng_equal"])]
        if o.get("completed"):
            d[name]["first_unrestored_event"] = bad[0] if bad else None
    for name in ("f", "f_l0"):
        f = outputs.get(name) or {}
        if f.get("completed"):
            forks = f["forks"]
            d[name]["forks"] = {"count": len(forks), "statuses": sorted({x["status"] for x in forks}),
                                "max_seconds": max((x["seconds"] for x in forks), default=None),
                                "first_error": next((x["error"][-800:] for x in forks if x.get("error")), None)}
            d["backend"] = f.get("backend")
    f0 = outputs.get("f0") or {}
    if f0.get("completed"):
        d["f0"] = {"ok": f0["ok"], "error": f0.get("error")}
    return d


LABELS = (("G0a", "G0a 物理"), ("G0b", "G0b 渲染"), ("G0c", "G0c 跨进程"), ("R_pass", "R"), ("F_pass", "F"),
          ("S_pass", "S"), ("control_perturb_detected", "扰动对照"), ("control_off_by_one_detected", "少一步对照"),
          ("control_s_without_python_detected", "S 对照"), ("F_noop_fork_safe", "空 fork 安全"),
          ("F0_headless_ok", "F0 无渲染"))


def yes(value):
    return "是" if value else "否"


def write_report(out, protocol, per, overall, details):
    lines = [f"# 仿真分支复现测试结果（Teacher Oracle 第 1 步）", "",
             f"状态：{'正式' if protocol['formal'] else '冒烟（非正式）'}运行；协议 `{protocol['protocol_doc']}`；"
             f"生成于 {timestamp()}。", "",
             f"结论：第 1 步{'通过' if overall['step1_passed'] else '未通过'}。"
             f"推荐分支方法：{overall['recommended_method'] or '无'}。"
             + (f"停止原因：{overall['stop_reason']}。" if overall["stop_reason"] else ""), "",
             "| episode | " + " | ".join(label for _, label in LABELS) + " |",
             "|---" * (len(LABELS) + 1) + "|"]
    for key, g in per.items():
        lines.append(f"| {key} | " + " | ".join(yes(g[k]) for k, _ in LABELS) + " |")
    lines += ["", "各方法的分项检查（全部 episode 都为“是”才算该方法通过）：", ""]
    sub = ("R_start_equal", "R_load_deterministic", "R_fresh_process_equal", "F_parent_unchanged",
           "F_child_start_equal", "F_load_deterministic", "F_agrees_with_R", "S_L0_restored", "S_L0_replay_equal",
           "S_L1_restored", "S_L1_replay_equal", "G0c_render")
    lines += ["| episode | " + " | ".join(sub) + " |", "|---" * (len(sub) + 1) + "|"]
    for key, g in per.items():
        lines.append(f"| {key} | " + " | ".join(yes(g[k]) for k in sub) + " |")
    lines += ["", "## 细节", ""]
    for key, d in details.items():
        g = per[key]
        lines.append(f"### {key}")
        if g["missing_jobs"]:
            lines.append(f"- 未完成的 job：{', '.join(g['missing_jobs'])}")
        for name in ("g0_A", "g0_B", "f", "f_l0", "s_L0", "s_L1", "s_nopy", "perturb"):
            if name in d:
                x = d[name]
                lines.append(f"- {name}：{x['frames']} 帧，物理不一致 {x['physics_mismatches']}，渲染不一致 "
                             f"{x['render_mismatches']}；首个物理不一致 {json.dumps(x['first_physics_mismatch'], ensure_ascii=False)}")
                if x.get("first_unrestored_event"):
                    e = x["first_unrestored_event"]
                    lines.append(f"  - 首个未完全恢复的分支点 t={e['t']}：sim 差异 {e['sim_diff'][:5]}，"
                                 f"Python 差异 {e['python_diff'][:5]}，RNG 一致={e['rng_equal']}")
                if x.get("error"):
                    lines.append(f"  - 异常：`{x['error'].strip().splitlines()[-1]}`")
                if "forks" in x:
                    fk = x["forks"]
                    lines.append(f"  - fork {fk['count']} 次，状态 {fk['statuses']}，单次最长 {fk['max_seconds']} 秒")
                    if fk["first_error"]:
                        lines.append(f"  - 首个子进程异常：`{fk['first_error'].strip().splitlines()[-1]}`")
        if "f0" in d:
            lines.append(f"- F0 无渲染影子：{'可运行' if d['f0']['ok'] else '不可运行'}"
                         + (f"（`{d['f0']['error']}`）" if d["f0"]["error"] else ""))
        for name, x in d["failed_jobs"].items():
            lines.append(f"- job {name} 失败（exit={x['exit']}）：`{x['log_tail'].strip().splitlines()[-1] if x['log_tail'].strip() else ''}`")
        if d.get("backend"):
            lines.append(f"- 后端：{d['backend']}")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))


def analyze(out):
    protocol = read_json(out / "protocol.json")
    per, details, rows = {}, {}, []
    for key, info in protocol["episodes"].items():
        outputs = {name: job_result(out, key, name) for name in info["jobs"]}
        per[key] = sb.evaluate_episode(outputs)
        details[key] = episode_details(outputs)
        rows += [{"episode": key, "check": k, "value": v} for k, v in per[key].items()]
    overall = sb.evaluate_overall(per)
    (out / "checks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    atomic_json(out / "summary.json", {"formal": protocol["formal"], "analyzed_utc": timestamp(), "overall": overall,
                                       "per_episode": per, "details": details})
    write_report(out, protocol, per, overall, details)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name in ("prepare", "all"):
            p.add_argument("--smoke", action="store_true", help="one short episode, 3 branch points; not formal")
        if name in ("run", "all"):
            p.add_argument("--parallel", type=int, default=4, help="episodes run concurrently")
    w = sub.add_parser("worker")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--episode", required=True)
    w.add_argument("--job", required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "worker":
        return worker(out, args.episode, args.job)
    if args.command == "prepare" or (args.command == "all" and not (out / "protocol.json").exists()):
        prepare(out, args.smoke)
    elif args.command == "all" and read_json(out / "protocol.json")["formal"] == args.smoke:
        raise SystemExit("--smoke does not match the existing protocol.json")
    if args.command in ("run", "all"):
        run(out, args.parallel)
    if args.command in ("analyze", "all"):
        analyze(out)


if __name__ == "__main__":
    main()
