#!/usr/bin/env python3
"""Resumable eight-GPU HarvestMem v3 E3 queue and reporter."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ECM = Path("/media/fulei/fuleivla/ecm_vla")
POLICY = Path("/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning")
POLICY_PY = "/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python"
SIM_PY = "/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python"
SCRIPT = ECM / "scripts/harvestmem_v3/eval_episode.py"
OUT = ECM / "results/harvestmem_v3_E3"

CHECKPOINTS = {
    "A": "/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/sg/ckpt/mme_vla_suite/sg/11999",
    "D": "/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999",
    "E": "/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999",
}
ARM_MEMORY = {"A": "uniform", "D": "uniform", "E": "none"}
ARM_NAME = {"A": "sg", "D": "k0", "E": "nm"}
GPU_LANES = {"A": [0, 1, 2], "D": [3, 4, 5], "E": [6, 7]}
TASKS = ("count", "occluded_ripe")
EPISODES = [(task, episode) for episode in range(200, 240) for task in TASKS]
LOCK = threading.Lock()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def trace_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def robot_detachments(rows):
    result = []
    if not rows:
        return result
    for i in range(1, len(rows)):
        before, after = rows[i - 1], rows[i]
        for actor, fruit in enumerate(after.get("fruits", [])):
            previous = before.get("fruits", [])[actor]
            if fruit.get("attached") and not previous.get("attached") and not fruit.get("wind_detached"):
                result.append((i, actor))
    return result


def rest_follow(rows, start, rest, window=200):
    for row in rows[start:start + window]:
        if row.get("tcp_pos") and math.dist(row["tcp_pos"], rest) <= 0.03:
            grip = row.get("gripper_qpos", [])
            if len(grip) >= 2 and min(grip[:2]) >= 0.0392:
                return True
    return False


def classify(arm, task, result_path: Path):
    result = load_json(result_path)
    rows = trace_rows(result_path.parent / "trace.jsonl") if (result_path.parent / "trace.jsonl").exists() else []
    timeline = trace_rows(result_path.parent / "subgoal_timeline.jsonl") if (result_path.parent / "subgoal_timeline.jsonl").exists() else []
    failures = set()
    if not rows:
        failures.add("engineering_error")
        return {"failures": sorted(failures), "return_follow": None, "target_follow": None, "events": {}}

    rest = rows[0].get("tcp_pos")
    detach = robot_detachments(rows)
    robot_detached = [actor for _, actor in detach]
    if not robot_detached:
        failures.add("never_pulled_fruit")
    if any(row.get("n_in_basket", 0) > result.get("N", 0) for row in rows):
        failures.add("multi_pick_count")
    if task == "count":
        n = result.get("N", 0)
        if any(row.get("oracle_phase") == "return" for row in rows):
            if rows[-1].get("n_in_basket", 0) < n:
                failures.add("stopped_short")
        if max((row.get("n_in_basket", 0) for row in rows), default=0) >= n and result.get("outcome") != "success":
            failures.add("count_reached_without_completion")
    if result.get("outcome") == "timeout":
        failures.add("timeout")
    if result.get("reason"):
        failures.add(str(result["reason"]))

    for i, row in enumerate(rows):
        if i >= 300:
            window = rows[i - 299:i + 1]
            if window[0].get("tcp_pos") and window[-1].get("tcp_pos") and math.dist(window[0]["tcp_pos"], window[-1]["tcp_pos"]) < 0.01:
                failures.add("tcp_stuck_300")
                break
    for i, actor in detach:
        fruit = rows[i].get("fruits", [])[actor]
        if not fruit.get("in_basket"):
            later = rows[i:min(len(rows), i + 300)]
            if not any(r.get("fruits", [])[actor].get("in_basket") for r in later if actor < len(r.get("fruits", []))):
                failures.add("dropped_outside_basket")
    if task == "occluded_ripe":
        dev = load_json(ECM / "results/harvestmem_v3_S1/replays/dev_expert" / task / f"ep{result['episode']}_attempt0/result.json", {})
        ripe = [bool(x.get("ripe")) for x in dev.get("slot_geometry", [])]
        slot_of = dev.get("slot_of", [])
        for _, actor in detach:
            if actor < len(slot_of) and slot_of[actor] < len(ripe) and not ripe[slot_of[actor]]:
                failures.add("occluded_green_picked")

    return_phase = next((i for i, row in enumerate(rows) if row.get("oracle_phase") == "return"), None)
    if arm == "A":
        return_start = return_phase
    else:
        n = result.get("N", 0)
        return_start = next((i for i, row in enumerate(rows) if row.get("n_in_basket", 0) >= n), None)
    return_follow = rest_follow(rows, return_start, rest) if return_start is not None else False

    target_follow = None
    if arm == "A":
        cycles = []
        selected = None
        for i, row in enumerate(rows):
            if "target_selected" in row.get("oracle_events", []):
                selected = (i, row.get("oracle_target_slot"))
            for j, actor in detach:
                if j == i and selected is not None:
                    slot_of = result.get("slot_of", [])
                    actual = slot_of[actor] if actor < len(slot_of) else None
                    cycles.append({"target_slot": selected[1], "detached_slot": actual,
                                   "followed": selected[1] == actual})
                    selected = None
        target_follow = (sum(c["followed"] for c in cycles) / len(cycles)) if cycles else None

    events = Counter(event for row in timeline for event in row.get("events", []))
    return {"failures": sorted(failures), "return_follow": return_follow,
            "target_follow": target_follow, "events": dict(events),
            "steps": result.get("steps"), "outcome": result.get("outcome"),
            "N": result.get("N"), "video": str(result_path.parent / "front_camera.mp4")}


def summarize():
    rows = []
    for arm in ARM_NAME:
        for task in TASKS:
            for path in sorted((OUT / arm / task).glob("ep*/result.json")):
                result = load_json(path)
                if result is None:
                    continue
                rows.append({"arm": arm, "task": task, "episode": result.get("episode"),
                             "success": result.get("outcome") == "success", **classify(arm, task, path)})
    lines = ["# HarvestMem v3 E3 Summary", "", f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "| Arm | Task | Done | Success | Rate | Mean steps | Return follow | Target follow |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARM_NAME:
        for task in TASKS:
            subset = [r for r in rows if r["arm"] == arm and r["task"] == task]
            if not subset:
                continue
            successes = sum(r["success"] for r in subset)
            mean_steps = sum(r.get("steps") or 0 for r in subset) / len(subset)
            returns = [r["return_follow"] for r in subset if r["return_follow"] is not None]
            targets = [r["target_follow"] for r in subset if r["target_follow"] is not None]
            return_rate = f"{sum(returns)/len(returns):.1%}" if returns else "n/a"
            target_rate = f"{sum(targets)/len(targets):.1%}" if targets else "n/a"
            lines.append(f"| {arm} | {task} | {len(subset)} | {successes} | {successes/len(subset):.1%} | "
                         f"{mean_steps:.1f} | {return_rate} | {target_rate} |")
    lines += ["", "## Count by N", "", "| Arm | N2-4 | N5-7 | N8-10 |", "|---|---:|---:|---:|"]
    for arm in ARM_NAME:
        subset = [r for r in rows if r["arm"] == arm and r["task"] == "count"]
        cells = []
        for lo, hi in ((2, 4), (5, 7), (8, 10)):
            group = [r for r in subset if isinstance(r.get("N"), (int, float))
                     and r.get("N") >= lo and r.get("N") <= hi]
            cells.append(f"{sum(r['success'] for r in group)}/{len(group)}" if group else "n/a")
        lines.append(f"| {arm} | {' | '.join(cells)} |")
    lines += ["", "## Failure categories", "", "| Arm | Task | Category | Count |", "|---|---|---|---:|"]
    failures = Counter((r["arm"], r["task"], failure) for r in rows for failure in r["failures"])
    for (arm, task, failure), count in sorted(failures.items()):
        lines.append(f"| {arm} | {task} | {failure} | {count} |")
    lines += ["", "## Event counts", "", "| Arm | Task | Event | Count |", "|---|---|---|---:|"]
    events = Counter((r["arm"], r["task"], event) for r in rows for event, count in r["events"].items() for _ in range(count))
    for (arm, task, event), count in sorted(events.items()):
        lines.append(f"| {arm} | {task} | {event} | {count} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


def run_one(arm, gpu, task, episode):
    out = OUT / arm / task / f"ep{episode}"
    result_path = out / "result.json"
    if result_path.exists() and load_json(result_path, {}).get("outcome") != "error":
        return
    out.mkdir(parents=True, exist_ok=True)
    port = 29000 + gpu
    server_dir = OUT / "servers" / f"{arm}_gpu{gpu}"
    server_dir.mkdir(parents=True, exist_ok=True)
    context = server_dir / "context.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{ECM}:{ECM}/src:{ECM}/scripts:{ECM}/scripts/harvestmem_v3"}
    cmd = [SIM_PY, "-u", str(SCRIPT), "--port", str(port), "--task", task,
           "--episode", str(episode), "--arm", ARM_NAME[arm], "--checkpoint", CHECKPOINTS[arm],
           "--memory", ARM_MEMORY[arm], "--context", str(context), "--out", str(out)]
    with (out / "client.log").open("w") as log:
        try:
            subprocess.run(cmd, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
        except subprocess.TimeoutExpired:
            result_path.write_text(json.dumps({"task": task, "episode": episode, "outcome": "error",
                                               "error": "client timeout 1800 s"}) + "\n")


def lane(arm, gpu, queue):
    server_dir = OUT / "servers" / f"{arm}_gpu{gpu}"
    server_dir.mkdir(parents=True, exist_ok=True)
    port = 29000 + gpu
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src:{ECM}/src:{ECM}/scripts"}
    log = (server_dir / "server.log").open("w")
    proc = subprocess.Popen([POLICY_PY, "-u", str(ECM / "scripts/overnight_r12/serve.py"),
                             "--checkpoint", CHECKPOINTS[arm], "--port", str(port),
                             "--context", str(server_dir / "context.json")], cwd=POLICY, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                    if response.read() == b"OK\n":
                        break
            except OSError:
                time.sleep(3)
        while True:
            with LOCK:
                if not queue:
                    return
                task, episode = queue.popleft()
            for attempt in range(3):
                run_one(arm, gpu, task, episode)
                result = load_json(OUT / arm / task / f"ep{episode}" / "result.json", {})
                if result.get("outcome") != "error":
                    break
                time.sleep(2)
            with LOCK:
                summarize()
    finally:
        proc.kill()
        log.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queues = {arm: deque(EPISODES) for arm in ARM_NAME}
    threads = []
    for arm, gpus in GPU_LANES.items():
        for gpu in gpus:
            thread = threading.Thread(target=lane, args=(arm, gpu, queues[arm]), daemon=False)
            thread.start(); threads.append(thread)
    for thread in threads:
        thread.join()
    summarize()


if __name__ == "__main__":
    main()
