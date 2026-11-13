#!/usr/bin/env python3
"""V-A analysis (plan §0A.3/§0A.6): selection audit, validity gates, paired bootstrap, four-state judgments.

    python3 scripts/analyze_v_a.py results/V_A_<run_id> [results/V_A_<extension_run_id> ...]

Writes analysis.json, per_episode.csv, selection_stats.csv, selection_audit.json and REPORT.md into the first
run directory. Several runs are merged (pre-registered seed extension); keys must not overlap.
"""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, SIM, VALID, read_json, read_jsonl, sha256, timestamp

sys.path.insert(0, str(ECM / "src"))
from memory_management.validation.online_variants import EpisodeState, choose
from memory_management.validation.partition import partition

PROTECTED_CONDITIONS = {"D4B", "RNDB", "RNDC", "JITB"}
PHASE_CONSERVING = {"D4B", "RNDB", "RNDC", "JITB"}
LABELS = {"degraded": "显著低于对照", "equivalent": "与对照等价（界限 δ）", "noninferior": "非劣于对照（界限 δ）",
          "undetermined": "未检测到差异，但不能排除 δ 以内的退化"}


def load(runs):
    protocol = read_json(runs[0] / "protocol.json")
    rows, planned = [], 0
    assert len(runs) <= 2, "only one preregistered extension is allowed"
    for index, run in enumerate(runs):
        own = read_json(run / "protocol.json")
        assert own["config"] == protocol["config"], "incompatible protocols"
        assert own["split"] == protocol["split"] == "val", "V-A must use val"
        plan = read_json(run / "plan.json")
        assert sha256(run / "plan.json") == own["plan_sha256"]
        key = lambda r: (r["task"], r["episode_id"], r["model_seed"], r["condition"])
        expected = {key(r) for r in plan}
        assert len(expected) == len(plan) == own["planned_episodes"], "invalid plan keys"
        if own.get("formal"):
            from run_v_a_closed_loop import verify
            verify(run)
            cfg = own["config"]
            assert own["tasks"] == cfg["tasks"] and own["episode_ids"] == list(range(cfg["episode_ids"]["start"], cfg["episode_ids"]["stop"]))
            if index == 0:
                assert own["experiment_stage"] == "primary" and own["model_seeds"] == cfg["model_seeds"]
                assert own["conditions"] == cfg["conditions"]
            else:
                assert protocol.get("formal") and own["experiment_stage"] == "extension"
                assert own["parent_protocol_sha256"] == sha256(runs[0] / "protocol.json")
                decision = read_json(runs[0] / "extension_decision.json")
                assert own["model_seeds"] == cfg["extension_seeds"] and own["conditions"] == decision["conditions"]
            cartesian = {(t, i, s, c) for t in own["tasks"] for i in own["episode_ids"]
                         for s in own["model_seeds"] for c in own["conditions"]}
            assert expected == cartesian, "formal plan has wrong scope"
        elif index:
            raise ValueError("cannot merge diagnostic runs as a formal extension")
        planned += own["planned_episodes"]
        for path in sorted(run.glob("server*/attempts.jsonl")):
            server = read_json(path.parent / "config.json")
            for row in read_jsonl(path):
                assert key(row) in expected, "attempt outside sealed plan"
                assert row["model_seed"] == server["model_seed"], "wrong server seed"
                directory = run / "episodes" / row["condition"] / row["task"] / f"ep{row['episode_id']:02d}_seed{row['model_seed']}"
                assert Path(row["attempt_directory"]).resolve() == directory.resolve()
                durable = read_json(directory / "result.json")
                assert all(row[k] == v for k, v in durable.items()), "durable result differs from attempts"
                if row["outcome"] in VALID:
                    metadata = read_json(SIM / f"src/robomme/env_metadata/val/record_dataset_{row['task']}_metadata.json")
                    match = [r for r in metadata["records"] if r["task"] == row["task"] and r["episode"] == row["episode_id"]]
                    assert len(match) == 1 and row["environment_seed"] == match[0]["seed"], "wrong environment seed"
                    assert row["difficulty"] == match[0]["difficulty"], "wrong difficulty"
                rows.append(row)
    keys = [(r["task"], r["episode_id"], r["model_seed"], r["condition"]) for r in rows]
    assert len(keys) == len(set(keys)), "overlapping attempts across runs"
    return protocol, rows, planned


def audit_episode(row):
    """Per-decision selection checks (plan §0A.3 validity gate 3), within the episode's own trajectory."""
    path = Path(row["attempt_directory"]) / "policy_decisions.jsonl"
    failures, decisions = [], read_jsonl(path)
    if not decisions:
        return decisions, [{"check": "missing_policy_decisions", "where": row["attempt_directory"]}]
    raw = read_jsonl(path.parent / "raw_frames.jsonl")
    state = EpisodeState(row["task"], row["episode_id"], row["model_seed"], decisions[0]["exec_start_idx"])
    if not raw or [r["frame_id"] for r in raw] != list(range(len(raw))):
        return decisions, [{"check": "missing_or_noncontiguous_raw_frames", "where": row["attempt_directory"]}]
    for d in decisions:
        v, sel, F = d["v_a"], d["selected_frame_ids"], d["available_max_frame_id"]
        c, where = row["condition"], (row["task"], row["episode_id"], row["model_seed"], row["condition"], F)
        full = len(v["d0_frame_ids"]) == 32 and F >= 32
        checks = {
            "count": len(sel) == min(F + 1, 32),
            "sorted_unique_causal": sel == sorted(set(sel)) and all(0 <= s <= F for s in sel),
            "official_d0": v["d0_frame_ids"] == list(partition(F).uniform),
            "source_positions": d["source_position_frame_ids"] == sel,
            "budget": d["token_budget"] == 512 and d["tokens_per_frame"] == 16 and d["num_views"] == 1
                      and d["valid_tokens"] == 16 * len(sel),
            "logged_selection": sel == sorted(set(v["d0_frame_ids"]) - set(v["deleted_d0_frames"])
                                              | set(v["added_frames"])),
        }
        if c in ("D0a", "D0b"):
            checks["d0_identity"] = sel == v["d0_frame_ids"]
        if c in PROTECTED_CONDITIONS and full:
            checks["protected"] = bool(v["protected_present"]) and partition(F).protected <= set(sel)
        if F >= len(raw):
            checks["raw_causal_coverage"] = False
        else:
            checks["selected_front_hashes"] = d["selected_front_sha256"] == [raw[i]["front_sha256"] for i in sel]
            causal = raw[:F + 1] if c in {"D4L", "D4B", "RNDB", "RNDC"} else []
            replay_sel, replay = choose(c, v["d0_frame_ids"], F, causal, state)
            checks["causal_selection_replay"] = replay_sel == sel
            checks["pins_replay"] = replay.get("pins", {}) == v.get("pins", {})
            checks["decision_sequence"] = d["decision_idx"] == v["decision_idx"] == replay["decision_idx"]
            checks["exec_start_constant"] = d["exec_start_idx"] == state.exec_start_idx
        if c == "D4B" and not v["pin_overflow"]:
            checks["pins_selected"] = set(v["pins"].values()) <= set(sel)
        active = not v["pin_overflow"] and not v.get("rndb_infeasible") and "d4b_deleted_d0_frames" in v
        if c == "RNDB" and active:
            checks["rndb_deleted_equals_d4b"] = v["deleted_d0_frames"] == v["d4b_deleted_d0_frames"]
        if c == "RNDC" and active:
            checks["rndc_deleted_count"] = len(v["deleted_d0_frames"]) == len(v["d4b_deleted_d0_frames"])
        if c in PHASE_CONSERVING and v.get("phase_borrow", 0) == 0:
            checks["phase_counts"] = (v["n_demo_slots"], v["n_exec_slots"]) == (v["n_demo_slots_d0"], v["n_exec_slots_d0"])
        failures += [{"check": k, "where": where} for k, ok in checks.items() if not ok]
    return decisions, failures


def selection_table(rows, decisions_by_key):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["task"], r["condition"])
        for d in decisions_by_key.get(id(r), []):
            v = d["v_a"]
            if len(v["d0_frame_ids"]) < 32:
                continue
            a = acc[key]
            a["replaced"].append(len(v["added_frames"]))
            a["max_gap"].append(v["max_gap"])
            a["cur_prev_gap"].append(v["current_to_prev_gap"])
            a["demo_slots"].append(v["n_demo_slots"])
            a["exec_slots"].append(v["n_exec_slots"])
            a["overflow"].append(bool(v["pin_overflow"]))
            a["borrow"].append(v.get("phase_borrow", 0) != 0)
            a["protected"].append(bool(v["protected_present"]))
            if "n_pin_frames" in v:
                a["pins"].append(v["n_pin_frames"])
            a["evidence_overlap"].extend(bool(x["evidence_overlap"]) for x in v.get("replacements", []))
            if "deleted_jaccard_vs_d4b" in v and v["deleted_d0_frames"]:
                a["jaccard_vs_d4b"].append(v["deleted_jaccard_vs_d4b"])
    out = []
    for (task, cond), a in sorted(acc.items()):
        rec = {"task": task, "condition": cond, "full_decisions": len(a["max_gap"])}
        for k in ("replaced", "max_gap", "cur_prev_gap", "demo_slots", "exec_slots", "pins", "jaccard_vs_d4b"):
            rec[f"{k}_median"] = float(np.median(a[k])) if a[k] else None
        for k in ("overflow", "borrow", "protected", "evidence_overlap"):
            rec[f"{k}_rate"] = float(np.mean(a[k])) if a[k] else None
        out.append(rec)
    return out


class PairedBootstrap:
    """Stratified paired bootstrap: resample IDs within task, keep all seeds of a drawn ID."""

    def __init__(self, outcomes, tasks, draws, seed=20261022):
        self.outcomes, self.tasks = outcomes, tasks
        rng = np.random.default_rng(seed)
        self.ids = {t: sorted({i for (tt, i, _s, _c) in outcomes if tt == t}) for t in tasks}
        self.index = {t: rng.integers(0, len(self.ids[t]), size=(draws, len(self.ids[t]))) for t in tasks}

    def task_arrays(self, task, a, b, value):
        sums, counts = [], []
        for i in self.ids[task]:
            diffs = [value(self.outcomes[(task, i, s, a)]) - value(self.outcomes[(task, i, s, b)])
                     for (t, ii, s, c) in self.outcomes if t == task and ii == i and c == a
                     and (task, i, s, b) in self.outcomes]
            sums.append(sum(diffs))
            counts.append(len(diffs))
        return np.asarray(sums, float), np.asarray(counts, float)

    def contrast(self, a, b, tasks=None, value=lambda r: float(r == "success")):
        tasks = self.tasks if tasks is None else tasks
        point, boot, per_task = [], [], {}
        for t in tasks:
            sums, counts = self.task_arrays(t, a, b, value)
            if counts.sum() == 0:
                continue
            est = sums.sum() / counts.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                draws = sums[self.index[t]].sum(1) / counts[self.index[t]].sum(1)
            point.append(est)
            boot.append(draws)
            per_task[t] = {"delta": est, "n_pairs": int(counts.sum()), **intervals(draws)}
        if not boot:
            return None
        stacked = np.vstack(boot)
        # A resample containing no valid pair for a task is undefined. Keep the
        # preregistered equal task weights by dropping that draw for every task.
        usable = np.isfinite(stacked).all(axis=0)
        if not usable.any():
            return None
        pooled = stacked[:, usable].mean(axis=0)
        delta = float(np.mean(point))
        p = float(min(1.0, 2 * min(np.nanmean(pooled <= 0), np.nanmean(pooled >= 0))))
        return {"delta": delta, **intervals(pooled), "p_two_sided": p, "tasks": list(per_task), "per_task": per_task}


def intervals(draws):
    q = np.nanpercentile(draws, [2.5, 5, 95, 97.5])
    return {"ci95": [float(q[0]), float(q[3])], "ci90": [float(q[1]), float(q[2])]}


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def judge(res, margin, alpha=0.05):
    lo90, hi90 = res["ci90"]
    flags = {"degraded": res["ci95"][1] < 0 and res["p_holm"] < alpha,
             "noninferior": lo90 > -margin, "equivalent": lo90 > -margin and hi90 < margin}
    # A significant degradation inside the margin is still routed as a degradation.
    state = ("degraded" if flags["degraded"] else "equivalent" if flags["equivalent"]
             else "noninferior" if flags["noninferior"] else "undetermined")
    return {**flags, "state": state, "wording": LABELS[state]}


def route(judgments):
    c3, c4 = judgments["C3"]["state"], judgments["C4"]["state"]
    if "degraded" in (c3, c4):
        return "有退化"
    if c3 in ("noninferior", "equivalent") and c4 in ("noninferior", "equivalent"):
        return "无明显退化"
    return "未能判定"


def progress(row):
    if row["outcome"] == "success":
        return 1.0
    n, k = row.get("n_subtasks") or 0, row.get("final_task_index")
    return float(k) / n if n and isinstance(k, (int, float)) and k >= 0 else 0.0


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    opts = p.parse_args()
    runs = [r.resolve() for r in opts.runs]
    protocol, rows, planned = load(runs)
    cfg = protocol["config"]
    stats, margin = cfg["stats"], cfg["stats"]["margin_pp"] / 100
    out_dir = runs[0]

    decisions_by_key, audit_failures = {}, []
    for r in rows:
        if r["outcome"] in VALID:
            try:
                decisions_by_key[id(r)], fails = audit_episode(r)
            except (KeyError, ValueError, IndexError, OSError) as exc:
                fails = [{"check": "unreadable_selection_evidence", "where": r["attempt_directory"], "error": str(exc)}]
            audit_failures += fails
    errors = [r for r in rows if r["outcome"] not in VALID]
    outcomes = {(r["task"], r["episode_id"], r["model_seed"], r["condition"]): r["outcome"]
                for r in rows if r["outcome"] in VALID}
    by_key = {(r["task"], r["episode_id"], r["model_seed"], r["condition"]): r for r in rows if r["outcome"] in VALID}
    tasks = [t for t in protocol["tasks"] if any(k[0] == t for k in outcomes)]
    boot = PairedBootstrap(outcomes, tasks, stats["bootstrap_draws"])

    contrasts = {}
    for name, (a, b) in {**stats["contrasts"], "C0": stats["aa_contrast"], **stats.get("secondary_contrasts", {})}.items():
        if not ({a, b} <= {k[3] for k in outcomes}):
            continue
        scope = [t for t in stats.get("c2_pooled_tasks", tasks) if t in tasks] if name in ("C2", "C2e") else None
        comparison = boot.contrast(a, b, scope)
        if comparison is not None:
            contrasts[name] = {"a": a, "b": b, **comparison}
    family = [n for n in stats["contrasts"] if n in contrasts]
    for name, padj in holm({n: contrasts[n]["p_two_sided"] for n in family}).items():
        contrasts[name]["p_holm"] = padj
    judgments = {n: judge(contrasts[n], margin) for n in family}

    secondary = {}
    for name in family:
        a, b = contrasts[name]["a"], contrasts[name]["b"]
        boot.outcomes = {k: by_key[k] for k in outcomes}
        secondary[name] = {
            "progress": boot.contrast(a, b, contrasts[name]["tasks"], value=progress),
            "timeout": boot.contrast(a, b, contrasts[name]["tasks"], value=lambda r: float(r["outcome"] == "timeout")),
        }
        boot.outcomes = outcomes
    c0 = contrasts.get("C0")
    validity = {
        "formal": bool(protocol.get("formal")),
        "error_rate": len(errors) / max(1, len(rows)),
        "error_rate_ok": len(errors) / max(1, len(rows)) <= stats["validity"]["max_error_rate"],
        "completed": len(rows), "planned": planned, "complete": len(rows) == planned,
        "aa_ok": bool(c0 and c0["ci95"][0] <= 0 <= c0["ci95"][1]
                      and abs(c0["delta"]) <= stats["validity"]["max_abs_aa_pp"] / 100),
        "selection_audit_failures": len(audit_failures), "selection_audit_ok": not audit_failures,
    }
    validity["valid"] = all(validity[k] for k in ("formal", "error_rate_ok", "complete", "aa_ok", "selection_audit_ok"))
    extension = sorted({c for n, j in judgments.items() if j["state"] == "undetermined"
                        for c in (contrasts[n]["a"], contrasts[n]["b"])})
    result = {"generated_utc": timestamp(), "runs": [str(r) for r in runs],
              "protocol_sha256": [sha256(r / "protocol.json") for r in runs], "extension_used": len(runs) > 1,
              "margin": margin, "tasks": tasks,
              "validity": validity, "contrasts": contrasts, "judgments": judgments, "secondary": secondary,
              "v_a_result": route(judgments) if validity["valid"] and {"C3", "C4"} <= set(judgments) else None,
              "preregistered_extension_conditions": extension if len(runs) == 1 else [],
              "remaining_undetermined_conditions": extension,
              "secondary_glmm": {"status": "not_fitted", "reason": "R/lme4 and statsmodels unavailable; primary inference uses preregistered paired bootstrap"}}

    table = []
    for task in tasks:
        for cond in protocol["conditions"]:
            own = [r for r in rows if r["task"] == task and r["condition"] == cond]
            ok = [r for r in own if r["outcome"] in VALID]
            table.append({"task": task, "condition": cond, "n": len(own), "errors": len(own) - len(ok),
                          "success_rate": np.mean([r["outcome"] == "success" for r in ok]) if ok else None,
                          "timeout_rate": np.mean([r["outcome"] == "timeout" for r in ok]) if ok else None,
                          "mean_progress": np.mean([progress(r) for r in ok]) if ok else None})
            successes = [r for r in ok if r["outcome"] == "success" and r.get("executed_steps") is not None]
            table[-1]["success_completion_steps_mean"] = float(np.mean([r["executed_steps"] for r in successes])) if successes else None
    result["per_task_condition"] = [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in t.items()}
                                    for t in table]
    sel = selection_table(rows, decisions_by_key)
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out_dir / "selection_audit.json").write_text(json.dumps(audit_failures[:2000], indent=1))
    write_csv(out_dir / "selection_stats.csv", sel, list(sel[0]) if sel else [])
    write_csv(out_dir / "per_episode.csv", rows, ["task", "episode_id", "model_seed", "condition", "outcome",
              "environment_seed", "difficulty", "final_task_index", "n_subtasks", "raw_frames", "exec_start_idx",
              "executed_steps", "elapsed_seconds", "server", "attempt_directory", "error"])
    write_report(out_dir / "REPORT.md", result, table, sel)
    print(json.dumps({"validity": validity, "v_a_result": result["v_a_result"],
                      "judgments": {n: j["state"] for n, j in judgments.items()},
                      **{n: {"delta": round(c["delta"], 4), "ci90": [round(x, 4) for x in c["ci90"]],
                             "ci95": [round(x, 4) for x in c["ci95"]]} for n, c in contrasts.items()}},
                     ensure_ascii=False, indent=1))
    return 0


def pp(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def write_report(path, result, table, sel):
    v, lines = result["validity"], []
    lines += ["# V-A 闭环对照报告", "", f"生成：{result['generated_utc']}；runs：{', '.join(Path(r).name for r in result['runs'])}", "",
              "## 有效性", "",
              f"- 完成 {v['completed']}/{v['planned']}；工程错误率 {100 * v['error_rate']:.2f}%（门槛 ≤ 2%）。",
              f"- A/A 对照 C0 通过：{v['aa_ok']}；选帧审计失败 {v['selection_audit_failures']} 项。",
              f"- 本轮结果{'有效' if v['valid'] else '无效，先排查原因，不得引用下面的判定'}。", "",
              "## 主对比（成功率差，百分点；δ = %.0f）" % (100 * result["margin"]), "",
              "| 对比 | Δ | 90% CI | 95% CI | Holm p | 判定 | 允许的表述 |", "|---|---|---|---|---|---|---|"]
    for name, c in result["contrasts"].items():
        j = result["judgments"].get(name)
        lines.append(f"| {name} = {c['a']} − {c['b']} | {pp(c['delta'])} | [{pp(c['ci90'][0])}, {pp(c['ci90'][1])}] | "
                     f"[{pp(c['ci95'][0])}, {pp(c['ci95'][1])}] | {c.get('p_holm', float('nan')):.3f} | "
                     f"{j['state'] if j else '次要/对照'} | {j['wording'] if j else ''} |")
    lines += ["", f"**V-A 结果（§0A.6）**：{result['v_a_result']}。"]
    if result["extension_used"]:
        lines.append("已用完预注册的一次性追加 seed {10, 11}；仍未能判定时按保守路线先分析，不再追加。")
    elif result["validity"]["valid"] and result["preregistered_extension_conditions"]:
        lines.append(f"未能判定的对比需按预注册追加 seed {{10, 11}}，涉及条件：{', '.join(result['preregistered_extension_conditions'])}。")
    lines += ["", "## 每任务成功率", "", "| 任务 | 条件 | n | 错误 | 成功率 | timeout 率 | 平均进度 | 成功场平均步数 |", "|---|---|---|---|---|---|---|---|"]
    for t in table:
        f = lambda x: "—" if x is None else f"{100 * x:.1f}%"
        lines.append(f"| {t['task']} | {t['condition']} | {t['n']} | {t['errors']} | {f(t['success_rate'])} | "
                     f"{f(t['timeout_rate'])} | {f(t['mean_progress'])} | {t['success_completion_steps_mean']} |")
    lines += ["", "## 选帧统计（满记忆决策）", "", "| 任务 | 条件 | 替换数 | 最大间隔 | 当前帧间隔 | 演示/执行槽 | 溢出率 | 借槽率 | 证据重叠率 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in sel:
        g = lambda k: "—" if s[k] is None else f"{s[k]:.3g}"
        lines.append(f"| {s['task']} | {s['condition']} | {g('replaced_median')} | {g('max_gap_median')} | "
                     f"{g('cur_prev_gap_median')} | {g('demo_slots_median')}/{g('exec_slots_median')} | "
                     f"{g('overflow_rate')} | {g('borrow_rate')} | {g('evidence_overlap_rate')} |")
    lines += ["", "## 分析范围", "", "主推断为预注册的分层配对 bootstrap；工程错误成对剔除，timeout 计为失败。",
              "次要 logistic GLMM 尚未拟合（当前无 R/lme4 或 statsmodels）；不影响预注册主对比路由，但不能引用 GLMM 结论。"]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
