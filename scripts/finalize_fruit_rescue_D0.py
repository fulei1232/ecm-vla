#!/usr/bin/env python3
"""Verify frozen stage-0 registry and close D0; never generates or evaluates layouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run_fruit_rescue_splits import REGISTRY, inventory
from v_a_common import ECM, atomic_json, read_json, sha256, timestamp


def main():
    results = ECM / "results/fruit_lowlevel_rescue_D0_20261102"
    registry = read_json(REGISTRY)
    generation = read_json(results / "split_generation/status.json")
    if not registry.get("frozen") or generation.get("state") != "complete":
        raise SystemExit("Stage 0 generation is not complete; refusing to finalize")
    rows, missing = inventory()  # Only opaque file hashes and manifest identifiers; no val layout analysis.
    assert not missing and len(rows) == 180 and rows == registry["manifests"]
    list_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert list_hash == registry["manifest_list_sha256"]
    assert all(sha256(ECM / path) == digest for path, digest in registry["generator"]["source_sha256"].items())
    counts = {split: {task: sum(r["split"] == split and r["task"] == task for r in rows)
                      for task in ("count", "occluded_ripe")} for split in ("dev_gate", "sealed_val")}
    assert counts == {"dev_gate": {"count": 40, "occluded_ripe": 40},
                      "sealed_val": {"count": 50, "occluded_ripe": 50}}
    summary = read_json(ECM / "reports/lowlevel_rescue_D0_20261102/summary.json")
    assert all(s["script_sha256"] == sha256(ECM / "scripts/analyze_fruit_failure_modes.py")
               and not s["failed_runs"] and not s["missing_expected_episodes"]
               for s in summary["analyses"].values())
    summary.update(state="complete", updated_utc=timestamp(), blocking_condition=None,
                   next_step="phase 1 execution guard and horizon experiments on dev 200-219; stop at G0",
                   split_generation=generation,
                   split_registry={"path": str(REGISTRY.relative_to(ECM)), "frozen": True,
                                   "present_manifests": 180, "expected_manifests": 180,
                                   "manifest_list_sha256": list_hash, "counts": counts,
                                   "registry_sha256": sha256(REGISTRY)})
    validation = {"verified_utc": timestamp(), "manifests": 180, "missing": 0,
                  "payload_files_sha256_verified": 360, "counts": counts,
                  "manifest_list_sha256": list_hash, "registry_sha256": sha256(REGISTRY),
                  "legacy_source_hashes_match": True, "val_layouts_analyzed": False,
                  "val_policy_rollouts": 0}
    summary["validation"]["split_verification"] = validation
    atomic_json(results / "split_verification.json", validation)
    atomic_json(results / "summary.json", summary)
    atomic_json(ECM / "reports/lowlevel_rescue_D0_20261102/summary.json", summary)
    status = read_json(ECM / "reports/lowlevel_rescue_status.json")
    status.update(phase=0, state="complete", updated_utc=timestamp())
    status["links"]["split_verification"] = str((results / "split_verification.json").relative_to(ECM))
    atomic_json(ECM / "reports/lowlevel_rescue_status.json", status)

    report = ECM / "reports/lowlevel_rescue_D0_20261102.md"
    text = report.read_text()
    lines = text.splitlines()
    lines[0] = "阶段 0：完成；快照、统一复核、新划分生成与哈希冻结均已完成。"
    lines[3] = "已确认：新划分 180/180 manifest 及 360 个内容文件哈希通过校验；登记 frozen=true。"
    lines[4] = "决策：按 §4.0 自动进入阶段 1；阶段 1 的 G0 结束后停下等用户确认。"
    text = "\n".join(lines) + "\n"
    start = text.index("## 新划分登记与失败运行")
    end = text.index("失败记录（全部保留）：", start)
    text = text[:start] + f"""## 新划分登记与失败运行

登记文件：`reports/split_registry_20261102.json`，已冻结：`frozen=true`、180/180 manifest，文件列表 sha256 为 `{list_hash}`。独立复核全部 manifest 及其 360 个 HDF5/GT 内容文件哈希，生成器源文件哈希与原 legacy 版本一致。验证依据：`results/fruit_lowlevel_rescue_D0_20261102/split_verification.json`；manifest 列表及各 sha256 均见登记文件。

目标按 §4.0 明确范围执行：dev 220–259 两任务各 40 条，val 100–149 两任务各 50 条，合计 180 条。任务书 §8 的“120 条”与明确范围不符，未采用该估算数。val 仅生成并作文件完整性验证，未分析其布局、运行策略或调参。冻结后不重新生成。

用户从有 GPU 权限的终端以 `nohup setsid --wait` 续跑同一个脚本，实际生成成功，解除了先前环境阻塞。完整命令、成功及失败批次均保留于 `results/fruit_lowlevel_rescue_D0_20261102/split_generation/status.json` 及各批次日志；`plan.json` 记录原生成器源文件哈希。

""" + text[end:]
    start = text.index("## 按规则做出的决策")
    text = text[:start] + f"""## 按规则做出的决策

阶段 0 已完成，报告、全部指标 summary、冻结登记与状态文件齐备。未进行 G0/G1 判定。按 §4.0 自动进入阶段 1，使用现有 K0 在 dev 200–219 上比较 A0–A4；不使用新门槛集或封存验证集调试。阶段 1 的 G0 结果出来后，按任务书停下等用户确认。

## 下一步

实现默认关闭的关节安全包络与独立执行步长，计算专家训练示范分位数、做必要单元测试和记忆一致性审计，再完成 A0–A4。所有 GPU 运行需通过已授权且具备设备访问能力的执行环境启动。

机器可读全部指标：`reports/lowlevel_rescue_D0_20261102/summary.json`（版本化副本）及 `results/fruit_lowlevel_rescue_D0_20261102/summary.json`（运行副本）。当前冻结登记 sha256：`{sha256(REGISTRY)}`；校验证据见 `split_verification.json`。
"""
    report.write_text(text)
    print(json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
