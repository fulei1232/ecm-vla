# Teacher Oracle 第 3 步扩展：BinFill 与 VideoUnmaskSwap（2026-10-24）

状态：方案，运行前封存。只使用 val split；V-A 结果目录只读。前提：第 2b 步（`results/phase_preflight_20261024`）两个任务的 P0–P4 全部通过；PatternLock 试跑（`results/teacher_oracle_20261024`）已完成。本方案在看到任何扩展结果之前写定。

## 1. 目的

PatternLock 试跑的结果：TO − S0 = +5.0 个百分点，90% CI [+1.7, +10.0]，按规则判为"没有可见空间"。方向是一致的：成功率 TO > S0 ≈ RS > AO。不过 PatternLock 的基线已经是 68%，可提升的空间有限。

BinFill 和 VideoUnmaskSwap 在 V-A 中的基线分别只有 40% 和 27%，记忆依赖也更强。扩展要回答同一个问题：特权老师每个决策换一帧记忆，能否提高成功率。

## 2. 与试跑方案相同的部分

以下内容与 `reports/teacher_oracle_pilot_protocol_20261024.md` 完全相同，不再重复：

- 第 3 节：五个条件 U0/S0/TO/AO/RS 的定义，以及 RS 在同组 TO 之后运行；
- 第 4 节：专家动作块的流程，包括方法 R 重放、逐位起点检查、20 步预算、可用条件和补齐规则；
- 第 5 节：候选（8 个候选帧 × 27 个历史槽）、记忆不跨决策累积、FP32 老师、CRN、16 个样本、批大小 16、接受规则（2 倍标准误）和四项审计；
- 第 6 节：服务器布局（6 个）、同组固定在同一服务器进程、续跑规则、关闭服务端心跳；
- 第 7 节：有效性门槛 G0–G5 及其阈值；
- 第 8 节：统计方法、δ = 12 个百分点、四种状态的定义和判定规则；
- 第 9 节：诊断项。

## 3. 不同之处

- **任务与规模**：BinFill 和 VideoUnmaskSwap，val episode 0–19，种子 7、8、9，共 120 组。
- **专家**：第 2b 步的阶段专家（`expert = "phase"`），参数与第 2b 步相同。
- **损失维度**：两个任务都有夹爪，打分算全部 8 维，与 V-B 口径相同。
- **单集墙钟上限**：14400 s。BinFill 在 V-A 中最长的 episode 有 1195 步，而每个决策都要分支和打分。
- **统计**：
  - 主对比把两个任务合并，任务等权；在任务内按 episode 重抽样、保留同一 episode 的全部种子，与 V-A 相同。C1–C3 三项做 Holm 校正，判定规则用合并结果。
  - 次要结果：两个任务分别报告 C1–C3，6 项一起做 Holm 校正。次要结果不改变判定。
- **冒烟**：每个任务一组，取种子 7 下 V-A D0a 执行步数最短的 episode；种子 7 的 2 个服务器并行跑。
- **成本**：BinFill 每个条件约 2000 个决策，每个打分决策要分支加打分约 30–50 s。6 个服务器估计 18–20 小时。

## 4. 命令与代码

```bash
python3 scripts/run_teacher_oracle.py all --stage extension --smoke --out results/teacher_oracle_ext_smoke_20261024
python3 scripts/run_teacher_oracle.py all --stage extension --out results/teacher_oracle_ext_20261024
```

代码与试跑相同。`run_teacher_oracle.py` 增加 `--stage`，用来选择任务、专家、墙钟上限和冒烟规则；试跑的参数不变。

## 5. 冒烟测试记录（正式运行前）

**运行情况**
- 目录：`results/teacher_oracle_ext_smoke_20261024`，2026-10-24 09:08–09:24（北京时间）。
- 两组按第 3 节的规则选取：BinFill ep01（V-A 253 步）、VideoUnmaskSwap ep11（98 步），种子 7，各占一个服务器。

**门槛与检查**
- G0–G5 全部通过。工程错误 0；专家可用率 100%（70/70 个决策）。
- 两组的 S0 与 U0 都逐位一致。
- 70 次分支起点全部逐位一致。
- 61 个打分决策的四项审计全部通过；缓存与完整前向的相对误差为 0；损失维度都是 8。
- 每组的五个条件都在同一个服务器进程内完成。

**耗时与成本**
- 分支耗时中位数 3.4 s，最长 4.8 s；打分中位数 14.4 s。
- BinFill 一个打分 episode（约 16 个决策）约 4.5–5 分钟。
- 按此估算，正式运行约 7–9 小时，短于第 3 节的估计。

**第 4 节的更正**
- 除 `run_teacher_oracle.py` 外，`eval_teacher_episode.py` 也有改动：它把各任务对应的专家（`params["experts"][task]`）传给分支进程；参数里没有 `experts` 时，行为与试跑相同。
- 改动在冒烟封存之前完成，冒烟时已经生效。
- 用改动后的代码重建试跑计划，与已封存的 `plan.json` 完全相同。在临时目录重新分析试跑结果，门槛、对比、各条件和诊断也全部相同。试跑的结果文件没有改动。

**冒烟后的改动**
- 冒烟后没有代码或参数改动；冒烟前后代码哈希一致。
- 正式运行使用新目录 `results/teacher_oracle_ext_20261024`。
