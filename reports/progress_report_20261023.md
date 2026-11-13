# Event-Aware Counterfactual Memory 项目进度报告

**报告日期：2026-10-23**（2026-10-24 有更新，见第十节）  
**项目目录：** `/media/fulei/fuleivla/ecm_vla`

## 一、结论

当前项目已经完成前置验证 Milestone V 和 Milestone 1，研究链路可以继续向事件检测、teacher 标签和价值预测器推进。

已经得到的可靠结论是：

1. **V-A 正式闭环有效。** 1260/1260 场完成，工程错误率为 0%，逐决策选帧审计全部通过。非均匀记忆分布在当前冻结策略上没有显示出超过预注册界限的明显退化。
2. **V-B 正式 teacher 探针有效。** 111/111 个合法边界、37 条专家示范、1,629,264 行原始损失完成；最小的单槽替换设置 `k=1` 在四个任务上同时满足可靠性 P2 和正收益 P3。
3. **Milestone 1 工程实现和验收完成。** 有界记忆、FIFO/Random 替换、策略实例接入、CPU/存储/真实 checkpoint 回归和 6 场 GPU 仿真验收均通过。
4. **目前不能声称任务成功率已经提升。** V-B 的正收益是反事实 flow loss 上的 teacher 信号，不是闭环成功率；V-A 也只证明记忆分布没有明显退化。成功率提升需要后续训练价值预测器并进行冻结策略闭环比较。

因此，当前项目状态是：**基础设施和可学习性前置证据已完成，下一步进入 Milestone 2：周期候选与因果事件检测。**

## 二、项目目标和验证顺序

项目目标是让 VLA 策略在固定记忆预算下，根据事件重要性选择、保留和替换历史帧。验证顺序按照实施计划执行：

1. 先用 V-A 验证非均匀选帧不会明显破坏冻结策略闭环；
2. 用 V-B 验证“替换哪些记忆槽会改变 teacher loss”，并确认该信号可靠且有正收益替换；
3. 实现不带学习的有界记忆框架；
4. 生成周期和事件候选；
5. 生成 teacher 标签并训练 value predictor；
6. 最后才进行冻结基础策略的正式闭环比较。

这个顺序的意义是先验证研究假设和工程正确性，再评估性能，避免把离线损失变化直接解释成机器人任务成功率提升。

## 三、Milestone V：前置验证

### 3.1 V-A：平衡选帧闭环对照

**结果目录：** `results/V_A_formal_20261023/`

- 任务：BinFill、VideoUnmaskSwap、PatternLock；
- 条件：7 种记忆/选帧条件；
- 每个任务和条件 60 个配对样本；
- 总运行量：1260/1260 场；
- 工程错误率：0%；
- A/A 对照 C0：通过；
- 逐决策选帧审计失败：0 项。

主要预注册对比为成功率差，单位为百分点：

| 对比 | 结果 | 判定 |
|---|---:|---|
| C1 = D4B − D4L | +13.9pp，95% CI [5.6, 21.7] | 非劣于对照 |
| C2 = D4B − RNDC | −1.7pp，95% CI [−7.5, 4.2] | 等价 |
| C3 = JITB − D0a | −1.7pp，95% CI [−5.6, 1.7] | 等价 |
| C4 = D4B − D0a | −3.3pp，95% CI [−7.2, 0.0] | 等价 |
| C0 = D0b − D0a | −0.6pp，95% CI [−2.8, 1.7] | A/A 对照正常 |

V-A 的正式结论是：**无明显退化**。这不是“性能提升”的结论，而是说明后续可以在冻结策略上继续研究记忆管理，不需要因为非均匀记忆分布立即进行策略适配。

各任务基线 D0a 成功率为：BinFill 40.0%、VideoUnmaskSwap 26.7%、PatternLock 71.7%。这些数值是当前实验协议下的基线，不代表所有任务或最终系统的总体成功率。

### 3.2 V-B：反事实 teacher 信号

**结果目录：** `results/V_B_fp32_full_20261022/`

- 合法边界：111/111；
- 专家示范：37 条；
- 原始损失记录：1,629,264 行；
- batch cached/full 正确性检查：运行中检查最大相对误差为 0；
- identity、repeat、CRN、causal 等正确性审计：通过；
- 最小通过设置：`k=1`。

`k=1` 在四个任务上的结果：

| 任务 | S1 排序可靠性 | SNR | 敏感边界比例 | 交叉拟合相对收益 | P2/P3 |
|---|---:|---:|---:|---:|---|
| BinFill | 0.991 | 33.65 | 60.0% | 10.5% [4.6%, 19.4%] | 通过/通过 |
| VideoUnmaskSwap | 0.972 | 8.12 | 90.0% | 2.5% [1.5%, 4.1%] | 通过/通过 |
| PatternLock | 0.983 | 25.40 | 61.9% | 12.4% [6.6%, 18.3%] | 通过/通过 |
| PickHighlight | 0.985 | 17.67 | 70.0% | 5.0% [3.1%, 7.0%] | 通过/通过 |

V-B 说明：teacher 能稳定识别部分有价值的记忆替换，而且单槽路线已经具备进入 value predictor 的信号基础。`k=4` 在 VideoUnmaskSwap 和 PatternLock 上没有通过，说明多槽收益不能简单假设为单槽收益相加，后续优先采用单槽或显式建模交互的路线。

V-B 的限制：探针主要在专家观测轨迹上运行；它使用 FP32 loss 探针和 BF16 checkpoint 恢复，不能直接等同于在线闭环成功率；BF16 计算本身的 cached loss 数值等价性尚未作为通过条件。因此 V-B 不能单独支持“成功率提升”或“最终策略有效”的表述。

### 3.3 V-A/V-B 联合决策

**联合路由：** `go_value_predictor`

含义是：V-A 没有发现明显闭环退化，V-B 又发现稳定且存在正收益的替换，因此按计划进入价值预测器路线；暂不触发策略适配 Milestone P。

联合证据：`results/V_A_formal_20261023/JOINT_REPORT.md`。

## 四、Milestone 1：有界记忆框架

**验收目录：** `results/M1_acceptance_20261023/`

### 4.1 已实现模块

代码位于 `src/memory_management/`，包括：

- `FrameRecord`：保存原始帧 ID、时间和只读特征；
- `RollingObservationBuffer`：连续流滚动缓存；
- `MemorySlot`、`BoundedMemoryBank`：固定容量记忆槽和事务式替换；
- `ManagedMemoryPacker`：将受管记忆打包成官方模型输入；
- `OnlineMemoryController`：在线接收、更新、准备和 reset；
- FIFO / Random 两种替换模式；
- `policy_adapter`：按策略实例接入，不修改基础模型接口；
- `baseline_framesamp` feature flag：可保持原始基线行为。

### 4.2 固定预算和模型接口

- memory bank：32 槽；
- rolling buffer：160 帧；
- 每帧：16 tokens；
- 最多持有约 192 个不同 `FrameRecord`；
- 长流输入 6201 帧压力测试仍保持有界；
- 支持超过 4096 帧的原始位置编码；
- 官方四张量接口保持兼容：
  - `static_image_emb`: `[512, 2048]`
  - `static_pos_emb`: `[512, 768]`
  - `static_state_emb`: `[512, 8]`
  - `static_mask`: `[512]`

保守的常驻特征上限记录为 22,026,240 bytes，约 21 MiB。该数字只统计记忆特征，不包含 checkpoint、模型工作区、JAX 编译缓存、传输中的视频和日志。

### 4.3 M1 验收结果

`acceptance.json` 的关键字段：

```json
{
  "passed": true,
  "completed": 6,
  "planned": 6,
  "engineering_errors": 0,
  "baseline_regression_passed": true,
  "cpu_storage_passed": true,
  "cpu_tests_and_provenance_passed": true,
  "both_modes_exercised_replacements": true
}
```

6 场 GPU 仿真 episode 全部完成逐决策工程审计：

- FIFO：BinFill、VideoUnmaskSwap、PatternLock 各 1 场；
- Random：BinFill、VideoUnmaskSwap、PatternLock 各 1 场；
- FIFO 和 Random 都实际发生了记忆替换；
- 每场的预算、阶段槽位、因果时间顺序、输入 shape、内存上限和日志完整性检查均通过。

这 6 场任务的 success/fail 是仿真任务结果，不是 M1 工程验收结果，也不是成功率实验。工程验收通过的依据是 `acceptance.json` 中的审计字段。

### 4.4 已修复的工程问题

第一次 GPU 运行中，VideoUnmaskSwap 的后续 websocket buffer 包把 `exec_start_idx` 写成 0。旧适配器把 0 误判成执行边界变化，导致第二个 buffer 包失败。

修复后改为：只有收到正的 `exec_start_idx` 才更新或校验边界；后续 0 沿用当前 episode 边界。修复提交为：

- `22212f0`：修复 websocket execution boundary；
- `7cdd0d5`：补充回归说明和文档。

旧失败目录 `results/M1_acceptance_20261023_boundary_bug/` 保留作故障诊断证据，不作为最终结果。

## 五、代码、测试和可复现性

ECM 目录已建立独立 Git 历史，当前关键提交为：

```text
795ca99  Snapshot completed Milestone V before bounded memory implementation
cc37ae9  Implement Milestone 1 bounded memory and sealed acceptance pipeline
22212f0  Handle omitted execution boundary in websocket buffers
7cdd0d5  Document websocket boundary regression fix
```

M1 验收中已执行：

- CPU 核心测试和旧验证回归：72 passed、1 skipped；
- policy venv 中的官方 packer / 小模型缓存回归：8 passed；
- 长轨迹存储压力测试；
- 真实 checkpoint baseline 动作回归；
- FIFO/Random 各 3 场 GPU 闭环；
- 协议、代码快照、依赖和结果 provenance 绑定。

被跳过的 1 项是系统 Python 缺少 JAX/Flax 依赖的测试；相同检查已在 policy venv 中实际执行并通过。

## 六、当前尚未完成的内容

以下内容目前没有被项目结果证明：

1. 事件检测器是否能稳定找到真实因果事件；
2. 在线事件候选与离线因果重放是否逐事件一致；
3. value predictor 是否能超过常数预测、视觉相似度和简单启发式；
4. learned manager 是否能在闭环中提升任务成功率；
5. 是否需要策略适配或微调；
6. 端到端时延、显存和大规模测试集上的最终性能。

特别需要避免的表述：不能把 V-B 的 2.5%–12.4% teacher 相对收益写成任务成功率提升，也不能把 M1 的 6 场 success/fail 结果写成总体性能结论。

## 七、下一步工作

### Milestone 2：周期候选与因果事件检测（工程实现已完成）

M2 已实现运动学显著性、因果 peak、视觉去重、周期/事件候选合并、离线重放、在线 controller 接入和诊断输出。合成因果验收与在线/离线 event ID 一致性通过，memory-management 测试为 `80 passed, 2 skipped`。

真实 M1 轨迹的 annotation-only 诊断发现固定尺度会产生大量连续误触发；自适应尺度可降低误触发，但由于没有人工事件真值，尚不能把该诊断当作正式 precision/recall。M2 正式验收还需要先确认 state 单位和采样频率，并补充人工或可靠任务事件标注、检测延迟和漏检统计。

### 后续里程碑

- **Milestone 3：** 正式单标签已通过。BinFill train episode 0、cutoff 240，候选帧 29 替换 slot 0 原帧 7；5 个变体、8 个 offsets、16 个共同随机种子共 640 行。base loss mean=`0.0215165305`，actual replacement loss mean=`0.0215131901`，折扣 gain mean=`1.0554e-4`；identity gain 最大绝对值为 0。CRN、重复性、mask、shape、因果性和无未来泄漏审计全部通过，结果为 `formal=true`。下一步进入 Milestone 4 批量标签生成与可靠性诊断；
- **Milestone 4：** BinFill train pilot 已完成：6/6 边界、864 个 k=1 槽位干预。k=1 的正 gain 比例为 1.000、负 gain 比例为 0、>2SE 比例为 1.000；split-half Spearman-Brown 中位数 0.978，SNR 5.68，memory-sensitive 边界比例 0.833。BinFill 单任务的 P2/P3 条件通过，但全局 combined P2/P3 仍为 false，因为当前只有一个任务，未满足至少两个任务的预注册门槛。已输出 `milestone4_diagnostic.json` 和人工复核案例；下一步扩展到其他 train 任务，再决定是否进入 1k–5k 批量生成。
- **Milestone 5：** 训练 value predictor，比较常数预测、视觉相似度、KEMO 式启发式和学习模型；
- **Milestone 6：** 冻结基础策略，进行 learned manager 的正式闭环比较；
- **Milestone P/7：** 仅当明确发现分布适配是实际障碍时，才进行最小策略适配和重新审计。

## 八、关键证据文件

- [实施计划](../../Event_Aware_Counterfactual_Memory_Implementation_Plan.md)
- [V-A 正式报告](../results/V_A_formal_20261023/REPORT.md)
- [V-A/V-B 联合决策](../results/V_A_formal_20261023/JOINT_REPORT.md)
- [V-B 正式报告](../results/V_B_fp32_full_20261022/REPORT.md)
- [M1 验收结果](../results/M1_acceptance_20261023/acceptance.json)
- [M1 验收日志](../results/M1_acceptance_20261023/pipeline.log)
- [M1 报告](../results/M1_acceptance_20261023/REPORT.md)
- [M1 实现说明](milestone1_20261023.md)

## 九、汇报用一句话

> **项目已完成 V-A/V-B 前置验证、Milestone 1 有界记忆工程验收和 Milestone 2 候选管线工程实现；M2 的真实事件 precision/recall 仍待标注和标定，下一步是 Milestone 3 的单标签 teacher 验证，目前尚未证明任务成功率提升。**

## 十、2026-10-24 更新：Teacher Oracle 闭环上限检验

在进入价值预测器之前，插入了一个闭环上限检验：每个决策由看得到专家下一段动作的特权老师，在官方选帧基础上换一帧，看任务成功率能否提高。前置的分支重放、规划器可用性和阶段专家检查均已通过。详细结果见 [第 3 步总结](teacher_oracle_step3_summary_20261024.md)。

| 任务 | U0 | TO（老师） | AO（反向老师） | RS（随机换帧） | 判定 |
|---|---|---|---|---|---|
| PatternLock | 68.3% | 73.3% | 58.3% | 66.7% | 没有可见空间（+5，不显著） |
| BinFill | 36.7% | 38.3% | 38.3% | 36.7% | 没有可见空间（与 VUS 合并：TO − S0 = 0.0，90% CI ±4.2） |
| VideoUnmaskSwap | 26.7% | 25.0% | 26.7% | 26.7% | 同上 |

两次运行的有效性门槛全部通过，工程错误为 0。老师在损失意义上起作用（替换增益约 4 倍标准误，动作更接近专家），但没有转化为成功率。结合 V-A（D4B 钉入证据帧无提升；删除近期窗口只伤害 PatternLock），这个冻结策略在 BinFill 和 VUS 上对记忆内容很不敏感。

**影响**：Milestone 5/6 的原设计（用反事实损失训练价值预测器选记忆）在这三个任务上缺乏依据，暂停。下一步做"记忆是否被使用"的闭环诊断：清空历史记忆或只留当前帧。根据结果再决定方向：策略适配（Milestone P）、换用依赖记忆的任务，或尝试多帧干预。

