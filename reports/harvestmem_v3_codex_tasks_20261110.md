# HarvestMem v3 任务清单（交给 Codex 执行）

日期：2026-11-10。审阅人：Claude。执行人：Codex。

## 0. 背景与方向（必读）

**过去为什么失败。** 端到端 VLA 靠"选对记忆帧"来提升采摘成功率，这条路已经验证了约 8 次，全部为零。

最近一次是 11-09 通宵诊断，结果见 `results/overnight_r12_20261109/SUMMARY.md`：

| 条件 | Count 成功 | Occluded 成功 |
|---|---|---|
| K0 均匀记忆 | 0/40 | 1–2/40 |
| KA 事件记忆 | 0/40 | 1–2/40 |
| OR（指令里直接写"已放 k 个"） | 0/40 | 1–2/40 |
| NM 无记忆 | 0/40 | 1–2/40 |

- Count 约 75% 的失败是没等绿灯就提前开工。
- OR 在篮子里已有 N 个时仍然继续摘（overpick）。

**结论：** 瓶颈不在选帧。策略学不会"开始 / 停止"这类稀疏决策，也不会把计数换算成该停下的动作。

**新方向：** 事件驱动的显式记忆 + 子目标条件 VLA（分层结构）。

- **高层（HarvestMem 状态跟踪器）**：用感知和采摘事件检测，显式维护任务状态：每个果子的位置和初始颜色、已摘过哪些、已摘数量。再把状态翻译成**当前子目标文本**。
- **低层（VLA）**：只负责执行子目标，例如"pick the fruit at <x, y>"、"return to rest and hold still"。
- **为什么这次可能成功：** 示范数据里每一帧都有子目标标签，监督是密集的；不再要求网络从几百条示范里自己学会数数。

**禁止重新设计任务。** 本轮只允许做一项任务改动：取消等灯（见 S1）。其他定义沿用 R12。

## 1. 已核实的现有代码事实

| 事实 | 位置 |
|---|---|
| R12 示范的 GT 每帧都有 `phase` 和 `subgoal`。子目标例如 `pick up the fruit on the upper left`、`put the fruit into the basket`、`return to rest to finish`、`hold open at rest until termination`；Occluded 带颜色，例如 `pick up the red fruit on the left` | `data/fruit_task_v2_r12_*/rm75b_eg2/train/*/gt/*.jsonl` |
| 子目标的方位词来自 `describe()`，精度只有左/中/右 × 上/中/下。果子有 4–6 个时**会重名、有歧义**，需要改成带定位的子目标 | `fruit_picking/solver.py:40` |
| 已构建的数据集里，每个样本已经有 `simple_subgoal` 和 `grounded_subgoal` 字段，但当前训练用的配置没有读这两个字段 | `data/finetune_kf/fruit_task_v2_r12_20261109/data/*.pkl` |
| MME-VLA 自带 symbolic 子目标通路，tokenizer 格式为 `Task: …;\nCurrent Subgoal: …;\nAction:`。官方 grounded 子目标格式为 `at <x, y>`，训练时还对坐标加了噪声增强 | `acm_vla/vendor/robomme_policy_learning/src/mme_vla_suite/training/config.py:110-210`、`training/dataset.py:171-187`、`models/config/robomme/symbolic-*.yaml` |
| 本地只有 `perceptual-framesamp-modul/79999` 的权重，**没有 symbolic 配置的权重** | `adaptive_evidence_acquisition_20261009/models/` |
| 在 prompt 末尾追加文本的训练方式已经跑通，可以照搬（OR 臂） | `scripts/overnight_r12/finetune_variant.py` |
| 闭环评测器、策略服务端、通宵队列已有可用版本，可以复用或改写 | `scripts/overnight_r12/{eval_episode.py,serve.py,run_queue.py}` |
| SAM-FDM 感知（颜色提示 + SAM + RGB-D 反投影 + 锚点）和事件检测都已实现 | `paper/experiments/perception.py`、`paper/experiments/exp_perception_distance.py`、`src/memory_management/event_keyframes.py`、`src/memory_management/fruit_v2_distance.py` |
| 等灯的实现：`TaskV2Spec.start_delay_max=600`、`early_start_failure`；count 的 revision 9 会强制打开 | `fruit_picking/task_v2_spec.py`、`fruit_picking/task_v2_env.py:33-36,111,166-170,201` |
| R12 数据生成入口，以及 count 环境的安装方式（遮挡罩 + 假底） | `scripts/audit_fruit_task_v2_r12_episode.py`、`scripts/overnight_r12/eval_episode.py:make_env` |

## 2. 全局规则

1. **只写新目录。** 新代码放 `scripts/harvestmem_v3/`；新数据放 `data/harvestmem_v3_*`；新结果放 `results/harvestmem_v3_*`；新训练放 `runs/harvestmem_v3_*`。不修改、不删除任何旧结果和旧数据。确实要改旧代码时，只能加一个默认关闭的开关，旧行为必须保持不变。
2. **GPU 使用。** 训练和闭环评测不能同时用同一张卡。11-09 就是因为这样，Vulkan 显存耗尽，148/160 个 episode 报错。每张卡最多同时跑 2 个仿真客户端。每个 episode 设 30 分钟超时；工程错误最多重试 2 次；错误单独统计，不计入成功率。
3. **先冒烟，再长跑。** 任何超过 1 小时的任务，先跑一个最小冒烟：训练 3 步，或 1 个完整 episode。确认冒烟成功后，才放到后台。
4. **后台无人值守。** 长任务用 `nohup setsid` 启动，过程必须可续跑（已完成的结果跳过）。结果自动汇总到 `SUMMARY.md`。
5. **不要过度流程化。** 不需要 sha256 封存、逐条审计证明、恢复分支这类机制。只需要：可复现（记录命令、git commit、种子）、结果诚实（失败如实报告）。
6. **每个阶段结束都停下来交付**，等审阅通过后再进入下一阶段。门槛没通过，就如实报告并给出你判断的原因，不要自行改门槛或改任务。
7. **数据划分。** train 0–199（种子 4100000+16e+attempt）；dev 200–259（种子 4200000+16e）用于开发和门槛判定；val 150–199（种子 4300000+16e）留到最终评测，本轮不使用。

## 3. 阶段任务与交付物

### S0 代码勘察与方案确认（只读，不用 GPU，约半天）

**要做的事**

1. 确定子目标进入模型的方式，二选一，并说明理由：
   - (a) 沿用 `perceptual-framesamp-modul` 模型和 79999 权重，把子目标拼进 prompt：`<instruction> current subgoal: <subgoal>.`
   - (b) 改用 MME-VLA 的 symbolic 配置。需要确认 79999 权重能否加载到 symbolic 结构上，哪些参数缺失或形状不匹配。
   
   **默认推荐 (a)**，因为它已经被 OR 臂验证过能训练。
2. 设计带定位的子目标格式。要求：
   - 同一场景内任意两个果子的子目标文本都不相同；
   - 能从前置相机图像上算出来。推荐用果心在 256×256 前置图像上的像素坐标，例如 `pick the fruit at <x, y>`。
   - 列出完整的子目标集合：pick、place、return_and_hold，以及 Occluded 中的 skip/none 等情况。
3. 统计 R12 现有示范：子目标标签是否每帧都有；在一个场景内用 `describe()` 产生的重名率是多少。

**交付** `reports/harvestmem_v3_S0.md`，内容包括：

- 选定的方案和理由；
- 子目标格式的定义，附 5 个示例；
- 重名率统计；
- 需要新增或修改的文件清单；
- 风险清单。

### S1 去掉等灯的任务变体 + 数据重新生成（约 1–1.5 天，主要是机器时间）

**要做的事**

1. 新增环境开关 `no_lamp_wait=True`：`start_delay` 固定为 0，`early_start_failure=False`。灯可以保留成常绿，也可以直接移除，二选一并说明。其余全部和 R12 一致：不会凑够 N 自动判成功、要求完成信号、补果、风落、遮挡罩 + 假底、N∈2..10、预算 2800；Occluded 保持原样。
2. 改数据记录：GT 每帧增加 `grounded_subgoal`（S0 定的格式）、`target_slot`、`n_picked_so_far`、`phase`。
3. dev 200–249 专家回放，每个任务 50 条。**门槛：专家成功率 ≥ 45/50。**
4. 生成训练数据：两个任务各 clean 0–99 + DART 100–199。DART 配方沿用 R12 修好后的两周期方案。只有成功的示范进入数据集。
5. 构建训练集（`build_fruit_dataset.py`），计算归一化统计。样本里写入 `grounded_subgoal`。

**交付** `reports/harvestmem_v3_S1.md`，内容包括：

- 环境改动的 diff 摘要；
- 专家门槛结果；
- 400 条示范的统计（N 分布、步数、尝试次数、失败原因）；
- 构建结果路径和样本数；
- 抽查 3 条示范：每条列出子目标序列，附关键帧截图。

### S2 训练（约 1 天，全是机器时间）

在 S1 的数据上训练以下 3 个模型，配方与 D5 相同：79999 初始化，12000 步，batch 64，学习率 2.5e-5→2.5e-6，EMA 0.999，seed 42，temporal_index=order。

| 臂 | 输入 | 作用 |
|---|---|---|
| **SG** | 均匀记忆 + 带定位的子目标（训练时用 GT） | 本文方法的低层 |
| **K0** | 均匀记忆，只有任务指令 | 端到端基线（MME-VLA 原版） |
| **NM** | 只有当前帧，只有任务指令 | 无记忆基线 |

SG 训练时对子目标坐标做小噪声增强，±4 px，参考官方的 `add_grounding_augmentation`。两个 4 卡任务并行跑，K0 和 NM 排在后面。

**交付** `reports/harvestmem_v3_S2.md`：每个臂的 checkpoint 路径、loss 曲线（首 / 1000 步 / 末）、训练耗时。

### S3 高层状态跟踪器（可以和 S2 并行，CPU 为主，约 1 天）

实现 `scripts/harvestmem_v3/tracker.py`。它是因果的：每帧输入观测和本体状态，输出当前子目标文本。逻辑如下：

1. **初始化（观察阶段）**：确定每个果子的位置、初始颜色，以及在前置图像上的像素坐标。
2. **事件检测**：沿用 `event_keyframes` 和 any_hanging 距离，加上滞回，检测"抓到 → 脱离 → 入篮"。每完成一次，已摘数加 1。
3. **决策**：
   - Count：已摘数小于 N 时，选下一个目标（规则与专家一致：取 y 最大的挂果）；等于 N 时，输出 return_and_hold。
   - Occluded：选下一个初始为红色、且还没摘的果子；全部摘完后输出 return_and_hold。
4. **两种信号源**：
   - `source=gt`：读取仿真真值，作为上界；
   - `source=perception`：用 SAM-FDM 处理前置 RGB-D 得到锚点和距离，作为部署版本。

**离线验证**：在 S1 的 dev 专家回放上逐帧运行跟踪器，与 GT 子目标比较。

**门槛**
- `gt` 源：每帧一致率 ≥ 99%；
- `perception` 源：每帧一致率 ≥ 95%，已摘数在每个回合末尾全部正确。

**交付** `reports/harvestmem_v3_S3.md`：

- 两种信号源各自的一致率，按任务和 N 分层；
- 不一致帧的分类（时机偏差几帧、目标选错、计数错）；
- 单元测试结果。

### S4 闭环评测与方向判定（约半天到 1 天，机器时间）

在 dev 200–239 上，每个臂每个任务跑 40 条，1 个模型种子，8 张卡全部空闲时才开始。

| 臂 | 低层模型 | 子目标来源 | 意义 |
|---|---|---|---|
| A | SG | 仿真真值脚本（oracle） | 系统上界 |
| B | SG | 跟踪器 `source=gt` | 跟踪逻辑本身的损耗 |
| C | SG | 跟踪器 `source=perception` | **本文完整方法** |
| D | K0 | 无 | 端到端记忆基线 |
| E | NM | 无 | 无记忆基线 |
| F | 不需要模型 | 计时脚本（clock_onset 的逻辑，复用 R0 的实现） | 时间捷径基线 |

**汇总指标**

- 成功率；
- Count 按 N 分层（2–4 / 5–7 / 8–10）；
- 失败分类：记忆类（多摘、少摘就停、摘错果）和操作类（掉果、规划失败、超时）；
- 平均步数。

**判定规则（开跑前就固定，跑完后不再更改）**

- **继续推进**：C 的 Count 成功率 ≥ 50%，且 C − D ≥ 25 个百分点 → 进入正式评测（val）和真机阶段。
- **低层有问题**：A 的成功率 < 40% → 问题在低层执行，不在记忆。报告主要的操作类失败，然后停下。
- **跟踪器有问题**：A 高但 C 低 → 修跟踪器（感知部分），然后停下报告。

**交付** `results/harvestmem_v3_S4/SUMMARY.md` 加 `reports/harvestmem_v3_S4.md`：结果表、判定结论，以及每个臂 3 个典型失败 episode 的视频或截图路径。

### S5 真机准备（S4 通过后再启动，先只写清单）

S4 通过后，再单独下发真机任务。届时需要：

- 采集格式与仿真 h5 对齐（关节、夹爪、前置 + 腕部 RGB、深度、时间戳）；
- 主从臂采集脚本；
- 部署服务；
- 真机上 SAM-FDM 的标定。

本轮不做。

## 4. 每次交付的报告格式

每份 `reports/harvestmem_v3_S*.md` 都要包括：

1. 一句话结论：通过 / 未通过 / 部分通过；
2. 做了什么（命令、git commit、耗时）；
3. 结果表（关键数字）；
4. 门槛对照；
5. 异常和未解决的问题（**如实写，不要省略**）；
6. 下一阶段需要审阅人决定的事项。
