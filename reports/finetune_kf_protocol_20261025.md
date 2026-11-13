# 事件关键帧微调（方案 B）协议

日期：2026-10-25。本文件在正式训练开始前封存（`runs/finetune_kf_20261025/protocol.json` 记录其 sha256）；闭环评测阶段（`run_teacher_oracle.py --stage finetune`）沿用同一文件，封存时再次核对 sha256。封存后不改任何条款；冒烟记录写在第 9 节，写完后才封存。

## 1. 问题

冻结的 MME-VLA（checkpoint 79999）上，按距离加密的事件记忆没有改善成功率，BinFill 还变差了（`results/event_memory_20261025`）。事后诊断（`results/event_memory_20261025_posthoc/binfill.json`，描述性）显示：均匀记忆本来就覆盖了每一次接近，按距离加密只是把同一次接近重复更多帧，挤掉了别的帧；策略又没有在这种记忆上训练过。

本协议检验：**用"每次接近只留关键帧、其余均匀"的 16 帧记忆微调后，策略的闭环成功率是否高于用纯均匀 16 帧记忆做同样微调的对照。** 关键帧最多 k 帧，k ∈ {4, 8, 12}；k = 0 是对照。

## 2. 记忆定义（`src/memory_management/event_keyframes.py`）

训练和闭环用同一个函数 `select(distances, t, KeyframeConfig)`，只决定帧号，距离不进入模型输入。

- **信号**：d_t 是末端到最近目标物体的距离（米），仿真里由 `event_memory.gt_distance` 从真值算出（PatternLock 在桌面平面内量到按钮；BinFill 量到目标颜色、仍在桌上的方块；VideoUnmaskSwap 量到箱子），未知记 NaN。真机上由距离模块提供，定义相同。
- **事件**：d_t ≤ r 为"近"，d_t > r 为"远"，未知帧沿用上一帧的状态（缺测不会开始或结束事件）。一次事件是一段连续的"近"帧，给出两个关键帧：
  - 接触帧：这段里距离第一次落到"本段最小值 + tol"以内的帧；
  - 离开帧：这段结束后第一个"远"帧（事件还在进行时没有）。BinFill 里一段覆盖抓取和搬运，离开帧就是方块落进箱子的时刻；PatternLock 里一段是一次按钮。
- **选帧**（第 t 帧，只用 0..t 帧及其距离）：t + 1 ≤ 16 时全选；否则先取最近若干次事件的关键帧（新事件优先，接触帧先于离开帧），最多 k 个；剩余 16 − (关键帧数) 帧在其余帧上等距选取（含首尾）。k = 0 时与官方均匀采样 `even_sampling_indices(t, 16)` 完全相同。
- **参数**：r = 0.03 m，tol = 0.005 m，三个任务相同。依据只来自 train 轨迹（前 30 条/任务）：r = 0.03 时 PatternLock 平均 7.3 次事件/episode，等于子目标数；BinFill 2.8 次，等于抓起的方块数（r 取 0.01–0.08 都不变）；VideoUnmaskSwap 1.6 次。

在 train 示范的执行帧上（每任务 40 条、每 4 帧取一个样本），三个 k 的记忆差别如下：

| 任务 | 事件/episode | 关键帧数均值（k=4 / 8 / 12） | 记忆不同的样本比例（k0 vs k4 / k4 vs k8 / k8 vs k12） |
|---|---|---|---|
| PatternLock | 7.25（最多 15） | 3.99 / 7.68 / 10.15 | 100% / 99.3% / 76.2% |
| BinFill | 2.73（最多 5） | 2.63 / 3.36 / 3.38 | 86.9% / 31.8% / 0.9% |
| VideoUnmaskSwap | 1.53（最多 2） | 1.05 / 1.05 / 1.05 | 62.0% / 0% / 0% |

所以 k 之间的差别主要在 PatternLock；BinFill 的 k = 8 与 k = 12 几乎相同；VideoUnmaskSwap 的三个 k 记忆完全相同（只会因为联合训练的权重不同而不同）。这是设计本身的性质，事先写明。

## 3. 训练数据

- 来源：RoboMME train split 的 PatternLock、BinFill、VideoUnmaskSwap 各 episode 0–99，用官方生成器重新生成并逐帧记录真值（`data/train_demos_gt`；完整性检查 300/300，`data/train_demos_gt/integrity_20261025.json`；与本地原有 train 示范逐像素、逐动作一致）。这些 episode 很可能就是 79999 训练时用过的。
- 预处理：`scripts/build_finetune_dataset.py`，每帧都走官方 `DatasetProcessor._process_episode`（输出格式由官方 `RoboMMEDataset` 原样读取），另为每个 episode 写 `distance.npy`。episode 按"任务顺序 × episode 升序"编号，样本号预先分配，4 张卡并行与官方顺序构建结果相同。输出 `data/finetune_kf/robomme3_train_gt`。
- 规模：300 个 episode，114,746 帧，88,977 个执行样本（PatternLock 10,391，BinFill 59,602，VideoUnmaskSwap 18,984）。
- 特征一致性：存储的 SigLIP 特征（官方构建器用的编码器）与服务端策略自身的视觉编码器在同一帧上的差异为 bfloat16 舍入量级（见第 9 节），与官方训练/推理链路相同。

## 4. 训练

- 初始化：79999 的参数（EMA 参数）与归一化统计（`assets/robomme/norm_stats.json`），历史配置 `perceptual-framesamp-modul.yaml`（预算 512 token；16 帧之外的槽位由官方打包器右侧补零并屏蔽）。
- 训练循环：官方 `scripts/train.py: main` 原样运行，只替换 `RoboMMEDataset.prepare_frame_sampling` 的选帧（`scripts/finetune_event_keyframes.py`）。
- 四个臂 k0 / k4 / k8 / k12，除 k 外完全相同：batch 64；每臂 2 张 A800（fsdp 2；单卡 batch 64 显存不够，见第 9 节），GPU 0–1 依次训 k0、k8，GPU 2–3 依次训 k4、k12；6000 步；AdamW（官方设置，梯度裁剪 1.0）；学习率 500 步线性预热到 2.5e-5，余弦衰减到 2.5e-6；EMA 0.999（保存的是 EMA 参数）；视觉编码器冻结（官方冻结规则）；随机种子 42（数据顺序和流匹配噪声四臂相同）；数据加载进程 12。每臂只保存最终 checkpoint（step 5999）。按冒烟测得的约 2.5–3 s/步，每臂约 4–5 小时，全部约 9–10 小时。
- 样本按官方方式在全部执行样本上均匀打乱，不按任务重新加权（BinFill 样本最多）。
- 重启规则：某臂没有最终 checkpoint 就整臂移到 `superseded/` 并从第 0 步重训（官方循环不保存优化器状态，无法续训）。
- 训练门槛 T1（`run_finetune_kf.py gate`，结果写 `gate_t1.json`）：四臂都得到最终 checkpoint；每臂最后一次运行的日志里 120 条损失记录（每 50 步的区间均值）齐全且全部有限。不满足则不做闭环评测，只报告工程问题。
- 执行：`scripts/run_finetune_kf_pipeline.sh` 依次训练四臂；有臂失败时按重启规则自动重训一次（只重训没有最终 checkpoint 的臂）；T1 通过后直接开始第 5 节的闭环评测。

## 5. 闭环评测

- val split，episode 20–49，三个任务，模型种子 7、8、9：每任务 90 组，共 270 组。不碰 test。val 0–19 是事件记忆实验用过、并引出本设计的那批 episode，所以不用；PatternLock 的 20–49 只在记忆预算标定（U0/B16/B8）里用过。
- 条件（同一组内五个条件共享环境种子、策略随机数重置和同一服务进程，顺序随机）：
  - K0、K4、K8、K12：各自微调后的 checkpoint（`runs/finetune_kf_20261025/ckpt/mme_vla_suite/k{k}/5999`），配各自的 16 帧记忆；
  - B16：未微调的 79999，官方均匀 16 帧（与记忆预算标定的 B16 相同）。
- 距离信号：真值（`GroundTruthDistance`，读客户端写的 `raw_frames.jsonl`），与训练时同一个 `gt_distance`。
- 每个模型种子一个服务进程，进程内装载全部五个 checkpoint，按 episode 上下文里的条件路由（`serve_teacher_policy.py` 的 `ConditionRouter`）。

## 6. 统计与判定

- 对每组（任务, episode, 种子）配对，bootstrap 20000 次（种子 20261024），成功率差以百分点计。
- 主对比，各任务分别，一个 Holm 族共 9 项：F1 = K4 − K0，F2 = K8 − K0，F3 = K12 − K0。α = 0.05，界限 δ = 12 个百分点。
- 状态：95% CI 下界 > 0 且 Holm p < α 为"提升"；上界 < 0 且 Holm p < α 为"退化"；90% CI 落在 (−δ, δ) 内为"小于界限"；其余"未定"。
- 每个任务的判定：有提升的 k 则报"这些 k 优于同样微调的均匀记忆 K0"（同时列出退化的 k）；否则有退化的 k 则报"差于 K0"；三项都小于界限则报"事件关键帧没有可见作用"；其余"未定"。未定不追加种子或 episode。
- 次要对比（单独一个 Holm 族，3 项，不改变判定）：F4 = K0 − B16，即同一 16 帧均匀记忆下微调本身的作用。
- 有效性门槛：G3 工程错误率 ≤ 2%；G5 每组五个条件在同一服务进程；GA 选帧审计——每个决策实际喂给模型的帧与该条件按记录的距离信号重算的结果完全一致。任何门槛不过，结果只作工程诊断。

## 7. 不在本协议内

- 真机距离模块的噪声和缺测稳健性测试（之后单独立协议，在仿真里给真值加噪声、丢帧）。
- 采摘数据集（RM75-B）上的训练和评测。
- 其余 RoboMME 任务：只在三个任务上微调，不评估其余任务上的遗忘。

## 8. 产出

- 训练：`runs/finetune_kf_20261025/`（`protocol.json`、`arms/*.json`、`logs/`、`train_status.json`、`ckpt/`）。
- 闭环：`results/finetune_kf_eval_20261025/`（`protocol.json`、`analysis.json`、`REPORT.md`，自动生成）。

## 9. 冒烟记录

以下都在封存前完成，结果不参与判定。

1. **数据构建**（`data/finetune_kf/robomme3_train_gt/meta/build.json`）：检查全部通过；300 个 episode，114,746 帧，88,977 个执行样本（PatternLock 10,391，BinFill 59,602，VideoUnmaskSwap 18,984）；`episodes.json` sha256 `75c56a8d…`。特征一致性（`meta/encoder_check.json`，episode 0 的 8 帧）：存储特征与服务端编码器的平均绝对差 0.0085，特征平均绝对值 1.45，最大差 1.0（数值大的元素上的 bfloat16 舍入）。
2. **训练冒烟一**（每臂 1 卡，batch 64）：显存不够（训练状态约 49 GB，加上约 67 GB 激活值）。因此改为每臂 2 卡 fsdp（第 4 节）。同一轮里冒烟的 500 步预热超过冒烟步数，学习率表报错；冒烟改用 5 步预热，正式设置不变。
3. **训练冒烟二**（`runs/finetune_kf_smoke_20261025`，30 步，fsdp 2）：四臂都得到最终 checkpoint。损失有限，0.0011–0.0020，四臂几乎相同（共享数据顺序和噪声）。每卡显存约 66 GB（预分配 92%）。
4. **训练冒烟三**（`runs/finetune_kf_smoke_tmpfs_20261025`，100 步，数据放在内存盘副本上，副本与原件核对一致）：第 15–100 步稳定在 2.7–3.0 s/步；每次运行在第 13 步前后都有一次约 70 s 的停顿，网络盘和内存盘上都出现，之后不再出现。网络盘上的冒烟二约 2.5 s/步，内存盘并不更快，所以正式训练直接读网络盘上的原件。
5. **损失水平**：第 0 步损失约 0.0011，说明 79999 对这些 train 示范已经拟合得很好（很可能训练时见过）。微调信号可能很弱，四个臂之间、以及与 B16 之间的差别都可能很小。次要对比 F4（K0 − B16）用来衡量微调本身的作用。这一点事先写明，不因结果调整。
6. **闭环冒烟**（`results/finetune_kf_eval_smoke_20261025`，用冒烟二的 step 29 checkpoint，val episode 20、种子 7，三个任务 × 五个条件）：G3、G5、GA 全部通过。选帧审计 166 个决策，不一致 0 个。每条结果记录的 checkpoint 路径与条件一一对应（K0–K12 为各自微调的 checkpoint，B16 为 79999）。一个服务进程装载五个 checkpoint 后显存约 61 GB。3 组共约 6.5 分钟（含装载）。成功与否不解读。
7. **正式评测耗时估计**：270 组，3 个服务进程并行，每组约 1.5 分钟，约 2–3 小时。
