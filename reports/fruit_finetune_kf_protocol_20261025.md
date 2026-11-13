# 采摘任务事件关键帧微调协议（RM75-B + EG2）

日期：2026-10-25。本文件在正式训练开始前封存（`runs/fruit_finetune_kf_20261025/protocol.json` 记录其 sha256）；闭环评测阶段（`run_teacher_oracle.py --stage fruit`）沿用同一文件，封存时再次核对 sha256。封存后不改任何条款；冒烟记录写在第 9 节，写完后才封存。

## 1. 问题

目标机械臂（RM75-B 七轴臂 + EG2 夹爪）做采摘。本协议检验：**在采摘示范上从 79999 微调时，"每次采摘只留关键帧、其余均匀"的 16 帧记忆，闭环成功率是否高于用纯均匀 16 帧记忆做同样微调的对照。** 关键帧最多 k 帧，k ∈ {4, 8}；k = 0 是对照。方法与 RoboMME 上的方案 B（`reports/finetune_kf_protocol_20261025.md`）相同，只换数据、机械臂和距离信号。

两个任务都需要记忆：

- **FruitCount**：树上 M 个红果，要求恰好摘 N 个（N < M，每个 episode 不同）放进篮子。篮子在前置相机视野外，策略只能靠记忆数自己摘了几个。成功：篮中恰好 N 个；失败：篮中超过 N 个或有果子掉地。
- **FruitOccludedRipe**：红果（熟）和绿果（生）混挂。开头的观察阶段有风把叶簇吹到两个果子前面（其中至少一个是红果，所以"被遮住"不暗示"熟"）。要求把所有红果放进篮子。成功：红果全在篮中且篮中没有绿果；失败：篮中有绿果或有果子掉地。
    10|
## 2. 记忆定义（`src/memory_management/event_keyframes.py`，与 RoboMME 相同）

训练和闭环用同一个函数 `select(distances, t, KeyframeConfig)`，只决定帧号，距离不进入模型输入。事件、关键帧和选帧规则与 RoboMME 协议第 2 节完全相同（接触帧、离开帧；新事件优先；其余帧等距；k = 0 与官方均匀采样相同）。

- **信号**（`src/memory_management/fruit_distance.py`）：d_t 是末端（TCP）到最近一个"仍挂在树上的可摘果子"的距离（米）。可摘：FruitCount 是所有果子，FruitOccludedRipe 是红果。树上没有可摘果子时记 10 m（"远"，不是未知）。这个定义不依赖示范者的计划，闭环里同样可算；真机上由距离模块提供，定义相同。
- **参数**：r = 0.03 m，tol = 0.005 m，与 RoboMME 相同。依据只来自 train 轨迹：r 取 0.01–0.04 时两个任务 100/100 个 episode 的事件数都等于摘下的果子数；一次事件从夹爪靠近果子开始，接触帧约在果子脱离树枝前 4 帧，离开帧就是脱离那一帧（被夹住的果子已不在树上，距离跳到下一个果子）。

在 train 示范的执行帧上（每任务 100 条、每 4 帧取一个样本，只计 t ≥ 16 的样本），记忆差别如下：

| 任务 | 事件/episode（最多） | 关键帧数均值（k=4 / 8） | 记忆不同的样本比例（k0 vs k4 / k4 vs k8） | 样本数 |
|---|---|---|---|---|
    20|| FruitCount | 2.85（5） | 2.63 / 3.14 | 83.8% / 20.9% | 11,482 |
| FruitOccludedRipe | 2.54（3） | 2.49 / 2.75 | 82.3% / 13.5% | 9,946 |

所以 k = 4 与 k = 0 的记忆在八成以上的样本上不同；k = 8 只在摘第 3 个以后的果子时与 k = 4 不同，差别小得多。这是设计本身的性质，事先写明。

## 3. 训练数据

- 来源：采摘仿真（`fruit_picking/`，FruitPick-v1，RM75-B + EG2，pd_joint_pos 控制）的脚本示范者在 train 种子上生成的成功示范，count 与 occluded_ripe 各 episode 0–99（`data/fruit_picking/rm75b_eg2/train`）。示范者某次尝试失败（规划失败、没摘下、没放进篮子、任务未完成）就丢弃并换种子重试，只保留成功示范。每帧记录仿真真值。
- 观测与动作：前置相机和手腕相机 RGB（256×256），状态 = 7 个关节角 + 第一个手指位置（8 维），动作 = 7 个关节目标角 + 夹爪指令（8 维），与 RoboMME 的格式相同，由官方 `DatasetProcessor` 原样读取。
- 预处理：`scripts/build_fruit_dataset.py`（复用 `build_finetune_dataset.py` 的分片构建，每帧走官方 `DatasetProcessor._process_episode`），另为每个 episode 写 `distance.npy`（第 2 节的信号，由真值逐帧算出）。输出 `data/finetune_kf/fruit_rm75b_train`。
- 规模：200 个 episode，88,607 个执行样本（FruitCount 47,369，FruitOccludedRipe 41,238；没有视频示范帧）。每个 episode 要摘的果子数：FruitCount 2–5 个，FruitOccludedRipe 2–3 个。
    30|- 归一化统计：官方 `scripts/compute_norm_stats.py` 的流程（同一数据加载与变换、`RunningStats`）在全部 train 样本上重算，asset id 仍为 `robomme`（服务端无需改动），写在 `data/finetune_kf/fruit_rm75b_train/assets/robomme/norm_stats.json`。不用 79999 自带的统计：RM75 的关节范围与 Panda 差别很大（16 步动作增量的 1%–99% 分位数，第 1 关节 ±1.3 rad，Panda 只有 ±0.19 rad）。
- 特征一致性：存储的 SigLIP 特征与服务端编码器在同一帧上的差异为 bfloat16 舍入量级（见第 9 节）。

## 4. 训练

- 初始化：79999 的参数（EMA 参数）；归一化统计用第 3 节的采摘统计；历史配置 `perceptual-framesamp-modul.yaml`（预算 512 token）。79999 从未见过 RM75-B，这是一次换机械臂的微调。
- 训练循环：官方 `scripts/train.py: main` 原样运行，只替换选帧（`scripts/finetune_event_keyframes.py`），与 RoboMME 相同。
- 三个臂 k0 / k4 / k8，除 k 外完全相同。除步数外，超参数与 RoboMME 协议第 4 节相同：batch 64；每臂 2 张 A800（fsdp 2），GPU 0–1 依次训 k0、k8，GPU 2–3 训 k4；12000 步（约 8.7 个 epoch）；AdamW（官方设置，梯度裁剪 1.0）；学习率 500 步线性预热到 2.5e-5，余弦衰减（12000 步）到 2.5e-6；EMA 0.999；视觉编码器冻结；随机种子 42；数据加载进程 12。每臂只保存最终 checkpoint（step 11999），checkpoint 里带采摘归一化统计。每臂约 9 小时，共两轮，约 18–19 小时。
- 步数的依据：RoboMME 用 6000 步（约 4.3 个 epoch）是因为 79999 已经拟合了那些示范（第 0 步损失约 0.0011）；这里是换机械臂，6000 步可能停在地板附近，所以加倍。步数在任何采摘训练或评测之前确定，不看结果调整。
- 重启规则与训练门槛 T1 与 RoboMME 相同：三臂都得到最终 checkpoint，每臂最后一次运行的 240 条损失记录（每 50 步的区间均值）齐全且有限；另加一条：每个最终 checkpoint 带的归一化统计与第 3 节的采摘统计一致。不满足则不做闭环评测。
- 执行：`scripts/run_fruit_finetune_pipeline.sh` 依次训练三臂（失败臂自动重训一次），T1 通过后直接开始第 5 节的闭环评测。

    40|## 5. 闭环评测

- 仿真：FruitPick-v1（CPU 物理，GPU 渲染），由 `src/memory_management/fruit_eval.py` 的 `FruitEnvRunner` 接到官方 `EpisodeEvaluator.eval_each_episode` 上，循环不改：每 16 步一次决策，步数上限 1300（官方值；train 示范最长 757 步），任务成功或失败时 episode 立即结束。
- val split，两个任务各 episode 0–49（示范者在 val 种子上验证过可解的布局），模型种子 7、8、9：每任务 150 组，共 300 组。val 的采摘 episode 此前没有用于任何评测或调参（只有第 9 节的工程检查用过 episode 0 和 7，不涉及成功率）。
- 条件（同一组内三个条件共享环境种子、策略随机数重置和同一服务进程，顺序随机）：K0、K4、K8，各自微调后的 checkpoint（`runs/fruit_finetune_kf_20261025/ckpt/mme_vla_suite/k{k}/11999`），配各自的 16 帧记忆。不设 79999 的对照：它没有见过这个机械臂，归一化统计也不对。
- 距离信号：外部模块接口（`event_memory.ExternalDistance`，工厂 `memory_management.fruit_distance:RecordedDistance`）。客户端每帧用与训练相同的定义从仿真状态算出距离并写入 `raw_frames.jsonl`，服务端逐帧读取，先核对图像摘要确认是同一帧。真机上把这个模块换成距离模块即可。
- 每个模型种子的服务进程装载三个 checkpoint，按 episode 上下文里的条件路由（`ConditionRouter`）。

## 6. 统计与判定

    50|- 对每组（任务, episode, 种子）配对，bootstrap 20000 次（种子 20261024），成功率差以百分点计。
- 主对比，各任务分别，一个 Holm 族共 4 项：G1 = K4 − K0，G2 = K8 − K0。α = 0.05，界限 δ = 12 个百分点。
- 状态：95% CI 下界 > 0 且 Holm p < α 为"提升"；上界 < 0 且 Holm p < α 为"退化"；90% CI 落在 (−δ, δ) 内为"小于界限"；其余"未定"。
- 每个任务的判定：有提升的 k 则报"这些 k 优于同样微调的均匀记忆 K0"（同时列出退化的 k）；否则有退化的 k 则报"差于 K0"；两项都小于界限则报"事件关键帧没有可见作用"；其余"未定"。未定不追加种子或 episode。
- 地板规则：某任务三个条件的成功率都 ≤ 5% 时，该任务报"地板效应，无法比较"，不报"没有可见作用"。
- 有效性门槛：G3 工程错误率 ≤ 2%；G5 每组三个条件在同一服务进程；GA 选帧审计——每个决策实际喂给模型的帧与该条件按记录的距离信号重算的结果完全一致。任何门槛不过，结果只作工程诊断。

## 7. 不在本协议内

- 真机距离模块的噪声、缺测和遮挡（仿真里被叶子挡住的红果距离仍由真值给出；真机模块看不到时会不同）的稳健性测试，之后单独立协议。
    60|- 训练步数或数据量的扫描；与 79999 或 RoboMME 微调 checkpoint 的比较。
- 真机部署。

## 8. 产出

- 训练：`runs/fruit_finetune_kf_20261025/`（`protocol.json`、`arms/*.json`、`logs/`、`train_status.json`、`gate_t1.json`、`ckpt/`）。
- 闭环：`results/fruit_finetune_kf_eval_20261025/`（`protocol.json`、`analysis.json`、`REPORT.md`，自动生成）。

## 9. 冒烟记录

    70|以下都在封存前完成，结果不参与判定。

1. **数据构建**（`data/finetune_kf/fruit_rm75b_train/meta/build.json`）：检查全部通过；200 个 episode，88,607 个执行样本；`episodes.json` sha256 `a7b67343…`。特征一致性（`meta/encoder_check.json`，episode 0 的 8 帧）：平均绝对差 0.0086，特征平均绝对值 1.38，最大差 1.0，与 RoboMME 构建（0.0085 / 1.45 / 1.0）相同。归一化统计：692 批 × 128 样本。
2. **仿真接口回放**（`results/fruit_runner_check_20261025.json`）：把 4 条 val 示范（两个任务各 episode 0、7）的动作逐步送进 `FruitEnvRunner` + 记录器，每一帧的图像、状态与训练数据逐像素一致，记录的距离与训练用的距离差 0；episode 在真值第一次报告成功的那一步结束（示范者成功后还多录约 20 步静置，闭环 episode 不会有这一段）。
3. **闭环管线冒烟**（79999、K4、FruitCount val episode 0、种子 7，只测管线）：服务端通过外部模块读到了全部 1297 个决策帧的距离，与客户端记录逐帧相同，图像摘要无一不符；79999 不会操作这个机械臂，1301 步超时，末端离果子最近 0.38 m，没有触发事件。完整一条超时 episode 约 100 s。
4. **训练冒烟**（`runs/fruit_finetune_kf_smoke_20261026`，100 步，5 步预热，三臂）：T1 通过，三个最终 checkpoint 带的归一化统计与采摘统计一致。损失（第 0 步 → 最后一个区间均值）：k0 0.1664 → 0.0274，k4 0.1664 → 0.0274，k8 0.1664 → 0.0274。总用时 14 分钟（k0、k4 并行，k8 随后，含装载和编译）。
5. **闭环冒烟**（`results/fruit_finetune_kf_eval_smoke_20261026`，冒烟 checkpoint step 99，val episode 0、种子 7，两个任务 × 三个条件）：G3、G5、GA 全部通过；选帧审计 492 个决策，不一致 0 个。结局：timeout 6（冒烟 checkpoint，不解读）。各卡显存峰值（含同机其他进程）：GPU 0 32.5 GB，GPU 1 0.0 GB，GPU 2 0.6 GB，GPU 3 0.0 GB。用时 6 分钟（含装载）。
6. **正式评测耗时估计**：冒烟里每条 episode 耗时中位数 50 s（冒烟 checkpoint 多数跑满步数，接近上限）；900 条、3 个服务进程并行，约 4.1 小时以内。
