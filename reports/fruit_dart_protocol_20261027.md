# 采摘任务事件关键帧微调协议：DART 纠偏示范（RM75-B + EG2）

日期：2026-10-27（冒烟于 2026-10-28 写进第 9 节，随后封存）。本文件在正式训练开始前封存（`runs/fruit_dart_kf_20261027/protocol.json` 记录其 sha256）；开发门（`run_teacher_oracle.py --stage fruit_dev`）和闭环评测（`--stage fruit_dart`）沿用同一文件，各自封存时再次核对 sha256。封存后不改任何条款。

## 1. 问题

旧采摘协议（`reports/fruit_finetune_kf_protocol_20261025.md`，已封存）在干净示范上微调 12000 步后，K0/K4/K8 在两个任务上的闭环成功率都 ≤ 5%（地板效应，`results/fruit_finetune_kf_eval_20261025/REPORT.md`），无法比较记忆。诊断（`results/fruit_diagnosis_20261027/`）表明：开环动作很准（示范状态上 TCP 误差中位 0.4 cm），train 与 val 开环一致，夹爪方向一致率 98–99%，但闭环在 train 布局上也是 0 成功，训练损失已到 0.0011——失败来自闭环误差累积，策略从没见过偏离示范的状态，而不是欠拟合或没见过 val。

对症的做法是给训练数据加入"偏离后如何回来"的标签（DART，Laskey et al. 2017）。本协议检验两件事：

1. **主问题（与旧协议相同）**：在干净 + DART 示范上微调时，事件关键帧 16 帧记忆（k ∈ {4, 8}）的闭环成功率是否高于同样微调的均匀 16 帧记忆（k = 0）。
2. **次要问题**：DART 示范是否让同一记忆（k = 0）脱离地板，即新 K0 与旧干净 K0（记作 C0）的差。

## 2. 记忆定义

与旧采摘协议第 2 节完全相同：同一个 `select(distances, t, KeyframeConfig)`，r = 0.03 m，tol = 0.005 m，信号为 TCP 到最近一个仍挂在树上的可摘果子的距离（`src/memory_management/fruit_distance.py`），没有可摘果子时记 10 m。距离只决定帧号，不进入模型输入。DART 示范的事件检测：100 条 DART episode 上，事件对果子脱离的精确率 0.998、召回率 1.000（`paper/results/event_detection.json`），定义无需改动。

## 3. 训练数据

- **干净示范**：count 与 occluded_ripe 各 train episode 0–99（`data/fruit_picking/rm75b_eg2/train`），即旧协议的全部训练数据。
- **DART 示范**：各 train episode 100–199（`data/fruit_picking_dart/rm75b_eg2/train`，`scripts/generate_fruit_dart.py`）。脚本示范者与记录器不变，只对仿真执行的关节目标加逐关节 Ornstein–Uhlenbeck 噪声（平稳标准差 σ = 0.04 rad，相关时间 τ = 10 步），h5 里记录示范者的**干净**目标作为动作标签；每一段运动都从扰动后的实际关节状态重新规划，所以偏离状态上的标签指回示范轨迹。噪声只在机械臂移动时施加（reach、pull、carry 全幅，approach 半幅），观察、抓取、释放、静置时关闭。σ 由标定集 `data/fruit_picking_dart_calib/` 选定（TCP 位移中位约 2.5 cm），在任何 DART 训练或评测之前确定。
- 只保留成功示范（与干净示范同一重试规则，种子 = split_base + 16 × episode + attempt）。DART 接受率 count 0.57（175 次尝试，失败原因：没放进篮子 52、没摘下 23）、occluded_ripe 0.66（152 次尝试）。执行时 TCP 相对规划的位移中位 2.50 / 2.54 cm，90% 分位约 5 cm。
- **规模**（`data/finetune_kf/fruit_dart_rm75b_train/meta/build.json`，`scripts/run_build_fruit_dart_dataset.sh`，多源 `--part` plan）：400 个 episode，173,791 个执行样本（FruitCount 92,860，FruitOccludedRipe 80,931），检查全部通过。
- 归一化统计：与旧协议相同的流程在全部 400 个 episode 上重算（asset id `robomme`），写在 `data/finetune_kf/fruit_dart_rm75b_train/assets/robomme/norm_stats.json`。
- **不进训练的数据**：干净 dev episode 200–219（`data/fruit_picking_dev/rm75b_eg2/train`，只用于第 5 节的开发门）；val 全部 episode。

## 4. 训练

- 与旧采摘协议第 4 节相同：79999 初始化；官方 `scripts/train.py: main` 原样运行，只替换选帧；三个臂 k0 / k4 / k8 除 k 外完全相同；12000 步；全局 batch 64；AdamW（官方设置，梯度裁剪 1.0）；500 步线性预热到 2.5e-5，余弦衰减到 2.5e-6；EMA 0.999；视觉编码器冻结；随机种子 42；只保存 step 11999。
- **步数不变的理由**：诊断显示问题不在拟合程度（损失 0.0011），加步数不对症；保持 12000 步使次要对比 D1 = K0 − C0 只差训练数据（数据量翻倍后约 4.4 个 epoch，旧协议约 8.7 个）。
- **切卡（本机 8 × L20 48 GB）**：旧协议每臂 2 张 80 GB 卡（fsdp 2）；两张 48 GB 合计 96 GB 放不下，改为每臂 4 卡 FSDP（`fsdp_devices = 4`），GPU 组 (0–3) 与 (4–7)。**全局 batch 保持 64**（每卡 16），学习率与调度不变，所以优化轨迹只在数值上（分片与归约顺序）与 2 卡不同。
- **顺序**：k0（GPU 0–3）与 k4（GPU 4–7）并行 → T1（k0、k4）→ 第 5 节开发门 D → k8（GPU 0–3）→ T1（全部三臂）→ 第 6 节闭环评测。开发门不过则不训 k8、不做正式评测。
- 重启规则与 T1 与旧协议相同（最终 checkpoint 齐全；每臂最后一次运行的 240 条损失区间均值齐全且有限；checkpoint 带的归一化统计与第 3 节一致）。
- 执行：`scripts/run_fruit_dart_pipeline.sh RUN_DIR DEV_DIR EVAL_DIR`。

## 5. 开发门 D（只决定是否继续，不报告为结果）

- 数据：干净 dev episode 200–219，两个任务共 40 条；模型种子 7。这些 episode 从未用于训练、标定或任何评测。
- 条件：K0（本协议 k0 臂）与 C0（旧协议 `runs/fruit_finetune_kf_20261025/ckpt/mme_vla_suite/k0/11999`），同一服务进程装两个 checkpoint；4 个服务进程（GPU 0–3）并行，仿真在 GPU 4–7。
- 通过条件（事先确定）：K0 成功率 > 5%（40 条中至少 3 条成功），且 80 条中工程错误 ≤ 2。
- 不通过：停止，不训 k8、不做第 6 节；如实报告"DART（σ = 0.04 rad）未使 K0 脱离地板"，另立协议再决定下一步（例如更大 σ、更多 DART 数据或在线纠偏）。开发门的成功率不作为结论引用。
- 执行：`scripts/run_fruit_dev_gate.py --out results/fruit_dart_dev_gate_20261027`，结果 `dev_gate.json`。

## 6. 闭环评测

- 仿真与客户端与旧协议第 5 节相同（`FruitEnvRunner` 接官方 `EpisodeEvaluator`，每 16 步一次决策，步数上限 1300；距离由外部模块接口读取客户端逐帧记录的距离，核对图像摘要）。
- **val episode 50–99**，两个任务，模型种子 7、8、9：每任务 150 组，共 300 组。val 0–49 已用于旧协议的正式评测，不再用于本协议的结论。val 50–99 此前没有用于任何评测或调参。
- 条件：K0、K4、K8（本协议三臂的 step 11999）和 C0（旧干净 K0）。同一组内四个条件共享环境种子和策略随机数重置，顺序随机。
- **服务进程（48 GB 显存约束）**：四个 FP32 checkpoint 约 43 GB，单卡放不下。每个模型种子两个服务进程：一个装 K0/K4/K8（约 32.5 GB，与旧协议的单进程相同），一个装 C0；6 个服务进程占 GPU 0–5，仿真在 GPU 6–7。主对比的三个条件仍在同一进程内。
- val count episode 86 的真值 sidecar 文件损坏（全零字节，h5 与 manifest 完好）。闭环只用 manifest 的环境种子，距离由客户端实时计算，不读 sidecar，所以不影响评测；该文件在封存前按同一种子重新生成并核对 h5 摘要一致后替换（第 9 节记录）。

## 7. 统计与判定

- 配对 bootstrap 20000 次（种子 20261024），成功率差以百分点计。
- **主对比**（各任务分别，一个 Holm 族共 4 项）：G1 = K4 − K0，G2 = K8 − K0。α = 0.05，界限 δ = 12 个百分点。状态与每任务判定规则与旧协议第 6 节相同。
- **次要对比**（单独报告，不进主族）：D1 = K0 − C0，同样的 bootstrap 与状态定义。
- 地板规则：某任务 K0、K4、K8 的成功率都 ≤ 5% 时，该任务主对比报"地板效应，无法比较"。C0 不参与地板判断（它就是旧的地板参照）。
- 有效性门槛：G3 工程错误率 ≤ 2%；G5 每组内 K0/K4/K8 在同一服务进程、C0 在同一模型种子的另一个服务进程（每组恰好两个服务会话）；GA 选帧审计全部一致。任何门槛不过，结果只作工程诊断。

## 8. 不在本协议内

- 真机距离模块的噪声、缺测、遮挡（感知侧的离线分析见 `paper/`，不改变本协议的判定）。
- σ、DART 数据量或训练步数的扫描；与 79999 的比较。
- 真机部署。

## 9. 冒烟记录

以下在封存前完成，结果不参与判定。

1. **数据构建**：见第 3 节；`episodes.json` sha256 `d1dd1015…`。
2. **训练冒烟**（`runs/fruit_dart_kf_smoke_20261027`，100 步，5 步预热，三臂，每臂 4 卡 FSDP，全局 batch 64；`scripts/run_fruit_dart_smoke.sh`，2026-10-28）：T1 通过（`gate_t1.json`），三个最终 checkpoint 带的归一化统计与第 3 节一致，20 个损失区间均值齐全且有限。损失（第 0 步 → 最后一个区间均值）：k0 0.1174 → 0.0289，k4 0.1174 → 0.0289，k8 0.1174 → 0.0289。无 OOM；各卡显存（5 秒采样）均为 41.5 GB，即 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` 的预分配量。稳态速度：k0 ∥ k4 两组并行时 3.37 s/步，k8 单独时 3.10 s/步。总用时 15.6 分钟（k0、k4 并行，k8 随后，含装载、编译和保存）。
3. **闭环冒烟**（`results/fruit_dart_kf_eval_smoke_20261027`，冒烟 checkpoint step 99，C0 为旧 `k0/11999`；val episode 50、种子 7，两个任务 × 四个条件，两进程拆分）：G3、G5、GA 全部通过；工程错误 0；选帧审计 656 个决策，不一致 0 个。服务进程 0 装 K0/K4/K8（GPU 0），服务进程 1 装 C0（GPU 1），仿真在 GPU 6、7。结局：timeout 8（冒烟 checkpoint，不解读）；一个跑满 1301 步的 episode 墙钟 36–57 秒。各卡显存峰值（5 秒采样）：GPU 0 32.3 GB，GPU 1 16.3 GB，GPU 6、7 各 0.5 GB，其余 0。用时 6.4 分钟（含装载）。
4. **开发门冒烟**（`results/fruit_dart_dev_gate_smoke_20261027`，`run_fruit_dev_gate.py --smoke --checkpoint-run runs/fruit_dart_kf_smoke_20261027 --checkpoint-step 99`；dev episode 200 两个任务，K0 为冒烟 checkpoint、C0 为旧 `k0/11999`）：管线跑通，`dev_gate.json` 正常写出（4/4 个 episode，工程错误 0，选帧审计 269 个决策、不一致 0 个；判定字段为不通过，冒烟 checkpoint 不作门槛决定）。结局：timeout 3，C0 在 FruitOccludedRipe 上 fail 1。显存峰值 GPU 0 16.3 GB。用时 3.1 分钟。`--checkpoint-run` 只在 `--smoke` 下可用，正式开发门读第 4 节训练的 `k0/11999`。
5. **val count episode 86 sidecar 重建**（2026-10-28，`data/regen_val86/`）：用同一种子（201376）重新生成；新 h5 与原 h5 的关节状态在全部 471 步逐位相同，动作差 ≤ 1.7e-15 rad、布局四元数差 ≤ 1.2e-16（浮点末位），步数、果子数、要摘数相同。只替换损坏的 sidecar，保留原 h5；manifest 的 `gt_sha256` 改为新文件摘要（`fb95ccf9…`），并在 `gt_regenerated` 字段记录原摘要与原因；损坏文件和原 manifest 备份在 `data/regen_val86/`。
6. **正式耗时估计**（按冒烟速度）：第一轮 k0 ∥ k4 约 12000 × 3.37 s ≈ 11.2 小时；开发门 40 组 × 2 条件、4 个服务进程，约 0.5 小时；第二轮 k8 约 12000 × 3.10 s ≈ 10.3 小时；闭环评测 300 组 × 4 条件、每个模型种子一条通道，最坏（全部跑满 1301 步）约 6 小时。合计约 28 小时，另加装载与保存。
