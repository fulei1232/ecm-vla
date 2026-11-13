# 交接文档：采摘地板诊断 → DART 纠偏示范（2026-10-27）

> **给下一台机器上的人或 Agent 用。**  
> 本文档在共享盘 `/media/fulei/fuleivla` 下，不依赖 `/root`。  
> 写完时：原机已无可用 GPU；数据生成进程已停；DART 相关代码**未提交、未封存、未正式跑**。  
>
> **若目标机是 8×48GB L20（不是 80GB 卡）：请改读  
> [`HANDOFF_fruit_dart_L20_8x48G_20261027.md`](HANDOFF_fruit_dart_L20_8x48G_20261027.md)**  
> （训练须 4 卡 FSDP、评测勿单卡装 4 个 checkpoint；冲突以 L20 文档为准。）

---

## 0. 一分钟结论

| 项 | 状态 |
|---|---|
| RoboMME 关键帧微调评测 | **已完成**（`results/finetune_kf_eval_20261025/`） |
| 采摘 k0/k4/k8 微调 + 正式评测 | **已完成，地板效应**（`results/fruit_finetune_kf_eval_20261025/`） |
| 地板原因诊断 | **已完成**：闭环误差累积，非欠拟合 / 非 val 泛化 / 非夹爪格式 |
| 用户选择的下一步 | **DART 带扰动示范** → 重训 k0/k4/k8 → 新 val 评测 |
| DART 数据生成 | **中断、未完成**（可 resume） |
| DART 代码 | **半成品在 working tree，未 commit** |
| 正式协议 / 冒烟 / 封存 / 训练 | **都还没做** |

**下一机第一件事：** 在 `/media/fulei/fuleivla/ecm_vla` 核对 GPU 与路径，**先补完数据生成**，再写协议与冒烟，**封存后再训练**。

---

## 1. 机器与路径（新机注意）

### 1.1 共享盘上该用的根目录

```
/media/fulei/fuleivla/
├── ecm_vla/                          # 主仓库（git），当前工作都在这里
├── acm_vla/vendor/robomme_policy_learning/   # POLICY（训练/服务代码）
├── adaptive_evidence_acquisition_20261009/
│   ├── vendor/robomme_policy_learning/.venv/ # POLICY_PYTHON
│   └── models/perceptual-framesamp-modul/79999/  # 初始化 checkpoint
├── robomme_benchmark/.venv/          # SIM_PYTHON（仿真 + 示范生成）
└── acm_vla/cache/openpi/             # OPENPI_DATA_HOME
```

路径常量在 `ecm_vla/scripts/v_a_common.py`（`ECM`、`POLICY`、`POLICY_PYTHON`、`SIM_PYTHON`、`CHECKPOINT`、`OPENPI_DATA_HOME`）。**不要改成依赖 `/root` 的路径。**

### 1.2 新机有、旧机没有 / 不要依赖的东西

| 可用（共享） | 不可用 / 不要依赖 |
|---|---|
| `/media/fulei/fuleivla/**` 全部代码、数据、runs、results、logs | `/root/**`（Cursor terminals、agent-stores、agent-transcripts） |
| 4× A100 80GB（新机） | 旧机 GPU（已无） |

Cursor 会话历史只在旧机 `/root`；**以本交接文档 + git + `/media` 产物为准**。

### 1.3 新机启动前自检

```bash
cd /media/fulei/fuleivla/ecm_vla
nvidia-smi
test -x /media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python
test -x /media/fulei/fuleivla/robomme_benchmark/.venv/bin/python
test -d /media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999
ls /etc/vulkan/icd.d/nvidia_icd.json   # 仿真客户端需要；路径若不同，改环境变量 VK_ICD_FILENAMES
git status
git log -1 --oneline   # 期望 HEAD = d32446f（见下）
```

仿真生成 / 评测客户端环境变量习惯：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json   # 按新机实际路径改
export CUDA_VISIBLE_DEVICES=...
```

---

## 2. Git 状态（交接时快照）

```
分支: main
HEAD: d32446f Fruit diagnosis: task-space (tcp) replay error, 10 cm near-fruit phase, --summarize-only
```

### 2.1 已提交（可用）

| Commit | 内容 |
|---|---|
| `d4d0d5d` | 采摘正式协议封存 + smoke 记录 |
| `9ac3473` / `9b39ae7` | 采摘微调 study、评测客户端、距离模块 |
| `5d4db54` | `diagnose_fruit_episode.py` + `run_fruit_diagnosis.py` |
| `d32446f` | TCP 空间诊断、`--summarize-only`、10 cm near-fruit |

### 2.2 未提交（必须带到新机；都在 `/media` working tree）

**已修改：**

- `scripts/build_fruit_dataset.py` — `--part SOURCE FIRST STOP` 多源 plan
- `scripts/run_finetune_kf.py` — study `fruit_dart`；`gate --arms`
- `scripts/run_teacher_oracle.py` — stage `fruit_dart`（val 50–99；K0/K4/K8/C0；次要对比 D1）
- `scripts/run_fruit_diagnosis.py` — `start_server(..., conditions=..., base_port=...)`
- `src/memory_management/validation/teacher_oracle.py` — `C0` ∈ `KEYFRAME`

**未跟踪：**

- `scripts/generate_fruit_dart.py` — DART 生成器（已冒烟通过 1 条 episode）

**尚未写出（计划中，文件不存在）：**

- `scripts/run_fruit_dev_gate.py` — 开发集 go/no-go（k0/k4 训完后）
- `scripts/run_fruit_dart_pipeline.sh` — 训练 → 开发门 → k8 → 正式评测
- `reports/fruit_dart_protocol_20261027.md` — **正式协议（封存前必须写完 + smoke）**
- 对应 tests；smoke 记录；正式 seal

> **建议：** 上新机后立刻 `git status` 确认上述 diff 还在；若担心误清，先 `git stash push -u -m "fruit_dart_wip"` 或直接 commit WIP（用户若要求再正式整理 message）。

### 2.3 不要动的已封存产物

以下已用于正式结论，**禁止改 sealed 脚本内容去“修”旧结果**：

- `reports/fruit_finetune_kf_protocol_20261025.md`
- `runs/fruit_finetune_kf_20261025/`
- `results/fruit_finetune_kf_eval_20261025/`
- 正式评测用过的客户端身份文件（如 `eval_fruit_episode.py` 等）——新诊断/开发门用**新脚本**，不要改 sealed 身份文件。

诊断客户端 `diagnose_fruit_episode.py` 可继续复用（不在旧 fruit 正式评测 CODE 身份里作为唯一真相来源；开发门用它是刻意的）。

---

## 3. 已完成工作摘要

### 3.1 RoboMME 关键帧微调（方案 B）

- 协议：`reports/finetune_kf_protocol_20261025.md`
- 结果：`results/finetune_kf_eval_20261025/REPORT.md`
- 要点：门槛通过；无 k 明显优于 K0；成功率整体偏低。

### 3.2 采摘微调（无扰动示范，12000 步）

- 协议：`reports/fruit_finetune_kf_protocol_20261025.md`
- 训练：`runs/fruit_finetune_kf_20261025/`（k0/k4/k8，step **11999**）
- 评测：`results/fruit_finetune_kf_eval_20261025/`
- **判定：两任务地板效应**（K0/K4/K8 成功率都 ≤ 5%）
  - FruitCount：各条件 0/150
  - FruitOccludedRipe：K0 1/150，K4/K8 0/150
- 描述性现象：能接近/脱离部分果，但几乎完不成进篮；事件数远低于示范；记忆帧几乎不变 → **卡在基础操作，比不了记忆**。

### 3.3 诊断（用户选「先诊断」）

- 驱动：`scripts/run_fruit_diagnosis.py`
- 单集：`scripts/diagnose_fruit_episode.py`（`closed` / `replay`）
- TCP 汇总：`scripts/fruit_diagnosis_tcp.py`
- 结果：`results/fruit_diagnosis_20261027/{summary,tcp_summary}.json`

**结论（决定了 DART）：**

1. **开环（示范状态上预测）很准**：TCP 中位误差 ~0.4 cm（h1）/ ~1.7–2.4 cm（h16），远好于 hold 基线 ~11 cm。  
2. **train ≈ val** 开环误差 → 不是「没见过 val 布局」。  
3. **夹爪方向一致率 ~98–99%** → 不是夹爪表示问题。  
4. **闭环在 train 布局上也是 0/20 成功** → 不是欠拟合（loss 已到 0.0011）。  
5. **机制：闭环误差累积**——沿示范轨迹会跟，一偏就回不来。  
6. **单纯加步数不对症**；要对症加「偏离状态上的纠偏标签」→ DART（用户已选）。

---

## 4. 当前任务：DART 纠偏示范 + 重训 + 新 val 评测

### 4.1 方法（已定）

- **DART**（Laskey et al.）：仿真执行时对关节目标加时间相关噪声；h5 **记录干净规划动作**（绝对关节目标）。规划器每段从**实际（已扰动）状态**重规划 → 标签是纠偏动作。
- 生成器：`scripts/generate_fruit_dart.py`（**不改** `fruit_picking/` 包内 sealed 身份文件；复用其 Recorder/Solver）。
- 噪声：**OU**，σ = **0.04 rad**/关节，τ = 10 步；相位缩放 `reach/pull/carry=1`，`approach=0.5`，observe/grasp/release/settle = 0。
- 标定（episode 300–309，先定规则再看）：σ=0.04 时 TCP 中位扰动 ~2.5 cm，接受率 ~0.83（干净 ~0.96）。标定产物：`data/fruit_picking_dart_calib/`。

### 4.2 数据规划（已定编号，避免污染旧 val）

| 集合 | 路径 | episode | 用途 |
|---|---|---|---|
| 干净 train（已有） | `data/fruit_picking/rm75b_eg2/train` | 0–99 / task | 继续进训练集 |
| DART train | `data/fruit_picking_dart/rm75b_eg2/train` | **100–199** / task | 纠偏示范 |
| 干净 **dev** | `data/fruit_picking_dev/rm75b_eg2/train` | **200–219** / task | 仅 go/no-go，**不进训练、不进正式评测** |
| 新 **val** | `data/fruit_picking/rm75b_eg2/val` | **50–99** / task | 正式 `fruit_dart` 评测（旧 0–49 已用于正式 fruit 评测，**不可再作正式调参/结论**） |

种子公式不变：`seed = SPLIT_BASE[split] + 16 * episode + attempt`（见 `fruit_picking/generate.py`）。

### 4.3 数据生成进度（交接时，进程已停）

| 集合 | count | occluded_ripe | 目标 | 状态 |
|---|---:|---:|---:|---|
| DART 100–199 | **35** | **51** | 100 / 100 | **未完成** |
| Dev 200–219 | **13** | **16** | 20 / 20 | **未完成** |
| Val 50–99（累计 val 含旧 0–49） | **76**（即新段约 26） | **82**（新段约 32） | 累计 100 / 100 | **未完成** |
| 干净 train 0–99 | 100 | 100 | 100 | 完成 |

日志目录：`data/fruit_picking/logs/`  
例如：`dart_count_train_100-124.log`、`dev_count_train_200-219.log`、`count_val_50-74.log` …

生成器 **按 manifest 跳过已完成 episode**，直接重跑同一命令即可 resume。

### 4.4 训练 / 评测设计（已在未提交代码里起草）

**数据集（计划）：**

```
data/finetune_kf/fruit_dart_rm75b_train/
  = clean train 0–99  +  DART 100–199
```

构建示例（数据齐后）：

```bash
# 通过 run_build_fruit_dataset.sh 或直接调用；plan 需支持 --part（未提交改动）
python3 scripts/build_fruit_dataset.py plan --out data/finetune_kf/fruit_dart_rm75b_train \
  --part data/fruit_picking/rm75b_eg2/train 0 100 \
  --part data/fruit_picking_dart/rm75b_eg2/train 100 200
# 然后 shard / finish / norm-stats / check-encoder（照 scripts/run_build_fruit_dataset.sh）
```

注意：现有 `run_build_fruit_dataset.sh` 仍只传 `--episodes`；要么临时改 shell 传多个 `--part`，要么写一个 `run_build_fruit_dart_dataset.sh`。

**训练 study `fruit_dart`（未提交）：**

- 目录：`runs/fruit_dart_kf_20261027/`
- 臂：k0/k4/k8，12000 步，own assets，与 fruit 相同超参
- 流水线意图：先训 k0∥k4 → **开发门 D** → 通过再训 k8 → 正式评测

**开发门 D（脚本未写完）：**

- 在 `data/fruit_picking_dev` 上闭环对比 **K0（新 DART 训练）** vs **C0（旧干净 K0 = `runs/fruit_finetune_kf_20261025/.../k0/11999`）**
- 通过条件草案：40 条（20×2 task）上 K0 成功率 **> 5%**，工程错误 ≤ 2
- 不过则**停**，不训 k8、不开正式评测（省 ~9h）

**正式评测 stage `fruit_dart`（未提交）：**

- val episode **50–99**，种子 7/8/9
- 条件：**K0, K4, K8, C0**
  - K*：新 DART run 的 checkpoint
  - C0：旧 fruit 的 k0/11999，记忆与 K0 相同（均匀 16）
- 主对比：G1=K4−K0，G2=K8−K0（地板规则**只看主对比条件**）
- 次要：D1=K0−C0（报告 DART 相对干净微调是否抬地板）
- 客户端：仍用 `eval_fruit_episode.py`；距离：`fruit_distance:RecordedDistance`

---

## 5. 新机上如何继续（按顺序做）

### Step A — 环境与 WIP 确认（15 min）

1. 完成 §1.3 自检。  
2. `cd /media/fulei/fuleivla/ecm_vla && git status`：应看到 §2.2 的 5 个 modified + 1 个 untracked。  
3. 可选：先 commit WIP，避免另一会话覆盖。

### Step B — 补完数据生成（CPU 仿真为主，可多进程；约 1–2 h）

生成用 **SIM_PYTHON**（`robomme_benchmark/.venv`），与之前相同：

```bash
cd /media/fulei/fuleivla/ecm_vla
PY=/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json   # 按机改
L=data/fruit_picking/logs

# DART（σ=0.04）；已有 manifest 的 episode 会跳过
for t in count occluded_ripe; do
  for r in 100-124 125-149 150-174 175-199; do
    CUDA_VISIBLE_DEVICES=$((RANDOM%4)) $PY scripts/generate_fruit_dart.py \
      --task $t --episodes $r --sigma 0.04 --out data/fruit_picking_dart \
      >> $L/dart_${t}_train_$r.log 2>&1 &
  done
done

# Dev 干净
for t in count occluded_ripe; do
  CUDA_VISIBLE_DEVICES=$((RANDOM%4)) $PY -m fruit_picking.generate \
    --task $t --split train --episodes 200-219 --out data/fruit_picking_dev \
    >> $L/dev_${t}_train_200-219.log 2>&1 &
done

# 新 val 50-99（写入原 val 树，与 0-49 并存）
for t in count occluded_ripe; do
  for r in 50-74 75-99; do
    CUDA_VISIBLE_DEVICES=$((RANDOM%4)) $PY -m fruit_picking.generate \
      --task $t --split val --episodes $r --out data/fruit_picking \
      >> $L/${t}_val_$r.log 2>&1 &
  done
done
wait

# 验收
for d in data/fruit_picking_dart/rm75b_eg2/train data/fruit_picking_dev/rm75b_eg2/train; do
  for t in count occluded_ripe; do echo -n "$d $t "; ls $d/$t/ep*_manifest.json | wc -l; done
done
# dart 期望各 100；dev 期望各 20
for t in count occluded_ripe; do echo -n "val $t "; ls data/fruit_picking/rm75b_eg2/val/$t/ep*_manifest.json | wc -l; done
# val 期望各 100（0-99）
```

任一 `ep*_FAILED.json` 需查原因；`MAX_ATTEMPTS=16` 仍失败才写 FAILED。

### Step C — 补齐缺失脚本与协议（代码，封存前）

必须补全（草案见本交接 §4.4 / 先前会话意图）：

1. `scripts/run_fruit_dev_gate.py`  
2. `scripts/run_fruit_dart_pipeline.sh`（或等价）：  
   - prepare study fruit_dart  
   - train `--arms k0 k4` → gate `--arms k0 k4`  
   - run_fruit_dev_gate → 失败则 exit  
   - train `--arms k8` → gate 全臂  
   - `run_teacher_oracle.py all --stage fruit_dart`  
3. `reports/fruit_dart_protocol_20261027.md`（照 `fruit_finetune_kf_protocol_20261025.md` 结构；写明数据、σ、dev 门、val 50–99、C0、统计与地板规则）  
4. 扩展 `run_build_fruit_dataset.sh` 或新 build 脚本支持 `--part`  
5. 测试：`fruit_dart` stage_params、C0∈KEYFRAME、finetune_route 地板只看主对比、`--part` plan  

**约束（长期有效）：**

- 正式跑前：**封存协议 + smoke 记录写进协议第 9 节**  
- 不放松门槛；不用旧 val 0–49 做正式调参  
- 不改 `sim_branch.py` / `planner_branch.py` / `phase_expert.py`  
- 不改已完成正式跑的 sealed code-identity 文件内容  
- 用户说「后台运行、不用监控」时：nohup/setsid 后不要轮询  

### Step D — 构建数据集（GPU 分片，约数小时）

```bash
# 数据齐且代码就绪后
bash scripts/run_build_fruit_dart_dataset.sh data/finetune_kf/fruit_dart_rm75b_train
# 或改版 run_build_fruit_dataset.sh
```

检查：`meta/build.json` passed、encoder check、`assets/robomme/norm_stats.json`。

### Step E — 冒烟 → 写协议记录 → commit → 封存 → 正式链

1. 训练冒烟：`prepare --study fruit_dart --smoke` → train → gate（100 步）  
2. 闭环冒烟：`run_teacher_oracle.py all --smoke --stage fruit_dart --checkpoint-run <smoke> --checkpoint-step 99`  
   - 冒烟替换 **仅 K\*** checkpoint；**C0 仍指向正式旧 fruit k0**（代码已按 `startswith("K")` 处理）  
3. 把 smoke 写入协议文档 → commit  
4. `prepare --study fruit_dart` 封存 `runs/fruit_dart_kf_20261027`  
5. 后台启动 pipeline（预计：**两轮训练各 ~9 h + 开发门 ~1 h + 正式评测 ~4–5 h**；若开发门失败则提前停）

### Step F — 若开发门失败

不要硬训 k8 / 不要开正式评测。可选后续（需新协议，且仍用未污染的 episode 编号）：

- 加大 σ 或提高 DART 比例  
- 真 DAgger（策略 rollout + `FruitPickSolver` 标注）  
- 增加干净/扰动混合比例扫描  

---

## 6. 关键结果与检查点路径速查

```
# 诊断
results/fruit_diagnosis_20261027/summary.json
results/fruit_diagnosis_20261027/tcp_summary.json

# 旧采摘正式（地板）
results/fruit_finetune_kf_eval_20261025/REPORT.md
runs/fruit_finetune_kf_20261025/ckpt/mme_vla_suite/k{0,4,8}/11999/

# 旧干净训练数据 / 特征
data/fruit_picking/rm75b_eg2/{train,val}/
data/finetune_kf/fruit_rm75b_train/

# DART 进行中
data/fruit_picking_dart/
data/fruit_picking_dev/
data/fruit_picking_dart_calib/
data/fruit_picking/logs/dart_*.log
```

解释器：

```
POLICY_PYTHON = .../adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python
SIM_PYTHON    = .../robomme_benchmark/.venv/bin/python
```

训练：每臂 **2 GPU FSDP**，batch 64；GPU 对 `(0,1)` / `(2,3)` 与 `run_finetune_kf.py` 中 `GPU_PAIRS` 一致。A100 80G 与原先 A800 80G 同量级，**超参可先不动**；若 OOM 再单独开协议改 batch（不要静默改）。

---

## 7. 站立规则（续作必须遵守）

1. **协议先于正式跑**；smoke 进协议后再 seal。  
2. **val 0–49 已用于正式 fruit 评测** → 新正式结论只用 **50+**。  
3. **不放宽门、不测集调参、不选择性报告。**  
4. **不改 sealed 身份文件** 去「美化」旧跑；新逻辑进新脚本。  
5. **不修改** `sim_branch.py` / `planner_branch.py` / `phase_expert.py`。  
6. 长任务：**后台跑、用户说不用监控就不要轮询。**  
7. 需要可恢复的 seal hash 时：**先 commit 再 seal。**  
8. 拒绝「把数据做漂亮就算方法」类操纵。

---

## 8. 给下一任 Agent 的最短提示词

可直接粘贴：

```
继续 /media/fulei/fuleivla/ecm_vla 的采摘 DART 工作。
先读 reports/HANDOFF_fruit_dart_20261027.md。
共享盘在 /media；不要依赖 /root。
HEAD 应为 d32446f；working tree 有未提交的 fruit_dart 改动 + scripts/generate_fruit_dart.py。
优先：resume 补完 DART/dev/val 数据 → 补 run_fruit_dev_gate.py、pipeline、协议文档 → build → smoke → seal → 后台正式链。
用户已选 DART（σ=0.04）；正式 val 用 episode 50–99；C0=旧干净 K0 checkpoint。
遵守交接文档第 7 节站立规则。回复用中文。
```

---

## 9. 本文档位置与维护

- 路径：`/media/fulei/fuleivla/ecm_vla/reports/HANDOFF_fruit_dart_20261027.md`  
- 写成时：2026-10-27；机器切换原因：原机无 GPU，新机 4×A100 80G 可挂同一 `/media`。  
- 数据生成数字以交接当日 `ls …/ep*_manifest.json | wc -l` 为准；resume 后请在本节或协议 smoke 中更新「完成数」。

---

## 附录 A — 未提交文件清单（checklist）

```
 M scripts/build_fruit_dataset.py
 M scripts/run_finetune_kf.py
 M scripts/run_fruit_diagnosis.py
 M scripts/run_teacher_oracle.py
 M src/memory_management/validation/teacher_oracle.py
?? scripts/generate_fruit_dart.py
（待写）scripts/run_fruit_dev_gate.py
（待写）scripts/run_fruit_dart_pipeline.sh
（待写）reports/fruit_dart_protocol_20261027.md
```

## 附录 B — 为何必须新 val 50–99

正式 `fruit` 评测已用完 val **0–49** × 3 seeds × 3 conditions。若再用同一批 episode 做 DART 后的正式结论，属于测集回看。协议与 `STAGES["fruit_dart"]["episode_ids"] = range(50, 100)` 已按此设计（代码未提交但意图已定）。
