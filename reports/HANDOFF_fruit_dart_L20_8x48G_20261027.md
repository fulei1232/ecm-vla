# 交接文档：换到 8×48GB L20 继续采摘 DART（2026-10-27）

> **给挂着同一 `/media`、配备 8 张 48GB L20 的机器上的人或 Agent 用。**  
> 不依赖 `/root`（Cursor terminals / agent-stores / transcripts 都在旧机本地，新机读不到）。  
> 通用进度与站立规则亦见同目录 `HANDOFF_fruit_dart_20261027.md`；**本文以 L20 显存与切卡为准，冲突时以本文为准。**

---

## 0. 一分钟结论

| 项 | 状态 |
|---|---|
| 采摘正式评测（干净示范） | **已完成，地板效应** → `results/fruit_finetune_kf_eval_20261025/` |
| 诊断 | **已完成**：闭环误差累积 → 用户选 **DART** |
| DART 数据 | **未完成，可 resume**（见 §4） |
| DART 代码 | **半成品未 commit**（见 §2） |
| 正式协议 / 冒烟 / 封存 / 训练 | **都还没做** |
| 本机 GPU | **8× L20 48GB**（不是 80GB A800/A100） |

**能不能跑？能。** 但训练不能照搬旧协议的「2 卡 FSDP + 全局 batch 64」；评测不要在一张 48GB 上同时装 4 个 checkpoint。改动写进**尚未封存**的 DART 协议即可，**不要改已封存的旧 fruit 训练超参**。

**上机第一件事：** 自检 → **先补完数据生成** → 改切卡代码进协议草案 → smoke 确认不 OOM → 封存 → 正式链。

---

## 1. 机器与路径

### 1.1 共享盘根目录（新机必须能访问）

```
/media/fulei/fuleivla/
├── ecm_vla/                          # 主仓库（git），所有续作在这里
├── acm_vla/vendor/robomme_policy_learning/          # POLICY
├── adaptive_evidence_acquisition_20261009/
│   ├── vendor/robomme_policy_learning/.venv/bin/python   # POLICY_PYTHON
│   └── models/perceptual-framesamp-modul/79999/          # 初始化 ckpt
├── robomme_benchmark/.venv/bin/python               # SIM_PYTHON
└── acm_vla/cache/openpi/                            # OPENPI_DATA_HOME
```

路径常量：`ecm_vla/scripts/v_a_common.py`。**禁止改成 `/root/...`。**

### 1.2 可用 / 不可用

| 可用 | 不可用 |
|---|---|
| `/media/fulei/fuleivla/**` | `/root/**` |
| 本机 `nvidia-smi` 应看到 **8 张 L20，各约 48GB** | 旧机 GPU；旧 Cursor 会话 |

### 1.3 上机自检（复制执行）

```bash
cd /media/fulei/fuleivla/ecm_vla
nvidia-smi --query-gpu=index,name,memory.total --format=csv
# 期望：8 行，L20，memory.total ≈ 46068 MiB

test -x /media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python
test -x /media/fulei/fuleivla/robomme_benchmark/.venv/bin/python
test -d /media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999

# Vulkan（仿真客户端）；路径因机而异
ls /etc/vulkan/icd.d/nvidia_icd.json || ls /usr/share/vulkan/icd.d/*.json

git status
git log -1 --oneline   # 期望 HEAD = d32446f
```

仿真 / 评测客户端环境变量：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json   # 按本机实际改
```

---

## 2. Git 状态（交接快照）

```
分支: main
HEAD: d32446f Fruit diagnosis: task-space (tcp) replay error, 10 cm near-fruit phase, --summarize-only
```

### 2.1 已提交（可直接用）

| Commit | 内容 |
|---|---|
| `d4d0d5d` | 采摘正式协议封存 + smoke |
| `9ac3473` / `9b39ae7` | 采摘微调 study、评测客户端、距离模块 |
| `5d4db54` / `d32446f` | 地板诊断客户端与驱动、TCP 汇总 |

### 2.2 未提交（working tree，须带到本机；都在 `/media`）

**已修改：**

- `scripts/build_fruit_dataset.py` — `--part` 多源 plan  
- `scripts/run_finetune_kf.py` — study `fruit_dart`；`gate --arms`（**仍写着旧 `GPU_PAIRS=([0,1],[2,3])`，上 L20 必须改，见 §5**）  
- `scripts/run_teacher_oracle.py` — stage `fruit_dart`（val 50–99；K0/K4/K8/C0；次要 D1）  
- `scripts/run_fruit_diagnosis.py` — `start_server` 支持多条件  
- `src/memory_management/validation/teacher_oracle.py` — `C0` ∈ `KEYFRAME`  

**未跟踪：**

- `scripts/generate_fruit_dart.py` — DART 生成器（已单条冒烟通过）  
- `reports/HANDOFF_fruit_dart_20261027.md` — 通用交接（A100 表述）  
- `reports/HANDOFF_fruit_dart_L20_8x48G_20261027.md` — **本文**  

**尚未写出：**

- `scripts/run_fruit_dev_gate.py`  
- `scripts/run_fruit_dart_pipeline.sh`  
- `reports/fruit_dart_protocol_20261027.md`  
- 支持 `--part` 的 build 包装脚本；相关 tests  

建议上机后立刻确认 `git status` 仍有上述 diff；可先 WIP commit 防丢。

### 2.3 禁止改动的已封存产物

- `reports/fruit_finetune_kf_protocol_20261025.md`  
- `runs/fruit_finetune_kf_20261025/`（含 `ckpt/.../k{0,4,8}/11999`，C0 要用）  
- `results/fruit_finetune_kf_eval_20261025/`  
- 正式评测 sealed 身份文件内容（如 `eval_fruit_episode.py`）— 新逻辑进新脚本  

---

## 3. 已完成工作（为何做 DART）

### 3.1 采摘干净微调（12000 步）→ 地板

- 报告：`results/fruit_finetune_kf_eval_20261025/REPORT.md`  
- FruitCount / FruitOccludedRipe：K0/K4/K8 成功率都 ≤ 5%，**无法比较记忆**  
- Checkpoint：`runs/fruit_finetune_kf_20261025/ckpt/mme_vla_suite/k{0,4,8}/11999/`  

### 3.2 诊断结论（决定 DART）

结果：`results/fruit_diagnosis_20261027/{summary,tcp_summary}.json`

1. 开环（示范状态）很准：TCP 中位 ~0.4 cm（h1）/ ~1.7–2.4 cm（h16）≪ hold ~11 cm  
2. train ≈ val 开环 → 不是「没见过 val」  
3. 夹爪方向一致 ~98–99% → 不是表示问题  
4. **闭环在 train 布局也 0 成功** → 不是欠拟合（loss 已到 0.0011）  
5. **机制：闭环误差累积** → 加步数不对症；要对症加偏离状态的纠偏标签 → **DART**（已选定）

---

## 4. 当前任务：DART 数据 + 重训 + 新 val

### 4.1 方法（已定）

- 脚本：`scripts/generate_fruit_dart.py`（包外；**不改** `fruit_picking/` 内 sealed 文件）  
- 执行时对关节目标加 OU 噪声；h5 记**干净**规划动作；规划器从扰动后状态重规划  
- σ = **0.04 rad**，τ = 10；`reach/pull/carry=1`，`approach=0.5`，observe/grasp/release/settle=0  
- 标定目录：`data/fruit_picking_dart_calib/`（TCP 中位扰动 ~2.5 cm，接受率 ~0.83）

### 4.2 数据编号（已定）

| 集合 | 路径 | episode | 用途 |
|---|---|---|---|
| 干净 train（已有） | `data/fruit_picking/rm75b_eg2/train` | 0–99 | 进训练集 |
| DART train | `data/fruit_picking_dart/rm75b_eg2/train` | **100–199** | 纠偏示范 |
| 干净 **dev** | `data/fruit_picking_dev/rm75b_eg2/train` | **200–219** | 仅 go/no-go，**不进训练/正式评测** |
| 新 **val** | `data/fruit_picking/rm75b_eg2/val` | **50–99** | 正式 `fruit_dart`（0–49 已用过，不可再作正式结论） |

### 4.3 生成进度（写本文时；进程已停）

| 集合 | count | occluded_ripe | 目标 |
|---|---:|---:|---:|
| DART 100–199 | 35 | 51 | 100 / 100 |
| Dev 200–219 | 13 | 16 | 20 / 20 |
| Val 累计（含 0–49） | 76 | 82 | 100 / 100 |

日志：`data/fruit_picking/logs/dart_*.log`、`dev_*.log`、`*_val_5*.log`  
生成器遇已有 `ep*_manifest.json` **会跳过** → 直接重跑同一命令即可 resume。

### 4.4 Resume 命令（本机 8 卡，可多开）

```bash
cd /media/fulei/fuleivla/ecm_vla
PY=/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
L=data/fruit_picking/logs
i=0

for t in count occluded_ripe; do
  for r in 100-124 125-149 150-174 175-199; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) $PY scripts/generate_fruit_dart.py \
      --task $t --episodes $r --sigma 0.04 --out data/fruit_picking_dart \
      >> $L/dart_${t}_train_$r.log 2>&1 &
    i=$((i+1))
  done
done

for t in count occluded_ripe; do
  CUDA_VISIBLE_DEVICES=$((i % 8)) $PY -m fruit_picking.generate \
    --task $t --split train --episodes 200-219 --out data/fruit_picking_dev \
    >> $L/dev_${t}_train_200-219.log 2>&1 &
  i=$((i+1))
done

for t in count occluded_ripe; do
  for r in 50-74 75-99; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) $PY -m fruit_picking.generate \
      --task $t --split val --episodes $r --out data/fruit_picking \
      >> $L/${t}_val_$r.log 2>&1 &
    i=$((i+1))
  done
done
wait

# 验收
for t in count occluded_ripe; do
  echo dart/$t $(ls data/fruit_picking_dart/rm75b_eg2/train/$t/ep*_manifest.json | wc -l)   # 100
  echo dev/$t  $(ls data/fruit_picking_dev/rm75b_eg2/train/$t/ep*_manifest.json | wc -l)    # 20
  echo val/$t  $(ls data/fruit_picking/rm75b_eg2/val/$t/ep*_manifest.json | wc -l)          # 100
done
```

---

## 5. L20 48GB：训练 / 评测必须改的地方

### 5.1 旧配置（80GB×4）为何不能原样搬

旧 fruit / RoboMME 微调协议：

- 全局 **batch 64**  
- 每臂 **2 GPU FSDP**（`fsdp_devices=2`）  
- `GPU_PAIRS = ([0,1], [2,3])` → 两臂并行  
- 代码注释写明：**batch 64 单卡 80GB 放不下**，所以 FSDP=2  

两张 48GB 合计 96GB < 两张 80GB 的 160GB → **原样 2 卡 FSDP 很可能 OOM**。

### 5.2 推荐训练切卡（写进未封存的 DART 协议）

| 项 | 旧（A800 80G×4） | **本机 L20 48G×8** |
|---|---|---|
| 全局 batch | 64 | **保持 64**（勿静默改；改了要写协议并说明） |
| `fsdp_devices` | 2 | **4** |
| 每臂 GPU | 2 | **4** |
| 并行臂数 | 2 | **2**（仍两轮：先 k0∥k4，开发门后再 k8） |
| GPU 分组 | `(0,1)` / `(2,3)` | **`(0,1,2,3)` / `(4,5,6,7)`** |
| 每卡样本 | ~32 | ~16（64÷4） |
| 每臂墙钟 | ~9 h（A800） | **预计更长**（L20 算力更低；smoke 后估正式时间） |

改代码落点（封存前）：

1. `scripts/run_finetune_kf.py`  
   - `TRAIN["fsdp_devices"] = 4`（或仅 `fruit_dart` study 覆盖）  
   - `GPU_PAIRS = ([0,1,2,3], [4,5,6,7])`  
2. `reports/fruit_dart_protocol_20261027.md` 第 4 节写明上述切卡与「为何不改全局 batch」  
3. **不要**改已封存 `fruit` / RoboMME 的 protocol 或旧 `runs/`  

约束：`batch_size % jax.device_count() == 0` → 64 % 4 == 0，OK。

### 5.3 训练冒烟（上正式前必做）

```bash
# 切卡改完并 WIP commit 之后
python3 scripts/run_finetune_kf.py prepare --study fruit_dart --smoke \
  --out runs/fruit_dart_kf_smoke_20261027 \
  --dataset data/finetune_kf/fruit_dart_rm75b_train   # 数据集建好后

python3 scripts/run_finetune_kf.py train --out runs/fruit_dart_kf_smoke_20261027
# 另开终端盯显存：
nvidia-smi -l 5
python3 scripts/run_finetune_kf.py gate --out runs/fruit_dart_kf_smoke_20261027
```

若仍 OOM：优先试 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`；再考虑 **fsdp=8、单臂占满 8 卡、臂串行**（仍保持全局 batch 64）。**降 batch 是最后手段**，必须写进协议并说明与旧 fruit 不可直接比绝对 loss 曲线以外的东西——成功率对比仍可做，但训练动力学变了要诚实写明。

### 5.4 闭环评测显存（48GB 更紧）

旧 fruit 冒烟：单卡装 **3** 个 checkpoint，峰值约 **32.5 GB**。  
DART 正式条件是 **K0/K4/K8/C0 共 4 个** → 按比例约 **~43 GB**，48GB 余量只够几 GB，JAX 编译临时缓冲容易顶满。

**推荐（写进协议）：**

- 不要让一张服务卡同时 load 4 个 ckpt。  
- 例如：每 seed 开 **2 个服务进程**，各挂 2 个条件；或 4 进程各挂 1 个（卡多够用）。  
- `sim_gpu` **不要**与正在 load 大模型的 `policy_gpu` 挤同一张（旧代码有 `sim_gpu = gpus[(k+2)%len]`，8 卡时重新核对）。  
- 服务端保持 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。  
- 闭环冒烟时用 `nvidia-smi` 记峰值，写入协议第 9 节。

`run_teacher_oracle.py` 里 `servers_per_seed` / `gpus` 列表在 `prepare` 时写入 `server*/config.json`；封存前按上面策略改 stage 或 runtime。

### 5.5 数据集构建

`shard` 每进程一卡即可；8 卡可 `--of 8`。48GB 对特征编码足够。  
plan 用多源（未提交的 `--part`）：

```text
clean train 0–99  +  DART train 100–199
→ data/finetune_kf/fruit_dart_rm75b_train/
```

现有 `scripts/run_build_fruit_dataset.sh` 只传 `--episodes`；需改 shell 或新建 `run_build_fruit_dart_dataset.sh` 传两个 `--part`。

---

## 6. 评测 / 训练设计（与卡无关的部分）

### 6.1 开发门 D（脚本待写）

- 数据：`data/fruit_picking_dev` ep 200–219 × 2 task = 40  
- 对比：**K0**（DART 新训）vs **C0**（旧干净 `.../k0/11999`）  
- 通过草案：K0 成功率 **> 5%**，工程错误 ≤ 2  
- 时机：k0∥k4 训完、**k8 之前**；不过则停，省约一整轮训练  

### 6.2 正式 stage `fruit_dart`（代码未提交但已起草）

- val **50–99**，seeds 7/8/9  
- 条件 K0/K4/K8/C0；主对比 G1/G2；次要 D1=K0−C0；地板规则**只看主对比条件**  
- 客户端：`eval_fruit_episode.py`；距离：`fruit_distance:RecordedDistance`  
- 跑目录计划：`runs/fruit_dart_kf_20261027/`，`results/fruit_dart_kf_eval_20261027/`  

### 6.3 流水线顺序（待写成 `run_fruit_dart_pipeline.sh`）

1. prepare study fruit_dart（封存后）  
2. train `--arms k0 k4` → gate `--arms k0 k4`  
3. `run_fruit_dev_gate.py` → 失败 exit  
4. train `--arms k8` → gate 全臂  
5. `run_teacher_oracle.py all --stage fruit_dart`  

用户说「后台运行、不用监控」→ `nohup setsid ... &` 后不要轮询。

---

## 7. 推荐执行顺序（本机 checklist）

| 步 | 内容 | 卡需求 |
|---|---|---|
| A | §1.3 自检；确认 §2.2 WIP 在 | — |
| B | Resume 补完 DART/dev/val（§4.3） | 渲染/小显存，可占多卡 |
| C | 改 `fsdp_devices=4` + `GPU_PAIRS` 两组四卡；补 dev_gate / pipeline / 协议草案 / build `--part` / tests | — |
| D | 构建 `fruit_dart_rm75b_train`（plan/shard/finish/norm/encoder） | 8 卡可并行 shard |
| E | **训练 smoke（100 步）+ 记显存**；不过则按 §5.3 升级切卡 | 先 4+4 |
| F | 闭环 smoke（注意 §5.4 拆 checkpoint）；写入协议第 9 节 → commit → **seal** | 多卡拆服务 |
| G | 后台正式 pipeline | 训练 4+4；评测按协议 |

预计墙钟：数据补齐 ~1–2 h；build 数小时；训练两轮各长于旧机 9 h；开发门 ~1 h；正式评测数小时（4 条件 × 50 ep × 3 seed）。

---

## 8. 站立规则（必须遵守）

1. **协议先于正式跑**；smoke 进协议后再 seal。  
2. **正式 val 只用 50–99**；0–49 已用于旧 fruit 正式评测。  
3. **不放宽门、不测集调参、不选择性报告。**  
4. **不改 sealed 旧跑身份文件**；L20 切卡只写进新 DART 协议。  
5. **不修改** `sim_branch.py` / `planner_branch.py` / `phase_expert.py`。  
6. 长任务后台跑；用户说不用监控就不要盯。  
7. Seal 前 **commit**，保证 hash 可恢复。  

9. **全局 batch 默认保持 64**；若被迫改，协议里单独成条说明。

---

## 9. 关键路径速查

```
诊断:   results/fruit_diagnosis_20261027/
旧正式: results/fruit_finetune_kf_eval_20261025/REPORT.md
旧 ckpt: runs/fruit_finetune_kf_20261025/ckpt/mme_vla_suite/k{0,4,8}/11999/
旧数据: data/fruit_picking/rm75b_eg2/{train,val}/
旧特征: data/finetune_kf/fruit_rm75b_train/
DART中: data/fruit_picking_dart/  data/fruit_picking_dev/  data/fruit_picking_dart_calib/
日志:   data/fruit_picking/logs/

POLICY_PYTHON = .../adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python
SIM_PYTHON    = .../robomme_benchmark/.venv/bin/python
```

---

## 10. 给下一任 Agent 的最短提示词

```
继续 /media/fulei/fuleivla/ecm_vla 的采摘 DART。
先读 reports/HANDOFF_fruit_dart_L20_8x48G_20261027.md（本机是 8×48GB L20，不是 80GB）。
不要依赖 /root。HEAD 应为 d32446f；working tree 有未提交 fruit_dart 改动 + generate_fruit_dart.py。
优先：resume 补数据 → 把训练改成每臂 4 卡 FSDP、GPU_PAIRS=(0-3)/(4-7)、全局 batch 仍 64 → 补协议/dev_gate/pipeline → build → smoke（确认不 OOM）→ seal → 后台正式链。
评测不要单卡同时装 4 个 checkpoint。正式 val 用 episode 50–99；C0=旧干净 K0。
遵守文档第 8 节。回复用中文。
```

---

## 附录 A — WIP 文件清单

```
 M scripts/build_fruit_dataset.py
 M scripts/run_finetune_kf.py          # 上机后还要改 fsdp / GPU_PAIRS
 M scripts/run_fruit_diagnosis.py
 M scripts/run_teacher_oracle.py
 M src/memory_management/validation/teacher_oracle.py
?? scripts/generate_fruit_dart.py
?? reports/HANDOFF_fruit_dart_20261027.md
?? reports/HANDOFF_fruit_dart_L20_8x48G_20261027.md
（待写）scripts/run_fruit_dev_gate.py
（待写）scripts/run_fruit_dart_pipeline.sh
（待写）reports/fruit_dart_protocol_20261027.md
```

## 附录 B — 与「4×A100 80GB」交接的差异

| | A100 80G×4 文档 | **本文 L20 48G×8** |
|---|---|---|
| 训练切卡 | 可沿用 2×FSDP | **必须 4×FSDP（或 8×串行）** |
| 评测 4 ckpt | 单卡可能紧 | **必须拆服务 / 少挂 ckpt** |
| GPU 列表 | 0–3 | **0–7** |
| 文档 | `HANDOFF_fruit_dart_20261027.md` | **本文** |

若实际机器是 80GB 卡，改回读通用交接，不要用本文的 4 卡 FSDP 假设去改已经按 80GB 冒烟通过的配置。
