# HarvestMem v3 评测任务清单（交给 Codex 执行）

日期：2026-11-11。审阅人：Claude。执行人：Codex。  
本清单接在 `harvestmem_v3_codex_tasks_20261110.md` 之后，全局规则继续有效，本文只补充评测相关的内容。

## 0. 背景：上一次评测作废（必读）

`runs/harvestmem_v3_eval_20261111/` 里的 SG 0/20、K0 0/20 **不是采摘任务的结果，整份作废**：

| 证据 | 位置 |
|---|---|
| 跑的是 RoboMME 基准任务 `VideoUnmask` / `VideoPlaceOrder`，`task_goal` 是 "watch the video carefully, then pick up the container hiding the green cube" | `sg/VideoUnmask/val_00/result.json` |
| 用的是 RoboMME 的 `val` 划分、种子 1060000，不是我们的 dev 种子 4200000+16e | 同上 |
| 回放画面是 Panda 机械臂 + 木桌 + 方块/杯子，没有果树、果篮、RM75 机械臂 | `/tmp/eval_frames/*.png` |
| 步数上限 1300（Count 应为 2800）；SG 的 prompt 里没有任何子目标 | 同上 |
| 能跑完不报错，只是因为两边状态恰好都是 8 维 | — |

训练本身经核查基本正确：数据集、79999 初始化、12000 步、batch 64、学习率、EMA、seed 都与方针一致；`norm_stats.json` 与数据集一致；SG 的 `grounded_subgoal` 在 tokenize 之前拼进 prompt，确实进入了模型。偏差见 E0。

SG 与 K0 的 loss 曲线几乎重合（0.1495 → 约 0.0023）是预期内的：开始/停止这类决策帧很稀疏，在平均 loss 里看不出来。**loss 不能作为结论，只能看采摘环境里的闭环。**

## 1. 已核实的代码事实

| 事实 | 位置 |
|---|---|
| SG checkpoint | `/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/sg/ckpt/mme_vla_suite/sg/11999` |
| K0 checkpoint | 同上，把 `sg` 换成 `k0` |
| 训练日志 | `runs/harvestmem_v3_S2_20261111/{sg,k0}/logs/*_12000.log`；起止 UTC 03:34 → 12:23，约 8.8 h |
| 训练集 prompt 是去掉等灯前缀后的小写指令，例如 `put exactly 2 fruits into the basket. do not count windfalls. when finished, return to the initial rest pose, open the gripper, and hold still.` | `data/harvestmem_v3_dataset_20261110/data/0.pkl` |
| SG 的 prompt 必须用 `compose_prompt(instruction, grounded_subgoal)`，即 `<指令小写> current subgoal: <子目标>.` | `scripts/harvestmem_v3/subgoal.py` |
| 可复用的采摘闭环客户端：30 步只记历史的观察段、16 步一块执行、`guard_v2`、`trace.jsonl`/`actions.jsonl`/`result.json` | `scripts/overnight_r12/eval_episode.py` |
| 可复用的服务端：`memory=uniform` 等价于训练时的均匀 16 帧（`events=0`），并装了 `temporal_index=order`；`memory=none` 对应 NM | `scripts/overnight_r12/serve.py` |
| 队列/GPU 分配、两个 Python 环境（服务端 `POLICY_PY`，仿真客户端 `SIM_PY`）| `scripts/overnight_r12/run_queue.py` |
| 无灯开关：`gym.make(..., no_lamp_wait=True)`；Count 指令会自动去掉 "Wait until the work light turns green, then " | `fruit_picking/task_v2_env.py:31-60` |
| S1 专家通过 gym.make 包装注入 `no_lamp_wait=True`，`task_revision=9`，`robot_uids='rm75b_eg2'`，种子 `4200000+16e+attempt` | `scripts/harvestmem_v3/run_episode.py:86-135` |
| **命名陷阱**：环境里 `u.attached[i]=True` 的意思是"已脱离枝条"；`u.hanging() = present & ~attached`；`u._grasped()` 返回当前被夹住的果子；`u.slot_of[actor]` 把补果后的新 actor 映射回原果位 | `fruit_picking/task_v2_env.py:124-134` |
| 专家选果规则：Count 在已投递数 ≥ N 或树上无果时停止，否则选挂果中世界坐标 `pos[1]` 最大者；Occluded 在初始为红的挂果中选 `pos[1]` 最大者 | `fruit_picking/task_v2_spec.py:110-134`、`scripts/fruit_r9_episode_support.py:166-180` |
| 子目标坐标：初始帧果位 3D 坐标（`u.original_geometry[slot]['pos']`）经前置相机内参/`extrinsic_cv` 投影到 256×256，再 `quantize_unique` 去重 | `scripts/harvestmem_v3/relabel_episode.py:110-118` |
| 训练标签切换时机：pick 覆盖 reach/approach/grasp/pull；carry 开始（目标果已脱枝）切到 place；篮中数量增加后切到下一个 pick；最后一次入篮后切到 return and hold | `scripts/harvestmem_v3/relabel_episode.py:15-58` |
| dev 专家回放（已带 `grounded_subgoal`），`result.json` 里有 `N` 和 `harvestmem_v3_slot_pixels`；GT 每帧有逐果 `pos/attached/present/wind_detached/in_basket`、`tcp_pos`、`gripper_qpos`、`n_in_basket` | `results/harvestmem_v3_S1/replays/dev_expert/{count,occluded_ripe}/ep2xx_attempt*/` |

## 2. 本轮新增规则

1. **禁止使用任何 RoboMME 评测入口**：`examples/robomme/eval.py`、`challenge_interface/`，以及 `VideoUnmask`、`VideoPlaceOrder` 等 RoboMME 任务。评测环境只允许 `FruitPickTaskV2-v0` + `rm75b_eg2` + `task_revision=9` + `no_lamp_wait=True`。
2. **每个 episode 开头做自检并写进 `result.json`**，任何一项失败即判为工程错误，不计入成功率：
   - env id、robot、`no_lamp_wait=True`、种子 = `4200000+16e`；
   - Count 指令里不含 "work light"；
   - 若 `dev_expert/<task>/ep<e>_attempt0/result.json` 存在且含 `harvestmem_v3_slot_pixels`，评测环境的 `N` 和逐果位像素坐标必须与其中的 `N`、`harvestmem_v3_slot_pixels` **完全相同**（这一项用来确认布局和投影与训练一致）；
   - 保存首帧前置图像 `first_front.png`。
3. **服务端启动时自检**：checkpoint 路径包含 `harvestmem_v3_S2_20261111`；加载到的 `norm_stats.json` 与 `data/harvestmem_v3_dataset_20261110/assets/robomme/norm_stats.json` 相同。每条 `actions.jsonl` 记录都写下这次 infer 用的 prompt。
4. **交付前自己看图**：至少看 2 个 episode 的首帧和中间帧，在报告里写明"画面是果树 + 果篮 + RM75"。
5. 新代码放 `scripts/harvestmem_v3/`，结果放 `results/harvestmem_v3_E*/`，不改旧代码。

## 3. 阶段任务

### E0 作废标记 + 补 S2 报告（不用 GPU，约 0.5 h）

1. 在 `runs/harvestmem_v3_eval_20261111/` 下新增 `INVALID.md`，写明第 0 节的原因。不删除、不改动原文件。
2. 写 `reports/harvestmem_v3_S2.md`：两个 checkpoint 路径、loss（首步 / 1000 步 / 末步）、训练耗时、命令。如实列出以下偏差：
   - SG **没有**做方针要求的 ±4 px 坐标噪声增强（`s2_train.py::install_grounded_prompt` 未实现）。本轮不重训：A 臂用的 oracle 坐标与训练坐标同源，不受影响；到 C 臂（感知源）前再处理。
   - NM 未训练。
   - checkpoint 落在 `acm_vla/vendor/.../runs/` 而不是 `ecm_vla/runs/`（`--run-dir` 相对于 `cwd=POLICY` 解析）。不移动，只记录。

### E1 Oracle 子目标 + 离线对齐（CPU，约半天）

实现 `scripts/harvestmem_v3/oracle.py`。它是一个因果状态机，只依赖每步快照（字段与 GT 行相同：逐果 `pos/attached/present/wind_detached/in_basket`、`tcp_pos`、`gripper_qpos`、`n_in_basket`、`timestep`）和回合常量（task、N、`slot_of`、各果位初始 ripe、各果位像素坐标）。闭环评测时由仿真实时构造快照，离线验证时直接读 GT 行，**两边用同一个函数**。

状态机：

1. `timestep < 30`：observe。
2. **PICK(target)**：按专家规则从挂果中选 target，子目标为 `pick_subgoal(task, slot_xy[slot_of[target]], ripe)`。target 一旦选定就保持不变，直到它脱枝，或因风落/消失而失效（失效时重新选）。
3. **PICK → PLACE**：target 已脱枝且不是风落（即被机器人拉下）。
4. **PLACE → 下一步**：`n_in_basket` 比进入 PLACE 时增加；满足停止条件则进入 RETURN，否则回到 PICK 并重新选果。
5. **PLACE → 掉果恢复**（训练数据里没有，仅评测用）：被拉下的果子不在篮中、离 TCP 超过 5 cm，且已静止 10 步，则回到 PICK 重新选果，并记一次 `drop` 事件。
6. **停止条件**：Count 为 `n_in_basket ≥ N`；Occluded 为不再有初始为红的挂果。RETURN 是吸收态。

**离线验证**：在 dev 200–249 两个任务的专家回放（每个 episode 取成功的那次 attempt）上逐帧运行，与 GT 的 `grounded_subgoal` 比较。

**门槛**
- 每帧一致率 ≥ 99%（两个任务分别统计）；
- 选错目标 0 次，计数错误 0 次；不一致只允许出现在切换点附近 ±5 帧以内的时机偏差。

**交付** `reports/harvestmem_v3_E1.md`：一致率按任务和 N 分层；不一致帧按"时机偏差 / 选错目标 / 计数错"分类；单元测试结果。这一步同时完成了 S3 的 `source=gt` 部分。

### E2 采摘评测器 + 开环检查（GPU 1–2 张，约 1–2 h）

1. 基于 `overnight_r12/eval_episode.py` 写 `scripts/harvestmem_v3/eval_episode.py`：
   - 环境：`no_lamp_wait=True`；预算 Count 2800、Occluded 1300；种子 `4200000+16e`；前 30 步只记历史；保留 `guard_v2`。
   - `--arm sg`：每次 infer 用 `compose_prompt(instruction, oracle 子目标)`；`--arm k0`：用 `instruction.lower()`。
   - 服务端直接用 `overnight_r12/serve.py`，参数为 `--checkpoint <上面的路径>`、`memory=uniform`。
   - 加上第 2 节的全部自检。
   - 输出：`result.json`、`trace.jsonl`、`actions.jsonl`（含 prompt）、`subgoal_timeline.jsonl`（只记切换点）、`first_front.png`、前置相机视频。
2. **开环检查**，用来确认服务端、归一化、prompt、记忆采样与训练一致：
   - 取 train clean 的 Count ep0、Occluded ep0，以及 dev 200、201 两个任务的专家回放；
   - 按闭环客户端完全相同的方式（30 步前缀、16 步一块、`add_buffer`）把记录的帧喂给服务端；SG 用记录的 `grounded_subgoal` 组 prompt；
   - 比较每块预测的 16 步关节动作与记录的专家动作，计算 7 个关节的平均绝对误差；基线是"原地不动"（16 步都等于当前关节角）的误差。
   - **门槛**：在 train 回放上，SG 和 K0 的"模型误差 / 原地不动误差"都 ≤ 0.3；dev 上的数值只报告。
   - **只报告，不设门槛**：SG 的子目标敏感性。把 pick 子目标的坐标换成同场景另一个果位，看预测动作的平均变化量，并与换成同一果位 ±4 px 时的变化量对比。
3. 冒烟：每个臂在 Count ep200 上跑一个 `--max-steps 200` 的闭环，人工检查首帧、prompt 日志和动作范围。

**交付** `reports/harvestmem_v3_E2.md`：开环误差表、敏感性数值、冒烟截图路径。

### E3 小规模闭环（8 卡全空闲时开始，机器时间）

| 臂 | 低层模型 | 子目标 | episode |
|---|---|---|---|
| A | SG | E1 的 oracle（仿真真值） | dev 200–209 × Count / Occluded |
| D | K0 | 无 | dev 200–209 × Count / Occluded |

共 40 个 episode，GPU 分配沿用 `run_queue.py`：每张卡最多 2 个仿真客户端；每个 episode 30 分钟超时；工程错误最多重试 2 次，单独统计。可续跑，结果自动汇总到 `results/harvestmem_v3_E3/SUMMARY.md`。

**每个 episode 自动做失败分类**（根据 trace 判断，可以同时命中多类）：

- 操作类：从未拉下任何果子；拉下后掉在篮外；卡住（未结束时连续 300 步 TCP 位移 < 1 cm）。
- 决策类：多摘（Count 中 `n_in_basket > N`）；少摘就停（`n < N` 时回到 rest 并停住）；凑够 N 后不停（凑够后直到预算耗尽也没有完成信号）；Occluded 摘了初始为绿的果子。
- A 臂额外统计**子目标跟随率**：每个采摘周期中，第一个被拉下的果子是否就是子目标指定的那个。
- 其他：超时，附上 `v2_reason`。

**判定规则（开跑前固定）**

- A 在任一任务 ≥ 4/10：低层基本可用。**预先批准**直接把 A、D 扩到 dev 200–239（每臂每任务 40 条），跑完再停下报告。
- A 在两个任务都 < 4/10：问题在低层执行。报告主要失败类别和子目标跟随率，每臂附 3 个典型失败视频，然后停下，不要改训练。
- 无论哪种情况都报告 A − D。注意 D 这次没有等灯问题，成功率可能高于 11-09 的 0/40，这是正常现象。

**交付** `results/harvestmem_v3_E3/SUMMARY.md` 和 `reports/harvestmem_v3_E3.md`：成功率、Count 按 N 分层（2–4 / 5–7 / 8–10）、失败分类表、子目标跟随率、平均步数、典型视频路径、判定结论。

### E-NM 与 E1 并行训练 NM（GPU 0–3，约 8.5 h，2026-11-11 补充批准）

- 必须使用新的绝对路径作为 run-dir：`/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111`。不能复用 `runs/harvestmem_v3_S2_20261111`，否则 `run_s2.py` 会覆盖 SG/K0 的 `formal_started.json`/`formal_finished.json`。用绝对路径后，checkpoint 会落在 `ecm_vla/runs/` 下。
- 冒烟沿用 11-11 09:35 那次（`runs/harvestmem_v3_S2_smoke_20261110/nm/`）：`s2_train.py`、`run_s2.py` 自 11-10 18:43 起没有改动。启动后盯日志，看到 Step 0 和 Step 50 的 loss、确认 checkpoint 目录在 `ecm_vla/runs/` 下以后再离开。
- 中途不可续跑：只在最后一步保存，且 `resume=False`、`overwrite=False`。崩溃就如实报告，不要往同一目录自动重启。
- 训完后在 `reports/harvestmem_v3_S2.md` 补一行 NM：loss（首步 / 1000 步 / 末步）、耗时、checkpoint 路径，并注明其位置与 SG/K0 不同。
- NM 训练期间：E1 只用 CPU；E2 只能用 GPU 4–7；E3 必须等 NM 训完、8 张卡全部空闲后再开始。

## 4. 本轮不做

- 不重训 SG（±4 px 增强留到 C 臂之前）；
- 不做感知版跟踪器；不跑 B、C、F 臂（NM 的闭环评测即 E 臂，也放到 E3 审阅之后）；
- 不用 val 150–199。

以上留到 E3 审阅之后再下发。
