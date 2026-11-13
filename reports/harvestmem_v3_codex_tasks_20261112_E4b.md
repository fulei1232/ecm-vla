# HarvestMem v3：E4b 任务单（给 Codex）

日期：2026-11-12。依据：E4a 原始结果（`results/harvestmem_v3_E4a/*/count/ep*/`），指挥者已逐条复算。

## 0. E4a 审阅结论（写进 E4b 报告开头）

E4a 通过。指挥者复算的成功数、触发数、工程门槛与 E4a 报告一致。另有三点 E4a 报告没写，E4b 报告需要引用：

1. **闭环近乎确定。** 同一回合加不加强制停止，停止前的结局几乎一样：
   - A 与 AR 的起步失败回合完全相同（12 个）；E 与 ER 完全相同（8 个）；D 的 33 个中，DR 有 31 个相同。
   - 不停止臂"凑够 N"的回合数，与强制停止臂的成功数基本相等：D 15 对 DR 16，E 22 对 ER 23，A 16 对 AR 16。
   - 也就是说，强制停止把凑够 N 的回合几乎全部变成了成功，没有带来新的失败。
   - 因此各臂比较一律按回合配对，提高精度靠增加回合数，不靠重复跑同一回合。
2. **强制停止之后剩下的失败（各 60 回合）：**

   | 臂 | 成功 | 起步失败 | 掉果 | 耗尽预算 | 起步成功的回合中的成功率 |
   |---|---:|---:|---:|---:|---:|
   | AR（SG） | 16 | 12 | 20 | 12 | 16/48 |
   | DR（K0） | 16 | 31 | 8 | 5 | 16/29 |
   | ER（NM） | 23 | 8 | 24 | 5 | 23/52 |

   - K0 的主要瓶颈仍是一个稀疏决策（起步）；NM 的主要瓶颈是执行（掉果）。
3. **SG 这条线到此结束。** 带子目标训练的 SG 不比 K0 好（AR 与 DR 都是 16/60），E4b 不再评测 SG。

## 1. 本轮目的

E4b 用专家规划器把夹爪送到跟踪器选中的果子前，再交给 VLA。要回答两个问题：

- **Count：** 由规划器完成观察段后的第一次出发，能否消除起步失败，把 K0 的成功率从 26.7% 提上去？
- **Occluded：** 每个目标都由规划器送到位，能否把"拉对果"的比例从 E3 的 35%–50% 提到 80% 以上？

本轮仍然不训练、不改任务、不改跟踪器。

## 2. 规则（沿用 E4a）

- 只写新目录；E4a 及更早的产物一律不改。
- `eval_episode.py` 只能用默认关闭的新开关扩展，开关关闭时行为与 E4a 相同。
- 训练与评测不共用 GPU；每个 GPU 通道同一时间最多 1 个客户端。
- 冒烟通过后再正式运行；后台运行、可续跑。
- 每个停点停下等审阅。

## 3. 实现

在 `eval_episode.py` 新增开关 `--planner-reach {off,first,all}`，默认 `off`。

### 3.1 规划器

- **构造**：与示范完全相同，`FruitPickSolver(env, step_fn, ik_mode="canonical_v31", ik_seed=0, ik_candidates=64)`，见 `scripts/harvestmem_v3/run_episode.py` 第 139 行。
- **调用链**：`canonical_v31` 的 `pick` 和 `move` 实际调用的是 `canonical_solver.py` 里的 `pick`、`move`。
  - 预抓取位姿按 `canonical_solver.pick` 第 127–129 行计算：用果子的**当前**位置，`PRE_GRASP = 0.12`。
  - 依次调用 `solver.refresh_obstacles(skip=(actor,))`、`solver.gripper = OPEN`、`solver.move(pre, {..., "phase": "reach", "target": actor})`。
  - 只执行 reach 段；接近、抓取、拉下、搬运、放果都交给 VLA。
- **注意事项**：
  - `canonical_v31_solver.plan` 会读 `solver.L.fruits[target]`。要确认补果后新出现的 actor 在 `L.fruits` 中有对应项，没有就停下汇报。
  - `canonical_pending_basket` 保持为 None。不调用 `pick`，所以 `check_pending_basket` 不应报错；一旦报错，说明调用方式不对，停下汇报。
- **`step_fn`**：
  - 接到 `Episode.step(action, guard=False)`，让轨迹、跟踪器、视频照常记录。
  - 动作写入 `actions.jsonl`，标记 `"source": "planner"`。
  - 同时用 `guard_v2` 计算这个动作是否会被裁剪，只记录，不实际裁剪。
  - 规划器运动期间的画面照常加入 VLA 的记忆缓冲区，用于下一次 `add_buffer`。

### 3.2 何时交给规划器

- **`first`**：只在观察段结束后、跟踪器第一次发出 `target_selected` 时交给规划器。之后全程由 VLA 执行。
- **`all`**：跟踪器每次发出 `target_selected` 都交给规划器。这包括观察段后的第一次、放果完成后选下一颗、掉果或风吹落后重新选目标、`reopen_after_no_target`。
  - 事件发生在 VLA 动作块中间时，立即中断动作块，与 E4a 的强制停止相同。
  - 规划器 reach 结束后，如果跟踪器的目标已经变了（例如 reach 途中目标被风吹落），且跟踪器仍处于 pick 状态，就为新目标再规划一次。连续最多 3 次，之后交回 VLA。
  - 交接时如果 `is_grasped_any` 为真（理论上不应出现），跳过规划器并记录 `planner_skipped_grasped`。
- **强制停止**：Count 臂全部同时打开 `--enforce-stop`，判定方式与 E4a 完全相同，并优先于规划器。
- **规划失败**：已定为交回 VLA 继续执行，记录 `planner_failure`（目标 slot、异常信息、时间步）。回合照常计入结果。

### 3.3 `result.json` 新增字段

- `planner_reach`（取值 off、first 或 all）
- `planner_calls`
- `planner_failures`
- `planner_replans`
- `planner_steps`（规划器执行的总步数）
- `planner_would_clip_steps`
- `handoffs`：每次交接一条记录，包含时间步、目标 slot、交接时的 TCP 位置、实际拉下的 slot

## 4. 冒烟

1. **规划器单独使用**：K0 加 `--planner-reach first --enforce-stop`，跑 count ep202、ep207（E4a 中这两个回合都是起步失败）。要求：
   - 第一次 reach 成功；
   - 交接后 VLA 有动作输出；
   - 视频中能看到机器人离开 rest，到达第一个目标果前。
2. **每个目标都交给规划器**：K0 加 `--planner-reach all`，跑 occluded ep200、ep201。要求：
   - 每次 `target_selected` 都对应一次规划器调用；
   - 截取每次交接时的前置相机画面，确认夹爪停在跟踪器目标的预抓取位置。
3. **开关关闭的确定性抽查**：
   - DR 配置（K0 + `--enforce-stop`，`--planner-reach off`）重跑 E4a 中的 count ep201、ep206、ep210、ep202、ep205。
   - 结局（outcome 和 reason）应与 E4a 相同。如果不同，在报告中列出，但不阻塞正式运行。

## 5. 正式运行

新建 `scripts/harvestmem_v3/run_e4b.py`，输出到 `results/harvestmem_v3_E4b/`，回合为 dev 200–259。

| 任务 | 臂 | checkpoint | 记忆 | 规划器 | 强制停止 |
|---|---|---|---|---|---|
| Count | DP1R | K0 | uniform | first | 是 |
| Count | DPaR | K0 | uniform | all | 是 |
| Count | EP1R | NM | none | first | 是 |
| Count | EPaR | NM | none | all | 是 |
| Occluded | D | K0 | uniform | off | — |
| Occluded | DPa | K0 | uniform | all | — |
| Occluded | E | NM | none | off | — |
| Occluded | EPa | NM | none | all | — |

- Count 的对照直接用 E4a 的 DR、ER，不重跑。
- Occluded 的对照 D、E 要在本轮代码下重跑 200–259，不复用 E3。
- 每个 checkpoint 的 4 个臂共用一组服务通道，队列按（回合，臂）交错。K0 用 GPU 0–3，NM 用 GPU 4–7。
- 共 480 回合，预计约 1 小时。

## 6. 开环检查（只报告，不设门槛）

在 dev 专家示范的每个 reach 结束点（专家 `phase` 从 reach 变为 approach 的那一帧），用现有 `open_loop.py` 计算 K0、NM 的"模型与保持不动的误差比"，与 E2 的整体误差比放在一起报告。

## 7. 预登记（正式运行前不得修改）

**工程门槛**（任一条不满足，只算工程问题，停下排查）：
- 规划失败次数不超过规划器调用次数的 5%；
- 所有 P 臂中，第一次规划器 reach 完成之前没有 `finish_signal`；
- Count 臂的强制停止门槛与 E4a 相同：overpick 为 0、触发时篮中正好 N 颗、300 步内以 `finish_signal` 结束。

**预测：**
1. DP1R 的 Count 成功率 ≥ 40%，对照为 DR 的 26.7%。依据：DR 起步成功的回合中有 16/29 成功。
2. EP1R 的 Count 成功率不低于 ER 减 5 个百分点（即不低于 33.3%）。NM 的起步失败本来就少，规划器出发主要是不应造成伤害。
3. DPa 和 EPa 在 Occluded 上的"拉对比例"（被拉下的果子中初始为熟果的比例）≥ 80%。对照为本轮重跑的 D、E。
4. DPa、EPa 的 Occluded 成功率高于各自配对对照。另报告是否达到 20%，作为任务层面是否有效的参考线。
5. Count 上 Pall 与 P1 的对比只做描述，不预测方向。

**判定：**
- 预测 1 和预测 3 都成立：起步、停止、选果三个任务层决策都已由跟踪器做出、结构方式执行。下一步做感知版跟踪器（C 臂），用感知替换真值跟踪器。
- 预测 1 不成立，但起步失败已降到 0：说明从规划器交接后，VLA 的执行变差了。比较 DP1R 和 DR 在起步成功回合中的掉果、卡死比例，汇报后停下。
- 预测 3 不成立：统计交接后 VLA 拉下的是不是目标果，以及没拉目标果时拉的是哪颗（离预抓取位置最近的果，还是别的果），汇报后停下。

## 8. 交付，完成后停下

1. `eval_episode.py` 的改动（附 diff）、`run_e4b.py`、冒烟结果。
2. `reports/harvestmem_v3_E4b.md`，包括：
   - 第 0 节的 E4a 审阅结论；
   - 8 个臂的成功率及 Wilson 95% 区间；
   - Count 按 N 分段（2–4、5–7、8–10）的成功率；
   - 用 `decision_metrics.py` 计算的决策层指标，以及决策失败与执行失败的分解；
   - 规划器统计：调用次数、失败次数、重新规划次数、会被 `guard_v2` 裁剪的步数；
   - 交接后目标一致率：交接后 VLA 拉下的果子是否就是交接时的目标；
   - 配对四格表：DP1R 对 DR、EP1R 对 ER、DPa 对 D、EPa 对 E；
   - 开环检查结果；
   - 视频：每个 P 臂 2 个成功回合、2 个失败回合，加上门槛中出现的任何异常回合。

## 9. 本轮不做

- 不训练。
- 不实现感知版跟踪器。
- 不评测 SG。
- 不用 val。
- 不改任务规则，不改跟踪器。
- 规划器卡住时不自动重新 reach，只记录"交接后卡死"。
