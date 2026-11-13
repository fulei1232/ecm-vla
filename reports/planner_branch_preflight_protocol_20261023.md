# 闭环 Teacher Oracle 第 2 步：分支内 planner 可用性测试方案（2026-10-23）

状态：方案，运行前封存。只使用 val split 和 V-A 已有的 D0a 录制，V-A 结果目录只读。前提是第 1 步已通过（`results/branch_preflight_20261023`），分支方法用 R：新建环境实例，reset 后把录制动作重放到分支点。

## 1. 目的

Teacher Oracle 在每个决策点从当前状态调用当前子目标的 planner，取它的前 20 步动作作为专家动作块（20 是模型的 action horizon），再用这个动作块给候选记忆打分。本测试回答四个问题：

1. planner 在闭环状态上能否给出动作块（可用性）；
2. 从这些状态交给 planner，任务能否完成（专家是否胜任）；
3. 同一状态重复规划，结果是否逐位相同（确定性）；
4. 每个决策点都重新规划时，这些动作块连起来是否仍是一个能完成任务的专家（作为打分目标是否自洽）。

另外，失败 episode 里哪些决策点还能被 planner 救回来，决定了 Teacher Oracle 最多能在哪里起作用，这一项作为诊断报告。

本测试不调用策略模型，不计算 flow-matching loss，也不做记忆选择。

## 2. 已确认的代码事实和探针结果

以下事实来自代码阅读，或来自一次未封存的接口探针（BinFill ep05 t=112，VideoUnmaskSwap ep16 t=16，PatternLock ep00 t=16）：

- **调用路径。** `DemonstrationWrapper.get_demonstration_trajectory` 生成 demo 的方式是：把 `FailAwarePandaArmMotionPlanningSolver`（PatternLock 用 stick 版，`joint_vel_limits=0.3`）绑定在 `DemonstrationWrapper` 上，调用 `task_list[i]["solve"](wrapper, planner)`。
  - screw 规划失败时回退到 RRT*：screw 尝试 `_demo_screw_max_attempts=1` 次，RRT* 尝试 `_demo_rrt_max_attempts=3` 次。两者都失败时，这一段运动返回 -1，`solve` 跳过它继续往下走。
  - planner 的每一步都走 `DemonstrationWrapper.step`，与闭环执行路径相同，子目标判定、截断计数和终止时的额外一步都照常发生。
  - 闭环外层的 `FailAwareWrapper` 不在这条路径上，step 中的异常会直接抛出。
- **故意失败的抓取不会触发。** 闭环中 `use_demonstrationwrapper=True`，`solve_pickup` / `solve_pickup_bin` 里的 `solve_pickup_fail` 分支不会执行；val 环境的 `robomme_failure_recovery` 为 False。
- **子目标随时切换。** `use_demonstrationwrapper=True` 且 `demonstration_record_traj=False` 时，每一步都允许切换子目标。
- **每个 `solve` 都从子目标起点开始。** 例如 `solve_pickup` 先 `open_gripper`（原地 6 步），再到 reach 位姿（z=0.15），再下降抓取。在子目标中途调用时，专家会先停 6 步或先回到 reach 位姿。探针中 VideoUnmaskSwap t=16 的专家前 6 步关节完全不动。
- **每次规划都从零速度起步。** screw 和 RRT* 的轨迹都从当前 qpos、速度为零开始做时间参数化。探针中 BinFill t=112 专家前几步的关节位移依次是 0.0016、0.0047、0.0079 rad……，而此刻策略的速度约 0.02 rad/步。所以中途的专家动作块开头相当于"先刹车再加速"。
- **夹爪状态必须初始化。** planner 构造时 `gripper_state=OPEN`。生成 demo 时同一个 planner 贯穿所有子目标，所以没问题；分支里新建的 planner 却不知道机械臂正夹着东西。探针 BinFill t=112（夹着方块，子目标"放进 bin"）：
  - 默认 OPEN 时，专家动作块的夹爪维度全是 +1，60 步后子目标仍未完成；
  - 按控制器目标推断为 CLOSED 后，60 步内方块放进 bin，子目标推进。
- **RRT\* 不是确定性的。** 它按 1 秒的规划时间截止；screw 规划是确定性的。
- **BinFill 的 planner 目标是固定的方块。** 拾取子目标的完成条件是"抓起任意方块"，但 `solve_pickup` 的目标是 `cube_collection[i]` 这个具体方块。放进 bin 的方块会被移到 (10, 10, 0)。如果策略之前抓的不是这一个，planner 的目标可能已经不在场景里。
- **动作格式。** planner 输出 8 维动作（7 个关节目标 + 夹爪 ±1），PatternLock 是 7 维。策略输出的夹爪值是连续的，大约在 ±1 附近。

## 3. 分支与 planner 构造（预先固定）

每个分支：

1. 用第 1 步的 R 方法新建环境实例，reset 后重放录制动作到决策点 t。
2. 检查分支起点（P0）。
3. 按第 2 节的方式构造 planner，参数和 screw→RRT* 重试与 `get_demonstration_trajectory` 完全相同。
4. 按第 4 节的模式运行专家，直到停止。

夹爪初始化规则：每次（重新）规划开始时，夹爪控制器的目标 `_target_qpos` 若大于 `[lower, upper]` 的中点，就设为 OPEN，否则设为 CLOSED。只用于机械臂任务。同时记录它与"上一步执行动作的夹爪符号"是否一致，作为交叉核对。

分支里拦截 `DemonstrationWrapper.step`，每一步记录：

- 动作；
- 状态（success / fail / timeout / ongoing）；
- 执行后的 `current_task_index`；
- 这一步的来源：夹爪开合、screw、RRT\*，或 `solve` 直接调用 step。

retry 包装另外统计 screw 失败、RRT\* 尝试与失败，以及"两者都失败"（记为 exhausted）发生在第几步。截断和重规划用继承 `BaseException` 的控制信号实现，避免被 retry 里的 `except Exception` 吞掉；每一步执行前和每次 `solve` 返回后都检查待处理的信号。

## 4. 两种专家模式

每个决策点开两个独立的 R 分支，分别运行：

- **T-full（一次规划，跑到底）。** 用同一个 planner 对当前子目标调用 `solve`。`solve` 返回后，无论子目标是否推进，都对当时的当前子目标再调用一次 `solve`，直到停止。这对应"planner 作为整段子目标的专家"。
- **T-rh（每 16 步重新规划）。** 与 T-full 相同，但每执行 16 步就中断当前 `solve`，按夹爪规则重新初始化，再对当时的当前子目标从头调用 `solve`。这对应 Teacher Oracle 的实际用法：每个决策点都从当前状态重新规划。不同的是，这里由专家自己执行它的前 16 步。

专家动作块定义为 T-full 的前 20 步。只有在 episode 于 20 步内终止时，才允许更短的动作块。

停止规则（两种模式相同）：

- episode 终止或截断，结局取这一步的状态；
- 执行步数累计（t 加专家步数）超过 1300，结局为 timeout。这与 `eval.py` 的判定相同，都在终止判定之前检查；
- 连续 400 步子目标没有推进，结局为 stuck；
- 连续 3 次 `solve` 正常返回而子目标没有推进，结局为 stuck（T-rh 中被重规划中断的调用不计入）；
- 其他异常，结局为 exception，并记录异常信息。

## 5. episode 与决策点

参照数据是 V-A 正式运行的 D0a 录制，也就是 Teacher Oracle 的基线条件 B0：均匀采样记忆，模型 seed 7。每个任务按 episode_id 升序，取前 4 个成功和前 4 个失败的 episode：

| 任务 | 成功 episode | 失败 episode | 决策点数 |
|---|---|---|---|
| BinFill | 01, 04, 05, 08 | 00, 02, 03, 06 | 270 |
| VideoUnmaskSwap | 05, 12, 13, 14 | 00, 01, 02, 03 | 69 |
| PatternLock | 00, 01, 02, 04 | 03, 07, 10, 11 | 46 |

- 测每个 episode 的全部执行期决策点 t = 0, 16, 32, …，共 385 个。
- PatternLock 不在第 1 步的测试范围内，它的 R 方法有效性由本测试的 P0 逐点检查。
- 重复性测试：每个 episode 取第一个、中间一个和最后一个决策点，在另一个进程里重做 T-full 和 T-rh。

## 6. 检查项与通过标准

逐位一致的定义同第 1 步，不使用数值容差。下面的阈值都按任务分别计算，运行前固定。

- **P0 分支起点。** 每个分支重放到 t 时，录制的帧数等于 `exec_start + t + 1`，且每一帧的物理和渲染字段都与参照逐位一致。任何一个分支不满足，该任务的结果无效。
- **P1 动作块可用性。** 至少 90% 的决策点能拿到动作块：
  - T-full 前 20 步内没有异常，也没有发生 exhausted；
  - 拿满 20 步，或 episode 在 20 步内终止。
- **P2 T-full 胜任。** 在策略成功的 episode 上，至少 90% 的决策点 T-full 的结局是 success。理由：这些状态策略自己都能走通，一个合格的专家也应该走得通。
- **P3 T-rh 胜任。** 在策略成功的 episode 上，至少 90% 的决策点 T-rh 的结局是 success。
- **P4 确定性。** 重复点上，两次运行都没用到 RRT\* 时，T-full 和 T-rh 的动作序列、结局与步数必须逐位相同。用到 RRT\* 的重复点单独报告。

## 7. 诊断项（不设门槛）

- **可挽回性。** 在失败 episode 上，按决策点报告 T-full 和 T-rh 的结局、可挽回的决策点比例，以及最后一个可挽回的决策点在 episode 中的位置。
- **动作块与策略动作块的对比。** 策略动作块是该决策点的 `decision_XXXX.npz`（20×8）。只比较 7 个关节：
  - 专家与策略的逐步 L2 均值；
  - 原地不动基线：当前 qpos 重复 20 次，与策略的逐步 L2 均值；
  - 第 20 步净位移（相对当前 qpos）的方向余弦；
  - 起步速度比：专家前 4 步平均每步位移 ÷ 策略前 4 步的；
  - 动作块中夹爪开合步（原地不动）的步数；
  - 夹爪符号一致的比例（机械臂任务）。

  以上按"策略成功 / 失败"和"子目标切换后的第一个决策点 / 子目标中途"分组报告。"切换后的第一个决策点"指参照中 t−16 到 t 之间 `current_task_index` 变过，t=0 也算。
- **planner 统计。** RRT\* 使用率、exhausted 次数、单次 `solve` 用时。
- **BinFill 目标有效性。** 决策点上当前子目标的 `segment` 物体是否已被移出场景（x > 5）。
- **夹爪初始化交叉核对。** 记录与上一步动作符号不一致的次数。

## 8. 判定与后续

一个任务满足 P0–P4 全部通过，才进入第 3 步（小规模 Teacher Oracle 试跑）。

- **P2 通过但 P3 不通过：** planner 能当整段子目标的专家，但每个决策点重规划得到的动作块连不起来，原因是从子目标起点和零速度重新开始。这时动作块不能直接当打分目标，需要设计"按阶段续接"的专家（例如跳过已完成的路点），作为第 2b 步另立方案测试，不在本次运行里打补丁。
- **P1 或 P2 不通过：** 按失败类型（目标无效、规划 exhausted、异常、stuck、任务失败）分类，再决定是改适配还是放弃这个任务。
- **只有 P4 不通过（仅 screw 时也不确定）：** 第 3 步前必须先找到非确定性的来源。
- **诊断项用于第 3 步的打分设计。** 例如起步速度比远小于 1 时，flow-matching 的目标在开头就是"刹车"；若改为只比较后段，必须在第 3 步方案里事先写明。

## 9. 冒烟测试

正式运行前，每个任务取第一个入选 episode 的前 3 个决策点（BinFill ep01、VideoUnmaskSwap ep05、PatternLock ep00）。重复点规则与正式运行相同，即第一个、中间一个和最后一个，所以这 3 个点都会重复。冒烟只用来发现代码问题，结果不计入结论。如果冒烟后要改代码，改动记在本文件末尾，正式运行用新的输出目录。

## 10. 实现与产物

代码：

- `src/memory_management/validation/planner_branch.py`：planner 构造、step 拦截、两种专家模式、动作块指标、判定；
- `scripts/run_planner_preflight.py`：封存、按 episode 分片并行运行、汇总，支持中断后恢复；
- `tests/memory_management/validation/test_planner_branch.py`：不依赖仿真的单元测试，包括停止规则、夹爪规则、指标、判定和 episode 选择。

结果目录 `results/planner_preflight_20261023/`：

- `protocol.json`：episode、决策点、参数、源码和参照文件的哈希，运行前封存；
- `workers/<episode>/<job>.json`：每个决策点的两种专家记录，包括动作块和策略动作块；
- `checks.jsonl`、`summary.json`、`REPORT.md`。

## 11. 成本估计

按探针的计时，每个分支的建环境、reset 和重放约 7 秒，专家运行按剩余步数约每秒 100 步，每次规划约 0.1–0.6 秒。385 个决策点 × 2 个分支，再加 72 个重复点（144 个分支），单进程约 2.5 小时。按每片不超过 6 个决策点分片，用 16 个进程并行，预计 20 分钟左右。并行时不改线程相关的环境变量，与 V-A 和第 1 步保持一致，以免影响逐位复现。

## 12. 冒烟测试后的修订（正式运行前）

冒烟结果在 `results/planner_preflight_smoke_20261023/`，16 个进程并行，共 1 分钟。结果不计入结论，这里只记录与代码正确性有关的观察：

- 9 个决策点的分支起点全部逐位一致；18 对重复分支逐位一致，其中 1 对用到了 RRT\*；夹爪状态推断与上一步动作符号 0 次不一致。
- T-full 在 9 个点上全部成功。
- BinFill 和 VideoUnmaskSwap 的 T-rh 在 6 个点上全部失败：5 个点 stuck（400 步内子目标没变），1 个点抛出异常。原因是每次重规划都从子目标开头执行，先张开夹爪 6 步，再从零速度移动。一个 16 步的窗口完成不了"接近—下探—闭合—抬起"，下一次重规划又把夹爪张开。BinFill 在 t=0 抛出的 `RuntimeError: Fail to parameterize path` 来自 mplib 的 TOPP：重规划时机械臂已在目标位姿附近，screw 路径退化。原版 `DemonstrationWrapper` 的 screw 重试和任务级包装都只捕获 `ScrewPlanFailure`，所以原版同样会抛出这个异常，插桩没有改变行为。这正是第 4 节设计 T-rh 时要检测的现象。按第 8 节，本次不修补，留到第 2b 步。
- PatternLock 的 `solve` 只有一段 screw 移动，T-rh 在 3 个点上全部成功。

冒烟后的唯一代码改动：`run_planner_preflight.py` 的报告里，没有样本的结局分组原来打印为空，改为"无样本"。判定逻辑和参数没有改动。正式运行使用新目录 `results/planner_preflight_20261023/`，封存时重新计算源码哈希。
