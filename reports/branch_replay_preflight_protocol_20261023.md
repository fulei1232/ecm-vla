# 闭环 Teacher Oracle 第 1 步：仿真分支复现测试方案（2026-10-23）

状态：方案草案，尚未实现、尚未运行。只使用 val split 和 V-A 已有录制，V-A 结果目录只读。

## 1. 目的

闭环 Teacher Oracle 在每个决策点需要：从当前状态分出一个分支，让 planner 在分支里试走一段，拿到专家动作块，再回到主环境继续。本测试只回答一个问题：能不能构造这样的分支，并同时满足两点：

1. 分支起点与主环境的物理状态和任务状态逐位一致；
2. 分支里做任何事都不影响主环境。

本测试不评估 planner 动作是否合理（那是第 2 步），也不涉及策略推理和记忆选择。

## 2. 为什么不能直接用 `set_state_dict`

以下事实已从代码确认：

- ManiSkill 的 `BaseEnv.set_state_dict` 只调用 `scene.set_sim_state`，恢复的只有非静态 actor 的位姿和速度，以及 articulation 的根位姿、qpos、qvel。`get_state_dict` 虽然包含控制器状态，`set_state_dict` 却不会把它设回去。求解器内部状态（如接触缓存）也不在状态字典里。
- RoboMME 任务有大量 Python 层状态不在状态字典里。
  - 通用：`current_task_index`、`task_list`、`elapsed_steps`、`successflag` / `failureflag`、`generator` 等随机数状态。
  - BinFill：`red_cubes_in_bin` 等计数。
  - VideoUnmaskSwap：`swap_schedule`、`swap_times`、`pick_times`。
  - PatternLock：`selected_buttons`、`achieved_list`、`match`。
  - PickHighlight：`target_cube_pickup_counts`、`_highlight_cache`。
- `DemonstrationWrapper` 也有自己的状态：
  - `steps_without_demonstration`（截断计数）；
  - `current_subgoal_segment_filled`；
  - `_prev_ee_quat_wxyz` / `_prev_ee_rpy_xyz`；
  - `episode_success`。

  它在终止时还会额外执行一步；`_augment_obs_and_info` 直接读取 `obs['sensor_data']['base_camera']['rgb']`。

因此本测试比较三种分支方法。"进程内存档"成本最低，但最可能失败。

## 3. 三种分支方法

| 代号 | 做法 | 每次分支成本 | 主要风险 |
|---|---|---|---|
| S 进程内存档 | 在主环境上保存 state dict、控制器状态、任务和 wrapper 的 Python 属性深拷贝、numpy/torch/环境随机数状态；分支结束后恢复 | 最低 | 求解器内部状态和遗漏的属性无法恢复；分支直接改动主环境 |
| R 重置重放 | 新建同 seed 的环境实例，reset 后按已执行动作重放到分支点，在这个实例上做分支 | 随时间线性增长 | 不同环境实例或进程之间的确定性 |
| F 影子进程 fork | 另开一个影子环境，与主环境同步执行相同动作；分支时 `os.fork()`，子进程做分支，父进程不动 | 基本恒定 | 渲染上下文和线程在 fork 后可能不安全；影子环境若能关闭渲染，风险较小 |

R 和 F 的共同优点是主环境从不被分支触碰，只需保证分支起点与主环境一致。

## 4. 参照数据与测试 episode

参照数据是 V-A 正式运行的 D0a 录制（val split，checkpoint79999，模型 seed 7）。每个 episode 包含：

- `actions/decision_XXXX.npz`：服务端原样输出的 float64 动作块（20×8），每块执行前 16 步。已核对它与 `control_actions.jsonl` 逐位一致。
- `raw_frames.jsonl`：每帧 front/wrist 图像和 8 维状态的 sha256，以及 GT 快照。快照包括 `current_task_index`、`elapsed_steps`、物体位置、`tcp_position`、各任务计数器和 `info`。
- `result.json`：结局、执行步数、环境 seed。

测试 episode（每个任务一成一败）：

| 任务 | episode | 结局 | 执行步数 | demo 帧数 | 执行期子目标切换（control step） |
|---|---|---|---|---|---|
| BinFill | ep05 seed7 | success | 701 | 0 | 107, 170, 282, 332, 550, 601 |
| BinFill | ep02 seed7 | fail | 834 | 0 | 114, 191, 405, 462, 581, 641 |
| VideoUnmaskSwap | ep16 seed7 | success | 253 | 114 | 106, 138 |
| VideoUnmaskSwap | ep03 seed7 | fail | 276 | 216 | 106, 139 |

这样选择的理由：

- BinFill 子目标多、计数状态多，能覆盖较长的物理历史；
- VideoUnmaskSwap 的 reset 期间由 planner 生成视频 demo，并带有交换时间表，能覆盖 reset 路径和时间相关的状态；
- 失败 episode 覆盖失败判定路径。

分支点在运行前由脚本按规则确定并写入 `protocol.json`：

- F 测所有执行期决策点（每 16 个 control step 一个）。
- R 和 S 每个 episode 最多测 8 个：第一个执行期决策点、终止前最后一个决策点、子目标切换后的第一个决策点（最多 3 个），其余按时间均匀补齐。

## 5. 分支负载

分支里必须"做点事"，才能检验隔离性。第 1 步只用确定性负载，不调用 planner：

- **L0 空负载**：分支后什么都不做。
- **L1 扰动负载**：从分支点起执行 32 步录制动作，每步 7 个关节目标都加 +0.05 rad，夹爪不变。这会改变机械臂姿态，并可能碰到物体，足以同时扰动物理状态和任务状态。

调用 planner 的负载放到第 2 步。

## 6. 检查项与通过标准

"逐位一致"的定义：

- 图像和状态数组的 sha256 相等；
- GT 快照的 JSON 完全相等；
- sim state dict 的每个张量字节相等。

不使用数值容差。

### G0 参照重放（所有方法的前提）

在新进程中按录制配置建环境：`BenchmarkEnvBuilder`，`dataset="val"`，`action_space="joint_angle"`，`max_steps=1300`。然后走与 V-A 相同的 `EnvRunner.get_init_obs` / `EnvRunner.step` 路径重放录制动作。

- **G0a 物理**：每帧 8 维状态的 sha256、GT 快照与录制一致；终局结果和执行步数与 `result.json` 一致。
- **G0b 渲染**：每帧 front/wrist 图像的 sha256 与录制一致。
- **G0c 跨进程**：两个新进程各重放一次，每个决策点导出的 sim state dict 逐位一致。导出的状态作为后续检查的参照状态。

结果的处理事先约定如下：

- 若 G0b 失败而 G0a、G0c 通过，记为"物理可复现、渲染不可复现"。R 和 F 可以继续，因为 planner 只读物理状态，闭环中主环境也不会被重放。但不能再声称逐位复现了 V-A 录制。
- 若 G0a 或 G0c 失败，立即停止，不放宽标准。

### G1 R 方法

每个分支点 t：

1. 在同一进程中新建环境实例，reset，重放 [0, t) 的动作。此时 state dict、GT 快照、状态 sha256 必须与 G0 参照在 t 处逐位一致。
2. 执行 L1，记录每步的 state dict。
3. 再新建一个实例重复第 1、2 步，两次 L1 的结果必须逐位一致（分支确定性）。

另取 2 个分支点在新进程中重做第 1 步，确认"同进程新实例"和"新进程"的结果相同。

### G2 F 方法

- **F0 无渲染影子的可行性。** 把影子环境的 `obs_mode` 改为不渲染，检查 `DemonstrationWrapper` 能否正常运行。由于 `_augment_obs_and_info` 直接读 rgb，预计需要一个只在影子环境中生效的补丁，补丁不得改动物理或任务逻辑。
- 若无渲染影子不可行，影子环境保留渲染，但必须另外通过 fork 安全性检查：子进程能渲染、不崩溃、不挂起，父进程 fork 前后的渲染结果不变。
- 影子环境按录制动作同步执行。在每个决策点：
  1. 父进程的 state dict 和 GT 快照与 G0 参照一致；
  2. fork 出子进程，子进程先导出起点状态，必须与参照一致；
  3. 子进程执行 L0 或 L1，写出结果后退出；
  4. 父进程等子进程退出后再导出一次状态，必须与 fork 前逐位一致，然后继续执行。
- 同一分支点 fork 两次，两次 L1 的结果逐位一致。
- 影子进程固定 `OMP_NUM_THREADS=1` 等线程设置，避免 fork 后线程锁死。记录每次 fork 的耗时和内存。

### G3 S 方法

1. 在单个环境里重放录制动作。
2. 在每个分支点保存：state dict、控制器状态、任务和 wrapper 的 Python 属性深拷贝、numpy/torch/环境随机数状态。
3. 执行负载，然后恢复。
4. 继续重放录制动作直到结束。

恢复后直到结束的每一帧，状态、GT 快照（G0b 通过时还包括图像）以及终局，都必须与 G0 参照逐位一致。

先测 L0：只保存再恢复、不做任何负载也必须通过。否则说明恢复操作本身就会扰动物理。

事先说明：S 很可能因为求解器内部状态或遗漏的属性而失败。S 失败不阻塞第 2 步。

### G4 灵敏度对照（保证检查不是空过）

- 在某个分支点，把一个录制动作的一个关节加 1e-5 rad。G0 式的比较必须检测到之后的偏离，并报告第一个偏离步。
- S：故意只恢复 state dict，不恢复 Python 属性（`elapsed_steps` 每步都会变，所以必然可测）。检查必须判为失败。
- R 和 F：故意把分支起点放在 t−1（少重放一步）。起点一致性检查必须判为失败。

任何一项对照没有按预期触发，相应方法的通过结果都无效。

## 7. 判定与后续

**第 1 步通过的条件：**

- G0a 和 G0c 通过；
- R 或 F 至少有一种在全部 4 个 episode、全部分支点上通过；
- G4 全部按预期触发。

**方法选择：**

- 优先用 F，因为它的分支成本不随时间增长；
- F 不可行时用 R；
- S 只有在全部检查都通过时才考虑，闭环中仍优先 R 或 F，因为 S 会直接改动主环境。

**停止条件：** G0a 或 G0c 失败，或 R 和 F 都失败。此时不放宽标准，Teacher Oracle 改走以下两条备选之一：

- 按 GT 物体位姿重新搭建场景再规划。这种做法不保证逐位一致，只能作为近似专家，结论要相应降级。
- 放弃闭环 Teacher Oracle，改做重复闭环的结果对照。

**通过后的第 2 步**复用同一套分支工具。在分支里调用当前子目标的 planner（`task_list[current_task_index]["solve"]`），planner 绑定 `DemonstrationWrapper`，与 demo 生成路径一致。统计以下指标：

- 规划成功率和失败类型；
- RRT* 回退的比例；
- 同一分支点重复规划时，动作块是否一致；
- 需要单独确认 `use_demonstrationwrapper` 等标志位，确保分支里不会触发 `solve_pickup_fail` 这类故意失败的动作。

## 8. 实现与产物（计划）

代码：

- `src/memory_management/validation/sim_branch.py`：参照重放、状态采集与比较、S/R/F 三种分支方法。
- `scripts/run_branch_preflight.py`：按封存的 protocol 依次运行 G0–G4，支持中断后恢复。
- `tests/memory_management/validation/test_sim_branch.py`：状态比较、protocol 封存、灵敏度对照的单元测试。

结果目录 `results/branch_preflight_<日期>/`：

- `protocol.json`：episode、分支点、负载、判定规则和源码哈希，运行前封存；
- `checks.jsonl`：每项检查一行，包含首个不一致的字段和步数；
- `reference_states/`：G0 导出的各决策点 state dict；
- `summary.json` 和 `REPORT.md`。

环境信息一并记录：物理后端（预期为 CPU，`gpu_sim_enabled=False`，需实际核对）、GPU 型号、SAPIEN/ManiSkill/mplib 版本、线程相关的环境变量。

## 9. 成本估计

本测试不需要策略服务，也不需要 GPU 推理。作为参照，V-A 中 BinFill ep05 的 701 步闭环连同策略推理共用时 41 秒。

主要工作量：

- G0：每个 episode 重放 2 次；
- R：每个分支点重放 2 次，平均长度为半个 episode；
- S：每个分支点从恢复处重放到结束；
- F：每个 episode 同步执行 1 次，外加若干次 fork。

合计约 80 次"半个到一个 episode"的重放。单 CPU 进程预计 1 小时内完成，按 episode 并行可以更快。

## 10. 冒烟测试后的修订（正式运行前）

冒烟测试只用 VideoUnmaskSwap ep16 的 3 个分支点（t=0、112、240），共跑两次：`results/branch_preflight_smoke_20261023` 暴露了代码问题，修复后在 `results/branch_preflight_smoke_20261023b` 重跑。判定标准和参数都没有改，修订只在实现层面：

- S 的 Python 状态采集支持 ManiSkill 的 `BatchedRNG`。它的 `get_state` 会被逐环境批量化，真实状态在 `.rngs` 里。无法指纹化或深拷贝的属性会列在结果里，不再让任务崩溃。
- R 的 off-by-one 对照在比较帧时去掉帧号和步号。否则两边标签必然不同，对照就成了空过。
- 新增 F 的诊断任务 `f_l0`：在每个决策点 fork 一个什么都不做的子进程，用来区分"fork 本身破坏父进程"和"子进程执行仿真破坏渲染"。它只作诊断，不进入 `F_pass`。
- F0 不写补丁，只把影子环境的 `obs_mode` 改为 `state`，记录能否运行。补丁要改 wrapper 的图像读取路径，等 F 确实需要时再做。
- worker 写完结果后用 `os._exit(0)` 退出。渲染设备丢失后，析构会让进程异常终止，而这时结果已经落盘。

第二次冒烟的结果：

- G0a、G0b、G0c 通过：两次独立进程的重放与 V-A 录制逐帧逐位一致，包括图像。
- R 通过；扰动对照和 off-by-one 对照都按预期触发。
- F 失败。空操作 fork 16 次都正常，父进程不受影响。但子进程第一次 step 就拿不到图像（`front_rgb_list` 为 None），此后父进程的 step 也拿不到图像；第一次冒烟中，进程退出时报 `vk::Device::waitIdle: ErrorDeviceLost`。也就是说，fork 继承的渲染上下文不能在子进程里使用。F0 同样不可运行，`obs_mode="state"` 下报 `IndexError: too many indices for tensor of dimension 2`。
- S 失败，原因已定位到 ManiSkill 的状态接口：
  - 机器人底座是固定的，它的根位姿经 `set_root_pose` 写入再读出并不恒等。这是 float32 舍入误差：位置 x、y 分量分别偏 7e-11 和 1.5e-8 m，每往返一次误差翻倍。
  - qpos、qvel、其他物体、Python 属性和随机数都逐位恢复，但这点误差使恢复后的第一步就开始偏离。
  - 正常重放中，底座位姿在所有决策点上都不变。
  - 按第 7 节，S 失败不阻塞第 2 步，正式运行也不为 S 增加修正版本。
