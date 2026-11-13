# Teacher Oracle 第 2b 步：按阶段续接专家的分支测试方案（2026-10-24）

状态：方案，运行前封存。只使用 val split 和 V-A 已有的 D0a 录制，V-A 结果目录只读。前提是第 1 步（`results/branch_preflight_20261023`）通过，分支方法用 R。本方案在看到任何第 2b 步分支结果之前写定。

## 1. 目的

第 2 步（`results/planner_preflight_20261023`）的结果：

- PatternLock 五项检查全部通过。
- VideoUnmaskSwap 一次走完整段子目标（T-full）时 100% 成功，但每 16 步重规划（T-rh）时 0% 成功。
- BinFill 的 T-full 是 85.2%，T-rh 是 0%。

T-rh 失败的原因是 RoboMME 的 `solve()` 都是开环的原语序列，每次调用都从子目标的第一个原语开始。中途重启时，它会重新张开夹爪 6 步，并退回预抓取位，所以一个 16 步的窗口完成不了抓取。BinFill 的 T-full 还有一类失败：原定方块已经被策略放进桶里，planner 仍在追它。T-full 卡住的 63 个分支中有 35 个属于这种情况。

本步测试一个"按阶段续接"的专家：路点与原版 solve 相同，但每执行完一个原语，就根据当前状态重新选择下一个原语。要回答的问题与第 2 步相同（P0–P4），重点是 P3：专家按自己的动作块每 16 步重规划时，能否完成任务。只有 P3 通过，动作块才能当闭环打分目标。

## 2. 专家定义

代码 `src/memory_management/validation/phase_expert.py`，入口 `PhaseSolver.solve(k)`，替换第 2 步的 `task_list[k]["solve"]`。planner 的构造、screw→RRT\* 重试、夹爪状态初始化、step 拦截都与第 2 步相同。

唯一的长度参数是通道半径 c = 0.02 m，约为 BinFill 方块的半边长。它在运行前写定，不按结果调整。"在通道内"指 TCP 与目标点的水平距离 ≤ c，并且高度在目标点高度 − c 与预抓取高度 + c 之间。

每个子目标的规则如下。路点的计算方式与原版 solve 逐行一致；每次判断都从当前物体位姿和 TCP 姿态重新计算。

**抓取**（BinFill "pick up the cube"，即原版 `solve_pickup`；VideoUnmaskSwap "pick up the container"，即原版 `solve_pickup_bin`）。按以下顺序判断，执行一个原语后重新判断：

1. 已抓住目标（`agent.is_grasping`）：夹爪指令已是闭合则抬起，抬起后 solve 结束；否则闭合。
2. 夹爪指令为闭合但没抓住：如果这是本次调用的第一个原语，且 TCP 与抓取位的距离 ≤ c，就继续闭合 6 步，这对应正在进行中的闭合。其他情况一律张开。
3. 夹爪张开且在下探通道内：移动到抓取位，然后闭合。
4. 其他情况：移动到预抓取位。

**放下**（BinFill "put it into the bin"，即原版 `solve_putonto_whenhold_binspecial`；VideoUnmaskSwap "put down the container"，即原版 `solve_putdown_whenhold`）：

- 夹爪指令为闭合：移动到释放位，张开，再移到撤离位。
- 已经张开：只移到撤离位。

**按钮**（BinFill "press the button"，即原版 `solve_button`）：

- 在按钮上方的通道内：先闭合（若尚未闭合），再按下、抬起。
- 不在通道内：移动到按钮上方的预备位。

其他规则：

- **BinFill 的换目标规则。** 抓取子目标原定的方块若已被移出场景（x > 5），改追同色、仍在桌上的方块中序号最小的一个。如果手里正抓着一个同色方块，就继续用它。
- **退化移动。** 如果 screw 规划因 TOPP 无法参数化路径而抛出 `RuntimeError: Fail to parameterize path`，说明 TCP 已在目标位姿，这段移动按"已完成、0 步"处理。第 2 步原版专家照旧抛出异常，本步计数报告。
- **原语上限。** 一次 solve 调用最多执行 20 个原语，超过就返回。返回后按第 2 步的规则计入"连续 3 次正常返回而子目标未变"的卡住判定。
- **规划失败。** 某段移动的 screw 和 RRT\* 都失败（exhausted）时，solve 立即返回。
- **未覆盖的子目标。** 不在上述列表里的子目标照旧调用原版 solve，并计数。在执行阶段，两个任务的子目标都在列表里。

## 3. 不变的部分

以下内容与第 2 步方案（`reports/planner_branch_preflight_protocol_20261023.md`）第 3–6 节完全相同：

- episode 选择和决策点：只测 BinFill 和 VideoUnmaskSwap，即第 2 步的 16 个 episode、339 个决策点。
- T-full 和 T-rh 两种模式，以及停止规则：步数预算 1300；400 步内子目标没变算卡住；连续 3 次正常返回而子目标没变也算卡住。
- 重复点规则：第一个、中间和最后一个决策点。
- P0–P4 的定义和阈值：P1、P2、P3 都要求 ≥ 90%，不放宽。

参数写在 `protocol.json` 的 `params` 中，`expert = "phase"`。

## 4. 判定

- 一个任务 P0–P4 全部通过，就可以把这个专家用作第 3 步的打分目标。
- P2 或 P3 不通过时，按失败类型分类报告：卡住、exhausted、异常、任务失败，以及卡住时所处的阶段。本次运行内不修改规则；如需修改，另立方案并使用新的输出目录。
- P4 不通过（不用 RRT\* 时也不确定）时，先找出非确定性的来源。

诊断项与第 2 步相同，另加阶段统计：每类阶段判断的次数、退化移动次数、换目标次数、回退原版 solve 的次数、触及原语上限的次数。报告并列第 2 步原版专家在同一批决策点上的结果，用于对比起步速度比、夹爪开合步数和方向余弦。

## 5. 冒烟测试与产物

正式运行前做冒烟测试：每个任务取第一个入选 episode 的前 3 个决策点，即 BinFill ep01 和 VideoUnmaskSwap ep05，重复规则与正式运行相同。冒烟只用来发现代码问题，结果不计入结论。冒烟后若改代码，改动记在本文件末尾，正式运行使用新的输出目录。

命令：

```bash
python3 scripts/run_planner_preflight.py all --expert phase --smoke --out results/phase_preflight_smoke_20261024
python3 scripts/run_planner_preflight.py all --expert phase --out results/phase_preflight_20261024
```

单元测试：`tests/memory_management/validation/test_phase_expert.py`，覆盖阶段规则、换目标规则、原语上限和回退；退化移动的处理在 `test_planner_branch.py` 中测试。

成本与第 2 步相近：16 个进程并行，约 20 分钟。

## 6. 冒烟测试记录（正式运行前）

冒烟结果在 `results/phase_preflight_smoke_20261024/`，不计入结论：

- 6 个决策点的分支起点全部逐位一致。
- 12 对重复分支逐位一致，其中 1 对用到了 RRT\*。
- T-full 和 T-rh 都是 6/6 成功。
- 没有退化移动、换目标、回退原版 solve 或触及原语上限的情况。

冒烟后没有修改代码。正式运行使用 `results/phase_preflight_20261024/`。
