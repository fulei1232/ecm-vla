阶段 1：实现与 CPU 验证完成，等待从有 GPU 权限的终端启动闭环；阶段未结束。
已确认：9 项 CPU 测试通过，包括执行真实官方循环 AST 的 16/8/4 步缓冲测试。
已确认：复用正式审计读取旧 K0 一条轨迹，82 个决策、1297 个已传入历史帧，0 项不一致。
待确认：新 8/4 步闭环各两条的记忆审计，以及 A0–A4 各 40 条实验，尚未运行。
决策：只有四条记忆冒烟全部通过才展开实验；完成后停在 G0，不自动跑 G1 或阶段 2。

## 做了什么

默认关闭的安全包络位于 `src/memory_management/fruit_execution.py`。在 `FruitEnvRunner.step` 送入仿真前，逐步读取实测关节角，取训练状态包络、q7 的 [−π,π] 包络和 measured±v_max 的交集，裁剪七个手臂目标，夹爪不变；交集为空则记工程错误。`FruitRawRecorder` 在开启包络或显式启用日志时同时写 proposed_action 与 action、逐关节裁剪标记，关闭开关时保持旧日志行为。

`--exec-horizon` 在客户端显式指定时覆盖官方 eval.py 向 get_action_chunk 传入的执行长度；未编辑 vendor 文件，官方循环、EpisodeState、pack_buffer 与 obs_horizon=16 保持原样。`diagnose_fruit_episode.py` 的原合并断言已处理，执行设置仅允许 closed 模式。原命令不加参数时，guard=off，执行长度沿用 obs_horizon，数值行为不变。

训练分位数来自 `scripts/compute_fruit_joint_guard.py`：只读取 clean train 0–99 与 DART train 100–199、两个任务共 400 条，状态数 173791。每个回合内相邻实测状态差取绝对值，逐关节 99.9 分位×1.5 得 v_max；状态上下界为 0.5%/99.5% 分位。没有读取开发集或验证集来定阈值。完整数值、manifest 及其哈希来源：`reports/fruit_joint_guard_v1_20261102.json`。

## 验证证据

- `results/fruit_exec_guard_dev_20261102/cpu_tests.log`：9 项测试通过，含夹爪保留、实测关节角限速、空交集/非法动作报错、q7±π、实际执行日志，以及原官方循环在 16/8/4 下逐帧写入与动作裁剪。
- `results/fruit_exec_guard_dev_20261102/archived_audit_preflight.json`：旧正式 K0 的 82 个决策审计 0 项不一致。只读原日志并在临时目录补充执行设置元数据，没有重跑旧验证集。首次临时审计漏链接距离日志，已纠正，未改旧文件。
- 已用 SIM_PYTHON 导入真实官方 evaluator 和新接口，obs_horizon 仍为 16；编排、报告和视频依赖导入成功。

## GPU 实验顺序

`run_fruit_exec_guard.py` 首先验证阶段 0 登记已冻结，再启动 GPU 0/1/2/3 上各一个现有 K0 服务，不超过每卡一个推理服务。先用首个服务跑 smoke_A2、smoke_A3，各在 FruitCount/FruitOccludedRipe ep200 跑一条，共四条；任何工程错误或记忆审计不一致立即停止，等待检查，不继续完整实验。

然后运行 A0(off,16)、A1(v1,16)、A2(v1,8)、A3(v1,4)、A4(off,8)。同一布局五个条件按 A0→A4 在同一个服务会话中依次运行；40 个布局分配到四个服务。种子均为 7，模型为 `runs/fruit_dart_kf_20261027/ckpt/mme_vla_suite/k0/11999`。观测/动作原始日志、每条记忆审计、服务日志与状态 JSON 均保留。显式续跑时完整配对布局保持原样，不完整布局全组移至 superseded 留档后重跑，不跨服务拼组。

实验全部完成后，`report_fruit_exec_guard.py` 统一调用 `analyze_fruit_failure_modes.py` 产生第 5 节全部指标，并比较 A0 与存档基线；最好条件优先满足两个 G0 门槛，再按首颗目标入篮、整任务成功、腕部偏离较少、A0–A4 固定顺序排序。附同一布局 A0/最好条件的视频抽帧。

## 下一步与停止点

从具备 GPU/Vulkan 权限的终端启动脚本。最终产物为 `results/fruit_exec_guard_dev_20261102/{report.md,summary.json}`；`status.json` 显示阶段进度。当前没有实验结果，不作任何 G0 判定。G0 结束必须按任务书汇报并等待用户确认。
