# 修订 12 阶段报告：R12_check_failed

CPU 视线检查通过；R11 的 350 条 R0 与信息门槛按批准复用，不重跑。

- 观测修订：固定视觉假底位于 z=0.310 m，四边压入内衬至少 2 mm，不增加碰撞形状。
- DART 修订：重试固定 clean 配对布局和 N，仅改变 attempt 噪声；每回合随机选择两个采摘周期启用原强度 OU 噪声。
- R11 已有 109 条成功示范不混入 R12；训练关闭。

## 当前决策

R12 child failed exit1: /media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r12_20261109/commands/data_plan/console.log

## 阶段产物

- [CPU optical proof](cpu_optics/result.json)
- [R11 reuse proof](approved_r11_reuse.json)
- [GPU validation](gpu_validation.json)
- [attempt ledger](data_attempt_ledger.json)

完整机器可读指标见 `summary.json`。GPU 验证若通过，400 条 R12 示范将从全新输出目录生成；构建与共用归一化完成后停下，不训练。
