# ECM VLA / HarvestMem v3

用于通过私有 GitHub 仓库审阅当前代码、交换修改建议的源码快照。

## 从哪里开始读

1. [AI_REVIEW.md](AI_REVIEW.md)：当前任务、代码导航与建议输出格式。
2. [原 E4b 任务单](reports/harvestmem_v3_codex_tasks_20261112_E4b.md)：预登记预测和原始判定树。
3. [E4b 补遗](reports/harvestmem_v3_E4b_erratum.md)：旧 planner 运行失效的依据。
4. [r1 先导报告](reports/harvestmem_v3_E4b_r1_preflight.md)：观测修复、工程门槛、对照差异和授权决定。
5. [闭环评测](scripts/harvestmem_v3/eval_episode.py) 与 [r1 正式队列](scripts/harvestmem_v3/run_e4b_r1.py)。

## 目录

| 路径 | 内容 |
|---|---|
| `src/memory_management/` | 记忆、服务适配与 guard |
| `fruit_picking/` | 任务环境、布局、专家规划与机器人资源 |
| `scripts/harvestmem_v3/` | 训练、评测、分析、工程检查与队列 |
| `scripts/overnight_r12/serve.py` | K0/NM 推理服务入口 |
| `reports/` | 实验设计、预登记、审阅及历史报告 |
| `tests/` | 项目原有测试 |
| `external/robomme_policy_learning/` | 实际使用的外部策略库源码副本，保留其许可证 |
| `SOURCE_SNAPSHOT.json` | 来源提交、文件哈希与快照范围 |

## 范围与使用

这是代码审阅快照，不是独立训练/仿真环境。模型权重、数据集、逐回合 trace、视频、运行日志和虚拟环境保留在实验主机，不上传到本仓库。`reports/` 中保留文本和结构化历史报告。历史报告可能已被后续补遗更正，应以最新审阅为准。

代码中的绝对路径对应实验主机的原目录；`external/` 为阅读提供上下文，不会自动替换原来的安装路径。原始工作目录不因本快照的提交或推送而改变。

同步时 r1 队列正在主机执行；本仓库没有该轮最终结果。该状态只描述这次快照，后续请核对新提交和报告。
