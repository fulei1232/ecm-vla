# 给网页版 AI 的审阅入口

请先读取本文件、README 和 `SOURCE_SNAPSHOT.json`，以当前提交为审阅基准。

## 当前实验

HarvestMem v3 E4b r1：只重跑 DP1R、DPaR、EP1R、EPaR、DPa、EPa 六个 planner 臂，dev200–259，共360回合；复用原 DR/ER/D/E 对照。原 E4b planner 臂因交接后使用过时观测而失效，不能把旧数据当作有效 reach 实验的负结果。

r1 修复后推理使用 planner 结束时观测，planner 帧保留在待发送历史中；guard 字典按真实标志记录，交接记录为 reach 结束状态。新门槛为动作跳变不超过0.3rad的比例至少95%、首次交接后第8步距目标不超过0.20m的比例至少90%。

中途比例停止按用户确认的60回合失败数限额执行：planner分母至少60，跳变分母至少60，距离分母60；每臂满60回合后按实际事件分母最终判定。Count停止违规、首次reach完成前出现finish_signal仍立即停止。详见 `engineering()`。

DR ep210 在不同服务进程有结局差异。固定输入探针确认观测、RNG和历史输入相同时动作仍可能跨进程不同，但未定位到具体计算环节。用户已同意将off抽查差异列为非阻塞诊断；其他门槛、原预测1–5和判定树保持不变。

## 建议的阅读顺序

- `scripts/harvestmem_v3/eval_episode.py`：观测/状态同步、历史缓冲、动作日志、planner调度、强制停止。
- `scripts/harvestmem_v3/run_e4b_r1.py`：八卡任务分配、代码指纹、累计门槛、恢复逻辑。
- `scripts/harvestmem_v3/e4b_r1_gates.py`：交接归属和距离采样。
- `scripts/harvestmem_v3/oracle.py`、`src/memory_management/fruit_rescue_platform.py`：目标选择和guard语义。
- `scripts/overnight_r12/serve.py`、`src/memory_management/policy_adapter.py`：服务与记忆配置。
- `external/robomme_policy_learning/src/mme_vla_suite/policies/policy.py`：采样RNG与add_buffer。
- `external/robomme_policy_learning/src/mme_vla_suite/models/integration/history_pi0.py`：模型推理。

## 修改建议请这样写

1. 给出当前提交、准确文件路径、相关函数和具体触发条件。
2. 区分已由代码/现有产物确认的事实与待验证的假设。
3. 说明结果影响，给出最小补丁或明确修改步骤。
4. 列出针对性验证方法，指出是否需要CPU、GPU或额外实验。
5. 若改变预登记、工程门槛、样本数、对照或任务语义，单独说明并等待用户决定。

评测运行期间，实验主机逐回合校验SOURCES文件哈希。请将建议作为待执行补丁，不要求正在运行时直接保存这些文件；由本地主机代理确认停点后应用。不要把旧planner成功率当作有效基线，也不要编造本仓库未包含的逐回合结果。

可复制给 AI 的请求：

> 请先读取 AI_REVIEW.md 和 r1 先导报告，再审阅 eval_episode.py、run_e4b_r1.py、e4b_r1_gates.py。以当前提交为准，输出按严重性排序的问题，每项附路径、函数、触发条件、证据、最小修改建议和验证命令。涉及预登记或门槛变更请单列，暂不直接修改实验主机。
