# V-A 正式闭环准备记录（2026-10-23）

状态：正式主实验已封存，实际 **0/1260**，尚无正式成功率和联合 Go 结论。

运行目录：`results/V_A_formal_20261023`。3 主任务、val ID 0–19、模型 seed 7/8/9、7 条件，6 服务分布在配置的 GPU 0–3；官方 BF16 策略推理，512 token，32 帧，动作块执行 16 步。V-B 的 FP32 loss 探针不改变此推理协议。

计划 SHA256：`01bda980c322ce5fa72562ccc61eeb9809faacc7726efaf40bee7b55bca21646`。封存了 21 个项目源码文件的快照与哈希、183 个外部源码/val 元数据/小型归一化资产文件的哈希、6 个服务配置。checkpoint 使用既有 79999 路径，未重算全部权重 shard 哈希。已绑定 V-B 正式 protocol/analysis/REPORT 三份文件的 SHA。

新增 `scripts/complete_v_a.py`，顺序执行正式主实验、逐决策因果选帧重放审计、主对比分析、预注册追加 seed 10/11 一次、合并分析、§0A.6 联合决策。仅主轮有效且 C1–C4 存在未能判定时，对所涉及条件追加；追加条件及初轮报告在追加前封存。无工程错误自动重试；中断后从持久记录恢复，锁阻止重复控制器。

修复缺少决策日志仍能通过审计、空任务范围意外合并其他任务、退化判定缺少 95% CI 上界检查、smoke 被当正式结果、无效结果仍有正式路由等问题。补充 val 元数据与环境 seed 核验、结果记录一致性、输入帧哈希与源时间位置、预算和保护槽检查。成功场次记录完成步数；次要 logistic GLMM 尚未拟合，当前没有 R/lme4 或 statsmodels，报告明确披露，不作 GLMM 结论。

验证：完整 validation 套件 35 passed、1 skipped（真实 JAX 模型 CPU 测试在系统 Python 中无依赖而跳过）；随后针对 bootstrap 修正的 V-A 6 项测试再次通过，无警告。旧 smoke 7/7 场的新增选帧重放审计 0 失败；因 `formal=false` 不产生正式路由。静态资产、60 个 val 场景与主计划完整性检查通过。启动脚本通过 `bash -n`。

实际启动尝试在 runtime preflight 的 `socket.socket()` 被工具沙箱以 `PermissionError: Operation not permitted` 拒绝，尚未启动策略服务或消耗正式场次。当前工具环境也无法访问 NVIDIA 驱动。`pipeline_state.json` 留有这次启动错误，终端重启入口会恢复；不需要删除结果目录。

GPU 终端执行：

```bash
bash /media/fulei/fuleivla/ecm_vla/run_v_a_formal_20261023.sh
```

后台日志：`results/V_A_formal_20261023/pipeline.log`。主轮进度：同目录 `progress.json`；整体进度与追加路径：`pipeline_state.json`。完成后查看 `REPORT.md`、`analysis.json`、`selection_audit.json`、`JOINT_REPORT.md` 与 `joint_decision.json`。中断后运行同一脚本恢复；已有持久结果不重跑。
