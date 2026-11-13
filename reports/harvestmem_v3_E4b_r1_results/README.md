# E4b r1 已完成运行摘要

本页导出已完成队列的结果，供代码与实验审阅。360/360 回合完成；队列报告运行错误为0、工程门槛问题为0。当前停在审阅点，尚未独立复核全部工程统计或完成预登记分支分析。

| 臂 | 任务 | 成功 | 成功率 |
|---|---|---:|---:|
| DP1R | count | 23/60 | 38.3% |
| DPaR | count | 58/60 | 96.7% |
| EP1R | count | 34/60 | 56.7% |
| EPaR | count | 58/60 | 96.7% |
| DPa | occluded_ripe | 55/60 | 91.7% |
| EPa | occluded_ripe | 57/60 | 95.0% |

文件：

- `rows.json`：360 回合的 result 汇总，包括各次交接与初始熟度。
- `engineering_totals.json`：队列计算的工程门槛累计值。
- `formal_admission.json`：准入时的源码与证据指纹。
- `formal_status.json`：队列完成状态。
- `EXPORT_MANIFEST.json`：各导出文件原始路径和SHA256。

本目录不含逐步动作、trace、视频及权重。对照沿用E4a的DR/ER和原E4b的D/E；原E4b planner臂实验无效。跨服务输出差异及继续授权见同级的r1先导报告。以上成功率是描述统计，不能替代熟果比例、交接miss及原预登记预测的完整审阅。
