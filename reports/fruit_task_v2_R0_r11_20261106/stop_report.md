# 修订 11 阶段报告：计数 R0 通过，数据生成因可见性门槛停止

阶段：修订 11 数据生成（阻塞）。
做了什么：完成 dev250–259 时钟校准并冻结；在带 fixed_v2 遮挡板的计数环境完成 350 条 R0 和信息可得性审计；门槛通过后开始生成 400 条示范。

## R0 结果与门槛对照

- 校准：专家回放 10/10 成功；绝对时钟拟合命中 4/10，相对拟合 8/10，参数已冻结，哈希见 [clock_fit.json](clock_fit.json)。
- 主队列 350/350 完成，工程/有效性检查通过。`history` 50/50（要求至少 45/50），包络裁剪率 61/65,744 = 0.093%（要求 <0.5%）。`current`、`initial_current`、`initial_ground`、`clock` 分别 0/50、0/50、0/50、11/50（各自要求 ≤15/50）。
- 只报告条件：`clock_onset` 40/50；`clock_onset_k0` 40/50。
- 信息门槛通过：KA 总体 50/50，N=2–4 为 14/14、N=5–7 为 21/21、N=8–10 为 15/15（要求总体 ≥95%、分段 ≥90%）；K0 的 N=8–10 为 0/15（要求 ≤50%）；recent16 为 0/50（要求 ≤5%）。
- 完整队列指标和其余检查见 [summary.json](summary.json)、[information_audit.json](information_audit.json)、[other_checks.json](other_checks.json)。

## 数据生成中止点

目标为 400 条成功示范。停机前已接受 109 条（clean 100、DART 9），启动 173 次尝试（clean 101、DART 72），165 次有终局、8 次在途中止；这些尝试都属于 count 任务，occluded 任务尚未启动。clean/DART 失败与在途尝试均保留；未构建特征、未计算共用归一化、未训练。阶段耗时记录见 [data_phase_timing.json](data_phase_timing.json)。

硬门槛触发在 ep113 attempt3、第 1230 步：hand_camera 的 `basket_fruit` 为 127 像素，front_camera 为 0；两相机的 `inactive_reserve` 均为 0。§19.2 要求篮内果和备用果在双相机始终为 0 像素，因此立即停止全局流程。遮挡板交叠计数为 1/173 = 0.58%，未达到其独立停机规则（次数至少 3 且比例 >2%）；这不是本次停止原因。

没有重启、重试、换种子、调参或改变任务定义。R11 没有活动回放进程。修订 10 无罩 R0 仍仅作历史；D5/D6、历史产物和论文未改。按规则在此等待审阅。

## 失败证据

- [失败检查 JSON](../../results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3/check_failure.json)
- [像素坐标及对象清单](../../results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3/check_failure_pixel_coordinates.json)
- [手部相机截图](../../results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3/check_failure_hand_camera.png)
- [正面相机截图](../../results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3/check_failure_front_camera.png)
- [原始相机分割和深度](../../results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3/check_failure_raw_sensors.npz)
- [逐步轨迹](../../results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3/trace.jsonl)

## 决策与下一步

按 §19.2 硬可见性门槛立即停机；按 §19.3 不进入后续构建、归一化或训练。等待用户审阅并决定后续处理。
