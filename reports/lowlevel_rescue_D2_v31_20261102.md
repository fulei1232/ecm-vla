阶段2 v3.1完成：六个核心项全部通过，已按§11.5选择v3.1并进入阶段3。
逐尝试成功率count=100/102、occluded_ripe=100/100；规划失败分别2/102、0/100。
规划确定性100/100，目标连续性100/100，方法一致率99/100；抓取q5/q6/q7各544/544一致。
83194个记录状态及指令q7均不越界；v3的24次失败中保存状态重规划成功23/24；v2/v3保留。
后续实际进度：400条新训练示范已生成，12000步K0训练和T1已完成；当前准备恢复单次G1。

## 做了什么

先完成全部24次v3失败碰撞诊断，再新增canonical_v31分支：路径中段0.018m，目标及TCP距目标≤0.05m用0.008m；目标果实沿用v3排除语义，不外扩。tuck的q2–q7固定为[-1,-.0061,2,-.181,.0534,-1.3933]，reach的q1取目标，carry取当前；顺序直线→tuck→high_ready→RRT。仅因目标在0.008m碰撞才更换IK，路径失败不会更换目标。carry仍直接rest，settle2、observe30、close10、open8及screw均不变。

RRT使用同一原始点云：端点可行时以0.018m搜索；端点需0.008m时，原生绑定无法提供随状态变化的碰撞半径，按§11.4.4用0.008m搜索后仅对中段外扩验证。全部方法共用分区验证器，RRT输出密集插值复核；每次调用保留半径和首个碰撞记录。

## 与v3对照

| 项目 | v3 | v3.1 | 目标 |
|---|---:|---:|---|
| ★规划确定性 | 100/100 | 100/100 | 100% |
| ★目标连续性 | 100/100 | 100/100 | ≥98% |
| ★方法一致率 | 97/100 | 99/100 | ≥95% |
| ★逐尝试成功 count | 100/118 | 100/102 | ≥90% |
| ★规划失败 count | 18/118 | 2/102 | ≤5% |
| ★逐尝试成功 occluded_ripe | 100/106 | 100/100 | ≥90% |
| ★规划失败 occluded_ripe | 6/106 | 0/100 | ≤5% |
| ★抓取q5一致性 | 540/540 | 544/544 | ≥95% |
| ★抓取q6一致性 | 540/540 | 544/544 | ≥95% |
| ★抓取q7一致性 | 540/540 | 544/544 | ≥95% |
| ★q7_states_outside_pi | 0/81730 | 0/83194 | 0 |
| ★q7_commands_outside_pi | 0/81730 | 0/83194 | 0 |
| reach q1多余行程P90 rad | 0.0 | 0.0 | 只报告 |
| carry q1多余行程P90 rad | 3.793369978666306e-05 | 0.001720595359802246 | 只报告 |
| v3回合长度中位/P10/P90 | [427.0, 302.0, 565.5] | — | 只报告 |
| v31回合长度中位/P10/P90 | [430.5, 302.0, 570.5999999999999] | — | 只报告 |

全部数值源：v3_extended_stats.json、v31_stats.json、planning/summary.json。选择理由：All revision3 core gates passed.。

## 3×2失败位置分布

Target attempt = one reach planning call, including rejected attempts; failure belongs to the target causing episode failure. left:y>.08, right:y<-.08, middle:closed interval; upper:z>=.54, lower:z<.54. Not all unvisited fruits in the layout.

| 任务/格子 | v3目标尝试 | v3失败 | v3.1目标尝试 | v3.1失败 |
|---|---:|---:|---:|---:|
| count/left/upper | 88 | 6 | 75 | 1 |
| count/left/lower | 85 | 4 | 77 | 0 |
| count/middle/upper | 66 | 4 | 62 | 0 |
| count/middle/lower | 54 | 4 | 53 | 1 |
| count/right/upper | 10 | 0 | 11 | 0 |
| count/right/lower | 16 | 0 | 16 | 0 |
| occluded_ripe/left/upper | 43 | 1 | 41 | 0 |
| occluded_ripe/left/lower | 44 | 2 | 41 | 0 |
| occluded_ripe/middle/upper | 43 | 0 | 43 | 0 |
| occluded_ripe/middle/lower | 55 | 1 | 53 | 0 |
| occluded_ripe/right/upper | 39 | 0 | 38 | 0 |
| occluded_ripe/right/lower | 39 | 2 | 38 | 0 |

每格失败归于导致整回合失败的目标果实；未访问果实不虚增尝试数。失败原因、种子与初始xyz均见failure_coverage.json。换种子使两版成功布局不完全配对。

## 运动与静止监督

方法/RRT计数（所有尝试）：{'v3': {'segments': 1144, 'rrt_used': 26, 'method_detour': 98, 'method_straight': 1020, 'method_rrt': 26}, 'v31': {'segments': 1095, 'rrt_used': 2, 'method_detour': 122, 'method_straight': 971, 'method_rrt': 2}}。

| 阶段 | v3命中/监督样本 | v3.1命中/监督样本 |
|---|---:|---:|
| observe | 1000/6000 | 1000/6000 |
| grasp | 2693/5400 | 2715/5440 |
| release | 1700/4320 | 1720/4352 |
| settle | 0/1080 | 0/1088 |
| done | 0/1000 | 0/1000 |
| other | 0/63930 | 0/65314 |

此处审计示范的observe仍保留原监督标记以便对照；阶段3新生成的全部400条示范才将前30帧设为历史，不删除任何帧。

## 碰撞诊断与24次失败重规划

完整诊断：[report.md](../results/fruit_expert_v31_audit_20261102/v3_failure_diagnosis/report.md)，目标两余量碰撞数{'0.008': 0, '0.018': 4}。原生RRT入口未发现把点云半径自动改回0.008m的代码。

v3.1从保存失败状态单次重规划成功23/24，逐条结果在v3_failure_replanning/summary.json。这不是24个完整物理回合的成功率。

## 示范抽帧


![示范](../results/fruit_expert_v31_audit_20261102/videos/count_ep0.png)
源：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/demos/rm75b_eg2/train/count/ep0_manifest.json；视频：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/videos/count_ep0.mp4。

![示范](../results/fruit_expert_v31_audit_20261102/videos/count_ep1.png)
源：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/demos/rm75b_eg2/train/count/ep1_manifest.json；视频：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/videos/count_ep1.mp4。

![示范](../results/fruit_expert_v31_audit_20261102/videos/occluded_ripe_ep0.png)
源：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/demos/rm75b_eg2/train/occluded_ripe/ep0_manifest.json；视频：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/videos/occluded_ripe_ep0.mp4。

![示范](../results/fruit_expert_v31_audit_20261102/videos/occluded_ripe_ep1.png)
源：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/demos/rm75b_eg2/train/occluded_ripe/ep1_manifest.json；视频：/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v31_audit_20261102/videos/occluded_ripe_ep1.mp4。

## 决策与下一步

按§11.5选择canonical_v3.1_20261102，不停下等待确认，自动执行§11.6阶段3。仅两版均因代码错误等无法使用时才停止；不再迭代专家。

阶段3数据、归一化统计、8卡布局是相对旧K0训练的三项允许差别。observe前30帧只作历史，评测A0+30步固定保持（计入1300预算），包络v2仅作不阻塞对照。

## 独立复核与后续实际进度

复核200个manifest、400个数据内容哈希、83194个q7记录、100组规划重复及100组连续性；v3的429份元数据、v2的236份元数据及v3源文件快照均保持原摘要。4段视频及抽帧已核对。唯一工作流源文件变更是阶段3分析器对物理保持前缀的读取修复，不改变专家、模型、控制、原始回放或指标定义。证据在independent_validation.json及阶段3physical_prefix_fix/validation.json。

阶段3已完成400条干净/DART新示范，共164639个物理帧，前缀12000帧保留为历史，监督样本152639。归一化统计重新计算，检查点携带统计与数据一致。T1的240个损失区间齐全且有限，末区间0.0029。训练规格只含获准的数据、统计和GPU布局差别；训练包装器较旧封存版增加了可选save_interval字段，本次未设置该字段，有效保存间隔仍为12000，不构成新增训练设置差别。
