阶段 2：审计不达标，停下等用户确认。
已确认：200条被接受示范来自217次尝试；逐尝试成功率 count=100/111、occluded_ripe=100/106，均未达95%。
已确认：确定性仅7/10布局通过，3个布局轨迹长度不一致；抓取q7偏离为279/547，q7越界状态286/91349。
已通过：reach/carry的q1摆幅P90为0.766/0.690rad，q7摆幅>π均为0/547；修订1补充项已交付。
决策：按§4.2停下等用户确认，不进入阶段3；平台保持A0，v1作废。

## 做了什么

新增 opt-in ik_mode=nearest，默认仍为 legacy。调用本机 mplib.IK，以局部确定性采样代替原生随机配置提供器；将 q7 选为距当前最近且在限位内的2π等价值，拒绝 |q7|>π，再按加权关节距离排序。权重为 [64,24,4,2,1,1,1]，基座/肩部较高；按顺序每次只向一个目标规划，优先碰撞检查通过的关节直线并使用原TOPP时间参数化，失败才退回RRT。每段记录 straight/rrt/screw。approach/pull 的 plan_screw 和所有保持步数不变。

所有新示范位于本结果目录 demos，固定布局重复位于 determinism；原数据、原D1结果、封存划分不覆盖。随机源与限制见 audit.json/planning_note。基础关节权重通过开发中的CPU运动学探针确定，未使用开发门槛集或验证集。

## 审计数值与目标对照

| 审计项 | 旧专家 | 新专家 | 目标/判定 |
|---|---:|---:|---|
| 专家成功率 count（含重试失败） | 100/104 | 100/111 | ≥95%；False |
| 已接受 episode count | 100/100 | 100/100 | 不替代逐尝试成功率 |
| 专家成功率 occluded_ripe（含重试失败） | 100/102 | 100/106 | ≥95%；False |
| 已接受 episode occluded_ripe | 100/100 | 100/100 | 不替代逐尝试成功率 |
| 10布局×3次确定性 | 未审计 | 7/10通过；共同控制步最大差异 5.813525778 rad，另有长度不一致 | 全部10布局<1e-6且长度一致；不通过 |
| reach q1摆幅P90 rad | 2.088454878330231 | 0.7661663055419922 | <1.0；True |
| reach q7摆幅>π | 28/539 | 0/547 | 0 |
| carry q1摆幅P90 rad | 2.096185290813446 | 0.689743357896805 | <1.0；True |
| carry q7摆幅>π | 21/539 | 0/547 | 0 |
| approach末q7偏离中位数>1rad | 51/539 | 279/547 | 0 |
| q7状态超出[−π,π] | 7016/88607 | 286/91349 | 0 |

新 q7 中心（approach末中位数）：-1.0632274150848389 rad。所有数值来自 legacy_stats.json/new_stats.json/determinism.json；逐示范及失败尝试来源在 audit.json 中列出。回合长度比较：{"legacy": {"episodes": 200, "median": 456.0, "p10_p90": [317.9, 607.1]}, "nearest": {"episodes": 200, "median": 463.0, "p10_p90": [324.0, 616.0]}}。

![回合长度直方图](../results/fruit_expert_v2_audit_20261102/episode_lengths.png)

因专家失败后仍按旧规则换种子，成功布局相对旧示范有 20/200 条种子不同；列表见 audit.json。长度及轨迹分布为各自成功示范的描述对照，不假定布局完全配对。

规划方法计数：{"straight": 1100, "screw": 1032, "rrt": 56}；保持段审计：True。成功率按每个实际尝试计，失败后换种子规则保留16次上限；原 manifest 的失败尝试也纳入旧专家分母。

失败原因分类及 no-plan 规划失败比例（含分母）见 audit.json/expert_failure_categories：{"legacy": {"counts": {"detach_failure": 6}, "all_attempts": 206, "no_plan_rate": {"numerator": 0, "denominator": 206, "fraction": 0.0}}, "nearest": {"counts": {"basket_miss": 4, "detach_failure": 10, "no_plan": 3}, "all_attempts": 217, "no_plan_rate": {"numerator": 3, "denominator": 217, "fraction": 0.013824884792626729}}}。

## 静止监督样本（修订1 §9.3）

动作块取前16步，尾部按官方方式重复末动作；以该样本的实测 joint_state 为位移参照，前8步手臂最大位移<0.01rad且后8步≥0.05rad。按原始 is_video_demo=False 的监督锚点统计。官方构建器实际保存20步动作块，本项按任务书显式只诊断前16步，未改构建器。

| 阶段 | 旧示范 命中/监督样本 | 新示范 命中/监督样本 |
|---|---:|---:|
| observe | 1000/6000 | 1000/6000 |
| grasp | 2687/5390 | 2737/5470 |
| release | 0/4312 | 0/4376 |
| settle | 1695/6468 | 1734/6564 |
| done | 0/1000 | 0/1000 |
| other | 0/65437 | 1/67939 |
| 合计 | 5382/88607 | 5472/91349 |

## 安全包络v2所需统计（仅提供统计，未实施包络）

来源 new_stats.json：状态分位为[0,0.5%,1%,50%,99%,99.5%,100%]，完整各关节数值在 audit.json；下表列0.5%/99.5%以及目标相对实测状态偏差。

| 关节 | 状态P0.5 | 状态P99.5 | 目标−状态绝对差 P99 | 目标−状态绝对差 P99.9 |
|---|---:|---:|---:|---:|
| 1 | -2.968464670 | 0.714717997 | 0.124779464 | 0.167959097 |
| 2 | -1.249326551 | 1.137238622 | 0.107712698 | 0.123317946 |
| 3 | -3.050225501 | 3.047982950 | 0.231655286 | 0.262378668 |
| 4 | -2.164558988 | 2.184727721 | 0.259053678 | 0.262655281 |
| 5 | -2.781991243 | 2.727871809 | 0.248190115 | 0.261457311 |
| 6 | -1.467961318 | 2.206537008 | 0.217239640 | 0.249852598 |
| 7 | -2.988855896 | 2.383325834 | 0.198804239 | 0.254936232 |

## 构建器能力评估（先不实施）

原生不支持中途非监督帧。官方 build_robomme_dataset.py:263 强制 is_video_demo == (step_idx < exec_start_idx)；ECM 计划器也按完整执行后缀计算样本数。使用真实官方函数/断言和合成HDF5验证：30帧历史前缀通过，中途40–44帧标历史会触发断言。文件哈希和案例见 builder_assessment.json。

阶段3按 §9.4 只将开头observe30帧作历史，settle/done暂保持监督。若后续获准扩展，需要独立supervision mask、保留每帧特征和原step_idx、重排连续PKL样本编号并同步改计划/验收/加载检查；此处只是方案，没有实施，也没有删除帧。

## D1 两个补充指标（§9.5）

来源 d1_new_metrics/<条件>/{summary.json,episodes.jsonl}；原 D1 所有旧指标和结果不变。离家为TCP距第0帧位置严格>5cm。每次入篮以各果实 in_basket 的 false→true 转换检测，同一步多个果实入篮记一次事件并列全部果实编号；每个事件独立计时，未恢复的事件右删失单列，未用0填补。中位数仅在观察到离家/恢复的样本中计算，未观察者计数必须一起看。

| 条件/任务 | leave_home中位步 | 未离开/回合 | 入篮后恢复中位步 | 结束未恢复/入篮事件 |
|---|---:|---:|---:|---:|
| A0/合计 | 138.0 | 0/40 | 214.5 | 3/15 |
| A0/FruitCount | 139.5 | 0/20 | 214.5 | 1/7 |
| A0/FruitOccludedRipe | 137.0 | 0/20 | 465.0 | 2/8 |
| A1/合计 | 138.0 | 0/40 | 629 | 3/14 |
| A1/FruitCount | 139.5 | 0/20 | 440.5 | 3/5 |
| A1/FruitOccludedRipe | 137.0 | 0/20 | 629 | 0/9 |
| A2/合计 | 956.5 | 10/40 | 424.5 | 1/3 |
| A2/FruitCount | 780.5 | 8/20 | 424.5 | 1/3 |
| A2/FruitOccludedRipe | 996.0 | 2/20 | None | 0/0 |
| A3/合计 | 766 | 5/40 | 256 | 0/1 |
| A3/FruitCount | 799.5 | 2/20 | 256 | 0/1 |
| A3/FruitOccludedRipe | 757 | 3/20 | None | 0/0 |
| A4/合计 | 956.5 | 10/40 | 459 | 1/4 |
| A4/FruitCount | 780.5 | 8/20 | 459 | 1/4 |
| A4/FruitOccludedRipe | 996.0 | 2/20 | None | 0/0 |

## 新专家视频抽帧

来源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/demos/rm75b_eg2/train/count/ep0_manifest.json`；视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/videos/count_ep0.mp4`。

![专家示范](../results/fruit_expert_v2_audit_20261102/videos/count_ep0.png)

来源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/demos/rm75b_eg2/train/count/ep1_manifest.json`；视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/videos/count_ep1.mp4`。

![专家示范](../results/fruit_expert_v2_audit_20261102/videos/count_ep1.png)

来源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/demos/rm75b_eg2/train/occluded_ripe/ep0_manifest.json`；视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/videos/occluded_ripe_ep0.mp4`。

![专家示范](../results/fruit_expert_v2_audit_20261102/videos/occluded_ripe_ep0.png)

来源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/demos/rm75b_eg2/train/occluded_ripe/ep1_manifest.json`；视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/videos/occluded_ripe_ep1.mp4`。

![专家示范](../results/fruit_expert_v2_audit_20261102/videos/occluded_ripe_ep1.png)

## 已确认、推断、决策与下一步

已确认的审计布尔结果：{"success_count": false, "success_occluded_ripe": false, "all_200_demos_present": true, "determinism": false, "reach_q1_p90_below_1": true, "reach_q7_swing_over_pi_zero": true, "carry_q1_p90_below_1": true, "carry_q7_swing_over_pi_zero": true, "grasp_q7_consistent": false, "all_q7_states_within_pi": false, "original_hold_lengths_unchanged": true}。所有失败尝试及重复失败均保留于 audit.json 和各日志；缺失布局不以成功布局替代。

静止监督与闭环等待机制的因果关系仍是推断；此阶段只给出新旧统计，不提前把阶段3的数据改动效果当作已证实。RRT确定性也以重复轨迹实测为准。

存在审计不达标项，按 §4.2 停下等用户确认，不进入阶段3。

## 独立复核与确定性数字说明

复核200个manifest、400个内容文件哈希及全部运行源文件；冻结划分未变。30次重复都完成，但只有7/10布局满足确定性，3/10轨迹长度不一致；共同控制步最大差异为5.813525778rad。自动汇总原先的0.0仅来自等长的7个布局，不能代表全部10个布局；原始自动汇总已另存audit_before_independent_review.json。独立复核没有改动作、轨迹、IK权重、控制器或审计判定。

| 布局 | 三次长度 | 共同控制步最大差异 rad |
|---|---|---:|
| occluded_ripe ep0 | [338, 340, 303] | 5.740575946 |
| occluded_ripe ep1 | [615, 477, 555] | 4.717311364 |
| occluded_ripe ep3 | [418, 367, 367] | 5.813525778 |

q7命令超出[−π,π]为295/91349，实测状态超出为286/91349；实测越界的阶段计数为{'approach': 69, 'grasp': 110, 'pull': 107}。逐回合首个样例及来源在independent_validation.json。该诊断用于区分规划目标与实测跟踪，不替代既定审计；不能仅凭越界断言都是PD滞后造成。

相对数据路径的基目录：`results/fruit_expert_v2_audit_20261102`。

回合长度的定性复核通过：旧/新中位数456/463步，P10–P90为317.9–607.1/324–616步，已检查直方图，分布同阶；未自行增加数值验收阈值。完整轨迹长度不一致时，最大差异字段为null，不能将等长7个布局的0rad当作全部通过；共同控制步最大差异另报5.813525778rad。
