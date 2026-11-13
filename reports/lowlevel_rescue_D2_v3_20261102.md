阶段2专家v3完成：五个核心项通过3项、失败2项，按§10.5停下等待审阅。
失败：连续性97/100（要求≥98/100）；reach/carry第1关节摆幅P90=2.045175/2.048776rad（要求各<1）。
通过：规划确定性100/100；抓取q5/q6/q7一致性各540/540；全部记录q7指令和实测均在[−π,π]。
完成200条示范/224次尝试；保留settle=2；D2三个布局固定动作各回放3次，所有已记录物理状态逐位一致。
决策：不进入阶段3，不再自行迭代v3；A0平台、legacy/nearest及v2产物保留，等待指挥者决定。

## 做了什么

新增独立 canonical 分支。carry 直接取 rest 手臂关节角；reach 先用 rest 初值执行原生 CLIK，失败才用确定性随机初值后备，等权按距 rest 排序，q7 映射到 rest 附近的等价值。目标在检查路径之前确定，不因路径碰撞改换 IK 分支。路径点云半径为0.018m（原0.008m+0.01m），依次直线、经固定rest、经固定high_ready，最后才RRT。固定high_ready数值与全部路径使用记录见代码和规划trace。

approach/pull仍用原plan_screw和0.008m点云，保持observe30、close10、open8、done5。canonical settle默认2；如任一任务或合计落篮失败占尝试>3%，只按指令恢复12并在独立目录重生成，短settle的完整结果保留。下一颗reach规划完成后、执行轨迹前检查前一颗；最后一颗在done后检查。规划本身不推进仿真时钟，因此该延迟并不凭空增加物理等待时间。

## 三方对照

| 项目 | 旧专家 | v2 nearest | v3 canonical | 目标 |
|---|---:|---:|---:|---|
| 逐尝试成功率 count | 100/104 (96.15%) | 100/111 (90.09%) | 100/118 (84.75%) | ≥90%，非核心 |
| 逐尝试成功率 occluded_ripe | 100/102 (98.04%) | 100/106 (94.34%) | 100/106 (94.34%) | ≥90%，非核心 |
| ★ 抓取末q5在自身中位数±1rad | 492/539 (91.28%) | 199/547 (36.38%) | 540/540 (100.00%) | 各≥95% |
| ★ 抓取末q6在自身中位数±1rad | 514/539 (95.36%) | 491/547 (89.76%) | 540/540 (100.00%) | 各≥95% |
| ★ 抓取末q7在自身中位数±1rad | 488/539 (90.54%) | 268/547 (48.99%) | 540/540 (100.00%) | 各≥95% |
| ★ reach的q1摆幅P90 rad | 2.088454878330231 | 0.7661663055419922 | 2.0451749294996264 | <1.0 |
| ★ carry的q1摆幅P90 rad | 2.096185290813446 | 0.689743357896805 | 2.0487763345241548 | <1.0 |
| ★ q7实测范围 rad | [-6.15802526473999, 6.071005344390869] | [-3.4181365966796875, 3.397477149963379] | [-1.8397624492645264, -1.0070732831954956] | 全在[−π,π] |
| ★ q7指令范围 rad | [-6.163645267486572, 6.087190863934422] | [-3.423551630956584, 3.406986623751667] | [-1.840708442332205, -1.0016437729673724] | 全在[−π,π] |
| q7实测越界 | 7016/88607 (7.92%) | 286/91349 (0.31%) | 0/81730 (0.00%) | 0 |
| q7指令越界 | 7066/88607 (7.97%) | 295/91349 (0.32%) | 0/81730 (0.00%) | 0 |
| ★ 同一保存状态的规划重复 | 未按新口径测 | 未按新口径测（D2物理+重规划7/10通过） | 100/100 | reach50+carry50，各3次完全一致 |
| ★ 目标连续性 | 未按新口径测 | 未按新口径测 | 97/100 | ≥98%，目标最大差<.05rad且方法不变 |
| RRT使用比例（含失败尝试） | 未记录 | 62/1136 (5.46%) | 26/1144 (2.27%) | ≤2%，非核心 |
| 被接受示范中使用过RRT | 未记录 | 55/1094 (5.03%) | 3/1080 (0.28%) | 描述对照 |

所有分布来自各自成功示范，失败换种子导致布局不完全配对。旧/v2来源为 /media/fulei/fuleivla/ecm_vla/results/fruit_expert_v2_audit_20261102/legacy_stats.json、new_stats.json 及其中 manifest；新增手腕三关节和指令范围均从原 HDF5 补算，未覆盖旧文件。v3来源 canonical_stats.json。逐尝试各类失败占比：`audit.json/failure_categories`。完整失败动作、GT及保存规划输入在选中数据根目录 attempts、hdf5_files、gt、planning_states。

非核心项：{'rrt_rate': {'numerator': 26, 'denominator': 1144, 'fraction': 0.022727272727272728}, 'rrt_rate_at_most_2percent': False, 'by_task': {'count': {'attempt_success_at_least_90percent': False, 'each_failure_category_at_most_3percent': False}, 'occluded_ripe': {'attempt_success_at_least_90percent': True, 'each_failure_category_at_most_3percent': False}}, 'all_attempt_reach_carry_methods': {'rrt': 26, 'straight': 1020, 'detour': 98}, 'all_attempt_q7': {'samples': 85640, 'state_outside_pi': 0, 'command_outside_pi': 0}, 'rrt_rate_accepted_demos': {'numerator': 3, 'denominator': 1080, 'fraction': 0.002777777777777778}}。RRT主比例的分母包含失败尝试中所有reach/carry规划调用，另列被接受示范的比例。失败原因三方对照：{'legacy': {'count': {'attempts': 104, 'categories': {'detach_failure': {'numerator': 4, 'denominator': 104, 'fraction': 0.038461538461538464}}}, 'occluded_ripe': {'attempts': 102, 'categories': {'detach_failure': {'numerator': 2, 'denominator': 102, 'fraction': 0.0196078431372549}}}}, 'nearest_v2': {'count': {'attempts': 111, 'categories': {'basket_miss': {'numerator': 4, 'denominator': 111, 'fraction': 0.036036036036036036}, 'detach_failure': {'numerator': 5, 'denominator': 111, 'fraction': 0.04504504504504504}, 'planning_failure': {'numerator': 2, 'denominator': 111, 'fraction': 0.018018018018018018}}}, 'occluded_ripe': {'attempts': 106, 'categories': {'detach_failure': {'numerator': 5, 'denominator': 106, 'fraction': 0.04716981132075472}, 'planning_failure': {'numerator': 1, 'denominator': 106, 'fraction': 0.009433962264150943}}}}, 'canonical_v3': {'count': {'attempts': 118, 'categories': {'planning_failure': {'numerator': 18, 'denominator': 118, 'fraction': 0.15254237288135594}}}, 'occluded_ripe': {'attempts': 106, 'categories': {'planning_failure': {'numerator': 6, 'denominator': 106, 'fraction': 0.05660377358490566}}}}}。

settle选择：{'settle_steps': 2, 'data_root': '/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/demos_settle2/rm75b_eg2/train', 'fallback_triggered': False, 'settle2_stats': '/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/settle2_stats.json', 'settle2_failure_breakdown': {'count': {'attempts': 118, 'categories': {'planning_failure': {'numerator': 18, 'denominator': 118, 'fraction': 0.15254237288135594}}}, 'occluded_ripe': {'attempts': 106, 'categories': {'planning_failure': {'numerator': 6, 'denominator': 106, 'fraction': 0.05660377358490566}}}}, 'fallback_rule': 'If either task or pooled basket_miss/all_attempts >3%, generate a separate settle12 dataset once.'}。该记录给出settle2逐任务失败类别和实际分母；触发时只恢复为12一次，两套数据均保留。

## 规划确定性与连续性

按 episode 顺序交错两个任务抽取成功示范中的 reach/carry 实例，保留输入状态、果实目标、点云、附着体、速度/加速度限值和源文件哈希。每个确定性实例从同一保存输入重复3次，比对全部关节路径、时间、速度、加速度和规划trace，要求逐位一致。连续性以原始规划为基准，五次分别在果实中心施加确定的±2mm坐标扰动、测量关节施加±1e−3rad，重新计算接近方向和预抓取位姿；以六次目标的逐关节最大两两差和方法是否一致判定。完整每例记录在planning目录。原生RRT无公共设种子API，实际调用一律保留，不假定它确定。

## D2三个布局的物理首次分叉

| 原D2布局 | 重复对 | 首次动作不同（0起始索引/阶段/幅度rad） | 首次已记录状态不同（已执行步数/字段/幅度） |
|---|---|---|---|
| ep0 | [0, 1] | 156 / ['reach', 'reach'] / 0.0006264726438096613 | 157 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.0002460479736328125, 'max_index': [6], 'first_value': -1.2487150430679321, 'second_value': -1.248961091041565, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 7.140636444091797e-05, 'max_index': [6], 'first_value': -0.5380036234855652, 'second_value': -0.5380750298500061, 'units': 'm_and_quaternion_components'}] |
| ep0 | [0, 2] | 156 / ['reach', 'reach'] / 0.00398777819980585 | 157 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.0015530148521065712, 'max_index': [2], 'first_value': -0.011659393087029457, 'second_value': -0.010106378234922886, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 0.001303344964981079, 'max_index': [5], 'first_value': 0.45662856101989746, 'second_value': 0.45793190598487854, 'units': 'm_and_quaternion_components'}] |
| ep0 | [1, 2] | 156 / ['reach', 'reach'] / 0.004182758722805574 | 157 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.0016287034377455711, 'max_index': [2], 'first_value': -0.011735081672668457, 'second_value': -0.010106378234922886, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 0.0012980103492736816, 'max_index': [5], 'first_value': 0.45663389563560486, 'second_value': 0.45793190598487854, 'units': 'm_and_quaternion_components'}] |
| ep1 | [0, 1] | 187 / ['reach', 'reach'] / 0.0027178659274609274 | 188 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.001056671142578125, 'max_index': [2], 'first_value': 2.930772066116333, 'second_value': 2.931828737258911, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 0.00036960840225219727, 'max_index': [5], 'first_value': 0.4610866606235504, 'second_value': 0.4607170522212982, 'units': 'm_and_quaternion_components'}] |
| ep1 | [0, 2] | 187 / ['reach', 'reach'] / 0.0007421090091555005 | 188 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.00029140710830688477, 'max_index': [6], 'first_value': -0.8043289184570312, 'second_value': -0.8046203255653381, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 0.0001316368579864502, 'max_index': [3], 'first_value': 0.462950199842453, 'second_value': 0.46281856298446655, 'units': 'm_and_quaternion_components'}] |
| ep1 | [1, 2] | 187 / ['reach', 'reach'] / 0.0027489167693990346 | 188 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.0010676383972167969, 'max_index': [2], 'first_value': 2.931828737258911, 'second_value': 2.9307610988616943, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 0.00045502185821533203, 'max_index': [5], 'first_value': 0.4607170522212982, 'second_value': 0.46117207407951355, 'units': 'm_and_quaternion_components'}] |
| ep3 | [0, 1] | 211 / ['reach', 'reach'] / 0.003656536451906778 | 212 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.0014371871948242188, 'max_index': [6], 'first_value': -1.6933956146240234, 'second_value': -1.6919584274291992, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 0.0004559755325317383, 'max_index': [6], 'first_value': -0.5395227670669556, 'second_value': -0.5390667915344238, 'units': 'm_and_quaternion_components'}] |
| ep3 | [0, 2] | 211 / ['reach', 'reach'] / 0.003656536451906778 | 212 / [{'field': 'robot_qpos', 'object': 'robot', 'max_abs_difference': 0.0014371871948242188, 'max_index': [6], 'first_value': -1.6933956146240234, 'second_value': -1.6919584274291992, 'units': 'rad'}, {'field': 'robot_tcp_pose', 'object': 'robot', 'max_abs_difference': 0.0004559755325317383, 'max_index': [6], 'first_value': -0.5395227670669556, 'second_value': -0.5390667915344238, 'units': 'm_and_quaternion_components'}] |
| ep3 | [1, 2] | None / None / None | 已记录字段全程一致 |

原D2重复中的首次状态差和首次动作差见physics/original_d2_divergence.json。对occluded_ripe的ep0/1/3，各固定采用D2 repeat0的同一动作数组，在三个独立进程重放，全程不调用规划器。记录机器人qpos/qvel/TCP、果实位姿/速度/附着/入篮、叶片位姿/push。首次逐位不同与首次>1e−6字段单位的不同分别报告；状态索引k表示已执行k步，避免把执行前记录误作第k个动作的结果。

| 布局 | 重放对 | 首次逐位分叉：已执行步数/对象/幅度 | 首次>1e−6 |
|---|---|---|---|
| ep0 | [0, 1] | 全程完全一致 | 全程完全一致 |
| ep0 | [0, 2] | 全程完全一致 | 全程完全一致 |
| ep0 | [1, 2] | 全程完全一致 | 全程完全一致 |
| ep1 | [0, 1] | 全程完全一致 | 全程完全一致 |
| ep1 | [0, 2] | 全程完全一致 | 全程完全一致 |
| ep1 | [1, 2] | 全程完全一致 | 全程完全一致 |
| ep3 | [0, 1] | 全程完全一致 | 全程完全一致 |
| ep3 | [0, 2] | 全程完全一致 | 全程完全一致 |
| ep3 | [1, 2] | 全程完全一致 | 全程完全一致 |

原D2没有保存关节速度、果实朝向/速度和叶片位姿，已记录字段一致不能证明完整物理状态一致。固定动作若仍发生分叉，才是物理回放微差的直接证据；与原D2规划切换之间的因果关系另需结合首次动作分叉步数解释。没有观测到的叶片或果实分叉不作推断。

## 静止监督与回合长度

| 阶段 | 旧专家 | v2 | v3 |
|---|---:|---:|---:|
| observe | 1000/6000 (16.67%) | 1000/6000 (16.67%) | 1000/6000 (16.67%) |
| grasp | 2687/5390 (49.85%) | 2737/5470 (50.04%) | 2693/5400 (49.87%) |
| release | 0/4312 (0.00%) | 0/4376 (0.00%) | 1700/4320 (39.35%) |
| settle | 1695/6468 (26.21%) | 1734/6564 (26.42%) | 0/1080 (0.00%) |
| done | 0/1000 (0.00%) | 0/1000 (0.00%) | 0/1000 (0.00%) |
| other | 0/65437 (0.00%) | 1/67939 (0.00%) | 0/63930 (0.00%) |
| 合计 | 5382/88607 (6.07%) | 5472/91349 (5.99%) | 5393/81730 (6.60%) |

静止监督定义沿用修订1：相对当前实测状态，前8动作最大关节位移<.01rad、后8≥.05rad，尾部重复末动作，仅统计原监督锚点。observe仍在D2原始记录中保留监督标记；仅阶段3构建时将前30帧设为历史。settle缩短不删帧，done仍保留监督。

长度中位数/P10/P90：{'legacy': {'median': 456.0, 'p10_p90': [317.9, 607.1]}, 'nearest_v2': {'median': 463.0, 'p10_p90': [324.0, 616.0]}, 'canonical_v3': {'median': 427.0, 'p10_p90': [302.0, 565.5]}}。保持段实际长度见canonical_stats.json/hold_segment_lengths。

![三方回合长度](../results/fruit_expert_v3_audit_20261102/episode_lengths.png)

## 4段示范抽帧

视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/videos/count_ep0.mp4`；源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/demos_settle2/rm75b_eg2/train/count/ep0_manifest.json`。

![示范](../results/fruit_expert_v3_audit_20261102/videos/count_ep0.png)

视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/videos/count_ep1.mp4`；源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/demos_settle2/rm75b_eg2/train/count/ep1_manifest.json`。

![示范](../results/fruit_expert_v3_audit_20261102/videos/count_ep1.png)

视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/videos/occluded_ripe_ep0.mp4`；源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/demos_settle2/rm75b_eg2/train/occluded_ripe/ep0_manifest.json`。

![示范](../results/fruit_expert_v3_audit_20261102/videos/occluded_ripe_ep0.png)

视频 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/videos/occluded_ripe_ep1.mp4`；源 `/media/fulei/fuleivla/ecm_vla/results/fruit_expert_v3_audit_20261102/demos_settle2/rm75b_eg2/train/occluded_ripe/ep1_manifest.json`。

![示范](../results/fruit_expert_v3_audit_20261102/videos/occluded_ripe_ep1.png)

## 决策与下一步

存在核心项失败，按§10.1/10.5停下等用户审阅，不再自行迭代v3，不生成阶段3数据或训练。

本阶段无新策略闭环指标；§5完整旧D1指标及§9.5补算已原样附入summary.json/D1_revision1_metrics，不能把专家结果当成K0改善。

## 独立复核与失败项定位

RRT使用对照按段统计“是否调用过RRT”，包含RRT失败后改用其他分支的段；v2全部尝试为62/1136，被接受示范为55/1094；v3分别为26/1144、3/1080。这与只统计最终成功路径为RRT的v2计数47/1094不同，口径与来源见independent_validation.json/rrt_usage_comparison。

独立校验200个manifest、400个成功内容文件哈希、48个失败内容文件哈希、17个运行源文件、236个v2元数据文件及7份v2源文件快照；冻结划分、交接文档、D1补算结果均未变。全部子进程退出码为0。逐示范重新读取81730个成功控制步，失败另3910步，全部q7记录无越界；手腕及q1摆幅独立重算一致。4段视频及抽帧、长度直方图已检查。完整证据见independent_validation.json。

连续性中，100/100实例的纯目标最大两两差都<0.05rad（最大0.043941080rad），失败来自3个实例的规划方法切换，其中1次RRT路径被碰撞/q7检查拒绝；不能把目标平滑误写成完整连续性达标。每坐标扰动为±2mm，对应欧氏位移3.464mm；测量关节每坐标为±0.001rad，五组规则在运行前固定。

| 连续性失败实例 | 最大目标差 rad | 方法（基准+5次） | 失败情况 |
|---|---:|---|---|
| count ep4 step165 | 0.028517655 | ['detour', 'detour', 'rrt', 'detour', 'detour', 'detour'] | ['Success', 'Success', 'RRT path failed canonical collision/q7 validation', 'Success', 'Success', 'Success'] |
| occluded_ripe ep5 step205 | 0.039253118 | ['straight', 'straight', 'straight', 'detour', 'straight', 'straight'] | ['Success', 'Success', 'Success', 'Success', 'Success', 'Success'] |
| count ep15 step168 | 0.026405048 | ['straight', 'detour', 'straight', 'detour', 'straight', 'straight'] | ['Success', 'Success', 'Success', 'Success', 'Success', 'Success'] |

reach目标与固定rest的q1绝对差P90=2.066036321rad；carry目标均已逐段核实等于rest。固定归位与reach目标之间的大角度距离与实测摆幅一致；本轮没有为通过门槛更改rest、权重、摆幅口径或绕行姿态。

D2三个布局在固定动作下均未观测到物理分叉，首次分叉步数、对象和幅度均为“无”。原D2已有记录中，ep0/1/3的首次不同动作分别位于0起始索引156/187/211，均为第二颗果实的reach；执行该动作后才出现已记录机器人状态差，即已执行157/188/212步时。原D2对应规划trace包含RRT第1候选成功与退到第2候选直线的分支差异。本次结果不支持把原分叉直接归因于物理微差，规划后备路径的不确定性是更直接的解释；后者仍应标作由动作、trace和固定回放共同支持的推断，原D2没有记录的速度/姿态不能倒推出数值。

settle静止监督命中由v2的1734/6564降到0/1080，但release从0/4376升到1700/4320；全阶段命中应一起看，不能只凭settle单项认定总体等待问题解决。三方总体命中率见audit.json/comparison各版本static_delayed_supervision/all。
