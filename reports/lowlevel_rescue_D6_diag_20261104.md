探索性、非预登记，不改变D5结论，不写入论文；判定C（按§14.5点估计规则）。
第一部分D5既有1200条及200条遮挡训练示范已分析；第二部分完成480/480条。
遮挡成功率下降7.50个百分点，遗漏+错果上升2.50个百分点。
D5中K0/K4/K8每次遮挡决策均保留第0帧；没有事件帧挤掉阵风前帧的证据。
已按§14.6在D6停止，等待用户审阅；D5确证性结论保持原样。

## 第一部分：遮挡任务的遗漏、查看与错果

来源：`/media/fulei/fuleivla/ecm_vla/results/fruit_memory_usage_diag_20261104/posthoc/summary.json`、episodes.jsonl和experts.json。只用既有D5回放，布局标签从回放种子重建并核对初始GT，没有读取val示范或产生新val回放。

| 范围/条件 | 遗漏hidden ripe | 查看hidden unripe | 查看/有hidden unripe回合 | 错果 | hidden错果 | visible错果 |
|---|---:|---:|---:|---:|---:|---:|
| main/K0 | 16/150 (10.67%) | 63/150 (42.00%) | 63/81 (77.78%) | 12/150 (8.00%) | 3/150 (2.00%) | 9/150 (6.00%) |
| main/K4 | 23/150 (15.33%) | 61/150 (40.67%) | 61/81 (75.31%) | 21/150 (14.00%) | 8/150 (5.33%) | 13/150 (8.67%) |
| main/K8 | 20/150 (13.33%) | 57/150 (38.00%) | 57/81 (70.37%) | 20/150 (13.33%) | 6/150 (4.00%) | 14/150 (9.33%) |
| sensitivity/K0 | 8/50 (16.00%) | 20/50 (40.00%) | 20/27 (74.07%) | 7/50 (14.00%) | 1/50 (2.00%) | 6/50 (12.00%) |
| sensitivity/K4 | 7/50 (14.00%) | 21/50 (42.00%) | 21/27 (77.78%) | 5/50 (10.00%) | 2/50 (4.00%) | 3/50 (6.00%) |
| sensitivity/K8 | 8/50 (16.00%) | 17/50 (34.00%) | 17/27 (62.96%) | 9/50 (18.00%) | 4/50 (8.00%) | 5/50 (10.00%) |

| 范围/模型/果实组 | 果实数 | 接近15cm | 接近后脱果 | 接近未脱果 | 实际叶片抬起 |
|---|---:|---:|---:|---:|---:|
| main/K0/hidden_ripe | 219 | 203/219 (92.69%) | 129/219 (58.90%) | 74/219 (33.79%) | 207/219 (94.52%) |
| main/K0/hidden_unripe | 81 | 66/81 (81.48%) | 3/81 (3.70%) | 63/81 (77.78%) | 77/81 (95.06%) |
| main/K0/visible_unripe | 237 | 188/237 (79.32%) | 9/237 (3.80%) | 179/237 (75.53%) | 0/237 (0.00%) |
| main/K4/hidden_ripe | 219 | 196/219 (89.50%) | 124/219 (56.62%) | 72/219 (32.88%) | 201/219 (91.78%) |
| main/K4/hidden_unripe | 81 | 69/81 (85.19%) | 8/81 (9.88%) | 61/81 (75.31%) | 73/81 (90.12%) |
| main/K4/visible_unripe | 237 | 179/237 (75.53%) | 13/237 (5.49%) | 166/237 (70.04%) | 0/237 (0.00%) |
| main/K8/hidden_ripe | 219 | 198/219 (90.41%) | 127/219 (57.99%) | 71/219 (32.42%) | 206/219 (94.06%) |
| main/K8/hidden_unripe | 81 | 63/81 (77.78%) | 6/81 (7.41%) | 57/81 (70.37%) | 74/81 (91.36%) |
| main/K8/visible_unripe | 237 | 184/237 (77.64%) | 14/237 (5.91%) | 170/237 (71.73%) | 0/237 (0.00%) |
| sensitivity/K0/hidden_ripe | 73 | 65/73 (89.04%) | 40/73 (54.79%) | 25/73 (34.25%) | 64/73 (87.67%) |
| sensitivity/K0/hidden_unripe | 27 | 21/27 (77.78%) | 1/27 (3.70%) | 20/27 (74.07%) | 25/27 (92.59%) |
| sensitivity/K0/visible_unripe | 79 | 61/79 (77.22%) | 6/79 (7.59%) | 55/79 (69.62%) | 0/79 (0.00%) |
| sensitivity/K4/hidden_ripe | 73 | 66/73 (90.41%) | 35/73 (47.95%) | 31/73 (42.47%) | 68/73 (93.15%) |
| sensitivity/K4/hidden_unripe | 27 | 23/27 (85.19%) | 2/27 (7.41%) | 21/27 (77.78%) | 23/27 (85.19%) |
| sensitivity/K4/visible_unripe | 79 | 59/79 (74.68%) | 3/79 (3.80%) | 56/79 (70.89%) | 0/79 (0.00%) |
| sensitivity/K8/hidden_ripe | 73 | 65/73 (89.04%) | 38/73 (52.05%) | 27/73 (36.99%) | 68/73 (93.15%) |
| sensitivity/K8/hidden_unripe | 27 | 21/27 (77.78%) | 4/27 (14.81%) | 17/27 (62.96%) | 24/27 (88.89%) |
| sensitivity/K8/visible_unripe | 79 | 60/79 (75.95%) | 5/79 (6.33%) | 55/79 (69.62%) | 0/79 (0.00%) |

接近/未脱果是几何代理，不等于主动查看；已有日志不能排除瞬时触碰或单步抓握。PUSH_START作用于移动叶簇中心，另列的果实中心15cm口径是用户指定的主指标。错果含未成熟脱果或错误入篮，另有仅wrong_in_basket和每颗果实明细。

| 布局聚类对比（非预登记） | 差异百分点 [95%区间] |
|---|---|
| main/K4-K0/success | -0.67 [-9.33, 8.67] |
| main/K4-K0/omission_hidden_ripe | 4.67 [-2.67, 12.00] |
| main/K4-K0/checked_hidden_unripe | -1.33 [-6.67, 3.33] |
| main/K4-K0/wrong_fruit | 6.00 [-1.33, 13.33] |
| main/K4-K0/wrong_hidden_unripe | 3.33 [-0.67, 8.00] |
| main/K4-K0/wrong_visible_unripe | 2.67 [-3.33, 8.67] |
| main/K4-K0/wrong_in_basket | 4.00 [-2.00, 10.00] |
| main/K4-K0/omission_plus_wrong_fruit | 10.67 [-0.00, 21.33] |
| main/K8-K0/success | 1.33 [-8.00, 10.67] |
| main/K8-K0/omission_hidden_ripe | 2.67 [-3.33, 9.33] |
| main/K8-K0/checked_hidden_unripe | -4.00 [-8.67, 0.67] |
| main/K8-K0/wrong_fruit | 5.33 [-1.33, 12.00] |
| main/K8-K0/wrong_hidden_unripe | 2.00 [-1.33, 6.00] |
| main/K8-K0/wrong_visible_unripe | 3.33 [-2.00, 8.67] |
| main/K8-K0/wrong_in_basket | 2.67 [-3.33, 8.67] |
| main/K8-K0/omission_plus_wrong_fruit | 8.00 [-1.33, 18.00] |
| sensitivity/K4-K0/success | -6.00 [-22.00, 10.00] |
| sensitivity/K4-K0/omission_hidden_ripe | -2.00 [-14.00, 10.00] |
| sensitivity/K4-K0/checked_hidden_unripe | 2.00 [-8.00, 12.00] |
| sensitivity/K4-K0/wrong_fruit | -4.00 [-16.00, 8.00] |
| sensitivity/K4-K0/wrong_hidden_unripe | 2.00 [-4.00, 8.00] |
| sensitivity/K4-K0/wrong_visible_unripe | -6.00 [-16.00, 4.00] |
| sensitivity/K4-K0/wrong_in_basket | -6.00 [-16.00, 4.00] |
| sensitivity/K4-K0/omission_plus_wrong_fruit | -6.00 [-22.00, 10.00] |
| sensitivity/K8-K0/success | -6.00 [-22.00, 10.00] |
| sensitivity/K8-K0/omission_hidden_ripe | 0.00 [-10.00, 10.00] |
| sensitivity/K8-K0/checked_hidden_unripe | -6.00 [-16.00, 4.00] |
| sensitivity/K8-K0/wrong_fruit | 4.00 [-8.00, 16.00] |
| sensitivity/K8-K0/wrong_hidden_unripe | 6.00 [-2.00, 14.00] |
| sensitivity/K8-K0/wrong_visible_unripe | -2.00 [-10.00, 6.00] |
| sensitivity/K8-K0/wrong_in_basket | -4.00 [-14.00, 6.00] |
| sensitivity/K8-K0/omission_plus_wrong_fruit | 4.00 [-12.00, 20.00] |

## 第一部分：阵风前帧保留

| 范围/条件 | 阶段 | pre_gust_retained/决策 | 第0帧/决策（全部） |
|---|---|---:|---:|
| main/K0 | before_target_1 | 1699/1699 (100.00%) | 6656/6656 (100.00%) |
| main/K0 | before_target_2 | 3243/3243 (100.00%) | 6656/6656 (100.00%) |
| main/K0 | before_target_3 | 1425/1425 (100.00%) | 6656/6656 (100.00%) |
| main/K0 | after_all_targets | 289/289 (100.00%) | 6656/6656 (100.00%) |
| main/K4 | before_target_1 | 1649/1649 (100.00%) | 5239/5239 (100.00%) |
| main/K4 | before_target_2 | 2350/2350 (100.00%) | 5239/5239 (100.00%) |
| main/K4 | before_target_3 | 991/991 (100.00%) | 5239/5239 (100.00%) |
| main/K4 | after_all_targets | 249/249 (100.00%) | 5239/5239 (100.00%) |
| main/K8 | before_target_1 | 1233/1233 (100.00%) | 6113/6113 (100.00%) |
| main/K8 | before_target_2 | 2945/2945 (100.00%) | 6113/6113 (100.00%) |
| main/K8 | before_target_3 | 1666/1666 (100.00%) | 6113/6113 (100.00%) |
| main/K8 | after_all_targets | 269/269 (100.00%) | 6113/6113 (100.00%) |
| sensitivity/K0 | before_target_1 | 524/524 (100.00%) | 1996/1996 (100.00%) |
| sensitivity/K0 | before_target_2 | 981/981 (100.00%) | 1996/1996 (100.00%) |
| sensitivity/K0 | before_target_3 | 378/378 (100.00%) | 1996/1996 (100.00%) |
| sensitivity/K0 | after_all_targets | 113/113 (100.00%) | 1996/1996 (100.00%) |
| sensitivity/K4 | before_target_1 | 671/671 (100.00%) | 1867/1867 (100.00%) |
| sensitivity/K4 | before_target_2 | 882/882 (100.00%) | 1867/1867 (100.00%) |
| sensitivity/K4 | before_target_3 | 240/240 (100.00%) | 1867/1867 (100.00%) |
| sensitivity/K4 | after_all_targets | 74/74 (100.00%) | 1867/1867 (100.00%) |
| sensitivity/K8 | before_target_1 | 473/473 (100.00%) | 1989/1989 (100.00%) |
| sensitivity/K8 | before_target_2 | 847/847 (100.00%) | 1989/1989 (100.00%) |
| sensitivity/K8 | before_target_3 | 581/581 (100.00%) | 1989/1989 (100.00%) |
| sensitivity/K8 | after_all_targets | 88/88 (100.00%) | 1989/1989 (100.00%) |

这里使用物理步<8；策略相对时钟=物理步−30，未把策略步<8误当阵风前。第1/2/3颗之前按已脱离成熟目标数分期，已摘尽单列。均匀取帧包含第0帧；事件关键帧仍有均匀剩余槽，也保留第0帧。保留证据不等于模型确实利用证据。

## 第一部分：专家示范对照

| 专家 | 遗漏 | 几何查看/全部 | 几何查看/有hidden unripe | 明确reach hidden ripe | 首次15cm接近时目标吻合 | 明确unripe目标数 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0/100 (0.00%) | 50/100 (50.00%) | 50/68 (73.53%) | 132/132 (100.00%) | 82/132 (62.12%) | 0 |
| DART | 0/100 (0.00%) | 49/100 (49.00%) | 49/60 (81.67%) | 140/140 (100.00%) | 77/140 (55.00%) | 0 |

专家会直接规划被遮挡成熟目标，却也经常路过未成熟邻居的15cm范围。因此该几何查看指标不能单独支持“策略逐个查看而不用记忆”的解释。训练与D5布局不同，专家比较仅作描述。

## 第一部分：计数掉果阶段与超时末段

| 范围/条件 | 失败回合 | 掉果事件 | 抓取 | 拉拽 | 搬运 | 释放 | 超时末段标签（回合数） |
|---|---:|---:|---:|---:|---:|---:|---|
| main/K0 | 39/150 | 39 | 1/39 (2.56%) | 26/39 (66.67%) | 11/39 (28.21%) | 1/39 (2.56%) | {'ongoing_motion': 10} |
| main/K4 | 42/150 | 42 | 6/42 (14.29%) | 24/42 (57.14%) | 11/42 (26.19%) | 1/42 (2.38%) | {'ongoing_motion': 15} |
| main/K8 | 35/150 | 35 | 5/35 (14.29%) | 18/35 (51.43%) | 11/35 (31.43%) | 1/35 (2.86%) | {'ongoing_motion': 9} |
| sensitivity/K0 | 13/50 | 13 | 1/13 (7.69%) | 5/13 (38.46%) | 5/13 (38.46%) | 2/13 (15.38%) | {'ongoing_motion': 3} |
| sensitivity/K4 | 12/50 | 12 | 1/12 (8.33%) | 8/12 (66.67%) | 1/12 (8.33%) | 2/12 (16.67%) | {'ongoing_motion': 9} |
| sensitivity/K8 | 12/50 | 12 | 3/12 (25.00%) | 7/12 (58.33%) | 2/12 (16.67%) | 0/12 (0.00%) | {'ongoing_motion': 2} |

阶段依据脱果、首次8cm分离、张开命令和脱果后TCP移动5cm构成观察性代理，不把落地时刻当成失手原因；分母为掉果果实事件，失败回合另列。超时末段200步的原始运动、夹爪和TCP统计在逐回合文件中。

## 第二部分：G1核对、消融实现与审计

五条G1逐位核对有4条完全一致，计数ep249不一致；首轮输入及记忆张量摘要相同，模型动作摘要首先出现差异，后续物理轨迹分叉。按既定规则不复用G1，种子7 normal全部80条重新回放（包含5条核对），480条比较样本均来自本次新回放。核对明细：`results/fruit_memory_usage_diag_20261104/pilot_comparison.json`。
默认不开启任何消融。启用诊断服务后仍需每条context显式指定recent16或no_pregust；normal无标记。封存源码全部不改。no_pregust排除物理步<20，初次只有11帧，沿用官方右padding；不删历史、不改帧号/位置编码。
每条审计的源文件摘要和逐决策检查：`/media/fulei/fuleivla/ecm_vla/results/fruit_memory_usage_diag_20261104/audits`。已审计480/480条，共17398次决策；本次未复用G1；审计只写入新目录，G1原始目录未改。

| 条件/任务 | 成功 | 首颗入篮 | 工程错误 | 执行腕偏离 |
|---|---:|---:|---:|---:|
| K0_normal/FruitCount | 52/80 (65.00%) | 74/80 (92.50%) | 0/80 (0.00%) | 0/80 (0.00%) |
| K0_normal/FruitOccludedRipe | 26/80 (32.50%) | 58/80 (72.50%) | 0/80 (0.00%) | 0/80 (0.00%) |
| K0_recent16/FruitCount | 48/80 (60.00%) | 63/80 (78.75%) | 0/80 (0.00%) | 0/80 (0.00%) |
| K0_recent16/FruitOccludedRipe | 15/80 (18.75%) | 45/80 (56.25%) | 0/80 (0.00%) | 0/80 (0.00%) |
| K0_no_pregust/FruitCount | 49/80 (61.25%) | 69/80 (86.25%) | 0/80 (0.00%) | 0/80 (0.00%) |
| K0_no_pregust/FruitOccludedRipe | 20/80 (25.00%) | 48/80 (60.00%) | 0/80 (0.00%) | 0/80 (0.00%) |

| 条件 | 遗漏hidden ripe | 几何查看hidden unripe | 错果 | 仅错误入篮 | hidden错果 | visible错果 | 遗漏+错果总分/回合 |
|---|---:|---:|---:|---:|---:|---:|---|
| K0_normal | 13/80 (16.25%) | 29/80 (36.25%) | 9/80 (11.25%) | 8/80 (10.00%) | 1/80 (1.25%) | 8/80 (10.00%) | 22/80 |
| K0_recent16 | 11/80 (13.75%) | 28/80 (35.00%) | 9/80 (11.25%) | 5/80 (6.25%) | 4/80 (5.00%) | 5/80 (6.25%) | 20/80 |
| K0_no_pregust | 12/80 (15.00%) | 24/80 (30.00%) | 12/80 (15.00%) | 7/80 (8.75%) | 4/80 (5.00%) | 8/80 (10.00%) | 24/80 |

| 相对normal（探索性） | 差异百分点 [95%布局聚类区间] |
|---|---|
| FruitCount/K0_recent16-K0_normal/success | -5.00 [-17.50, 6.25] |
| FruitOccludedRipe/K0_recent16-K0_normal/success | -13.75 [-23.75, -3.75] |
| FruitOccludedRipe/K0_recent16-K0_normal/omission_hidden_ripe | -2.50 [-13.75, 8.75] |
| FruitOccludedRipe/K0_recent16-K0_normal/checked_hidden_unripe | -1.25 [-10.00, 7.50] |
| FruitOccludedRipe/K0_recent16-K0_normal/wrong_fruit | 0.00 [-8.75, 8.75] |
| FruitOccludedRipe/K0_recent16-K0_normal/wrong_in_basket | -3.75 [-12.50, 5.00] |
| FruitOccludedRipe/K0_recent16-K0_normal/wrong_hidden_unripe | 3.75 [0.00, 8.75] |
| FruitOccludedRipe/K0_recent16-K0_normal/wrong_visible_unripe | -3.75 [-12.50, 5.00] |
| FruitOccludedRipe/K0_recent16-K0_normal/omission_plus_wrong_fruit | -2.50 [-17.50, 12.50] |
| FruitOccludedRipe/K0_recent16-K0_normal/omission_or_wrong_fruit | -1.25 [-15.00, 11.25] |
| FruitCount/K0_no_pregust-K0_normal/success | -3.75 [-17.50, 10.00] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/success | -7.50 [-20.00, 5.00] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/omission_hidden_ripe | -1.25 [-12.50, 10.00] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/checked_hidden_unripe | -6.25 [-13.75, 0.00] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/wrong_fruit | 3.75 [-3.75, 11.25] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/wrong_in_basket | -1.25 [-8.75, 6.25] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/wrong_hidden_unripe | 3.75 [-1.25, 8.75] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/wrong_visible_unripe | 0.00 [-5.00, 5.00] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/omission_plus_wrong_fruit | 2.50 [-11.25, 17.50] |
| FruitOccludedRipe/K0_no_pregust-K0_normal/omission_or_wrong_fruit | 2.50 [-11.25, 16.25] |

| 推理条件 | 记忆中阵风前帧/决策 | 第0帧/决策 |
|---|---:|---:|
| K0_normal | 3487/3487 (100.00%) | 3487/3487 (100.00%) |
| K0_recent16 | 0/3146 (0.00%) | 0/3146 (0.00%) |
| K0_no_pregust | 0/3547 (0.00%) | 0/3547 (0.00%) |

| 推理条件/计数 | 失败回合 | 抓取事件 | 拉拽事件 | 搬运事件 | 释放事件 | 超时末段 |
|---|---:|---:|---:|---:|---:|---|
| K0_normal | 21/80 | 1/21 (4.76%) | 12/21 (57.14%) | 7/21 (33.33%) | 1/21 (4.76%) | {'ongoing_motion': 7} |
| K0_recent16 | 22/80 | 2/22 (9.09%) | 10/22 (45.45%) | 9/22 (40.91%) | 1/22 (4.55%) | {'ongoing_motion': 10} |
| K0_no_pregust | 24/80 | 6/24 (25.00%) | 11/24 (45.83%) | 6/24 (25.00%) | 1/24 (4.17%) | {'ongoing_motion': 7} |

计数recent16相对normal的成功率差为-5.00 [-17.50, 6.25]个百分点；每80回合的掉果果实事件数差（recent−normal）为{'grasp': 1, 'pull': -2, 'carry': 2, 'release': 0}。结合超时末段与掉果阶段作描述，不把记忆消融后的控制退化误认为计数任务恢复了计数需求。历史运动/阶段信息或帧分布改变都是可能解释，本诊断不定位唯一原因。

## §14.5判定与局限

判定 **C**。no_pregust相对normal的遮挡成功率由26/80降至20/80（下降7.50个百分点，介于5与10之间）；遗漏+错果分数从22/80增至24/80（上升2.50个百分点）。两项均未达到A的10个百分点门槛，但成功下降未满足B的严格<5个百分点，故按原规则判C。成功率差no_pregust−normal的95%布局聚类区间为[−20.00, 5.00]个百分点，错误合计差区间为[−11.25, 17.50]；区间只作探索性描述，不参与改判。
合计错误=遗漏回合指示+commission回合指示，同一回合可计2，另列并集及仅错误入篮。按有符号的成功下降/错误上升和预先指定10pp、5pp门槛判定；CI不参与改判。这是探索性诊断，不是新的确证性记忆效果结论。
计数recent16即使下降，也可能因控制阶段信息或输入帧分布改变；计数环境达N即终止的结构缺陷依然存在，不能据此证明在计数。所有任务重设计都留给用户决定。

## 结果解释与独立核验

recent16的遮挡成功率为15/80，相对normal下降13.75个百分点（探索性布局区间[−23.75, −3.75]），但§14.5预先指定的判定只使用no_pregust，不能用recent16替换后改判A。no_pregust的遗漏为12/80（normal为13/80），错果为12/80（normal为9/80）；本次合计上升主要来自错果，未观察到遗漏增加。计数recent16下降5.00个百分点且区间跨0，仅作操作行为参照，不改变计数环境的结构问题。

独立核验从原始回放重新计算遗漏、错果和判定，并重跑逐条记忆审计：480/480条、160组配对样本、17398次决策全部通过，工程错误0/480。核验代码：`scripts/finalize_fruit_memory_usage_diag.py`；逐例证据：`reports/lowlevel_rescue_D6_diag_20261104/independent_validation.json`；机器摘要的`independent_validation`记录证据与核验代码sha256。全部回放和策略服务已退出，不再追加实验。

## 保全与下一步

保全摘要：{'D5_artifacts_unchanged': True, 'sealed_files_sha256': 'dd476f53c788aa0d95f900c8f555fd61f4a76181064a4a0a8472c03ebe3acf91', 'D5_tree_manifest_sha256': 'edbe073aa8daeb77191437fee30fd236befa4b9c5541d958c8aa7ddcdd8b185f'}。D5所有文件大小/mtime、链接及封存摘要核对不变；新数据只来自dev220–259种子7/8。
来源：`/media/fulei/fuleivla/ecm_vla/results/fruit_memory_usage_diag_20261104/posthoc`、`/media/fulei/fuleivla/ecm_vla/results/fruit_memory_usage_diag_20261104/episodes`、`/media/fulei/fuleivla/ecm_vla/results/fruit_memory_usage_diag_20261104/audits`、`/media/fulei/fuleivla/ecm_vla/results/fruit_memory_usage_diag_20261104/summary.json`与执行计划。
D6交付后停止等待用户审阅；全部标注探索性、非预登记，不改变D5确证性结论，不写入论文。
