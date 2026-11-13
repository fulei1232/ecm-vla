修订11：R11_check_failed；已停止。
罩内计数R0完成350/350；信息门槛：True；遮挡R0已定稿，不重跑。
新数据成功109/400；构建与共用归一化：未完成；不训练。
ep112保留为正常DART robot_drop失败，不进入训练，不再要求单条诊断成功。
修订10的R0只作无遮挡板历史记录；旧83条数据不混用；D5/D6及所有历史结果保持，不使用val、不改论文。

## 固定规则与执行顺序

计数 v2 修订11：开跑前固定的规则（探索性，不改变D5结论，不写论文）。

依据交接文档§19及用户本次批准。固定fixed_v2纯视觉罩正式纳入计数v2的R0、数据、后续训练观测和评测定义：环境FruitPickCountShieldV2-v0，count_shield=fixed_v2；遮挡任务FruitPickTaskV2-v0，不加罩。沿用构建前偏移修复、14个渲染盒、13个新增障碍审计盒、零新增碰撞体。任务内部revision=9，专家canonical_v31、DART、包络、随机流和预算均不变。原任务与D5默认行为不变。

ep112结果保持robot_drop正常失败，排除训练数据；不重跑它，不再要求单条DART整回合成功。修订10的R0仅是无遮挡板的历史结果，旧83条示范不混入新数据。遮挡R0已定稿，不重跑。已接受的张量兼容、训练/推理顺序编码、提前开工9例、20条预算/平均每步性能、补果外观及构建/几何检查只复制原证据及sha256，不重做。

1. 首先dev250–259，history专家10条，种子4200000+16×episode；全部完成后拟合绝对时钟及相对时钟参数并冻结sha256，不重拟合。
2. 然后dev200–249，种子4200000+16×episode；每队列50条，顺序history→current→initial_current→initial_ground→clock_onset→clock_onset_k0→clock，共350条。history成功≥45/50，包络裁剪率<0.5%；current、initial_current、initial_ground和clock成功≤15/50；两个onset条件只报告。队列完成后先判对应门槛，再进入下一队列。R0不重试、不换种子。
3. 350条后按冻结的§17.4定义计算信息可得性：KA总体≥95%且N2–4/5–7/8–10各≥90%，K0在N8–10≤50%，recent16总体≤5%。K8与strict区间只报告。缺终点按未覆盖。它是专家轨迹的信息上限，不是模型学到使用记忆的证据。
4. 其余有效性：补果外观、灯亮前提前开工、预算耗尽恰好N等沿用原定义，逐回放复核动作、噪声、标签、脚本信息限制和事件信号any_hanging。篮内果basket_fruit及停放备用果inactive_reserve在双相机必须始终0像素，任意非零立即全局停机。篮子本体的壁、沿、底只记录，不判失败；遮挡板自身不计入篮子。本体统计每任务及clean/DART的可见回合数、帧数（双相机任一可见的一次观测计一帧）、最大单相机像素数；按累计双相机像素排序，交付前5回合的峰值双相机原生截图。报告全部尝试及成功子集，保留失败和不完整尝试。
5. 果实穿板：沿用已接受的严格sphere-vs新增障碍审计几何。R0专家、脚本和校准均立即停机；数据生成单次尝试记success=false/outcome=shield_intersection，关闭并保留该次HDF5及原始证据，不进训练链接，使用下一个原定尝试。全局累计次数≥3且次数/已经实际启动的全部数据尝试>2%时立即停机（恰好2%不触发）。分母包括两个任务、clean及DART、失败和在途尝试；启动和结果入账互斥，触发后不再启动进程，终止在途进程，不为稀释比例额外启动。
6. 夹爪只报告：每次尝试逐观测记录手指或TCP与14个完整渲染盒的接触/交叠帧数及非负最小欧氏距离。手指采用原URDF中与可见几何一致的两个局部盒，经实际link世界位姿变换，用原环境已有的native FCL计算OBB距离；TCP按点到盒计算；接触/交叠距离为0。无遮罩距离null。按R0专家、clean、DART三组及任务分开汇总，不据此改动作或判定。
7. R0全过才生成400条成功示范：两任务各clean train0–99、DART train100–199。每回合最多16次原定尝试，seed=4100000+16×episode+attempt，普通失败也完整保留；仅成功数据入构建。两任务全部重新生成，不混旧83条。前30帧只作历史，等待和完成保持段作监督，any_hanging信号；构建一份数据，计算一次共用归一化供K0/K8/KA使用；顺序时间编码，16帧和模型结构不变。
8. 报告全部数据生成并行总墙钟、含构建/归一化的总耗时、每任务按N及clean/DART分组的单尝试耗时分布；失败尝试单列，不把8卡耗时之和当作总墙钟。重做train信息可得性只报告。
9. 任何未获例外的门槛、单元检查、工程或一致性检查失败即停，不调参、不改定义、不换审计种子。数据全部完成即停审阅，不训练、不使用val150–199、不改论文。D5/D6及历史回放、报告、76份源码以sha256保护；只更新修订11独立目录及全局状态入口。

开跑前冻结本说明、split registry、86份源码、新检查、复用证明和platform_manifest的sha256。执行入口仅scripts/run_fruit_task_v2_r11.py --run；该进程内显式启用新观测和判定规则，旧文件不编辑。


## 计数主队列

| 条件 | 成功/完成 | 通过 | 裁剪/检查步 |
|---|---:|---|---:|
| history | 50/50 | True | 61/65744 |
| current | 0/50 | True | 61/66843 |
| initial_current | 0/50 | True | 61/66529 |
| initial_ground | 0/50 | True | 61/66843 |
| clock_onset | 40/50 | 只报告 | 61/64371 |
| clock_onset_k0 | 40/50 | 只报告 | 61/64373 |
| clock | 11/50 | True | 57/63787 |

## 信息可得性

| 条件/N段 | 全采摘覆盖 | 比例 |
|---|---:|---:|
| K0/all | 27/50 | 0.54 |
| K0/2-4 | 14/14 | 1.0 |
| K0/5-7 | 13/21 | 0.6190476190476191 |
| K0/8-10 | 0/15 | 0.0 |
| K8/all | 20/50 | 0.4 |
| K8/2-4 | 14/14 | 1.0 |
| K8/5-7 | 6/21 | 0.2857142857142857 |
| K8/8-10 | 0/15 | 0.0 |
| KA/all | 50/50 | 1.0 |
| KA/2-4 | 14/14 | 1.0 |
| KA/5-7 | 21/21 | 1.0 |
| KA/8-10 | 15/15 | 1.0 |
| recent16/all | 0/50 | 0.0 |
| recent16/2-4 | 0/14 | 0.0 |
| recent16/5-7 | 0/21 | 0.0 |
| recent16/8-10 | 0/15 | 0.0 |

## 篮子本体与夹爪（均只报告）

| 任务/组别（全部尝试） | 尝试 | 本体可见回合/帧 | 单帧最大像素 | 夹爪进入帧 | 最小距离(m) |
|---|---:|---:|---:|---:|---:|
| count/R0_scripts/clock | 50 | 0/0 | 0 | 0 | 0.08544488251209259 |
| count/R0_scripts/clock_onset | 50 | 0/0 | 0 | 0 | 0.08544488251209259 |
| count/R0_scripts/clock_onset_k0 | 50 | 0/0 | 0 | 0 | 0.08544488251209259 |
| count/R0_scripts/current | 50 | 0/0 | 0 | 0 | 0.08544488251209259 |
| count/R0_expert | 50 | 0/0 | 0 | 0 | 0.08544488251209259 |
| count/R0_scripts/initial_current | 50 | 0/0 | 0 | 0 | 0.08544488251209259 |
| count/R0_scripts/initial_ground | 50 | 0/0 | 0 | 0 | 0.08544488251209259 |
| count/calibration | 10 | 0/0 | 0 | 0 | 0.09303376078605652 |
| count/clean_data | 101 | 0/0 | 0 | 0 | 0.08542056381702423 |
| count/DART_data | 65 | 0/0 | 0 | 0 | 0.00445827841758728 |

逐尝试、按任务和clean/DART分组、全部尝试及成功子集：[visibility_gripper_audit.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r11_20261106/visibility_gripper_audit.json)。
篮内果和备用果仍为零像素硬门槛；夹爪用原URDF可见手指盒和TCP相对14个罩子渲染盒的欧氏距离，无遮罩距离为null。

## 数据耗时

数据尚未完成；已有阶段耗时：{'started_utc': '2026-11-09T03:33:17.013896+00:00', 'raw_generation_wall_seconds': None, 'feature_build_wall_seconds': None, 'normalization_and_delivery_wall_seconds': None, 'completed': False, 'total_wall_seconds': None, 'elapsed_wall_seconds_at_stop': 1734.4660653211176, 'stopped_utc': '2026-11-09T04:02:11.479963+00:00'}。

## 决策与下一步

SHIELD process/check failed exit1: /media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r11_20261106/replays/train_dart/count/history/ep113_attempt3/console.log
机器汇总：[summary.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r11_20261106/summary.json)。
