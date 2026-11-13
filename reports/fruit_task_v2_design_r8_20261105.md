R0 修订8的既定门槛全部通过，已停止，等待用户审阅；不进入数据生成或训练。
全新门槛回放450/450条，校准和压力30/30条；480条记录审计通过，工程和检查错误均为0。
计数专家50/50、绝对时钟13/50；遮挡当前帧/首帧加当前帧/专家为0/50、46/50、48/50，均达预定门槛。
1800步预算检查通过：专家最慢1636步、余164步；不停摘最迟1683步触发多摘。四项入篮检查通过，D5末帧核查0/1200异常。
开工时刻时钟45/50仅作报告：每次放果的事件记忆尚未被证明必需；D5/D6结论与封存产物保持原样。

## 开跑前固定的任务定义与信息边界

# R0修订8：物理回放前固定的设计与审计规则

依据交接文档§16及用户本轮指令。v2在本提交中原地更新；旧候选以6eb6197为准，原结果目录/报告保留。D5、D6的产物、封存文件和v1源码不改。本轮无训练数据、无模型训练、无论文改动。

## 1. 任务语义表

| 项目 | 计数v2 | 遮挡v2 |
|---|---|---|
| 开关 | 显式task_version=v2；默认v1不启用新机制，原FruitPick-v1入口不改 | 同左 |
| 入篮几何 | abs(rel_x)<0.09、abs(rel_y)<0.09、0<rel_z<0.12 | 同左；唯一新增语义修补 |
| 几何依据 | 原env.py底板局部中心0.003、半厚0.003，位姿z=0.15，所以底板下/上表面0.150/0.156，壁顶0.270m | 同左 |
| N | 新独立随机流U{2,3,4,5}，不依赖树果M（4–6） | 原2–3颗成熟果目标定义不改 |
| 指令 | 只有N变量，写明等绿灯、风落不计、完成后回rest张开静止 | 记住初始红果；阵风后摘取那些果实 |
| 随机开工 | 30步观察后再等D~U{0…600}步；D与N、M、风落使用独立随机流 | 不加灯、不加随机等待 |
| 指示灯 | 前置相机预定归一化位置约(0.12,0.12)，半径0.02m纯视觉球，不加碰撞；红→绿后保持绿 | 无 |
| 提前开工 | 灯前离rest不判失败，单记early_start；本轮既定脚本都等绿灯 | 不适用 |
| 风落 | 1–2颗，时间为绿灯后34–66步；只选当前挂树、未抓握、距TCP>0.15m的果实；不足时延后 | 原颜色阵风8–20步和叶簇抬起保留 |
| 补果 | 机器人果实入篮且连续2步未被抓握时触发；TCP距原挂果位<0.15m时延后，记录延迟 | 无补果 |
| 外观 | 新果半径、RGBA材质、粗糙度、果顶小球、姿态与原果一致 | 所有果实在8–20步渐变到统一RGB(.55,.55,.55)，之后不恢复 |
| 多摘/掉果 | 第N+1颗机器人脱果立即失败；机器人篮外落到z<0.06立即失败；风落单独标记 | 原未成熟入篮/机器人掉果失败逻辑不变 |
| 完成信号 | 只在绿灯后计连续40步：臂关节距rest≤.05rad、速度≤.05rad/s、两指开度≥.03m、无持果 | 原所有成熟入篮且无未成熟入篮即成功 |
| 预算 | 暂1800物理步；到时一律失败，包括恰好N颗已入篮的情况，后者单独统计 | 保持1300步及原终止行为 |
| success/terminated | 新count_outcome仅完成信号时可成功；ManiSkill仍按success OR fail置terminated | 父类evaluate调用v2修补的in_basket；成功/失败定义不改 |

预算与完成同时到达时按预算失败，不允许用预算结束代替计数停止。前30步不累计完成静止。条件中的“树上果数不变”指机器人完成放果并补回后，不再随累计摘数递减；搬运中暂空、风落位置不补，严格遵循用户给定触发条件。

## 2. 备用实体与运动规划

每位置预建N个备用果（加原果共N+1份），足够供N次成功放果后的补回，以及摘下第N+1颗以检测多摘。实体仅在_load_scene创建，运行中不新增。备用果停放于x≥100、y=100、z=10的互不相交位置，超出两相机10m远裁剪面；碰撞组全0，不能接触任何物体。启用时恢复原碰撞组、移到原位、设为kinematic。

物理实体编号永久保留：旧果仍留在篮子，新果使用备用实体，因此v3.1原有延迟入篮检查仍指向旧果，不改规划器、IK、碰撞几何、tuck、路径校验或2步settle。挂树选择只看当前可见果的y，补果后会反复摘同一位置；风落使目标消失时取消该技能，再观察选择，不轮换位置。

每次抓放后，低层技能等本次入篮确认和补回，再允许下次选果；不向有限信息选择器传送累计放果次数或补果次数。备用、风落、已入篮实体均不作为挂树候选。事件距离只计算真实挂树集合，备用实体完全排除。

有限信息接口只给稳定的挂果位置编号，按位置编号排序；不暴露物理实体ID、代数、备用池长度或遍历顺序，防止补果次数从新ID泄露。地面果按几何位置排序，不给风落标签或实体代数。

## 3. 入篮单元检查与D5核查

(a) 篮底静置果心z=.156+半径：应入篮；(b) 篮下地面果心z=-.4+半径：不入篮；(c) 篮口以上：不入篮；边界rel_z=0或.12不入篮。
(d) 用ep245原90帧真值检查新规则，再在原R7场景执行其89条原执行动作，唯一行为改动为v2篮底下界。该回归是用户明确要求的例外，不用于新任务性能统计。

D5仅扫描既有1200条末帧的位置与in_basket标记，查rel_z≤0的果实，报告果实数和回合；不重跑、不重判、不改D5。

## 4. 补果等效性和截图

Count每次补果后，以及每个实际进入完成保持段的回合末，比较前置图像与第0帧。按稳定位置合并分割掩码，剔除风落位置与指示灯掩码；ROI是两掩码并集。门槛IoU≥.98、RGB平均绝对差≤2/255，最大差另报。像素按原始坐标逐像素比较，不对齐、不缩放、不更改曝光或阈值。

预期失败（例如多摘）若没有完成保持段，末帧比较记N/A，不能记通过；仍保留并检查此前每次补果后的相同ROI。没有可比较帧的规划失败同样明示N/A，而非补跑或换seed。有可比较帧且检查不过立即停。

Count阶段：灯前、灯后、风落后（落地后首帧）、抓前（首条关闭命令前）、释放、补回后、完成保持最后一帧，各给双相机原始截图。Occluded阶段保持开头、阵风后、抓前、释放。逐步记录两相机的灯、篮子、篮内果实、停放备用果的分割像素。篮子、篮内果实或停放备用果任何正像素立即停；灯的腕视可见性只如实报告，前置初始灯须可见。

## 5. 冻结的脚本队列和输入

所有脚本共享同一v3.1运动技能、包络v2、30步观察及逐控制步轨迹；每16条命令记执行块，不重复一个专家动作16次。原任务默认及K0/K4/K8方法完全不改。有限信息策略只看当前可见≥16像素的果实、几何、颜色/叶簇提示；G为当前可见且处于地面高度附近的果实数，不给实际风落数。

| 队列 | 可额外使用的信息 | 停止规则 | 门槛 |
|---|---|---|---|
| Count(i) current | 只有当前帧及指令N | 当前树上可见果≤1或当前篮内可见果≥N则停 | ≤15/50 |
| Count(ii-a) initial_current | 第0帧+当前帧 | M−R≥N或可见篮内果≥N则停 | 与ii-g取较高值，≤15/50 |
| Count(ii-g) initial_ground | 同上，并使用当前可见地面果数G | M−R−G≥N或可见篮内果≥N则停 | 同上 |
| Count(iii) history | 完整历史及自己的放果尝试事件 | 完成第N次放果后回rest保持；实际入篮由环境验证 | ≥45/50且裁剪率<.5% |
| Count(iv) clock | 首帧+当前帧+绝对控制步t | round((t−a)/b+偏置)≥N则停 | ≤15/50 |
| Count(iv-o) clock_onset | 额外记住第一次观察到绿灯的时刻 | 将t替换为t−绿灯时刻，用相同校准批次拟合的相对模型 | 只报告，不设门槛 |
| Occluded(i) current | 只有当前帧 | 摘当前可见红果或有可见叶簇标记的果；优先y最大 | ≤15/50 |
| Occluded(ii) initial_current | 第0帧+当前帧 | 摘初始红果中当前可见的y最大果；若无可见候选，按初始红果位置从头重新查看，未保存跨放果的已完成集合 | ≥45/50 |
| Occluded(iii) history | 完整历史、成熟目标跟踪 | 原专家选择未摘成熟目标 | ≥45/50且裁剪率<.5% |

Occluded(ii)的查看使用原v3.1 pre-grasp姿态和冻结规划器；每次选果重新扫描初始位置，低层技能局部搜索不缓存以前回合进程。当前看不到不能凭空给出其当前颜色或存在状态。所有初始信息都来自第0帧可见区域。

有限信息输入不含种子、实际风落数、补果数、全局入篮数、放果历史、真实实体代数。t只给iv/iv-o，绿灯时刻只给iv-o且只能在观察到绿灯之后给。原始真值仅供独立审计，不能反传给有限信息选择器。

用户已明确选择：不增加“提前开工时钟”队列。early_start不罚仍可能留下红灯先开工再按时钟停止的风险；本次只记录该风险，既定门槛不改变。

## 6. 校准、预算与实际运行顺序

几何种子沿用4200000+16×episode，不换seed、不筛成功、不重试。dev200–249是50布局门槛样本；dev250–259只用于校准/预算检查，不计入门槛。30条独立诊断为：10条随机N/D专家校准，10条N=5/D=600专家压力检查，10条N=5/D=600一直摘不停。压力覆盖是预先定义的诊断重置参数，不修改普通v2的随机分布。

预算两条检查：10个最坏参数专家尝试都完成，最慢完成（含40步信号）距1800至少150步；10个一直摘不停的尝试都在1800之前摘下第6颗并触发多摘。若有规划/物理失败使检查无法完成，写明无法验证，不将延长预算说成能修复规划；若时间不够，提出由实测最慢耗时与最大相邻间隔构造的预算建议，然后停止，不自动改1800。

时钟在10条随机校准上以真实首次入篮序列拟合t_k=a+b·k（k从1起），OLS解a/b；偏置在这些校准回合的所有停止阈值分界点、相邻浮点和区间中择成功回合最多者，平局取绝对值最小、再取较小值。round用numpy.rint（正中取偶）。成功指首次停止恰在N次放果之后，前面未提前停。校准参考失败不算时钟成功。

iv-o使用同一批数据减去绿灯时刻后独立拟合a_o/b_o及偏置，让它代表真正“记住开工再按节奏”上界；不是把包含平均随机等待的绝对截距直接套到相对时间上。两模型均在门槛审计前保存sha256，不能根据dev200–249结果再拟合。

顺序：ep245物理回归→Count最坏参数dev250和Occluded专家dev200渲染/机制预览（均计入对应新批次）→完成两组预算压力检查→随机校准并冻结两套时钟→Count专家50→Occluded专家50→Occluded(ii)50→Occluded(i)50→Count(i)50→Count(ii-a)50→Count(ii-g)50→iv-o50→iv50。这样只报告的iv-o不会被最后时钟门槛失败跳过；仍在任何先行门槛失败时立即停下，不为了补齐后续队列而越过停止点。

每个门槛队列满50条后判定；单个预期规划/抓握失败保留在成功率分母，不能重试。任何语义、备用果、可见性或外观检查错误立即终止在途回放。全部通过也在R0停止。专家裁剪率去掉最初30步，包含红灯等待与完成保持，严格<0.5%。

## 7. 事件信号特权信息

原fruit_distance的最近“成熟且挂树”距离读取成熟度真值；在遮挡中性化后，事件帧选择可能隐含成熟度，即使距离数值不直接输入模型。新候选any_hanging只接受TCP、位置和挂树标记，不接收成熟度。它仍用几何/状态真值，不能说成完全无特权感知。两个版本均排除停放备用、风落和已摘果。

只做对照，不替用户选择最终版本。对新专家轨迹用原event_keyframes.detect_events，radius=.03m、tol=.005m，分别报告事件数及灯变绿、补果同一步的近场边沿/关键帧数。指示灯纯颜色不进入距离；补果时距TCP≥.15m，高于近场半径，但仍按实测日志报告重合。原“release”关键帧常对应脱果后距离变远，不能把名称当成真实入篮/放果时刻。两版采用哪一个留到R0审阅，之后三个模型统一使用，K0/K4/K8定义不改。

## 8. 划分、统计与权限

新登记仅声明按修订8重新生成train0–199/dev200–259；本轮只实例化dev诊断，不写训练格式、不建HDF5。val150–199不生成、不查看、不评测；val0–149全部记已用。等待段会增加示范长度并占均匀帧位，原因是堵时钟捷径，不为偏袒任何方法；后续预登记如实写明，不据此调任务。

报告给出按N分组的开工后首颗入篮时间、相邻间隔的均值/标准差/范围，补果推迟次数/步数、各条件典型失败、预算恰N但超时的计数、双相机逐步像素最大值、等效性最大差值、四项几何检查和D5只读核查。所有数字链接到新R8目录。不存在训练或数据生成的自动后续；在R0等待审阅。


## 入篮几何与回归检查

从冻结的 `fruit_picking/env.py` 核实：篮子位姿z=0.150 m为底板下表面，底板世界范围[0.150,0.156] m，篮壁/篮口范围[0.150,0.270] m。v2两任务使用 |rel_x|、|rel_y|<0.09 m且0<rel_z<0.12 m；v1父类和行为保留。读法与§16.3一致。

| 检查 | 结果 | 依据 |
|---|---|---|
| (a) 果实静置在篮底 | 入篮，通过 | geometry_unit_checks.json |
| (b) 篮子正下方地面 | 排除，通过 | geometry_unit_checks.json |
| (c) 果心高于篮口 | 排除，通过 | geometry_unit_checks.json |
| (d) ep245原命令物理重放 | 89条命令完成，无wind_basket_anomaly | ep245_native_replay.json |

(a)–(c)及v1兼容性是原生方法单元检查；(d)另有90个旧记录状态的只读重算，并完成了独立原命令物理重放。重放使用提交6eb6197的旧R0场景，只替换v2入篮下界，不用新任务重解释旧轨迹。开跑前14项CPU单元检查见[preflight.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/preflight.json)，其中“pending”是开跑前状态；物理完成证据见[ep245_native_replay.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/ep245_native_replay.json)。

D5只读扫描1200条已保存末帧：rel_z≤0且原判入篮的果实0颗、受影响回合0个。这里只支持末帧核查，不外推未保存的中间时刻；没有重跑、重判或修改D5。逐例记录：[D5_readonly_basket_audit.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/D5_readonly_basket_audit.json)。

## 预算实测（dev250–259，D=600、N=5）

| 固定压力检查 | 实测 | 门槛 | 结果 |
|---|---|---|---|
| 专家完成并保持40步 | 10/10完成；最慢1636步，余164步 | 全完成，且余量≥150步 | 通过 |
| 一直摘不停 | 10/10摘下第6颗；最迟1683步 | 全部在1800步前多摘 | 通过 |

保持预算1800，不调参、不换种子。以上是预定10个布局的实测核算，未声称穷尽全部几何。证据：[budget_check.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/budget_check.json)，对应`replays/budget_expert`、`replays/budget_nonstop`。

## 捷径与专家审计

| 任务/信息条件 | 成功 | 门槛 | 通过 | 裁剪步/检查步 | 典型失败 |
|---|---:|---|---|---:|---|
| count/history | 50/50 | ≥45/50 | True | 35/46212 | {} |
| occluded_ripe/history | 48/50 | ≥45/50 | True | 32/16742 | {'planning_or_grasp_failure': 1, 'original_fail_or_timeout': 1} |
| occluded_ripe/initial_current | 46/50 | ≥45/50 | True | 28/20177 | {'planning_or_grasp_failure': 1, 'original_fail_or_timeout': 3} |
| occluded_ripe/current | 0/50 | ≤15/50 | True | 0/63500 | {'original_fail_or_timeout': 50} |
| count/current | 0/50 | ≤15/50 | True | 35/47315 | {'overpick': 50} |
| count/initial_current | 0/50 | ≤15/50 | True | 35/46373 | {'overpick': 44, 'finish_signal': 6} |
| count/initial_ground | 0/50 | ≤15/50 | True | 35/47159 | {'overpick': 49, 'finish_signal': 1} |
| count/clock_onset | 45/50 | 只报告 | None | 35/45315 | {'finish_signal': 5} |
| count/clock | 13/50 | ≤15/50 | True | 31/44605 | {'overpick': 24, 'finish_signal': 13} |

计数(ii)取ii-a和ii-g较高者；当前两者完成数50/50、50/50。未满两队列时不伪造较高成功率。
专家裁剪率分母去掉前30步，包含红灯等待和完成保持；要求严格<0.5%。时钟iv-o无门槛。

## 补果外观与延迟

| 条件 | 比较次数 | 最低IoU | 最大平均色差/255原始值 | 最大单像素色差原始值 | 完成保持末帧比较数 | 延迟次数/步数统计 |
|---|---:|---:|---:|---:|---:|---|
| count/history | 230 | 1.0 | 0.0 | 0 | 50 | 0/{'n': 0, 'mean': None, 'std': None, 'range': None} |
| count/current | 180 | 1.0 | 0.0 | 0 | 0 | 0/{'n': 0, 'mean': None, 'std': None, 'range': None} |
| count/initial_current | 180 | 1.0 | 0.0 | 0 | 6 | 0/{'n': 0, 'mean': None, 'std': None, 'range': None} |
| count/initial_ground | 180 | 1.0 | 0.0 | 0 | 1 | 0/{'n': 0, 'mean': None, 'std': None, 'range': None} |
| count/clock_onset | 225 | 1.0 | 0.0 | 0 | 50 | 0/{'n': 0, 'mean': None, 'std': None, 'range': None} |
| count/clock | 192 | 1.0 | 0.0 | 0 | 26 | 0/{'n': 0, 'mean': None, 'std': None, 'range': None} |

平均色差门槛是原始像素值≤2，即≤2/255；最大差值只报告。无完成保持段的失败回合记N/A，另检查每次补果后的相同ROI，不将N/A记为通过。

## 专家时间统计（dev200–249）

单位为控制步；标准差为总体标准差。首颗时刻从实际灯变绿起算，相邻间隔按首次入篮记录计算。

| N | 回合 | 首颗：均值±标准差；范围 | 相邻间隔：均值±标准差；范围 |
|---|---:|---|---|
| all | 50 | 142.12 ± 15.20；126–181（n=50） | 145.43 ± 15.02；129–186（n=130） |
| 2 | 10 | 133.90 ± 5.82；126–144（n=10） | 139.10 ± 5.73；131–149（n=10） |
| 3 | 14 | 145.43 ± 16.77；130–181（n=14） | 145.89 ± 16.22；129–186（n=28） |
| 4 | 12 | 143.17 ± 14.82；127–176（n=12） | 144.31 ± 12.30；132–181（n=36） |
| 5 | 14 | 143.79 ± 16.39；126–180（n=14） | 147.05 ± 16.68；130–185（n=56） |

本批50条专家的红灯等待D：356.78 ± 151.55；30–600（n=50）；等待占前30步之后全部动作的逐回合平均比例为37.71%。这会占用均匀选帧位置，后续协议应如实记录，不据此调整任务。逐例见[summary.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/summary.json)及`replays/audit/count/history/ep*/result.json`。

## 时钟拟合与预先固定的停止规则

仅用dev250–259的10条随机N/D专家回放，38个首次入篮记录。两套参数在门槛队列前冻结。

| 模型 | a | b | 偏置 | 校准回合成功 |
|---|---:|---:|---:|---:|
| absolute | 327.468548387097 | 174.604032258065 | -0.073384940118 | 4/10 |
| relative | 11.938709677419 | 142.125806451613 | 0.000000000000 | 9/10 |

冻结文件：[clock_fit.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/clock_fit.json)；sha256=`818c31322cc3f725c8a3286c90146df82a200c218c30d6c7dfefd66051f55451`。a/b为OLS拟合，偏置按开跑前冻结的断点搜索和平局规则选择；没有在dev200–249上再拟合。

拟合及偏置选择仅用dev250–259，之后冻结。iv不给灯亮时刻；iv-o记住首次观测到的灯亮时刻、使用同一校准批次拟合的相对时间模型，只作解释。

## 事件信号特权信息检查

原信号读取成熟度真值；遮挡中性化后，K4/K8的选帧仍可能间接泄露成熟度。计数全为成熟果，两版本应一致。any_hanging不接收成熟度参数，但仍使用几何和挂树状态真值，不能宣称完全无特权信息。
指示灯只改颜色，不进入距离函数；补果改变候选几何，但生成时TCP至少距原位15cm，远于3cm近场半径。下表报告实测同一步的近场边沿和关键帧重合，不把重合当作因果证明。
| 任务/版本 | 专家回合 | 近场事件 | 灯变化边沿/关键帧 | 补果边沿/关键帧 |
|---|---:|---:|---|---|
| count/mature | 50 | 180 | 0/0 | 0/0 |
| count/any_hanging | 50 | 180 | 0/0 | 0/0 |
| occluded_ripe/mature | 50 | 127 | 0/0 | 0/0 |
| occluded_ripe/any_hanging | 50 | 127 | 0/0 | 0/0 |

两个版本目前均为诊断候选，采用哪一个留给用户R0审阅，之后三个模型统一使用；没有修改K0/K4/K8定义。

## 相机截图和可见性

count图例来自audit/dev200，N=5；按阶段完整度、episode序选择，不按成功结果筛图。
该例双相机各物体最大像素数：{'front_camera': {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 169}, 'hand_camera': {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 591}}。全部逐步像素在trace.jsonl，所有篮子/篮内果实像素须为0。

before_green，物理步0：
![count before_green front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/before_green_front_camera.png)
![count before_green hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/before_green_hand_camera.png)

after_green，物理步395：
![count after_green front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/after_green_front_camera.png)
![count after_green hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/after_green_hand_camera.png)

after_wind，物理步441：
![count after_wind front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/after_wind_front_camera.png)
![count after_wind hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/after_wind_hand_camera.png)

before_grasp，物理步455：
![count before_grasp front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/before_grasp_front_camera.png)
![count before_grasp hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/before_grasp_hand_camera.png)

release，物理步526：
![count release front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/release_front_camera.png)
![count release hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/release_hand_camera.png)

after_replenishment，物理步531：
![count after_replenishment front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/after_replenishment_front_camera.png)
![count after_replenishment hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/after_replenishment_hand_camera.png)

finish_hold，物理步1131：
![count finish_hold front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/finish_hold_front_camera.png)
![count finish_hold hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/history/ep200/finish_hold_hand_camera.png)
occluded_ripe图例来自audit/dev200，N=2；按阶段完整度、episode序选择，不按成功结果筛图。
该例双相机各物体最大像素数：{'front_camera': {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0}, 'hand_camera': {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0}}。全部逐步像素在trace.jsonl，所有篮子/篮内果实像素须为0。

opening，物理步0：
![occluded_ripe opening front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/opening_front_camera.png)
![occluded_ripe opening hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/opening_hand_camera.png)

post_gust，物理步20：
![occluded_ripe post_gust front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/post_gust_front_camera.png)
![occluded_ripe post_gust hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/post_gust_hand_camera.png)

before_grasp，物理步93：
![occluded_ripe before_grasp front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/before_grasp_front_camera.png)
![occluded_ripe before_grasp hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/before_grasp_hand_camera.png)

release，物理步167：
![occluded_ripe release front_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/release_front_camera.png)
![occluded_ripe release hand_camera](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep200/release_hand_camera.png)

## 全部门槛回放的像素最大值

| 条件 | 前置相机：篮子/篮内果/停放备用/灯 | 腕部相机：篮子/篮内果/停放备用/灯 |
|---|---|---|
| count/history | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 169} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 894} |
| occluded_ripe/history | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0} |
| occluded_ripe/initial_current | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0} |
| occluded_ripe/current | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 0} |
| count/current | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 169} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 894} |
| count/initial_current | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 169} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 894} |
| count/initial_ground | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 169} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 894} |
| count/clock_onset | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 169} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 894} |
| count/clock | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 169} | {'basket': 0, 'basket_fruit': 0, 'inactive_reserve': 0, 'lamp': 894} |

| 条件 | early_start回合 | 预算耗尽但恰好N颗入篮 |
|---|---:|---:|
| count/history | 0 | 0 |
| occluded_ripe/history | 0 | 0 |
| occluded_ripe/initial_current | 0 | 0 |
| occluded_ripe/current | 0 | 0 |
| count/current | 0 | 0 |
| count/initial_current | 0 | 0 |
| count/initial_ground | 0 | 0 |
| count/clock_onset | 0 | 0 |
| count/clock | 0 | 0 |

## 停止与产物

完整机器摘要：`/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/summary.json`。源数据、动作、有限信息输入及逐例有效性：`/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays`。
划分：`/media/fulei/fuleivla/ecm_vla/reports/split_registry_fruit_task_v2_r8_20261105.json`；val150–199仅保留，不生成、不用；val0–149排除。
用户明确选择不增加提前开工时钟队列。early_start不罚会留下提前开工再按时钟停的风险，本次如实记录，既定门槛不据此改变。
所有门槛/几何/外观/可见性/预算检查失败即停，不调参、不换种子、不自行修改任务定义；不生成训练数据、不训练、不写论文。

## 独立交叉核验与解释边界

独立核验[记录](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/independent_validation.json)：480条有效新回放、429,584帧、2,052次决策；无未审计尝试，无检查失败文件。逐帧重算v2入篮几何、有效挂树集合和两种距离信号，距离误差最大值为0；核对图像及原始记录哈希、源文件、包络计数、灯时序和补果安全距离。像素证据是在原仿真中测量，核验未新渲染或新回放。23个固定源文件和D5/D6/旧R0保护清单全部保持。

计数两种信号的距离逐帧相同；遮挡专家18,292帧中有8,428帧距离数值不同，但50条专家的完整事件区间全部相同，事件数均为127。专家只趋近成熟目标，区间一致不能证明原信号没有成熟度特权。候选any_hanging不接收成熟度，但仍读几何和挂树真值；采用哪版留给用户R0审阅，三个模型之后统一使用。

计数开工时刻时钟45/50，是按既定规则只报告的结果：记住灯亮时刻再按节奏推算，已能获得90%成功率。因此本次只确认指定门槛通过，不能声称逐次放果事件记忆是成功的必要条件。另一个已记录风险是红灯提前开工不罚，可能存在提前开工的绝对时钟策略；用户已明确保留第16节队列，本轮未增加该审计，也未改变失败定义。

计数所有条件的early_start均为0，预算耗尽且恰好N的回合均为0；两任务所有门槛回放的篮子、篮内果实及停放备用果双相机最大像素均为0。所有计数条件补果推迟次数为0，实际外观比较的最低IoU=1、平均及最大像素色差均为0。外观比较范围是非风落挂果位置的果实ROI，排除灯，不声称含机器人和地面风落果的整幅画面完全一致。未发生完成保持的失败回合，其末帧比较记N/A，逐次补果比较仍保留。

| 条件 | 典型失败证据（原结局标签保持） |
|---|---|
| occluded_ripe/history | [ep202](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep202/result.json)：planning_or_grasp_failure，t=105；[ep241](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/history/ep241/result.json)：original_fail_or_timeout，t=245 |
| occluded_ripe/initial_current | [ep202](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/initial_current/ep202/result.json)：planning_or_grasp_failure，t=105；[ep241](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/initial_current/ep241/result.json)：original_fail_or_timeout，t=405 |
| occluded_ripe/current | [ep200](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/current/ep200/result.json)：original_fail_or_timeout，t=1300；[ep201](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/occluded_ripe/current/ep201/result.json)：original_fail_or_timeout，t=1300 |
| count/current | [ep200](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/current/ep200/result.json)：overpick，t=1150；[ep201](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/current/ep201/result.json)：overpick，t=1179 |
| count/initial_current | [ep200](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/initial_current/ep200/result.json)：overpick，t=1150；[ep201](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/initial_current/ep201/result.json)：overpick，t=1179 |
| count/initial_ground | [ep200](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/initial_ground/ep200/result.json)：overpick，t=1150；[ep201](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/initial_ground/ep201/result.json)：overpick，t=1179 |
| count/clock_onset | [ep209](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/clock_onset/ep209/result.json)：finish_signal，t=515；[ep223](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/clock_onset/ep223/result.json)：finish_signal，t=1342 |
| count/clock | [ep200](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/clock/ep200/result.json)：overpick，t=1150；[ep201](/media/fulei/fuleivla/ecm_vla/results/fruit_task_v2_R0_r8_20261105/replays/audit/count/clock/ep201/result.json)：overpick，t=1179 |

遮挡专家ep202在t=105记录“fruit 3 did not detach”；ep241在t=245原环境fail=True，成熟果3在篮外z=-0.06368 m（落果），未重试，也未更改原`original_fail_or_timeout`结局标签。完整原始状态在`replays/audit/occluded_ripe/history/ep241/trace.jsonl`。

下图为开跑顺序固定的Count预算预览dev250与Occluded专家预览dev200，覆盖要求的双相机阶段；不是按成功率挑选的性能样本。原图22张路径和sha256见[清单](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/fixed_preview_sources.json)，完整分辨率见[拼图](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/fixed_preview_montage.png)。

![固定预览阶段与双相机截图](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/fixed_preview_montage.png)

运行源码提交：`d09f3660b650ed497237423265bdbd15677deb24`。R0既定门槛通过后按§16.7停止。下一步仅等待用户审阅任务定义、解释边界和事件信号选择；不自动生成训练数据、不训练、不使用val150–199，不改D5/D6或论文。

本次审阅交付文件与sha256见[review_manifest.json](/media/fulei/fuleivla/ecm_vla/reports/fruit_task_v2_R0_r8_20261105/review_manifest.json)；这是R0审阅清单，不是正式实验协议封存。
