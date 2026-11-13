# 低层修复后采摘任务：K0/K4/K8正式评测协议（修订5）

日期：2026-11-04。停点A已获用户批准，按HANDOFF §13（修订5）执行。
分类试运行通过后直接封存并执行主评测900条及无包络敏感性300条，到停点B交付；有效性失败立即停止。
封存状态、正文提交和摘要见第12节登记区；完整文件SHA256以配套fruit_lowlevel_v2_seal_20261104.json为准，运行前核验。
协议正文哈希排除封存登记区，以避免自引用；实际正文提交、分析代码和平台摘要均固定。

## 1. 问题、主假设与历史说明

在同一份新专家干净+DART数据训练、同一低层执行平台下，事件关键帧记忆是否提高整任务成功率：
H1：K4−K0>0；H2：K8−K0>0。两个任务分别比较，合计4个单侧检验构成唯一的主假设族，Holm校正，α=.05。
不会沿用旧协议的12个百分点界限、旧开发门、旧地板判定或旧val范围。

专家版本canonical_v3.1_20261102只影响训练数据。G1已通过：合计成功41/80、首颗61/80、执行与原始输出腕偏离均0/80、工程错误0/80。
遮挡任务dev2/20与G1的12/40样本都小、置信区间重叠，按抽样波动处理，不作解读。
包络v2在G1至少一个关节被裁剪11185/49470步（约22.6%），专家动作626/164639步（约0.38%）。
策略原始输出腕偏离也是0/80；包络主要限速，取舍已在G1前固定，不因本轮性能改变。
正式报告逐条件列裁剪率，并用无包络敏感性分析检查主对照方向是否依赖包络。
旧正式结果results/fruit_dart_kf_eval_20261027只作历史背景，不作新旧统计比较，不用于阈值选择。

## 2. 数据、记忆与训练

直接复用data/finetune_kf/fruit_dart_v2_rm75b_train：400回合、164639物理帧，12000帧为observe历史，152639监督样本。
干净train0–99与DART100–199均为新专家生成；保留完整时间结构，release/done监督不删；canonical settle2。
本阶段不重新构建数据、不重新计算统计。三条件共用统计SHA256：
`6da229fba3c9c2564ddacdc2f7093cb69c672a0ddd651cb6b18665ec57e08568`。
数据来源及所有400份distance.npy摘要记录在results/fruit_lowlevel_v2_stage5_20261103/plan.json。

K0：runs/fruit_k0v2_lowlevel_rescue_20261102/ckpt/mme_vla_suite/k0/11999，直接复用，不重训。
K4/K8：runs/fruit_lowlevel_v2_k4k8_20261103/ckpt/mme_vla_suite/k4/11999及k8/11999。
从79999初始化，种子42，12000步，batch64、warmup500、peak_lr2.5e-5、decay_lr2.5e-6、EMA.999，GPU0–7/FSDP8，优化器与编码器冻结设置同K0。
逐项复制K0配置；只有arm名称、输出目录和sampler.events有差别，唯一有效训练差别为记忆选帧。K4与K8依次训练，各自T1通过，固定11999，不挑检查点。
训练专用配置文件fruit_lowlevel_v2_training_20261103.md及运行protocol.json只记录训练来源；正式评测单独按本协议封存。

三条件都是16帧记忆；事件槽上限分别0/4/8，r=.03m、tol=.005m，沿用同一个event_keyframes.select函数。
距离为TCP到最近仍挂树的可摘成熟果实，距离只选择帧，不进入模型输入；没有可摘果实记10m。选帧只用决策当时可见历史。
距离近区事件提供contact和release两帧；后者是首次离开近区，本采摘任务中作为detach代理，真值脱果另计，算法不变。

## 3. 三条件共用执行平台及SHA256

基线A0，加G1前已确定的包络v2；步长16，obs_horizon16，前30个物理控制步固定保持初始关节角并张开夹爪，第31步交给策略，30步计入1300总预算。
采用原官方EpisodeEvaluator、同一个fruit_rescue_platform与记录器；K4/K8客户端仅扩展条件/模型种子路由。
包络v2：新专家状态[.5%,99.5%]各外扩.1rad，限速为1.5×P99.9|joint_action−joint_state|；夹爪不裁剪。
交集为空时先裁剪到范围、再按限速向范围移动，记guard_projection，属于平台事件，不是工程错误。原前30步保持不经包络改写。
guard_v2.json与guard_choice.json沿用G1，不重新估计或重新选择。

平台逐文件完整SHA256及清单摘要见下方停点A记录，以及results/fruit_lowlevel_v2_stage5_20261103/platform_identity.json。
清单包含包络/选择文件、专家统计、物理前缀实现、客户端、审计、服务端、距离/选帧函数和官方评测入口。封存时写入本协议并固定，运行前逐项验证。
K0/K4/K8检查点manifest、元数据、归一化统计摘要也在停点A记录；只使用各自11999。

## 4. 开发集有效性、冒烟与停点

K4、K8各dev200–219×两个任务，种子7，各40条；不使用val。平台与G1完全相同。
有效性条件：工程错误0；40/40回合的记忆审计全部通过，包括输入图像摘要、物理/策略时钟、因果可见性、原始源帧、位置编码、padding/mask与选帧重算。
逐回合报告contact数、detach代理关键帧数、实际脱果数、事件槽/有效槽比例；事件统计同时列全部物理帧与最后一次决策前已观察事件。
事件槽分母为所有决策的有效帧槽，不含padding；另列均匀槽偶然选到事件帧的比例。
K4/K8全程没有事件槽的回合须逐例解释：在任何决策前都未进入3cm近区，或首次接触发生在末次决策之后。记录最小距离和原始事件证据，不因此添加性能门槛。
若决策时已经观察到事件却全程没有事件槽，或任何选帧/事件重算不符，属于有效性失败，立即停止。其他已在执行中的回合只收尾，不再发新回合。
性能全部记录，绝不据此调整模型、统计、触发阈值、步长、包络或训练超参数。
冒烟只用dev200两个任务×三个条件，共6条、种子7，同任务三个条件在同一个服务会话；不接触val布局。
上述停点A流程已完成，用户在修订5中批准；分类试运行通过后按§13.4直接封存并正式评测，不再等待A。

## 5. 正式评测集与配对（停点A批准并封存后执行）

val100–149，来自reports/split_registry_20261102.json的冻结布局；模型种子7/8/9。
每任务、每条件50布局×3种子=150组；两个任务×三个条件共900回合。按“任务、布局×模型种子”配对，共享环境种子及每次条件开始时的策略随机数重置。
每个模型种子一个服务进程装载三个条件，同组恰在同一服务会话；三个种子分开服务。条件顺序按固定种子20261103生成的置换安排，记录实际运行顺序。
不得用val预览、调参、检查点挑选或冒烟。任务失败、超时、掉果及所有工程错误均保留，不删结果或填补缺组。
客户端单次进程墙钟上限1800秒，服务装载上限1800秒；与1300物理控制步预算无关，墙钟超时仍先核查已有结局。
工程重跑仅限尚无任务结局的崩溃/进程超时/审计无法运行等工程中断，同一环境种子、模型种子与条件，最多重跑2次（含初次最多3次）。
已经产生success/fail/timeout的回合一律不重跑；即使result.json缺失，也从控制日志或最后物理真值识别已有结局。已有结局后审计失败直接判有效性失败。
所有尝试（包括最终采纳尝试）原样保存在superseded/并列入报告；main/sensitivity仅链接到通过审计的采纳尝试。2次重跑后仍为工程错误就停止。
同组内始终要求三个条件共享服务会话；若已有条件结束后服务进程丢失，不能重跑已结束条件来恢复配对，直接有效性失败。
客户端无结局崩溃可在同一活服务上重跑；服务崩溃且本组尚无已结束条件时，可按相同配置重启服务用于剩余重跑机会。除此不自行补救。
有效性门槛：工程错误0、全部900回合齐全、每组条件/布局/种子/会话正确、逐决策记忆审计0不一致。任何门槛失败时主结果只作工程诊断，不报告确证性结论。
封存后不修改协议或分析定义；确需改变时停止并报告，不把修改后的分析当作原计划结果。

## 6. 主分析与统计

主终点为result.json整任务success二值（统一分析器核对），分别计算每个任务K4−K0、K8−K0的成功率差，单位百分点。
每个配对差保留同一布局的三个模型种子。对50个布局按布局聚类、有放回重采样10000次，每次保留三个种子及所有条件，共同计算差值；报告两侧95%百分位bootstrap区间（2.5%、97.5%）。不把150组当作150个独立布局。
单侧p值使用布局聚类配对随机化：先对每个布局求三个种子的平均差，在零假设下以布局为单位整体交换两个条件（等价于对该布局差值取±号）。
100000次符号置换，以平均差≥实测差的次数b计算p=(b+1)/(100000+1)，含等号；差值全0时p=1。
统计基础种子20261103；各任务/对比用其名称的SHA256生成固定子流，不受运行顺序影响；保存bootstrap和置换摘要。
4个p值合并做一次Holm：从小到大排序，校正p为截至当前的max[(4−排序位置+1)×p]，截断到1并映射回原检验。
只有观测差>0且Holm校正后p≤.05，才判对应主假设获得支持；置信区间与p值都完整报告，不用次要结果替换主结论。
不事后合并任务或种子、改变主假设族，也不使用历史结果作为对照。

## 7. 预定义次要指标和互斥失败类型

统一报告HANDOFF §5及§9.5全部指标，合计与分任务均列分子/分母，包括正确/错误入篮果实总数、计数恰好N、放下未成功子集冻结、掉果率（outcome=fail）及超时率。
按清单中的每个配额N分别列样本数和结果，不根据性能合并N。次要结果只描述，不进入主假设族。

以下定义在封存前写死，以逐帧真值与动作计算。每个有效回合恰好归入成功、记忆类或操作类之一；任何记忆错误优先归入记忆类，即使也发生操作错误，保留所有操作标签。不从模型动作推断不可观察的内部意图。

- 计数任务的记忆类：任一时刻入篮数>N；或首次达到N之后，新果实发生脱离，或产生一次新的近果事件（该事件start晚于首次达到N，近区阈值仍为3cm）。
- 结构性说明：环境evaluate的计数分支在入篮数恰好N时立即判成功，FruitEnvRunner与官方循环随即终止。上述“超过N”和“达到N后继续取果”结构上不会发生；保留条款用于检查，非零按环境终止异常处理，归invalid并作为工程诊断停止，不当作记忆效果。计数任务实际记忆类仅由下述提前停止代理构成。
- 计数任务的提前停止代理：仅对outcome=timeout且最终入篮不足N。最后连续200个控制步，7臂关节相邻目标变化均≤.005rad、夹爪张开、TCP始终距该窗口起点≤5cm、无新增脱离/入篮；窗口起点还有成熟果实挂树，且当时总预算至少剩余200步。满足时归记忆类，标签count_premature_stop_proxy。
- 该代理使“剩余时间充足”和“停止取果”可计算；它仍可能混入操作停滞，报告中明确这一限制，并单独列该标签数量，不把分类直接解释成记忆的因果机制。
- 遮挡任务的记忆类：任一帧wrong_in_basket>0（入篮果不在目标集合）；或任何不成熟果实脱离，包含之后掉落的情况。
- 果实只有连续抓握满DETACH_STEPS（当前为2）控制步才会脱落；未成熟果脱落必经双指抓握，不能被碰掉。分类仍归记忆类。新增只描述、不改变分类的adjacent_misgrasp_possible子标签：在该未成熟果这次成功连续抓握的第一步，它距最近的仍挂树的目标成熟果实中心<.04m。
- 起点为首次脱果帧 − (DETACH_STEPS−1)（当前为脱果前一帧）；逐步日志的该帧给出果实位置和挂树目标集合。无候选目标时距离记null、子标签false；严格小于4cm，等于不触发。逐回合和逐未成熟果事件分别报告计数，作为可能操作误抓的限制说明，不更改互斥归类。
- 操作类：没有上述记忆错误且整任务未成功的有效回合。标签包括never_detached、near_miss、掉果（outcome=fail）、仍未完成的操作/停滞超时；保留原始标签。近失阈值与统一分析器相同，不另定义。
- 结果为success且无记忆错误归成功；工程/审计无效归invalid并使正式有效性门槛失败，不能塞进两种机制类。

分别报告记忆类和操作类在三个条件中的场次率（分母为全部有效回合），及失败内部占比（分母为分类中的记忆类+操作类回合数），并报告K4−K0、K8−K0、K8−K4的差异与布局聚类95%区间。
若result.success伴有记忆错误，主成功率仍按原始success统计，机制分类优先记忆类，并单列这类冲突回合数，避免混淆分母。
基础分类代码scripts/fruit_lowlevel_v2_metrics.py保持停点A原样；scripts/fruit_lowlevel_v2_metrics_r5.py逐回合调用它，追加修订5的结构异常处理及描述子标签。两版及统一指标代码均纳入分析摘要。机制检查不进入Holm族，不据此重新命名主终点。

## 8. 诊断指标与无包络敏感性分析

每个条件及任务报告包络裁剪率：任一关节和7个逐关节裁剪步数/总控制步数，分母包含前30步物理保持；另列guard_projection事件数。
报告leave_home_step、post_release_resume_steps（观察到的恢复中位数及右删失分子/分母）、腕偏离执行值/原始输出；新专家q7中心固定为−1.5764960646629333rad。全部条件使用同一中心，不重新估计。

事件召回诊断（只报告、不调阈值）：每条件、每任务列“距离代理脱离事件数/真值脱果数”，取完整物理回合，不限已被服务观察的前缀。它是事件数比，不做匹配、不截断到1；分母为0记无定义。停点A开发集K4为80/91、K8为86/95。

敏感性分析非确证性，不进入Holm：val100–149，仅模型种子7，K0/K4/K8各两任务100回合，合计300回合。只关闭包络，其余平台、检查点、配对、步长16、30步保持及总预算1300不变。
预先报告每任务K4−K0、K8−K0、K8−K4三组差异方向与布局聚类bootstrap10000次95%区间，与包络开启主分析对照。
包络关闭裁剪率按定义为0，仍记录原始动作与执行动作、超限/偏离及所有错误；不得因敏感性结果改变主平台或主结论标准。敏感性也须全部组完整、工程错误0、审计一致，否则只报告有效性失败。

## 9. 交付、封存与停点

停点A交付：K4/K8各自T1、40条有效性完整指标、逐回合事件统计、6条dev200冒烟和本草案。平台与模型完整SHA256如下方记录。
停点A已审阅通过；按§13.4完成条款修改及dev分类试运行后直接封存，保存正文提交、协议完整文件SHA256、分析与平台清单SHA256及全部依赖，运行前逐项验证。封存后按原规则运行900+300条，出现有效性失败就停止汇报。
正式评测完成后在停点B交付results/fruit_lowlevel_v2_eval_2026110X/REPORT.md，包括配对差、95%区间、Holm校正p、全部次要/诊断指标、失败类型差异和敏感性分析，按§6写15行内摘要；不写进论文。

## 10. 历史停点A结果与冒烟记录（以下为当时尚未封存的原始交付）（运行后自动填写，以上定义不变）

<!-- STOP_A_RESULTS_BEGIN -->

阶段5停点A已到达：K4/K8训练、有效性和dev200冒烟全部完成。
K0直接复用11999；K4/K8已依次从79999完成12000步，数据、统计、种子42及8卡FSDP均与K0相同。
T1：K0=True；K4=True；K8=True。
有效性：K4工程错误0/40 (0.00%)、记忆审计40/40；K8工程错误0/40 (0.00%)、记忆审计40/40。
协议为未封存草案；没有访问val布局或启动正式/敏感性评测。按§12.5停点A等待用户审阅。

## 训练与T1

来源：`/media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/plan.json`、`protocol.json`、`arms/`及`gate_t1_k4.json`、`gate_t1_k8.json`。
K0来源：`/media/fulei/fuleivla/ecm_vla/runs/fruit_k0v2_lowlevel_rescue_20261102/gate_t1_k0.json`；数据/统计/源码保全摘要见`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_stage5_20261103/plan.json`。

| 条件 | 日志区间 | 全部有限 | 首/末区间损失 | 统计匹配 |
|---|---:|---|---|---|
| K0 | 240/240 | True | 0.1543/0.0029 | True |
| K4 | 240/240 | True | 0.1543/0.0029 | True |
| K8 | 240/240 | True | 0.1543/0.0029 | True |

## 开发集有效性与性能记录

只用dev200–219、两个任务、种子7；性能只记录，不调整设置。每项完整统一指标均在配套summary.json/validity/<条件>/performance中，含合计与分任务分母。

| 条件/任务 | 成功 | 首颗入篮 | 腕偏离执行/输出 | 工程错误 |
|---|---:|---:|---|---:|
| K4/合计 | 18/40 (45.00%) | 31/40 (77.50%) | 0/40 (0.00%)/0/40 (0.00%) | 0/40 (0.00%) |
| K4/FruitCount | 13/20 (65.00%) | 18/20 (90.00%) | 0/20 (0.00%)/0/20 (0.00%) | 0/20 (0.00%) |
| K4/FruitOccludedRipe | 5/20 (25.00%) | 13/20 (65.00%) | 0/20 (0.00%)/0/20 (0.00%) | 0/20 (0.00%) |
| K8/合计 | 18/40 (45.00%) | 31/40 (77.50%) | 0/40 (0.00%)/0/40 (0.00%) | 0/40 (0.00%) |
| K8/FruitCount | 14/20 (70.00%) | 20/20 (100.00%) | 0/20 (0.00%)/0/20 (0.00%) | 0/20 (0.00%) |
| K8/FruitOccludedRipe | 4/20 (20.00%) | 11/20 (55.00%) | 0/20 (0.00%)/0/20 (0.00%) | 0/20 (0.00%) |

## 逐回合事件触发

contact为原算法近区事件；detach关键帧为原算法release，即第一次离开近区，是脱果代理。真值实际脱果另列；不把代理与物理事件混同。事件槽占比的分母为所有决策的有效帧槽，排除padding；另报告均匀帧偶然包含的事件帧。事件选择逐决策按当时可见历史重算。

| 条件/任务/布局 | contact | detach代理 | 实际脱果 | 事件槽/有效槽 | 零事件解释 |
|---|---:|---:|---:|---:|---|
| K4/FruitCount/200 | 3 | 3 | 3 | 72/416 (17.31%) | — |
| K4/FruitCount/201 | 2 | 2 | 2 | 42/384 (10.94%) | — |
| K4/FruitCount/202 | 5 | 5 | 5 | 151/752 (20.08%) | — |
| K4/FruitCount/203 | 2 | 2 | 2 | 36/272 (13.24%) | — |
| K4/FruitCount/204 | 3 | 3 | 3 | 104/544 (19.12%) | — |
| K4/FruitCount/205 | 4 | 4 | 5 | 137/736 (18.61%) | — |
| K4/FruitCount/206 | 2 | 2 | 2 | 60/464 (12.93%) | — |
| K4/FruitCount/207 | 1 | 1 | 1 | 2/80 (2.50%) | — |
| K4/FruitCount/208 | 2 | 2 | 2 | 22/224 (9.82%) | — |
| K4/FruitCount/209 | 2 | 2 | 2 | 42/304 (13.82%) | — |
| K4/FruitCount/210 | 3 | 3 | 3 | 80/464 (17.24%) | — |
| K4/FruitCount/211 | 0 | 0 | 1 | 0/80 (0.00%) | No distance<=0.03m event was observed before a policy decision; the policy did not reach the near region, or its first contact occurred only after the final decision. |
| K4/FruitCount/212 | 5 | 5 | 5 | 142/720 (19.72%) | — |
| K4/FruitCount/213 | 2 | 2 | 2 | 71/512 (13.87%) | — |
| K4/FruitCount/214 | 2 | 2 | 3 | 38/432 (8.80%) | — |
| K4/FruitCount/215 | 2 | 2 | 2 | 44/336 (13.10%) | — |
| K4/FruitCount/216 | 5 | 5 | 5 | 121/608 (19.90%) | — |
| K4/FruitCount/217 | 4 | 4 | 4 | 106/544 (19.49%) | — |
| K4/FruitCount/218 | 2 | 2 | 2 | 51/384 (13.28%) | — |
| K4/FruitCount/219 | 3 | 3 | 4 | 100/544 (18.38%) | — |
| K4/FruitOccludedRipe/200 | 2 | 2 | 2 | 60/464 (12.93%) | — |
| K4/FruitOccludedRipe/201 | 2 | 2 | 3 | 85/496 (17.14%) | — |
| K4/FruitOccludedRipe/202 | 1 | 1 | 1 | 152/1280 (11.88%) | — |
| K4/FruitOccludedRipe/203 | 1 | 1 | 2 | 20/208 (9.62%) | — |
| K4/FruitOccludedRipe/204 | 1 | 1 | 1 | 150/1280 (11.72%) | — |
| K4/FruitOccludedRipe/205 | 1 | 1 | 2 | 130/1104 (11.78%) | — |
| K4/FruitOccludedRipe/206 | 3 | 3 | 3 | 116/592 (19.59%) | — |
| K4/FruitOccludedRipe/207 | 1 | 1 | 1 | 154/1280 (12.03%) | — |
| K4/FruitOccludedRipe/208 | 0 | 0 | 1 | 0/272 (0.00%) | No distance<=0.03m event was observed before a policy decision; the policy did not reach the near region, or its first contact occurred only after the final decision. |
| K4/FruitOccludedRipe/209 | 2 | 2 | 3 | 125/720 (17.36%) | — |
| K4/FruitOccludedRipe/210 | 2 | 2 | 2 | 284/1280 (22.19%) | — |
| K4/FruitOccludedRipe/211 | 1 | 1 | 1 | 154/1280 (12.03%) | — |
| K4/FruitOccludedRipe/212 | 0 | 0 | 1 | 0/64 (0.00%) | No distance<=0.03m event was observed before a policy decision; the policy did not reach the near region, or its first contact occurred only after the final decision. |
| K4/FruitOccludedRipe/213 | 0 | 0 | 1 | 0/64 (0.00%) | No distance<=0.03m event was observed before a policy decision; the policy did not reach the near region, or its first contact occurred only after the final decision. |
| K4/FruitOccludedRipe/214 | 1 | 1 | 1 | 4/96 (4.17%) | — |
| K4/FruitOccludedRipe/215 | 2 | 2 | 2 | 56/496 (11.29%) | — |
| K4/FruitOccludedRipe/216 | 0 | 0 | 0 | 0/1280 (0.00%) | No distance<=0.03m event was observed before a policy decision; the policy did not reach the near region, or its first contact occurred only after the final decision. |
| K4/FruitOccludedRipe/217 | 3 | 3 | 3 | 107/768 (13.93%) | — |
| K4/FruitOccludedRipe/218 | 1 | 1 | 1 | 149/1280 (11.64%) | — |
| K4/FruitOccludedRipe/219 | 2 | 2 | 2 | 62/464 (13.36%) | — |
| K8/FruitCount/200 | 3 | 3 | 3 | 78/400 (19.50%) | — |
| K8/FruitCount/201 | 3 | 3 | 3 | 87/448 (19.42%) | — |
| K8/FruitCount/202 | 5 | 5 | 5 | 223/752 (29.65%) | — |
| K8/FruitCount/203 | 2 | 2 | 2 | 71/512 (13.87%) | — |
| K8/FruitCount/204 | 3 | 3 | 3 | 72/384 (18.75%) | — |
| K8/FruitCount/205 | 4 | 4 | 4 | 480/1280 (37.50%) | — |
| K8/FruitCount/206 | 2 | 2 | 2 | 54/416 (12.98%) | — |
| K8/FruitCount/207 | 2 | 2 | 2 | 56/416 (13.46%) | — |
| K8/FruitCount/208 | 3 | 3 | 3 | 85/432 (19.68%) | — |
| K8/FruitCount/209 | 2 | 2 | 2 | 44/320 (13.75%) | — |
| K8/FruitCount/210 | 3 | 3 | 3 | 79/416 (18.99%) | — |
| K8/FruitCount/211 | 2 | 2 | 2 | 43/496 (8.67%) | — |
| K8/FruitCount/212 | 5 | 5 | 5 | 247/880 (28.07%) | — |
| K8/FruitCount/213 | 2 | 2 | 2 | 66/496 (13.31%) | — |
| K8/FruitCount/214 | 1 | 1 | 2 | 102/880 (11.59%) | — |
| K8/FruitCount/215 | 1 | 1 | 1 | 152/1280 (11.88%) | — |
| K8/FruitCount/216 | 4 | 4 | 4 | 508/1280 (39.69%) | — |
| K8/FruitCount/217 | 4 | 4 | 4 | 104/464 (22.41%) | — |
| K8/FruitCount/218 | 2 | 2 | 2 | 49/368 (13.32%) | — |
| K8/FruitCount/219 | 4 | 4 | 4 | 121/528 (22.92%) | — |
| K8/FruitOccludedRipe/200 | 1 | 1 | 2 | 26/272 (9.56%) | — |
| K8/FruitOccludedRipe/201 | 3 | 3 | 3 | 79/432 (18.29%) | — |
| K8/FruitOccludedRipe/202 | 1 | 1 | 1 | 153/1280 (11.95%) | — |
| K8/FruitOccludedRipe/203 | 2 | 2 | 2 | 42/352 (11.93%) | — |
| K8/FruitOccludedRipe/204 | 1 | 1 | 1 | 4/80 (5.00%) | — |
| K8/FruitOccludedRipe/205 | 2 | 2 | 2 | 79/592 (13.34%) | — |
| K8/FruitOccludedRipe/206 | 1 | 1 | 1 | 152/1280 (11.88%) | — |
| K8/FruitOccludedRipe/207 | 1 | 1 | 1 | 154/1280 (12.03%) | — |
| K8/FruitOccludedRipe/208 | 0 | 0 | 1 | 0/608 (0.00%) | No distance<=0.03m event was observed before a policy decision; the policy did not reach the near region, or its first contact occurred only after the final decision. |
| K8/FruitOccludedRipe/209 | 2 | 2 | 2 | 260/1280 (20.31%) | — |
| K8/FruitOccludedRipe/210 | 3 | 3 | 3 | 174/1264 (13.77%) | — |
| K8/FruitOccludedRipe/211 | 1 | 1 | 2 | 42/384 (10.94%) | — |
| K8/FruitOccludedRipe/212 | 3 | 3 | 3 | 108/672 (16.07%) | — |
| K8/FruitOccludedRipe/213 | 1 | 1 | 1 | 4/96 (4.17%) | — |
| K8/FruitOccludedRipe/214 | 2 | 2 | 2 | 41/528 (7.77%) | — |
| K8/FruitOccludedRipe/215 | 2 | 2 | 3 | 128/640 (20.00%) | — |
| K8/FruitOccludedRipe/216 | 1 | 1 | 2 | 43/400 (10.75%) | — |
| K8/FruitOccludedRipe/217 | 0 | 0 | 1 | 0/368 (0.00%) | No distance<=0.03m event was observed before a policy decision; the policy did not reach the near region, or its first contact occurred only after the final decision. |
| K8/FruitOccludedRipe/218 | 1 | 1 | 2 | 113/992 (11.39%) | — |
| K8/FruitOccludedRipe/219 | 1 | 1 | 2 | 136/1152 (11.81%) | — |

逐决策证据：`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_stage5_20261103/validity/<条件>/<任务>/ep<ID>/event_statistics.json`；汇总和所有零事件解释见相邻`event_summary.json`。

## 冒烟（dev200）

| 条件 | 完整回合 | 记忆审计 | 审计决策 | 工程错误 |
|---|---:|---:|---:|---:|
| K0 | 2/2 | 2/2 | 31 | 0/2 (0.00%) |
| K4 | 2/2 | 2/2 | 40 | 0/2 (0.00%) |
| K8 | 2/2 | 2/2 | 42 | 0/2 (0.00%) |

冒烟来源：`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_stage5_20261103/smoke`；每个任务的三个条件在同一服务会话内，环境种子及模型随机数重置相同。

## 平台与模型SHA256

| 文件 | SHA256 |
|---|---|
| /media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_stage5_20261103/platform_identity.json | 0aec2ee6035113edb0af802335ef384a56ff4521f47c4356318b3f929fe7f8cb |
| /media/fulei/fuleivla/ecm_vla/results/fruit_k0v2_gate_20261102/guard_v2.json | 899c15bbd9927396b68cac063ab8b72dd2ef0f654b58bc3a4a7d79efe350a092 |
| /media/fulei/fuleivla/ecm_vla/results/fruit_k0v2_gate_20261102/guard_choice.json | e7ae2ab456885d83fa5030b8065022f0b6c96e0b02b92344f8e50ff1d613466b |
| /media/fulei/fuleivla/ecm_vla/results/fruit_k0v2_gate_20261102/expert_profile.json | fc6a547c24e89a12c73765022e4365d89d310595288408142bf7f16fd190f56a |
| /media/fulei/fuleivla/ecm_vla/src/memory_management/fruit_rescue_platform.py | 62131ece3a41995f0cd3e4e05c24a549e9dc01e46a4639450bbf294fc05120f3 |
| /media/fulei/fuleivla/ecm_vla/scripts/evaluate_fruit_rescue_episode.py | cbf0cd6887ef97ed70449a24ca19645184c09cddca6eca87510c719d1caa0300 |
| /media/fulei/fuleivla/ecm_vla/scripts/evaluate_fruit_lowlevel_v2_episode.py | 57e2d5958c8e2930221614a805685dde04defa0f7cdd4ca947c12217628abd8c |
| /media/fulei/fuleivla/ecm_vla/scripts/audit_fruit_rescue_memory.py | 2b9e3f2e0e8d47669e73934fc60c0d7f18fdd9a38426c803e7080b4aeb0c5a26 |
| /media/fulei/fuleivla/ecm_vla/scripts/run_fruit_diagnosis.py | a412d00e621aa65ca72c9a3131a6ddc01a610007aef9cf3c1841a7093193d585 |
| /media/fulei/fuleivla/ecm_vla/scripts/serve_teacher_policy.py | 6d2043b30b3ce5775eef598867dc4bb6fe267e4a0feb97e120bfb6490b305f07 |
| /media/fulei/fuleivla/ecm_vla/src/memory_management/fruit_eval.py | d567b0ec5cc8dd778600c23ed45d3b8ee59616ad5cdbee6c8383297836d886d8 |
| /media/fulei/fuleivla/ecm_vla/src/memory_management/fruit_distance.py | df4af894320d411089ab405e9208156e6e089a51b8a56337a3c24f6ba264d592 |
| /media/fulei/fuleivla/ecm_vla/src/memory_management/event_keyframes.py | d554086f2112a3c443879f4e81164b770e941fa78d8439af7b2a280080c76161 |
| /media/fulei/fuleivla/ecm_vla/src/memory_management/validation/teacher_oracle.py | 5d935f63ccefb06b1bd9be30220806c49621d5f3ce480f1f4af7a9112b881f8a |
| /media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/examples/robomme/eval.py | ae6b2d4f451476113c95e5356e75271b8a1db3f3c65b5b0f5a082b200acee349 |
| /media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/examples/robomme/subgoal_predictor.py | ad982b1434d10558fce1f8acf0a9825ed38bb178dc68693584c0697e323e0ae3 |
| /media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/scripts/serve_policy.py | 8174f5ba7dd8a0e84c0ad39315ec6617c1f30a1911b154c590f5d486a620163f |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_k0v2_lowlevel_rescue_20261102/ckpt/mme_vla_suite/k0/11999/params/manifest.ocdbt | ab4732b24bc9085c1772fa715d7181f30cf5d25618d4eb67325071b25324b24c |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_k0v2_lowlevel_rescue_20261102/ckpt/mme_vla_suite/k0/11999/assets/robomme/norm_stats.json | 6da229fba3c9c2564ddacdc2f7093cb69c672a0ddd651cb6b18665ec57e08568 |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_k0v2_lowlevel_rescue_20261102/ckpt/mme_vla_suite/k0/11999/_CHECKPOINT_METADATA | cc3fdb97b6a3e33e787177d27ee8252d5e19c50a703f2bb37a61bc5e2f5c5ce8 |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/ckpt/mme_vla_suite/k4/11999/params/manifest.ocdbt | a8e10c40048026939b558d0250d57b04711f003737f119c2bcdd5a7e08e68aef |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/ckpt/mme_vla_suite/k4/11999/assets/robomme/norm_stats.json | 6da229fba3c9c2564ddacdc2f7093cb69c672a0ddd651cb6b18665ec57e08568 |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/ckpt/mme_vla_suite/k4/11999/_CHECKPOINT_METADATA | c2aad80e3062d9697cf11ca0eaf92d39f5e464cccf471650cb700f43a98b7978 |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/ckpt/mme_vla_suite/k8/11999/params/manifest.ocdbt | c9ea8c0808a312c69698086e9c7a1c23de996a27086757e8d2fbfb6da5a3f258 |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/ckpt/mme_vla_suite/k8/11999/assets/robomme/norm_stats.json | 6da229fba3c9c2564ddacdc2f7093cb69c672a0ddd651cb6b18665ec57e08568 |
| /media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/ckpt/mme_vla_suite/k8/11999/_CHECKPOINT_METADATA | ee33b1a83ba05cec2b4df1038fdc8ae1e37e42086f8479fdca60d6fcfaf2cf7f |

## 决策与下一步

按§12.5到达停点A，交付T1、有效性、冒烟及协议草案；等待用户审阅后再封存。
协议：`/media/fulei/fuleivla/ecm_vla/reports/fruit_lowlevel_v2_protocol_20261103.md`；目前未封存。主检验4项Holm、布局聚类bootstrap10000次、失败分类和不开包络敏感性分析均已预定义。

当前未执行val100–149；未启动正式900条或敏感性300条评测，未写入论文。

## 独立复核与事件汇总

来源：`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_stage5_20261103/independent_validation.json`；逐项核对训练配方、来源摘要、检查点、全部86条原始轨迹，重新计算3236次记忆/事件决策审计及全部统一指标，与自动产物一致。
K4有效性完成时间早于K8训练启动；每模型仅有11999检查点。411份数据/K0保全文件和32份运行来源摘要保持一致。协议在启动前写下的全部定义与准备提交324134f相同，仅补入实测证据。

| 条件 | contact | detach代理 | 实际脱果 | 事件槽/有效槽 | 零事件回合 |
|---|---:|---:|---:|---:|---:|
| K4 | 80 | 80 | 91 | 3229/23568 (13.70%) | 5/40 |
| K8 | 86 | 86 | 95 | 4459/26400 (16.89%) | 2/40 |

零事件回合的具体解释（全物理帧最小距离，不只看决策前缀）：

| 条件/任务/布局 | 最小距离cm | 实际脱果数 | 原因 |
|---|---:|---:|---|
| K4/FruitCount/211 | 3.702202 | 1 | 全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。 |
| K4/FruitOccludedRipe/208 | 4.504099 | 1 | 全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。 |
| K4/FruitOccludedRipe/212 | 3.404738 | 1 | 全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。 |
| K4/FruitOccludedRipe/213 | 3.456067 | 1 | 全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。 |
| K4/FruitOccludedRipe/216 | 3.296567 | 0 | 全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。 |
| K8/FruitOccludedRipe/208 | 3.994397 | 1 | 全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。 |
| K8/FruitOccludedRipe/217 | 3.004431 | 1 | 全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。 |

这些回合没有已观察到事件却漏选事件槽的证据；不改变3cm阈值或选帧设置。脱果代理与实际脱果可能不同，二者均保留。

## 全部§5及§9.5指标

每个条件均列合计与分任务；这里的性能只作记录，未用来调整任何设置。

### all

| 指标 | K4 | K8 |
|---|---:|---:|
| success | 18/40 (45.00%) | 18/40 (45.00%) |
| fail | 15/40 (37.50%) | 15/40 (37.50%) |
| timeout | 7/40 (17.50%) | 7/40 (17.50%) |
| engineering_error | 0/40 (0.00%) | 0/40 (0.00%) |
| first_target_in_basket | 31/40 (77.50%) | 31/40 (77.50%) |
| never_detached | 1/40 (2.50%) | 0/40 (0.00%) |
| near_miss | 1/40 (2.50%) | 0/40 (0.00%) |
| wrist_over_limit | 0/40 (0.00%) | 0/40 (0.00%) |
| wrist_deviation_executed | 0/40 (0.00%) | 0/40 (0.00%) |
| wrist_deviation_proposed | 0/40 (0.00%) | 0/40 (0.00%) |
| freeze_200 | 0/40 (0.00%) | 0/40 (0.00%) |
| freeze_200_basket_unsuccessful | 0/15 (0.00%) | 0/18 (0.00%) |
| count_exact | 13/20 (65.00%) | 14/20 (70.00%) |
| correct_in_basket（果实总数/有数据回合） | 76/40 | 80/40 |
| wrong_in_basket（果实总数/有数据回合） | 0/40 | 2/40 |
| start_idle中位步 | 31.0 | 31.0 |
| leave_home_step中位步 | 36.0 | 36.0 |
| post_release_resume_steps中位步 | 15.0 | 15.0 |
| 从未离家 | 0/40 (0.00%) | 0/40 (0.00%) |
| 入篮后未恢复（右删失） | 18/76 (23.68%) | 20/82 (24.39%) |
| 任一关节裁剪/全部控制步 | 6634/24452 (27.13%) | 7374/27321 (26.99%) |
| 关节1裁剪/全部控制步 | 4665/24452 (19.08%) | 5592/27321 (20.47%) |
| 关节2裁剪/全部控制步 | 2953/24452 (12.08%) | 2000/27321 (7.32%) |
| 关节3裁剪/全部控制步 | 1420/24452 (5.81%) | 842/27321 (3.08%) |
| 关节4裁剪/全部控制步 | 2102/24452 (8.60%) | 3328/27321 (12.18%) |
| 关节5裁剪/全部控制步 | 4667/24452 (19.09%) | 4083/27321 (14.94%) |
| 关节6裁剪/全部控制步 | 3950/24452 (16.15%) | 3329/27321 (12.18%) |
| 关节7裁剪/全部控制步 | 3167/24452 (12.95%) | 3320/27321 (12.15%) |
| 偏离起始carrying_fruit | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始closed_empty | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始open_after_basket | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始open_before_first_detach | 0/0（不适用/无观测） | 0/0（不适用/无观测） |

### FruitCount

| 指标 | K4 | K8 |
|---|---:|---:|
| success | 13/20 (65.00%) | 14/20 (70.00%) |
| fail | 7/20 (35.00%) | 3/20 (15.00%) |
| timeout | 0/20 (0.00%) | 3/20 (15.00%) |
| engineering_error | 0/20 (0.00%) | 0/20 (0.00%) |
| first_target_in_basket | 18/20 (90.00%) | 20/20 (100.00%) |
| never_detached | 0/20 (0.00%) | 0/20 (0.00%) |
| near_miss | 0/20 (0.00%) | 0/20 (0.00%) |
| wrist_over_limit | 0/20 (0.00%) | 0/20 (0.00%) |
| wrist_deviation_executed | 0/20 (0.00%) | 0/20 (0.00%) |
| wrist_deviation_proposed | 0/20 (0.00%) | 0/20 (0.00%) |
| freeze_200 | 0/20 (0.00%) | 0/20 (0.00%) |
| freeze_200_basket_unsuccessful | 0/5 (0.00%) | 0/6 (0.00%) |
| count_exact | 13/20 (65.00%) | 14/20 (70.00%) |
| correct_in_basket（果实总数/有数据回合） | 51/20 | 55/20 |
| wrong_in_basket（果实总数/有数据回合） | 0/20 | 0/20 |
| start_idle中位步 | 31.0 | 31.0 |
| leave_home_step中位步 | 36.0 | 36.0 |
| post_release_resume_steps中位步 | 15.0 | 15 |
| 从未离家 | 0/20 (0.00%) | 0/20 (0.00%) |
| 入篮后未恢复（右删失） | 13/51 (25.49%) | 14/55 (25.45%) |
| 任一关节裁剪/全部控制步 | 172/9246 (1.86%) | 2448/12928 (18.94%) |
| 关节1裁剪/全部控制步 | 6/9246 (0.06%) | 2105/12928 (16.28%) |
| 关节2裁剪/全部控制步 | 8/9246 (0.09%) | 500/12928 (3.87%) |
| 关节3裁剪/全部控制步 | 0/9246 (0.00%) | 93/12928 (0.72%) |
| 关节4裁剪/全部控制步 | 0/9246 (0.00%) | 873/12928 (6.75%) |
| 关节5裁剪/全部控制步 | 92/9246 (1.00%) | 1545/12928 (11.95%) |
| 关节6裁剪/全部控制步 | 9/9246 (0.10%) | 1345/12928 (10.40%) |
| 关节7裁剪/全部控制步 | 138/9246 (1.49%) | 1276/12928 (9.87%) |
| 偏离起始carrying_fruit | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始closed_empty | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始open_after_basket | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始open_before_first_detach | 0/0（不适用/无观测） | 0/0（不适用/无观测） |

### FruitOccludedRipe

| 指标 | K4 | K8 |
|---|---:|---:|
| success | 5/20 (25.00%) | 4/20 (20.00%) |
| fail | 8/20 (40.00%) | 12/20 (60.00%) |
| timeout | 7/20 (35.00%) | 4/20 (20.00%) |
| engineering_error | 0/20 (0.00%) | 0/20 (0.00%) |
| first_target_in_basket | 13/20 (65.00%) | 11/20 (55.00%) |
| never_detached | 1/20 (5.00%) | 0/20 (0.00%) |
| near_miss | 1/20 (5.00%) | 0/20 (0.00%) |
| wrist_over_limit | 0/20 (0.00%) | 0/20 (0.00%) |
| wrist_deviation_executed | 0/20 (0.00%) | 0/20 (0.00%) |
| wrist_deviation_proposed | 0/20 (0.00%) | 0/20 (0.00%) |
| freeze_200 | 0/20 (0.00%) | 0/20 (0.00%) |
| freeze_200_basket_unsuccessful | 0/10 (0.00%) | 0/12 (0.00%) |
| count_exact | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| correct_in_basket（果实总数/有数据回合） | 25/20 | 25/20 |
| wrong_in_basket（果实总数/有数据回合） | 0/20 | 2/20 |
| start_idle中位步 | 31.0 | 31.0 |
| leave_home_step中位步 | 36.0 | 36.0 |
| post_release_resume_steps中位步 | 24.5 | 18 |
| 从未离家 | 0/20 (0.00%) | 0/20 (0.00%) |
| 入篮后未恢复（右删失） | 5/25 (20.00%) | 6/27 (22.22%) |
| 任一关节裁剪/全部控制步 | 6462/15206 (42.50%) | 4926/14393 (34.22%) |
| 关节1裁剪/全部控制步 | 4659/15206 (30.64%) | 3487/14393 (24.23%) |
| 关节2裁剪/全部控制步 | 2945/15206 (19.37%) | 1500/14393 (10.42%) |
| 关节3裁剪/全部控制步 | 1420/15206 (9.34%) | 749/14393 (5.20%) |
| 关节4裁剪/全部控制步 | 2102/15206 (13.82%) | 2455/14393 (17.06%) |
| 关节5裁剪/全部控制步 | 4575/15206 (30.09%) | 2538/14393 (17.63%) |
| 关节6裁剪/全部控制步 | 3941/15206 (25.92%) | 1984/14393 (13.78%) |
| 关节7裁剪/全部控制步 | 3029/15206 (19.92%) | 2044/14393 (14.20%) |
| 偏离起始carrying_fruit | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始closed_empty | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始open_after_basket | 0/0（不适用/无观测） | 0/0（不适用/无观测） |
| 偏离起始open_before_first_detach | 0/0（不适用/无观测） | 0/0（不适用/无观测） |

离家与恢复中位数仅计算观察到的事件，未观察值单列，不填0。count_exact只适用于FruitCount；裁剪分母包括30步保持，未去掉回合中段帧。

## 停点A的审阅事项

请按HANDOFF §12.5审阅两模型T1、有效性与冒烟，以及协议草案。草案预定义四项单侧Holm检验、10000次布局聚类bootstrap、互斥失败分类和种子7无包络敏感性分析；计数提前停止明确为最后200步的可观察代理，限制已列明。
当前停在A，等待审阅后再封存和正式评测；没有读取val布局、执行正式/敏感性回合或写论文。

<!-- STOP_A_RESULTS_END -->

## 11. 修订5分类试运行

# 修订5分类代码开发集试运行

只使用既有dev200–219回放，不读val；不基于这些结果改变定义或阈值。

| 条件 | 成功类 | 记忆类 | 操作类 | 无效 | 结构异常 | 邻近误抓子标签（回合/果实） | 代理脱离/真值脱果 |
|---|---:|---:|---:|---:|---:|---|---|
| K0 | 15/40 | 5/40 | 20/40 | 0/40 | 0 | 0/0 | 74/84 |
| K4 | 18/40 | 3/40 | 19/40 | 0/40 | 0 | 0/0 | 80/91 |
| K8 | 18/40 | 4/40 | 18/40 | 0/40 | 0 | 0/0 | 86/95 |

邻近误抓列是回合计数/果实事件计数，不是比例；所有分类子标签及逐回合证据在summary.json。

## 全部分类子标签（回合计数，不改变分类）

| 子标签 | K0 | K4 | K8 |
|---|---:|---:|---:|
| adjacent_misgrasp_possible | 0/40 | 0/40 | 0/40 |
| count_continued_picking_after_quota | 0/40 | 0/40 | 0/40 |
| count_over_quota | 0/40 | 0/40 | 0/40 |
| count_premature_stop_proxy | 0/40 | 0/40 | 0/40 |
| dropped_fruit_outcome_fail | 12/40 | 15/40 | 15/40 |
| near_miss | 1/40 | 1/40 | 0/40 |
| never_detached | 1/40 | 1/40 | 0/40 |
| occluded_unripe_detached | 5/40 | 3/40 | 4/40 |
| occluded_wrong_in_basket | 2/40 | 0/40 | 2/40 |
| timeout_operating_or_stalled | 13/40 | 7/40 | 7/40 |
| timeout_stopped | 0/40 | 0/40 | 0/40 |

## 分任务分类与事件召回诊断

| 条件/任务 | 成功 | 记忆 | 操作 | 无效 | 代理脱离/真值脱果 |
|---|---:|---:|---:|---:|---|
| K0/FruitCount | 13/20 | 0/20 | 7/20 | 0/20 | 49/52 |
| K0/FruitOccludedRipe | 2/20 | 5/20 | 13/20 | 0/20 | 25/32 |
| K4/FruitCount | 13/20 | 0/20 | 7/20 | 0/20 | 54/58 |
| K4/FruitOccludedRipe | 5/20 | 3/20 | 12/20 | 0/20 | 26/33 |
| K8/FruitCount | 14/20 | 0/20 | 6/20 | 0/20 | 57/58 |
| K8/FruitOccludedRipe | 4/20 | 4/20 | 12/20 | 0/20 | 29/37 |

未成熟果实脱落由连续双指抓握产生；邻近误抓子标签只描述，不改变记忆类归属。计数超额及达额后继续取果为环境终止结构异常，不当作记忆效果。

试运行判定：True。未使用本次结果修改定义、阈值、平台、模型或统计。

全部逐例证据及输入摘要：`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_eval_20261104/dev_classification_trial/summary.json`。未使用val布局，未基于结果修改定义。

## 12. 封存登记

<!-- SEAL_RECORD_BEGIN -->

封存授权：HANDOFF §13.4；分类试运行通过，无需再次审阅。

协议正文封存提交哈希：`0f69e6235e06f606d3e2d4be64d594c1e6422f8c`。

协议正文SHA256（仅排除本登记区）：`b19655e30443028ecf5a222bdc46bcee35dd0a1ba3a8165666745e0df491a4f2`。

分析代码清单SHA256：`b8f23e8386f16253ae2ad191273ebc381a2eca6c042418adf3de8db4acc00629`。

平台清单SHA256：`26818831ff72a104cbb1433b789d0f477854503aee1091443c0cd41df4997f1d`。

协议完整文件SHA256：见[配套封存清单](fruit_lowlevel_v2_seal_20261104.json)的protocol_file_sha256字段；完整文件摘要置于旁侧清单，避免把文件自身摘要写回文件造成自引用。正文提交与此登记区独立核验。

<!-- SEAL_RECORD_END -->

## 12.1 启动前工程记录

只读ready命令在任何服务或val回合启动前发现状态写入函数参数重名。修正参数命名并补充ready命令回归测试后重新封存，旧封存清单及原始诊断保留在results/fruit_lowlevel_v2_eval_20261104/prelaunch_seal_superseded/0。该启动代码修正不改变协议定义、统计规则、模型或平台行为。
