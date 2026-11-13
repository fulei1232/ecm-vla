# Low-level rescue 阶段3执行协议（修订3）

依据：HANDOFF_lowlevel_rescue_20261102.md §4.3、§9.4、§11.5–11.6。旧协议、旧数据、v2/v3结果不覆盖。本协议在新训练前冻结，具体专家由v3.1审计的selection.json决定。

1. 专家选择：六个新核心项全部达标用v3.1；只有逐尝试成功率项失败时按合计成功尝试/全部尝试比较（同时完整报告逐任务），相同选v3.1；其他核心项退步用已采纳的v3。§11.5未单列仅目标/方法一致性退步的情况，按“采纳v3基线且直接进入阶段3”的总指令回退v3，不新增确认点。只有两版都因代码问题等无法用于阶段3时停止；不再迭代专家。
2. v3.1限时12小时，从2026-11-02 14:39:53 UTC计；中途重启不重置。到限时仍未完成时保留全部部分结果，用已可用的v3进入阶段3，不进行下一轮专家改造。
3. 数据：选定专家全部重新生成，clean train0–99双任务到data/fruit_picking_v2，DART train100–199双任务到data/fruit_picking_dart_v2。DART原NoisyExecution/NoisyRecorder直接复用，σ=.04rad、τ=10步、reach/pull/carry全幅、approach半幅、其余无噪声，种子default_rng([layout_seed,1])，标签为干净目标；16次布局重试规则不变。失败尝试也保留。
4. observe前30帧在HDF5中标is_video_demo=True，其余False；保留所有帧及原时刻。settle2、release8、done5保持监督，不引入中途监督mask，不删除帧。构建到data/finetune_kf/fruit_dart_v2_rm75b_train，使用原官方处理器并重算归一化统计，重新验证编码特征。
5. K0训练与runs/fruit_dart_kf_20261027/arms/k0.json完全一致，仅允许数据、归一化统计、8卡布局（fsdp_devices=8，global batch仍64）三项差别；输出目录区别仅为产物位置。相同79999初始化、12000步、warmup500、LR2.5e-5→2.5e-6、AdamW/梯度裁剪/EMA.999/冻结视觉/seed42/num_workers12/log_interval50不变。仍只保存11999。新spec逐字段与旧spec比对。T1用原run_finetune_kf.gate；未完成训练最多重启一次并保留旧日志/检查点。
6. 评测：新客户端接原官方eval_each_episode循环。其init_episode之后先以初始实测关节角、夹爪张开固定保持30个物理控制步，全部观测按原时间顺序进入历史，first exec_start_idx=30，首个策略动作是第31步。epstate.count包含保持步；环境在第1300步截断，避免原循环的1301步边界。所有条件一致，30步计入预算。
7. 原始帧仍使用物理control_step，另记policy_control_step=physical−30；frame_id不变。逐条审计首决策摄入31帧，其后每16帧，图像/状态哈希、选帧、源时间位置编码、mask、WebSocket输入输出全部一致；内存审计错误计入工程错误。原记忆和事件定义不变。
8. 安全包络v2为不阻塞对照：以400条新示范状态[.5%,99.5%]各外扩.1rad为关节范围，以1.5×P99.9(|干净joint_action−实测joint_state|)为目标相对状态限制。每步取交集；空交集时将目标先裁到范围，再从实测状态按限速向该目标移动，记录guard_projection，不中止。夹爪不改。30步固定保持统一绕过包络。
9. 先验证所有专家记录动作过包络的裁剪率<.5%；不满足则不采用，不阻塞。满足时在dev200–219双任务40布局同服务会话、种子7，配对off/v2。仅当首颗入篮不低于off、腕部偏离更低，且无工程/记忆错误，才采用v2。否则off。G1前写guard_choice.json，之后冻结。
10. dev200–219同时给出leave_home_step中位数≤60的参考检查、放下后冻结与D1 A0对照，不作门槛。腕部偏离中心为选定专家400条训练示范approach末q7中位数；每个条件报告全部§5与§9.5指标。
11. G1只用冻结dev220–259、双任务各40、模型种子7一次。记录结果后不重跑，工程失败也保留；中断恢复仅执行尚无result.json的布局。先封存checkpoint manifest、guard决定及registry的哈希。门槛：总成功≥25%，每任务≥15%，首颗≥50%，腕部偏离场次≤5%，工程错误0。不开val100–149。
12. G1结束按§6交付D3报告、summary.json、状态和提交，按§4.3等待用户确认；通过后拟阶段5，未通过拟阶段4。
