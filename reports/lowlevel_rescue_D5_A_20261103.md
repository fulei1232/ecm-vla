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
