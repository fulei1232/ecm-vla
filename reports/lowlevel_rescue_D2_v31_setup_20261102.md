阶段2修订3：v3.1实现、失败诊断与CPU预检完成，等待GPU终端启动完整审计。
v3作为已采纳基线保留；v3.1限时到2026-11-03 02:39:53 UTC，重启不重置。
先诊断的24次v3失败中，0个目标在0.008m碰撞、4个在0.018m碰撞；全部诊断结果保留。
v3.1从这些失败快照单次重规划成功23/24；此数不是整回合成功率，正式200条审计尚未运行。
决策：按§11.5自动选专家并进入阶段3，不再等专家确认、不再迭代；仅两版都无法使用时停止。

## 已完成

完整阅读交接文档§11。按其顺序，先从全部24个失败快照检查目标在两半径下的碰撞，定位直线/rest/high_ready首个碰撞路点、TCP位置、关节角及碰撞对象，再实现独立canonical_v31规划分支。v3代码保留，原canonical、nearest、legacy路径只增加分支入口，去掉v3.1入口后的solver AST与9d1082f逐位相同；canonical_planning、canonical_solver、joint_planning、generate和原DART脚本均无改动。

诊断来源：`results/fruit_expert_v31_audit_20261102/v3_failure_diagnosis/report.md`、`summary.json`及24份逐尝试JSON。失败阶段为reach19、carry4、approach1。原直线/rest首碰撞均可定位，23次涉及其他果实、1次树；high_ready为20次其他果实、4次树。叶片没有碰撞体，目标果实在v3原点云中已经排除。原始失败RRT输出未保存，无法倒推出它的首碰撞点，诊断用一次重新规划明确标记，保留原失败快照不变。检查原生Python入口未发现把点云半径自动改回0.008m的逻辑，故不把§11.3中的该推断写成事实。

v3.1路径优先级为直线→tuck→high_ready→RRT；tuck固定q2–q7为[-1,-.0061,2,-.181,.0534,-1.3933]，reach的q1取目标，carry取当前，q7取rest。真实URDF检查此姿态在q1=0时TCP高约0.626m，最远链接原点半径约0.215m。中段点云半径0.018m，目标及TCP距目标≤0.05m末段0.008m，目标果实沿用v3排除语义。仅目标0.008m碰撞允许替换IK，路径碰撞不替换目标。carry仍直接rest，screw和各保持步数与v3相同。

RRT使用同一原始点云：端点允许时0.018m搜索；需要末段0.008m时，原生绑定只能接受单一点云半径，按§11.4.4使用0.008m搜索，输出再按同一分区策略密集校验。每段保留半径、首次碰撞及方法记录。

重规划来源：`v3_failure_replanning/summary.json`。23/24成功均为tuck；失败的一次保留原结果，未重复尝试以挑选结果。

## 审计和专家选择

正式train0–99、双任务共200条新示范写入独立v3.1目录。核心：reach50+carry50各3次规划逐位一致；目标连续性≥98/100；方法一致率≥95/100；抓取q5/q6/q7各≥95%在自身中位数±1rad；q7指令/状态全在[-π,π]；各任务逐尝试成功≥90%、规划失败≤5%。

q1只报告Σ|Δq1|−|q1末−q1初|的P90。v3补算为reach0、carry0.0000379337rad，来源`v3_extended_stats.json`。3×2覆盖格定义预先固定：left:y>.08，right:y<−.08，middle闭区间；upper:z≥.54，lower:z<.54。分母为实际reach目标尝试，失败归于导致整回合失败的初始果实位置，未访问果实不计入。v3及v3.1两版、后续阶段3干净/DART新数据均报告该分布。

全部核心达标选v3.1；仅逐尝试成功项失败按合计成功尝试/全部尝试比较，相同选v3.1，逐任务同时报告；其他核心项退步回退已采纳v3。§11.5未单列仅目标/方法退步的情况，按“不再迭代、直接进入阶段3”的总指令选择v3，不增加确认点。

## 验证与阶段3准备

49项相关测试全部通过，覆盖目标碰撞允许换解、路径碰撞不换解、tuck的q1规则、包络空交集投影、真实旧训练规格差异、目标覆盖格及原官方评测循环。真实原生规划的reach/carry各一保存状态重复3次均逐位一致；官方HDF5前30帧历史约定及50帧断言通过。预检来源`preflight.json`，没有把小探针等同于完整核心审计。

串联脚本已接好阶段3：按选择专家重新生成clean0–99/DART100–199，原OU参数直接复用；所有帧保留，前30帧历史；原构建器、重算统计、79999初始化、原K0配方12000步、8卡FSDP、原T1。评测原循环先固定保持30步（首个策略动作第31步），全部观测摄入历史，总预算准确1300步；CPU测试已验证该边界。包络v2先按专家裁剪率验收，再在dev200–219作不阻塞配对；G1前冻结设置，只跑dev220–259一次，记录后不重跑。

新协议：`reports/fruit_lowlevel_rescue_stage3_protocol_20261102.md`。阶段3产物：`data/fruit_picking_v2`、`data/fruit_picking_dart_v2`、`data/finetune_kf/fruit_dart_v2_rm75b_train`、`runs/fruit_k0v2_lowlevel_rescue_20261102`、`results/fruit_k0v2_gate_20261102`。这些新目录启动前均不存在；不会覆盖v2/v3及旧训练产物。

## 启动

当前执行环境没有GPU设备访问，需要用户在已有GPU权限的终端启动。该命令不是再次审批专家；之后按已授权规则自动继续阶段3。

```bash
cd /media/fulei/fuleivla/ecm_vla
if nvidia-smi -L; then
  nohup setsid --wait python3 -u scripts/run_fruit_v31_rescue.py \
    --out results/fruit_expert_v31_audit_20261102 \
    >> results/fruit_expert_v31_audit_20261102/launcher.log 2>&1 < /dev/null &
  printf '已启动，PID=%s\n' "$!"
fi
```

运行状态：`results/fruit_expert_v31_audit_20261102/status.json`；进入阶段3后同时查看`results/fruit_k0v2_gate_20261102/status.json`和训练日志。v3.1完整报告按§6写入独立D2_v31报告与summary；G1结束按§4.3汇报D3后等待确认。
