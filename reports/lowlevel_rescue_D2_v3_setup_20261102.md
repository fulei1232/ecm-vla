阶段2专家v3：实现与CPU预检完成，等待有GPU权限的终端启动完整审计。
已完整阅读修订2第10节；本轮限时一天，五个核心项决定是否进入阶段3。
legacy、nearest原有执行逻辑保持不变，v2源文件快照及产物校验清单另存，不覆盖v2。
两个任务的真实mplib保存输入各重复规划3次，路径逐位相同；这只是预检，不能替代100实例正式审计。
下一步：生成200条新示范、完成规划与物理回放审计及三方报告；任一核心项失败就停止。

## 实现

新增 `fruit_picking/canonical_planning.py`、`canonical_solver.py`，仅 `ik_mode="canonical"` 调用。carry目标直接取rest手臂角；reach先从rest初值CLIK求解，失败才用v2相同的局部确定性随机发生器后备，以七关节等权距离选择离rest最近的有效解，q7按rest参考映射。先确定目标，路径失败时不切换IK分支。

reach/carry的点云半径为0.018m，即原POINT_RADIUS=0.008m加1cm。先检查关节直线，再尝试经固定rest和固定high_ready的两段直线，RRT最后使用并逐次记日志。high_ready固定为 `[-1.7306,-0.9000,-0.0061,1.4000,-0.1810,0.0534,-1.3933]`。直线进行密集碰撞检查、TOPP时间参数化和输出检查，显式补上目标端点；RRT输出也重新检查。

approach/pull继续原plan_screw及原POINT_RADIUS，不改close10、open8、observe30、done5。canonical settle默认2，前颗果实在下一reach规划完成后、执行前检查；最后一颗在done后检查。规划不推进仿真时钟。若任一任务或合计落篮失败/实际尝试>3%，按指令只恢复12一次，另外生成一套完整数据，保留settle2全部产物。

所有失败尝试保留HDF5、GT、原因、命令/状态q7统计及规划输入，不只保留被接受布局。最多16次换种子规则沿用原生成器。

## 审计与停止规则

- 规划确定性：交错抽取两个任务的reach50、carry50，从保存的完整规划输入各重复3次，精确比较路径、时间、速度、加速度及trace。
- 连续性：100个reach，各做5组果实坐标±2mm、实测关节±0.001rad扰动，重新计算预抓取位姿；比较含基准的最大两两目标差<0.05rad且方法相同，要求≥98%。
- 抓取手腕：approach末q5、q6、q7分别至少95%落在各自中位数±1rad内。
- reach/carry实测q1摆幅P90分别<1rad；q7全部记录指令与实测状态在[-π,π]内。
- RRT比例、逐尝试成功率、失败类别比例为非核心项；三方对照同时给出静止监督与长度。
- D2 occluded_ripe的ep0、ep1、ep3，各把D2 repeat0原动作独立回放3次，记录机器人、叶片、果实的完整状态，定位首次精确分叉及>1e-6的分叉，仅报告不设门槛。
- 全部核心项通过才进入阶段3，按§9.4与§10.5执行，无需再次批准；任一核心项失败不再自行迭代v3。

## 已完成的检查

`results/fruit_expert_v3_audit_20261102/preflight.json`：真实URDF、原生CLIK、碰撞检测和TOPP，两任务各一保存输入，恢复后重复3次与原路径逐位一致。count走固定high_ready绕行，occluded_ripe走直线。没有运行GPU仿真，也没有把2个探针宣称为正式审计通过。

AST比对f992da7确认：去掉canonical专用分支后，solver原构造/规划/执行/保持/抓取/主循环，以及generate.run_episode/verify逻辑相同；joint_planning.py、v2运行/重复/统计/报告脚本原文件无改动。

相关单元测试37项全部通过。正式运行源文件17项已冻结哈希，v2产物236项及7份v2源文件快照校验通过，冻结划分不变。准备摘要见 `reports/lowlevel_rescue_D2_v3_setup_20261102/summary.json`。

旧专家与v2的q5/q6/q7一致性从原HDF5补算到独立v3目录：旧专家492/539、514/539、488/539；v2为199/547、491/547、268/547。来源为 `legacy_extended_stats.json`、`nearest_v2_extended_stats.json`，每项附原manifest及内容哈希。

原D2三个布局的首次动作差与首次已记录状态差定位见 `physics/original_d2_divergence.json`。已有记录不包含关节速度、果实朝向/速度和叶片完整位姿，不能提前断言分叉由物理微差引起，须结合固定动作回放。

## 启动和产物

当前会话不能访问 `/dev/nvidia*`；需要在用户已有GPU权限的终端启动。运行器有独占锁、源文件和冻结划分哈希校验、24小时截止时间、独立输出、可恢复记录及完成后重复启动保护。

本轮准备起始为2026-11-02 13:53:58 UTC，截止为2026-11-03 13:53:58 UTC，重启不会重置限时。

```bash
cd /media/fulei/fuleivla/ecm_vla
if nvidia-smi -L; then
  nohup setsid --wait python3 -u scripts/run_fruit_expert_v3_audit.py \
    --out results/fruit_expert_v3_audit_20261102 --gpus 0 1 2 3 4 5 6 7 \
    >> results/fruit_expert_v3_audit_20261102/launcher.log 2>&1 < /dev/null &
  printf '已启动，PID=%s\n' "$!"
fi
```

运行状态：`results/fruit_expert_v3_audit_20261102/status.json`。正式交付为同目录 `report.md`、`audit.json`、`summary.json`、4段视频及抽帧；版本化报告为 `reports/lowlevel_rescue_D2_v3_20261102.md`，不会覆盖D2 v2。
