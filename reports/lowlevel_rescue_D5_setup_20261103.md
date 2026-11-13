阶段5准备完成：尚未启动K4/K8训练，尚未到停点A。
K0直接复用11999；K4/K8配置逐项与K0相同，唯一有效差别为事件记忆槽4/8。
同一164639帧数据及归一化统计直接复用，不重建、不重算；训练种子42、8卡FSDP、12000步。
新增9项测试通过，隔离的停点A报告生成检查通过；现有K0/G1来源摘要保持一致。
当前会话没有GPU设备，需要在有GPU权限的终端启动；完成T1/有效性/冒烟后自动在A停下。

## 执行顺序

K4训练→T1→dev200–219有效性40条→K8训练→T1→有效性40条→dev200两个任务三条件冒烟6条→停点A。训练或有效性失败即停止，保留全部证据，不按性能调整设置，不自动重试。

## 协议草案与启动前核验

草案：`/media/fulei/fuleivla/ecm_vla/reports/fruit_lowlevel_v2_protocol_20261103.md`，已包含完整平台SHA256、val100–149×种子7/8/9的配对、4项单侧Holm、布局聚类bootstrap10000次、互斥失败分类、逐条件裁剪率及种子7无包络敏感性分析。当前未封存、未访问val布局。

训练配置：`/media/fulei/fuleivla/ecm_vla/runs/fruit_lowlevel_v2_k4k8_20261103/arms/k4.json`、`arms/k8.json`、`plan.json`和训练专用`protocol.json`。K0原有T1通过；K4/K8的T1、有效性及冒烟目前均未运行，不报预期数为实测结果。

预检来源：`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_stage5_20261103/preflight.json`；只读验证411份数据/K0保全文件和32份运行源文件；平台清单SHA256 `0aec2ee6035113edb0af802335ef384a56ff4521f47c4356318b3f929fe7f8cb`。

## 下一步

在之前的GPU终端执行以下命令，预估两次训练合计约13小时，另加有效性和冒烟；脚本没有封存或正式评测命令。

```bash
cd /media/fulei/fuleivla/ecm_vla
if nvidia-smi -L; then
  nohup setsid --wait python3 -u scripts/run_fruit_lowlevel_v2_stage5.py --run \
    >> results/fruit_lowlevel_v2_stage5_20261103/launcher.log 2>&1 < /dev/null &
  printf '已启动，PID=%s\n' "$!"
fi
```
