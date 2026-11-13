阶段5修订5已封存：分类试运行通过，正式评测尚未启动。
协议§5/§7/§8已按§13.2修改，定义、阈值和模型平台未据试运行调整。
dev既有120条均有效，计数结构异常为0；记忆类K0=5/40、K4=3/40、K8=4/40。
10项新增测试及真实封存ready预检通过；启动前状态参数错误已修复并保留旧清单和诊断，未产生val回合。
当前会话无GPU设备；待GPU终端启动后，自动执行正式900条与无包络300条，在B或有效性失败处停止。

## 封存与试运行

协议：`/media/fulei/fuleivla/ecm_vla/reports/fruit_lowlevel_v2_protocol_20261103.md`；正文提交 `0f69e6235e06f606d3e2d4be64d594c1e6422f8c`；完整文件SHA256 `35b0b35d683b6df78f9bb59f1579015f8fc2146bd784d1bfcb3d6cf2b08bd331`。
封存清单：`/media/fulei/fuleivla/ecm_vla/reports/fruit_lowlevel_v2_seal_20261104.json`；分析清单SHA256 `b8f23e8386f16253ae2ad191273ebc381a2eca6c042418adf3de8db4acc00629`；平台清单SHA256 `26818831ff72a104cbb1433b789d0f477854503aee1091443c0cd41df4997f1d`。
完整文件摘要保存在旁侧清单，正文摘要排除封存登记区；正文提交和所有源码逐项核验。

开发集分类试运行：`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_eval_20261104/dev_classification_trial/report.md`及summary.json，版本化副本为reports/fruit_lowlevel_v2_classification_trial_20261104.md/.json。成功/记忆/操作/无效及所有子标签、逐回合输入摘要完整保留，未使用val布局调整设置。

## 执行与错误边界

三个模型分别固定11999。每个模型种子7/8/9各一个服务会话装载三个模型，GPU0–2；仿真GPU3–5。主评测每任务每条件150条，共900条；敏感性只种子7，共300条，唯一改变为关闭包络。
仅无结局工程中断可同种子最多重跑2次；所有尝试保留在superseded，采纳回合仅链接。已有success/fail/timeout的回合不重跑。组内配对、逐决策记忆、事件或环境终止异常一旦构成有效性失败即停止，只报工程诊断。

## 启动命令

```bash
cd /media/fulei/fuleivla/ecm_vla
if nvidia-smi -L; then
  nohup setsid --wait python3 -u scripts/run_fruit_lowlevel_v2_eval.py --run \
    >> results/fruit_lowlevel_v2_eval_20261104/launcher.log 2>&1 < /dev/null &
  printf '已启动，PID=%s\n' "$!"
fi
```

预检：`/media/fulei/fuleivla/ecm_vla/results/fruit_lowlevel_v2_eval_20261104/preflight.json`；实时状态：status.json；完成后REPORT.md与summary.json，停止在B等待用户审阅，不写论文。
