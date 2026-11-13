# V-B FP32 运行准备记录（2026-10-22）

更新（2026-10-23）：完整 V-B 已完成并核验，k=1 四任务均通过 P2/P3；正式验收见 [v_b_report.md](v_b_report.md)。以下保留运行准备与修复记录。

- 结果：`results/V_B_cached_loss_g0b_fp32.json`，50 个随机 noise/time 样本，单观测 VideoUnmaskSwap ep00/F216。
- FP32 计算、BF16 checkpoint 恢复、最高矩阵乘法精度。最大相对误差 `2.9794896e-5`，真实动作维误差 `2.3936394e-5`，阈值 `1e-3`；重复缓存计算逐位一致。
- BF16 计算报告 `results/V_B_cached_loss_g0b_diagnostic.json` 仍失败。
- CPU 验证：`python3 -m pytest tests/memory_management/validation -q`，29 passed、1 skipped。另用 policy Python 在 CPU 上执行真实一层 HistoryGemma 缓存/完整前向与 batch 测试，1 passed（包括同 batch 前缀和不同记忆 mask 的对照）。
- 合成数据完成 Parquet→分析报告集成测试。
- 实际 GPU smoke 已完成 VideoUnmaskSwap ep20 的 266 帧编码，但在 F184 的首个批次停止：identity/repeat 通过，batch full/cached 最大相对误差 `0.0022688619792461395 > 0.001`。尚无正式 teacher 标签或结论。单观测 G0b 不能覆盖此批量/观测配置。
- GPU 诊断 `results/V_B_fp32_smoke/cache_batch_diagnostic.json` 已完成：原 B=1 前缀广播路径与 B=4 完整前向有数值差异；B=4 前缀的有效 token KV 与 B=4 完整前向逐位相同。全部 16 个 seed、4 种参考记忆的 B=4 cached/full 三项损失误差均为 0；完整前向自身 KV 的 suffix 对照误差也为 0。此证据限定于 VideoUnmaskSwap ep20/F184 的诊断，不等于正式实验通过。
- runner 已修改为在前缀 JIT **外**复制单观测，以实际评估 batch 大小编码前缀，保留缓存 batch 维并检查尺寸；阈值仍为 `1e-3`。旧失败协议与诊断产物保留。
- 新增 `--reuse-features-from RUN`：新协议只导入兼容且经过哈希校验的帧特征，绑定原协议、数据和编码依赖；不导入旧损失标签。新 smoke 已封存为 `results/V_B_fp32_smoke_batch4`，复用 VideoUnmaskSwap ep20 的 266 帧特征。
- 修复后的 smoke 已完成：VideoUnmaskSwap ep20/F184、offset=0、16 seed、9 种变体，共 144 行原始损失。identity/repeat/CRN/causal 检查均通过，batch full/cached 最大相对误差为 0。已重新核对 Parquet 校验哈希、配对条件、完整单元格、协议源码哈希及分析报告；CSV、JSON、Markdown 和图均已落盘。`formal=false`，P2/P3 的 false 不代表正式实验失败。

运行入口 `scripts/run_v_b_teacher_probe.py` 支持 CPU 准备、单边界 smoke、协议冻结和按边界恢复；每个边界保存逐变体/偏移/seed 的 Parquet、校验哈希及运行门槛。真实帧逐帧编码，保留官方位置编码；每个偏移复用前缀 KV，并检查 batch full/cached 等价、identity、重复计算、因果记忆和共享 noise/time。

分析入口 `scripts/analyze_v_b.py` 输出 `analysis.json`、`REPORT.md`、CSV 和图，涵盖拆半可靠性、信噪比、集合可加性、事件相关、按 episode 划分的启发式交叉验证、折扣敏感性和交叉拟合收益。贪心组合仅用偶数 seed 选择、奇数 seed 评估，并从固定候选池的对称交叉拟合中排除。

正式数据为四任务 val ep20–29，共 111 个合法边界。PatternLock ep20/25/28 不足三个合法决策，预先排除；51 个边界候选少于 6 个，保留并记录不足，不人为补造显著性峰。

先执行以下单边界 smoke（默认 batch=4，16 个 seed，仅 offset=0，少量替换）。它自动生成分析产物，但不会判定正式 P2/P3：

```bash
cd /media/fulei/fuleivla/ecm_vla
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python \
-u scripts/run_v_b_teacher_probe.py --resume results/V_B_fp32_smoke_batch4
```

`results/V_B_fp32_smoke_batch4/protocol.json` 在最终代码检查后用系统 Python 准备，绑定代码、数据、复用帧特征和 G0b 报告哈希。修改代码后应创建新运行目录，不能直接恢复旧协议。系统 Python 用于 Parquet 写入和分析，避免向策略环境添加 pyarrow/sklearn。

完整运行协议为 `results/V_B_fp32_full_20261022/protocol.json`：四任务 val ep20–29、111 边界、16 seed、有效未来偏移 0..7，固定 FP32/highest、batch=4，并复用经校验的 ep20 帧特征。启动命令：

```bash
cd /media/fulei/fuleivla/ecm_vla
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python \
-u scripts/run_v_b_teacher_probe.py --resume results/V_B_fp32_full_20261022
```

中断后用相同命令继续，已完成边界按哈希校验后跳过。全部完成后自动分析。正式运行期间仍逐边界/偏移执行正确性检查，若失败应保留现场并诊断。V-A 仍需正式运行与验收；本次工作不授予价值预测器或完整管理器的 Go 结论。
