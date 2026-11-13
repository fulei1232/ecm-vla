阶段 0：完成；快照、统一复核、新划分生成与哈希冻结均已完成。
已确认：统一脚本读取 980 场历史轨迹，正式表与 K0 失控起始情境完全复现（明细见下列结果路径）。
已确认：dev 起始静止中位数按一基控制步为 78.5/112；入篮未成功冻结为 4/13、10/16，差异证据见下文。
已确认：新划分 180/180 manifest 及 360 个内容文件哈希通过校验；登记 frozen=true。
决策：按 §4.0 自动进入阶段 1；阶段 1 的 G0 结束后停下等用户确认。

## 做了什么

完整阅读并按 `reports/HANDOFF_lowlevel_rescue_20261102.md` 执行。未找到适用的 AGENTS.md。未改动已封存协议、旧结果、专家、记忆、SAM 或论文。没有训练，也没有运行任何新门槛集或新验证集评测。

1. 审查原有修改并提交快照：`c57a16409983319e8dd052479273e39dd1bc5dd8`，提交信息为 `WIP: DAgger rounds 1–3 scripts as run`，包含 12 个既有未跟踪脚本和 2 个已修改文件。
2. 新增 `scripts/analyze_fruit_failure_modes.py`。输入 episodes 根目录、目录模式、编号及种子过滤，输出逐回合 CSV/JSONL 和合计/分任务 summary；所有后续阶段统一使用此脚本。
3. 对原正式 K0/K4/K8 各 300 场及 dev 原 K0/候选各 40 场离线复算；仅读取历史评测日志，未读取已用尽 val 0–99 的布局来运行新任务。
4. 新增 `scripts/run_fruit_rescue_splits.py`，固定当前 legacy 生成器源文件哈希，支持跳过并校验已有 manifest、失败留档、8 进程分批生成、dev 完成后才生成 val、完成后冻结登记。GPU 不可用导致首批环境初始化失败。

### 快照中的默认行为审查

`scripts/build_fruit_dataset.py` 的 `skip_missing=False` 保持旧行为，只有显式 `--skip-missing` 才跳过缺失 manifest。归一化批大小由固定 128 改为 `min(128, execution_samples)`：这确实改变了不足 128 样本时的默认行为，已在此记录；正式训练数据有 173791 个 execution_samples，因此封存配方仍用 128。来源：`data/finetune_kf/fruit_dart_rm75b_train/meta/stats.json`。

`scripts/finetune_event_keyframes.py` 仅在训练 spec 显式给出 `save_interval` 时改变保存间隔；缺省仍为总训练步数。12 个新增脚本有独立入口，审查了导入、输出位置和调用关系，不会由旧正式入口自动执行。对这些新增脚本中的旧 DAgger 逻辑只做原样快照，不继续采集。14 个快照脚本均通过语法解析。

## 关键数字与任务书的对照

### 正式评测：任务书数值 → 本次复算

每列的输入与逐回合文件如下，下面本列所有数字均来自相应 summary：

- `formal_K0`：输入 `/media/fulei/fuleivla/ecm_vla/results/fruit_dart_kf_eval_20261027/episodes/K0`；输出 `results/fruit_lowlevel_rescue_D0_20261102/formal_K0/summary.json`、`results/fruit_lowlevel_rescue_D0_20261102/formal_K0/episodes.jsonl`。
- `formal_K4`：输入 `/media/fulei/fuleivla/ecm_vla/results/fruit_dart_kf_eval_20261027/episodes/K4`；输出 `results/fruit_lowlevel_rescue_D0_20261102/formal_K4/summary.json`、`results/fruit_lowlevel_rescue_D0_20261102/formal_K4/episodes.jsonl`。
- `formal_K8`：输入 `/media/fulei/fuleivla/ecm_vla/results/fruit_dart_kf_eval_20261027/episodes/K8`；输出 `results/fruit_lowlevel_rescue_D0_20261102/formal_K8/summary.json`、`results/fruit_lowlevel_rescue_D0_20261102/formal_K8/episodes.jsonl`。
- `dev_baseline`：输入 `/media/fulei/fuleivla/ecm_vla/results/fruit_dart_dev_gate_20261027/episodes/K0`；输出 `results/fruit_lowlevel_rescue_D0_20261102/dev_baseline/summary.json`、`results/fruit_lowlevel_rescue_D0_20261102/dev_baseline/episodes.jsonl`。
- `dev_candidate`：输入 `/media/fulei/fuleivla/ecm_vla/runs/fruit_dagger_full_task_20261031/dev40/candidate`；输出 `results/fruit_lowlevel_rescue_D0_20261102/dev_candidate/summary.json`、`results/fruit_lowlevel_rescue_D0_20261102/dev_candidate/episodes.jsonl`。

| 指标 | K0 | K4 | K8 |
|---|---:|---:|---:|
| 成功 | 3/300 → 3/300 | 1/300 → 1/300 | 2/300 → 2/300 |
| q7 指令绝对值 > 6.28 | 184/300 → 184/300 | 194/300 → 194/300 | 190/300 → 190/300 |
| 掉果 outcome=fail | 88/300 → 88/300 | 79/300 → 79/300 | 80/300 → 80/300 |
| 已入篮且未成功子集的 freeze_200 | 26/87 → 26/87 | 18/87 → 18/87 | 14/68 → 14/68 |

这些项全部一致。对冻结子集复算双侧 Fisher：K0–K4 p=0.221903，K0–K8 p=0.201658；均不显著，仅作过程指标。依据：`reports/lowlevel_rescue_D0_20261102/summary.json` 的 `fisher_two_sided_freeze`，由上表列联表计算。

### dev：任务书数值 → 本次复算

| 指标 | 原 K0 | 全任务 DAgger |
|---|---:|---:|
| 成功 | 3/40 → 3/40 | 1/40 → 1/40 |
| q7 越限 | 18/40 → 18/40 | 23/40 → 23/40 |
| 掉果 | 12/40 → 12/40 | 7/40 → 7/40 |
| 一颗未摘下 | 13/40 → 13/40 | 16/40 → 16/40 |
| 旧表标为放下至少一颗后冻结 | 5 → 严格子集 4/13 | 11 → 严格子集 10/16 |
| 全部回合 freeze_200（另列，非子集） | 5/40 | 11/40 |
| start_idle 中位数 | 76.5 → 78.5 控制步 | 110 → 112 控制步 |

**已确认的差异：** dev 原 K0 的 `FruitCount/ep218_seed7` 和候选的 `FruitCount/ep218` 都是 timeout、最终入篮 0，最长冻结分别为 350 和 222 步。它们满足全体 freeze_200，但不满足“已入篮且未成功”子集。原表冻结计数恰好等于全体计数；新脚本保留第 5 节的子集定义。逐回合依据：`results/fruit_lowlevel_rescue_D0_20261102/{dev_baseline,dev_candidate}/episodes.jsonl`；原始路径见该文件每行 `sources`。

**已确认的索引关系：** `start_idle` 使用 `control_actions.jsonl` 的一基 `control_step`。相邻差分数组的零基位置等于该值减 2，因此两个中位数减 2 恰为旧表 76.5/110。**推断：** 旧分析可能输出差分数组位置；未拿到当时的分析代码，不能把该原因说成已证实。没有为匹配旧表而改定义。没有检测到运动的回合输出 null 并记录右删失，本批全部检测到运动。

### K0 首次腕部偏离前的情境

来源：`results/fruit_lowlevel_rescue_D0_20261102/formal_K0/summary.json` 的 `deviation_onset_context`。共 209/300 场发生偏离，取首次偏离前 32 控制步的快照，夹爪按该步指令正负分类；初始 t=0 夹爪按张开处理。

| 情境 | 任务书 | 复算 |
|---|---:|---:|
| 第一颗前且张开 | 119/209 | 119/209 |
| 已入篮且张开 | 40/209 | 40/209 |
| 闭合但无果实 | 26/209 | 26/209 |
| 搬运果实 | 24/209 | 24/209 |

四项完全一致。实现中前三类优先判定后，第四类为剩余张开情境；该类可能也包含“已脱离后丢失但尚未入篮”，不能仅凭类别标签证明绝无摘下。分类口径已在 summary 的 definitions 中写明。

### 合计与分任务：关键指标

所有比例均列分子/分母。下表按上述各组 summary 的 `all` / `by_task` 读取；correct/wrong 是果实总数，不是回合比例。第 5 节全部指标（包括原始/执行腕部偏离、near_miss、count_exact、每关节裁剪率等）均保存在机器可读汇总，未以缺失值充当 0。

| 数据组 | 任务 | 成功 | fail | timeout | 首目标入篮 | correct/wrong 果实数 | 未摘下 | 工程错误 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| formal_K0 | 合计 | 3/300 | 88/300 | 209/300 | 54/300 | 97/1 | 133/300 | 0/300 |
| formal_K0 | FruitCount | 3/150 | 46/150 | 101/150 | 32/150 | 49/0 | 70/150 | 0/150 |
| formal_K0 | FruitOccludedRipe | 0/150 | 42/150 | 108/150 | 22/150 | 48/1 | 63/150 | 0/150 |
| formal_K4 | 合计 | 1/300 | 79/300 | 220/300 | 31/300 | 79/15 | 151/300 | 0/300 |
| formal_K4 | FruitCount | 1/150 | 39/150 | 110/150 | 18/150 | 36/5 | 82/150 | 0/150 |
| formal_K4 | FruitOccludedRipe | 0/150 | 40/150 | 110/150 | 13/150 | 43/10 | 69/150 | 0/150 |
| formal_K8 | 合计 | 2/300 | 80/300 | 218/300 | 38/300 | 76/2 | 154/300 | 0/300 |
| formal_K8 | FruitCount | 1/150 | 38/150 | 111/150 | 19/150 | 33/0 | 85/150 | 0/150 |
| formal_K8 | FruitOccludedRipe | 1/150 | 42/150 | 107/150 | 19/150 | 43/2 | 69/150 | 0/150 |
| dev_baseline | 合计 | 3/40 | 12/40 | 25/40 | 13/40 | 19/0 | 13/40 | 0/40 |
| dev_baseline | FruitCount | 2/20 | 7/20 | 11/20 | 7/20 | 9/0 | 8/20 | 0/20 |
| dev_baseline | FruitOccludedRipe | 1/20 | 5/20 | 14/20 | 6/20 | 10/0 | 5/20 | 0/20 |
| dev_candidate | 合计 | 1/40 | 7/40 | 32/40 | 11/40 | 18/0 | 16/40 | 0/40 |
| dev_candidate | FruitCount | 1/20 | 3/20 | 16/20 | 7/20 | 8/0 | 10/20 | 0/20 |
| dev_candidate | FruitOccludedRipe | 0/20 | 4/20 | 16/20 | 4/20 | 10/0 | 6/20 | 0/20 |

## 新划分登记与失败运行

登记文件：`reports/split_registry_20261102.json`，已冻结：`frozen=true`、180/180 manifest，文件列表 sha256 为 `67d6b1c64ad9724950c15a499c30d67e4de36339dcbf17c2a4869ca267a101fc`。独立复核全部 manifest 及其 360 个 HDF5/GT 内容文件哈希，生成器源文件哈希与原 legacy 版本一致。验证依据：`results/fruit_lowlevel_rescue_D0_20261102/split_verification.json`；manifest 列表及各 sha256 均见登记文件。

目标按 §4.0 明确范围执行：dev 220–259 两任务各 40 条，val 100–149 两任务各 50 条，合计 180 条。任务书 §8 的“120 条”与明确范围不符，未采用该估算数。val 仅生成并作文件完整性验证，未分析其布局、运行策略或调参。冻结后不重新生成。

用户从有 GPU 权限的终端以 `nohup setsid --wait` 续跑同一个脚本，实际生成成功，解除了先前环境阻塞。完整命令、成功及失败批次均保留于 `results/fruit_lowlevel_rescue_D0_20261102/split_generation/status.json` 及各批次日志；`plan.json` 记录原生成器源文件哈希。

失败记录（全部保留）：

- 首次使用 `nohup setsid ... &` 后启动 shell 退出，worker 未持续运行且日志为空；改用 `nohup setsid --wait ...` 让工具保留监督会话，随后得到可审计的失败退出。
- NVIDIA 检查退出码 9；沙箱 `/dev/nvidia*` 不存在。正确 SIM_PYTHON 检查得到 `torch.cuda.is_available=False`、device_count=0。另有一次诊断探针错误解析了虚拟环境 Python 的符号链接，导致基础 Python 缺 torch；已纠正，该失败未用作 GPU 证据。完整 stdout/stderr：`results/fruit_lowlevel_rescue_D0_20261102/environment_check.json`。
- legacy 生成器首批退出码 1，SAPIEN 报 CUDA 不可用以及 `vk::createInstanceUnique: ErrorIncompatibleDriver`。完整日志：`results/fruit_lowlevel_rescue_D0_20261102/split_generation/dev_gate_count_220_224_attempt0.log`。

已确认当前沙箱无法使用 NVIDIA/Vulkan；这不能证明宿主机驱动本身损坏。当前权限配置禁止 sandbox_approval，不能在本会话自行切到沙箱外运行，也未尝试绕过。

## 验证

统一分析脚本的 7 项 CPU 测试通过：覆盖 200/201 步冻结边界、运动严格阈值、near_miss 距离边界、原始/执行动作分离、夹爪排除、目标顺序、偏离前 32 步上下文，以及缺 result/缺帧/缺距离的工程错误记录。日志：`results/fruit_lowlevel_rescue_D0_20261102/metrics_tests.log`。增强了缺失帧检查后已重跑全部 980 场，仍无缺回合、工程错误或帧控制步不一致。新脚本均通过语法解析。

## 按规则做出的决策

阶段 0 已完成，报告、全部指标 summary、冻结登记与状态文件齐备。未进行 G0/G1 判定。按 §4.0 自动进入阶段 1，使用现有 K0 在 dev 200–219 上比较 A0–A4；不使用新门槛集或封存验证集调试。阶段 1 的 G0 结果出来后，按任务书停下等用户确认。

## 下一步

实现默认关闭的关节安全包络与独立执行步长，计算专家训练示范分位数、做必要单元测试和记忆一致性审计，再完成 A0–A4。所有 GPU 运行需通过已授权且具备设备访问能力的执行环境启动。

机器可读全部指标：`reports/lowlevel_rescue_D0_20261102/summary.json`（版本化副本）及 `results/fruit_lowlevel_rescue_D0_20261102/summary.json`（运行副本）。当前冻结登记 sha256：`cfc430da817e779dc3d2b50e10a5c07a4d7cecc6e66ecf15f7f2bfd831ebb943`；校验证据见 `split_verification.json`。
