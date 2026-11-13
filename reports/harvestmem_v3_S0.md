# HarvestMem v3 S0：代码勘察与方案确认

日期：2026-11-10  
结论：**S0 部分通过，可提交审阅；不进入 S1。** 已完成代码、数据字段、重名率和 checkpoint 结构只读核查。子目标方案与格式已给出明确建议；GT 可逐帧计算的像素定位和模型 prompt 接线仍需在 S1 实现并做数据/训练 smoke，因此这里不声称端到端验证完成。

## 1. 子目标进入模型的方案

**建议选 (a)：保留 `perceptual-framesamp-modul` 与 checkpoint 79999，在任务 prompt 后附加当前 grounded 子目标。** 训练和推理统一使用：

```text
<instruction> current subgoal: <grounded_subgoal>.
```

理由：

- 79999 是 `perceptual-framesamp-modul` 权重，现有 R12 OR 变体已经验证可将 oracle 文本后缀接入 prompt 并完成训练；这条数据通路最贴近本轮已验证的训练流程。
- MME-VLA 的 symbolic tokenizer 确实支持 grounded 子目标，但当前自带 `symbolic-grounded-subgoal.yaml` 配置为 `integration_type: context`；79999 对应 `modulation`，checkpoint 中有 modulation 专属的 Gemma `mem_attn` 参数及 `mem_encoder` 参数。按本机代码用 symbolic 配置创建参数树并对照 79999 checkpoint `_METADATA`：symbolic 树预期 51 个参数叶子，checkpoint 61 个；按参数名无缺失叶子，但有 10 个 checkpoint 独有叶子。checkpoint 是 4 卡 Orbax 分片格式，单进程 `_METADATA` 的 `write_shape` 不能直接当作全局 shape，因此没有把局部 shape 读数冒充最终兼容结论。完整 CPU restore smoke 已启动但在约 8 分钟、仍未输出 restore 完成时中止；所以 (b) 的“真实整包可加载”在 S0 **未确认**，仍需单独 smoke。
- 官方 symbolic tokenizer 的格式是 `Task: ...; Current Subgoal: ...; Action:`；坐标 grounding augmentation 已有实现。但本轮不需要为此承担模型结构迁移风险。

训练接线建议只对 SG arm 在读取 sample 时附加子目标（或提供新的 prompt 字段），K0/NM 保持原指令。不要改写基线 prompt，也不要修改 vendor 代码；确认 tokenizer 的 max token 长度足够容纳任务指令和坐标子目标。79999 权重路径：`/media/fulei/fuleivla/adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999`。

## 2. Grounded 子目标规范

统一把果子**初始观察帧**中的果心作为稳定锚点，使用 256×256 前置相机 RGB 坐标，左上角为原点，x 向右、y 向下，取整数像素并裁剪到 `[0,255]`。首选从实例分割 mask 求质心；如果只拿到 3D GT，则使用该帧的前置相机内参和外参投影果心到像平面。坐标只在初始化时确定，跟踪器之后用 slot/锚点关联，不因遮挡或轻微果实摆动改写目标身份。

训练记录增加结构字段 `target_slot`，区分固定果位；自然语言 grounding 使用坐标。`target_slot` 不拼进模型文本，以免模型依赖仿真专属编号。pick 的坐标来自目标果；place/return/hold/skip/none 使用固定任务语义标签，不虚构篮子坐标。坐标 x/y 对同场景不同果位保证不同；若投影量化碰撞，在 S1 以更高精度的浮点投影排序后分配最近未占用像素，验证仍无碰撞。

完整子目标集合：

| 情况 | Grounded 子目标 | 说明 |
|---|---|---|
| 初始化观察 | `observe the fruit tree` | 观察阶段，无 target slot。 |
| Count 选择果子 | `pick the fruit at <x, y>` | 只给所选目标果的初始像素锚点。 |
| Occluded 选择初始红果 | `pick the red fruit at <x, y>` | 颜色为初始属性，gust 后仍按记忆锚点定位。 |
| 放入篮子 | `place the fruit in the basket` | 放置当前已抓目标，不附树上坐标。 |
| 回到 rest | `return to rest` | 完成目标数/目标集合后执行。 |
| rest 等待终止 | `hold open at rest until termination` | 对齐现有 terminal hold 行为。 |
| Occluded 跳过非红果/已摘果 | `skip the fruit at <x, y>` | 仅用于高层逐个显式排除候选；若不执行跳过动作，可将其作为 tracker 内部决策而不进入低层训练标签。 |
| 无可执行目标 | `no target; hold open at rest` | 空候选/终止保护状态；不向低层发 pick。 |

示例（五条）：

1. `pick the fruit at <82, 61>`
2. `pick the red fruit at <174, 93>`
3. `place the fruit in the basket`
4. `return to rest`
5. `hold open at rest until termination`

建议仅让 `pick ... at <x, y>` / `pick the red fruit ...` 成为逐果 grounded 动作目标；place 和 return 是阶段目标。skip 与 none 属于 tracker/控制器的显式状态，若要进入策略数据，必须为其定义对应专家动作，不能只造语言标签。训练加噪需将坐标限制在图像边界；文档要求的 ±4 px 与 vendor 现有默认 ±8 px 不同，S2 应用独立配置实现 ±4，勿静默复用默认值。

## 3. R12 示例数据核查

统计对象是已构建 R12 训练集 `data/finetune_kf/fruit_task_v2_r12_20261109/meta/episodes.json` 中 400 个 episode 指向的 GT JSONL，而不是随机抽样。对每个 episode：逐行解析全部 GT 帧，检查 `phase` 和 `subgoal` 字段；在 timestep 0 按 `present=true` 且 `front_pixels>0` 选取可见果位，使用 `fruit_picking/solver.py::describe()` 的同一阈值生成描述。Occluded 的 `ripe` 来自对应成功示范 `r12_episode.json` 的 `result.slot_geometry`。冲突对数按同场景同名字符串的组合数 `n*(n-1)/2` 计算。

| 任务 | 示范数 | GT 帧数 | 缺 phase/subgoal 帧 | 场景重名 | 重名果位对 | 可见果位 |
|---|---:|---:|---:|---:|---:|---:|
| Count | 200 | 263,959 | 0 / 0 | 48/200（24%） | 48 | 936 |
| Occluded | 200 | 73,962 | 0 / 0 | 32/200（16%） | 32 | 926 |
| **合计** | **400** | **337,921** | **0 / 0** | **80/400（20%）** | **80** | **1,862** |

总重名率以可见果位计为 80/1,862 = **4.30%**；也可报告场景层面 20%。比如 Count episode 8 的初始可见果位中，`the fruit on the lower middle` 出现两次。当前 GT 每帧均有 `phase`、`subgoal`，但没有 `grounded_subgoal`、`target_slot`、`n_picked_so_far` 这些新 schema 字段；现有构建样本 pkl 已有 `simple_subgoal`、`grounded_subgoal` 及其 online 版本，但还没有上述三个结构字段。注意当前旧 `grounded_subgoal` 仍等同方位描述，不代表已实现坐标 grounding。

## 4. 需要新增/修改的文件

按全局规则，新代码/新数据/新结果放指定新目录；不得就地修改旧数据、checkpoint 或旧结果。具体文件落点留待 S1/S2 实施方案冻结。

| 位置 | 预期工作 |
|---|---|
| `scripts/harvestmem_v3/`（新） | GT/预测统一的 grounded 子目标格式与坐标投影 helper；S1 新 episode 生成/专家回放/验证/汇总入口；S2 新训练 variant；S3 `tracker.py` 和单测。 |
| `fruit_picking/task_v2_spec.py`、`fruit_picking/task_v2_env.py`（旧代码，窄改） | 若 S1 没有独立 env wrapper 可复用，则新增默认关闭的 `no_lamp_wait` 开关；默认值保持 R12 原行为。 |
| `fruit_picking` 记录器/新 S1 生成器（优先新入口） | 新 GT 帧记录 grounded_subgoal、target_slot、n_picked_so_far、phase；与 HDF5 info 子目标同步，避免 JSONL/pkl 标签分叉。 |
| `scripts/build_fruit_dataset.py` 或新目录下的构建入口 | 仅当现有 builder 无法读取新 schema 时，增加新入口/新 adapter；不覆盖 `data/finetune_kf/fruit_task_v2_r12_20261109`。 |
| `scripts/harvestmem_v3/test_*.py`（新） | 坐标唯一性/边界、标签映射、phase 和计数语义、tracker 因果性测试。 |
| `data/harvestmem_v3_*`、`runs/harvestmem_v3_*`、`results/harvestmem_v3_*`（新） | 后续阶段各自的专属输出；S0 未创建这些目录。 |
| `reports/harvestmem_v3_S0.md`（本文件） | 本阶段只新增的交付报告。 |

## 5. 风险和待审阅事项

- **像素 grounding 可观测性：** GT 生成可用分割质心或相机投影；部署端 SAM-FDM 可能输出 anchor 而非精确果心。S1/S3 要测量定位偏差和遮挡下关联，±4 px 训练噪声是否足够需要数据验证。
- **坐标标签的因果对齐：** 用初始图像锚点可避免动作过程中目标位置漂移，但 DART 扰动可能导致果实摆动/脱离；要定义 slot identity 和事件完成的权威规则，不能把“分割消失”直接当作入篮。
- **终止监督：** 现有示范含 return/terminal hold 语义；新的子目标必须明确何时从 return 切到 hold，并确保 terminal observation 与训练 action chunk 对齐。
- **prompt 长度与训练/部署一致性：** (a) 改文本输入，不改参数树，但追加后缀可能截断原指令或与 lowercase/tokenizer 清洗不一致。训练与策略服务必须使用同一拼接格式、max token 长度和坐标序列化。
- **count/occluded 颜色语义：** Occluded 的初始红/绿属性不能从 gust 后当前 RGB 重新判定；跟踪器需要在初始化时记住颜色，并与固定 slot 绑定。
- **S1 环境改动边界：** `TaskV2Spec` 当前 `start_delay_max=600`、revision 9 Count 会打开 `early_start_failure`；`no_lamp_wait` 应同时令 delay=0 且 early-start 检查关闭，并保持其余 R12 规则完全不变。
- **重名统计口径：** 上表是在初始帧里由 `front_pixels>0` 定义的可见果位，非全场景全部果子。全场景 target 唯一性由 slot 坐标 schema 保证；被遮挡果子的部署定位需结合初始化锚点/遮挡前观察。
- **基线隔离：** SG 才接受子目标；K0、NM 的文本输入和数据过滤必须保持一致且不含该 oracle 信息。
- **当前示范的阶段覆盖：** 本轮标签字段检查证明每帧存在 phase/subgoal，但需在 S1 单独统计各类 phase 的帧数，并为 skip/none 决定是 tracker 内部状态还是实际策略动作。

## 6. 命令、版本和门槛

- 只读命令：`git status --short --branch`、`sed`/`find`/`grep` 源码与数据检查、用 Python 标准 JSON/pickle 读取 400 episode 的 GT 和构建样本、使用 policy 环境在 CPU 上创建 symbolic 参数树并与 79999 zip 的 Orbax `_METADATA` 逐叶比较。
- Git commit：`414ef44b9afb7a84e6cec9573382700cba4b3151`（`Prepare verified revision 11 Count R0 queue`；执行前 HEAD）。工作区原本已有多项用户修改和未跟踪文件；本阶段只新增本报告，未更改/清理其它内容。
- 耗时：约 15 分钟（含数据统计、参数树审计和一次中止的 CPU restore smoke）。
- GPU/仿真：未使用。训练/闭环：未运行。曾启动一次 CPU-only symbolic restore smoke；因约 8 分钟无进度输出而中止，未修改 checkpoint。
- S0 门槛：方案决策、坐标格式/集合、5 个示例、重名率、文件清单、风险清单均已写入；symbolic 整包 restore smoke 未完成，**建议不选 (b)，代码和数据 smoke 留给 S1/S2**，因此结论为部分通过，等待审阅。

## 7. 下一阶段审阅决定

请审阅并确认：

1. 是否批准方案 (a) 作为 SG 的唯一子目标输入路径；
2. 是否批准 `<x, y>` 表示 256×256 初始前置图像像素坐标、目标身份固定绑定初始帧；
3. `skip`/`none` 是否仅作高层 tracker 状态，不进入低层策略训练标签；
4. 是否据此进入 S1。

按任务规则，S0 报告交付后停止；没有启动 S1。
