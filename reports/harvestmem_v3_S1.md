# HarvestMem v3 S1：无灯任务变体、示范生成与数据集

日期：2026-11-10  
结论：**S1 通过。** 无灯 Count/Occluded 环境、400 条成功示范、3D 果心投影 grounded 标签、子目标切换审计和训练集构建均完成；按阶段规则停在 S1，不启动 S2 训练。

## 1. 做了什么

- 在 `fruit_picking/task_v2_env.py` 增加默认关闭的 `no_lamp_wait` 开关：`start_delay_max=0`、`early_start_failure=False`；Count 指令移除等灯前缀，其他 R12 任务规则保持不变。
- 新增 `scripts/harvestmem_v3/`：共享 prompt/投影格式、episode 运行器、GT/HDF5 重标注与 S1 可续跑驱动。
- 子目标使用初始前置帧的 3D 果心通过相机内参/`extrinsic_cv` 投影到 256×256 图像，未使用分割 mask 质心；像素冲突做确定性去重。
- 训练/推理 prompt 拼接函数已固定为：`<instruction lowercase> current subgoal: <grounded_subgoal>.`。
- 将 `return to rest to finish` 与 terminal hold 合并为 `return to rest and hold still`；skip/none 未进入训练标签。
- 命令记录在 `results/harvestmem_v3_S1/driver_logs/`；本阶段未创建 git commit，执行基线为 `414ef44b9afb7a84e6cec9573382700cba4b3151`。
- 运行期间先做 Count/Occluded 各 1 个完整 smoke，再跑 dev 200–249，最后生成 train clean 0–99 + DART 100–199。

## 2. 结果表

### 专家门槛

| split | Count | Occluded | 结论 |
|---|---:|---:|---|
| dev 200–249 | 50/50 | 50/50 | 通过 ≥45/50 |
| train clean | 100/100 | 100/100 | 全部成功 |
| train DART | 100/100 | 100/100 | 全部成功 |

总计生成 400 条训练示范，clean 与 DART 各 200 条。DART 每条与对应 clean episode 复用同一布局和 N；每回合按两周期配方施加扰动。clean 共 5 次重试，DART 共 92 次重试，均未影响最终成功数。

### 示范统计

| 任务/组 | episodes | 平均步骤 | 步数范围 | N 分布 | 总帧数 |
|---|---:|---:|---:|---|---:|
| Count clean | 100 | 1015.9 | 339–1912 | 2:9, 3:7, 4:6, 5:17, 6:14, 7:6, 8:12, 9:13, 10:16 | 101,843 |
| Count DART | 100 | 1019.0 | 343–1909 | 与 clean 完全一致 | 101,843 |
| Occluded clean | 100 | 369.0 | 275–521 | 2:51, 3:49 | 36,981 |
| Occluded DART | 100 | 368.6 | 273–517 | 与 clean 完全一致 | 36,981 |
| **合计** | **400** | — | — | — | **277,648** |

训练数据目录：

- `data/harvestmem_v3_clean_20261110/rm75b_eg2/train/`
- `data/harvestmem_v3_dart_20261110/rm75b_eg2/train/`

### 子目标帧数

| 任务/组 | observe | pick | place | return and hold |
|---|---:|---:|---:|---:|
| Count clean | 3,000 | 49,546 | 34,965 | 4,332 |
| Count DART | 3,000 | 49,482 | 34,803 | 4,558 |
| Occluded clean | 3,000 | 22,177 | 11,784 | 100 |
| Occluded DART | 3,000 | 22,105 | 11,757 | 100 |

Count 的 `return to rest and hold still` 每回合 43–44 帧，clean 平均 43.32，DART 平均 45.58；这解决了原 R12 只有约 3 帧 return 监督的问题。Occluded 沿用原任务“最后入篮后立即终止”的定义，因此只有 1 个 terminal hold 标注帧，没有人为添加动作帧。

## 3. 子目标切换门槛

逐帧审计规则：

- `pick → place`：目标果脱离枝条后进入 carry 的帧；
- `place → next pick`：确认入篮、篮中数量增加后进入下一 reach 的帧；
- `place → return`：最后一次入篮确认后进入 return/terminal hold 的帧；
- Count 不允许出现 `wait_for_green`。

训练 400 条示范共审计 **3,984 次子目标转移**：

| 检查项 | 结果 |
|---|---:|
| 转移总数 | 3,984 |
| 不一致转移 | 0 |
| 不一致率 | 0/3,984 = 0% |
| wait_for_green 帧 | 0 |
| 3D 投影越界/量化碰撞 | 0 |

所有 GT JSONL 和 HDF5 `info` 均写入 `grounded_subgoal`、`simple_subgoal`、`target_slot`、`n_picked_so_far`、`phase`。构建后的 pkl 已确认包含 `grounded_subgoal`、`simple_subgoal` 及 online 版本；结构化审计字段以 GT/HDF5 为权威记录。

## 4. Token 长度检查

使用项目实际 `PaligemmaTokenizer`、完整任务指令和 400 条示范全部 277,648 帧的 grounded 子目标测量：

| 指标 | 结果 |
|---|---:|
| `max_token_len` | 64 |
| 最大 token 数 | 60 |
| 超过 64 的帧 | 0 |
| p99 token 数 | 49 |

因此当前 prompt 拼接不会截断。训练和服务端必须继续复用 `scripts/harvestmem_v3/subgoal.py::compose_prompt` 的同一格式。

## 5. 数据集构建

最终构建目录：`data/harvestmem_v3_dataset_20261110/`。

| 项目 | 结果 |
|---|---:|
| episodes | 400 |
| 总帧数 | 277,648 |
| execution samples | 265,648 |
| Count samples | 197,686 |
| Occluded samples | 67,962 |
| feature shards | 400/400 |
| build 校验 | `passed=true` |
| distance signal | `any_hanging` |
| norm batches | 2,075 |

关键文件：

- `data/harvestmem_v3_dataset_20261110/meta/episodes.json`
- `data/harvestmem_v3_dataset_20261110/meta/build.json`
- `data/harvestmem_v3_dataset_20261110/meta/stats.json`
- `data/harvestmem_v3_dataset_20261110/meta/norm_stats.json`

中途第一次 shard 启动因 `OPENPI_DATA_HOME` 指向错误目录，8 个 shard 都报缺少 `pi05_vision_encoder/siglip_params.pkl`；已改为复用本机已有 cache `/media/fulei/fuleivla/acm_vla/cache/openpi` 后完整重跑通过。该问题不改变数据和任务定义，失败日志保留在 `results/harvestmem_v3_S1/driver_logs/build.log`。

## 6. 抽查示范

### Count clean episode 0

- 子目标序列：observe → pick `<100, 110>` → place → pick `<100, 110>` → place → return and hold。
- 关键帧：`results/harvestmem_v3_S1/replays/train_clean/count/ep0_attempt0/before_grasp_front_camera.png`、`release_front_camera.png`、`finish_hold_front_camera.png`。

### Occluded clean episode 0

- 子目标序列：observe → pick red `<80, 138>` → place → pick red `<110, 130>` → place → pick red `<151, 136>` → place → return and hold。
- 关键帧：`results/harvestmem_v3_S1/replays/train_clean/occluded_ripe/ep0_attempt0/opening_front_camera.png`、`post_gust_front_camera.png`、`release_front_camera.png`。

### Count DART episode 100

- 与 clean episode 0 复用同一布局和 N；子目标序列相同，动作轨迹在两个采摘周期加入 DART 扰动。
- 关键帧：`results/harvestmem_v3_S1/replays/train_dart/count/ep100_attempt0/before_grasp_front_camera.png`、`release_front_camera.png`、`finish_hold_front_camera.png`。

## 7. 异常和未解决问题

- DART 的工程重试次数高于 clean，但最终 200/200 成功；重试原因和目录均保留在各 manifest 的 `failed_attempts`。
- Occluded 的 return/hold 仍只有 terminal observation 1 帧，这是“Occluded 保持原样”的结果；本轮只合并标签，没有改变任务终止定义。
- 当前数据集 builder 的 pkl 结构保留 grounded/simple 子目标，但没有把 `phase`、`target_slot`、`n_picked_so_far` 作为独立 pkl key 暴露；GT/HDF5 已完整记录，后续如训练或 tracker 需要直接读取这些字段，应使用 GT/HDF5 adapter 或在新 builder 中显式透传。
- 尚未训练 SG/K0/NM，也没有闭环成功率结果；这些属于 S2/S4，不在本阶段宣称。

## 8. 下一阶段审阅决定

S1 已完成并停下。请审阅：

1. 是否批准使用 `data/harvestmem_v3_dataset_20261110/` 进入 S2；
2. 是否接受 Occluded 的 1 帧 terminal `return and hold` 监督；
3. 是否批准开始 S2 的 3 步 smoke，然后训练 SG、K0、NM。
