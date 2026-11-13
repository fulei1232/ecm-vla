# S0 审阅意见（Claude，2026-11-10）

**结论：通过，批准进入 S1。** 报告扎实，统计口径清楚，风险列得全。symbolic 方案 (b) 不需要再做加载冒烟，直接放弃。

## 对报告第 7 节 4 个问题的答复

1. **方案 (a)：批准**，这是 SG 唯一的子目标输入路径。拼接格式固定为 `<instruction lowercase> current subgoal: <grounded_subgoal>.`，训练和服务端共用同一个函数，放在 `scripts/harvestmem_v3/subgoal.py`。S1 构建完成后，用最长的指令加最长的子目标测一次 token 数（max_token_len=64），确认没有被截断，并在报告里写出这个数字。
2. **坐标：批准。** 用 256×256 初始前置帧的像素坐标，目标身份绑定初始帧的 slot。**GT 标签一律用 3D 果心 + 相机内外参投影来算**，不用分割质心，这样结果确定，也不依赖 mask。部署时 SAM-FDM 的质心偏差在 S3 单独测量。量化后坐标撞车的处理方案同意。
3. **skip / none：批准**，只作为 tracker 内部状态，不进入低层训练标签。`observe the fruit tree` 可以保留，前 30 帧本来就只作历史、不参与监督。

## 补充要求（S1 必须落实）

4. **合并"回到原位"和"静止等待"。** 我查了 R12 的 Count 示范，`return to rest to finish` 每回合只有大约 3 帧（例如 ep3 的 1150→1153），因为放果点就在 rest 附近。这么稀疏的监督学不会。现改为**一个子目标** `return to rest and hold still`，覆盖从最后一次 settle 结束到回合终止的所有帧，包括 finish 保持段和 terminal hold。集合更新为：`pick the fruit at <x, y>`、`pick the red fruit at <x, y>`、`place the fruit in the basket`、`return to rest and hold still`，外加只用于历史帧的 `observe the fruit tree`。
5. **子目标切换时刻要和 tracker 规则一致**，按现有示范的实际时序定义如下：
   - pick → place：目标果**脱离果枝**（`attached` 变为 True）那一帧，对应示范里 pull→carry 的切换；
   - place → 下一个 pick 或 return：本次**入篮确认之后**，即 release + settle 结束、`n_in_basket` 加 1 之后；
   - 选目标的规则和专家完全一致。Count 取 y 最大的挂果，补果后通常还是同一个 slot，所以连续几个 pick 子目标的坐标相同，这是正常的。
   
   S1 报告要统计 GT 子目标转移和上面规则的逐帧一致率，目标是 100%。这样 S3 的 tracker 就可以直接对照。
6. **Count 去掉等灯以后**，观察阶段结束马上进入第一个 pick，不再有 `wait_for_green` 段。S1 报告要列出各个子目标的帧数分布，重点看 `return to rest and hold still` 每回合有多少帧。
7. **坐标噪声**：S2 用独立配置，±4 px，裁剪在图像边界内。不要复用 vendor 默认的 ±8。

## 不需要做的

- symbolic 整包加载冒烟：不做。
- 对 K0 / NM 的 prompt 做任何改动：不做，两者保持原指令。

进入 S1，完成后停下并交付 `reports/harvestmem_v3_S1.md`。
