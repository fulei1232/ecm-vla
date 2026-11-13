# 事件记忆正式运行：中断与恢复记录

本说明是 `results/event_memory_20261025` 的附录。封存协议 `reports/event_memory_protocol_20261025.md` 未改动：sha256 仍为 `aecb367ce811…`，与 `protocol.json` 中的 `protocol_doc_sha256` 一致。

## 经过（时间为 UTC+8）

- 09:34：`progress.json` 最后一次更新，此时完成 59/180 组。
- 约 09:44：机器重启（09:59 时 uptime 为 15 分钟）。日志里没有内存不足的记录。运行中的 supervisor、策略服务和仿真进程全部终止。
- 10:00：用原命令恢复：`python3 scripts/run_teacher_oracle.py all --stage event --out results/event_memory_20261025`，日志追加到 `results/event_memory_20261025.log`。flock 锁随进程退出自动释放。
- 10:00:19：运行器按预注册规则处理未完成的组，即只要组内有 episode 没写完，整组移入 `superseded/20261025T020019/` 后整组重跑。共移走 6 组、18 个 episode 目录：

  | 组 | 移走的条件目录数 |
  |---|---|
  | BinFill/ep03_seed7 | 5 |
  | BinFill/ep10_seed7 | 2 |
  | BinFill/ep11_seed9 | 5 |
  | PatternLock/ep00_seed9 | 3 |
  | VideoUnmaskSwap/ep06_seed8 | 2 |
  | VideoUnmaskSwap/ep17_seed8 | 1 |

  清单记录在 `superseded.jsonl`。这些目录不参与分析。
- 恢复前检查了已完成部分：3052 个 episode json 文件和全部 jsonl 行都能正常解析。
- 10:48：180/180 组完成，分析和报告由同一命令按封存规则自动生成。门槛全部通过：G3 工程错误率 0%，G5 同进程，GA 选帧审计 14428 个决策，不一致 0 个。

## 对结论的影响

被作废的组是整组重跑的：同一组的所有条件使用相同的种子和 episode，配对结构不变。重跑的组不是按结果挑选的，而是由"中断时尚未写完"这一点决定，与结果无关。判定规则、对比族和统计方法都没有改动。
