"""Stop-A report. No protocol sealing or formal evaluation side effects."""
from pathlib import Path
from v_a_common import ECM, atomic_json, read_json, sha256, timestamp


def ratio(row):
    n, d = row['numerator'], row['denominator']
    return f'{n}/{d} ({n/d:.2%})' if d else '0/0（无观测）'


def deliver_stage5(out, run, k0, g1, draft, state):
    complete = state['state'] == 'awaiting_protocol_review_A'
    t1 = {arm: read_json(p) for arm, p in [('K0', k0/'gate_t1_k0.json'),
        ('K4', run/'gate_t1_k4.json'), ('K8', run/'gate_t1_k8.json')] if p.exists()}
    validity = {arm: read_json(p) for arm in ('K4', 'K8')
                if (p := out/f'validity/{arm}/validity.json').exists()}
    smoke = {arm: read_json(p) for arm in ('K0', 'K4', 'K8')
             if (p := out/f'smoke/{arm}/validity.json').exists()}
    if complete:
        assert len(t1) == 3 and all(r['passed'] for r in t1.values())
        assert len(validity) == 2 and len(smoke) == 3
        assert all(r['passed'] for r in [*validity.values(), *smoke.values()])
    checkpoints = {'K0': str(k0/'ckpt/mme_vla_suite/k0/11999'),
                   **{arm: str(run/f'ckpt/mme_vla_suite/{arm.lower()}/11999') for arm in ('K4', 'K8')}}
    digests = {arm: {f: sha256(Path(p)/f) for f in ('params/manifest.ocdbt', 'assets/robomme/norm_stats.json', '_CHECKPOINT_METADATA')}
               for arm, p in checkpoints.items() if (Path(p)/'params/manifest.ocdbt').exists()}
    summary = {'phase': 5, 'revision': 4, 'delivery': 'stop A before sealing', 'complete_to_stop_A': complete,
        'generated_utc': timestamp(), 'state': state, 'T1': t1, 'validity': validity, 'smoke': smoke,
        'checkpoints': checkpoints, 'checkpoint_sha256': digests,
        'platform_identity_sha256': sha256(out/'platform_identity.json'),
        'platform': read_json(out/'platform_identity.json'), 'plan': read_json(out/'plan.json'),
        'evaluation_protocol_sealed': False, 'formal_evaluation_started': False, 'val_layouts_accessed': False,
        'requires_user_review': True, 'next': 'User review at A; seal only after approval.'}
    lines = [('阶段5停点A已到达：K4/K8训练、有效性和dev200冒烟全部完成。' if complete else
              '阶段5停止：有效性或工程检查未通过，保存现有证据等待审阅。'),
        'K0直接复用11999；K4/K8计划依次从79999训练12000步，沿用同一份数据、统计、种子42和8卡FSDP；实际完成见T1。',
        'T1：'+'；'.join(f'{arm}={r["passed"]}' for arm, r in t1.items())+'。',
        '有效性：'+'；'.join(f'{arm}工程错误{ratio(r["engineering_errors"])}、记忆审计{r["memory_audits_passed"]}/{r["layouts"]}'
                           for arm, r in validity.items())+'。',
        '协议为未封存草案；没有访问val布局或启动正式/敏感性评测。按§12.5停点A等待用户审阅。', '',
        '## 训练与T1','', f'来源：`{run}/plan.json`、`protocol.json`、`arms/`及`gate_t1_k4.json`、`gate_t1_k8.json`。',
        f'K0来源：`{k0}/gate_t1_k0.json`；数据/统计/源码保全摘要见`{out}/plan.json`。', '',
        '| 条件 | 日志区间 | 全部有限 | 首/末区间损失 | 统计匹配 |','|---|---:|---|---|---|']
    for arm, gate in t1.items():
        row = gate['arms'][arm.lower()]
        lines.append(f'| {arm} | {row["logged"]}/{row["expected_logged"]} | {row["finite"]} | {row["first_loss"]}/{row["last_loss"]} | {row.get("norm_stats_match")} |')
    lines += ['', '## 开发集有效性与性能记录','',
        '只用dev200–219、两个任务、种子7；性能只记录，不调整设置。每项完整统一指标均在配套summary.json/validity/<条件>/performance中，含合计与分任务分母。', '',
        '| 条件/任务 | 成功 | 首颗入篮 | 腕偏离执行/输出 | 工程错误 |','|---|---:|---:|---|---:|']
    for arm, v in validity.items():
        for name, r in [('合计', v['performance']['all']), *v['performance']['by_task'].items()]:
            lines.append(f'| {arm}/{name} | {ratio(r["success"])} | {ratio(r["first_target_in_basket"])} | {ratio(r["wrist_deviation_executed"])}/{ratio(r["wrist_deviation_proposed"])} | {ratio(r["engineering_error"])} |')
    lines += ['', '## 逐回合事件触发','',
        'contact为原算法近区事件；detach关键帧为原算法release，即第一次离开近区，是脱果代理。真值实际脱果另列；不把代理与物理事件混同。'
        '事件槽占比的分母为所有决策的有效帧槽，排除padding；另报告均匀帧偶然包含的事件帧。事件选择逐决策按当时可见历史重算。', '',
        '| 条件/任务/布局 | contact | detach代理 | 实际脱果 | 事件槽/有效槽 | 零事件解释 |','|---|---:|---:|---:|---:|---|']
    for arm, v in validity.items():
        for r in v['event_statistics']['per_episode']:
            lines.append(f'| {arm}/{r["task"]}/{r["episode_id"]} | {r["contact_events"]} | {r["detach_event_keyframes"]} | {r["physical_detach_events"]} | {ratio(r["designated_event_slot_fraction"])} | {r["zero_event_slot_explanation"] or "—"} |')
    lines += ['', f'逐决策证据：`{out}/validity/<条件>/<任务>/ep<ID>/event_statistics.json`；汇总和所有零事件解释见相邻`event_summary.json`。', '',
        '## 冒烟（dev200）','', '| 条件 | 完整回合 | 记忆审计 | 审计决策 | 工程错误 |','|---|---:|---:|---:|---:|']
    for arm, v in smoke.items():
        lines.append(f'| {arm} | {v["layouts"]}/2 | {v["memory_audits_passed"]}/2 | {v["audited_decisions"]} | {ratio(v["engineering_errors"])} |')
    lines += ['', f'冒烟来源：`{out}/smoke`；每个任务的三个条件在同一服务会话内，环境种子及模型随机数重置相同。', '',
        '## 平台与模型SHA256', '', '| 文件 | SHA256 |', '|---|---|',
        f'| {out}/platform_identity.json | {summary["platform_identity_sha256"]} |']
    for path, digest in summary['platform']['files_sha256'].items():
        lines.append(f'| {path} | {digest} |')
    for arm, files in digests.items():
        for name, digest in files.items():
            lines.append(f'| {checkpoints[arm]}/{name} | {digest} |')
    lines += ['',
        '## 决策与下一步','',
        ('按§12.5到达停点A，交付T1、有效性、冒烟及协议草案；等待用户审阅后再封存。' if complete else
         f'停止原因：{state.get("error")}。保留所有结果；没有基于性能改变设置或自动重试失败检查。'),
        f'协议：`{draft}`；目前未封存。主检验4项Holm、布局聚类bootstrap10000次、失败分类和不开包络敏感性分析均已预定义。', '',
        '当前未执行val100–149；未启动正式900条或敏感性300条评测，未写入论文。']
    body = '\n'.join(lines)+'\n'
    (out/'report.md').write_text(body)
    # Append only evidence; the preregistered definitions above the marker do not change.
    text = draft.read_text()
    begin, end = '<!-- STOP_A_RESULTS_BEGIN -->', '<!-- STOP_A_RESULTS_END -->'
    assert text.count(begin) == text.count(end) == 1
    evidence = '\n\n'+body+f'\n平台清单SHA256：`{summary["platform_identity_sha256"]}`。\n'
    text = text[:text.index(begin)+len(begin)] + evidence + text[text.index(end):]
    draft.write_text(text)
    summary['draft_sha256_at_stop_A'] = sha256(draft)
    atomic_json(out/'summary.json', summary)
    report = ECM/'reports/lowlevel_rescue_D5_A_20261103'
    report.mkdir(parents=True, exist_ok=True)
    atomic_json(report/'summary.json', summary)
    (ECM/'reports/lowlevel_rescue_D5_A_20261103.md').write_text(body)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root['links'].update(D5_A_report=str(ECM/'reports/lowlevel_rescue_D5_A_20261103.md'),
        D5_A_summary=str(report/'summary.json'), D5_draft=str(draft))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)
