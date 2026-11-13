"""Revision10 progress and delivery, with the original R9 failure retained."""
from v_a_common import read_json,timestamp
from fruit_r10_common import OUT,TRACKED,REPORT,DOC,proof,preserve


def report(state,support):
    rows=support.collect();cohorts={}
    for level in support.COHORTS:
        own=[r for r in rows if r['level']==level]
        checked=sum(r['guard_checked_steps'] for r in own);clipped=sum(r['guard_clipped_steps'] for r in own)
        cohorts[level]={'episodes':len(own),'successes':sum(r['success'] for r in own),
            'report_only':level in support.REPORT_ONLY,
            'passed':None if level in support.REPORT_ONLY else support.gate(level,own),
            'guard_checked_steps':checked,'guard_clipped_steps':clipped,
            'guard_clip_rate':clipped/checked if checked else None}
    extra={name:read_json(OUT/name) if (OUT/name).exists() else None for name in
        ('performance_reassessment_r10.json','budget_check.json','approved_reuse.json',
         'clock_fit.json','information_audit.json','other_checks.json','R0_gate.json',
         'data_phase_timing.json','data_generation_timing.json','data_delivery.json')}
    preserve()
    value={'revision':10,'task_definition_revision':9,'created_utc':timestamp(),'state':state,
        'count_R0_replays':len(rows),'count_R0_target':350,'cohorts':cohorts,'checks':extra,
        'passed_checks_and_pressure_not_rerun':True,'old_R9_failure_retained_unchanged':True,
        'D5_D6_R7_R8_R9_artifacts_unchanged':True,'training':False,'val150_199_used':False,
        'cases':[{k:r[k] for k in ('task','level','episode','N','success','outcome','steps','source')} for r in rows]}
    proof('summary.json',value)
    stopped=state['state'].startswith('R10_');done=state['state']=='awaiting_data_review'
    perf=extra['performance_reassessment_r10.json'];budget=extra['budget_check.json']
    info=extra['information_audit.json'];delivery=extra['data_delivery.json']
    lines=[f"修订10状态：{state['state']}；{'已停止等待审阅' if stopped or done else '准备/执行中，未宣称全部通过'}。",
        f"既有20条按每步平均耗时重判通过，最大比例{perf['maximum_ratio']:.6f}≤2；旧整体耗时失败记录保持，不重跑。",
        f"既有预算汇总通过：专家最慢{budget['max_finish_step_including_hold']}步、余{budget['minimum_remaining_steps']}步，不停摘最迟{budget['max_Nplus1_detach_step']}步摘第11颗；保持2800步。",
        f"新计数主队列{len(rows)}/350；信息门槛={info['passed'] if info else '未执行'}；数据={'已完成400条及构建/归一化' if delivery else '尚未完成'}。",
        'D5/D6与R8/R9历史产物保持；复用已批准证据，不训练、不写论文、不使用val150–199。','',
        '## 本修订的固定执行规则','',DOC.read_text(),'',
        '## 计数主队列','', '| 条件 | 成功 | 门槛 | 通过 | 裁剪步/检查步 |', '|---|---:|---|---|---:|']
    for level,c in cohorts.items():
        gate='只报告' if c['report_only'] else '≥45/50' if level=='history' else '≤15/50'
        lines.append(f"| {level} | {c['successes']}/{c['episodes']} | {gate} | {c['passed']} | {c['guard_clipped_steps']}/{c['guard_checked_steps']} |")
    lines+=['', '## 信息可得性','']
    if info:
        lines+=['| 条件/N段 | 全采摘覆盖 | 比例 | 严定义全覆盖 |','|---|---:|---:|---:|']
        for condition,bands in info['conditions'].items():
            for band,r in bands.items():lines.append(f"| {condition}/{band} | {r['all_broad']}/{r['episodes']} | {r['broad_fraction']} | {r['all_strict']}/{r['episodes']} |")
        lines+=['',f"门槛：{info['criteria']}。专家信息上限不等于模型已学会使用。"]
    else:lines.append('未执行，不计为通过。')
    lines+=['','## 数据耗时与评测成本参考','']
    if delivery:
        costs=delivery['data_generation_timing'];phase=costs['data_stage_wall_times']
        lines+=[f"原始示范生成总墙钟{phase['raw_generation_wall_seconds']:.3f}秒；包含构建、归一化、检查和统计的完整数据阶段总墙钟{phase['total_wall_seconds']:.3f}秒。这两个并行墙钟值不等于逐回合耗时之和。",'',
            '| 任务/N/范围 | 回合数 | 平均进程耗时(s) | 标准差 | 范围(s) |','|---|---:|---:|---:|---|']
        for task,groups in costs['by_task_and_N'].items():
            for group in ('all_attempts','accepted','clean_accepted','DART_accepted'):
                for N,v in groups[group].items():
                    d=v['process_wall_seconds']
                    lines.append(f"| {task}/{N}/{group} | {d['n']} | {d['mean']} | {d['std']} | {d['range']} |")
        lines+=['','进程耗时包含初始化、专家规划、仿真、渲染和记录；正式VLA评测另有模型推理开销。所有尝试和成功交付分别统计，不用成功子集代替完整尝试成本。']
    else:lines.append('数据阶段尚未完成；总墙钟及按N回合耗时将在实际生成后报告，不预填为0。')
    lines+=['','## 证据与决定','',
        f'[性能新判定]({TRACKED}/performance_reassessment_r10.json)、[逐步耗时CSV]({TRACKED}/per_step_timing.csv)、[原预算函数汇总]({TRACKED}/budget_check.json)。',
        f'[复用清单]({TRACKED}/approved_reuse.json)、[机器摘要]({TRACKED}/summary.json)。',
        f"当前原因：{state.get('reason','按指定顺序推进，任何失败停止')}。",
        '全部门槛通过才解锁数据；数据完成即停，不启动训练。旧失败仅由本修订的新规则另作判定，原记录未改。','']
    text='\n'.join(lines);REPORT.write_text(text);(OUT/'REPORT.md').write_text(text)
    return value
