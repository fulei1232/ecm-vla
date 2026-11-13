"""Chinese stage reporting for the independent fixed-shield continuation."""
from v_a_common import read_json, timestamp
from fruit_shield_v2_common import OUT, TRACKED, REPORT, DOC, proof, preserve


def report(state,support):
    rows=support.collect();cohorts={}
    for level in support.COHORTS:
        own=[r for r in rows if r['level']==level]
        checked=sum(r['guard_checked_steps'] for r in own);clipped=sum(r['guard_clipped_steps'] for r in own)
        cohorts[level]={'episodes':len(own),'successes':sum(r['success'] for r in own),
            'report_only':level in support.REPORT_ONLY,
            'passed':None if level in support.REPORT_ONLY else support.gate(level,own),
            'guard_checked_steps':checked,'guard_clipped_steps':clipped,
            'guard_clip_rate':clipped/checked if checked else None,
            'early_start':sum(bool(r['early_start_steps']) for r in own),
            'budget_exact_N':sum(r['budget_exhausted_exact_N'] for r in own),
            'max_visibility':{cam:{key:max((r['max_visibility'][cam][key] for r in own),default=0)
                for key in ('basket','basket_fruit','inactive_reserve')} for cam in ('front_camera','hand_camera')}}
    checks={name:read_json(OUT/name) if (OUT/name).exists() else None for name in
        ('new_tests.json','optical_geometry_check.json','approved_reuse.json','ep112_diagnostic.json','clock_fit.json',
         'information_audit.json','other_checks.json','R0_gate.json','data_phase_timing.json','data_delivery.json')}
    progress={}
    for purpose in ('train_clean','train_dart'):
        for f in (OUT/'replays'/purpose).glob('*/*/ep*/validity.json'):
            r=read_json(f.parent/'result.json');key=r['task']+'/'+purpose
            group=progress.setdefault(key,{'completed_attempts':0,'successful_episodes':0})
            group['completed_attempts']+=1;group['successful_episodes']+=bool(r['success'])
    preserve()
    summary={'created_utc':timestamp(),'state':state,'condition':'count_fixed_shield_v2/occluded_original',
        'count_R0_replays':len(rows),'count_R0_target':350,'cohorts':cohorts,'checks':checks,
        'data_progress':progress,'old_R10_83_episodes_archived_not_mixed':True,
        'all_previous_artifacts_unchanged':True,'occluded_R0_not_rerun':True,
        'training':False,'val150_199_used':False,
        'cases':[{k:r[k] for k in ('task','level','episode','N','success','outcome','steps','source')} for r in rows]}
    proof('summary.json',summary)
    diag=checks['ep112_diagnostic.json'];info=checks['information_audit.json'];delivery=checks['data_delivery.json']
    diagnostic_label=diag['passed'] if diag else ('已启动但未通过完整诊断' if (OUT/'commands/ep112_diagnostic/command.json').exists() else '未执行')
    lines=[f"固定遮挡v2续跑：{state['state']}；{'停止等待审阅' if state['state'].startswith('SHIELD_') or state['state']=='awaiting_data_review' else '准备/执行中，未宣称完成'}。",
        f"ep112原动作对照及原种子完整DART诊断：{diagnostic_label}；不放宽零像素门槛。",
        f"新计数主队列{len(rows)}/350；信息门槛：{info['passed'] if info else '未执行'}；遮挡任务R0不重跑。",
        f"新数据成功{sum(v['successful_episodes'] for v in progress.values())}/400；构建和归一化：{'完成' if delivery else '尚未完成'}；不训练。",
        '旧83条示范、原失败及D5/D6/R7–R10产物保持；新观测数据另建目录，不使用val150–199、不写论文。','',
        '## 做了什么与固定规则','',DOC.read_text(),'',
        '## 计数主队列','', '| 条件 | 成功/完成 | 通过 | 裁剪步/检查步 |','|---|---:|---|---:|']
    for level,c in cohorts.items():
        lines.append(f"| {level} | {c['successes']}/{c['episodes']} | {c['passed'] if not c['report_only'] else '只报告'} | {c['guard_clipped_steps']}/{c['guard_checked_steps']} |")
    lines+=['','## 信息可得性','']
    if info:
        lines+=['| 条件/N段 | 全采摘覆盖 | 比例 |','|---|---:|---:|']
        for condition,bands in info['conditions'].items():
            for band,r in bands.items():lines.append(f"| {condition}/{band} | {r['all_broad']}/{r['episodes']} | {r['broad_fraction']} |")
    else:lines.append('尚未执行。')
    lines+=['','## 数据时间与成本','']
    if delivery:
        cost=delivery['data_generation_timing'];phase=cost['data_stage_wall_times']
        lines += [f"原始示范并行总墙钟{phase['raw_generation_wall_seconds']:.3f}秒；完整数据流水线{phase['total_wall_seconds']:.3f}秒。",'',
            '| 任务/N/范围 | 回合数 | 平均进程秒数 | 标准差 | 范围 |','|---|---:|---:|---:|---|']
        for task,groups in cost['by_task_and_N'].items():
            for scope,bands in groups.items():
                for N,v in bands.items():
                    d=v['process_wall_seconds'];lines.append(f"| {task}/{N}/{scope} | {d['n']} | {d['mean']} | {d['std']} | {d['range']} |")
        lines+=['','并行墙钟不等于逐尝试耗时之和；生成包含规划、渲染与HDF5记录，后续VLA评测另有推理成本。']
    elif checks['data_phase_timing.json']:
        lines.append(f"数据阶段未完成，当前时间记录：{checks['data_phase_timing.json']}。")
    else:lines.append('数据阶段未开始，不预填总耗时。')
    lines += ['', '## 决策和下一步','',state.get('reason','ep112诊断通过才进入新计数审计；全部门槛通过才生成数据，生成完停下。'),
        f'完整指标和证据路径：[summary.json]({TRACKED}/summary.json)。任一门槛或一致性检查失败即停止，不自动修改、换种子或重跑。','']
    text='\n'.join(lines);REPORT.write_text(text);(OUT/'REPORT.md').write_text(text)
    return summary
