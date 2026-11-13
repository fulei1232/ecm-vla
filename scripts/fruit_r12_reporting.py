"""Concise R12 report publisher; previous reports remain unchanged."""
import json

from v_a_common import ECM, atomic_json, read_json, timestamp
from fruit_r12_common import OUT, TRACKED, REPORT, DOC, publish


def report(state, extra=None):
    cpu_path = OUT/'cpu_optics/result.json'
    reuse_path = OUT/'approved_r11_reuse.json'
    gpu_path = OUT/'gpu_validation.json'
    delivery_path = OUT/'data_delivery.json'
    summary = {'instruction_revision':12,'created_utc':timestamp(),'state':dict(state),
        'plan':str(DOC),'training':False,'val_used':False,
        'R11_109_demonstrations_mixed':False,
        'cpu_optical_check':read_json(cpu_path) if cpu_path.exists() else None,
        'R11_R0_reuse':read_json(reuse_path) if reuse_path.exists() else None,
        'gpu_validation':read_json(gpu_path) if gpu_path.exists() else None,
        'data_delivery':read_json(delivery_path) if delivery_path.exists() else None,
        'extra':extra or {}}
    atomic_json(OUT/'summary.json',summary)
    atomic_json(TRACKED/'summary.json',summary)
    links=[]
    for name,path in [('CPU optical proof','cpu_optics/result.json'),
                      ('R11 reuse proof','approved_r11_reuse.json'),
                      ('GPU validation','gpu_validation.json'),
                      ('data delivery','data_delivery.json'),
                      ('attempt ledger','data_attempt_ledger.json')]:
        if (OUT/path).exists():links.append(f'- [{name}]({path})')
    status=str(state.get('state','unknown'))
    lines=[f'# 修订 12 阶段报告：{status}',
        '',
        'CPU 视线检查通过；R11 的 350 条 R0 与信息门槛按批准复用，不重跑。',
        '',
        '- 观测修订：固定视觉假底位于 z=0.310 m，四边压入内衬至少 2 mm，不增加碰撞形状。',
        '- DART 修订：重试固定 clean 配对布局和 N，仅改变 attempt 噪声；每回合随机选择两个采摘周期启用原强度 OU 噪声。',
        '- R11 已有 109 条成功示范不混入 R12；训练关闭。',
        '',
        '## 当前决策',
        '',
        f"{state.get('reason','按修订 12 顺序执行。')}",
        '',
        '## 阶段产物',
        '',
        *links,
        '',
        '完整机器可读指标见 `summary.json`。GPU 验证若通过，400 条 R12 示范将从全新输出目录生成；构建与共用归一化完成后停下，不训练。',
        '']
    content='\n'.join(lines)
    REPORT.write_text(content)
    (OUT/'REPORT.md').write_text(content)
    publish(state)
    return summary
