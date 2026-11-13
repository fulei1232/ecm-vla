#!/usr/bin/env python3
"""Prepare revision5 artifacts, then attest one committed protocol payload.

The protocol records the committed payload hash and commit. Its full-file hash
is in the sidecar seal, avoiding an impossible full-file self-reference.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from fruit_lowlevel_v2_eval_common import OUT, DOC, SEAL, PLATFORM, STAGE5, G1, BEGIN, END, payload_sha
from v_a_common import ECM, POLICY, atomic_json, read_json, sha256, timestamp


def prepare_protocol():
    text = DOC.read_text()
    if '## 11. 修订5分类试运行' in text:
        a, b = text.index('## 11. 修订5分类试运行'), text.index('## 12. 封存登记')
        trial = (OUT/'dev_classification_trial/report.md').read_text()
        DOC.write_text(text[:a]+'## 11. 修订5分类试运行\n\n'+trial+
            f'\n全部逐例证据及输入摘要：`{OUT}/dev_classification_trial/summary.json`。未使用val布局，未基于结果修改定义。\n\n'+text[b:])
        return
    prefix, historical = text.split('## 10. 停点A结果与冒烟记录', 1)
    prefix = prefix.replace('# 低层修复后采摘任务：K0/K4/K8正式评测协议草案', '# 低层修复后采摘任务：K0/K4/K8正式评测协议（修订5）')
    a = prefix.index('日期：');b = prefix.index('\n## 1.')
    prefix = prefix[:a]+('日期：2026-11-04。停点A已获用户批准，按HANDOFF §13（修订5）执行。\n'
        '分类试运行通过后直接封存并执行主评测900条及无包络敏感性300条，到停点B交付；有效性失败立即停止。\n'
        '封存状态、正文提交和摘要见第12节登记区；完整文件SHA256以配套fruit_lowlevel_v2_seal_20261104.json为准，运行前核验。\n'
        '协议正文哈希排除封存登记区，以避免自引用；实际正文提交、分析代码和平台摘要均固定。\n')+prefix[b:]
    prefix = prefix.replace('训练专用配置文件fruit_lowlevel_v2_training_20261103.md及运行protocol.json只记录训练来源，不等于正式评测协议已封存。',
        '训练专用配置文件fruit_lowlevel_v2_training_20261103.md及运行protocol.json只记录训练来源；正式评测单独按本协议封存。')
    prefix = prefix.replace('训练、有效性或冒烟失败就保留证据并停下；全部通过也在§12.5停点A交用户审阅，不自动封存或正式评测。',
        '上述停点A流程已完成，用户在修订5中批准；分类试运行通过后按§13.4直接封存并正式评测，不再等待A。')
    old = '不得用val预览、调参、检查点挑选或冒烟。停止原因、超时、掉果、所有工程错误均保留；不删失败，不填补缺组，不按结果挑选重跑。'
    new = ('不得用val预览、调参、检查点挑选或冒烟。任务失败、超时、掉果及所有工程错误均保留，不删结果或填补缺组。\n'
        '客户端单次进程墙钟上限1800秒，服务装载上限1800秒；与1300物理控制步预算无关，墙钟超时仍先核查已有结局。\n'
        '工程重跑仅限尚无任务结局的崩溃/进程超时/审计无法运行等工程中断，同一环境种子、模型种子与条件，最多重跑2次（含初次最多3次）。\n'
        '已经产生success/fail/timeout的回合一律不重跑；即使result.json缺失，也从控制日志或最后物理真值识别已有结局。已有结局后审计失败直接判有效性失败。\n'
        '所有尝试（包括最终采纳尝试）原样保存在superseded/并列入报告；main/sensitivity仅链接到通过审计的采纳尝试。2次重跑后仍为工程错误就停止。\n'
        '同组内始终要求三个条件共享服务会话；若已有条件结束后服务进程丢失，不能重跑已结束条件来恢复配对，直接有效性失败。\n'
        '客户端无结局崩溃可在同一活服务上重跑；服务崩溃且本组尚无已结束条件时，可按相同配置重启服务用于剩余重跑机会。除此不自行补救。')
    assert old in prefix;prefix = prefix.replace(old, new)
    old = '- 计数任务的记忆类：任一时刻入篮数>N；或首次达到N之后，新果实发生脱离，或产生一次新的近果事件（该事件start晚于首次达到N，近区阈值仍为3cm）。'
    new = (old+'\n- 结构性说明：环境evaluate的计数分支在入篮数恰好N时立即判成功，FruitEnvRunner与官方循环随即终止。上述“超过N”和“达到N后继续取果”结构上不会发生；保留条款用于检查，非零按环境终止异常处理，归invalid并作为工程诊断停止，不当作记忆效果。计数任务实际记忆类仅由下述提前停止代理构成。')
    assert old in prefix;prefix = prefix.replace(old, new)
    old = '- 遮挡任务的记忆类：任一帧wrong_in_basket>0（入篮果不在目标集合）；或任何不成熟果实脱离，包含之后掉落的情况。'
    new = (old+'\n- 果实只有连续抓握满DETACH_STEPS（当前为2）控制步才会脱落；未成熟果脱落必经双指抓握，不能被碰掉。分类仍归记忆类。新增只描述、不改变分类的adjacent_misgrasp_possible子标签：在该未成熟果这次成功连续抓握的第一步，它距最近的仍挂树的目标成熟果实中心<.04m。\n'
        '- 起点为首次脱果帧 − (DETACH_STEPS−1)（当前为脱果前一帧）；逐步日志的该帧给出果实位置和挂树目标集合。无候选目标时距离记null、子标签false；严格小于4cm，等于不触发。逐回合和逐未成熟果事件分别报告计数，作为可能操作误抓的限制说明，不更改互斥归类。')
    assert old in prefix;prefix = prefix.replace(old, new)
    prefix = prefix.replace('分类代码为scripts/fruit_lowlevel_v2_metrics.py；封存时纳入分析代码摘要。',
        '基础分类代码scripts/fruit_lowlevel_v2_metrics.py保持停点A原样；scripts/fruit_lowlevel_v2_metrics_r5.py逐回合调用它，追加修订5的结构异常处理及描述子标签。两版及统一指标代码均纳入分析摘要。')
    old = '敏感性分析非确证性，不进入Holm：'
    prefix = prefix.replace(old, '事件召回诊断（只报告、不调阈值）：每条件、每任务列“距离代理脱离事件数/真值脱果数”，取完整物理回合，不限已被服务观察的前缀。它是事件数比，不做匹配、不截断到1；分母为0记无定义。停点A开发集K4为80/91、K8为86/95。\n\n'+old)
    prefix = prefix.replace('用户审阅通过后再封存：填写封存记录和提交哈希，保存协议与全部依赖摘要的封存清单，运行前验证；当前不执行该步骤。',
        '停点A已审阅通过；按§13.4完成条款修改及dev分类试运行后直接封存，保存正文提交、协议完整文件SHA256、分析与平台清单SHA256及全部依赖，运行前逐项验证。封存后按原规则运行900+300条，出现有效性失败就停止汇报。')
    trial = (OUT/'dev_classification_trial/report.md').read_text()
    DOC.write_text(prefix+'## 10. 历史停点A结果与冒烟记录（以下为当时尚未封存的原始交付）'+historical+
        '\n## 11. 修订5分类试运行\n\n'+trial+
        '\n逐条件、分任务的全部分类子标签及逐例输入摘要见dev_classification_trial/summary.json；未成熟果抓握起点均可由既有日志计算。未使用val布局，结果不用于修改定义。\n'+
        '\n## 12. 封存登记\n\n'+BEGIN+'\n待登记已提交的协议正文；该区不计入正文哈希。\n'+END+'\n')


def prepare():
    from run_fruit_lowlevel_v2_stage5 import verify_inputs
    verify_inputs(read_json(STAGE5/'plan.json'))
    assert read_json(OUT/'dev_classification_trial/summary.json')['passed']
    assert read_json(STAGE5/'summary.json')['complete_to_stop_A']
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_protocol()
    schedule = []
    rng = np.random.default_rng(20261103)
    for stage, seeds in (('main', (7, 8, 9)), ('sensitivity', (7,))):
        for seed in seeds:
            for episode in range(100, 150):
                for task in ('FruitCount', 'FruitOccludedRipe'):
                    schedule.append({'stage': stage, 'task': task, 'episode': episode, 'seed': seed,
                        'condition_order': rng.permutation(['K0', 'K4', 'K8']).tolist()})
    atomic_json(OUT/'schedule.json', schedule)
    original = read_json(PLATFORM)
    extra = [ECM/'scripts/run_fruit_lowlevel_v2_eval.py', ECM/'fruit_picking/env.py',
        ECM/'fruit_picking/agent.py', ECM/'fruit_picking/layout.py',
        POLICY/'src/mme_vla_suite/policies/policy.py']
    platform = {'base_platform_identity_sha256': sha256(PLATFORM),
        'base_platform': original, 'files_sha256': {**original['files_sha256'], **{str(p): sha256(p) for p in extra}},
        'main': {'guard': 'v2', 'exec_horizon': 16, 'observe_hold_steps': 30, 'total_step_budget': 1300},
        'sensitivity': {'guard': 'off', 'other_platform_settings_identical': True},
        'seed_routing': 'One unchanged policy server per seed7/8/9; three checkpoints per process; simulator GPU3/4/5.'}
    atomic_json(OUT/'platform_manifest.json', platform)
    sources = [ECM/'scripts'/n for n in ('analyze_fruit_failure_modes.py', 'fruit_lowlevel_v2_metrics.py',
        'fruit_lowlevel_v2_metrics_r5.py', 'fruit_lowlevel_v2_statistics.py', 'fruit_lowlevel_v2_eval_common.py',
        'run_fruit_lowlevel_v2_eval.py', 'report_fruit_lowlevel_v2_eval.py', 'seal_fruit_lowlevel_v2_eval.py')]
    analysis = {'files_sha256': {str(p): sha256(p) for p in sources}, 'bootstrap_draws': 10000,
        'permutation_draws': 100000, 'Holm_family_size': 4, 'base_stats_seed': 20261103,
        'classification_revision': 5, 'r4_classifier_preserved': True}
    atomic_json(OUT/'analysis_manifest.json', analysis)
    registry = read_json(ECM/'reports/split_registry_20261102.json')
    vals = [r for r in registry['manifests'] if r['split'] == 'sealed_val']
    assert len(vals) == 100 and {r['episode'] for r in vals} == set(range(100, 150))
    approved = read_json(STAGE5/'summary.json')
    checkpoints = approved['checkpoints']
    files = {**analysis['files_sha256'], **platform['files_sha256']}
    for arm, path in checkpoints.items():
        for name, digest in approved['checkpoint_sha256'][arm].items():
            assert sha256(Path(path)/name) == digest
            files[str(Path(path)/name)] = digest
    for p in (OUT/'schedule.json', OUT/'platform_manifest.json', OUT/'analysis_manifest.json',
              OUT/'dev_classification_trial/summary.json', ECM/'reports/split_registry_20261102.json'):
        files[str(p)] = sha256(p)
    copies = {'schedule.json': 'fruit_lowlevel_v2_schedule_20261104.json',
        'platform_manifest.json': 'fruit_lowlevel_v2_platform_20261104.json',
        'analysis_manifest.json': 'fruit_lowlevel_v2_analysis_20261104.json',
        'dev_classification_trial/summary.json': 'fruit_lowlevel_v2_classification_trial_20261104.json',
        'dev_classification_trial/report.md': 'fruit_lowlevel_v2_classification_trial_20261104.md'}
    for source, name in copies.items():
        destination = ECM/'reports'/name
        destination.write_bytes((OUT/source).read_bytes())
        files[str(destination)] = sha256(destination)
    # The registry already commits to these manifests; their contents are not
    # opened until seal verification immediately before evaluation.
    files.update({str(ECM/r['manifest']): r['sha256'] for r in vals})
    plan = {'revision': 5, 'protocol_payload_sha256': payload_sha(DOC.read_text()),
        'analysis_manifest_sha256': sha256(OUT/'analysis_manifest.json'),
        'platform_manifest_sha256': sha256(OUT/'platform_manifest.json'),
        'files_sha256': files, 'checkpoints': checkpoints, 'val_registry': vals,
        'q7_center': read_json(G1/'expert_profile.json')['q7_center'],
        'dev_classification_trial_passed': True,
        'schedule': {'main_seeds': [7, 8, 9], 'sensitivity_seeds': [7], 'episodes': list(range(100, 150)),
                     'main_episodes': 900, 'sensitivity_episodes': 300},
        'retry_policy': 'Only no-outcome engineering aborts; same seeds/condition; at most2 retries; preserve every attempt; never retry a completed task outcome.'}
    atomic_json(OUT/'seal_plan.json', plan)
    print({'prepared_for_payload_commit': True, 'protocol_payload_sha256': plan['protocol_payload_sha256'],
           'analysis_sha256': plan['analysis_manifest_sha256'], 'platform_sha256': plan['platform_manifest_sha256']}, flush=True)


def seal(commit):
    plan = read_json(OUT/'seal_plan.json')
    if SEAL.exists():
        from fruit_lowlevel_v2_eval_common import verify_seal
        verify_seal();print('Already sealed', flush=True);return
    assert payload_sha(DOC.read_text()) == plan['protocol_payload_sha256']
    committed = subprocess.check_output(['git', 'show', f'{commit}:reports/fruit_lowlevel_v2_protocol_20261103.md'], cwd=ECM, text=True)
    assert payload_sha(committed) == plan['protocol_payload_sha256']
    for name, digest in read_json(OUT/'analysis_manifest.json')['files_sha256'].items():
        data = subprocess.check_output(['git', 'show', f'{commit}:{Path(name).relative_to(ECM)}'], cwd=ECM)
        assert hashlib.sha256(data).hexdigest() == digest
    text = DOC.read_text()
    attestation = ('\n\n封存授权：HANDOFF §13.4；分类试运行通过，无需再次审阅。\n\n'
        f'协议正文封存提交哈希：`{commit}`。\n\n'
        f'协议正文SHA256（仅排除本登记区）：`{plan["protocol_payload_sha256"]}`。\n\n'
        f'分析代码清单SHA256：`{plan["analysis_manifest_sha256"]}`。\n\n'
        f'平台清单SHA256：`{plan["platform_manifest_sha256"]}`。\n\n'
        '协议完整文件SHA256：见[配套封存清单](fruit_lowlevel_v2_seal_20261104.json)的protocol_file_sha256字段；'
        '完整文件摘要置于旁侧清单，避免把文件自身摘要写回文件造成自引用。正文提交与此登记区独立核验。\n\n')
    DOC.write_text(text[:text.index(BEGIN)+len(BEGIN)]+attestation+text[text.index(END):])
    sealed = {**plan, 'sealed_utc': timestamp(), 'payload_commit': commit,
        'protocol_file_sha256': sha256(DOC), 'protocol_doc': str(DOC)}
    atomic_json(SEAL, sealed)
    from fruit_lowlevel_v2_eval_common import verify_seal
    verify_seal()
    print({'sealed': True, 'payload_commit': commit, 'protocol_file_sha256': sealed['protocol_file_sha256']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'seal'))
    p.add_argument('--payload-commit')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        assert args.payload_commit
        seal(args.payload_commit)
