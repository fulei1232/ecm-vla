"""D6 transport, clock, source-frame, position, padding and ablation-definition audit."""
from pathlib import Path
import numpy as np

from fruit_memory_diag_core import select_frames
from v_a_common import read_json, read_jsonl


def audit(directory, ablation, oracle):
    directory = Path(directory)
    row = read_json(directory/'result.json')
    raw = read_jsonl(directory/'raw_frames.jsonl')
    controls = read_jsonl(directory/'control_actions.jsonl')
    client = read_jsonl(directory/'client_ingest.jsonl');server = read_jsonl(directory/'policy_ingest.jsonl')
    decisions = read_jsonl(directory/'policy_decisions.jsonl');sent = read_jsonl(directory/'client_decisions.jsonl')
    received = read_jsonl(directory/'policy_actions.jsonl');oracles = read_jsonl(directory/'oracle_decisions.jsonl')
    errors = []
    def check(ok, why):
        if not ok: errors.append(why)
    check(row.get('executed_steps', 1301) <= 1300 and len(raw) == len(controls)+1, 'budget/raw_count')
    check(len(controls) >= 30 and all(c.get('observe_hold') for c in controls[:30]), 'physical_hold_prefix')
    if len(controls) >= 30:
        actions = np.asarray([c['action'] for c in controls[:30]])
        check(np.array_equal(actions, np.repeat(actions[:1], 30, axis=0)) and np.all(actions[:, 7] == 1), 'fixed_prefix_pose')
    check(all(r['frame_id'] == i and r['control_step'] == i and r['gt']['control_steps'] == i
              and r['policy_control_step'] == i-30 and r['exec_start_idx'] == 30 and r['is_demo'] == (i < 30)
              for i, r in enumerate(raw)), 'physical_vs_policy_clock')
    check(bool(decisions) and len(client) == len(server) == len(decisions) == len(sent) == len(received) == len(oracles), 'decision_counts')
    cursor = 0
    for i, (c, s, d, tx, rx, o) in enumerate(zip(client, server, decisions, sent, received, oracles)):
        t = 30+i*16
        check(c['first_frame_id'] == s['first_frame_id'] == cursor, f'ingest_start/{i}')
        check(c['last_frame_id'] == s['last_frame_id'] == d['available_max_frame_id'] == t, f'ingest_end/{i}')
        check(c['samples'] == s['samples'] == (31 if i == 0 else 16), f'ingest_samples/{i}')
        check(d['control_step'] == i*16 and d['exec_start_idx'] == 30 and d['history_count'] == t+1, f'policy_clock/{i}')
        check(tx['input_sha256'] == rx['input_sha256'] and tx['actions_sha256'] == rx['actions_sha256'], f'action_transport/{i}')
        for f in s['frames']:
            k = f['frame_id']
            check(f['front_sha256'] == raw[k]['front_sha256'] and f['state_sha256'] == raw[k]['state_sha256'], f'raw_source/{i}/{k}')
        expected = select_frames(t, ablation)
        check(d['selected_frame_ids'] == expected, f'ablation_selection/{i}')
        check(d.get('memory_ablation') == o.get('memory_ablation') == ablation, f'ablation_flag/{i}')
        if ablation is not None:
            check(o['selected_frame_ids'] == expected, f'oracle_selected/{i}')
        check(d['source_position_frame_ids'] == expected, f'original_positions/{i}')
        check(d['selected_front_sha256'] == [raw[k]['front_sha256'] for k in expected], f'selected_images/{i}')
        check(d['selected_control_steps'] == [raw[k]['policy_control_step'] for k in expected], f'position_clock/{i}')
        check(d['valid_tokens'] == len(expected)*d['tokens_per_frame'] and d['tokens_per_frame'] == 16, f'valid_tokens/{i}')
        check(d['mask'] == [True]*d['valid_tokens']+[False]*d['padding_tokens']
              and d['valid_tokens']+d['padding_tokens'] == d['token_budget'] == 512, f'padding/{i}')
        cursor = t+1
    normal = None
    lifecycle = [r for r in read_jsonl(directory/'policy_lifecycle.jsonl') if r.get('kind') == 'reset']
    check(len(lifecycle) == 1 and lifecycle[0]['model_seed'] == row['model_seed'], 'model_seed_reset')
    if ablation is None:
        from audit_fruit_rescue_memory import audit as original_audit
        normal = original_audit(directory, oracle)
        check(normal['passed'], 'original_normal_audit')
    return {'passed': not errors, 'errors': errors, 'decisions': len(decisions), 'ablation': ablation,
        'physical_clock_definition': 'frame_id==physical_control_step; policy clock is physical minus30',
        'normal_baseline_audit': normal}
