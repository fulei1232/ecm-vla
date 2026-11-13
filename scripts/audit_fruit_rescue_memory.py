"""Stage3 prefix-aware audit; original source frames and physical clock retained."""
from pathlib import Path
import numpy as np

from run_teacher_oracle import selection_audit
from v_a_common import read_json,read_jsonl


def audit(directory,oracle):
    row=read_json(directory/'result.json');raw=read_jsonl(directory/'raw_frames.jsonl')
    client=read_jsonl(directory/'client_ingest.jsonl');server=read_jsonl(directory/'policy_ingest.jsonl')
    decisions=read_jsonl(directory/'policy_decisions.jsonl');sent=read_jsonl(directory/'client_decisions.jsonl')
    received=read_jsonl(directory/'policy_actions.jsonl');controls=read_jsonl(directory/'control_actions.jsonl')
    errors=[]
    def check(ok,why):
        if not ok: errors.append(why)
    check(row.get('executed_steps',1301)<=1300,'total_budget')
    check(len(controls)>=30 and all(c.get('observe_hold') for c in controls[:30]),'physical_prefix_steps')
    if len(controls)>=30:
        actions=np.asarray([c['action'] for c in controls[:30]])
        check(np.array_equal(actions,np.repeat(actions[:1],30,axis=0)) and np.all(actions[:,7]==1),'fixed_prefix_pose_open_gripper')
    check(all(r['control_step']==i and r['exec_start_idx']==30 and r['is_demo']==(i<30) for i,r in enumerate(raw)),
          'raw_clock_and_prefix_flags')
    check(bool(decisions) and len(client)==len(server)==len(decisions)==len(sent)==len(received),'decision_record_counts')
    cursor=0
    for i,(c,s,d,tx,rx) in enumerate(zip(client,server,decisions,sent,received)):
        maximum=30+i*16
        check(c['first_frame_id']==s['first_frame_id']==cursor,f'ingest_start/{i}')
        check(c['last_frame_id']==s['last_frame_id']==d['available_max_frame_id']==maximum,f'ingest_end/{i}')
        check(c['samples']==s['samples']==(31 if i==0 else 16),f'ingest_samples/{i}')
        check(d['exec_start_idx']==30 and d['control_step']==i*16 and d['history_count']==maximum+1,f'policy_clock/{i}')
        check(tx['input_sha256']==rx['input_sha256'] and tx['actions_sha256']==rx['actions_sha256'],f'websocket_hashes/{i}')
        for frame in s['frames']:
            k=frame['frame_id'];check(frame['front_sha256']==raw[k]['front_sha256'] and frame['state_sha256']==raw[k]['state_sha256'],f'raw_hash/{k}')
        selected=d['selected_frame_ids']
        check(d['source_position_frame_ids']==selected and d['selected_front_sha256']==[raw[k]['front_sha256'] for k in selected],f'selected_sources/{i}')
        check(d['mask']==[True]*d['valid_tokens']+[False]*d['padding_tokens'],f'padding/{i}')
        cursor=maximum+1
    formal=selection_audit([{**row,'attempt_directory':str(directory),'block':str(directory),'oracle':read_jsonl(directory/'oracle_decisions.jsonl')}],
                         oracle['capacity'],oracle.get('event'),oracle['keyframes'])
    check(formal['decisions']>0 and formal['mismatches']==0,'formal_selection_audit')
    return {'passed':not errors,'errors':errors,'mismatches':len(errors),'decisions':len(decisions),
            'physical_prefix_steps':30,'formal_selection_audit':formal}
