"""Paths and preservation for revision9; no training entry point."""
from pathlib import Path

from v_a_common import ECM, read_json, sha256, atomic_json, timestamp
from posthoc_fruit_memory_usage import tree_state

OUT = ECM / 'results/fruit_task_v2_R0_r9_20261106'
TRACKED = ECM / 'reports/fruit_task_v2_R0_r9_20261106'
REPORT = ECM / 'reports/fruit_task_v2_design_r9_20261106.md'
DOC = ECM / 'reports/fruit_task_v2_R0_r9_plan_20261106.md'
DATASET = ECM / 'data/finetune_kf/fruit_task_v2_r9_20261106'
RAW_CLEAN = ECM / 'data/fruit_task_v2_r9_clean_20261106/rm75b_eg2/train'
RAW_DART = ECM / 'data/fruit_task_v2_r9_dart_20261106/rm75b_eg2/train'
R8 = ECM / 'results/fruit_task_v2_R0_r8_20261105'
GUARD = ECM / 'results/fruit_k0v2_gate_20261102/guard_v2.json'
COHORTS = ['history', 'current', 'initial_current', 'initial_ground',
           'clock_onset', 'clock_onset_k0', 'clock']
REPORT_ONLY = {'clock_onset', 'clock_onset_k0'}


def preserve():
    for f in (OUT / 'preservation').glob('*_tree.json'):
        name = f.name[:-len('_tree.json')]
        assert tree_state(ECM / 'results' / name) == read_json(f), name
    for name, expected in read_json(OUT / 'preservation/files_sha256.json').items():
        assert sha256(name) == expected, name
    baseline = read_json(OUT / 'baseline_sources/manifest.json')
    for name, expected in baseline['files_sha256'].items():
        assert sha256(OUT / 'baseline_sources' / name) == expected, name


def publish(record, **changes):
    record.update(changes, updated_utc=timestamp())
    atomic_json(OUT / 'status.json', record)
    root = read_json(ECM / 'reports/lowlevel_rescue_status.json')
    root.update(phase='R0', instruction_revision=9, state=record['state'], updated_utc=record['updated_utc'],
                decision='Revision9: any failed gate/check stops; accepted occluded R0 is not rerun.',
                next='All R0 gates passing authorizes data generation only; stop after data, never train.')
    root['links'].update(R0_r9_report=str(REPORT), R0_r9_status=str(OUT / 'status.json'),
                         R0_r9_summary=str(TRACKED / 'summary.json'))
    atomic_json(ECM / 'reports/lowlevel_rescue_status.json', root)


def proof(name, value):
    TRACKED.mkdir(exist_ok=True)
    for d in (OUT, TRACKED):atomic_json(d / name, value)


def d5_cases():
    import json
    rows = [json.loads(x) for x in (ECM / 'results/fruit_lowlevel_v2_eval_20261104/per_episode.jsonl').read_text().splitlines()]
    return sorted([r for r in rows if r['stage'] == 'main' and r['condition'] == 'K0'
                   and r['model_seed'] == 7 and 100 <= r['episode'] < 110],
                  key=lambda r: (r['task'], r['episode']))


def recorded_distances(directory):
    import numpy as np
    rows = [__import__('json').loads(x) for x in (Path(directory) / 'event_signal.jsonl').read_text().splitlines()]
    values = []
    for row in rows:
        assert row['first_frame'] == len(values)
        values.extend(np.nan if v is None else v for v in row['distance'])
    return np.asarray(values, dtype=float)
