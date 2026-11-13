"""Revision5 evaluation constants and immutable seal verification."""
from pathlib import Path
import hashlib
import subprocess

from v_a_common import ECM, read_json, sha256

OUT = ECM/'results/fruit_lowlevel_v2_eval_20261104'
DOC = ECM/'reports/fruit_lowlevel_v2_protocol_20261103.md'
SEAL = ECM/'reports/fruit_lowlevel_v2_seal_20261104.json'
PLATFORM = ECM/'results/fruit_lowlevel_v2_stage5_20261103/platform_identity.json'
STAGE5 = ECM/'results/fruit_lowlevel_v2_stage5_20261103'
G1 = ECM/'results/fruit_k0v2_gate_20261102'
VAL = ECM/'data/fruit_picking/rm75b_eg2/val'
TASKS = ('FruitCount', 'FruitOccludedRipe')
CONDITIONS = ('K0', 'K4', 'K8')
VALID = frozenset(('success', 'fail', 'timeout'))
BEGIN = '<!-- SEAL_RECORD_BEGIN -->'
END = '<!-- SEAL_RECORD_END -->'


def protocol_payload(text):
    assert text.count(BEGIN) == text.count(END) == 1
    return text[:text.index(BEGIN)+len(BEGIN)]+'\n'+text[text.index(END):]


def payload_sha(text):
    return hashlib.sha256(protocol_payload(text).encode()).hexdigest()


def verify_seal():
    seal = read_json(SEAL)
    assert seal['revision'] == 5 and seal['dev_classification_trial_passed']
    assert sha256(DOC) == seal['protocol_file_sha256']
    assert payload_sha(DOC.read_text()) == seal['protocol_payload_sha256']
    for name, digest in seal['files_sha256'].items():
        assert sha256(name) == digest, name
    for name, digest in read_json(PLATFORM)['files_sha256'].items():
        assert sha256(name) == digest, name
    committed = subprocess.check_output(['git', 'show', f"{seal['payload_commit']}:reports/fruit_lowlevel_v2_protocol_20261103.md"], cwd=ECM, text=True)
    assert protocol_payload(committed) == protocol_payload(DOC.read_text())
    assert seal['schedule']['main_seeds'] == [7, 8, 9] and seal['schedule']['sensitivity_seeds'] == [7]
    assert seal['schedule']['episodes'] == list(range(100, 150))
    return seal


def terminal_outcome(directory):
    """A terminal task outcome anywhere forbids retry, even if result writing failed."""
    import json
    d = Path(directory)
    f = d/'result.json'
    if f.exists():
        try:
            result = read_json(f)
            if result.get('outcome') in VALID:
                return result['outcome']
        except (ValueError, OSError):
            pass
    f = d/'control_actions.jsonl'
    if f.exists():
        for line in f.read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('outcome') in VALID:
                return row['outcome']
    # A crash between raw-frame and control/result writes must not erase a
    # terminal physical state. Recompute the unchanged environment predicates.
    task = None
    for name in ('result.json', 'started.json'):
        if (d/name).exists():
            try:
                task = read_json(d/name).get('task') or task
            except (ValueError, OSError):
                pass
    f = d/'raw_frames.jsonl'
    if task and f.exists():
        for line in reversed(f.read_text(errors='replace').splitlines()):
            try:
                gt = json.loads(line)['gt']
                inside = gt['in_basket']
                dropped = any(a and not b and p[2] < .06 for a, b, p in zip(gt['detached'], inside, gt['fruit_positions']))
                if task == 'FruitCount':
                    success, failed = sum(inside) == gt['n_pick'], sum(inside) > gt['n_pick'] or dropped
                else:
                    targets = set(gt['targets'])
                    wrong = any(b and i not in targets for i, b in enumerate(inside))
                    success = all(inside[i] for i in targets) and not wrong
                    failed = wrong or dropped
                if success or failed:
                    return 'success' if success else 'fail'
                break
            except (ValueError, KeyError, TypeError, IndexError):
                continue
    return None


def retry_allowed(directory, attempt):
    return attempt < 2 and terminal_outcome(directory) is None
