import json
from pathlib import Path
import sys

import pytest

ECM = Path(__file__).resolve().parents[3]
ACM_SCRIPTS = Path("/media/fulei/fuleivla/acm_vla/scripts")
M3_RUN = Path("/media/fulei/fuleivla/acm_vla/results/20261012T122037_m3_diagnostic80")
sys.path.insert(0, str(ECM))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def m3_episodes():
    """Recorded official D0 rollouts (test split; used only to validate selection code)."""
    if not M3_RUN.exists():
        pytest.skip("ACM M3 run not available")
    out = []
    for task in ("BinFill", "VideoUnmaskSwap", "PatternLock", "PickHighlight"):
        for ep in (0, 1, 2, 3):
            d = M3_RUN / "episodes" / task / f"ep{ep:02d}"
            rows = read_lines(d / "raw_frames.jsonl")
            decisions = read_lines(d / "policy_decisions.jsonl")
            out.append({"task": task, "episode_id": ep, "rows": rows, "decisions": decisions,
                        "exec_start_idx": decisions[0]["exec_start_idx"]})
    return out
