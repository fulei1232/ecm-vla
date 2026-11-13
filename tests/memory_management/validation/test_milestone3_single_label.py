import json
from pathlib import Path

from scripts.run_milestone3_single_label import make_label


RUN = Path(__file__).resolve().parents[3] / "results/V_B_fp32_full_20261022"


def test_milestone3_identity_label_is_zero_and_audited():
    result = make_label(RUN, "BinFill_ep20_F0064.json", slot=0)
    assert result["schema"] == "milestone3-single-teacher-label-v1"
    assert result["formal"] is False
    assert result["candidate"]["candidate_is_historical"] is True
    assert result["replacement"]["same_frame"] is False
    assert result["replacement"]["actual_label"] == "k1_c0_j0"
    assert result["label"]["same_frame_gain_max_abs"] == 0.0
    assert result["label"]["discounted_gain_mean"] == 0.0
    assert "actual_discounted_gain_mean" in result["label"]
    assert all(result["audits"].values())
    assert result["shape_and_mask_audit"]["all_offsets_valid"] is True


def test_milestone3_label_is_json_serializable():
    result = make_label(RUN, "BinFill_ep20_F0064.json", slot=0)
    encoded = json.dumps(result, allow_nan=False)
    assert "cached" not in encoded.lower() or result["schema"].startswith("milestone3")
