import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

ECM = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def kf():
    sys.path.insert(0, str(ECM / "scripts"))
    spec = importlib.util.spec_from_file_location("run_finetune_kf", ECM / "scripts/run_finetune_kf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATS = {"norm_stats": {k: {"mean": [0.1] * 8, "std": [1.0] * 8, "q01": [-1.0] * 8, "q99": [1.0] * 8}
                        for k in ("state", "actions")}}


def dataset(root):
    (root / "meta").mkdir(parents=True)
    (root / "meta/build.json").write_text(json.dumps({"passed": True}))
    (root / "meta/episodes.json").write_text("[]")
    (root / "assets/robomme").mkdir(parents=True)
    (root / "assets/robomme/norm_stats.json").write_text(json.dumps(STATS))
    return root


def arm(out, name):
    return json.loads((out / f"arms/{name}.json").read_text())


def test_fruit_study_trains_with_the_dataset_norm_stats(kf, tmp_path):
    ds, out = dataset(tmp_path / "ds"), tmp_path / "run"
    kf.prepare(out, ds, smoke=False, study="fruit")
    p = kf.verify(out)
    assert p["name"] == "fruit_finetune_kf" and p["study"] == "fruit" and sorted(p["arms"]) == ["k0", "k4", "k8"]
    assert p["protocol_doc"] == str(ECM / "reports/fruit_finetune_kf_protocol_20261025.md")
    assert p["assets_dir"] == str(ds / "assets") and p["assets_norm_stats_sha256"] == kf.sha256(ds / "assets/robomme/norm_stats.json")
    k0, k4, k8 = (arm(out, a) for a in ("k0", "k4", "k8"))
    assert k8["train"] == dict(kf.TRAIN, steps=12000) and k8["sampler"]["events"] == 8
    assert k0["assets_dir"] == k8["assets_dir"] == str(ds / "assets")
    assert k0["gpus"] == k8["gpus"] == [0, 1] and k4["gpus"] == [2, 3]
    code = {Path(r["path"]).name for r in p["code_identity"]}
    assert {"run_fruit_finetune_pipeline.sh", "build_fruit_dataset.py", "fruit_distance.py"} <= code
    assert "run_finetune_kf_pipeline.sh" not in code
    (ds / "assets/robomme/norm_stats.json").write_text(json.dumps({"norm_stats": {}}))
    with pytest.raises(AssertionError, match="norm stats"):
        kf.verify(out)


def test_robomme_study_keeps_its_arms_and_the_checkpoint_norm_stats(kf, tmp_path):
    ds, out = dataset(tmp_path / "ds"), tmp_path / "run"
    kf.prepare(out, ds, smoke=False)
    p = kf.verify(out)
    assert p["name"] == "finetune_kf" and sorted(p["arms"]) == ["k0", "k12", "k4", "k8"] and "assets_dir" not in p
    spec = arm(out, "k12")
    assert set(spec) == {"arm", "sampler", "dataset", "init_checkpoint", "checkpoint_base_dir", "train", "gpus"}
    assert spec["train"] == kf.TRAIN and spec["gpus"] == [2, 3]
    assert "run_finetune_kf_pipeline.sh" in {Path(r["path"]).name for r in p["code_identity"]}
    kf.prepare(tmp_path / "smoke", ds, smoke=True, study="fruit")
    assert arm(tmp_path / "smoke", "k4")["train"] == dict(kf.TRAIN, steps=100, warmup_steps=5, log_interval=5)


def finish_arm(out, spec, stats):
    step = kf_arm_dir(out, spec)
    (step / "params").mkdir(parents=True)
    (step / "params/manifest.ocdbt").write_text("m")
    (step / "_CHECKPOINT_METADATA").write_text("{}")
    (step / "assets/robomme").mkdir(parents=True)
    (step / "assets/robomme/norm_stats.json").write_text(json.dumps(stats))
    (out / "logs").mkdir(exist_ok=True)
    lines = [f"Step {s}: grad_norm=0.0150, loss=0.0010, param_norm=1875.2" for s in range(0, 100, 5)]
    (out / f"logs/{spec['arm']}_20261026T000000.log").write_text("\n".join(lines) + "\n")


def kf_arm_dir(out, spec):
    return out / "ckpt/mme_vla_suite" / spec["arm"] / str(spec["train"]["steps"] - 1)


def test_gate_requires_the_dataset_norm_stats_in_every_checkpoint(kf, tmp_path):
    ds, out = dataset(tmp_path / "ds"), tmp_path / "run"
    kf.prepare(out, ds, smoke=True, study="fruit")
    for name in ("k0", "k4", "k8"):
        finish_arm(out, arm(out, name), STATS)
    assert kf.gate(out) == 0
    assert all(a["norm_stats_match"] and a["logged"] == 20 for a in json.loads((out / "gate_t1.json").read_text())["arms"].values())
    other = json.loads(json.dumps(STATS))
    other["norm_stats"]["actions"]["q99"][0] = 1.5
    (kf_arm_dir(out, arm(out, "k4")) / "assets/robomme/norm_stats.json").write_text(json.dumps(other))
    assert kf.gate(out) == 1
    shutil.rmtree(kf_arm_dir(out, arm(out, "k4")) / "assets")
    assert kf.gate(out) == 1
