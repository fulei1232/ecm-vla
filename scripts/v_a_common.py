"""Paths, interpreters and small I/O helpers shared by the V-A runner scripts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ECM = Path(__file__).resolve().parents[1]
WORKSPACE = ECM.parent
POLICY = WORKSPACE / "acm_vla/vendor/robomme_policy_learning"
SIM = POLICY / "third_party/robomme_benchmark"
POLICY_PYTHON = WORKSPACE / "adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python"
SIM_PYTHON = WORKSPACE / "robomme_benchmark/.venv/bin/python"
CHECKPOINT = WORKSPACE / "adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999"
OPENPI_DATA_HOME = WORKSPACE / "acm_vla/cache/openpi"
VALID = {"success", "fail", "timeout"}


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def append_jsonl(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def git_head(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
