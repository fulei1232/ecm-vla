#!/usr/bin/env python3
"""Official policy server with an explicit baseline/fifo/random instance-level switch."""
import argparse
import importlib.util
import logging
from pathlib import Path
import sys

from v_a_common import ECM, POLICY, append_jsonl, atomic_json, read_json

sys.path.insert(0, str(ECM / "src"))
from memory_management.config import MemoryConfig
from memory_management.policy_adapter import install_memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    args = p.parse_args()
    cfg = read_json(args.config)
    spec = importlib.util.spec_from_file_location("upstream_m1_server", POLICY / "scripts/serve_policy.py")
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    factory = entry.create_policy

    def context():
        own = read_json(args.context) if args.context.exists() else {}
        return {**own, "episode_key": f"{own.get('task')}/{own.get('episode_id')}/{cfg['mode']}"}

    def emit(row):
        own = context()
        directory = Path(own["directory"])
        append_jsonl(directory / "policy_decisions.jsonl", row)
        atomic_json(directory / "memory_latest.json", row)

    def instrument(opts):
        policy = factory(opts)
        install_memory(policy, cfg["mode"], MemoryConfig(**cfg["memory"]), context, emit)
        return policy

    entry.create_policy = instrument
    logging.basicConfig(level=logging.INFO, force=True)
    entry.main(entry.Args(seed=cfg["model_seed"], port=cfg["port"], policy=entry.Checkpoint(
        config=cfg["policy_config"], dir=cfg["checkpoint_path"])))


if __name__ == "__main__":
    main()
