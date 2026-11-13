#!/usr/bin/env python3
"""Official policy server entry point; only the returned policy instance is wrapped by VALogger."""
import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys

from v_a_common import ECM, POLICY

sys.path.insert(0, str(ECM / "src"))
from memory_management.validation.v_a_logger import VALogger  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    opts = parser.parse_args()
    config = json.loads(opts.config.read_text())
    spec = importlib.util.spec_from_file_location("official_server_entry", POLICY / "scripts/serve_policy.py")
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    factory = entry.create_policy

    def instrument(args):
        policy = factory(args)
        VALogger(policy, context_path=opts.context)
        return policy

    entry.create_policy = instrument
    logging.basicConfig(level=logging.INFO, force=True)
    entry.main(entry.Args(seed=config["model_seed"], port=config["port"],
                          policy=entry.Checkpoint(config=config["policy_config"], dir=config["checkpoint_path"])))


if __name__ == "__main__":
    main()
