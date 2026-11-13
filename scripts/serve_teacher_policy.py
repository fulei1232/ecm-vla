#!/usr/bin/env python3
"""Official policy server entry point for the Teacher Oracle pilot.

The returned policy instance is wrapped by OracleLogger, which also holds an FP32 teacher copy of the
checkpoint. Scoring blocks the event loop for tens of seconds, so the server sends no keepalive pings
(the official client tolerates 600 s); everything else is the official server.

With config["checkpoints"] ({condition: checkpoint}) the server loads one policy per distinct checkpoint,
each wrapped by its own OracleLogger, and every episode runs on the policy of the condition named in the
episode context at reset.
"""
import argparse
import dataclasses
import importlib.util
import json
import logging
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, POLICY

sys.path.insert(0, str(ECM / "src"))
from memory_management.validation.teacher_oracle import (CONDITIONS, SCORED, OracleLogger, OracleScorer,  # noqa: E402
                                                         loss_dims)


class ConditionRouter:
    """The served object: reset() picks the policy of the context's condition; the rest is delegated."""

    def __init__(self, policies, by_condition, context_path):
        self.policies, self.by_condition, self.context_path = policies, by_condition, context_path
        self.active = None

    @property
    def metadata(self):
        return next(iter(self.policies.values())).metadata

    def reset(self):
        context = json.loads(self.context_path.read_text()) if self.context_path.exists() else {}
        condition = context.get("condition")
        if condition not in self.by_condition:
            raise ValueError(f"no checkpoint for condition {condition!r}")
        self.active = self.policies[self.by_condition[condition]]
        return self.active.reset()

    def add_buffer(self, obs):
        return self.active.add_buffer(obs)

    def infer(self, obs):
        return self.active.infer(obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    opts = parser.parse_args()
    config = json.loads(opts.config.read_text())
    oracle = config["oracle"]
    spec = importlib.util.spec_from_file_location("official_server_entry", POLICY / "scripts/serve_policy.py")
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    wps = entry.websocket_policy_server

    class Server(wps.WebsocketPolicyServer):
        async def run(self):
            async with wps._server.serve(self._handler, self._host, self._port, compression=None, max_size=None,
                                         process_request=wps._health_check, ping_interval=None) as server:
                await server.serve_forever()

    wps.WebsocketPolicyServer = Server
    factory = entry.create_policy

    def instrument(args):
        if config.get("checkpoints"):
            if SCORED & set(config["conditions"]):
                raise ValueError("scored conditions need the single-checkpoint server")
            policies = {}
            for path in dict.fromkeys(config["checkpoints"].values()):
                policies[path] = factory(dataclasses.replace(args, policy=entry.Checkpoint(
                    config=config["policy_config"], dir=path)))
                OracleLogger(policies[path], context_path=opts.context, scorer=None, params=oracle)
                logging.info("loaded %s", path)
            return ConditionRouter(policies, config["checkpoints"], opts.context)
        policy = factory(args)
        scorer = None
        if SCORED & set(config.get("conditions", CONDITIONS)):
            dims = sorted({loss_dims(t) for t in config["tasks"]})
            scorer = OracleScorer(config["policy_config"], config["checkpoint_path"], config["model_seed"],
                                  oracle["batch_size"], oracle["mc_samples"], dims, oracle["rel_err_max"])
            blank = np.zeros((256, 256, 3), np.uint8)
            scorer.warmup({"observation/image": blank, "observation/wrist_image": blank,
                           "observation/state": np.zeros(8, np.float32), "prompt": "warmup"}, dims,
                          n=oracle.get("capacity", 32))
            logging.info("Teacher Oracle scorer ready (dims %s)", dims)
        OracleLogger(policy, context_path=opts.context, scorer=scorer, params=oracle)
        return policy

    entry.create_policy = instrument
    logging.basicConfig(level=logging.INFO, force=True)
    entry.main(entry.Args(seed=config["model_seed"], port=config["port"],
                          policy=entry.Checkpoint(config=config["policy_config"], dir=config["checkpoint_path"])))


if __name__ == "__main__":
    main()
