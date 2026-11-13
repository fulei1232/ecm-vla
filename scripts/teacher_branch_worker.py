#!/usr/bin/env python3
"""Branch worker of one live Teacher Oracle episode (oracle_branch.serve); started by eval_teacher_episode.py."""
import argparse
import json
import os
from pathlib import Path
import sys

from v_a_common import ECM, POLICY, SIM, SIM_PYTHON


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", required=True, help="JSON: directory, task, episode_id, model_seed, environment_seed")
    parser.add_argument("--params", required=True, help="JSON expert parameters")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--reply-fd", type=int, required=True)
    opts = parser.parse_args()
    sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"), str(SIM / "src"),
                    str(ECM / "src")]
    import robomme
    import mani_skill
    assert Path(robomme.__file__).resolve() == (SIM / "src/robomme/__init__.py").resolve(), robomme.__file__
    maniskill = SIM_PYTHON.parent.parent / "lib/python3.11/site-packages/mani_skill"
    assert Path(mani_skill.__file__).resolve().parent == maniskill.resolve(), mani_skill.__file__
    from memory_management.validation import oracle_branch
    with os.fdopen(opts.reply_fd, "w") as replies:
        oracle_branch.serve(json.loads(opts.info), json.loads(opts.params), opts.scratch, sys.stdin, replies)
    sys.stdout.flush()
    sys.stderr.flush()
    # As in steps 1-2: skip renderer teardown; every reply is already written.
    os._exit(0)


if __name__ == "__main__":
    main()
