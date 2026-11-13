#!/usr/bin/env python3
"""D6 client adapter. Only add an explicit optional context flag; platform unchanged."""
import argparse
from pathlib import Path
import sys

import evaluate_fruit_lowlevel_v2_episode as original


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--memory-ablation', choices=('recent16', 'no_pregust'), default=None)
    args, remaining = parser.parse_known_args()
    context = Path(remaining[remaining.index('--context')+1])
    saved = original.atomic_json
    def write(path, value):
        if args.memory_ablation is not None and Path(path) == context:
            value = {**value, 'memory_ablation': args.memory_ablation}
        return saved(path, value)
    original.atomic_json = write
    sys.argv = [sys.argv[0], *remaining]
    try:
        return original.main()
    finally:
        original.atomic_json = saved


if __name__ == '__main__':
    raise SystemExit(main())
