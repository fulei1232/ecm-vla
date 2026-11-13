#!/usr/bin/env python3
"""Write newline-delimited JSON from stdin as an atomic Parquet file.

Kept outside the policy process so the system Python's existing pandas/pyarrow
can be used without changing the JAX policy environment.
"""
import argparse
import io
import os
from pathlib import Path
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_json(io.StringIO(sys.stdin.read()), lines=True, precise_float=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, args.out)


if __name__ == "__main__":
    main()
