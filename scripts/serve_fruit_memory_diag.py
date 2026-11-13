#!/usr/bin/env python3
"""D6 server adapter; absent --enable-memory-diagnostics is exactly the original entry."""
import argparse
from functools import partial
import sys

import serve_teacher_policy as original


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--enable-memory-diagnostics', action='store_true', default=False)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    before = original.OracleLogger
    if args.enable_memory_diagnostics:
        from memory_management.fruit_memory_diag import DiagnosticOracleLogger
        original.OracleLogger = partial(DiagnosticOracleLogger, diagnostics_enabled=True)
    try:
        return original.main()
    finally:
        original.OracleLogger = before


if __name__ == '__main__':
    main()
