#!/usr/bin/env python3
"""Explicit R9 server entry; absent flag delegates to the original server unchanged."""
import argparse
from functools import partial
import sys
import json

import serve_teacher_policy as original


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--enable-fruit-v2-r9', action='store_true')
    parser.add_argument('--temporal-index', choices=['absolute','order'], default='absolute')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if not args.enable_fruit_v2_r9:
        if args.temporal_index != 'absolute':raise ValueError('Order mode requires explicit R9 flag')
        return original.main()
    from memory_management.validation import teacher_oracle as teacher
    from memory_management.policy_adapter import install_temporal_index
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', required=True)
    config_args, _ = config_parser.parse_known_args(remaining)
    with open(config_args.config) as stream:config = json.load(stream)
    signal = config['oracle']['event']['signal']
    if signal != {'kind':'external','factory':'memory_management.fruit_v2_distance:RecordedDistance'}:
        raise ValueError('R9 inference requires the any_hanging signal factory')
    previous = original.OracleLogger

    class R9Logger(previous):
        def __init__(self, policy, *positional, **kwargs):
            install_temporal_index(policy, args.temporal_index)
            super().__init__(policy, *positional, **kwargs)

    old_conditions, old_keyframe = teacher.CONDITIONS, teacher.KEYFRAME
    teacher.CONDITIONS = (*teacher.CONDITIONS, 'KA')
    teacher.KEYFRAME = teacher.KEYFRAME | {'KA'}
    original.OracleLogger = R9Logger
    try:
        return original.main()
    finally:
        original.OracleLogger = previous
        teacher.CONDITIONS, teacher.KEYFRAME = old_conditions, old_keyframe


if __name__ == '__main__':main()
