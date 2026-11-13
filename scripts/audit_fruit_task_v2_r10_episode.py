#!/usr/bin/env python3
"""Revision10 continuation uses the unchanged revision9 task/rollout implementation."""
from fruit_r10_common import activate,OUT


if __name__=='__main__':
    activate()
    import audit_fruit_task_v2_r9_episode as original
    original.OUT=OUT
    original.main()
