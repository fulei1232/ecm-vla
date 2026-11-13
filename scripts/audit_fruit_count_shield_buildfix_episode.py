#!/usr/bin/env python3
"""Run the unchanged episode CLI with the approved build-record implementation."""
from fruit_shield_buildfix_common import activate


if __name__=='__main__':
    activate()
    from fruit_picking.count_shield_v2_buildfix import install
    install()
    import runpy
    runpy.run_module('audit_fruit_count_shield_v2_episode',run_name='__main__')
