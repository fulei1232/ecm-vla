#!/usr/bin/env python3
"""Same ep112 commands and geometry, with pre-attachment decoration offsets."""
from fruit_shield_buildfix_common import activate


if __name__=='__main__':
    activate()
    from fruit_picking.count_shield_v2_buildfix import install
    install()
    import diagnose_fruit_count_shield_v2_ep112 as original
    original.main()
