#!/usr/bin/env python3
"""Standalone full-dimension CPU storage stress; outputs measurements, not success-rate evidence."""
import argparse
from dataclasses import asdict
from pathlib import Path
import resource
import sys
import time
import numpy as np

from v_a_common import ECM, atomic_json, timestamp
sys.path.insert(0, str(ECM / "src"))
from memory_management.config import MemoryConfig
from memory_management.types import FrameRecord
from memory_management.online_controller import OnlineMemoryController
from memory_management.validation.partition import partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = MemoryConfig()
    measurements = []
    started = time.monotonic()
    for mode in ("fifo", "random"):
        c = OnlineMemoryController(cfg, mode)
        c.reset(mode, 280)
        for i in range(6201):
            f = FrameRecord(mode, i, c.decision_id, None,
                            np.full((1, 16, 2048), i / 1024, np.float16),
                            np.full((1, 16, 768), i, np.float32), np.full(8, i, np.float32))
            c.ingest(f)
            if i in (31, 32, 100, 467, 999, 1339, 3999, 4097, 6200):
                packed = c.prepare()
                st = c.stats()
                assert st["bank_frames"] == min(i + 1, 32)
                assert st["rolling_frames"] <= 160 and st["retained_unique_frames"] <= 192
                assert packed["static_mask"].sum() == min(i + 1, 32) * 16
                assert st["retained_unique_bytes"] <= f.nbytes * 192
                if i <= 1339:
                    assert st["recent_frame_ids"] == list(partition(i).recent) and not st["recent_window_approx"]
                measurements.append({"mode": mode, "observed_frames": i + 1, **st,
                                     "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    result = {"schema": "m1-cpu-storage-v1", "generated_utc": timestamp(), "passed": True,
              "config": asdict(cfg), "feature_byte_bound": f.nbytes * 192,
              "elapsed_seconds": time.monotonic() - started, "measurements": measurements,
              "scope": "retained feature arrays; process RSS also includes Python/runtime/output tensors; synthetic features, no policy weights"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print(f"CPU storage stress passed: {args.out}; retained feature bound={result['feature_byte_bound']} bytes")


if __name__ == "__main__":
    main()
