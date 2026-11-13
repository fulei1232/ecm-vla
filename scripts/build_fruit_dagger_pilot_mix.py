#!/usr/bin/env python3
"""Build an isolated, provenance-recorded short-run mixture for the K0 DAgger pilot.

Sample a fixed number of execution steps from every clean+DART training episode,
then add all expert correction samples. Direct correction steps get one extra
copy. Original feature directories are linked; correction pkl episode ids are
remapped to follow the 400 base episodes. No source dataset is edited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp


def add_link(src: Path, dest: Path):
    if not src.exists():
        raise FileNotFoundError(src)
    dest.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path, default=ECM / "data/finetune_kf/fruit_dart_rm75b_train")
    p.add_argument("--corrections", type=Path, default=ECM / "data/finetune_kf/fruit_dagger_corrections_20261030")
    p.add_argument("--late", type=Path, default=ECM / "data/finetune_kf/fruit_dagger_carry_late_20261030")
    p.add_argument("--extra-corrections", type=Path, help="additional preprocessed correction dataset")
    p.add_argument("--extra-manifests", type=Path, help="HDF5 train root for the additional corrections")
    p.add_argument("--additional-source", nargs=3, action="append", default=[],
                   metavar=("LABEL", "DATASET", "MANIFEST_ROOT"),
                   help="repeatable preprocessed correction source")
    p.add_argument("--base-per-episode", type=int, default=25)
    p.add_argument("--direct-extra", type=int, default=1)
    a = p.parse_args()
    if int(np.__version__.split(".")[0]) != 1:
        p.error("run this builder with the policy training interpreter (NumPy 1 pickle format)")
    out, base, corrections, late = (x.resolve() for x in (a.out, a.base, a.corrections, a.late))
    if bool(a.extra_corrections) != bool(a.extra_manifests):
        p.error("--extra-corrections and --extra-manifests must be provided together")
    sources = [("corrections", corrections,
                ECM / "data/fruit_dagger_corrections_20261030/rm75b_eg2/train"),
               ("late", late,
                ECM / "data/fruit_dagger_corrections_late_20261030/rm75b_eg2/train")]
    if a.extra_corrections:
        sources.append(("expanded", a.extra_corrections.resolve(), a.extra_manifests.resolve()))
    for label, dataset, manifests in a.additional_source:
        sources.append((label, Path(dataset).resolve(), Path(manifests).resolve()))
    if len({label for label, _, _ in sources}) != len(sources):
        p.error("correction source labels must be unique")
    if a.base_per_episode < 1 or a.direct_extra < 0:
        p.error("base-per-episode must be positive and direct-extra nonnegative")
    if (out / "meta/stats.json").exists():
        p.error(f"mixture already exists: {out}")
    for sub in ("data", "features", "meta", "assets/robomme"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for src in (base, *(item[1] for item in sources)):
        if not read_json(src / "meta/build.json")["passed"]:
            raise ValueError(f"source build did not pass: {src}")

    base_rows = read_json(base / "meta/episodes.json")
    if len(base_rows) != 400 or [r["global_episode"] for r in base_rows] != list(range(400)):
        raise ValueError("expected 400 indexed clean+DART base episodes")
    next_sample = 0
    selected = []
    for row in base_rows:
        g = row["global_episode"]
        add_link(base / f"features/episode_{g}", out / f"features/episode_{g}")
        ids = np.linspace(row["first_sample"], row["first_sample"] + row["exec_samples"] - 1,
                          num=min(a.base_per_episode, row["exec_samples"]), dtype=int)
        for sample_id in sorted(set(ids.tolist())):
            add_link(base / f"data/{sample_id}.pkl", out / f"data/{next_sample}.pkl")
            next_sample += 1
        selected.append({"episode": g, "source_task": row["task"], "source_train_episode": row["train_episode"],
                         "selected": len(set(ids.tolist()))})
    base_count = next_sample

    direct_copies = []
    correction_counts = {}
    episode_offset = 400
    for label, src, manifest_root in sources:
        rows = read_json(src / "meta/episodes.json")
        for row in rows:
            source_g = row["global_episode"]
            new_g = episode_offset + source_g
            add_link(src / f"features/episode_{source_g}", out / f"features/episode_{new_g}")
            source_manifest = (manifest_root /
                               ("count" if row["task"] == "FruitCount" else "occluded_ripe") /
                               f"ep{row['train_episode']}_manifest.json")
            man = read_json(source_manifest)
            direct_steps = man["correction_steps"]
            if direct_steps > row["exec_samples"]:
                raise ValueError(f"invalid direct correction count: {source_manifest}")
            for local in range(row["exec_samples"]):
                source_sample = row["first_sample"] + local
                with (src / f"data/{source_sample}.pkl").open("rb") as f:
                    sample = pickle.load(f)
                sample["epis_idx"] = np.asarray([new_g], dtype=sample["epis_idx"].dtype)
                dest = out / f"data/{next_sample}.pkl"
                with dest.open("wb") as f:
                    pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
                if local < direct_steps:
                    direct_copies.append(dest)
                next_sample += 1
            correction_counts[label] = correction_counts.get(label, 0) + row["exec_samples"]
        episode_offset += len(rows)
    correction_count = next_sample - base_count
    for _ in range(a.direct_extra):
        for original in direct_copies:
            add_link(original, out / f"data/{next_sample}.pkl")
            next_sample += 1
    repeated_direct = len(direct_copies) * a.direct_extra

    shutil.copy2(base / "assets/robomme/norm_stats.json", out / "assets/robomme/norm_stats.json")
    atomic_json(out / "meta/stats.json", {"execution_samples": next_sample,
                                          "total_samples": sum(r["timesteps"] for r in base_rows) +
                                                           sum(read_json(src / "meta/build.json")["total_samples"]
                                                           for _, src, _ in sources)})
    report = {"passed": True, "created_utc": timestamp(), "base": str(base),
              "corrections": str(corrections), "late": str(late),
              "correction_sources": {label: str(src) for label, src, _ in sources},
              "source_build_sha256": {label: sha256(src / "meta/build.json")
                                      for label, src, _ in [("base", base, None), *sources]},
              "base_per_episode": a.base_per_episode, "direct_extra": a.direct_extra,
              "base_episodes": len(base_rows), "base_samples": base_count,
              "correction_episodes": episode_offset - len(base_rows),
              "correction_samples": correction_count, "direct_correction_unique": len(direct_copies),
              "direct_correction_repeats": repeated_direct,
              "correction_counts": correction_counts, "execution_samples": next_sample,
              "norm_stats_sha256": sha256(out / "assets/robomme/norm_stats.json"),
              "script_sha256": sha256(Path(__file__).resolve())}
    atomic_json(out / "meta/base_selection.json", selected)
    atomic_json(out / "meta/build.json", report)
    if not (out / f"data/{next_sample - 1}.pkl").exists():
        raise RuntimeError("mixture sample ids are incomplete")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
