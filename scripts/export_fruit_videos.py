#!/usr/bin/env python3
"""Export fruit-picking demonstrations (RoboMME-format h5) to mp4: front | wrist camera with a caption bar.

    python scripts/export_fruit_videos.py data/fruit_picking/rm75b_eg2 --out data/fruit_picking/rm75b_eg2/videos
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BAR = 48


def text(value):
    value = value[()] if hasattr(value, "shape") else value
    return value.decode() if isinstance(value, bytes) else str(value)


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frames(h5, manifest):
    with h5py.File(h5, "r") as f:
        episode = f[next(iter(f.keys()))]
        steps = sorted((k for k in episode.keys() if k.startswith("timestep_")), key=lambda k: int(k.split("_")[1]))
        small = font(13)
        for i, key in enumerate(steps):
            step = episode[key]
            image = np.concatenate([step["obs/front_rgb"][()], step["obs/wrist_rgb"][()]], axis=1)
            canvas = Image.new("RGB", (image.shape[1], image.shape[0] + BAR), (20, 20, 20))
            canvas.paste(Image.fromarray(image), (0, BAR))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 4), manifest["instruction"], fill=(255, 255, 255), font=small)
            subgoal = text(step["info/simple_subgoal"])
            gripper = "closed" if bool(step["obs/is_gripper_close"][()]) else "open"
            draw.text((6, 24), f"t={i}/{len(steps) - 1}  gripper {gripper}  {subgoal}", fill=(255, 210, 120), font=small)
            yield np.asarray(canvas)


def export(job):
    h5, manifest, out, fps = job
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8) as writer:
        n = 0
        for frame in frames(h5, manifest):
            writer.append_data(frame)
            n += 1
    return str(out), n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="robot directory with <split>/<task>/ep*_manifest.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--limit", type=int, help="first N episodes per split and task")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    jobs = []
    for task_dir in sorted(d for d in a.root.glob("*/*") if any(d.glob("ep*_manifest.json"))):
        manifests = sorted(task_dir.glob("ep*_manifest.json"), key=lambda m: int(m.name[2:].split("_")[0]))
        for m in manifests[:a.limit]:
            manifest = json.loads(m.read_text())
            name = Path(manifest["h5"]).stem + ".mp4"
            jobs.append((task_dir / manifest["h5"], manifest, a.out / task_dir.relative_to(a.root) / name, a.fps))
    with ProcessPoolExecutor(a.workers) as pool:
        for path, n in pool.map(export, jobs):
            print(f"{path}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
