#!/usr/bin/env python3
"""Restore the shared Python 3.11.15 runtimes without reinstalling dependencies.

Run with system python3 from the GPU-enabled terminal. Downloads only the missing
interpreter; backs up each changed venv configuration and interpreter link.
--check-only does not repair; --cpu-only explicitly skips GPU/render validation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ECM = Path(__file__).resolve().parents[1]
WS = ECM.parent
RUNTIME = WS / ".runtime"
VERSION = "3.11.15"
BASE = RUNTIME / "python" / f"cpython-{VERSION}-linux-x86_64-gnu/bin/python3.11"
POLICY = WS / "acm_vla/vendor/robomme_policy_learning"
VENVS = {
    "policy": WS / "adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv",
    "sim": WS / "robomme_benchmark/.venv",
}


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def repair_venv(venv, interpreter, backup):
    """Change only home and bin/python; roll both back if the smoke check fails."""
    cfg = venv / "pyvenv.cfg"
    link = venv / "bin/python"
    if not link.is_symlink():
        raise RuntimeError(f"Expected an existing symlink, refusing to replace: {link}")
    original = cfg.read_text()
    if f"version_info = {VERSION}" not in original:
        raise RuntimeError(f"Unexpected Python version in {cfg}")
    if not (venv / "lib/python3.11/site-packages").is_dir():
        raise RuntimeError(f"Missing existing dependencies: {venv}")
    old_target = os.readlink(link)
    updated = re.sub(r"(?m)^home = .*?$", f"home = {interpreter.parent}", original)
    if updated == original and link.resolve() == interpreter.resolve():
        print(f"Already repaired: {venv}", flush=True)
        return
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cfg, backup / "pyvenv.cfg")
    (backup / "python-link.json").write_text(json.dumps({"path": str(link), "target": old_target}, indent=2))
    cfg_tmp = cfg.with_name(f"pyvenv.cfg.repair-{os.getpid()}")
    link_tmp = link.with_name(f"python.repair-{os.getpid()}")
    try:
        cfg_tmp.write_text(updated)
        os.replace(cfg_tmp, cfg)
        link_tmp.symlink_to(os.path.relpath(interpreter, link.parent))
        os.replace(link_tmp, link)
        run([str(link), "-I", "-c", "import sys, ssl, sqlite3, ctypes; "
             f"assert sys.version_info[:3] == (3, 11, 15); "
             f"assert sys.prefix == {str(venv)!r}; print(sys.executable, sys.version)"], timeout=30)
    except BaseException:
        shutil.copy2(backup / "pyvenv.cfg", cfg)
        link_tmp.unlink(missing_ok=True)
        link_tmp.symlink_to(old_target)
        os.replace(link_tmp, link)
        raise
    finally:
        cfg_tmp.unlink(missing_ok=True)
        link_tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--expected-gpus", type=int, default=8)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = RUNTIME / "repairs" / stamp
    out.mkdir(parents=True)
    report = {"created_utc": stamp, "cpu_only": args.cpu_only, "checks": [], "passed": False}
    report_path = out / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    try:
        if not args.check_only:
            if not BASE.exists():
                uv = shutil.which("uv")
                if not uv:
                    raise RuntimeError("uv is required to restore Python 3.11.15")
                print("Installing Python 3.11.15 into the shared workspace...", flush=True)
                run([uv, "python", "install", VERSION, "--install-dir", str(RUNTIME / "python"),
                     "--no-bin", "--cache-dir", str(RUNTIME / "uv-cache"), "--no-config"], timeout=600)
            run([str(BASE), "-I", "-c", "import sys; assert sys.version_info[:3] == (3, 11, 15)"], timeout=30)
            for name, venv in VENVS.items():
                repair_venv(venv, BASE, out / "backup" / name)

        common = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                  "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "WANDB_MODE": "disabled",
                  "OPENPI_DATA_HOME": str(WS / "acm_vla/cache/openpi"), "OMP_NUM_THREADS": "2"}
        common.pop("PYTHONHOME", None)
        # Resolve imports exactly as the existing training launcher does.
        policy_env = {**common, "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src:{ECM}/src"}
        sim_env = {**common, "PYTHONPATH": f"{ECM}:{ECM}/src:{WS}/robomme_benchmark/src"}

        def check(name, command, env, cwd=ECM, timeout=180):
            log = out / f"{name}.log"
            print(f"Checking {name}...", flush=True)
            try:
                with log.open("w") as stream:
                    result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=timeout)
                code = result.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                with log.open("a") as stream:
                    stream.write(f"\n{error}\n")
                code = -1
            report["checks"].append({"name": name, "passed": code == 0, "returncode": code, "log": str(log)})
            save()
            print(f"{'PASS' if code == 0 else 'FAIL'} {name}: {log}", flush=True)
            if code:
                print(log.read_text(errors="replace")[-4000:], flush=True)
            return code == 0

        def pycheck(name, which, code, env, cwd=ECM):
            return check(name, [str(VENVS[which] / "bin/python"), "-u", "-c", code], env, cwd)

        policy_ok = pycheck("policy_imports", "policy", """
import sys, importlib, importlib.metadata as md, runpy
assert sys.version_info[:3] == (3, 11, 15)
for name in ['numpy', 'jax', 'jaxlib', 'flax', 'orbax.checkpoint', 'torch', 'transformers', 'h5py', 'openpi', 'openpi_client', 'mme_vla_suite.training.config']:
    importlib.import_module(name)
    print('import OK:', name, flush=True)
for name in ['numpy', 'jax', 'jaxlib', 'flax', 'orbax-checkpoint', 'torch', 'transformers', 'h5py']:
    print(name, md.version(name), flush=True)
runpy.run_path('scripts/train.py', run_name='runtime_import_check')
print('official training entry imports OK')
""", {**policy_env, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}, POLICY)
        sim_ok = pycheck("sim_imports", "sim", """
import sys, importlib, importlib.metadata as md
assert sys.version_info[:3] == (3, 11, 15)
for name in ['numpy', 'torch', 'torchvision', 'sapien', 'mani_skill', 'h5py', 'gymnasium', 'openpi_client', 'fruit_picking.generate', 'fruit_picking.env', 'fruit_picking.solver']:
    importlib.import_module(name)
    print('import OK:', name, flush=True)
for name in ['numpy', 'torch', 'torchvision', 'sapien', 'h5py']:
    print(name, md.version(name), flush=True)
""", {**sim_env, "CUDA_VISIBLE_DEVICES": ""})
        if not args.cpu_only:
            gpu_env = {**common}
            gpu_env.pop("CUDA_VISIBLE_DEVICES", None)
            smi_ok = check("nvidia_smi", ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"], gpu_env)
            check("gpu_topology", ["nvidia-smi", "topo", "-m"], gpu_env)
            if policy_ok and smi_ok:
                p_gpu = {**policy_env, "JAX_PLATFORMS": "cuda"}
                p_gpu.pop("CUDA_VISIBLE_DEVICES", None)
                pycheck("policy_jax_cuda", "policy", f"""
import jax, jax.numpy as jnp, numpy as np
devices = jax.devices('gpu')
print(devices, flush=True)
assert len(devices) == {args.expected_gpus}, devices
for d in devices:
    with jax.default_device(d):
        x = jnp.ones((64, 64), dtype=jnp.float32)
        y = jax.jit(lambda a: a @ a)(x).block_until_ready()
        np.testing.assert_allclose(np.asarray(y), 64)
    print('CUDA compute OK:', d, flush=True)
""", p_gpu, POLICY)
            if sim_ok and smi_ok:
                s_gpu = {**sim_env}
                s_gpu.pop("CUDA_VISIBLE_DEVICES", None)
                pycheck("sim_torch_cuda", "sim", f"""
import torch
print('torch', torch.__version__, 'CUDA runtime', torch.version.cuda, flush=True)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == {args.expected_gpus}
for i in range(torch.cuda.device_count()):
    x = torch.ones((64, 64), device=f'cuda:{{i}}')
    torch.testing.assert_close((x @ x).cpu(), torch.full((64, 64), 64.0))
    print('CUDA compute OK:', i, torch.cuda.get_device_name(i), flush=True)
""", s_gpu)
                icds = [Path('/etc/vulkan/icd.d/nvidia_icd.json'), Path('/usr/share/vulkan/icd.d/nvidia_icd.json')]
                icd = next((p for p in icds if p.is_file()), None)
                if icd:
                    s_gpu['VK_ICD_FILENAMES'] = str(icd)
                s_gpu['CUDA_VISIBLE_DEVICES'] = '0'
                pycheck("fruit_render", "sim", """
import gymnasium as gym
import fruit_picking.env
env = gym.make('FruitPick-v1', task='count', robot_uids='rm75b_eg2', obs_mode='rgb+depth+segmentation',
               control_mode='pd_joint_pos', render_mode='rgb_array', sim_backend='cpu')
try:
    obs, _ = env.reset(seed=100000)
    for camera in ['front_camera', 'hand_camera']:
        rgb = obs['sensor_data'][camera]['rgb']
        assert tuple(rgb.shape[-3:]) == (256, 256, 3), rgb.shape
        assert rgb.float().std().item() > 0
        print('render OK:', camera, tuple(rgb.shape), flush=True)
finally:
    env.close()
""", s_gpu)
        report["passed"] = all(c["passed"] for c in report["checks"])
    except Exception as error:
        report["error"] = str(error)
        print(f"Repair stopped: {error}", file=sys.stderr, flush=True)
    finally:
        save()
        print(f"Report: {report_path}", flush=True)
    if report["passed"]:
        print("PASS (CPU checks only; GPU not validated)" if args.cpu_only else "PASS (imports, CUDA compute and fruit rendering)")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
