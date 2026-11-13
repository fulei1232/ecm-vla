#!/usr/bin/env python3
"""Build the RealMan arm + EG2 two-finger gripper URDF (and its SRDF) used by the fruit-picking simulation.

Sources (Apache-2.0, RealManRobot/rm_models): RM75/urdf/RM75-B (7 joints, default) or RM65/urdf/RM65-B (6 joints),
and thirdparty/Two-finger Electric Gripper.urdf (EG2-4C2). The gripper is mounted on the arm's last link (flange
face at its origin, +z outwards). The EG2 base keeps its real mesh. Its four-bar
fingers are closed-chain (mimic joints), which neither SAPIEN nor mplib enforces, so the two fingertips
are modelled as parallel prismatic jaws with the EG2 fingertip span: half-opening 0..0.04 m along the
gripper x axis, pads from z = 0.100 to 0.144 m above the flange.

mplib plans against `<mesh>.convex.stl` hulls, written next to every mesh.

The `tcp` link sits between the pads (z = 0.125 m) and follows the ManiSkill two-finger convention:
z = approach direction, y = closing direction.

    /media/fulei/fuleivla/robomme_benchmark/.venv/bin/python ecm_vla/fruit_picking/assets/build_rm_eg2.py \
        --arm RM75-B --rm-models /media/fulei/fuleivla/third_party/rm_models
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ARMS = {"RM65-B": ("RM65/urdf/RM65-B", "link_6"), "RM75-B": ("RM75/urdf/RM75-B", "link_7")}
GRIPPER = "thirdparty/Two-finger Electric Gripper.urdf"

FINGER_Z0 = 0.100
FINGER_HALF = (0.004, 0.011, 0.022)
FINGER_TRAVEL = 0.04
TCP_Z = 0.125


def element(tag, parent=None, **attrib):
    node = ET.Element(tag, {k: str(v) for k, v in attrib.items()})
    if parent is not None:
        parent.append(node)
    return node


def vec(values):
    return " ".join(f"{v:g}" for v in values)


def box_link(name, center, half, mass, rgba):
    link = element("link", name=name)
    inertial = element("inertial", link)
    element("origin", inertial, xyz=vec(center), rpy="0 0 0")
    element("mass", inertial, value=mass)
    sx, sy, sz = (2 * h for h in half)
    element("inertia", inertial, ixx=mass * (sy**2 + sz**2) / 12, ixy=0, ixz=0,
            iyy=mass * (sx**2 + sz**2) / 12, iyz=0, izz=mass * (sx**2 + sy**2) / 12)
    for tag in ("visual", "collision"):
        shape = element(tag, link)
        element("origin", shape, xyz=vec(center), rpy="0 0 0")
        element("box", element("geometry", shape), size=vec(2 * h for h in half))
        if tag == "visual":
            element("color", element("material", shape, name=f"{name}_material"), rgba=vec(rgba))
    return link


def joint(name, kind, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0), axis=None, limit=None):
    node = element("joint", name=name, type=kind)
    element("parent", node, link=parent)
    element("child", node, link=child)
    element("origin", node, xyz=vec(xyz), rpy=vec(rpy))
    if axis is not None:
        element("axis", node, xyz=vec(axis))
    if limit is not None:
        element("limit", node, lower=limit[0], upper=limit[1], effort=limit[2], velocity=limit[3])
    return node


def relink_meshes(root, prefix):
    for mesh in root.iter("mesh"):
        mesh.set("filename", f"meshes/{prefix}{Path(mesh.get('filename')).name}")


def robot_name(arm: str) -> str:
    return arm.lower().replace("-", "") + "_eg2"


def build(rm_models: Path, arm_name: str):
    ARM, flange = ARMS[arm_name]
    name = robot_name(arm_name)
    out = HERE / name
    if out.exists():
        shutil.rmtree(out)
    (out / "meshes").mkdir(parents=True)
    for src in (rm_models / ARM / "meshes").glob("*.STL"):
        shutil.copy(src, out / "meshes" / src.name)
    shutil.copy(rm_models / GRIPPER / "meshes/4C2_baselink.STL", out / "meshes/eg2_4C2_baselink.STL")
    shutil.copy(rm_models / "LICENSE", out / "LICENSE.rm_models")

    arm = ET.parse(rm_models / ARM / f"urdf/{arm_name}.urdf").getroot()
    relink_meshes(arm, "")
    robot = element("robot", name=name)
    for child in arm:
        robot.append(copy.deepcopy(child))

    gripper = ET.parse(rm_models / GRIPPER / "urdf/EG2-4C2.urdf").getroot()
    base = copy.deepcopy(next(l for l in gripper.findall("link") if l.get("name") == "4C2_baselink"))
    base.set("name", "eg2_base")
    relink_meshes(base, "eg2_")
    robot.append(base)
    robot.append(joint("eg2_mount", "fixed", flange, "eg2_base"))

    hx = FINGER_HALF[0]
    for side, sign in (("left", 1), ("right", -1)):
        robot.append(box_link(f"{side}_finger", (sign * hx, 0, FINGER_HALF[2]), FINGER_HALF, 0.03, (0.15, 0.15, 0.15, 1)))
        robot.append(joint(f"{side}_finger_joint", "prismatic", "eg2_base", f"{side}_finger", xyz=(0, 0, FINGER_Z0),
                           axis=(sign, 0, 0), limit=(0, FINGER_TRAVEL, 40, 0.1)))
    tcp = element("link", name="tcp")
    robot.append(tcp)
    # tcp y axis = closing direction (gripper x); z = approach
    robot.append(joint("tcp_joint", "fixed", "eg2_base", "tcp", xyz=(0, 0, TCP_Z), rpy=(0, 0, 1.5707963267948966)))

    import trimesh
    for mesh in (out / "meshes").glob("*.STL"):
        trimesh.load(mesh).convex_hull.export(mesh.with_name(mesh.name + ".convex.stl"))

    ET.indent(robot)
    urdf = out / f"{name}.urdf"
    urdf.write_text('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n")
    (out / "NOTICE").write_text(
        "Meshes and arm kinematics from RealManRobot/rm_models (Apache License 2.0, Copyright 2024 realman-robotics):\n"
        f"  {ARM} and {GRIPPER} (EG2-4C2 base only). See LICENSE.rm_models.\n"
        "Modified: combined into one URDF; EG2 four-bar fingers replaced by parallel prismatic jaws; tcp link added.\n")
    return urdf


def write_srdf(path: Path, pairs):
    root = element("robot", name=path.stem)
    for (a, b), reason in sorted(pairs.items()):
        element("disable_collisions", root, link1=a, link2=b, reason=reason)
    ET.indent(root)
    path.write_text('<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n")


def build_srdf(urdf: Path, samples=20000, always=0.95, seed=0):
    """Disable self-collision checks for adjacent links, the two jaws, and pairs in contact in nearly every
    random configuration (mplib 0.1.1's own generator is broken)."""
    import mplib
    import numpy as np
    srdf = urdf.with_suffix(".srdf")
    tree = ET.parse(urdf).getroot()
    pairs = {tuple(sorted((j.find("parent").get("link"), j.find("child").get("link")))): "Adjacent"
             for j in tree.findall("joint")}
    pairs[("left_finger", "right_finger")] = "Adjacent"
    write_srdf(srdf, pairs)
    planner = mplib.Planner(urdf=str(urdf), move_group="tcp", srdf=str(srdf))
    np.random.seed(seed)
    counts = {}
    for _ in range(samples):
        planner.robot.set_qpos(planner.pinocchio_model.get_random_configuration(), True)
        for c in planner.planning_world.collide_full():
            key = tuple(sorted((c.link_name1, c.link_name2)))
            counts[key] = counts.get(key, 0) + 1
    for key, n in counts.items():
        if n >= always * samples:
            pairs.setdefault(key, "Default")
    write_srdf(srdf, pairs)
    return srdf, {f"{a}|{b}": n / samples for (a, b), n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(ARMS), default="RM75-B")
    parser.add_argument("--rm-models", type=Path, default=Path("/media/fulei/fuleivla/third_party/rm_models"))
    parser.add_argument("--no-srdf", action="store_true")
    args = parser.parse_args()
    urdf = build(args.rm_models, args.arm)
    print("wrote", urdf)
    if not args.no_srdf:
        srdf, frequent = build_srdf(urdf)
        print("wrote", srdf, "| most frequent self-collisions:", frequent)


if __name__ == "__main__":
    main()
