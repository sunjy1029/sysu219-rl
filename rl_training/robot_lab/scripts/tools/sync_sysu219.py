"""Export SYSU219 from ROS source xacro; --check performs read-only static validation.

Run with ROS's Python (xacro, PyYAML and NumPy required). No Isaac Sim imports.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xacro
import xacro.substitution_args as substitutions
import yaml

from sysu219_meshes import build_isaac_files


LEGS = ("FR", "FL", "RR", "RL")
JOINTS = [f"{leg}_{kind}_joint" for leg in LEGS for kind in ("hip", "thigh", "calf")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def origin_transform(element):
    transform = np.eye(4)
    if element is None:
        return transform
    roll, pitch, yaw = map(float, element.get("rpy", "0 0 0").split())
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    transform[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                           [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                           [-sp, cp*sr, cp*cr]]
    transform[:3, 3] = list(map(float, element.get("xyz", "0 0 0").split()))
    return transform


def export(source):
    original_find = substitutions._eval_find
    # Resolve this package to source, never to a possibly stale ROS install tree.
    substitutions._eval_find = lambda name: str(source) if name == "sysu219_description" else original_find(name)
    try:
        document = xacro.process_file(str(source / "xacro/robot.xacro"), mappings={
            "DEBUG": "false", "GAZEBO": "false", "CLASSIC": "false"})
    finally:
        substitutions._eval_find = original_find
    original = ET.fromstring(document.toxml())
    root = copy.deepcopy(original)
    for child in list(root):
        if child.tag in ("ros2_control", "gazebo", "transmission"):
            root.remove(child)
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    active = {joint.get("name"): joint for joint in joints if joint.get("type") != "fixed"}
    require(list(active) == JOINTS, "Expected FR/FL/RR/RL x hip/thigh/calf, exactly 12 revolute joints")
    require(all(joint.get("type") == "revolute" for joint in active.values()), "Unexpected joint type")
    require(len(links) == len(root.findall("link")), "Duplicate links")
    require(len({j.get("name") for j in joints}) == len(joints), "Duplicate joints")
    children = [j.find("child").get("link") for j in joints]
    require(len(set(children)) == len(children), "A link has multiple parents")
    require(set(links) - set(children) == {"base"}, "Expected floating root link base")
    for joint in joints:
        require(joint.find("parent").get("link") in links and joint.find("child").get("link") in links,
                "Joint references an unknown link")
    mass = 0.0
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None:
            continue  # massless fixed frames must not acquire invented inertias
        require(ET.tostring(inertial) == ET.tostring(original.find(f"link[@name='{name}']/inertial")),
                f"Changed source inertia: {name}")
        link_mass = float(inertial.find("mass").get("value"))
        values = inertial.find("inertia").attrib
        xx, yy, zz, xy, xz, yz = [float(values[k]) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        eigenvalues = np.linalg.eigvalsh([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        require(link_mass > 0 and np.all(eigenvalues > 0), f"Invalid inertia: {name}")
        require(eigenvalues[-1] <= sum(eigenvalues[:2]) + 1e-10, f"Inertia triangle violation: {name}")
        mass += link_mass
    mesh_files = {}
    for mesh in root.findall(".//mesh"):
        path = Path(mesh.get("filename").removeprefix("file://"))
        require(path.parent == source / "meshes" and path.is_file(), f"Invalid mesh: {path}")
        mesh_files[f"meshes/{path.name}"] = path.read_bytes()
        mesh.set("filename", f"../meshes/{path.name}")
    for leg in LEGS:
        foot_joint = root.find(f"joint[@name='{leg}_foot_fixed']")
        require(foot_joint is not None, f"Missing {leg} foot joint")
        foot_joint.set("dont_collapse", "true")
        require(links[f"{leg}_foot"].find("collision/geometry/sphere") is not None, "Missing foot collision")
    control = yaml.safe_load((source / "config/robot_control.yaml").read_text())["sysu219_guide_controller"]["ros__parameters"]
    require(control["joints"] == JOINTS and len(control["stand_pos"]) == 12, "ROS controller order mismatch")
    parameters = {}
    for name, default in zip(JOINTS, control["stand_pos"]):
        joint = active[name]
        axis = np.array(list(map(float, joint.find("axis").get("xyz").split())))
        require(np.isclose(np.linalg.norm(axis), 1), f"Invalid axis: {name}")
        limit = {k: float(v) for k, v in joint.find("limit").attrib.items()}
        require(limit["lower"] <= default <= limit["upper"], f"Default pose outside limits: {name}")
        require(limit["effort"] > 0 and limit["velocity"] > 0, f"Invalid limits: {name}")
        dynamics = joint.find("dynamics")
        parameters[name] = dict(default_position=float(default), **limit,
            viscous_friction=float(dynamics.get("damping")),
            coulomb_friction=float(dynamics.get("friction")),
            armature=float(joint.find("identified_dynamics").get("Ja")))
    # Forward kinematics checks connectivity, reachability and the initial foot height.
    transforms = {"base": np.eye(4)}
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in transforms]
        require(ready, "Disconnected or cyclic kinematic tree")
        for joint in ready:
            transform = origin_transform(joint.find("origin"))
            if joint.get("name") in parameters:
                q = parameters[joint.get("name")]["default_position"]
                axis = np.array(list(map(float, joint.find("axis").get("xyz").split())))
                x, y, z = axis
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                rotation = np.eye(4)
                rotation[:3, :3] = np.eye(3) + np.sin(q)*cross + (1-np.cos(q))*(cross @ cross)
                transform = transform @ rotation
            transforms[joint.find("child").get("link")] = transforms[joint.find("parent").get("link")] @ transform
            pending.remove(joint)
    require(set(transforms) == set(links), "Unreachable links")
    feet = {leg: transforms[f"{leg}_foot"][:3, 3].tolist() for leg in LEGS}
    radii = {leg: float(links[f"{leg}_foot"].find("collision/geometry/sphere").get("radius")) for leg in LEGS}
    # Identity base orientation; lowest foot sphere has 1 cm initial ground clearance.
    height = max(radii[leg] - feet[leg][2] for leg in LEGS) + 0.01
    inputs = [*sorted((source / "xacro").glob("*.xacro")), source / "config/robot_control.yaml"]
    manifest = dict(source_package="sysu219_description", source_entry="xacro/robot.xacro",
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        mesh_sha256={name: hashlib.sha256(data).hexdigest() for name, data in sorted(mesh_files.items())},
        total_mass_kg=mass, trunk_mass_kg=float(links["trunk"].find("inertial/mass").get("value")),
        joint_names=JOINTS, joints=parameters, stand_kp=float(control["stand_kp"]), stand_kd=float(control["stand_kd"]),
        initial_height_m=height, feet_in_base_m=feet,
        initial_foot_clearance_m={leg: height + feet[leg][2] - radii[leg] for leg in LEGS})
    ET.indent(root, space="  ")
    urdf = '<?xml version="1.0"?>\n<!-- Generated by scripts/tools/sync_sysu219.py from ROS SOURCE xacro. -->\n'
    urdf += ET.tostring(root, encoding="unicode") + "\n"
    return {"urdf/robot.urdf": urdf.encode(), "model_parameters.json": (json.dumps(manifest, indent=2) + "\n").encode(), **mesh_files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-description", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "source/robot_lab/data/Robots/SYSU219")
    parser.add_argument("--check", action="store_true", help="Compare against source without writing files")
    args = parser.parse_args()
    files = export(args.source_description.resolve())
    files.update(build_isaac_files(files))
    for name, data in files.items():
        path = args.output / name
        if args.check:
            require(path.is_file() and path.read_bytes() == data, f"Outdated or missing asset: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    params = json.loads(files["model_parameters.json"])
    print(f"SYSU219: {len(params['joint_names'])} joints, mass={params['total_mass_kg']:.9f} kg")
    print(f"Initial height={params['initial_height_m']:.6f} m; foot clearance={params['initial_foot_clearance_m']}")
    print(f"{'Checked' if args.check else 'Exported'} {len(files)} files; no simulator launched.")


if __name__ == "__main__":
    main()
