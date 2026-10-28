"""Read-only AST/URDF checks. Does not import Isaac Lab, torch, or task modules.

These checks cannot validate USD import, tensor execution, or policy convergence.
"""

import argparse
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from sysu219_paths import isaaclab_source


TASK_RELATIVE = Path("source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/sysu219")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assigned(node):
    result = {}
    for item in node.body:
        if isinstance(item, ast.Assign):
            result.update({t.id: item.value for t in item.targets if isinstance(t, ast.Name)})
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            result[item.target.id] = item.value
    return result


def transform(origin):
    matrix = np.eye(4)
    if origin is None:
        return matrix
    r, p, y = map(float, origin.get("rpy", "0 0 0").split())
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    matrix[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                       [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    matrix[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-lab", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--isaaclab-source", type=Path, default=isaaclab_source(), help="Path containing isaaclab/__init__.py")
    args = parser.parse_args()
    task = args.robot_lab / TASK_RELATIVE
    asset = args.asset_dir or args.robot_lab / "source/robot_lab/data/Robots/SYSU219"
    parameters = json.loads((task / "task_parameters.json").read_text())
    model = json.loads((asset / "model_parameters.json").read_text())
    trees = {}
    paths = [*sorted(task.rglob("*.py")), args.robot_lab / "scripts/tools/check_sysu219_trot.py"]
    for path in paths:
        text = path.read_text()
        require(all(line == line.rstrip() for line in text.splitlines()), f"Trailing whitespace: {path}")
        trees[path.name] = ast.parse(text, filename=str(path))
    env_tree = trees["trot_env_cfg.py"]
    classes = {node.name: node for node in ast.walk(env_tree) if isinstance(node, ast.ClassDef)}
    functions = {node.name: node for node in trees["mdp.py"].body if isinstance(node, ast.FunctionDef)}
    lab_functions = {}
    lab_classes = {}
    # Read definitions only: importing torch/Isaac Lab could initialize JIT/runtime.
    for path in args.isaaclab_source.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                lab_classes.setdefault(node.name, node)
        if "envs/mdp/" in path.as_posix():
            lab_functions.update({node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)})
    # Resolve every local/core MDP callable and required manager parameter.
    for call in ast.walk(env_tree):
        if not isinstance(call, ast.Call):
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        func = keywords.get("func")
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            continue
        namespace = func.value.id
        if namespace not in ("mdp", "lab_mdp"):
            continue
        definitions = functions if namespace == "mdp" else lab_functions
        require(func.attr in definitions, f"Unknown callable: {namespace}.{func.attr}")
        definition = definitions[func.attr]
        names = [arg.arg for arg in definition.args.args]
        required = set(names[:len(names)-len(definition.args.defaults)]) - {"env", "env_ids"}
        params = keywords.get("params")
        supplied = {ast.literal_eval(key) for key in params.keys} if isinstance(params, ast.Dict) else set()
        require(required <= supplied, f"Missing params for {func.attr}: {required - supplied}")
        require(supplied <= set(names), f"Unknown params for {func.attr}: {supplied - set(names)}")

    def fields(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen or name not in lab_classes:
            return set()
        seen.add(name)
        cls = lab_classes[name]
        result = set(assigned(cls))
        for base in cls.bases:
            base_name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
            result |= fields(base_name, seen)
        return result

    for tree in (env_tree, trees["rsl_rl_ppo_cfg.py"]):
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            name = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id if isinstance(call.func, ast.Name) else ""
            name = {"ObsTerm": "ObservationTermCfg", "RewTerm": "RewardTermCfg", "DoneTerm": "TerminationTermCfg",
                    "EventTerm": "EventTermCfg"}.get(name, name)
            if name in lab_classes:
                require({k.arg for k in call.keywords} <= fields(name), f"Unknown config fields for {name}")
    policy = assigned(classes["PolicyCfg"])
    expected = parameters["policy_observation_terms"]
    require(list(policy) == [name for name, _ in expected], "Policy observation order differs from contract")
    dimensions = {"base_ang_vel": 3, "projected_gravity": 3, "joint_pos": 12, "joint_vel": 12, "actions": 12, "gait_clock": 3}
    require(all(dimensions[name] == dim for name, dim in expected), "Policy term dimensions inconsistent")
    require(sum(dim for _, dim in expected) == parameters["policy_observation_dimension"] == 45, "Policy size mismatch")
    require(list(assigned(classes["CriticCfg"])) == [name for name, _ in parameters["critic_extra_terms"]], "Critic order mismatch")
    require(45 + sum(dim for _, dim in parameters["critic_extra_terms"]) == parameters["critic_observation_dimension"] == 52,
            "Critic size mismatch")
    require(not assigned(classes["CommandsCfg"]), "In-place task must not generate velocity commands")
    require(not ({"stand_still", "feet_air_time", "feet_contact_without_cmd", "joint_pos_penalty"} & set(assigned(classes["RewardsCfg"]))),
            "Velocity-gated rewards must not be inherited")
    require("episode_length_buf" not in {n.attr for n in ast.walk(trees["mdp.py"]) if isinstance(n, ast.Attribute)},
            "Gait clock must not depend on randomized episode length")
    registrations = ast.parse((task / "__init__.py").read_text())
    loop = next(n for n in registrations.body if isinstance(n, ast.For))
    entries = ast.literal_eval(loop.iter)
    require(entries[0][0] == parameters["task_id"] and all(name in classes for _, name in entries), "Invalid task entry points")
    agent = ast.parse((task / "agents/rsl_rl_ppo_cfg.py").read_text())
    runner = next(n for n in agent.body if isinstance(n, ast.ClassDef))
    require(ast.literal_eval(assigned(runner)["obs_groups"]) == {"actor": ["policy"], "critic": ["critic"]}, "Actor/critic mapping mismatch")
    print("AST, local MDP signatures, config API fields, task registration, 45/52 observation contract: OK")

    p = parameters
    nominal = p["ros_nominal"]
    require(p["clearance_m"] == 0.07, "Expected 7 cm peak foot clearance")
    require(abs(p["physics_dt_s"] - nominal["actuator_dt_s"]) < 1e-12, "Actuator timing mismatch")
    require(abs(p["physics_dt_s"]*p["decimation"] - 0.02) < 1e-12, "Policy must remain 50 Hz")
    require(p["period_s"] > 0 and p["ramp_s"] > 0 and p["settle_s"] >= 0, "Invalid clock timing")
    require(0.5 <= p["stance_ratio"] < 1, "Invalid stance ratio")
    require(0 <= p["transition_margin"] < (1-p["stance_ratio"])/2, "No scored swing interval")
    require(p["foot_phase_offsets"] == [0, 0.5, 0.5, 0], "Expected FR/FL/RR/RL diagonal phase pairs")
    names = model["joint_names"]
    require(len(names) == len(p["default_joint_positions"]) == len(p["action_scales"]) == 12, "Joint count mismatch")
    pose = dict(zip(names, p["default_joint_positions"]))
    root = ET.parse(asset / "urdf/robot.urdf").getroot()
    for name, scale in zip(names, p["action_scales"]):
        limits = nominal["joint_limits"][name]
        source = model["joints"][name]
        low = max(limits["lower"], pose[name]-p["action_clip"]*scale)
        high = min(limits["upper"], pose[name]+p["action_clip"]*scale)
        require(scale > 0 and source["lower"] <= low < high <= source["upper"],
                f"Clipped action envelope outside source limits: {name}")
    w, x, y, z = p["initial_root_rotation_wxyz"]
    require(abs(w*w+x*x+y*y+z*z-1) < 1e-10, "Initial quaternion not normalized")
    reference = np.eye(4)
    reference[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                           [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                           [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    require(np.allclose(reference[:3, :3].T @ [0,0,-1], p["nominal_projected_gravity_b"]), "Gravity reference mismatch")
    transforms = {"base": reference}
    pending = root.findall("joint")
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in transforms]
        require(ready, "Invalid kinematic tree")
        for joint in ready:
            matrix = transform(joint.find("origin"))
            if joint.get("name") in pose:
                q = pose[joint.get("name")]
                x, y, z = map(float, joint.find("axis").get("xyz").split())
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                rotation = np.eye(4)
                rotation[:3, :3] = np.eye(3) + math.sin(q)*cross + (1-math.cos(q))*(cross @ cross)
                matrix = matrix @ rotation
            transforms[joint.find("child").get("link")] = transforms[joint.find("parent").get("link")] @ matrix
            pending.remove(joint)
    for index, leg in enumerate(("FR", "FL", "RR", "RL")):
        require(np.allclose(transforms[f"{leg}_foot"][:2, 3], p["nominal_feet_xy_m"][index], atol=1e-8, rtol=0), "Nominal foot XY mismatch")
        sphere = root.find(f"link[@name='{leg}_foot']/collision/geometry/sphere")
        require(float(sphere.get("radius")) == p["foot_radius_m"], "Foot radius mismatch")
        bottom = transforms[f"{leg}_foot"][2, 3] - p["foot_radius_m"]
        require(abs(bottom + p["target_base_height_m"]) < 1e-8, "Nominal feet are not coplanar on ground")
    # Planar IK verifies the complete vertical 7 cm swing, including joint/action bounds.
    for leg in ("FR", "FL", "RR", "RL"):
        first = root.find(f"joint[@name='{leg}_calf_joint']/origin")
        second = root.find(f"joint[@name='{leg}_foot_fixed']/origin")
        first_xyz = np.array(list(map(float, first.get("xyz").split())))
        second_xyz = np.array(list(map(float, second.get("xyz").split())))
        require(np.allclose(first_xyz[:2], 0) and np.allclose(second_xyz[:2], 0), "Planar IK assumes axial links")
        l1, l2 = -first_xyz[2], -second_xyz[2]
        for height in np.linspace(0, p["clearance_m"], 101):
            target = transforms[f"{leg}_foot"][:3,3] + np.array([0.,0.,height])
            delta = reference[:3,:3].T @ (target-transforms[f"{leg}_thigh"][:3,3])
            x, z = delta[0], delta[2]
            cosine = (x*x+z*z-l1*l1-l2*l2)/(2*l1*l2)
            require(-1 <= cosine <= 1, "Swing outside workspace")
            calf = -math.acos(cosine)
            thigh = math.atan2(-x,-z)-math.atan2(l2*math.sin(calf),l1+l2*math.cos(calf))
            for kind, angle in (("thigh", thigh), ("calf", calf)):
                name = f"{leg}_{kind}_joint"
                i = names.index(name)
                limits = nominal["joint_limits"][name]
                low = max(limits["lower"], pose[name]-p["action_clip"]*p["action_scales"][i])
                high = min(limits["upper"], pose[name]+p["action_clip"]*p["action_scales"][i])
                require(low-1e-8 <= angle <= high+1e-8, "7 cm reference exceeds action/ROS bounds: " + name)
    print("7 cm vertical swing: reachable inside ROS command/action bounds (static IK only)")
    for phase in np.linspace(0.0, 1.0, 100, endpoint=False):
        phases = (phase + np.array(p["foot_phase_offsets"])) % 1.0
        stance = phases < p["stance_ratio"]
        require(stance[0] == stance[3] and stance[1] == stance[2] and sum(stance) in (2, 4), "Invalid diagonal schedule")
    print("URDF forward kinematics, coplanar neutral feet, target height and bounded actions: OK")
    print(f"Diagonal schedule: {p['period_s']} s period, {p['stance_ratio']:.0%} stance, "
          f"{2*p['stance_ratio']-1:.0%} cycle four-foot support; no aerial phase")
    print("Static checks only: no task imports, compilation, simulator, tensor execution, or training.")


if __name__ == "__main__":
    main()
