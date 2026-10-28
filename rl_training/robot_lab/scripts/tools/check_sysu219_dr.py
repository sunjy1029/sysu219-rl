"""Static checks for SYSU219 DR: no torch, Isaac Sim, compilation or training."""

import argparse
import ast
import json
import math
from sysu219_paths import isaaclab_source
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-lab", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--isaaclab-source", type=Path,
                        default=isaaclab_source())
    args = parser.parse_args()
    root = args.robot_lab
    task = root / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/sysu219"
    ranges = json.loads((task / "domain_randomization.json").read_text())
    parameters = json.loads((task / "task_parameters.json").read_text())
    require(ranges["schema"] == "sysu219_dr_v1", "Unknown DR schema")
    require(0 < ranges["nominal_fraction"] < 1, "Invalid nominal fraction")
    for key, value in ranges.items():
        if isinstance(value, list):
            require(len(value) == 2 and all(math.isfinite(v) for v in value) and value[0] <= value[1], "Invalid range: " + key)
    require(ranges["friction_static"][1] <= 1, "DR plane/min setup cannot cover this friction")
    require(0 < ranges["friction_dynamic_ratio"][0] <= ranges["friction_dynamic_ratio"][1] <= 1, "Invalid dynamic/static ratio")
    require(parameters["physics_dt_s"] == 0.002, "DR delay history assumes 2 ms physics steps")
    for key in ("action_delay_steps", "observation_delay_steps"):
        require(ranges[key] == [0, 2], "DR history size does not cover configured delays")
    for key in ("mass_scale", "kp_scale", "kd_scale", "coulomb_scale", "viscous_scale", "armature_scale"):
        require(0 < ranges[key][0] <= 1 <= ranges[key][1], "Scale must include nominal: " + key)
    require(parameters["ros_nominal"]["kp"]*ranges["kp_scale"][1] <= 500, "kp exceeds ROS gain clamp")
    require(parameters["ros_nominal"]["kd"]*ranges["kd_scale"][1] <= 5, "kd exceeds ROS gain clamp")
    paths = [*task.rglob("*.py"), root / "source/robot_lab/robot_lab/assets/sysu219_actuator.py",
             root / "scripts/reinforcement_learning/rsl_rl/source_setup.py",
             root / "scripts/reinforcement_learning/rsl_rl/play.py"]
    trees = {}
    for path in paths:
        source = path.read_text()
        require(all(line == line.rstrip() for line in source.splitlines()), "Trailing whitespace: " + str(path))
        trees[path.name] = ast.parse(source, filename=str(path))
    functions = {n.name: n for n in trees["dr_mdp.py"].body if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(trees["trot_env_cfg.py"]):
        if not isinstance(node, ast.Call):
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        func = keywords.get("func")
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "dr_mdp"):
            continue
        require(func.attr in functions, "Unknown DR MDP function")
        definition = functions[func.attr]
        required = {a.arg for a in definition.args.args} - {"env", "env_ids"}
        params = keywords.get("params")
        supplied = {ast.literal_eval(k) for k in params.keys} if isinstance(params, ast.Dict) else set()
        require(required == supplied, "DR MDP parameters mismatch: " + func.attr)
    # Delay steps are used as integer indices. Advanced-index assignment does
    # not implicitly cast Long samples into a Float observation-delay buffer.
    reset_function = functions["reset"]
    storage = next(n for n in reset_function.body if isinstance(n, ast.FunctionDef) and n.name == "storage")
    require(storage.args.args[-1].arg == "dtype", "Storage must accept an explicit dtype")
    delayed_calls = [n for n in ast.walk(reset_function) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "storage"
                     and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "observation_delay"]
    require(len(delayed_calls) == 1, "Missing observation-delay storage")
    dtype = next((k.value for k in delayed_calls[0].keywords if k.arg == "dtype"), None)
    require(isinstance(dtype, ast.Attribute) and isinstance(dtype.value, ast.Name)
            and dtype.value.id == "torch" and dtype.attr == "long", "Delay storage must use torch.long")
    # Check properties/setters used by randomization against this checkout,
    # rather than assuming an API from another IsaacLab version.
    articulation = ast.parse((args.isaaclab_source / "assets/articulation/articulation.py").read_text())
    methods = {n.name for n in ast.walk(articulation) if isinstance(n, ast.FunctionDef)}
    for name in ("write_joint_friction_coefficient_to_sim", "write_joint_armature_to_sim",
                 "write_root_state_to_sim", "write_joint_state_to_sim", "find_joints"):
        require(name in methods or name == "find_joints", "Missing articulation method: " + name)
    data_source = (args.isaaclab_source / "assets/articulation/articulation_data.py").read_text()
    for name in ("default_joint_friction_coeff", "default_joint_viscous_friction_coeff", "default_joint_armature"):
        require(name in data_source, "Missing default property: " + name)
    print("DR JSON ranges, timing/history capacity, nominal inclusion, gain clamps, AST, MDP signatures and IsaacLab API: OK")
    print("No simulator/tensor execution performed; first short training run must validate runtime shapes and physics.")


if __name__ == "__main__":
    main()
