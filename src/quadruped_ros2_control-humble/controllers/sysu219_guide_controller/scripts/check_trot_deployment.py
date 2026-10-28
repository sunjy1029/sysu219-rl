"""Read-only model/contract/source checks. Does not compile or run a ROS controller."""

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import onnx
import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    package = Path(__file__).resolve().parents[1]
    workspace = package.parents[3]
    config_root = workspace / "src/quadruped_ros2_control-humble/descriptions/sysu219/sysu219_description/config"
    configured_model = yaml.safe_load((config_root / "gazebo.yaml").read_text())["sysu219_guide_controller"]["ros__parameters"]["rl_model_path"]
    model_path = Path(configured_model)
    if not model_path.is_absolute():
        model_path = package / "model" / model_path
    report = json.loads(model_path.with_suffix(".json").read_text())
    contract = report["contract"]
    require(hashlib.sha256(model_path.read_bytes()).hexdigest() == report["onnx_sha256"], "Model hash mismatch")
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == report["contract_sha256"], "Contract hash mismatch")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    require(metadata["sysu219.contract_sha256"] == digest, "ONNX metadata mismatch")
    require(metadata["sysu219.checkpoint_sha256"] == report["checkpoint_sha256"], "Checkpoint metadata mismatch")
    for item, name, shape in ((model.graph.input[0], "obs", [1, 45]), (model.graph.output[0], "actions", [1, 12])):
        require(item.name == name and item.type.tensor_type.elem_type == onnx.TensorProto.FLOAT, "Tensor name/type mismatch")
        require([d.dim_value for d in item.type.tensor_type.shape.dim] == shape, "Tensor shape mismatch")
    header = (package / "include/sysu219_guide_controller/control/TrotPolicyConfig.h").read_text()
    require(f'kContractSha256 = "{digest}"' in header, "C++ contract hash mismatch")
    require(f'kModelFile = {json.dumps(model_path.name)};' in header, "C++ model filename mismatch")
    arrays = {name: [float(v.strip()) for v in values.split(",")] for name, values in
              re.findall(r"std::array<double, 12> (\w+) = \{([^}]+)\};", header)}
    for name, key in (("kDefaultPosition", "default_joint_positions"), ("kDefaultVelocity", "default_joint_velocities"),
                      ("kActionScale", "action_scales"), ("kTargetMin", "target_min"), ("kTargetMax", "target_max"),
                      ("kKp", "kp"), ("kKd", "kd"), ("kEffortLimit", "effort_limits")):
        require(arrays[name] == contract[key] and len(arrays[name]) == 12, f"C++ array mismatch: {name}")
    names = re.search(r"kJointNames = \{([^}]+)\}", header).group(1)
    require(json.loads("[" + names + "]") == contract["joint_names"], "C++ joint order mismatch")
    scalars = {name: float(value) for name, value in re.findall(r"double (\w+) = ([^;]+);", header)}
    for name, value in (("kPolicyDt", contract["policy_dt_s"]), ("kActionClip", contract["action_clip"]),
                        ("kPeriod", contract["clock"]["period_s"]), ("kSettle", contract["clock"]["settle_s"]),
                        ("kRamp", contract["clock"]["ramp_s"])):
        require(scalars[name] == value, f"C++ scalar mismatch: {name}")
    for prefix, term in (("AngularVelocity", "base_ang_vel"), ("Gravity", "projected_gravity"),
                         ("JointPosition", "joint_pos"), ("JointVelocity", "joint_vel"), ("LastAction", "actions")):
        expected = contract["observation_preprocessing"][term]
        require([scalars[f"k{prefix}ClipMin"], scalars[f"k{prefix}ClipMax"]] == expected["clip"], "Clip mismatch")
        require(scalars[f"k{prefix}Scale"] == expected["scale"], "Scale mismatch")
    config_root = workspace / "src/quadruped_ros2_control-humble/descriptions/sysu219/sysu219_description/config"
    description = ET.parse(config_root.parent / "xacro/robot.xacro").getroot()
    imu_joint = description.find("joint[@name='imu_joint']")
    require(imu_joint.find("parent").get("link") == "trunk" and
            imu_joint.find("origin").get("rpy") == "0 0 0", "IMU-to-trunk rotation requires observation transform")
    pose_mismatches = []
    for filename in ("gazebo.yaml", "robot_control.yaml"):
        config = yaml.safe_load((config_root / filename).read_text())
        parameters = config["sysu219_guide_controller"]["ros__parameters"]
        require(parameters["rl_model_path"] == configured_model, "Real/Gazebo model paths differ")
        require(parameters["joints"] == contract["joint_names"], "Configured controller joint names differ")
        stand = parameters["stand_pos"]
        require(len(stand) == 12 and all(math.isfinite(q) for q in stand), "Invalid ROS stand_pos")
        if any(abs(q - saved) > 1e-9 for q, saved in zip(stand, contract["default_joint_positions"])):
            pose_mismatches.append(filename + ": model default pose differs from ROS stand_pos; retraining/export required")
        require(0 < parameters["rl_pose_transition_s"] <= parameters["rl_entry_timeout_s"], "Invalid transition")
        require(parameters["rl_pose_transition_s"] <= contract["clock"]["settle_s"], "Takeover exceeds saved standing phase")
        if filename == "gazebo.yaml":
            low_level = config["leg_pd_controller"]["ros__parameters"]
            require(low_level["joints"] == contract["joint_names"], "Gazebo PD order mismatch")
            require(low_level["effort_limits"] == contract["effort_limits"], "Gazebo torque limits differ")
            for i in range(12):
                require(low_level["position_min"][i] <= contract["target_min"][i] <= contract["target_max"][i] <= low_level["position_max"][i], "Policy targets exceed hardware-aligned Gazebo limits")
    state = (package / "src/FSM/StateRLWalk.cpp").read_text()
    require("getVelocity()" not in state and "isaac_to_ctrl_" not in state, "Legacy observation/order remains")
    for index in ("[i]", "[3+i]", "[6+i]", "[18+i]", "[30+i]", "[42]", "[43]", "[44]"):
        require("observation_" + index in state, "Missing actor input segment: " + index)
    require("policy_.forward(observation_)" in state and "last_action_ = action" in state, "Actor action history missing")
    require("std::clamp(action[i], -cfg::kActionClip, cfg::kActionClip)" in state, "Action clipping missing")
    require(state.index("target[i] = std::clamp") < state.index("target[i] = q_start_[i] + blend*(target[i] - q_start_[i])") < state.index("last_action_ = action"), "Takeover action/history order mismatch")
    require("target[i] = std::clamp(default_position_[i] + cfg::kActionScale[i]*action[i], cfg::kTargetMin[i], cfg::kTargetMax[i])" in state, "Target transform mismatch")
    require("period.seconds()" in state and "runPolicy(0.0)" in state, "Policy timing/reset missing")
    for path in (package / "scripts").glob("*trot*.py"):
        ast.parse(path.read_text(), filename=str(path))
    ET.parse(package / "package.xml")
    # Preserve the original checked-in 48D policy byte for byte.
    relative = str((package / "model/policy.onnx").relative_to(workspace))
    previous = subprocess.run(["git", "show", "HEAD:" + relative], cwd=workspace, capture_output=True, check=True).stdout
    require(previous == (package / "model/policy.onnx").read_bytes(), "Original policy was modified")
    print(json.dumps({"model": str(model_path), "contract_sha256": digest, "static_checks": "passed",
                      "deployment_ready": not pose_mismatches, "blocking_issues": pose_mismatches,
                      "legacy_policy_preserved": True, "cpp_compiled_or_executed": False}, indent=2))
    if pose_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
