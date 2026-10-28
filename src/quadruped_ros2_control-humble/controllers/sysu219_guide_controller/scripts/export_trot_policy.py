"""Export the saved RSL-RL actor and its deployment contract, without Isaac Sim.

Run with conda lab's Python. Reads saved params, never the current training config.
No source build, simulator startup, environment installation or critic export.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
import yaml


class SavedConfigLoader(yaml.SafeLoader):
    """Only decode the two passive sequence tags emitted by saved Isaac configs."""


for tag in ("tag:yaml.org,2002:python/tuple", "tag:yaml.org,2002:python/object/apply:builtins.slice"):
    SavedConfigLoader.add_constructor(tag, lambda loader, node: loader.construct_sequence(node))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_contract(env, agent):
    action = env["actions"]["joint_pos"]
    names = action["joint_names"]
    expected = [f"{leg}_{kind}_joint" for leg in ("FR", "FL", "RR", "RL") for kind in ("hip", "thigh", "calf")]
    require(names == expected and action["preserve_order"] and action["use_default_offset"], "Unexpected action contract")
    policy = env["observations"]["policy"]
    terms = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "gait_clock"]
    require([key for key, value in policy.items() if isinstance(value, dict) and "func" in value] == terms, "Unexpected observation order")
    randomization = env.get("domain_randomization") or {}
    is_dr = randomization.get("enabled", False) and randomization.get("ranges", {}).get("schema") == "sysu219_dr_v1"
    require(policy["concatenate_terms"] and not policy["history_length"], "Unexpected policy preprocessing")
    require(not policy["enable_corruption"] or is_dr, "Unsupported observation corruption")
    if is_dr:
        # These wrappers return raw gyro/gravity/q/dq with training-only errors.
        # ROS still performs the same saved clip/scale and default subtraction.
        for term, function in (("base_ang_vel", "base_ang_vel"), ("projected_gravity", "projected_gravity"),
                               ("joint_pos", "joint_pos"), ("joint_vel", "joint_vel")):
            require(policy[term]["func"].endswith(".sysu219.dr_mdp:" + function), "Unsupported DR observation function")
    require(agent["obs_groups"]["actor"] == ["policy"], "Actor uses privileged observations")
    require(not agent["actor"]["obs_normalization"], "Normalized actor requires a new deployment contract")
    for term in terms:
        require(not policy[term]["history_length"] and policy[term]["modifiers"] is None, "Unsupported history/modifier")
        require(policy[term].get("noise") is None, "Unsupported manager noise; DR wrappers handle training-only noise")
    for term in ("joint_pos", "joint_vel"):
        selection = policy[term]["params"]["asset_cfg"]
        require(selection["joint_names"] == names and selection["preserve_order"], "Joint observation order mismatch")
    initial = env["scene"]["robot"]["init_state"]
    joints = env["scene"]["robot"]["actuators"]["legs"]

    def ordered(value):
        return [float(value[name]) for name in names] if isinstance(value, dict) else [float(value)] * 12

    preprocessing = {term: {"clip": policy[term]["clip"], "scale": 1.0 if policy[term]["scale"] is None else policy[term]["scale"]} for term in terms}
    require(preprocessing["gait_clock"] == {"clip": None, "scale": 1.0}, "Unsupported clock preprocessing")
    clock = env["trot"]
    require(clock["period_s"] > 0 and clock["ramp_s"] > 0 and clock["settle_s"] >= 0, "Invalid saved clock")
    return {
        "schema": "sysu219_trot_v1", "observation_dimension": 45, "action_dimension": 12,
        "joint_names": names, "observation_terms": terms, "observation_preprocessing": preprocessing,
        "default_joint_positions": ordered(initial["joint_pos"]),
        "default_joint_velocities": ordered(initial["joint_vel"]),
        "action_scales": ordered(action["scale"]), "action_clip": float(agent["clip_actions"]),
        "target_min": [float(action["clip"][name][0]) for name in names],
        "target_max": [float(action["clip"][name][1]) for name in names],
        "kp": ordered(joints["stiffness"]), "kd": ordered(joints["damping"]),
        "effort_limits": ordered(joints["effort_limit"]),
        "policy_dt_s": float(env["sim"]["dt"] * env["decimation"]),
        "clock": {key: float(clock[key]) for key in ("period_s", "settle_s", "ramp_s")},
        "input_name": "obs", "output_name": "actions",
    }


def write_header(contract, digest, path, model_file="policy_trot_in_place.onnx"):
    lines = ["// Generated from the checkpoint's SAVED env.yaml and agent.yaml. Do not edit by hand.",
             "#pragma once", "#include <array>", "#include <cstddef>", "", "namespace sysu219_trot {",
             "inline constexpr std::size_t kObservationDim = 45;", "inline constexpr std::size_t kActionDim = 12;",
             f'inline constexpr const char* kContractSha256 = "{digest}";',
             f'inline constexpr const char* kModelFile = {json.dumps(model_file)};']

    def scalar(name, value):
        lines.append(f"inline constexpr double {name} = {float(value):.17g};")

    def array(name, values):
        lines.append(f"inline constexpr std::array<double, 12> {name} = {{{', '.join(format(v, '.17g') for v in values)}}};")

    lines.append("inline constexpr std::array<const char*, 12> kJointNames = {" + ", ".join(json.dumps(n) for n in contract["joint_names"]) + "};")
    for name, key in (("kDefaultPosition", "default_joint_positions"), ("kDefaultVelocity", "default_joint_velocities"),
                      ("kActionScale", "action_scales"), ("kTargetMin", "target_min"), ("kTargetMax", "target_max"),
                      ("kKp", "kp"), ("kKd", "kd"), ("kEffortLimit", "effort_limits")):
        array(name, contract[key])
    for name, value in (("kActionClip", contract["action_clip"]), ("kPolicyDt", contract["policy_dt_s"]),
                        ("kPeriod", contract["clock"]["period_s"]), ("kSettle", contract["clock"]["settle_s"]),
                        ("kRamp", contract["clock"]["ramp_s"])):
        scalar(name, value)
    for name, term in (("AngularVelocity", "base_ang_vel"), ("Gravity", "projected_gravity"),
                       ("JointPosition", "joint_pos"), ("JointVelocity", "joint_vel"), ("LastAction", "actions")):
        data = contract["observation_preprocessing"][term]
        scalar(f"k{name}ClipMin", data["clip"][0])
        scalar(f"k{name}ClipMax", data["clip"][1])
        scalar(f"k{name}Scale", data["scale"])
    lines.extend(("}  // namespace sysu219_trot", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model_2999.pt")
    parser.add_argument("--model-name", default="policy_trot_in_place", help="Output filename stem")
    parser.add_argument("--output-dir", type=Path, default=package / "model")
    parser.add_argument("--header", type=Path, default=package / "include/sysu219_guide_controller/control/TrotPolicyConfig.h")
    args = parser.parse_args()
    require(Path(args.model_name).name == args.model_name and args.model_name not in ("", ".", ".."), "Invalid model filename stem")
    checkpoint = args.run_dir / args.checkpoint
    env_path = args.run_dir / "params/env.yaml"
    agent_path = args.run_dir / "params/agent.yaml"
    env = yaml.load(env_path.read_text(), Loader=SavedConfigLoader)
    agent = yaml.safe_load(agent_path.read_text())
    contract = make_contract(env, agent)
    # Reject incompatible checkpoints before writing the model/header files.
    config_root = package.parents[3] / "src/quadruped_ros2_control-humble/descriptions/sysu219/sysu219_description/config"
    for filename in ("gazebo.yaml", "robot_control.yaml"):
        parameters = yaml.safe_load((config_root / filename).read_text())["sysu219_guide_controller"]["ros__parameters"]
        require(parameters["joints"] == contract["joint_names"], "ROS joint order differs: " + filename)
        stand = parameters["stand_pos"]
        require(len(stand) == 12 and all(math.isfinite(q) for q in stand), "Invalid ROS stand_pos: " + filename)
        require(all(abs(q - saved) <= 1e-9 for q, saved in zip(stand, contract["default_joint_positions"])),
                "Saved training default pose differs from ROS stand_pos (" + filename +
                "); retrain with the ROS pose before exporting")
    contract_digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_cfg = copy.deepcopy(agent["actor"])
    require(actor_cfg.pop("class_name") == "MLPModel", "Only feed-forward MLP actor is supported")
    observations = TensorDict({"policy": torch.zeros(1, 45)}, batch_size=[1])
    actor = MLPModel(observations, agent["obs_groups"], "actor", 12, **actor_cfg)
    actor.load_state_dict(state["actor_state_dict"], strict=True)
    actor.eval()
    export_model = actor.as_onnx(verbose=False).cpu().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (args.model_name + ".onnx")
    torch.onnx.export(export_model, (torch.zeros(1, 45),), str(output),
                      input_names=["obs"], output_names=["actions"], opset_version=18, dynamo=False)
    model = onnx.load(output)
    require(model.ir_version <= 10, "ONNX IR is newer than the installed C++ ONNX Runtime 1.18 supports")
    onnx.helper.set_model_props(model, {
        "sysu219.contract_sha256": contract_digest, "sysu219.checkpoint_sha256": sha256(checkpoint),
        "sysu219.task": "RobotLab-Isaac-Trot-InPlace-SYSU219-DR-v0" if env.get("domain_randomization") else "RobotLab-Isaac-Trot-InPlace-SYSU219-v0",
    })
    onnx.checker.check_model(model)
    onnx.save(model, output)
    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(219)
    max_error = 0.0
    for index in range(256):
        sample = rng.normal(0, 0.5, (1, 45)).astype(np.float32)
        sample[:, 3:6] = [0, 0, -1]
        sample[:, 30:42] = np.clip(sample[:, 30:42], -contract["action_clip"], contract["action_clip"])
        elapsed = index * 0.02
        running = max(elapsed - contract["clock"]["settle_s"], 0)
        phase = running / contract["clock"]["period_s"] % 1
        sample[:, 42:45] = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), min(running/contract["clock"]["ramp_s"], 1)]
        with torch.inference_mode():
            original = actor(TensorDict({"policy": torch.from_numpy(sample)}, batch_size=[1])).numpy()
        exported = evaluator.run(["actions"], {"obs": sample})[0]
        require(np.isfinite(exported).all(), "Nonfinite ONNX output")
        max_error = max(max_error, float(np.abs(original-exported).max()))
        require(np.allclose(original, exported, atol=2e-5, rtol=2e-5), f"ONNX numerical mismatch: sample {index}")
    write_header(contract, contract_digest, args.header, output.name)
    report = {"contract": contract, "contract_sha256": contract_digest,
              "training_randomization": env.get("domain_randomization"),
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
              "env_sha256": sha256(env_path), "agent_sha256": sha256(agent_path),
              "onnx_sha256": sha256(output), "onnx_ir_version": model.ir_version,
              "verification": {"backend": "onnx.reference.ReferenceEvaluator", "samples": 256,
                               "max_abs_error": max_error, "cpp_compiled_or_executed": False}}
    (args.output_dir / (args.model_name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"onnx": str(output), "shape": [1, 45, 12], "contract": contract_digest,
                      "samples": 256, "max_abs_error": max_error}, indent=2))


if __name__ == "__main__":
    main()
