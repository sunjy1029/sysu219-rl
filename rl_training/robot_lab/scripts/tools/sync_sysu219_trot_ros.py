"""Generate/check the trot's ROS nominal parameters without importing Isaac Lab.

Reads ROS source, the canonical URDF and the existing training parameters.
Does not compile, convert USD, start simulation or modify ROS files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from check_sysu219_trot import transform, require
from sysu219_paths import workspace_root


TASK = Path("source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/sysu219")


def derive(robot_lab, workspace):
    package = workspace / "src/quadruped_ros2_control-humble"
    description = package / "descriptions/sysu219/sysu219_description"
    hardware = package / "hardwares/hardware_sysu219/src/HardwareSysu219.cpp"
    controller = package / "controllers/sysu219_guide_controller"
    paths = [description / "config/robot_control.yaml", description / "config/gazebo.yaml",
             hardware, controller / "src/Sysu219GuideController.cpp",
             controller / "src/FSM/StateTrotting.cpp", controller / "launch/slippery_ground.world",
             *sorted((description / "xacro").glob("*.xacro"))]
    real = yaml.safe_load(paths[0].read_text())["sysu219_guide_controller"]["ros__parameters"]
    gazebo = yaml.safe_load(paths[1].read_text())
    sim = gazebo["sysu219_guide_controller"]["ros__parameters"]
    low = gazebo["leg_pd_controller"]["ros__parameters"]
    names = real["joints"]
    require(names == sim["joints"] == low["joints"], "ROS joint orders differ")
    for field in ("stand_pos", "stand_kp", "stand_kd"):
        require(real[field] == sim[field], "Real/Gazebo nominal mismatch: " + field)
    types = {kind: list(map(float, values.split(","))) for kind, values in
             re.findall(r'joint_type_limits_\["(hip|thigh|calf)"\]\s*=\s*\{([^}]+)\}', hardware.read_text())}
    require(len(types) == 3, "Cannot resolve hardware command limits")
    asset = robot_lab / "source/robot_lab/data/Robots/SYSU219"
    model = json.loads((asset / "model_parameters.json").read_text())
    # Never silently reuse an asset after its physical source has changed.
    for path in (description / "xacro").glob("*.xacro"):
        key = str(path.relative_to(description))
        require(model["source_sha256"][key] == hashlib.sha256(path.read_bytes()).hexdigest(),
                "Model asset must be resynced from changed ROS source: " + key)
    require(names == model["joint_names"], "Asset joint order mismatch")
    p = json.loads((robot_lab / TASK / "task_parameters.json").read_text())
    p["default_joint_positions"] = real["stand_pos"]
    limits = {}
    for i, name in enumerate(names):
        urdf = model["joints"][name]
        hw = types[name.split("_")[1]]
        limits[name] = dict(lower=max(urdf["lower"], hw[0], low["position_min"][i]),
                            upper=min(urdf["upper"], hw[1], low["position_max"][i]),
                            velocity=min(urdf["velocity"], abs(hw[2]), hw[3], low["velocity_limits"][i]),
                            effort=min(urdf["effort"], abs(hw[4]), hw[5], low["effort_limits"][i]))
        require(limits[name]["lower"] <= real["stand_pos"][i] <= limits[name]["upper"], "Stand pose outside limits")
    root = ET.parse(asset / "urdf/robot.urdf").getroot()
    matrices = {"base": np.eye(4)}
    pose = dict(zip(names, real["stand_pos"]))
    pending = root.findall("joint")
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in matrices]
        require(ready, "Disconnected URDF")
        for j in ready:
            matrix = transform(j.find("origin"))
            if j.get("name") in pose:
                q = pose[j.get("name")]
                x, y, z = map(float, j.find("axis").get("xyz").split())
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                rotation = np.eye(4)
                rotation[:3, :3] = np.eye(3) + math.sin(q)*cross + (1-math.cos(q))*(cross @ cross)
                matrix = matrix @ rotation
            matrices[j.find("child").get("link")] = matrices[j.find("parent").get("link")] @ matrix
            pending.remove(j)
    feet = np.array([matrices[f"{leg}_foot"][:3, 3] for leg in ("FR", "FL", "RR", "RL")])
    pitch = math.atan2(feet[0, 2]-feet[2, 2], feet[0, 0]-feet[2, 0])
    c, s = math.cos(pitch), math.sin(pitch)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    feet_world = feet @ rotation.T
    require(np.ptp(feet_world[:, 2]) < 1e-8, "ROS pose cannot place all feet on a horizontal plane")
    p["initial_root_rotation_wxyz"] = [math.cos(pitch/2), 0.0, math.sin(pitch/2), 0.0]
    p["nominal_projected_gravity_b"] = [s, 0.0, -c]
    p["nominal_feet_xy_m"] = feet_world[:, :2].tolist()
    p["target_base_height_m"] = p["foot_radius_m"] - float(feet_world[0, 2])
    wave = re.search(r'make_shared<WaveGenerator>\(([^,]+),\s*([^,]+),\s*Vec4\(([^)]+)\)', paths[3].read_text())
    require(wave is not None, "Cannot resolve ROS gait parameters")
    ratio = wave[2].strip().split("/")
    p["period_s"] = float(wave[1])
    p["stance_ratio"] = float(ratio[0]) / float(ratio[1]) if len(ratio) == 2 else float(ratio[0])
    p["foot_phase_offsets"] = list(map(float, wave[3].split(",")))
    p["clearance_m"] = float(re.search(r'gait_height_\s*=\s*([\d.]+)', paths[4].read_text())[1])
    # One actuator update per 500 Hz ROS PD interval; policy remains 50 Hz.
    p["physics_dt_s"] = 1.0 / low["update_rate"]
    p["decimation"] = round(0.02 / p["physics_dt_s"])
    require(abs(p["physics_dt_s"]*p["decimation"] - 0.02) < 1e-12, "Policy period must remain 20 ms")
    world = ET.parse(paths[5]).getroot()
    ode = world.find(".//friction/ode")
    mu, mu2 = float(ode.findtext("mu")), float(ode.findtext("mu2"))
    require(mu == mu2, "Anisotropic Gazebo friction requires a different model")
    p["ros_nominal"] = dict(kp=real["stand_kp"], kd=real["stand_kd"], joint_limits=limits,
                            torque_response_time_s=low["torque_response_time"],
                            actuator_dt_s=p["physics_dt_s"], ground_friction=mu,
                            controller_rate_hz=real["update_rate"], actuator_rate_hz=low["update_rate"],
                            source_sha256={str(f.relative_to(workspace)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-lab", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--ros-workspace", type=Path, default=workspace_root())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    values = derive(args.robot_lab, args.ros_workspace)
    path = args.robot_lab / TASK / "task_parameters.json"
    if args.check:
        require(json.loads(path.read_text()) == values, "ROS nominal parameters are stale; run sync without --check")
    else:
        path.write_text(json.dumps(values, indent=2) + "\n")
    print("ROS nominal pose, gait, gains, command limits, contact baseline and actuator response: OK")


if __name__ == "__main__":
    main()
