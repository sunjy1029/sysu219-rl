"""SYSU219 model exported from the lab's ROS source description.

Requires Isaac Sim >= 5.0: joint friction is expressed as torque, not the
dimensionless coefficient used in 4.5. This is an asset, not a training task.
"""

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.version import get_isaac_sim_version

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


if get_isaac_sim_version().major < 5:
    raise RuntimeError("SYSU219 requires Isaac Sim >= 5.0 for torque-based Coulomb/viscous friction.")

_ASSET_DIR = Path(ISAACLAB_ASSETS_DATA_DIR) / "Robots" / "SYSU219"
SYSU219_MODEL_PARAMETERS = json.loads((_ASSET_DIR / "model_parameters.json").read_text())
"""Source hashes, URDF joint dynamics and the ROS fixed-stand pose."""
SYSU219_JOINT_NAMES = SYSU219_MODEL_PARAMETERS["joint_names"]
"""Explicit controller order: FR/FL/RR/RL x hip/thigh/calf (do not assume USD order)."""
SYSU219_FOOT_NAMES = [f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL")]
_JOINTS = SYSU219_MODEL_PARAMETERS["joints"]


SYSU219_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_ASSET_DIR / "urdf" / "robot_isaac.urdf"),
        fix_base=False,
        # Import view has root trunk; foot joints use dont_collapse=true.
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # Explicit actuator computes PD torque; do not also enable USD drives.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SYSU219_MODEL_PARAMETERS["initial_height_m"]),
        joint_pos={name: values["default_position"] for name, values in _JOINTS.items()},
        joint_vel={name: 0.0 for name in SYSU219_JOINT_NAMES},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=SYSU219_JOINT_NAMES,
            stiffness=SYSU219_MODEL_PARAMETERS["stand_kp"],
            damping=SYSU219_MODEL_PARAMETERS["stand_kd"],
            effort_limit={name: values["effort"] for name, values in _JOINTS.items()},
            effort_limit_sim={name: values["effort"] for name, values in _JOINTS.items()},
            velocity_limit_sim={name: values["velocity"] for name, values in _JOINTS.items()},
            # Ja is a custom ROS extension: explicitly map it to joint inertia.
            armature={name: values["armature"] for name, values in _JOINTS.items()},
            # Native passive joint friction, separate from the controller's kd.
            friction={name: values["coulomb_friction"] for name, values in _JOINTS.items()},
            dynamic_friction={name: values["coulomb_friction"] for name, values in _JOINTS.items()},
            viscous_friction={name: values["viscous_friction"] for name, values in _JOINTS.items()},
        ),
    },
)
