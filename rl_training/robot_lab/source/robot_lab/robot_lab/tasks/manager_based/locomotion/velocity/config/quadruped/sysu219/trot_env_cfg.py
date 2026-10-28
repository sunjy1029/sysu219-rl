"""Initial flat-ground SYSU219 in-place trot task; no forward velocity command."""

import json
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as lab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from robot_lab.assets.sysu219 import SYSU219_CFG, SYSU219_FOOT_NAMES, SYSU219_JOINT_NAMES, SYSU219_MODEL_PARAMETERS

from robot_lab.assets.sysu219_actuator import RosPDActuatorCfg, RandomizedRosPDActuator

from . import mdp, dr_mdp


PARAMETERS = json.loads(Path(__file__).with_name("task_parameters.json").read_text())
_POSE = dict(zip(SYSU219_JOINT_NAMES, PARAMETERS["default_joint_positions"], strict=True))
_SCALES = dict(zip(SYSU219_JOINT_NAMES, PARAMETERS["action_scales"], strict=True))
_LIMITS = PARAMETERS["ros_nominal"]["joint_limits"]
_MODEL_JOINTS = SYSU219_MODEL_PARAMETERS["joints"]
_NOMINAL = PARAMETERS["ros_nominal"]
_TARGET_CLIP = {
    name: (
        max(_LIMITS[name]["lower"], q - PARAMETERS["action_clip"] * _SCALES[name]),
        min(_LIMITS[name]["upper"], q + PARAMETERS["action_clip"] * _SCALES[name]),
    )
    for name, q in _POSE.items()
}


def feet_asset():
    return SceneEntityCfg("robot", body_names=SYSU219_FOOT_NAMES, preserve_order=True)


def feet_sensor():
    return SceneEntityCfg("contact_forces", body_names=SYSU219_FOOT_NAMES, preserve_order=True)


def joints_asset():
    return SceneEntityCfg("robot", joint_names=SYSU219_JOINT_NAMES, preserve_order=True)


@configclass
class TrotCfg:
    period_s: float = PARAMETERS["period_s"]
    stance_ratio: float = PARAMETERS["stance_ratio"]
    clearance_m: float = PARAMETERS["clearance_m"]
    settle_s: float = PARAMETERS["settle_s"]
    ramp_s: float = PARAMETERS["ramp_s"]
    transition_margin: float = PARAMETERS["transition_margin"]
    foot_phase_offsets: list[float] = PARAMETERS["foot_phase_offsets"]
    foot_radius_m: float = PARAMETERS["foot_radius_m"]
    nominal_feet_xy_m: list[list[float]] = PARAMETERS["nominal_feet_xy_m"]
    target_base_height_m: float = PARAMETERS["target_base_height_m"]
    nominal_projected_gravity_b: list[float] = PARAMETERS["nominal_projected_gravity_b"]
    ground_friction: float = _NOMINAL["ground_friction"]


@configclass
class TrotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min", restitution_combine_mode="min",
            static_friction=_NOMINAL["ground_friction"], dynamic_friction=_NOMINAL["ground_friction"], restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
    )
    robot = SYSU219_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))


@configclass
class CommandsCfg:
    """No velocity commands; zero speed does not mean stop stepping in this task."""


@configclass
class ActionsCfg:
    joint_pos = lab_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=SYSU219_JOINT_NAMES, preserve_order=True,
        scale=_SCALES, use_default_offset=True, clip=_TARGET_CLIP,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 45 dimensions, all observable on hardware. No position/velocity truth.
        base_ang_vel = ObsTerm(func=lab_mdp.base_ang_vel, scale=0.25, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=lab_mdp.projected_gravity, clip=(-1.0, 1.0))
        joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel, params={"asset_cfg": joints_asset()}, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel, params={"asset_cfg": joints_asset()}, scale=0.05, clip=(-100.0, 100.0))
        actions = ObsTerm(func=lab_mdp.last_action, clip=(-2.0, 2.0))
        gait_clock = ObsTerm(func=mdp.gait_clock)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(PolicyCfg):
        # Seven additional privileged dimensions, excluded from exported actor.
        base_lin_vel = ObsTerm(func=lab_mdp.base_lin_vel, clip=(-100.0, 100.0))
        anchor_error = ObsTerm(func=mdp.anchor_error, clip=(-100.0, 100.0))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    material = EventTerm(func=mdp.set_nominal_material, mode="startup")
    reset = EventTerm(func=mdp.reset_trot, mode="reset")


@configclass
class RewardsCfg:
    contact_schedule = RewTerm(func=mdp.contact_schedule, weight=3.0, params={"sensor_cfg": feet_sensor()})
    foot_height = RewTerm(func=mdp.foot_height_tracking, weight=4.0, params={"asset_cfg": feet_asset(), "std": 0.01})
    planar_velocity = RewTerm(func=mdp.planar_velocity_l2, weight=-2.0)
    anchor_position = RewTerm(func=mdp.anchor_position_l2, weight=-10.0)
    anchor_yaw = RewTerm(func=mdp.anchor_yaw_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-30.0)
    flat_orientation = RewTerm(func=mdp.nominal_orientation_l2, weight=-2.0)
    vertical_velocity = RewTerm(func=lab_mdp.lin_vel_z_l2, weight=-2.0)
    roll_pitch_velocity = RewTerm(func=lab_mdp.ang_vel_xy_l2, weight=-0.1)
    yaw_velocity = RewTerm(func=mdp.yaw_velocity_l2, weight=-0.2)
    foot_xy = RewTerm(func=mdp.foot_xy_l2, weight=-20.0, params={"asset_cfg": feet_asset()})
    foot_slide = RewTerm(
        func=mdp.foot_slide_l2, weight=-0.5,
        params={"asset_cfg": feet_asset(), "sensor_cfg": feet_sensor()},
    )
    landing_velocity = RewTerm(func=mdp.landing_velocity_l2, weight=-0.2, params={"asset_cfg": feet_asset()})
    contact_force = RewTerm(func=mdp.contact_force_excess, weight=-1.0e-5, params={"sensor_cfg": feet_sensor()})
    joint_torques = RewTerm(func=lab_mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": joints_asset()})
    joint_acceleration = RewTerm(func=lab_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": joints_asset()})
    joint_limits = RewTerm(func=lab_mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": joints_asset()})
    action_rate = RewTerm(func=lab_mdp.action_rate_l2, weight=-0.01)
    failed = RewTerm(func=lab_mdp.is_terminated, weight=-5.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)
    fallen = DoneTerm(func=lab_mdp.bad_orientation, params={"limit_angle": math.radians(35.0)})
    low_base = DoneTerm(func=lab_mdp.root_height_below_minimum, params={"minimum_height": 0.28})
    drift = DoneTerm(func=mdp.excessive_drift)
    trunk_contact = DoneTerm(
        func=lab_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk"]), "threshold": 5.0},
    )


@configclass
class SYSU219TrotEnvCfg(ManagerBasedRLEnvCfg):
    scene: TrotSceneCfg = TrotSceneCfg(num_envs=1024, env_spacing=2.5)
    trot: TrotCfg = TrotCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = PARAMETERS["decimation"]
        self.episode_length_s = PARAMETERS["episode_length_s"]
        self.sim.dt = PARAMETERS["physics_dt_s"]
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        # ROS fixed-stand pose; keep the source model and ROS configuration unchanged.
        self.scene.robot.init_state.joint_pos = _POSE.copy()
        self.scene.robot.init_state.rot = tuple(PARAMETERS["initial_root_rotation_wxyz"])
        self.scene.robot.actuators["legs"] = RosPDActuatorCfg(
            joint_names_expr=SYSU219_JOINT_NAMES,
            stiffness=_NOMINAL["kp"], damping=_NOMINAL["kd"],
            effort_limit={name: values["effort"] for name, values in _LIMITS.items()},
            effort_limit_sim={name: values["effort"] for name, values in _LIMITS.items()},
            velocity_limit_sim={name: values["velocity"] for name, values in _LIMITS.items()},
            armature={name: values["armature"] for name, values in _MODEL_JOINTS.items()},
            friction={name: values["coulomb_friction"] for name, values in _MODEL_JOINTS.items()},
            dynamic_friction={name: values["coulomb_friction"] for name, values in _MODEL_JOINTS.items()},
            viscous_friction={name: values["viscous_friction"] for name, values in _MODEL_JOINTS.items()},
            response_time_s=_NOMINAL["torque_response_time_s"], update_dt_s=self.sim.dt,
        )
        if abs(self.sim.dt - _NOMINAL["actuator_dt_s"]) > 1e-12:
            raise ValueError("Physics step must match the ROS PD update interval")
        self.scene.robot.init_state.pos = (
            0.0, 0.0, self.trot.target_base_height_m + PARAMETERS["spawn_clearance_m"],
        )
        if self.trot.period_s <= 0 or self.trot.ramp_s <= 0 or self.trot.settle_s < 0:
            raise ValueError("Invalid trot timing")
        if not 0.5 <= self.trot.stance_ratio < 1.0:
            raise ValueError("Trot requires alternating swing with optional four-foot support")
        if not 0 <= self.trot.transition_margin < (1 - self.trot.stance_ratio) / 2:
            raise ValueError("Contact transition margin removes the entire swing interval")
        if self.trot.clearance_m <= 0 or self.trot.foot_radius_m <= 0:
            raise ValueError("Foot clearance and radius must be positive")
        if list(self.trot.foot_phase_offsets) != [0.0, 0.5, 0.5, 0.0]:
            raise ValueError("Expected FR/FL/RR/RL phases for diagonal trot")


@configclass
class SYSU219TrotEnvCfg_PLAY(SYSU219TrotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16


@configclass
class DomainRandomizationCfg:
    enabled: bool = True
    ranges: dict = json.loads(Path(__file__).with_name("domain_randomization.json").read_text())


@configclass
class DRPolicyCfg(ObservationsCfg.PolicyCfg):
    base_ang_vel = ObsTerm(func=dr_mdp.base_ang_vel, scale=0.25, clip=(-100.0, 100.0))
    projected_gravity = ObsTerm(func=dr_mdp.projected_gravity, clip=(-1.0, 1.0))
    joint_pos = ObsTerm(func=dr_mdp.joint_pos, params={"asset_cfg": joints_asset()}, clip=(-100.0, 100.0))
    joint_vel = ObsTerm(func=dr_mdp.joint_vel, params={"asset_cfg": joints_asset()}, scale=0.05, clip=(-100.0, 100.0))

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = True


@configclass
class DRObservationsCfg(ObservationsCfg):
    policy: DRPolicyCfg = DRPolicyCfg()
    # Critic inherits clean ground-truth observations, never biased actor data.


@configclass
class DREventsCfg:
    material = EventTerm(func=dr_mdp.startup, mode="startup")
    reset = EventTerm(func=dr_mdp.reset, mode="reset")


@configclass
class SYSU219TrotDREnvCfg(SYSU219TrotEnvCfg):
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    observations: DRObservationsCfg = DRObservationsCfg()
    events: DREventsCfg = DREventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["legs"].class_type = RandomizedRosPDActuator
        # All per-environment coefficients are <= 0.8; min(plane=1,robot=mu)
        # makes the randomized robot material the effective contact friction.
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0


@configclass
class SYSU219TrotDREnvCfg_PLAY(SYSU219TrotDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
