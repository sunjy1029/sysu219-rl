"""Phase-conditioned in-place trot. No velocity-command gates or simulator startup."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def set_nominal_material(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    """Assign fixed coefficients to every robot shape; no domain randomization."""
    robot = env.scene["robot"]
    ids = torch.arange(env.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    materials = robot.root_physx_view.get_material_properties()
    materials[ids, :, 0:2] = env.cfg.trot.ground_friction
    materials[ids, :, 2] = 0.0
    robot.root_physx_view.set_material_properties(materials, ids)


def nominal_orientation_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize tilt relative to ROS stand, including its nominal pitch."""
    reference = torch.tensor(env.cfg.trot.nominal_projected_gravity_b, device=env.device)
    return (env.scene["robot"].data.projected_gravity_b - reference).square().sum(dim=1)


def reset_trot(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Reset robot and clock together, including partial vectorized resets.

    Never derive gait time from episode_length_buf: RSL-RL may randomize that
    buffer for rollout collection without resetting the physical robot.
    """
    lab_mdp.reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    if not hasattr(env, "_sysu219_trot_start_step"):
        env._sysu219_trot_start_step = torch.full(
            (env.num_envs,), env.common_step_counter, device=env.device, dtype=torch.long
        )
    env._sysu219_trot_start_step[env_ids] = env.common_step_counter


def _clock(env: ManagerBasedRLEnv):
    cfg = env.cfg.trot
    start = getattr(env, "_sysu219_trot_start_step", None)
    # Observation managers infer dimensions before the first reset event.
    if start is None:
        elapsed = torch.zeros(env.num_envs, device=env.device)
    else:
        elapsed = (env.common_step_counter - start).to(torch.float32) * env.step_dt
    running_time = (elapsed - cfg.settle_s).clamp_min(0.0)
    phase = torch.remainder(running_time / cfg.period_s, 1.0)
    amplitude = (running_time / cfg.ramp_s).clamp(0.0, 1.0)
    return phase, amplitude


def gait_clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Actor clock: sin(phase), cos(phase), startup amplitude; resets to [0,1,0]."""
    phase, amplitude = _clock(env)
    return torch.stack((torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase), amplitude), dim=1)


def _reference(env: ManagerBasedRLEnv):
    cfg = env.cfg.trot
    phase, amplitude = _clock(env)
    offsets = torch.tensor(cfg.foot_phase_offsets, device=env.device)
    foot_phase = torch.remainder(phase[:, None] + offsets, 1.0)
    progress = ((foot_phase - cfg.stance_ratio) / (1.0 - cfg.stance_ratio)).clamp(0.0, 1.0)
    height = cfg.clearance_m * amplitude[:, None] * torch.sin(math.pi * progress).square()
    stance = (foot_phase < cfg.stance_ratio) | (amplitude[:, None] < 0.2)
    margin = cfg.transition_margin
    valid = ((foot_phase - cfg.stance_ratio).abs() > margin) & (foot_phase > margin) & (foot_phase < 1 - margin)
    valid |= amplitude[:, None] < 0.2
    return foot_phase, height, stance, valid


def _upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (-env.scene["robot"].data.projected_gravity_b[:, 2] / 0.7).clamp(0.0, 1.0)


def _contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float):
    force = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return force[:, :, 2] > threshold


def contact_schedule(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    _, _, stance, valid = _reference(env)
    matched = (_contact(env, sensor_cfg, threshold) == stance).float()
    # Transition windows carry no reward rather than inventing perfect contacts.
    return (matched * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1) * _upright(env)


def foot_height_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.01) -> torch.Tensor:
    _, target, _, _ = _reference(env)
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = foot_z - env.scene.env_origins[:, 2:3] - env.cfg.trot.foot_radius_m
    return torch.exp(-((clearance - target) / std).square()).mean(dim=1) * _upright(env)


def anchor_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged critic input: world-frame XY drift, relative yaw, height error."""
    data = env.scene["robot"].data
    anchor = data.default_root_state[:, :3] + env.scene.env_origins
    xy = data.root_pos_w[:, :2] - anchor[:, :2]
    yaw = euler_xyz_from_quat(data.root_quat_w)[2]
    default_yaw = euler_xyz_from_quat(data.default_root_state[:, 3:7])[2]
    yaw_error = wrap_to_pi(yaw - default_yaw)
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - env.cfg.trot.target_base_height_m
    return torch.cat((xy, yaw_error[:, None], height[:, None]), dim=1)


def planar_velocity_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_lin_vel_w[:, :2].square().sum(dim=1)


def yaw_velocity_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_ang_vel_b[:, 2].square()


def anchor_position_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return anchor_error(env)[:, :2].square().sum(dim=1)


def anchor_yaw_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return anchor_error(env)[:, 2].square()


def base_height_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return anchor_error(env)[:, 3].square()


def foot_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, :2]
    nominal = torch.tensor(env.cfg.trot.nominal_feet_xy_m, device=env.device)
    target = env.scene.env_origins[:, None, :2] + nominal[None, :, :]
    return (feet - target).square().sum(dim=2).mean(dim=1)


def foot_slide_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float = 5.0
) -> torch.Tensor:
    velocity = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return (velocity.square().sum(dim=2) * _contact(env, sensor_cfg, threshold)).mean(dim=1)


def landing_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_downward_speed: float = 0.15) -> torch.Tensor:
    phase, _, _, _ = _reference(env)
    vz = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    _, amplitude = _clock(env)
    late_swing_start = env.cfg.trot.stance_ratio + 0.75 * (1.0 - env.cfg.trot.stance_ratio)
    late_swing = (phase > late_swing_start) & (amplitude[:, None] > 0)
    return ((-vz - max_downward_speed).clamp_min(0).square() * late_swing).mean(dim=1)


def contact_force_excess(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 350.0) -> torch.Tensor:
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    peak = forces.norm(dim=-1).amax(dim=1)
    return (peak - threshold).clamp_min(0).square().mean(dim=1)


def excessive_drift(env: ManagerBasedRLEnv, distance: float = 0.35, yaw_limit: float = 0.7) -> torch.Tensor:
    error = anchor_error(env)
    return (error[:, :2].norm(dim=1) > distance) | (error[:, 2].abs() > yaw_limit)
