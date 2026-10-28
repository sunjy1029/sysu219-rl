"""ROS-centred domain randomization; actor remains 45D, critic stays clean.

Physics parameters are sampled once per environment at startup. Sensor biases,
delays and initial perturbations are resampled on reset. White noise is sampled
once per observation. No random physical parameters are fed to the actor.
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

from . import mdp
from robot_lab.assets.sysu219 import SYSU219_JOINT_NAMES


def _uniform(bounds, shape, device):
    return torch.empty(shape, device=device).uniform_(float(bounds[0]), float(bounds[1]))


def _raw(env):
    robot = env.scene["robot"]
    ids = env._sysu219_dr["joint_ids"]
    return torch.cat((robot.data.root_ang_vel_b, robot.data.projected_gravity_b,
                      robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids]), dim=1)


def startup(env, env_ids):
    """Set each environment's material, mass/inertia/COM and joint parameters."""
    robot = env.scene["robot"]
    cfg = env.cfg.domain_randomization.ranges
    n = env.num_envs
    ids = torch.arange(n, device="cpu")
    random_cpu = torch.ones(n, dtype=torch.bool)
    random_cpu[torch.randperm(n)[:round(n*cfg["nominal_fraction"])]] = False
    mask = random_cpu.to(env.device)
    joint_ids, _ = robot.find_joints(SYSU219_JOINT_NAMES, preserve_order=True)
    actuator = robot.actuators["legs"]
    env._sysu219_dr = dict(mask=mask, joint_ids=joint_ids, cfg=cfg,
                           history=torch.zeros((2, n, 30), device=env.device),
                           head=0, cache=None, cache_step=-1)
    state = env._sysu219_dr
    # The shared plane is 1.0/min. These per-robot values therefore determine
    # effective contact friction, including the nominal subset's exact 0.2.
    materials = robot.root_physx_view.get_material_properties().clone()
    mu = _uniform(cfg["friction_static"], (n,), "cpu")
    ratio = _uniform(cfg["friction_dynamic_ratio"], (n,), "cpu")
    mu[~random_cpu] = env.cfg.trot.ground_friction
    ratio[~random_cpu] = 1.0
    materials[:, :, 0] = mu[:, None]
    materials[:, :, 1] = (mu*ratio)[:, None]
    materials[:, :, 2] = 0.0
    robot.root_physx_view.set_material_properties(materials, ids)
    masses = robot.root_physx_view.get_masses().clone()
    inertias = robot.root_physx_view.get_inertias().clone()
    scale = _uniform(cfg["mass_scale"], tuple(masses.shape), "cpu")
    scale[~random_cpu] = 1.0
    # Preserve the importer's epsilon foot masses rather than inventing payload.
    for i, name in enumerate(robot.body_names):
        if name.endswith("_foot"):
            scale[:, i] = 1.0
    robot.root_physx_view.set_masses(masses*scale, ids)
    robot.root_physx_view.set_inertias(inertias*scale[..., None], ids)
    coms = robot.root_physx_view.get_coms().clone()
    offset = _uniform(cfg["trunk_com_offset_m"], (n, 3), "cpu")
    offset[~random_cpu] = 0.0
    coms[:, robot.body_names.index("trunk"), :3] += offset
    robot.root_physx_view.set_coms(coms, ids)

    def scale_property(value, bounds):
        factors = _uniform(bounds, tuple(value.shape), env.device)
        factors[~mask] = 1.0
        return value.clone()*factors

    actuator.stiffness[:] = scale_property(actuator.stiffness, cfg["kp_scale"])
    actuator.damping[:] = scale_property(actuator.damping, cfg["kd_scale"])
    # Use the articulation's native joint order for these physics tensors.
    static = scale_property(robot.data.default_joint_friction_coeff, cfg["coulomb_scale"])
    viscous = scale_property(robot.data.default_joint_viscous_friction_coeff, cfg["viscous_scale"])
    # Hip/calf B values are effectively zero and are kept exactly at baseline.
    viscous[:, [i for i, name in enumerate(robot.joint_names) if "thigh" not in name]] = robot.data.default_joint_viscous_friction_coeff[:, [i for i, name in enumerate(robot.joint_names) if "thigh" not in name]]
    armature = scale_property(robot.data.default_joint_armature, cfg["armature_scale"])
    robot.write_joint_friction_coefficient_to_sim(static, static, viscous)
    robot.write_joint_armature_to_sim(armature)
    actuator.response_times[:] = _uniform(cfg["response_time_s"], (n, 1), env.device)
    actuator.response_times[~mask] = actuator.cfg.response_time_s
    actuator.update_response()
    # Sensor history is captured at each 2 ms actuator tick. Observation delay
    # zero reads fresh post-physics state, delays 1/2 read t-2/t-4 ms histories.
    def record():
        state["head"] = (state["head"]+1) % 2
        state["history"][state["head"]] = _raw(env)
    actuator.observation_callback = record
    print(f"[SYSU219 DR] {int((~mask).sum())}/{n} nominal environments; remaining randomized; schema={cfg['schema']}")


def reset(env, env_ids):
    mdp.reset_trot(env, env_ids)
    state = env._sysu219_dr
    cfg = state["cfg"]
    robot = env.scene["robot"]
    actuator = robot.actuators["legs"]
    ids = env_ids
    count = len(ids)
    mask = state["mask"][ids]

    def sample(key, width):
        out = _uniform(cfg[key], (count, width), env.device)
        out[~mask] = 0.0
        return out

    def storage(name, width, dtype=torch.float32):
        if name not in state:
            state[name] = torch.zeros((env.num_envs, width), dtype=dtype, device=env.device)
        return state[name]

    bias = sample("encoder_bias_rad", 12)
    storage("encoder_bias", 12)[ids] = bias
    # Encoder zero error affects the policy AND the PD feedback in the same
    # logical joint coordinate; it is not an independent target-angle offset.
    order = [SYSU219_JOINT_NAMES.index(name) for name in actuator.joint_names]
    actuator.encoder_bias[ids] = bias[:, order]
    storage("gyro_bias", 3)[ids] = sample("gyro_bias_rad_s", 3)
    tilt = sample("imu_tilt_bias_deg", 2)*math.pi/180
    quat = quat_from_euler_xyz(tilt[:, 0], tilt[:, 1], torch.zeros(count, device=env.device))
    if "imu_bias" not in state:
        state["imu_bias"] = torch.zeros((env.num_envs, 4), device=env.device)
        state["imu_bias"][:, 0] = 1
    state["imu_bias"][ids] = quat
    for key, target in (("action_delay_steps", actuator.delay_steps),
                        ("observation_delay_steps", storage("observation_delay", 1, dtype=torch.long))):
        low, high = cfg[key]
        delay = torch.randint(low, high+1, (count,), dtype=torch.long, device=env.device)
        delay[~mask] = 0
        if target.ndim == 1:
            target[ids] = delay
        else:
            target[ids, 0] = delay
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    tilt = sample("initial_tilt_deg", 2)*math.pi/180
    perturbation = quat_from_euler_xyz(tilt[:, 0], tilt[:, 1], torch.zeros(count, device=env.device))
    root[:, 3:7] = quat_mul(perturbation, root[:, 3:7])
    root[:, 7:9] += sample("initial_planar_velocity_m_s", 2)
    robot.write_root_state_to_sim(root, env_ids=ids)
    q = robot.data.default_joint_pos[ids].clone()
    q[:, state["joint_ids"]] += sample("initial_joint_offset_rad", 12)
    robot.write_joint_state_to_sim(q, robot.data.default_joint_vel[ids], env_ids=ids)
    # Do not change default_joint_pos: it defines actor offsets and ROS targets.
    state["history"][:, ids] = _raw(env)[ids].unsqueeze(0)
    state["cache"] = None


def _observed(env):
    state = getattr(env, "_sysu219_dr", None)
    if state is None or "encoder_bias" not in state:
        # ObservationManager infers dimensions before startup/reset events.
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(SYSU219_JOINT_NAMES, preserve_order=True)
        return torch.cat((robot.data.root_ang_vel_b, robot.data.projected_gravity_b,
                          robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids]), dim=1)
    step = env._sim_step_counter
    if state["cache"] is not None and state["cache_step"] == step:
        return state["cache"]
    result = _raw(env).clone()
    delay = state["observation_delay"][:, 0].long()
    for lag in (1, 2):
        selected = delay == lag
        result[selected] = state["history"][(state["head"]-(lag-1)) % 2, selected]
    result[:, 0:3] = quat_apply_inverse(state["imu_bias"], result[:, 0:3]) + state["gyro_bias"]
    result[:, 3:6] = quat_apply_inverse(state["imu_bias"], result[:, 3:6])
    result[:, 6:18] += state["encoder_bias"]
    if env.cfg.observations.policy.enable_corruption:
        for key, start, end in (("gyro_noise_rad_s", 0, 3), ("joint_pos_noise_rad", 6, 18),
                                ("joint_vel_noise_rad_s", 18, 30)):
            noise = _uniform(state["cfg"][key], (env.num_envs, end-start), env.device)
            noise[~state["mask"]] = 0.0
            result[:, start:end] += noise
    state["cache"], state["cache_step"] = result, step
    return result


def base_ang_vel(env):
    return _observed(env)[:, :3]


def projected_gravity(env):
    return _observed(env)[:, 3:6]


def joint_pos(env, asset_cfg):
    return _observed(env)[:, 6:18] - env.scene["robot"].data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel(env, asset_cfg):
    return _observed(env)[:, 18:30] - env.scene["robot"].data.default_joint_vel[:, asset_cfg.joint_ids]
