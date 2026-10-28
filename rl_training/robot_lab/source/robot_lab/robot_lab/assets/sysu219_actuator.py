"""Gazebo leg_pd_controller's bounded PD and first-order torque response.

The response time is a Gazebo assumption, not identified hardware bandwidth.
"""

import math

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass


class RosPDActuator(IdealPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.response_time_s < 0 or cfg.update_dt_s <= 0:
            raise ValueError("Invalid ROS actuator timing")
        self._alpha = -math.expm1(-cfg.update_dt_s / cfg.response_time_s) if cfg.response_time_s else 1.0
        self._filtered_effort = torch.zeros_like(self.applied_effort)

    def reset(self, env_ids):
        super().reset(env_ids)
        self._filtered_effort[env_ids] = 0.0

    def compute(self, control_action, joint_pos, joint_vel):
        # Base class computes and clips the SUM of PD and feedforward torque.
        result = super().compute(control_action, joint_pos, joint_vel)
        self._filtered_effort += self._alpha * (self.applied_effort - self._filtered_effort)
        self.applied_effort[:] = self._filtered_effort
        result.joint_efforts = self.applied_effort
        return result


@configclass
class RosPDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = RosPDActuator
    response_time_s: float = 0.004
    update_dt_s: float = 0.002


class RandomizedRosPDActuator(RosPDActuator):
    """Same nominal MIT controller, with per-environment response and q delay."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.response_times = torch.full_like(self.applied_effort[:, :1], cfg.response_time_s)
        self.encoder_bias = torch.zeros_like(self.applied_effort)
        self.delay_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._command_history = torch.zeros((3, *self.applied_effort.shape), device=self._device)
        self._history_ready = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._head = 0
        self._env_ids = torch.arange(self._num_envs, device=self._device)
        self.observation_callback = None
        self.update_response()

    def update_response(self):
        times = self.response_times
        self._alpha = torch.where(times > 0,
                                  -torch.expm1(-self.cfg.update_dt_s/times.clamp_min(1e-12)),
                                  torch.ones_like(times))

    def reset(self, env_ids):
        super().reset(env_ids)
        self._history_ready[env_ids] = False

    def compute(self, control_action, joint_pos, joint_vel):
        if self.observation_callback is not None:
            self.observation_callback()
        command = control_action.joint_positions
        # Initialize partial resets with their current target, never zero angles
        # or commands carried over from the previous episode.
        missing = ~self._history_ready
        self._command_history[:, missing] = command[missing].unsqueeze(0)
        self._history_ready[:] = True
        self._head = (self._head+1) % 3
        self._command_history[self._head] = command
        control_action.joint_positions = self._command_history[
            (self._head-self.delay_steps) % 3, self._env_ids].clone()
        return super().compute(control_action, joint_pos+self.encoder_bias, joint_vel)


@configclass
class RandomizedRosPDActuatorCfg(RosPDActuatorCfg):
    class_type: type = RandomizedRosPDActuator
