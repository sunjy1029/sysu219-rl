"""SYSU219 in-place trot task registrations (configurations are loaded by entry point)."""

import gymnasium as gym


for task_id, env_class in (
    ("RobotLab-Isaac-Trot-InPlace-SYSU219-v0", "SYSU219TrotEnvCfg"),
    ("RobotLab-Isaac-Trot-InPlace-SYSU219-Play-v0", "SYSU219TrotEnvCfg_PLAY"),
    ("RobotLab-Isaac-Trot-InPlace-SYSU219-DR-v0", "SYSU219TrotDREnvCfg"),
    ("RobotLab-Isaac-Trot-InPlace-SYSU219-DR-Play-v0", "SYSU219TrotDREnvCfg_PLAY"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.trot_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SYSU219TrotPPORunnerCfg",
        },
    )
