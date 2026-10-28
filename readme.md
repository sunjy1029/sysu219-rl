# SYSU219：ROS 控制与强化学习

本仓库基于原 `219_ws` 工作区，增加 SYSU219 原地 trot 强化学习训练及 ROS 部署支持。

## 相对原仓库的改动

- 在 `rl_training/robot_lab/` 纳入 robot_lab 源码，新增 SYSU219 模型、名义参数任务及域随机化任务。
- 以 ROS xacro、默认站姿和控制参数为训练基准，增加对角步态相位、7 cm 抬腿目标及对应奖励。
- 适配 ROS `RLPolicy` 和 `StateRLWalk`，对齐45维观测、关节顺序、动作解释及50 Hz推理时钟。
- RL入口保留当前承重命令，在0.5秒内直接平滑接管，取消先回默认站姿；传统TROTTING控制逻辑保留。
- 增加 ONNX 导出、部署一致性检查，以及最终训练配置和checkpoint归档。

## 当前部署策略

使用 **ROS 名义参数＋域随机化、7 cm 抬腿目标的原地 trot 策略**：

- ROS模型：[policy_trot_ros_dr_7cm.onnx](src/quadruped_ros2_control-humble/controllers/sysu219_guide_controller/model/policy_trot_ros_dr_7cm.onnx)。
- 训练来源：`2026-10-27_17-25-46_ros_dr_7cm/model_2999.pt`。
- 归档：[最终checkpoint与训练配置](rl_training/experiments/sysu219/ros_dr_7cm/)。
- Gazebo和真机配置中的 `rl_model_path` 均指向该模型；按 `7` 进入RLWALK。

训练、回放、导出和测试命令见 [SYSU219 RL 工作流](docs/sysu219_rl.md)。
