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

| 策略文件 | action_rate | 额外动作/观测延迟 |
|---|---:|---:|
| [policy_trot_dr7cm_ar002_q010_dq030.onnx](src/quadruped_ros2_control-humble/controllers/sysu219_guide_controller/model/policy_trot_dr7cm_ar002_q010_dq030.onnx) | -0.02 | 0～4 ms |
| [policy_trot_dr7cm_ar003_q010_dq030.onnx](src/quadruped_ros2_control-humble/controllers/sysu219_guide_controller/model/policy_trot_dr7cm_ar003_q010_dq030.onnx) | -0.03 | 0～4 ms |
| [policy_trot_dr7cm_ar002_delay008_q010_dq030.onnx](src/quadruped_ros2_control-humble/controllers/sysu219_guide_controller/model/policy_trot_dr7cm_ar002_delay008_q010_dq030.onnx) | -0.02 | 0～8 ms |

以上三版初始关节角扰动均为±0.10 rad、初始关节速度±0.30 rad/s，各附同名部署JSON。
当前Gazebo选择延迟鲁棒版；真机配置仍为上一版 `policy_trot_ros_dr_7cm.onnx`。
通过对应YAML的 `rl_model_path` 选择模型，重启后生效，按7进入RLWALK。
上一版归档见 [checkpoint与训练配置](rl_training/experiments/sysu219/ros_dr_7cm/)。

## 训练与测试命令

创建好conda环境后，在工作区根目录执行：

```bash
export WS_ROOT="$PWD"
cd "$WS_ROOT/rl_training/robot_lab"
```

Domain Randomization正式训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-DR-v0 \
  --num_envs 1024 --max_iterations 3000 --device cuda:0 --headless \
  --logger tensorboard --seed 42 --run_name ros_dr_7cm
```

名义条件回放仓库归档的最终DR策略：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-Play-v0 \
  --checkpoint "$WS_ROOT/rl_training/experiments/sysu219/ros_dr_7cm/model_2999.pt" \
  --num_envs 1 --device cuda:0 --real-time --skip-export
```

随机条件回放：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-DR-Play-v0 \
  --checkpoint "$WS_ROOT/rl_training/experiments/sysu219/ros_dr_7cm/model_2999.pt" \
  --num_envs 16 --device cuda:0 --real-time --skip-export
```

测试新训练的策略时，将 `--checkpoint` 替换为对应的 `logs/rsl_rl/sysu219_trot_in_place/<运行目录>/model_*.pt`。

ROS Gazebo测试（完成ROS构建安装后，在新终端执行）：

```bash
cd /你的工作区路径
source /opt/ros/humble/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="/opt/onnxruntime/lib:${LD_LIBRARY_PATH:-}"
ros2 launch sysu219_guide_controller gazebo.launch.py pkg_description:=sysu219_description
```

另开终端并加载相同ROS环境，运行 `ros2 run keyboard_input keyboard_input`；按2依次到默认站立，稳定后按7进入RL，按2返回站立。

ONNX导出及完整步骤见 [SYSU219 RL 工作流](docs/sysu219_rl.md)。
