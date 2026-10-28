# SYSU219 原地 trot：第一阶段配置

本次只新增源码和静态检查，不执行编译、USD 转换、仿真或训练。
当前配置是待加载验证的训练起点，不能据此认为已经能稳定踏步或直接部署真机。
资产仍要求 Isaac Sim >= 5.0，原因见 `data/Robots/SYSU219/README.md`。

## 任务入口与文件

- 训练配置任务 ID：`RobotLab-Isaac-Trot-InPlace-SYSU219-v0`
- 小规模观察配置任务 ID：`RobotLab-Isaac-Trot-InPlace-SYSU219-Play-v0`
- `trot_env_cfg.py`：独立平地场景、观测、动作、奖励、重置和终止配置。
- `mdp.py`：按环境独立重置的时钟、对角足端参考和奖励。
- `task_parameters.json`：初始步态参数、名义姿态、动作及观测接口记录。
- `agents/rsl_rl_ppo_cfg.py`：初始 PPO 参数，actor/critic 分别使用 policy/critic 观测组。

该任务不继承 Go2 的速度命令及奖励配置，也没有“零命令时站着不动”的开关。
不修改 ROS xacro、当前模型资产、旧 Isaac Lab SYSU219 任务或原 ROS RLWALK 部署代码。

## 初始步态与场景

- 平地、固定物理参数，无推扰、质量/增益/姿态随机化，无观测噪声。
- 物理/执行器时间步 0.002 s，decimation=10，策略 50 Hz，每回合 20 s。
- 初始训练场景 1024 个环境，Play 配置 16 个环境；尚未评估显存需求。
- 周期 0.975 s，支撑占比 2/3，两组相位差半周期；每周期共 1/3 时间四足支撑，无预期腾空段。
- FR/FL/RR/RL 足端相位偏移 `[0, 0.5, 0.5, 0]`，即 FL+RR 同步、FR+RL 同步。
- 重置后 0.5 s 站稳阶段，随后 1 s 内抬脚参考幅度增加到 0.07 m（7 cm）。
  此为参考轨迹的渐入，不是强制位置控制；训练初期策略仍可能在站稳阶段乱动。
- 摆动段高度使用 `clearance * amplitude * sin(pi * swing_progress)^2`，端点高度和速度连续。
  7 cm 指平地足底相对地面的目标峰值高度（已扣除足端球半径），不是机身高度或关节角幅度；
  支撑期目标为 0。足底高度奖励权重仍为 4.0，误差尺度 `std` 仍为 0.01 m。
- 时钟源是 `common_step_counter - 每个环境最近reset时的counter`，不使用可能被 RSL-RL 随机化的
  `episode_length_buf`。部分环境重置不改变其他环境时钟。
- 时钟观测为 `[sin(2*pi*phase), cos(2*pi*phase), amplitude]`，重置后是 `[0, 1, 0]`。

训练使用 ROS `stand_pos`：前腿 `[0, 0.9, -1.53]`，后腿 `[0, 0.9, -1.3]`。
由 URDF 正运动学计算，使四足落在水平地面时基座名义俯仰约 +2.456°、高度约 0.4373655 m；
初始根部额外高 0.01 m。姿态奖励跟踪该名义重力方向，不要求机身强行水平。
这些是无形变几何参考，不能当作承重后的实际稳定姿态或机身质心高度。

名义增益为 260/3.8；执行器每 2 ms 计算一次总 PD 力矩、按 100 Nm 限幅，再经过 4 ms 一阶响应。
该响应复现 Gazebo 配置假设，不是真机带宽辨识；真机驱动的前馈限幅不能证明固件总 PD 力矩也按此限幅。
关节摩擦、黏性摩擦与 armature 沿用 xacro 辨识值；动作目标限制使用 xacro、Gazebo、硬件命令范围交集。
速度参数取三者较小值（hip/thigh 19.95、calf 20 rad/s），PhysX 速度限制只是仿真模型约束，
不同于硬件对速度命令的裁剪，也不保证真实电机具有相同饱和行为。

地面及机器人碰撞形状使用固定摩擦系数 0.2、恢复系数 0，地面采用 min 组合，覆盖当前 Gazebo 低摩擦基准。
这是接触模型近似：ROS 脚部 `{name}_foot` 标签拼写问题未修改，不能宣称各形状实际材料或两个引擎接触行为完全相同。
本次没有加入域随机化；启动事件只设置固定材料，不随机采样不同物理参数。

`task_parameters.json` 中 `ros_nominal` 保存来源哈希和基准参数。
`scripts/tools/sync_sysu219_trot_ros.py --check` 可检查 ROS 源码是否发生变化；
去掉 `--check` 可更新基准配置，不会修改 ROS 文件。若 xacro 改变，需先同步模型资产。

## 奖励与终止

正奖励：足端接触与相位匹配、足底高度跟踪；不依赖非零速度命令。
接触切换附近不计接触奖励，早期幅度小于 0.2 时按四足支撑处理。
正奖励乘以直立程度，避免倒地后继续刷接触/高度分。

负项：水平运动、累计XY漂移、偏航角及角速度、高度误差、倾斜和机身垂向/横滚俯仰运动；
足端固定落点偏移、支撑足滑动、摆动末段较大下落速度及接触力峰值；
关节力矩、加速度、越限、动作变化和非超时终止。
所有权重、5 N 接触阈值、350 N 冲击阈值都是初始设置，未经过训练验证。

终止条件：20 s 超时；倾斜超过35度；根部高度低于0.28 m；
水平漂移超过0.35 m或偏航变化超过0.7 rad；trunk 接触地面。

位置/速度/接触真值只用于仿真奖励、终止和 critic，不能在真机分析中当作估计量的真值。
相对初始位置的惩罚只能鼓励少漂移；actor没有绝对位置输入，不保证能回到原点。

## 策略输入输出契约

关节和动作顺序严格为：

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

观测按表中顺序拼接，第一版没有观测归一化和历史堆叠：

| actor索引 | 内容 | 缩放 |
|---|---|---|
| 0..2 | 机身坐标系角速度 | 0.25 |
| 3..5 | 机身坐标系重力单位向量 | 1 |
| 6..17 | 实际关节角 - 当前任务名义角 | 1 |
| 18..29 | 关节速度（默认速度为零） | 0.05 |
| 30..41 | 上次动作，RSL wrapper裁剪后、关节缩放前 | 1 |
| 42..44 | sin相位、cos相位、起步幅度 | 1 |

actor 共45维，无机身线速度、位置或接触真值。
角速度/关节观测显式限制在[-100,100]，重力限制在[-1,1]，上次动作限制在[-2,2]；
Isaac Lab ObservationManager 顺序是先clip再scale，因此部署时也应先裁剪原始量再缩放。
critic 在这45维之后添加机身坐标系线速度3维，以及世界系XY位置误差、相对偏航角、机身高度误差4维，共52维。

12维策略动作先由 RSL wrapper 裁剪到[-2,2]，之后：

```text
q_target = q_default + scale * action
scale（每条腿）= [0.08, 0.20, 0.25]
q_default（前腿）= [0.0, 0.9, -1.53]
q_default（后腿）= [0.0, 0.9, -1.3]
```

JointPositionAction 还对处理后的绝对关节目标做裁剪，范围为动作包络与 xacro、ROS 硬件/Gazebo 命令限幅的交集。
关节执行器使用 RosPDActuator 及辨识参数，不输出足力命令。

ROS 已有45维 RL 接口，并要求模型站姿匹配 `stand_pos`。重新训练后须成组导出 ONNX、JSON 与头文件；
旧 baseline 模型仍是旧站姿/时序，不能直接沿用，也不能只修改元数据。

## 静态检查

在 robot_lab 根目录执行：

```bash
/usr/bin/python3 scripts/tools/check_sysu219_trot.py \
  --isaaclab-source $WS_ROOT/rl_training/IsaacLab/source/isaaclab/isaaclab

/usr/bin/python3 scripts/tools/sync_sysu219.py \
  --source-description $WS_ROOT/src/quadruped_ros2_control-humble/descriptions/sysu219/sysu219_description \
  --check
```

检查脚本只读AST、JSON和URDF，使用NumPy进行正运动学核对；不导入torch、Isaac Lab或任务模块，
不触发这些库的JIT/运行时初始化。它检查语法、MDP函数及参数、配置字段、任务入口、观测契约、
名义足端位置与高度、动作限位和对角支撑时序；不替代实际环境加载或张量执行验证。

下一阶段应先加载小规模场景，验证USD足端body、传感器、初始姿态、观测维度与奖励数值，
再决定是否开始训练。本次没有执行训练命令。


## 域随机化版本（独立任务）

- 训练：`RobotLab-Isaac-Trot-InPlace-SYSU219-DR-v0`。
- 随机条件回放：`RobotLab-Isaac-Trot-InPlace-SYSU219-DR-Play-v0`，保留物理随机化和观测误差。
- 名义条件回放：原 `RobotLab-Isaac-Trot-InPlace-SYSU219-Play-v0` 可加载同一 DR checkpoint。
- 原名义任务保持不变；45/52 维 actor/critic、站姿、限幅、步态、7 cm 目标保持相同。

范围集中在 `domain_randomization.json`，为建议覆盖范围，并非真机误差统计。
约20%的环境（按数量取整）固定为名义参数、无观测误差及重置扰动；单环境 DR 回放为随机环境。
其余环境启动时一次性采样各部件质量 ±5%（惯量同倍率、epsilon 足质量不变）、
trunk COM 各轴 ±5 mm、kp/kd ±10%、库仑摩擦 ±20%、thigh 黏性摩擦 ±20%、
calf armature ±10%、响应时间 0～8 ms。零或近零的其余辨识参数保留名义值。
物理参数每环境固定，不逐帧跳变，也不累乘历史随机值。

材料为每机器人静摩擦 0.15～0.8、动摩擦为静摩擦的0.8～1倍、恢复系数0。
共享地面设1.0且采用min组合，让机器人材料决定有效摩擦；名义子集有效摩擦仍为0.2。
当前对每个机器人的所有碰撞形状设置相同材料，避免固定关节合并后依赖特定脚碰撞形状序号。

每回合重置时采样动作及观测延迟0/2/4 ms；编码器零位 ±0.005 rad、陀螺仪偏置 ±0.01 rad/s、
IMU roll/pitch偏差 ±1°。编码器零位同时影响 PD 反馈和 actor 观测，IMU偏差同时旋转角速度与重力。
每次策略观测叠加有界均匀噪声：q ±0.003 rad、dq ±0.03 rad/s、gyro ±0.01 rad/s。
噪声在clip/scale之前加入；重力通过旋转保持单位长度。
观测延迟使用2 ms历史采样，并非20 ms策略帧延迟；部分重置只清理对应环境历史。
策略动作历史与时钟保持真实输入，不加入虚假噪声。

初始关节角 ±0.02 rad、roll/pitch ±2°、水平速度每轴 ±0.03 m/s；
不修改 default_joint_pos，避免随机化改变 ROS 的观测偏置和动作默认角。
没有外力推扰、地形、强度衰减或随机硬限位。critic 和奖励使用干净仿真状态。

正式训练示例（用户执行；先50迭代短训）：

```bash
conda activate lab
cd $ROBOT_LAB_ROOT
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-DR-v0 \
  --num_envs 1024 --max_iterations 3000 --device cuda:0 --headless \
  --logger tensorboard --seed 42 --run_name ros_dr_7cm
```

日志仍在 `logs/rsl_rl/sysu219_trot_in_place/`，以运行名区分。
训练前只读检查：`python scripts/tools/check_sysu219_dr.py`。
导出脚本支持此版本的训练噪声，并在JSON记录范围；部署不人工添加这些噪声/延迟。
必须从 DR 训练保存的 env/agent 配置导出，不能把名义模型改名充当 DR 策略。
本次仅静态检查，未启动 Isaac Sim、训练或编译；首次短训仍需验证运行时 API 和张量形状。


路径约定：在工作区根目录设置 `export WS_ROOT="$PWD"`、`export ROBOT_LAB_ROOT="$WS_ROOT/rl_training/robot_lab"`。最新复现步骤见工作区 `docs/sysu219_rl.md`。
