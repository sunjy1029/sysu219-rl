# SYSU219 原地 trot：ROS 名义站姿与 7 cm 模型

当前 Gazebo、真机 YAML 指定的模型均为 `policy_trot_ros_dr_7cm.onnx`，
相对于安装包 `share/sysu219_guide_controller/model/`。
来源为 `2026-10-27_17-25-46_ros_dr_7cm/model_2999.pt`，包含域随机化训练。
对应 JSON 记录 checkpoint、模型哈希、保存配置哈希和完整部署契约。
旧 `policy.onnx`、`policy_trot_in_place.onnx`、`policy_trot_ros_nominal_7cm.onnx` 及其 JSON 保留用于溯源。

## 当前部署约定

- 关节顺序 FR/FL/RR/RL，每腿 hip/thigh/calf。
- 默认站姿与 ROS `stand_pos` 相同：前腿 `[0, 0.9, -1.53]`，后腿 `[0, 0.9, -1.3]`。
- actor 输入 float32 `obs[1,45]`，输出 float32 `actions[1,12]`。
- 输入依次为机身角速度、投影重力、关节角减默认角、关节速度、上一帧裁剪动作、三维步态时钟。
- 裁剪和缩放顺序、幅度均取训练保存配置；无机身平动位置/速度或接触真值输入。
- 动作先裁剪至 ±2；关节尺度为 `[0.08, 0.20, 0.25]`，加默认角后再裁剪绝对目标。
- 前腿目标范围 hip ±0.16、thigh [0.5,1.3]、calf [-2.03,-1.03] rad。
- 后腿目标范围 hip ±0.16、thigh [0.5,1.3]、calf [-1.8,-0.888] rad。
- 策略 50 Hz，周期 0.975 s，站稳时钟 0.5 s，参考幅度渐入 1 s。
- MIT 命令为目标位置、速度 0、前馈力矩 0、kp 260、kd 3.8。
- 7 cm 是训练足底轨迹的峰值参考，不是 ROS 侧额外叠加的抬腿量。

RLWALK 读取 ROS `stand_pos`，与生成头文件不一致时拒绝启用。
模型部署契约摘要与生成头文件不一致时也拒绝加载，不能手改摘要绕过。
初始姿态前后腿不对称的名义俯仰已在训练中处理，ROS 使用实际 IMU 姿态构造重力观测。
RLWALK 不再先回到 stand_pos。满足原有关节角/速度入口检查后，立即启动策略时钟，
以进入时实际下发的位置命令和增益为起点，在0.5秒内直接融合到策略目标及训练增益。
`rl_pose_transition_s=0.5` 现在表示这一直接接管时长，不能超过训练站稳时长；
旧 `rl_action_transition_s` 已移除，避免叠加两段过渡。第一帧保持原承重命令，不额外等待。
训练默认角保持不变，继续用于观测偏差和动作解释；不把实测角重新定义为默认角。
接管期间 last_action 使用实际下发目标反算的等效动作并裁剪到±2；
若入口命令暂在策略动作包络外，该历史是裁剪近似。接管完成后恢复原始裁剪动作历史。
入口位置命令与训练默认角偏差超过入口容差会拒绝接管；等待反馈就绪时保持原命令。
策略时钟中的0.5秒站稳和1秒幅度渐入照常运行，接管融合与站稳阶段重叠。
这是部署接管行为变化，需用户重新编译安装并在Gazebo验证；1切PASSIVE，2切FIXEDSTAND。
不依赖 InEKF 里程计就绪门控；传统 TROTTING/BALANCETEST 保持原门控。

## 导出和只读检查

在 219_ws，使用 conda lab 的 Python：

实际导出入口为
`src/quadruped_ros2_control-humble/controllers/sysu219_guide_controller/scripts/export_trot_policy.py`，
参数为 `--run-dir rl_training/experiments/sysu219/ros_dr_7cm`
及 `--checkpoint model_2999.pt --model-name policy_trot_ros_dr_7cm`。
导出同时生成 ONNX、同名 JSON 和 `TrotPolicyConfig.h`，不会覆盖旧命名模型。
只读检查入口为同目录 `check_trot_deployment.py`，自动按 Gazebo YAML 指定的模型路径选择 JSON。

## 验证与测试前提

当前 DR 模型与训练 actor 的256组 ONNX参考执行器对比最大误差为9.536743e-7；
预编译 ORT 1.18.1 的256组误差见 `219_ws/analysis/sysu219_ros_dr_7cm_export/native_ort_report.json`。
模型、名义站姿、动作范围与时钟的静态检查通过。
当前 DR 和上一版名义模型的部署契约完全相同，当前已安装控制器包含相同的契约摘要；
已为新模型及JSON补齐安装目录符号链接，描述包安装配置也链接到更新后的源码 YAML。
因此本次更换模型可重启使用这些 YAML 的 ROS 进程后测试，无需为换模型重新编译。
上述仅适用于此前的模型权重切换；新增策略接管融合修改了C++控制逻辑，需用户重新编译安装后生效。
控制器仅在激活创建 RL 状态时加载模型，正在运行的进程不会因文件路径修改而自动换模型。
本次未编译、未启动 ROS、未改写控制器二进制。

DR 训练使用的物理/传感器误差范围记录于模型 JSON 的 `training_randomization`；
ROS 部署不人工加入噪声或随机化参数。训练使用域随机化不代表已通过真机验证。
先在 Gazebo 验证切换、抬腿、漂移和接触表现；硬件通信和 ROS 真机参数保持原样。


工作区已纳入 `rl_training/robot_lab/` 源码和最终实验归档；最新可移植命令见工作区 `docs/sysu219_rl.md`。历史JSON中的原路径用于来源记录，不作为当前运行路径。
