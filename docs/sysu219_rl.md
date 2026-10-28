# SYSU219：训练、导出与 ROS 部署

本工作区将 ROS 与 RL 源码一起管理。`rl_training/COLCON_IGNORE` 防止 ROS 构建工具扫描训练代码。
`robot_lab` 为源码快照，不是嵌套 Git 仓库；来源见其 `SYSU219_UPSTREAM.md`，保留上游许可证。

## 目录与环境

- `src/.../sysu219_description`：权威 xacro、网格、ROS 站姿配置。
- `rl_training/robot_lab`：SYSU219 模型资产、名义任务、DR 任务及训练入口。
- `rl_training/IsaacLab`：本地 IsaacLab 源码。
- `rl_training/experiments/sysu219/ros_dr_7cm`：最终 DR checkpoint、保存的配置和校验记录。
- ROS 控制器的 `model/`：ONNX、部署 JSON；对应 `TrotPolicyConfig.h` 必须匹配部署契约。

训练使用 Python 3.11、Isaac Sim 5.1、IsaacLab 2.3.2、RSL-RL 5.0.1、PyTorch 2.7.0+cu128。
完整已验证版本清单见 `rl_training/experiments/sysu219/environment.json`。
ROS 使用 Humble，原生 ONNX Runtime 使用外部安装的1.18.1。
版本清单不是一键安装脚本；首次环境配置仍需安装这些外部依赖。ROS Python 与训练 Python 分开使用。
训练入口会优先使用本仓库源码，无需把 robot_lab 重新 pip 安装到固定绝对路径。

在**当前工作区根目录**设置路径（不要假设目录必须叫219_ws）：

```bash
export WS_ROOT="$PWD"
export ROBOT_LAB_ROOT="$WS_ROOT/rl_training/robot_lab"
export CTRL_ROOT="$WS_ROOT/src/quadruped_ros2_control-humble/controllers/sysu219_guide_controller"
```

## 静态检查

以下不启动仿真、训练或编译：

```bash
/usr/bin/python3 "$ROBOT_LAB_ROOT/scripts/tools/check_sysu219_trot.py"
/usr/bin/python3 "$ROBOT_LAB_ROOT/scripts/tools/check_sysu219_dr.py"
/usr/bin/python3 "$ROBOT_LAB_ROOT/scripts/tools/sync_sysu219_trot_ros.py" --check

conda activate lab
python "$CTRL_ROOT/scripts/check_trot_deployment.py"
```

模型同步还需要 ROS Python 的 xacro、PyYAML、NumPy：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 "$ROBOT_LAB_ROOT/scripts/tools/sync_sysu219.py" \
  --source-description "$WS_ROOT/src/quadruped_ros2_control-humble/descriptions/sysu219/sysu219_description" \
  --check
```

若权威 xacro 改变，先去掉模型同步的 `--check` 更新资产，再更新名义参数：
`sync_sysu219_trot_ros.py`（不带 `--check`）。改模型后必须重新验证，而不是覆盖已有训练的保存配置。

## 训练

以下由用户在具备仿真环境时执行。首次先短训确认运行时 API、张量形状与物理参数有效。

```bash
conda activate lab
cd "$ROBOT_LAB_ROOT"
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-DR-v0 \
  --num_envs 128 --max_iterations 50 --device cuda:0 --headless \
  --logger tensorboard --seed 42 --run_name ros_dr_7cm_smoke
```

正式训练使用同一命令，将环境数改为1024、迭代数改为3000、运行名改为 `ros_dr_7cm`。
不加 `--resume` 即从头训练。输出位于新目录下的 `logs/rsl_rl/sysu219_trot_in_place/`，已忽略。
名义任务为 `RobotLab-Isaac-Trot-InPlace-SYSU219-v0`。
3000次迭代是首轮预算，是否继续应看随机条件下的回合长度、失败率及回放表现。

DR 范围见任务目录 `domain_randomization.json`；约20%环境保持名义条件。
默认角为前腿 `[0,0.9,-1.53]`、后腿 `[0,0.9,-1.3]`，抬腿目标7 cm，策略50 Hz，步态周期0.975秒。
域随机化不代表已经验证真机稳定性。

## 回放归档模型

```bash
python "$ROBOT_LAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-Play-v0 \
  --checkpoint "$WS_ROOT/rl_training/experiments/sysu219/ros_dr_7cm/model_2999.pt" \
  --num_envs 1 --device cuda:0 --real-time --skip-export
```

随机条件回放改用 `RobotLab-Isaac-Trot-InPlace-SYSU219-DR-Play-v0`、`--num_envs 16`。
前者使用名义条件，后者保留物理随机化、传感器偏差与噪声。

## 导出 ONNX

导出器只读取训练保存的 env/agent 和 checkpoint，不导入或启动 Isaac Sim。
以下会重新生成对应 ONNX、JSON 和 C++ 部署头文件；三者应成组保存。

```bash
python "$CTRL_ROOT/scripts/export_trot_policy.py" \
  --run-dir "$WS_ROOT/rl_training/experiments/sysu219/ros_dr_7cm" \
  --checkpoint model_2999.pt --model-name policy_trot_ros_dr_7cm
python "$CTRL_ROOT/scripts/check_trot_deployment.py"
```

ONNX 不包含训练噪声注入，ROS 使用真实传感器反馈和约定的输入预处理。
当前 ROS YAML 指向 `policy_trot_ros_dr_7cm.onnx`；相对路径基于控制器安装目录 `model/`。
变更默认角、动作解释、时钟或 C++ 接管逻辑后，需要用户重新编译安装；仅换同契约权重时可以只更新模型。
本机安装目录和符号链接不会随 Git 上传，新克隆必须按 ROS 安装流程生成它们。

## ROS 测试

ROS 构建及仿真由用户执行，本次整理未执行编译。完成构建安装后，新开终端：

```bash
source /opt/ros/humble/setup.bash
source "$WS_ROOT/install/setup.bash"
export LD_LIBRARY_PATH="/opt/onnxruntime/lib:${LD_LIBRARY_PATH:-}"
ros2 launch sysu219_guide_controller gazebo.launch.py pkg_description:=sysu219_description
```

另一个同样 source 环境的终端运行 `ros2 run keyboard_input keyboard_input`。
按2依次经过全趴、半趴、默认站立，等待稳定后按7进入RL；按2返回站立。
传统TROTTING从站立按4进入，仍受InEKF就绪和姿态检查约束；RL中不能直接按4切传统模式。
按1为PASSIVE，会失去主动支撑。

RL入口保留进入时承重位置命令/增益，在0.5秒内直接融合到策略目标，取消强制回默认角。
这属于 ROS 部署逻辑，归档 checkpoint 并未因此修改。传统控制逻辑保持独立。

## 本地整理与 Git

保留上游 LICENSE、来源commit、当前模型、最终checkpoint和配置；不提交 build/install/log、训练logs、视频和缓存。
`analysis/` 为本机临时分析目录；精选导出校验报告已归档到 experiments。
实验保存配置中的旧绝对路径保留作来源记录；使用上述相对仓库路径运行，无需原机器目录。
本次整理不创建提交、不推送、不更改 origin。最终提交应推送到你自己的 `219_ws` 仓库。
