# SYSU219 正式训练与回放入口

使用 conda `lab`。训练和回放沿用 `train.py`、`play.py`，不再依赖分析目录中的临时检查脚本。
本次补齐入口和无更新验证模式；没有执行 PPO 更新或正式训练。

## 源码路径与任务注册

两个脚本在导入 Isaac Lab 前调用同目录的 `source_setup.py`：

- 本进程优先使用当前 robot_lab 源码，不依赖它是否通过 pip 安装。
- Isaac Lab 优先取 `--isaaclab-root`，其次环境变量 `ISAACLAB_ROOT`。
- 未指定时寻找 robot_lab 同级的 `IsaacLab`，再寻找同级 `219_ws/rl_training/IsaacLab`。
- 没有找到本地源码时才使用已安装包；显式指定的路径错误时直接报错。
- 不修改 conda 的 editable 安装记录，不安装任何依赖。
- SYSU219 的两个任务ID自动设置 `ROBOT_LAB_TASK_PACKAGES`，正常导入真实的 robot_lab 包，
  只注册 SYSU219 所在任务包；不模拟包，也不扫描其他机器人缺失的 cusrl 配置。
- 未指定任务包时，保留原来的全库任务扫描行为。其他工具若需要定向注册，也可显式设置该环境变量。

RSL-RL 采用 lab 中的5.0.1；通过当前Isaac Lab已有的 `handle_deprecated_rsl_rl_cfg`，
把兼容配置转换为actor/critic及Gaussian distribution配置，不降级后端。
普通训练是否能稳定完成PPO更新和收敛，仍应单独进行短训验证。

## 1. 无训练入口验证

```bash
conda activate lab
cd $ROBOT_LAB_ROOT

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-v0 \
  --num_envs 1 --device cuda:0 --headless \
  --validate-only
```

此模式实际创建环境、actor/critic、优化器和rollout storage，验证确定性推理、checkpoint保存和重新加载。
不调用 `runner.learn()`，不收集训练rollout，不更新参数。会打印 `[VALIDATE] SUCCESS`。
验证文件独立保存在 `logs/rsl_rl/_entry_validation/sysu219_trot_in_place/` 下，
checkpoint名为 `entry_validation.pt`，其中标明 `trained=False`，不会混入正式训练的自动续训目录。

## 2. 小规模短训（下一步，由用户启动）

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-v0 \
  --num_envs 128 --max_iterations 50 \
  --device cuda:0 --headless --logger tensorboard --seed 42 --run_name smoke
```

确认更新和checkpoint正常，再尝试1024环境、3000次迭代。
正式日志位于 `logs/rsl_rl/sysu219_trot_in_place/<时间戳_运行名>/`。

## 3. 有限步回放检查

把路径替换为实际checkpoint，验证入口时可以使用未训练的 `entry_validation.pt`：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-Play-v0 \
  --num_envs 1 --device cuda:0 --headless \
  --checkpoint /实际路径/entry_validation.pt \
  --skip-export --max_steps 5
```

正常完成打印 `[PLAY] Completed 5 policy steps.`。这只验证加载和推理，不代表trot训练效果。
界面回放时去掉 `--headless` 并添加 `--real-time`；去掉 `--max_steps` 可持续回放。
`--skip-export` 避免回放时自动导出JIT/ONNX；需要导出时去掉该参数。
原地trot没有速度命令，不支持 `--keyboard`，使用时会给出明确报错。

## 4. 续训与观察

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Trot-InPlace-SYSU219-v0 \
  --num_envs 1024 --max_iterations 1000 --headless \
  --resume --load_run 实际运行目录名 --checkpoint model_49.pt

tensorboard --logdir logs/rsl_rl/sysu219_trot_in_place
```

所有命令仍受“禁止源码构建”的约束；运行仿真需要的JIT/着色器编译按已有授权处理。


路径约定：在工作区根目录设置 `export WS_ROOT="$PWD"`、`export ROBOT_LAB_ROOT="$WS_ROOT/rl_training/robot_lab"`。最新复现步骤见工作区 `docs/sysu219_rl.md`。
