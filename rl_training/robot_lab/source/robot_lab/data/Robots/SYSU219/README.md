# SYSU219 模型接入

本目录的唯一模型来源是 ROS 源码包 `sysu219_description` 中的
`xacro/robot.xacro`、其引用的 xacro 及 `meshes/`。
不使用 `219_ws/rl_training` 内的旧 URDF、USD 或网格副本。

当前状态：已完成静态接入，并在 conda lab / Isaac Sim 5.1 中验证单机器人零动作仿真。
未运行策略训练；这不代表已经学会 trot。
`assets/sysu219.py` 提供 `SYSU219_CFG`，本次不注册训练任务。

## 更新与静态检查

在 robot_lab 根目录运行以下命令。使用安装了 xacro、PyYAML 和 NumPy 的 ROS Python；
ROS Python 需具备这些依赖。这两个命令不导入 Isaac Lab，不编译或启动仿真。

```bash
/usr/bin/python3 scripts/tools/sync_sysu219.py \
  --source-description $WS_ROOT/src/quadruped_ros2_control-humble/descriptions/sysu219/sysu219_description

/usr/bin/python3 scripts/tools/sync_sysu219.py \
  --source-description $WS_ROOT/src/quadruped_ros2_control-humble/descriptions/sysu219/sysu219_description \
  --check
```

生成文件：源模型视图 `urdf/robot.urdf`、五个原始 DAE、导入视图 `urdf/robot_isaac.urdf`、
五个派生 OBJ 及 `model_parameters.json`。
JSON 保存源码和网格的 SHA256、关节限位、辨识参数、默认姿态及足端正运动学结果。
更新时只覆盖这些生成文件；资产配置和本说明不会被覆盖。

## 导入配置

```python
from robot_lab.assets.sysu219 import SYSU219_CFG, SYSU219_JOINT_NAMES, SYSU219_FOOT_NAMES

# 在后续场景配置中使用：
# robot = SYSU219_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

- 使用 `UrdfFileCfg` 加载 `robot_isaac.urdf` 并转换 USD。
- URDF 保留源码的关节轴、限位、质量、惯性、碰撞形状和网格缩放。
- 移除 ROS 控制、transmission 和 Gazebo 专用标签；根部不连接 world。
- 固定关节默认合并，四个 `*_foot_fixed` 增加 `dont_collapse="true"`，以保留足端 body 名称。
  这只是导入配置，仍需实际检查 USD 中的足端 body、碰撞体及固定 link 质量处理。
- 显式 IdealPD 执行器使用 ROS `robot_control.yaml` 的站立增益作为初始值；
  不使用旧训练配置的 165/200 Nm 电机模型。力矩和速度限制取自当前 xacro。
- `identified_dynamics/Ja` 显式映射为 armature，B 映射为 viscous friction，
  Fc 同时作为 static/dynamic friction 的初始值；源码没有单独辨识静摩擦值。
  PhysX 摩擦模型与辨识模型的低速行为仍需后续验证。
- 要求 Isaac Sim >= 5.0，避免 4.5 的无量纲摩擦系数与 Fc（Nm）混用。
  代码遇到旧版本会明确报错，不静默替代参数。
- 主动关节显式顺序为 FR、FL、RR、RL，每腿 hip、thigh、calf。
  后续动作和观测均应使用该列表并保留顺序，不能假定 USD 原始顺序相同。
- 初步禁用自碰撞，沿用库内常见模型配置；训练前需单独评估自碰撞与碰撞体重叠。

## 当前静态结果与边界

- 12 个主动关节，模型质量之和约 41.281929590 kg，trunk 约 23.262328 kg。
  这是源码模型质量之和，不是整机实测质量。
- 所有有惯性定义的 link 均通过质量正值、惯性正定和主惯量三角不等式检查。
- 所有网格路径有效，复制文件 SHA256 与 ROS 原文件一致。
- 默认角度直接使用 ROS `stand_pos`，全部在 URDF 硬限位内。
- 当前站立参数前腿 calf=-1.53 rad、后腿 calf=-1.30 rad。
  在机身水平时，前后足底高度相差约 0.032045 m，不能声称四足已同时平放地面。
  初始 base 高度按最低足底离地 0.01 m 计算，约 0.467347 m；
  后足初始离地 0.01 m，前足约 0.042045 m。
  此姿态来自当前 ROS 配置，后续训练前需决定是否采用对称姿态或调整机身初始姿态。
- 静态正运动学和文件检查不能验证 Isaac Sim 的导入结果、视觉朝向、
  接触稳定性、电机动态响应或模型与真机的一致性。

修改 URDF 或网格后，首次加载应重新生成 USD；仅更改网格时，Isaac Lab 的
转换缓存未必自动失效，应在需要更新时设置 `force_usd_conversion=True`。

## Isaac Sim 5.1 实际导入发现与修正

原 DAE 的 accessor/count 写成了浮点数总数，而非顶点个数；标准导入器会报 Unsupported Format。
`sysu219_meshes.py` 针对当前网格结构验证每个顶点和三角索引，导出 OBJ；保持坐标、三角面和
URDF 的 0.001 缩放，不进行简化。转换工具拒绝未支持的场景变换或外部纹理，避免静默丢失几何。
原始 DAE 和 ROS 描述保持不变。OBJ 材质沿用 URDF 定义，不导出 COLLADA 材质。

空的 base 和恒等 floating_base 固定连接会使 importer 将 trunk 合并改名为 base。
导入视图只去掉这个无惯性、无碰撞的包装；物理根节点为 trunk，坐标关系不变。

原脚 link 无惯性；保留它用于接触传感器时，PhysX 自动按球体体积生成约0.03351032 kg/脚。
导入视图显式设置每脚1e-6 kg（1 mg）及相应实心球惯量，避免自动增加约134 g整机质量。
这是固定足端的数值正则化，不是实测足质量；整机模型质量预期比源模型增加4e-6 kg。
所有原本有惯性定义的 link 的惯性保持不变。

首次加载检查使用独立注册 SYSU219 的脚本；全库自动扫描在当前 lab 环境会因其他机器人
的可选 cusrl 后端缺失而失败。检查脚本不安装这个无关后端，也不修改 conda 的安装路径记录。


路径约定：在工作区根目录设置 `export WS_ROOT="$PWD"`、`export ROBOT_LAB_ROOT="$WS_ROOT/rl_training/robot_lab"`。最新复现步骤见工作区 `docs/sysu219_rl.md`。
