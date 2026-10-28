# SYSU219 ROS DR 7 cm 最终模型归档

来源：`2026-10-27_17-25-46_ros_dr_7cm/model_2999.pt`。

- `model_2999.pt`：唯一归档的最终checkpoint；不包含中间checkpoint和TensorBoard日志。
- `params/env.yaml`、`params/agent.yaml`：本次训练保存的原始配置，保留历史绝对路径。
- `native_ort_report.json`：预编译 ONNX Runtime 的256组数值一致性验证结果。
- `manifest.json`：来源与文件SHA256。

使用工作区 `docs/sysu219_rl.md` 中的仓库相对路径回放或重新导出。
部署模型为 ROS 控制器 `model/policy_trot_ros_dr_7cm.onnx`，完整部署契约记录在同名JSON中。
当前 ROS 接管融合属于后续部署修改，不是此checkpoint中训练出的附加行为。
