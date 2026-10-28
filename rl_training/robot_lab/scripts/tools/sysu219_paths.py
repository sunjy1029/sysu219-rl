"""Locate the enclosing ROS workspace without machine-specific home paths."""

from pathlib import Path


def workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "src/quadruped_ros2_control-humble").is_dir() and (parent / "rl_training").is_dir():
            return parent
    raise FileNotFoundError("Cannot locate enclosing 219_ws; keep robot_lab under rl_training/")


def isaaclab_source():
    return workspace_root() / "rl_training/IsaacLab/source/isaaclab/isaaclab"
