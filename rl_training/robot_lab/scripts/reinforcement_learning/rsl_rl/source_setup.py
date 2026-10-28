"""Resolve local source checkouts before importing Isaac Lab (stdlib only).

No installation, package emulation, or modification of conda's editable records.
"""

import argparse
import os
from pathlib import Path
import sys


SYSU219_TASKS = {
    "RobotLab-Isaac-Trot-InPlace-SYSU219-v0",
    "RobotLab-Isaac-Trot-InPlace-SYSU219-Play-v0",
    "RobotLab-Isaac-Trot-InPlace-SYSU219-DR-v0",
    "RobotLab-Isaac-Trot-InPlace-SYSU219-DR-Play-v0",
}
SYSU219_PACKAGE = "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.sysu219"


def setup_sources(argv=None, robot_lab_root=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--task")
    options, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = Path(robot_lab_root) if robot_lab_root else Path(__file__).resolve().parents[3]
    robot_source = root / "source/robot_lab"
    if not (robot_source / "robot_lab/__init__.py").is_file():
        raise FileNotFoundError(f"Cannot find robot_lab source under {root}")
    explicit = options.isaaclab_root or os.environ.get("ISAACLAB_ROOT")
    candidates = [Path(explicit)] if explicit else [
        root.parent / "IsaacLab",
        root.parent / "219_ws/rl_training/IsaacLab",
    ]
    selected = None
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "source/isaaclab/isaaclab/app/__init__.py").is_file():
            selected = candidate
            break
    if explicit and selected is None:
        raise FileNotFoundError(f"Invalid Isaac Lab checkout: {explicit}")
    paths = [robot_source]
    if selected:
        paths.extend(sorted(path for path in (selected / "source").iterdir()
                            if path.is_dir() and (path / path.name / "__init__.py").is_file()))
    for path in reversed(paths):
        value = str(path.resolve())
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    if options.task and options.task.split(":")[-1] in SYSU219_TASKS:
        os.environ["ROBOT_LAB_TASK_PACKAGES"] = SYSU219_PACKAGE
    print(f"[INFO][robot_lab]: Isaac Lab source: {selected or 'installed package'}", flush=True)
    print(f"[INFO][robot_lab]: Task discovery: {os.environ.get('ROBOT_LAB_TASK_PACKAGES', 'all tasks')}", flush=True)
    return selected
