# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import importlib

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
_selected_packages = os.environ.get("ROBOT_LAB_TASK_PACKAGES")
if _selected_packages:
    for _package in _selected_packages.split(","):
        _package = _package.strip()
        if not _package.startswith(__name__ + ".") or not all(part.isidentifier() for part in _package.split(".")):
            raise ValueError(f"Expected a package under {__name__}, got {_package!r}")
        importlib.import_module(_package)
else:
    import_packages(__name__, _BLACKLIST_PKGS)
