# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Tuple

import torch
from polymetis.utils.data_dir import PKG_ROOT_DIR

try:
    torch.classes.load_library(
        f"{os.environ['CONDA_PREFIX']}/lib/libtorchscript_pinocchio.so"
    )
except OSError:
    lib_path = os.path.abspath(
        os.path.join(
            PKG_ROOT_DIR,
            "../../build/libtorchscript_pinocchio.so",
        )
    )
    print(
        f"Warning: Failed to load 'libtorchscript_pinocchio.so' from CONDA_PREFIX, loading from default build directory instead: '{lib_path}'"
    )
    torch.classes.load_library(lib_path)


class RobotModelPinocchio(torch.nn.Module):
    """
    A robot model able to compute kinematics & dynamics of a robot given an urdf.

    Implemented as a ``torch.nn.Module`` wrapped around a C++ custom class that leverages
    `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`_ -
    a C++ rigid body dynamics library.
    """

    def __init__(self, urdf_filename: str, ee_joint_name: str):
        super().__init__()
        self.model = torch.classes.torchscript_pinocchio.RobotModelPinocchio(
            urdf_filename, ee_joint_name
        )

    def get_joint_angle_limits(self) -> torch.Tensor:
        return self.model.get_joint_angle_limits()

    def get_joint_velocity_limits(self) -> torch.Tensor:
        return self.model.get_joint_velocity_limits()

    def forward_kinematics(
        self, joint_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes end-effector position and orientation from a given joint position.

        Args:
            joint_positions: A given set of joint angles.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: End-effector position, end-effector orientation as quaternion
        """
        pos, quat = self.model.forward_kinematics(joint_positions)
        return pos.to(joint_positions), quat.to(joint_positions)

    def compute_jacobian(self, joint_positions: torch.Tensor) -> torch.Tensor:
        return self.model.compute_jacobian(joint_positions).to(joint_positions)

    def inverse_dynamics(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_accelerations: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the desired torques to achieve a certain joint acceleration from
        given joint positions and velocities.

        Returns:
            torch.Tensor: desired torques
        """
        return self.model.inverse_dynamics(
            joint_positions, joint_velocities, joint_accelerations
        ).to(joint_positions)

    def inverse_kinematics(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        eps: float = 1e-4,
        max_iters: int = 1000,
        dt: float = 0.1,
        damping: float = 1e-6,
    ) -> torch.Tensor:
        """Computes joint positions that achieve a given end-effector pose.

        Returns:
            torch.Tensor: joint positions
        """
        return self.model.inverse_kinematics(
            ee_pos.squeeze(), ee_quat.squeeze(), eps, max_iters, dt, damping
        ).to(ee_pos)
