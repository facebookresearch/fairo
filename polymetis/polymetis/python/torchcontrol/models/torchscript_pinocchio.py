# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Tuple, Optional

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
            "../../build/torch_isolation/libtorchscript_pinocchio.so",
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

    Args:
        urdf_filename (str): path to the urdf file.
        ee_link_name (str, optional): name of the end-effector link. Defaults to None.
                                      Having a value of either None or "" would require
                                      you to specify link_name when using methods which
                                      require a link frame; otherwise, the end-effector
                                      link will be used by default.
    """

    def __init__(self, urdf_filename: str, ee_link_name: Optional[str] = None):
        super().__init__()
        self.model = torch.classes.torchscript_pinocchio.RobotModelPinocchio(
            urdf_filename, False
        )
        self.ee_link_name = None
        self.ee_link_idx = None
        self.set_ee_link(ee_link_name)

    def set_ee_link(self, ee_link_name: Optional[str] = None):
        """Sets the `ee_link_name`, `ee_link_idx` using pinocchio::ModelTpl::getBodyId."""
        # Set ee_link_name
        if ee_link_name == "":  # also treat an empty string as default value
            self.ee_link_name = None
        else:
            self.ee_link_name = ee_link_name

        # Set ee_link_idx
        if self.ee_link_name is not None:
            self.ee_link_idx = self.model.get_link_idx_from_name(self.ee_link_name)
        else:
            self.ee_link_idx = None

    def _get_link_idx_or_use_ee(self, link_name: str) -> int:
        """
        Get the link index from the link name, or use the end-effector link index
        if link_name is None.
        """
        if not link_name:
            frame_idx = self.ee_link_idx
            assert frame_idx, (
                "No end-effector link set during initialization, so link_name must "
                + "be either input as parameter or set as default using `set_ee_link`."
            )
        else:
            frame_idx = self.model.get_link_idx_from_name(link_name)
        return frame_idx

    def get_link_name_from_idx(self, link_idx: int):
        return self.model.get_link_name_from_idx(link_idx)

    def get_joint_angle_limits(self) -> torch.Tensor:
        return self.model.get_joint_angle_limits()

    def get_joint_velocity_limits(self) -> torch.Tensor:
        return self.model.get_joint_velocity_limits()

    def forward_kinematics(
        self,
        joint_positions: torch.Tensor,
        link_name: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes link position and orientation from a given joint position.

        Args:
            joint_positions: A given set of joint angles.
            link_name (str, optional): name of the link desired. Defaults to the
                                       end-effector link, if it was set during initialization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Link position, link orientation as quaternion
        """
        frame_idx = self._get_link_idx_or_use_ee(link_name)
        pos, quat = self.model.forward_kinematics(joint_positions, frame_idx)
        return pos.to(joint_positions), quat.to(joint_positions)

    def compute_jacobian(
        self, joint_positions: torch.Tensor, link_name: str = ""
    ) -> torch.Tensor:
        """Computes the Jacobian relative to the link frame.

        Args:
            joint_positions: A given set of joint angles.
            link_name (str, optional): name of the link desired. Defaults to the
                                       end-effector link, if it was set during initialization.

        Returns:
            torch.Tensor, torch.Tensor: The Jacobian relative to the link frame.
        """
        frame_idx = self._get_link_idx_or_use_ee(link_name)
        return self.model.compute_jacobian(joint_positions, frame_idx).to(
            joint_positions
        )

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
        link_pos: torch.Tensor,
        link_quat: torch.Tensor,
        link_name: str = "",
        rest_pose: torch.Tensor = None,
        eps: float = 1e-4,
        max_iters: int = 1000,
        dt: float = 0.1,
        damping: float = 1e-6,
    ) -> torch.Tensor:
        """Computes joint positions that achieve a given end-effector pose.
        Uses CLIK algorithm from
        https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html

        Args:
            link_pos (torch.Tensor): desired link position
            link_quat (torch.Tensor): desired link orientation
            link_name (str, optional): name of the link desired. Defaults to the
                                       end-effector link, if it was set during initialization.
            rest_pose (torch.Tensor): (optional) initial solution for IK
            eps (float): (optional) maximum allowed error
            max_iters (int): (optional) maximum number of iterations
            dt (float): (optional) time step for integration
            damping: (optional) damping factor for numerical stability

        Returns:
            torch.Tensor: joint positions
        """
        frame_idx = self._get_link_idx_or_use_ee(link_name)
        if rest_pose is None:
            rest_pose = torch.zeros(self.model.get_joint_angle_limits()[0].numel())
        return self.model.inverse_kinematics(
            link_pos.squeeze(),
            link_quat.squeeze(),
            frame_idx,
            rest_pose.squeeze(),
            eps,
            max_iters,
            dt,
            damping,
        ).to(link_pos)
