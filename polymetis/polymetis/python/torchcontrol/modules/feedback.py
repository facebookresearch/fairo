# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from torchcontrol.utils.tensor_utils import to_tensor, diagonalize_gain


class LinearFeedback(toco.ControlModule):
    """
    Linear feedback control: :math:`u = Kx`

    nA is the action dimension and nS is the state dimension

    Module parameters:
        - K: Gain matrix of shape (nA, nS)
    """

    def __init__(self, K: torch.Tensor):
        """
        Args:
            K: Gain matrix of shape (nA, nS) or shape (nS,) representing a nS-by-nS diagonal matrix (if nA=nS)
        """
        super().__init__()
        self.K = torch.nn.Parameter(diagonalize_gain(to_tensor(K)))

    def forward(self, x_current: torch.Tensor, x_desired: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_current: Current state of shape (nS,)
            x_desired: Desired state of shape (nS,)

        Returns:
            Output action of shape (nA,)
        """
        return self.K @ (x_desired - x_current)


class JointSpacePD(toco.ControlModule):
    """
    PD feedback control in joint space

    nA is the action dimension and N is the number of degrees of freedom

    Module parameters:
        - Kp: P gain matrix of shape (nA, N)
        - Kd: D gain matrix of shape (nA, N)
    """

    def __init__(self, Kp: torch.Tensor, Kd: torch.Tensor):
        """
        Args:
            Kp: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
        """
        super().__init__()

        Kp = diagonalize_gain(to_tensor(Kp))
        Kd = diagonalize_gain(to_tensor(Kd))
        assert Kp.shape == Kd.shape

        self.Kp = torch.nn.Parameter(Kp)
        self.Kd = torch.nn.Parameter(Kd)

    def forward(
        self,
        joint_pos_current: torch.Tensor,
        joint_vel_current: torch.Tensor,
        joint_pos_desired: torch.Tensor,
        joint_vel_desired: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            joint_pos_current: Current joint position of shape (N,)
            joint_vel_current: Current joint velocity of shape (N,)
            joint_pos_desired: Desired joint position of shape (N,)
            joint_vel_desired: Desired joint velocity of shape (N,)

        Returns:
            Output action of shape (nA,)
        """
        return self.Kp @ (joint_pos_desired - joint_pos_current) + self.Kd @ (
            joint_vel_desired - joint_vel_current
        )


class HybridJointSpacePD(toco.ControlModule):
    """
    PD feedback control in joint space
    Uses both constant joint gains and adaptive operational space gains

    nA is the action dimension and N is the number of degrees of freedom

    Module parameters:
        - Kq: P gain matrix of shape (nA, N)
        - Kqd: D gain matrix of shape (nA, N)
        - Kx: P gain matrix of shape (6, 6)
        - Kxd: D gain matrix of shape (6, 6)
    """

    def __init__(
        self, Kq: torch.Tensor, Kqd: torch.Tensor, Kx: torch.Tensor, Kxd: torch.Tensor
    ):
        """
        Args:
            Kq: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kqd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kx: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            Kxd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
        """
        super().__init__()

        Kq = diagonalize_gain(to_tensor(Kq))
        Kqd = diagonalize_gain(to_tensor(Kqd))
        assert Kq.shape == Kqd.shape
        Kx = diagonalize_gain(to_tensor(Kx))
        Kxd = diagonalize_gain(to_tensor(Kxd))
        assert Kx.shape == torch.Size([6, 6])
        assert Kxd.shape == torch.Size([6, 6])

        self.Kq = torch.nn.Parameter(Kq)
        self.Kqd = torch.nn.Parameter(Kqd)
        self.Kx = torch.nn.Parameter(Kx)
        self.Kxd = torch.nn.Parameter(Kxd)

    def forward(
        self,
        joint_pos_current: torch.Tensor,
        joint_vel_current: torch.Tensor,
        joint_pos_desired: torch.Tensor,
        joint_vel_desired: torch.Tensor,
        jacobian: torch.Tensor,
    ) -> torch.Tensor:
        """
        nA is the action dimension and N is the number of degrees of freedom

        Args:
            joint_pos_current: Current joint position of shape (N,)
            joint_vel_current: Current joint velocity of shape (N,)
            joint_pos_desired: Desired joint position of shape (N,)
            joint_vel_desired: Desired joint velocity of shape (N,)
            jacobian: End-effector jacobian of shape (N, 6)

        Returns:
            Output action of shape (nA,)
        """
        Kp = jacobian.T @ self.Kx @ jacobian + self.Kq
        Kd = jacobian.T @ self.Kxd @ jacobian + self.Kqd
        return Kp @ (joint_pos_desired - joint_pos_current) + Kd @ (
            joint_vel_desired - joint_vel_current
        )


class CartesianSpacePDFast(toco.ControlModule):
    """
    PD feedback control in SE3 pose space

    Module parameters:
        - Kp: P gain matrix of shape (6, 6)
        - Kd: D gain matrix of shape (6, 6)
    """

    def __init__(self, Kp: torch.Tensor, Kd: torch.Tensor):
        """
        Args:
            Kp: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            Kd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
        """
        super().__init__()

        Kp = diagonalize_gain(to_tensor(Kp))
        Kd = diagonalize_gain(to_tensor(Kd))
        assert Kp.shape == torch.Size([6, 6])
        assert Kd.shape == torch.Size([6, 6])

        self.Kp = torch.nn.Parameter(Kp)
        self.Kd = torch.nn.Parameter(Kd)

    def forward(
        self,
        pos_current: torch.Tensor,
        quat_current: torch.Tensor,
        twist_current: torch.Tensor,
        pos_desired: torch.Tensor,
        quat_desired: torch.Tensor,
        twist_desired: torch.Tensor,
    ):
        """
        Args:
            pos_current: Current position of shape (3,)
            quat_current: Current quaternion of shape (4,)
            twist_current: Current twist of shape (6,)
            pos_desired: Desired position of shape (3,)
            quat_desired: Desired quaternion of shape (4,)
            twist_desired: Desired twist of shape (6,)

        Returns:
            Output wrench of shape (6,)
        """
        # Compute pose error (from https://frankaemika.github.io/libfranka/cartesian_impedance_control_8cpp-example.html)
        pos_err = pos_desired - pos_current

        quat_curr_inv = R.functional.invert_quaternion(quat_current)
        quat_err = R.functional.quaternion_multiply(quat_curr_inv, quat_desired)
        quat_err_n = R.functional.normalize_quaternion(quat_err)
        ori_err = R.functional.quat2matrix(quat_current) @ quat_err_n[0:3]

        pose_err = torch.cat([pos_err, ori_err])

        # Compute twist error
        twist_err = twist_desired - twist_current

        # Compute feedback
        return self.Kp @ pose_err + self.Kd @ twist_err


class CartesianSpacePD(CartesianSpacePDFast):
    """
    PD feedback control in SE3 pose space

    Logically identical as CartesianSpacePDFast but with torchcontrol.transform.TransformationObj inputs.
    Slower implementation due to object creation and member access.
    """

    def forward(
        self,
        pose_current: T.TransformationObj,
        twist_current: torch.Tensor,
        pose_desired: T.TransformationObj,
        twist_desired: torch.Tensor,
    ):
        """
        Args:
            pose_current: Current ee pose
            twist_current: Current ee twist of shape (6,)
            pose_desired: Desired ee pose
            twist_desired: Desired ee twist of shape (6,)

        Returns:
            Output wrench of shape (6,)
        """
        pos_current = pose_current.translation()
        quat_current = pose_current.rotation().as_quat()
        pos_desired = pose_desired.translation()
        quat_desired = pose_desired.rotation().as_quat()

        return super().forward(
            pos_current,
            quat_current,
            twist_current,
            pos_desired,
            quat_desired,
            twist_desired,
        )
