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


class CartesianSpacePD(toco.ControlModule):
    """
    PD feedback control in Cartesian space

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
        ee_pose_current: T.TransformationObj,
        ee_twist_current: torch.Tensor,
        ee_pose_desired: T.TransformationObj,
        ee_twist_desired: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ee_pose_current: Current ee pose
            ee_twist_current: Current ee twist of shape (6,)
            ee_pose_desired: Desired ee pose
            ee_twist_desired: Desired ee twist of shape (6,)

        Returns:
            Output wrench of shape (6,)
        """
        # Compute operational space position & velocity
        ee_pose_err = (ee_pose_current.inv() * ee_pose_desired).as_twist()
        ee_twist_err = ee_twist_desired - ee_twist_current

        # Feedback law
        output = self.Kp @ ee_pose_err + self.Kd @ ee_twist_err

        # Return forces
        return output


class OperationalSpacePD(toco.ControlModule):
    """
    PD feedback control in operational space.
    Errors are computed in Cartesian space, then projected back into joint space to compute joint torques.

    nA is the action dimension and N is the number of degrees of freedom

    Module parameters:
        - Kp: P gain matrix of shape (nA, N)
        - Kd: D gain matrix of shape (nA, N)
    """

    def __init__(
        self, Kp: torch.Tensor, Kd: torch.Tensor, robot_model: torch.nn.Module
    ):
        """
        Args:
            Kp: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            robot_model: A valid robot model module from torchcontrol.models
        """
        super().__init__()

        Kp = diagonalize_gain(to_tensor(Kp))
        Kd = diagonalize_gain(to_tensor(Kd))
        assert Kp.shape == torch.Size([6, 6])
        assert Kd.shape == torch.Size([6, 6])

        self.Kp = torch.nn.Parameter(Kp)
        self.Kd = torch.nn.Parameter(Kd)

        self.robot_model = robot_model

    def forward(
        self,
        joint_pos_current: torch.Tensor,
        joint_vel_current: torch.Tensor,
        ee_pose_desired: T.TransformationObj,
        ee_twist_desired: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            joint_pos_current: Current joint position of shape (N,)
            joint_vel_current: Current joint velocity of shape (N,)
            ee_pose_desired: Desired ee pose
            ee_twist_desired: Desired ee twist of shape (6,)

        Returns:
            Output torques of shape (nA,)
        """
        # Get current ee state & jacobian pinv
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)

        ee_pose_current = T.TransformationObj(
            rotation=T.RotationObj(ee_quat_current),
            translation=ee_pos_current,
        )
        ee_twist_current = jacobian @ joint_vel_current

        ee_pose_err = (ee_pose_current.inv() * ee_pose_desired).as_twist()
        ee_twist_err = ee_twist_desired - ee_twist_current

        output = jacobian.T @ (self.Kp @ ee_pose_err + self.Kd @ ee_twist_err)

        return output


class OperationalSpacePositionPD(toco.ControlModule):
    """
    PD feedback control in operational space, but with position only.
    Feedback forces are computed in Cartesian space, then projected back into joint space.

    nA is the action dimension and N is the number of degrees of freedom

    Module parameters:
        - op_space_pd.Kp: P gain matrix of shape (6, 6)
        - op_space_pd.Kd: D gain matrix of shape (6, 6)
    """

    def __init__(
        self,
        Kp: torch.Tensor,
        Kd: torch.Tensor,
        joint_pos_current: torch.Tensor,
        robot_model: torch.nn.Module,
    ):
        """
        Args:
            Kp: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix (if nA=6)
            Kd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix (if nA=6)
            joint_pos_current: Current joint position of shape (N,)
            robot_model: A valid robot model module from torchcontrol.models
        """
        super().__init__()
        self.op_space_pd = OperationalSpacePD(Kp, Kd, robot_model)
        self.robot_model = self.op_space_pd.robot_model
        _, self.ee_quat_desired = self.robot_model.forward_kinematics(joint_pos_current)

    def forward(
        self,
        joint_pos_current: torch.Tensor,
        joint_vel_current: torch.Tensor,
        ee_pos_desired: torch.Tensor,
        ee_vel_desired: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            joint_pos_current: Current joint position of shape (N,)
            joint_vel_current: Current joint velocity of shape (N,)
            ee_pos_desired: Desired ee position of shape (3,)
            ee_vel_desired: Desired ee velocity of shape (3,)

        Returns:
            Output action of shape (nA,)
        """
        ee_pose_desired = T.from_rot_xyz(
            rotation=R.from_quat(self.ee_quat_desired), translation=ee_pos_desired
        )
        ee_twist_desired = torch.cat([ee_vel_desired, torch.zeros(3)])
        return self.op_space_pd(
            joint_pos_current, joint_vel_current, ee_pose_desired, ee_twist_desired
        )
