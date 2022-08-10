# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import torch

import torchcontrol as toco
from torchcontrol.transform import Transformation as T
from torchcontrol.utils.tensor_utils import to_tensor, stack_trajectory


class JointTrajectoryExecutor(toco.PolicyModule):
    def __init__(
        self,
        joint_pos_trajectory: List[torch.Tensor],
        joint_vel_trajectory: List[torch.Tensor],
        Kq,
        Kqd,
        Kx,
        Kxd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Executes a joint trajectory by using a joint PD controller to stabilize around waypoints in the trajectory.

        Args:
            joint_pos_trajectory: Joint position trajectory as list of tensors
            joint_vel_trajectory: Joint position trajectory as list of tensors
            Kq: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kqd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kx: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            Kxd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise

        (Note: nA is the action dimension and N is the number of degrees of freedom)
        """
        super().__init__()

        self.joint_pos_trajectory = to_tensor(stack_trajectory(joint_pos_trajectory))
        self.joint_vel_trajectory = to_tensor(stack_trajectory(joint_vel_trajectory))

        self.N = self.joint_pos_trajectory.shape[0]
        assert self.joint_pos_trajectory.shape == self.joint_vel_trajectory.shape

        # Control modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Parse current state
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Query plan for desired state
        joint_pos_desired = self.joint_pos_trajectory[self.i, :]
        joint_vel_desired = self.joint_vel_trajectory[self.i, :]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            joint_pos_desired,
            joint_vel_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        # Increment & termination
        self.i += 1
        if self.i == self.N:
            self.set_terminated()

        return {"joint_torques": torque_out}


class EndEffectorTrajectoryExecutor(toco.PolicyModule):
    def __init__(
        self,
        ee_pose_trajectory: List[T.TransformationObj],
        ee_twist_trajectory: List[torch.Tensor],
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity: bool = True,
    ):
        """
        Executes a EE pose trajectory by using a Cartesian PD controller to stabilize around waypoints in the trajectory.

        Args:
            ee_pose_trajectory: End effector pose trajectory as a list of TransformationObj
            ee_twist_trajectory: End effector twist (velocity + angular velocity) trajectory as list of tensors
            Kp: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            Kd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        self.ee_pos_trajectory = to_tensor(
            stack_trajectory([pose.translation() for pose in ee_pose_trajectory])
        )
        self.ee_quat_trajectory = to_tensor(
            stack_trajectory([pose.rotation().as_quat() for pose in ee_pose_trajectory])
        )
        self.ee_twist_trajectory = to_tensor(stack_trajectory(ee_twist_trajectory))

        self.N = self.ee_pos_trajectory.shape[0]
        assert self.ee_pos_trajectory.shape == torch.Size([self.N, 3])
        assert self.ee_quat_trajectory.shape == torch.Size([self.N, 4])
        assert self.ee_twist_trajectory.shape == torch.Size([self.N, 6])

        # Control
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kp, Kd)

        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Parse current state
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_twist_current = jacobian @ joint_vel_current

        # Query plan for desired state
        ee_pos_desired = self.ee_pos_trajectory[self.i, :]
        ee_quat_desired = self.ee_quat_trajectory[self.i, :]
        ee_twist_desired = self.ee_twist_trajectory[self.i, :]

        # Control logic
        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            ee_pos_desired,
            ee_quat_desired,
            ee_twist_desired,
        )
        torque_feedback = jacobian.T @ wrench_feedback

        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis

        torque_out = torque_feedback + torque_feedforward

        # Increment & termination
        self.i += 1
        if self.i == self.N:
            self.set_terminated()

        return {"joint_torques": torque_out}


class iLQR(toco.PolicyModule):
    """Executes a time-varying linear feedback policy (output of iLQR optimization)"""

    i: int

    def __init__(self, Kxs, x_desireds, u_ffs):
        """
        Definitions:
            state_dim = number of state dimensions
            num_dofs = number of action dimensions

        Args:
            Kxs: [ time_horizon x num_dofs x state_dim ] series of gain matrices
            x_desireds: [ time_horizon x state_dim ] series of desired state
            u_ffs: [ time_horizon x num_dofs ] series of desired torques
        """
        super().__init__()

        # Dimension checks
        assert Kxs.ndim == 3
        assert x_desireds.ndim == 2
        assert u_ffs.ndim == 2
        assert Kxs.shape[0] == x_desireds.shape[0]
        assert Kxs.shape[0] == u_ffs.shape[0]
        assert Kxs.shape[2] == x_desireds.shape[1]
        assert Kxs.shape[1] == u_ffs.shape[1]

        # Initialize params & modules
        self.Kxs = torch.nn.Parameter(to_tensor(Kxs))
        self.x_desireds = torch.nn.Parameter(to_tensor(x_desireds))
        self.u_ffs = torch.nn.Parameter(to_tensor(u_ffs))

        self.feedback = toco.modules.LinearFeedback(torch.zeros_like(self.Kxs[0, :, :]))

        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: Robot state dictionary

        Returns:
            A dictionary containing the controller output
        """
        # Extract state vector
        x = torch.cat(
            [state_dict["joint_positions"], state_dict["joint_velocities"]], dim=-1
        )

        # Select linear feedback gains & reference
        self.feedback.update({"K": self.Kxs[self.i, :, :]})

        x_desired = self.x_desireds[self.i, :]
        u_ff = self.u_ffs[self.i, :]

        # Increment & termination
        self.i += 1
        if self.i == self.Kxs.shape[0]:
            self.set_terminated()

        u_output = self.feedback(x, x_desired) + u_ff

        return {"joint_torques": u_output}
