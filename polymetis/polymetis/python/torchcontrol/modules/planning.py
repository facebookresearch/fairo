#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T


def min_jerk_spaces(
    N: int, T: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a 1-dim minimum jerk trajectory from 0 to 1 in N steps & T seconds.
    Assumes zero velocity & acceleration at start & goal.
    The resulting trajectories can be scaled for different start & goals.

    Args:
        N: Number of sample points

    Returns:
        Position trajectory of shape (N,)
        Velocity trajectory of shape (N,)
        Acceleration trajectory of shape (N,)
    """
    assert N > 1, "Number of planning steps must be larger than 1."

    t_traj = torch.linspace(0, 1, N)
    p_traj = 10 * t_traj ** 3 - 15 * t_traj ** 4 + 6 * t_traj ** 5
    pd_traj = (30 * t_traj ** 2 - 60 * t_traj ** 3 + 30 * t_traj ** 4) / T
    pdd_traj = (60 * t_traj - 180 * t_traj ** 2 + 120 * t_traj ** 3) / T ** 2

    return p_traj, pd_traj, pdd_traj


class JointSpaceMinJerkPlanner(toco.ControlModule):
    """
    Primitive joint space minimum jerk trajectory planner.
    Assumes zero velocity & acceleration at start & goal.

    N is the number of degrees of freedom
    """

    def __init__(
        self, start: torch.Tensor, goal: torch.Tensor, steps: int, time_to_go: float
    ):
        """Initializes planner object and plans the trajectory

        Args:
            start: Start joint position of shape (N,)
            goal: Goal joint position of shape (N,)
            steps: Number of steps
        """
        super().__init__()

        p_traj, pd_traj, pdd_traj = min_jerk_spaces(steps, time_to_go)

        # Plan
        D = goal - start
        self.q_traj = start[None, :] + D[None, :] * p_traj[:, None]
        self.qd_traj = D[None, :] * pd_traj[:, None]
        self.qdd_traj = D[None, :] * pdd_traj[:, None]

    def forward(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries the planned trajectory for the desired states at the input step

        Args:
            step: Step index

        Returns:
            Desired position of shape (N,)
            Desired velocity of shape (N,)
            Desired acceleration of shape (N,)
        """
        return (
            self.q_traj[step, :],
            self.qd_traj[step, :],
            self.qdd_traj[step, :],
        )


class CartesianSpaceMinJerkPlanner(toco.ControlModule):
    """
    Cartesian space minimum jerk trajectory planner.
    Assumes zero velocity & acceleration at start & goal.
    """

    def __init__(
        self,
        start: T.TransformationObj,
        goal: T.TransformationObj,
        steps: int,
        time_to_go: float,
    ):
        """Initializes planner object and plans the trajectory

        Args:
            start: Start pose
            goal: Goal pose
            steps: Number of steps
        """
        super().__init__()

        p_traj, pd_traj, pdd_traj = min_jerk_spaces(steps, time_to_go)
        dt = time_to_go / steps

        # Plan translation
        x_start = start.translation()
        x_goal = goal.translation()

        D = x_goal - x_start
        x_traj = x_start[None, :] + D[None, :] * p_traj[:, None]
        xd_traj = D[None, :] * pd_traj[:, None]
        xdd_traj = D[None, :] * pdd_traj[:, None]

        # Plan rotation
        r_start = start.rotation()
        r_goal = goal.rotation()
        r_delta = r_goal * r_start.inv()
        rv_delta = r_delta.as_rotvec()

        r_traj = torch.empty([steps, 4])
        for i in range(steps):
            r = R.from_rotvec(rv_delta * p_traj[i]) * r_start
            r_traj[i, :] = r.as_quat()
        rd_traj = rv_delta[None, :] * pd_traj[:, None]
        rdd_traj = rv_delta[None, :] * pdd_traj[:, None]

        # Combine results
        self.ee_pose_traj = torch.cat([x_traj, r_traj], dim=-1)
        self.ee_twist_traj = torch.cat([xd_traj, rd_traj], dim=-1)
        self.ee_accel_traj = torch.cat([xdd_traj, rdd_traj], dim=-1)

    def forward(
        self, step: int
    ) -> Tuple[T.TransformationObj, torch.Tensor, torch.Tensor]:
        """Queries the planned trajectory for the desired states at the input step

        Args:
            step: Step index

        Returns:
            Desired pose
            Desired twist of shape (6,)
            Desired accel of shape (6,)
        """
        ee_pose = T.from_rot_xyz(
            rotation=R.from_quat(self.ee_pose_traj[step, 3:]),
            translation=self.ee_pose_traj[step, :3],
        )
        return (
            ee_pose,
            self.ee_twist_traj[step, :],
            self.ee_accel_traj[step, :],
        )


class PositionMinJerkPlanner(JointSpaceMinJerkPlanner):
    """
    Minimum jerk trajectory planner through XYZ space.
    Assumes zero velocity & acceleration at start & goal.
    Basically a joint space planner with 3 joints.
    """

    def __init__(self, start, goal, steps: int, time_to_go: float):
        """Initializes planner object and plans the trajectory

        Args:
            start: start joint position of shape (3,)
            goal: goal joint position of shape (3,)
            steps: Number of steps
        """
        assert start.shape == torch.Size([3])
        assert goal.shape == torch.Size([3])
        super().__init__(start, goal, steps, time_to_go)


class CartesianSpaceMinJerkJointPlanner(toco.ControlModule):
    """
    Cartesian space minimum jerk trajectory planner, but outputs plan in joint space.
    Assumes zero velocity & acceleration at start & goal.
    """

    def __init__(
        self,
        joint_pos_start: torch.Tensor,
        ee_pose_goal: T.TransformationObj,
        steps: int,
        time_to_go: float,
        robot_model: torch.nn.Module,
    ):
        """Initializes planner object and plans the trajectory

        Args:
            start: Start pose
            goal: Goal pose
            steps: Number of steps
            robot_model: A valid robot model module from torchcontrol.models
        """
        super().__init__()

        dt = time_to_go / steps

        # Compute start pose
        ee_pos_start, ee_quat_start = robot_model.forward_kinematics(joint_pos_start)
        ee_pose_start = T.from_rot_xyz(
            rotation=R.from_quat(ee_quat_start), translation=ee_pos_start
        )
        cartesian_planner = CartesianSpaceMinJerkPlanner(
            ee_pose_start, ee_pose_goal, steps, time_to_go
        )

        # Extract plan & convert to joint space
        self.q_traj = torch.zeros(steps, joint_pos_start.shape[0])
        self.qd_traj = torch.zeros(steps, joint_pos_start.shape[0])
        self.qdd_traj = torch.zeros(steps, joint_pos_start.shape[0])

        self.q_traj[0, :] = joint_pos_start
        for i in range(0, steps - 1):
            # Get current joint state & jacobian
            joint_pos_current = self.q_traj[i, :]
            jacobian = robot_model.compute_jacobian(joint_pos_current)
            jacobian_pinv = torch.pinverse(jacobian)

            # Query Cartesian plan for next step & compute diff
            ee_pose_desired, ee_twist_desired, ee_accel_desired = cartesian_planner(
                i + 1
            )

            # Convert next step to joint plan
            self.qdd_traj[i + 1, :] = jacobian_pinv @ ee_accel_desired
            self.qd_traj[i + 1, :] = jacobian_pinv @ ee_twist_desired
            q_delta = self.qd_traj[i + 1, :] * dt
            self.q_traj[i + 1, :] = joint_pos_current + q_delta

            # Null space correction (norm of correction clamped to norm of current action)
            null_space_proj = (
                torch.eye(joint_pos_start.shape[0]) - jacobian_pinv @ jacobian
            )
            q_null_err = -null_space_proj @ self.q_traj[i + 1, :]
            q_null_err_norm = q_null_err.norm()
            q_null_err_clamped = (
                q_null_err / q_null_err_norm * min(q_null_err_norm, q_delta.norm())
            )
            self.q_traj[i + 1, :] = self.q_traj[i + 1, :] + q_null_err_clamped

    def forward(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries the planned trajectory for the desired states at the input step

        Args:
            step: Step index

        Returns:
            Desired position of shape (N,)
            Desired velocity of shape (N,)
            Desired acceleration of shape (N,)
        """
        return (
            self.q_traj[step, :],
            self.qd_traj[step, :],
            self.qdd_traj[step, :],
        )
