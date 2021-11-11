# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from torchcontrol.utils import to_tensor


class JointSpaceMoveTo(toco.PolicyModule):
    def __init__(
        self,
        joint_pos_current,
        joint_pos_desired,
        time_to_go: float,
        hz: float,
        *args,
        **kwargs,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            joint_pos_desired: Desired joint positions
            time_to_go: Duration of trajectory in seconds
            hz: Frequency of controller
            Kp: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise

        (Note: nA is the action dimension and N is the number of degrees of freedom)
        """
        super().__init__()

        # Parse inputs
        joint_pos_current = to_tensor(joint_pos_current)
        joint_pos_desired = to_tensor(joint_pos_desired)

        # Plan
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=joint_pos_current,
            goal=joint_pos_desired,
            time_to_go=time_to_go,
            hz=hz,
        )

        # Create joint executor
        self.plan_executor = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
            joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
            *args,
            **kwargs,
        )

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.plan_executor(state_dict)
        if self.plan_executor.is_terminated():
            self.set_terminated()
        return output


class CartesianTargetJointMoveTo(toco.PolicyModule):
    def __init__(
        self,
        joint_pos_current,
        ee_pos_desired,
        time_to_go: float,
        hz: float,
        robot_model: torch.nn.Module,
        ee_quat_desired=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            ee_pos_desired: Desired end-effector position (3D)
            ee_quat_desired: Desired end-effector orientation (None if current orientation is desired)
            time_to_go: Duration of trajectory in seconds
            hz: Frequency of controller
            Kp: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise

        (Note: nA is the action dimension and N is the number of degrees of freedom)
        """
        super().__init__()

        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = robot_model.forward_kinematics(
            joint_pos_current
        )

        # Plan
        ee_pos_desired = to_tensor(ee_pos_desired)
        if ee_quat_desired is None:
            ee_quat_desired = ee_quat_current
        else:
            ee_quat_desired = to_tensor(ee_quat_desired)

        ee_pose_desired = T.from_rot_xyz(
            rotation=R.from_quat(ee_quat_desired), translation=ee_pos_desired
        )

        waypoints = toco.planning.generate_cartesian_target_joint_min_jerk(
            joint_pos_start=joint_pos_current,
            ee_pose_goal=ee_pose_desired,
            time_to_go=time_to_go,
            hz=hz,
            robot_model=robot_model,
        )

        # Create joint plan executor
        self.plan_executor = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
            joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
            robot_model=robot_model,
            *args,
            **kwargs,
        )

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.plan_executor(state_dict)
        if self.plan_executor.is_terminated():
            self.set_terminated()
        return output


class CartesianSpaceMoveTo(toco.PolicyModule):
    """
    Plans and executes a trajectory to a desired end-effector pose
    """

    def __init__(
        self,
        joint_pos_current,
        ee_pos_desired,
        time_to_go: float,
        hz: float,
        robot_model: torch.nn.Module,
        ee_quat_desired=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            ee_pos_desired: Desired end-effector position (3D)
            Kp: P gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            Kd: D gain matrix of shape (6, 6) or shape (6,) representing a 6-by-6 diagonal matrix
            time_to_go: Duration of trajectory in seconds
            hz: Frequency of controller
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
            ee_quat_desired: Desired end-effector orientation (None if current orientation is desired)
        """
        super().__init__()

        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = robot_model.forward_kinematics(
            joint_pos_current
        )
        ee_pose_current = T.from_rot_xyz(
            rotation=R.from_quat(ee_quat_current), translation=ee_pos_current
        )

        # Plan
        ee_pos_desired = to_tensor(ee_pos_desired)
        if ee_quat_desired is None:
            ee_quat_desired = ee_quat_current
        else:
            ee_quat_desired = to_tensor(ee_quat_desired)
        ee_pose_desired = T.from_rot_xyz(
            rotation=R.from_quat(ee_quat_desired), translation=ee_pos_desired
        )

        waypoints = toco.planning.generate_cartesian_space_min_jerk(
            start=ee_pose_current,
            goal=ee_pose_desired,
            time_to_go=time_to_go,
            hz=hz,
        )

        # Create joint plan executor
        self.plan_executor = toco.policies.EndEffectorTrajectoryExecutor(
            ee_pose_trajectory=[waypoint["pose"] for waypoint in waypoints],
            ee_twist_trajectory=[waypoint["twist"] for waypoint in waypoints],
            robot_model=robot_model,
            *args,
            **kwargs,
        )

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.plan_executor(state_dict)
        if self.plan_executor.is_terminated():
            self.set_terminated()
        return output
