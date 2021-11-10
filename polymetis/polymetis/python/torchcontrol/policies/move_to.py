# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from torchcontrol.utils import to_tensor


class JointPlanExecutor(toco.PolicyModule):
    """
    Plans and executes a trajectory to a desired joint position
    """

    def __init__(
        self,
        plan: torch.nn.Module,
        steps: int,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity: bool = True,
    ):
        """
        Args:
            plan: A plan module from torchcontrol.modules.planning
            steps: Length of plan in number of steps
            Kp: P gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            Kd: D gain matrix of shape (nA, N) or shape (N,) representing a N-by-N diagonal matrix (if nA=N)
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise

        (Note: nA is the action dimension and N is the number of degrees of freedom)
        """
        super().__init__()

        self.plan = plan
        self.N = steps

        # Control modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Parse current state
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Query plan for desired state
        joint_pos_desired, joint_vel_desired, _ = self.plan(self.i)

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            joint_pos_desired,
            joint_vel_desired,
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


class JointSpaceMoveTo(JointPlanExecutor):
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
        # Parse inputs
        joint_pos_current = to_tensor(joint_pos_current)
        joint_pos_desired = to_tensor(joint_pos_desired)

        # Plan
        steps = int(time_to_go * hz)
        plan = toco.modules.planning.JointSpaceMinJerkPlanner(
            start=joint_pos_current,
            goal=joint_pos_desired,
            steps=steps,
            time_to_go=time_to_go,
        )

        # Initialize joint plan executor
        super().__init__(plan, steps, *args, **kwargs)


class CartesianTargetJointMoveTo(JointPlanExecutor):
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
        # Parse inputs
        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = robot_model.forward_kinematics(
            joint_pos_current
        )

        ee_pos_desired = to_tensor(ee_pos_desired)
        if ee_quat_desired is None:
            ee_quat_desired = ee_quat_current
        else:
            ee_quat_desired = to_tensor(ee_quat_desired)

        ee_pose_desired = T.from_rot_xyz(
            rotation=R.from_quat(ee_quat_desired), translation=ee_pos_desired
        )

        # Plan
        steps = int(time_to_go * hz)
        plan = toco.modules.planning.CartesianSpaceMinJerkJointPlanner(
            joint_pos_start=joint_pos_current,
            ee_pose_goal=ee_pose_desired,
            steps=steps,
            time_to_go=time_to_go,
            robot_model=robot_model,
        )

        # Initialize joint plan executor
        super().__init__(plan, steps, robot_model=robot_model, *args, **kwargs)


class CartesianSpaceMoveTo(toco.PolicyModule):
    """
    Plans and executes a trajectory to a desired end-effector pose
    """

    def __init__(
        self,
        joint_pos_current,
        ee_pos_desired,
        Kp,
        Kd,
        time_to_go: float,
        hz: float,
        robot_model: torch.nn.Module,
        ignore_gravity: bool = True,
        ee_quat_desired=None,
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

        self.N = int(time_to_go * hz)
        self.plan = toco.modules.planning.CartesianSpaceMinJerkPlanner(
            start=ee_pose_current,
            goal=ee_pose_desired,
            steps=self.N,
            time_to_go=time_to_go,
        )

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
        ee_posquat_desired, ee_twist_desired, _ = self.plan(self.i)

        # Control logic
        force_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            ee_posquat_desired[0:3],
            ee_posquat_desired[3:7],
            ee_twist_desired,
        )
        torque_feedback = jacobian.T @ force_feedback

        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis

        torque_out = torque_feedback + torque_feedforward

        # Increment & termination
        self.i += 1
        if self.i == self.N:
            self.set_terminated()

        return {"joint_torques": torque_out}
