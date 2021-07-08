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
    """Executes a joint trajectory plan by stabilizing around it"""

    N: int
    i: int

    def __init__(
        self,
        plan: torch.nn.Module,
        robot_model: torch.nn.Module,
        steps: int,
        Kp,
        Kd,
        ignore_gravity: bool = True,
    ):
        """
        Args:
            plan: A valid plan from torchcontrol.modules.planning that outputs joint references
            robot_model: A valid robot model module from torchcontrol.models
            steps: Number of steps in plan
            Kp: P gains in joint space
            Kd: D gains in joint space
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        self.plan = plan
        self.robot_model = robot_model
        self.N = steps

        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.impedance = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Executes joint PD around the planned joint trajectory
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Select desired state
        q_desired, qd_desired, _ = self.plan(self.i)

        # Control logic
        torque_feedback = self.impedance(
            joint_pos_current, joint_vel_current, q_desired, qd_desired
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )
        torque_out = torque_feedback + torque_feedforward

        # Increment & termination
        self.i += 1
        if self.i == self.N:
            self.set_terminated()

        return {"joint_torques": torque_out}


class JointSpaceMoveTo(toco.PolicyModule):
    """
    Plans and executes a trajectory to a desired joint position
    """

    def __init__(
        self,
        joint_pos_current,
        joint_pos_desired,
        time_to_go: float,
        hz: float,
        **kwargs,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            joint_pos_desired: Desired joint positions
            time_to_go: Duration of trajectory in seconds
            hz: Frequency of controller
        """
        super().__init__()

        joint_pos_current = to_tensor(joint_pos_current)
        joint_pos_desired = to_tensor(joint_pos_desired)

        # Planning
        N = int(time_to_go * hz)
        plan = toco.modules.planning.JointSpaceMinJerkPlanner(
            start=joint_pos_current,
            goal=joint_pos_desired,
            steps=N,
            time_to_go=time_to_go,
        )

        # Instantiate plan executor
        self.plan_executor = JointPlanExecutor(plan=plan, steps=N, **kwargs)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.plan_executor(state_dict)
        if self.plan_executor.is_terminated():
            self.set_terminated()

        return output


class OperationalSpaceMoveTo(toco.PolicyModule):
    """
    Plans and executes a trajectory to a desired end-effector pose
    """

    def __init__(
        self,
        joint_pos_current,
        ee_pos_desired,
        robot_model: torch.nn.Module,
        time_to_go: float,
        hz: float,
        ee_quat_desired=None,
        **kwargs,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            ee_pos_desired: Desired end-effector position (3D)
            time_to_go: Duration of trajectory in seconds
            hz: Frequency of controller
            ee_quat_desired: Desired end-effector orientation (None if current orientation is desired)
        """
        super().__init__()

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

        # Planning
        N = int(time_to_go * hz)
        plan = toco.modules.planning.CartesianSpaceMinJerkJointPlanner(
            joint_pos_start=joint_pos_current,
            ee_pose_goal=ee_pose_desired,
            steps=N,
            time_to_go=time_to_go,
            robot_model=robot_model,
        )

        # Instantiate plan executor
        self.plan_executor = JointPlanExecutor(
            plan=plan, steps=N, robot_model=robot_model, **kwargs
        )

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.plan_executor(state_dict)
        if self.plan_executor.is_terminated():
            self.set_terminated()

        return output
