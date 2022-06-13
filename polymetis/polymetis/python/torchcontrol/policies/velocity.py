# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Union

import torch

import torchcontrol as toco
from torchcontrol.utils import to_tensor


class JointVelocityControl(toco.PolicyModule):
    """
    Velocity control in joint space.
    """

    hz: int
    dt: float
    is_initialized: bool

    def __init__(
        self,
        joint_vel_desired: Union[List, torch.Tensor],
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        hz: int,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_vel_desired: Desired joint velocities
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            hz: int: frequency of the calls to forward()
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.dt = 1.0 / (hz)
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)
        self.is_initialized = False

        # Reference velocity
        self.joint_vel_desired = torch.nn.Parameter(to_tensor(joint_vel_desired))
        # Initialize position desired
        self.joint_pos_desired = torch.zeros_like(self.joint_vel_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]
        current_timestamp = state_dict["timestamp"]

        if not self.is_initialized:
            self.joint_pos_desired = torch.clone(joint_pos_current)
            self.is_initialized = True

        # original formulation of the PI controller is to integrate (v_desired - v_current) * dt, which is the integral
        # of v_desired * dt (desired position resulting from all desired velocities) minus the integral of v_current * dt (current position).
        self.joint_pos_desired += torch.mul(self.joint_vel_desired, self.dt)

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}
