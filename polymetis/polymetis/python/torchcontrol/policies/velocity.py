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

    def __init__(
        self,
        joint_vel_desired: Union[List, torch.Tensor],
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_vel_desired: Desired joint velocities
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)
        self.previous_timestamp = torch.zeros(2, dtype=torch.int32)
        self.timestep_initialized = False
        self.timestep_to_seconds_transformation = torch.Tensor([1, 1e-9])

        # Reference velocity
        self.joint_vel_desired = torch.nn.Parameter(to_tensor(joint_vel_desired))

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

        # Approximate discretization timestep dt ( assuming dt is consistent between the calls of self.forward() )
        # current_timestamp = state_dict["timestamp"]
        if self.timestep_initialized:
            dt = torch.dot((state_dict["timestamp"]-self.previous_timestamp).to(torch.float), self.timestep_to_seconds_transformation)
        else:
            # In the first timestep, dt is impossible to estimate so best guess is 0.0
            dt = torch.tensor(0.0)
            self.timestep_initialized = True
        self.previous_timestamp = torch.clone(state_dict["timestamp"])

        joint_pos_desired = joint_pos_current + torch.mul(self.joint_vel_desired, dt)

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}
