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
        hz=torch.tensor(0, dtype=torch.int32),
        ignore_gravity=True,
    ):
        """
        Args:
            joint_vel_desired: Desired joint velocities
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            hz: Optional[Int]: frequency of the calls to forward(). If this is omited then it is estimated on the fly
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.hz = hz
        if self.hz > 0:
            self.dt = 1./(self.hz.to(torch.float))
        else:
            # In the first timestep, dt is impossible to estimate so best guess is 0.0
            self.dt = torch.tensor(0.0)
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)
        self.previous_timestamp = torch.zeros(2, dtype=torch.int32)
        self.is_initialized = False
        self.timestep_to_seconds_transformation = torch.Tensor([1, 1e-9])

        # Reference velocity
        self.joint_vel_desired = torch.nn.Parameter(to_tensor(joint_vel_desired))
        # Initialize position desired
        self.joint_pos_desired = torch.zeros_like(joint_vel_desired)

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

        # Approximate discretization timestep dt if needed ( assuming dt is consistent between the calls of self.forward() )
        if self.is_initialized:
            if self.hz == 0:
                self.dt = torch.dot((current_timestamp-self.previous_timestamp).to(torch.float), self.timestep_to_seconds_transformation)
                self.previous_timestamp = torch.clone(current_timestamp)
        else:
            self.joint_pos_desired = torch.clone(joint_pos_current)
            self.is_initialized = True
            if self.hz == 0:
                self.previous_timestamp = torch.clone(current_timestamp)

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
