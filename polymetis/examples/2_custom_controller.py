# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import numpy as np
import torch

from polymetis import RobotInterface
import torchcontrol as toco


class MySinePolicy(toco.PolicyModule):
    """
    Custom policy that executes a sine trajectory on joint 6
    (magnitude = 0.5 radian, frequency = 1 second)
    """

    def __init__(self, time_horizon, hz, magnitude, period, kq, kqd, **kwargs):
        """
        Args:
            time_horizon (int):         Number of steps policy should execute
            hz (double):                Frequency of controller
            kq, kqd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.hz = hz
        self.time_horizon = time_horizon
        self.m = magnitude
        self.T = period

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

        # Initialize variables
        self.steps = 0
        self.q_initial = torch.zeros_like(kq)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_pos"]
        qd_current = state_dict["joint_vel"]

        # Initialize
        if self.steps == 0:
            self.q_initial = q_current.clone()

        # Compute reference position and velocity
        q_desired = self.q_initial.clone()
        q_desired[5] = self.q_initial[5] + self.m * torch.sin(
            np.pi * self.steps / (self.hz * self.T)
        )
        qd_desired = torch.zeros_like(qd_current)

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, q_desired, torch.zeros_like(qd_current)
        )

        # Check termination
        if self.steps > self.time_horizon:
            self.set_terminated()
        self.steps += 1

        return {"torque_desired": output}


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    robot.go_home()

    # Create policy instance
    hz = robot.metadata.hz
    default_kq = torch.Tensor(robot.metadata.default_Kq)
    default_kqd = torch.Tensor(robot.metadata.default_Kqd)
    policy = MySinePolicy(
        time_horizon=5 * hz,
        hz=hz,
        magnitude=0.5,
        period=2.0,
        kq=default_kq,
        kqd=default_kqd,
    )

    # Run policy
    print("\nRunning custom sine policy ...\n")
    state_log = robot.send_torch_policy(policy)
