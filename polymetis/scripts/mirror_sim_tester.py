# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import numpy as np
import torch

from polymetis import RobotInterface
import torchcontrol as toco

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(ip_address="localhost", port=50052)

    # Reset
    robot.go_home()

    hz = robot.metadata.hz
    Kq_default = torch.Tensor(robot.metadata.default_Kq)
    Kqd_default = torch.Tensor(robot.metadata.default_Kqd)
    Kx_default = torch.Tensor(robot.metadata.default_Kx)
    Kxd_default = torch.Tensor(robot.metadata.default_Kxd)
    target = robot.get_joint_positions()
    target[0] += 1.0
    waypoints = toco.planning.generate_joint_space_min_jerk(
        start=robot.get_joint_positions(),
        goal=target,
        time_to_go=5,
        hz=hz,
    )
    print("Creating policy...")
    policy = toco.policies.JointTrajectoryExecutor(
        joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
        joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
        Kq=Kq_default,
        Kqd=Kqd_default,
        Kx=Kx_default,
        Kxd=Kxd_default,
        robot_model=robot.robot_model,
        ignore_gravity=robot.use_grav_comp,
    )

    # Run policy
    print("\nRunning custom policy ...\n")
    state_log = robot.send_torch_policy(policy)
