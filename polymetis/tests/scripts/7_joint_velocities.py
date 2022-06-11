# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch

from polymetis import RobotInterface
from utils import check_episode_log


def test_new_joint_vel(state_log, joint_vel_desired):
    joint_vel = state_log[-1].joint_velocities
    print(f"Desired joint velocities: {joint_vel_desired}")
    print(f"Last joint velocities: {joint_vel}")
    assert torch.allclose(torch.Tensor(joint_vel), joint_vel_desired, atol=0.01)
    return joint_vel


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )
    hz = robot.metadata.hz
    robot.go_home()
    time.sleep(0.5)

    # Joint velocity control
    print("=== RobotInterface.start_joint_velocity_control ===")
    joint_vel = robot.get_joint_velocities()
    joint_vel_limits = robot.robot_model.get_joint_velocity_limits()
    delta_joint_vel_desired = torch.Tensor([0.0, 0.0, 0.0, 0.1, 0.0, -0.1, 0.0])
    joint_vel_desired = joint_vel + delta_joint_vel_desired

    joint_vel_desired = torch.Tensor([0.1, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1])

    robot.start_joint_velocity_control(joint_vel_desired, hz=hz)
    time.sleep(3)
    state_log = robot.terminate_current_policy()
    joint_vel = test_new_joint_vel(state_log, joint_vel_desired)
