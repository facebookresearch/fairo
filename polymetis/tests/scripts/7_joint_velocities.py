# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch

from polymetis import RobotInterface
from utils import check_episode_log


def test_new_joint_vel(robot, joint_vel_desired):
    joint_vel = robot.get_joint_velocities()
    print(f"Desired joint velocities: {joint_vel_desired}")
    print(f"New joint velocities: {joint_vel}")
    assert torch.allclose(joint_vel, joint_vel_desired, atol=0.01)
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
    print("=== RobotInterface.start_joint_velocity ===")
    joint_vel = robot.get_joint_velocities()
    joint_vel_limits = robot.robot_model.get_joint_velocity_limits()
    delta_joint_vel_desired = torch.Tensor([0.0, 0.0, 0.0, 0.1, 0.0, -0.1, 0.0])
    joint_vel_desired = joint_vel + delta_joint_vel_desired

    joint_vel_desired = torch.Tensor([0.1, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1])

    robot.start_joint_velocity(joint_vel_desired)
    # for _ in range(20):
    #     joint_vel += 0.05 * delta_joint_vel_desired
    #     assert torch.all(joint_vel < joint_vel_limits)
    #     robot.update_desired_joint_velocities(joint_vel)
    #     time.sleep(0.1)
    time.sleep(0.2)
    state_log = robot.terminate_current_policy()
    print(state_log)

    joint_vel = test_new_joint_vel(robot, joint_vel_desired)
    # state_log = robot.terminate_current_policy()
