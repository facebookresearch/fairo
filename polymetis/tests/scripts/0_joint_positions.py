# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch

from polymetis import RobotInterface
from utils import check_episode_log


def test_new_joint_pos(robot, joint_pos_desired):
    joint_pos = robot.get_joint_angles()
    print(f"Desired joint positions: {joint_pos_desired}")
    print(f"New joint positions: {joint_pos}")
    assert torch.allclose(joint_pos, joint_pos_desired, atol=0.01)
    return joint_pos


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )
    hz = robot.metadata.hz
    robot.go_home()
    time.sleep(0.5)

    # Get joint positions
    joint_pos = robot.get_joint_angles()
    print(f"Initial joint positions: {joint_pos}")

    # Go to joint positions
    print("=== RobotInterface.set_joint_positions ===")
    delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0])
    joint_pos_desired = joint_pos + delta_joint_pos_desired

    state_log = robot.set_joint_positions(joint_pos_desired)
    time.sleep(0.5)

    joint_pos = test_new_joint_pos(robot, joint_pos_desired)
    check_episode_log(state_log, int(robot.time_to_go_default * hz))

    # Move by delta joint positions
    print("=== RobotInterface.move_joint_positions ===")
    delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5])
    joint_pos_desired = joint_pos + delta_joint_pos_desired

    state_log = robot.move_joint_positions(delta_joint_pos_desired)
    time.sleep(0.5)

    joint_pos = test_new_joint_pos(robot, joint_pos_desired)
    check_episode_log(state_log, int(robot.time_to_go_default * hz))

    # Go home
    print("=== RobotInterface.go_home ===")
    joint_pos_desired = torch.Tensor(robot.home_pose)

    state_log = robot.go_home()
    time.sleep(0.5)

    joint_pos = test_new_joint_pos(robot, joint_pos_desired)
    check_episode_log(state_log, int(robot.time_to_go_default * hz))
