# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from polymetis import RobotInterface


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    robot.go_home()

    # Get joint positions
    joint_positions = robot.get_joint_positions()
    print(f"Current joint positions: {joint_positions}")

    # Command robot to pose (move 4th and 6th joint)
    joint_positions_desired = torch.Tensor(
        [-0.14, -0.02, -0.05, -1.57, 0.05, 1.50, -0.91]
    )
    print(f"\nMoving joints to: {joint_positions_desired} ...\n")
    state_log = robot.move_to_joint_positions(joint_positions_desired, time_to_go=2.0)

    # Get updated joint positions
    joint_positions = robot.get_joint_positions()
    print(f"New joint positions: {joint_positions}")
