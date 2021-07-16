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
    joint_pos = robot.get_joint_angles()
    print(f"Current joint positions: {joint_pos}")

    # Command robot to pose (move 4th and 6th joint)
    # note: can also be done with robot.move_joint_positions
    delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0])
    joint_pos_desired = joint_pos + delta_joint_pos_desired
    print(f"\nMoving joints to: {joint_pos_desired} ...\n")
    state_log = robot.set_joint_positions(joint_pos_desired, time_to_go=2.0)

    # Get updated joint positions
    joint_pos = robot.get_joint_angles()
    print(f"New joint positions: {joint_pos}")
