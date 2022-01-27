# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import time

from polymetis import RobotInterface


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    robot.go_home()

    # Joint impedance control
    joint_positions = robot.get_joint_positions()

    robot.start_joint_impedance()

    for i in range(20):
        joint_positions += torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0])
        robot.update_desired_joint_positions(joint_positions)
        time.sleep(0.1)

    robot.terminate_impedance_controller()

    # Cartesian impedance control
    ee_pos, ee_quat = robot.get_ee_pose()

    robot.start_cartesian_impedance()

    for i in range(20):
        ee_pos += torch.Tensor([0.0, 0.0, -0.01])
        time.sleep(0.1)

    robot.terminate_impedance_controller()
