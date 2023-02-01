# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import time

from ipaddress import ip_address
from polymetis import RobotInterface
from polymetis import GripperInterface


def do_grasps(n):
    # Do n iterations of a basic grasping motion.
    robot.go_home(time_to_go=1)
    for i in range(0, n):
        robot.move_to_joint_positions([90, 90, 90, 90, 90, -20, 0, 0, 0, 0], 0.2)
        time.sleep(1)
        robot.move_to_joint_positions([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.2)
        time.sleep(1)


if __name__ == "__main__":
    # Define the a set of joint angle positions to iterate through to test each joint on
    # the robot arm.
    time_scale = 12
    test_sequence = [
        [[-1, -0.1, 0, 0, 1.2, 0.5, 0.4], time_scale / 2],
        [[-1, 0.1, 0, 0, 1.2, 0, -0.4], time_scale / 4],
        [[0, -0.1, 0, 0, -1.2, 0, 0.4], time_scale / 2],
        [[0, 0.1, -1, 0, 1.2, 0.5, -0.4], time_scale / 2],
        [[0, -0.1, -1, 0, 1.2, 0, 0.4], time_scale / 2],
        [[-1, 0.1, 0, 0, 1.2, 0.5, -0.4], time_scale / 2],
        [[-1, -0.1, 0, 0, 1.2, 0, 0.4], time_scale / 4],
        [[0, 0.1, 0, 0, -1.2, 0, -0.4], time_scale / 2],
        [[0, -0.1, -1, 0, 1.2, 0.5, 0.4], time_scale / 2],
        [[0, 0.1, -1, 0, 1.2, 0, -0.4], time_scale / 2],
        [[0, 0, 0, 0, 0, 0, 0], time_scale / 3],
    ]

    robot = RobotInterface(ip_address="127.0.0.1", port=50052)
    robot.get_robot_state()
    print(robot.get_joint_positions())
    print(robot.get_joint_velocities())
    robot.set_home_pose([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Initialize robot interface
    panda = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    robot.go_home(time_to_go=1)
    # Get the current joint positions
    home_positions = [float(x) for x in panda.get_joint_positions()]

    for sequence, time_to_go in test_sequence:
        # Create an initial target based upon the home positions
        target_positions = list(home_positions)
        for joint, angle in enumerate(sequence):
            if angle == 0:
                continue

            # Update the target position
            target_positions[joint] = (44 / 21) * angle

        do_grasps(1)

        # Create a tensor based on the target positions
        joint_positions_desired = torch.Tensor(target_positions)

        # Move the robot to the target positions
        state_log = panda.move_to_joint_positions(
            joint_positions_desired, time_to_go=time_to_go
        )
