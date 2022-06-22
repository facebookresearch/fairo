# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import time

from ipaddress import ip_address
from polymetis import RobotInterface
from polymetis import GripperInterface


if __name__ == "__main__":

    # Define the a set of joint angle positions to iterate through to test each joint on
    # the robot arm.
    time_scale = 12
    test_sequence = [
        [[0, 0, 0, 0, 0, 0, 0], time_scale/3],
        [[-1, -0.1, 0, 0, 1.2, 0.5, 0.4], time_scale/2],
        [[-1, 0.1, 0, 0, 1.2, 0, -0.4], time_scale/4],
        [[0, -0.1, 0, 0, -1.2, 0, 0.4], time_scale/2],
        [[0, 0.1, -1, 0, 1.2, 0.5, -0.4], time_scale/2],
        [[0, -0.1, -1, 0, 1.2, 0, 0.4], time_scale/2],
        [[-1, 0.1, 0, 0, 1.2, 0.5, -0.4], time_scale/2],
        [[-1,-0.1, 0, 0, 1.2, 0, 0.4], time_scale/4],
        [[0, 0.1, 0, 0, -1.2, 0, -0.4], time_scale/2],
        [[0,-0.1, -1, 0, 1.2, 0.5, 0.4], time_scale/2],
        [[0, 0.1, -1, 0, 1.2, 0, -0.4], time_scale/2],
        [[0, 0, 0, 0, 0, 0, 0], time_scale/3],

    ]

    gripper = GripperInterface(ip_address="localhost")
    state = gripper.get_state()
    print(f"state: {state}")

    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    robot.go_home()

    # Get the current joint positions
    home_positions = [float(x) for x in robot.get_joint_positions()]

    for sequence, time_to_go in test_sequence:

        # Create an initial target based upon the home positions
        target_positions = list(home_positions)
        for joint, angle in enumerate(sequence):
            if angle == 0:
                continue

            # Update the target position
            target_positions[joint] = (44 / 21) * angle

        gripper.goto(width=1, speed=0.8, force=.1)
        time.sleep(2.0)
        state = gripper.get_state()
        print(f"state: {state}")

        gripper.goto(width=0, speed=0.8, force=.1)
        time.sleep(2.0)
        state = gripper.get_state()
        print(f"state: {state}")

        # Create a tensor based on the target positions
        joint_positions_desired = torch.Tensor(target_positions)

        # Move the robot to the target positions
        state_log = robot.move_to_joint_positions(
            joint_positions_desired, time_to_go=time_to_go
        )
