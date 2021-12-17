# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import threading

import torch

from polymetis import RobotInterface
from utils import check_episode_log


def connect_and_send_policy():
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
    check_episode_log(state_log, int(robot.time_to_go_default * hz))

    joint_pos = robot.get_joint_angles()
    assert torch.allclose(joint_pos, joint_pos_desired, atol=0.01)

    return True


if __name__ == "__main__":
    thread = threading.Thread(target=connect_and_send_policy)
    thread.start()
    thread.join()
