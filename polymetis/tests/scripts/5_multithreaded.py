# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import threading

import torch

from polymetis import RobotInterface
from utils import check_episode_log


success = []
exceptions = []


def connect_and_send_policy():
    try:
        # Initialize robot interface
        robot = RobotInterface(
            ip_address="localhost",
        )
        hz = robot.metadata.hz
        robot.go_home()
        time.sleep(0.5)

        # Get joint positions
        joint_pos = robot.get_joint_positions()
        print(f"Initial joint positions: {joint_pos}")

        # Go to joint positions
        print("=== RobotInterface.move_to_joint_positions ===")
        delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0])
        joint_pos_desired = joint_pos + delta_joint_pos_desired

        time_to_go = 4.0
        state_log = robot.move_to_joint_positions(
            joint_pos_desired, time_to_go=time_to_go
        )
        check_episode_log(state_log, int(time_to_go * hz))

        joint_pos = robot.get_joint_positions()
        assert torch.allclose(joint_pos, joint_pos_desired, atol=0.01)

        success.append(True)
    except Exception as e:
        exceptions.append(e)


if __name__ == "__main__":
    thread = threading.Thread(target=connect_and_send_policy)
    thread.start()
    thread.join()

    assert success, f"Exception: {exceptions[0]}"
