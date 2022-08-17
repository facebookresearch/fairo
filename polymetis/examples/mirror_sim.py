# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from polymetis import RobotInterface
import numpy as np
import hydra
from polymetis.utils.data_dir import PKG_ROOT_DIR, which


@hydra.main(config_name="launch_robot")
def main(cfg):
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )
    mirror_sim_client = hydra.utils.instantiate(cfg.robot_client)

    # Reset
    robot.go_home()

    # set state
    print("Running state setter...")
    curr_state = robot.get_robot_state()
    coef = 0.0001
    while True:
        curr_state.joint_positions[:] += coef * np.ones_like(curr_state.joint_positions)
        # robot.set_state(curr_state)
        mirror_sim_client.set_robot_state(curr_state)


# this should be preceded by a call to launch_robot.py robot_client=*_sim on the same machine
if __name__ == "__main__":
    main()
