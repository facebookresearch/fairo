# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict
from polymetis import RobotInterface
import torchcontrol as toco
import torch
import numpy as np
import hydra
from polymetis.utils.data_dir import PKG_ROOT_DIR, which
from polymetis.utils.grpc_utils import check_server_exists

hw_ip = "localhost"
hw_port = 50052
sim_ip = "localhost"
sim_port = 50051


@hydra.main(config_name="launch_robot")
def main(cfg):
    # launch remote CM server + robot client
    # launch local CM server
    assert check_server_exists(
        ip=hw_ip, port=hw_port
    ), f"HW CM server must be started at {hw_ip}:{hw_port}"
    assert check_server_exists(
        ip=sim_ip, port=sim_port
    ), f"Sim CM server must be started at {sim_ip}:{sim_port}"
    # launch sim client, TODO: move mirror sim into robot interface
    mirror_sim_client = hydra.utils.instantiate(cfg.robot_client)
    mirror_sim_client.init_robot_client()

    # create hw and sim robots
    sim_robot = RobotInterface(ip_address=sim_ip, port=sim_port)
    hw_robot = RobotInterface(ip_address=hw_ip, port=hw_port)

    # Create policy instance
    hz = hw_robot.metadata.hz
    Kq_default = torch.Tensor(hw_robot.metadata.default_Kq)
    Kqd_default = torch.Tensor(hw_robot.metadata.default_Kqd)
    Kx_default = torch.Tensor(hw_robot.metadata.default_Kx)
    Kxd_default = torch.Tensor(hw_robot.metadata.default_Kxd)

    print("Planning...")
    waypoints = toco.planning.generate_joint_space_min_jerk(
        start=hw_robot.get_joint_positions(),
        goal=hw_robot.get_joint_positions() + 0.3,
        time_to_go=5,
        hz=hz,
    )
    print("Creating policy...")
    policy = toco.policies.JointTrajectoryExecutor(
        joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
        joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
        Kq=Kq_default,
        Kqd=Kqd_default,
        Kx=Kx_default,
        Kxd=Kxd_default,
        robot_model=hw_robot.robot_model,
        ignore_gravity=hw_robot.use_grav_comp,
    )

    # Reset and test in sim, TODO: not easily doable until the mirror sim is moved into RoboInterface
    # sim_robot.go_home()
    # sim_robot.send_torch_policy(policy)

    # Mirror
    print("Homing robot...")
    hw_robot.go_home()
    print("Syncing...")
    mirror_sim_client.sync(hw_robot)  # must be non-blocking
    print("Sending policy...")
    hw_robot.send_torch_policy(policy)
    print("Unsyncing...")
    mirror_sim_client.unsync()


# this should be preceded by a call to launch_robot.py robot_client=None on the same machine
if __name__ == "__main__":
    main()
