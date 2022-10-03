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

    hw_robot = RobotInterface(
        ip_address=hw_ip,
        port=hw_port,
        use_mirror_sim=True,
        mirror_cfg=cfg,
        mirror_ip=sim_ip,
        mirror_port=sim_port,
    )

    # Create policy instance
    hz = hw_robot.metadata.hz
    Kq_default = torch.Tensor(hw_robot.metadata.default_Kq)
    Kqd_default = torch.Tensor(hw_robot.metadata.default_Kqd)
    Kx_default = torch.Tensor(hw_robot.metadata.default_Kx)
    Kxd_default = torch.Tensor(hw_robot.metadata.default_Kxd)

    print("Planning...")
    target = hw_robot.get_joint_positions()
    target[0] += 0.5
    waypoints = toco.planning.generate_joint_space_min_jerk(
        start=hw_robot.get_joint_positions(),
        goal=target,
        time_to_go=3,
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

    print("Setting up mirror for forward...")
    hw_robot.setup_mirror_for_forward()
    print("Homing simulation...")
    hw_robot.go_home(use_mirror=True)
    print("Running forward sim...")
    hw_robot.send_torch_policy(policy, use_mirror=True)
    hw_robot.clean_mirror_after_forward()

    # Mirror
    print("Syncing...")
    hw_robot.sync_with_mirror()
    print("Homing robot...")
    hw_robot.go_home()
    print("Sending policy...")
    hw_robot.send_torch_policy(policy)
    print("Unsyncing...")
    hw_robot.unsync_with_mirror()


# this should be preceded by a call to launch_robot.py robot_client=None on the same machine
if __name__ == "__main__":
    main()
