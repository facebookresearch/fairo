#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from polymetis import RobotInterface
import argparse
import json
import torch

# run
# launch_robot.py robot_client.executable_cfg.readonly=true


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("--ip", default="192.168.0.10")
    args = parser.parse_args()

    # Initialize robot interface
    robot = RobotInterface(ip_address=args.ip, enforce_version=False)

    ee_pos, ee_quat = robot.get_ee_pose()
    q = robot.get_joint_positions()
    print(f"Current ee position: {ee_pos}")
    print(f"Current ee orientation: {ee_quat}  (xyzw)")
    print(f"Joint angles {q}")

    if args.filename:
        with open(args.filename, "w") as f:
            f.write(
                json.dumps(
                    {
                        "position": ee_pos.numpy().tolist(),
                        "orientation": ee_quat.numpy().tolist(),
                        "joints": q.numpy().tolist(),
                    }
                )
            )
        print("pose saved to ", args.filename)
