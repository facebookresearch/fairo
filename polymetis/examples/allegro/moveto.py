#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ast import arg
import time

import torch
import json
from polymetis import RobotInterface
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pose', action='append', default=[])
    parser.add_argument('-t', '--time2go', default=1, type=float)
    args = parser.parse_args()
    print(args)

    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Move to joint positions
    for pose in args.pose:
        if ',' in pose:
            pose, _ = pose.split(',')
            time_to_go = float(_)
        else:
            time_to_go = args.time2go

        print(f'pose {pose} {time_to_go}')
        with open(pose) as f:
            pos=torch.tensor(json.load(f))
        robot.move_to_joint_positions(pos)

#    for i in range(10):
#        print(i)
#        robot.move_to_joint_positions(pos)
