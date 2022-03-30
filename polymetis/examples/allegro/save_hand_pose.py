#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import polymetis
import torchcontrol as toco
from typing import Dict
import argparse
import json
import torch


class ZeroController(toco.PolicyModule):
    def forward(self, state_dict: Dict[str, torch.Tensor]):
        return {"joint_torques": torch.zeros_like(state_dict["joint_positions"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    args = parser.parse_args()

    robot = polymetis.RobotInterface(ip_address="localhost", enforce_version=False)
    robot.send_torch_policy(ZeroController(), blocking=False)

    q = robot.get_joint_positions().numpy().tolist()
    print(q)
    if args.filename:
        with open(args.filename, "w") as f:
            f.write(json.dumps(q))
        print("pose saved to ", args.filename)
