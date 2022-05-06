# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import polymetis
import torchcontrol as toco
from typing import Dict


class ZeroController(toco.PolicyModule):
    def forward(self, state_dict: Dict[str, torch.Tensor]):
        return {"joint_torques": torch.zeros_like(state_dict["joint_positions"])}


if __name__ == "__main__":
    robot = polymetis.RobotInterface(ip_address="172.16.0.1", enforce_version=False)
    robot.send_torch_policy(ZeroController())
