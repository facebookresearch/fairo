# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch

import torchcontrol as toco
from torchcontrol.utils import to_tensor


class DefaultController(toco.PolicyModule):
    """
    PD control over current state.
    """

    def __init__(self, Kq, Kqd, **kwargs):
        super().__init__()

        self.impedance = toco.modules.feedback.JointSpacePD(Kq, Kqd)
        self.running = False

        num_dofs = self.impedance.Kp.shape[0]
        self.joint_pos_desired = torch.nn.Parameter(torch.zeros(num_dofs))

    @torch.jit.export
    def reset(self):
        self.running = False

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        joint_pos_current = state_dict["joint_pos"]
        joint_vel_current = state_dict["joint_vel"]

        # Set reference joint position
        if not self.running:
            self.joint_pos_desired[:] = joint_pos_current[:]
            self.running = True

        # PD around current state
        torque_feedback = self.impedance(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            torch.zeros_like(self.joint_pos_desired),
        )

        return {"torque_desired": torque_feedback}
