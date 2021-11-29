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

        self.joint_pd = toco.modules.feedback.JointSpacePD(Kq, Kqd)
        self.running = False

        num_dofs = self.joint_pd.Kp.shape[0]
        self.joint_pos_desired = torch.nn.Parameter(torch.zeros(num_dofs))

    @torch.jit.export
    def reset(self):
        self.running = False

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Set reference joint position
        if not self.running:
            self.joint_pos_desired[:] = joint_pos_current[:]
            self.running = True

        # PD around current state
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            torch.zeros_like(self.joint_pos_desired),
        )

        return {"joint_torques": torque_feedback}
