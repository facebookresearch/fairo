# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch

import torchcontrol as toco
from torchcontrol.utils.tensor_utils import to_tensor


class iLQR(toco.PolicyModule):
    """Executes a time-varying linear feedback policy (output of iLQR optimization)"""

    i: int

    def __init__(self, Kxs, x_desireds, u_ffs):
        """
        Definitions:
            state_dim = number of state dimensions
            num_dofs = number of action dimensions

        Args:
            Kxs: [ time_horizon x num_dofs x state_dim ] series of gain matrices
            x_desireds: [ time_horizon x state_dim ] series of desired state
            u_ffs: [ time_horizon x num_dofs ] series of desired torques
        """
        super().__init__()

        # Dimension checks
        assert Kxs.ndim == 3
        assert x_desireds.ndim == 2
        assert u_ffs.ndim == 2
        assert Kxs.shape[0] == x_desireds.shape[0]
        assert Kxs.shape[0] == u_ffs.shape[0]
        assert Kxs.shape[2] == x_desireds.shape[1]
        assert Kxs.shape[1] == u_ffs.shape[1]

        # Initialize params & modules
        self.Kxs = torch.nn.Parameter(to_tensor(Kxs))
        self.x_desireds = torch.nn.Parameter(to_tensor(x_desireds))
        self.u_ffs = torch.nn.Parameter(to_tensor(u_ffs))

        self.feedback = toco.modules.LinearFeedback(torch.zeros_like(self.Kxs[0, :, :]))

        # Initialize step count
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: Robot state dictionary

        Returns:
            A dictionary containing the controller output
        """
        # Extract state vector
        x = torch.cat(
            [state_dict["joint_positions"], state_dict["joint_velocities"]], dim=-1
        )

        # Select linear feedback gains & reference
        self.feedback.update({"K": self.Kxs[self.i, :, :]})

        x_desired = self.x_desireds[self.i, :]
        u_ff = self.u_ffs[self.i, :]

        # Increment & termination
        self.i += 1
        if self.i == self.Kxs.shape[0]:
            self.set_terminated()

        u_output = self.feedback(x, x_desired) + u_ff

        return {"joint_torques": u_output}
