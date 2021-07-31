# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Dict

import torch

from polymetis import RobotInterface
import torchcontrol as toco

WARMUP_STEPS = 5
TOTAL_STEPS = 30


class TimestampCheckController(toco.PolicyModule):
    i: int
    warmup_steps: int
    total_steps: int

    def __init__(self, hz, warmup_steps, total_steps):
        super().__init__()

        self.dt = torch.tensor(1.0 / hz)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.i = 0
        self.ts_prev = torch.zeros(2).to(torch.int32)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        ts = state_dict["timestamp"]

        # Check timestamp
        if self.i > self.warmup_steps:
            t_diff = toco.utils.timestamp_diff_seconds(ts, self.ts_prev)
            assert torch.allclose(t_diff, self.dt, atol=1e-3)  # millisecond accuracy

        # Update
        self.i += 1
        self.ts_prev = ts.clone()

        # Termination
        if self.i > self.total_steps:
            self.set_terminated()

        return {"torque_desired": torch.zeros(7)}


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Run policy
    policy = TimestampCheckController(
        hz=robot.metadata.hz, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS
    )
    robot.send_torch_policy(policy)
