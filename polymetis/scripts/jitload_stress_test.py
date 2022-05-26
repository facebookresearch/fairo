from typing import Dict
import time

import torch
import polymetis
import torchcontrol as toco

MAX_TRIES = 10


class TmpPolicy(toco.PolicyModule):
    def __init__(self):
        super().__init__()
        self.data = torch.rand([10000, 1000])

    def forward(self, robot_state: Dict[str, torch.Tensor]):
        if not self.is_terminated():
            self.set_terminated()
        return {"joint_torques": torch.zeros(7)}


robot = polymetis.RobotInterface()
try:
    for _ in range(MAX_TRIES):
        robot.send_torch_policy(TmpPolicy())
except KeyboardInterrupt:
    pass
