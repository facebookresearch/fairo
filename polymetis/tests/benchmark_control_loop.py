from typing import Dict

import numpy as np
import torch
import hydra

from polymetis import RobotInterface
import torchcontrol as toco


class MyPolicy(toco.PolicyModule):
    steps: int
    i: int

    def __init__(self, steps, *args, **kwargs):
        super().__init__()

        self.impedance = toco.policies.JointImpedanceControl(*args, **kwargs)

        self.steps = steps
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        output = self.impedance(state_dict)

        self.i += 1
        if self.i == self.steps:
            self.set_terminated()

        return output


@hydra.main(config_path="", config_name="config.yml")
def main(cfg):
    robot = RobotInterface()

    policy = MyPolicy(
        steps=cfg.steps,
        joint_pos_current=robot.get_joint_angles(),
        Kp=robot.metadata.default_Kq,
        Kd=robot.metadata.default_Kqd,
        robot_model=robot.robot_model,
    )

    duration_ls = []
    for i in range(cfg.iterations):
        # Send policy
        state_log = robot.send_torch_policy(policy)

        # Record control loop durations
        durations = [state.control_loop_ms for state in state_log]
        print(durations)

        duration_ls.append(durations)

    # Save to file
    with open("results.npy", "wb") as f:
        np.save(f, np.array(duration_ls))


if __name__ == "__main__":
    main()
