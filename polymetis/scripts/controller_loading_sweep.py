from typing import Dict
import time
import click
import random

import torch

import polymetis
import torchcontrol as toco

POLICY_SIZE_LIST = [100, 1000, 10000]
# Note: 2 sec joint traj = 100


class FatPolicy(toco.PolicyModule):
    def __init__(self, data_size):
        super().__init__()
        self.data = torch.rand([data_size, 256])

    def forward(self, robot_state: Dict[str, torch.Tensor]):
        if not self.is_terminated():
            self.set_terminated()
        return {"joint_torques": torch.zeros(7)}


def run_experiment(robot, policy_size):
    state_log = robot.send_torch_policy(FatPolicy(policy_size))
    time.sleep(0.1)
    return state_log


@click.command()
@click.option("--priority", "-p", default=-1)
@click.option("--repeats", "-r", default=100)
def main(priority, repeats):
    # Initialize robot
    robot = polymetis.RobotInterface()
    robot.get_robot_state(prio=priority)

    # Initialize data
    data = {"policy_sizes": POLICY_SIZE_LIST}
    for policy_size in POLICY_SIZE_LIST:
        data[policy_size] = []

    # Sweep through experiments
    for r in range(repeats):
        for policy_size in random.sample(POLICY_SIZE_LIST, len(POLICY_SIZE_LIST)):
            state_log = run_experiment(robot, policy_size)
            data[policy_size].append(state_log)

    # Save data
    filename = f"jitload_sweep_{priority}.pt"
    torch.save(data, filename)


if __name__ == "__main__":
    main()
