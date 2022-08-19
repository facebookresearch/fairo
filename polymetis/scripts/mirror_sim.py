# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from polymetis import RobotInterface
import numpy as np
import hydra
from polymetis.utils.data_dir import PKG_ROOT_DIR, which
from polymetis.utils.grpc_utils import check_server_exists

hw_ip = "localhost"
hw_port = 50052
sim_ip = "localhost"
sim_port = 50051


class MyPolicy(toco.PolicyModule):
    """
    Custom policy that executes a sine trajectory on joint 6
    (magnitude = 0.5 radian, frequency = 1 second)
    """

    def __init__(self, time_horizon, hz, magnitude, period, kq, kqd, coef, **kwargs):
        """
        Args:
            time_horizon (int):         Number of steps policy should execute
            hz (double):                Frequency of controller
            kq, kqd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.hz = hz
        self.time_horizon = time_horizon
        self.m = magnitude
        self.T = period

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

        # Initialize variables
        self.steps = 0
        self.q_initial = torch.zeros_like(kq)
        self.coef = coef

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        # Initialize
        if self.steps == 0:
            self.q_initial = q_current.clone()

        # Compute reference position and velocity
        q_desired = self.q_initial.clone()
        q_desired += np.ones_like(q_desired) * self.coef
        qd_desired = torch.zeros_like(qd_current)

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, q_desired, torch.zeros_like(qd_current)
        )

        # Check termination
        if self.steps > self.time_horizon:
            self.set_terminated()
        self.steps += 1

        return {"joint_torques": output}


@hydra.main(config_name="launch_robot")
def main(cfg):
    # launch remote CM server + robot client
    # launch local CM server
    assert check_server_exists(
        ip=hw_ip, port=hw_port
    ), f"HW CM server must be started at {hw_ip}:{hw_port}"
    assert check_server_exists(
        ip="localhost", port=50051
    ), f"Sim CM server must be started at {sim_ip}:{sim_port}"

    # launch sim client, TODO: move mirror sim into robot interface
    mirror_sim_client = hydra.utils.instantiate(cfg.robot_client)
    mirror_sim_client.init_connection()

    # create hw and sim robots
    sim_robot = RobotInterface(ip=sim_ip, port=sim_port)
    hw_robot = RobotInterface(ip=hw_ip, port=sim_port)

    # Create policy instance
    hz = hw_robot.metadata.hz
    default_kq = torch.Tensor(hw_robot.metadata.default_Kq)
    default_kqd = torch.Tensor(hw_robot.metadata.default_Kqd)
    policy = MyPolicy(
        time_horizon=5 * hz,
        hz=hz,
        magnitude=0.5,
        period=2.0,
        kq=default_kq,
        kqd=default_kqd,
        coef=0.001,
    )

    # Reset and test in sim
    sim_robot.go_home()
    sim_robot.send_torch_policy(policy)

    # Mirror
    hw_robot.go_home()
    mirror_sim_client.sync(hw_robot)  # must be non-blocking
    hw_robot.send_torch_policy(policy)
    mirror_sim_client.unsync()


# this should be preceded by a call to launch_robot.py robot_client=None on the same machine
if __name__ == "__main__":
    main()
