# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra

import polymetis_pb2
import polysim


class BulletManipulator:
    def __init__(
        self,
        robot_model_cfg: DictConfig,
        gui: bool,
        use_grav_comp: bool = True,
        gravity: float = 9.81,
    ):
        self.cfg = robot_model_cfg
        self.use_grav_comp = use_grav_comp

        # Initialize PyBullet simulation
        if self.gui:
            self.sim = BulletClient(connection_mode=pybullet.GUI)
        else:
            self.sim = BulletClient(connection_mode=pybullet.DIRECT)

        self.robot_id = self.sim.loadURDF(abs_urdf_path)

        self.sim.setGravity(0, 0, -gravity)

        # Initialize states
        self.arm_state = polymetis_pb2.RobotState()
        self.gripper_state = polymetis_pb2.GripperState()

    def get_arm_state(self) -> polymetis_pb2.RobotState:
        # Timestamp
        robot_state.timestamp.GetCurrentTime()

        # Joint pos & vel
        joint_cur_states = self.sim.getJointStates(
            self.robot_id, self.controlled_joints
        )
        self.arm_state.joint_positions[:] = [
            joint_cur_states[i][0] for i in range(self.n_dofs)
        ]
        self.arm_state.joint_velocities[:] = [
            joint_cur_states[i][1] for i in range(self.n_dofs)
        ]

        return self.arm_state

    def get_gripper_state(self) -> polymetis_pb2.GripperState:
        pass  # TODO

    def apply_arm_control(self, cmd: polymetis_pb2.TorqueCommand):
        # Extract torques
        commanded_torques = np.array(list(cmd.joint_torques))

        # Compute grav comp
        if self.use_grav_comp:
            joint_pos = list(self.arm_state.joint_positions)
            grav_comp_torques = self.sim.calculateInverseDynamics(
                joint_pos=joint_pos,
                joint_vel=[0] * len(joint_pos),
                joint_acc=[0] * len(joint_pos),
            )
        else:
            grav_comp_torques = np.zeros_like(commanded_torques)

        # Set sim torques
        applied_torques = commanded_torques + grav_comp_torques
        self.sim.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.cfg.controlled_joints,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=applied_torques,
        )

        # Populate torques in state
        self.arm_state.prev_joint_torques_computed[:] = commanded_torques
        self.arm_state.prev_joint_torques_computed_safened[:] = commanded_torques
        self.arm_state.motor_torques_measured[:] = applied_torques
        self.arm_state.motor_torques_external[:] = np.zeros_like(applied_torques)

    def apply_gripper_control(self, cmd: polymetis_pb2.GripperCommand):
        pass  # apply gripper action

    def step(self):
        self.sim.stepSimulation()


@hydra.main()
def main(cfg):
    # Create sim
    sim = BulletManipulator(...)  # TODO

    # Connect to Polymetis sim interface
    ps_interface = polysim.SimInterface(cfg.sim_cfg)  # TODO
    ps_interface.register_control_callback(
        server_ip=cfg.arm.ip,
        server_port=cfg.arm.port,
        server_type=polysim.ControlType.ARM,
        state_callback=sim.get_arm_state,
        action_callback=sim.apply_arm_control,
    )
    ps_interface.register_control_callback(
        server_ip=cfg.gripper.ip,
        server_port=cfg.gripper.port,
        server_type=polysim.ControlType.GRIPPER,
        state_callback=sim.get_gripper_state,
        action_callback=sim.apply_gripper_control,
    )
    ps_interface.register_step_callback(sim.step)

    # Start sim client
    ps_interface.run()


if __name__ == "__main__":
    main()
