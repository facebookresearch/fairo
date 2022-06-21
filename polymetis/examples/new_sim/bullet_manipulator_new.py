# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import hydra
from omegaconf.dictconfig import DictConfig
import pybullet
from pybullet_utils.bullet_client import BulletClient

import polymetis_pb2
import polysim
from polymetis.utils.data_dir import get_full_path_to_urdf


class BulletManipulator:
    def __init__(
        self,
        cfg: DictConfig,
        gui: bool,
        gravity: float = 9.81,
    ):
        self.cfg = cfg

        # Initialize PyBullet simulation
        if gui:
            self.sim = BulletClient(connection_mode=pybullet.GUI)
        else:
            self.sim = BulletClient(connection_mode=pybullet.DIRECT)

        urdf_path = get_full_path_to_urdf(self.cfg.robot_description_path)
        self.robot_id = self.sim.loadURDF(
            urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
        )

        for i in range(self.cfg.num_dofs):
            self.sim.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.cfg.controlled_joints[i],
                targetValue=self.cfg.rest_pose[i],
                targetVelocity=0,
            )

        # Initialize states
        self.arm_state = polymetis_pb2.RobotState()
        self.arm_state.prev_joint_torques_computed[:] = np.zeros(7)
        self.arm_state.prev_joint_torques_computed_safened[:] = np.zeros(7)
        self.arm_state.motor_torques_measured[:] = np.zeros(7)
        self.arm_state.motor_torques_external[:] = np.zeros(7)

        self.arm_state.prev_command_successful = True
        self.arm_state.error_code = 0

        self.gripper_state = polymetis_pb2.GripperState()

    def get_arm_state(self) -> polymetis_pb2.RobotState:
        # Timestamp
        self.arm_state.timestamp.GetCurrentTime()

        # Joint pos & vel
        joint_cur_states = self.sim.getJointStates(
            self.robot_id, self.cfg.controlled_joints
        )
        self.arm_state.joint_positions[:] = [joint_cur_states[i][0] for i in range(7)]
        self.arm_state.joint_velocities[:] = [joint_cur_states[i][1] for i in range(7)]

        return self.arm_state

    def get_gripper_state(self) -> polymetis_pb2.GripperState:
        # TODO
        return self.gripper_state

    def apply_arm_control(self, cmd: polymetis_pb2.TorqueCommand):
        # Extract torques
        commanded_torques = np.array(list(cmd.joint_torques))

        # Compute grav comp
        joint_pos = list(self.arm_state.joint_positions)
        grav_comp_torques = self.sim.calculateInverseDynamics(
            self.robot_id,
            joint_pos,
            [0] * len(joint_pos),
            [0] * len(joint_pos),
        )

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

        self.arm_state.prev_command_successful = True
        self.arm_state.error_code = 0

    def apply_gripper_control(self, cmd: polymetis_pb2.GripperCommand):
        pass  # TODO

    def step(self):
        self.sim.stepSimulation()


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    # Create sim
    sim = BulletManipulator(cfg.robot_model, gui=cfg.gui)

    # Connect to Polymetis sim interface
    ps_interface = polysim.SimInterface(cfg.robot_client.metadata_cfg, cfg.hz)
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
