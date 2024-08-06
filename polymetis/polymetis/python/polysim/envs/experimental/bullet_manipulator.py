# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import pybullet
from pybullet_utils.bullet_client import BulletClient

import polymetis_pb2
import polysim
from polymetis.utils.data_dir import get_full_path_to_urdf


class BulletManipulator:
    def __init__(
        self,
        hz: int,
        cfg: DictConfig,
        gui: bool,
        gravity: float = 9.81,
    ):
        self.cfg = cfg
        self.dt = 1.0 / hz

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

        for i in range(7):
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

        self.t = 0

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
        # Timestamp
        self.gripper_state.timestamp.GetCurrentTime()

        # Gripper states
        joint_cur_states = self.sim.getJointStates(
            self.robot_id, self.cfg.gripper.controlled_joints
        )
        self.gripper_state.width = float(
            joint_cur_states[0][0] + joint_cur_states[1][0]
        )
        self.gripper_state.is_grasped = False  # TODO
        self.gripper_state.is_moving = np.all(
            [
                abs(joint_cur_states[i][1]) < self.cfg.gripper.moving_threshold
                for i in range(2)
            ]
        )

        return self.gripper_state

    def apply_arm_control(self, cmd: polymetis_pb2.TorqueCommand):
        # Extract torques
        commanded_torques = np.array(list(cmd.joint_torques))

        # Compute grav comp
        joint_pos = list(self.arm_state.joint_positions)
        finger_pos = [self.gripper_state.width / 2.0] * 2
        grav_comp_torques = self.sim.calculateInverseDynamics(
            self.robot_id,
            joint_pos + finger_pos,
            [0] * 9,
            [0] * 9,
        )[:7]

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

    def apply_gripper_control(self, cmd: polymetis_pb2.GripperCommand):
        self.sim.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.cfg.gripper.controlled_joints,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=[cmd.width / 2.0] * 2,
        )

    def step(self):
        self.sim.stepSimulation()
        self.t += self.dt


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    # Create sim
    sim_wrapper = BulletManipulator(cfg.hz, cfg.robot_model, gui=cfg.gui)

    # Connect to Polymetis sim interface
    sim_client = polysim.SimInterface(cfg.hz)
    sim_client.register_arm_control(
        server_address=f"{cfg.arm.ip}:{cfg.arm.port}",
        state_callback=sim_wrapper.get_arm_state,
        action_callback=sim_wrapper.apply_arm_control,
        dof=7,
        kp_joint=cfg.robot_client.metadata_cfg.default_Kq,
        kd_joint=cfg.robot_client.metadata_cfg.default_Kqd,
        kp_ee=cfg.robot_client.metadata_cfg.default_Kx,
        kd_ee=cfg.robot_client.metadata_cfg.default_Kxd,
        urdf_path="franka_panda/panda_arm.urdf",
        rest_pose=cfg.robot_model.rest_pose,
        ee_link_name=cfg.robot_model.ee_link_name,
    )
    sim_client.register_gripper_control(
        server_address=f"{cfg.gripper.ip}:{cfg.gripper.port}",
        state_callback=sim_wrapper.get_gripper_state,
        action_callback=sim_wrapper.apply_gripper_control,
        max_width=cfg.robot_model.gripper.max_width,
    )
    sim_client.register_step_callback(sim_wrapper.step)

    # Start sim client
    sim_client.run()


if __name__ == "__main__":
    main()
