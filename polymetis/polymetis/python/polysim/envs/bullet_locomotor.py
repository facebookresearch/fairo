# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pybullet

from daisy_env.daisy_environments import DaisyBasicEnv, DaisyWalkForward
import daisy_env.configs as DaisyConfigs
from polysim.env.abstract_env import AbstractControlledEnv
from polysim.env.bullet_sim import BulletSimulation


class DaisyLocomotorEnv(AbstractControlledEnv):
    def __init__(
        self,
        hz,
        gui,
        n_dofs,
        robot_state_dim,
        use_controller_manager_py=False,
        control_mode="position",
        joint_limits=[3.14] * 18,
        time_warp=1.0,
        torque_limits=[50.0] * 18,
        init_shoulder_pos=0.0,
        init_elbow_pos=1.57,
        init_height=1.0,
        self_collision=False,
        restitution=0.05,
        lateral_friction=1.5,
        cfg=DaisyConfigs.fast_standing_6legged_config,
        default_controller_args={},
    ):
        AbstractControlledEnv.__init__(
            self,
            hz,
            torque_limits=torque_limits,
            default_controller_args=default_controller_args,
            time_warp=time_warp,
            use_controller_manager_py=use_controller_manager_py,
        )
        self.dt = 1.0 / hz
        self.n_dofs = n_dofs

        self.state_dim = robot_state_dim

        self.robot_state = dict()
        self.robot_state["robot_timestamp"] = 0.0
        self.robot_state["dt"] = self.dt
        self.torque_limits = torque_limits

        cfg["render"] = gui
        cfg["sim_numSolverIterations"] = 100
        cfg["self_collision"] = self_collision
        cfg["sim_timestep"] = self.dt
        cfg["lateralFriction"] = lateral_friction
        cfg["restitution"] = restitution
        cfg["control_mode"] = control_mode
        cfg["joint_limit_motor_1"] = joint_limits[0]
        cfg["joint_limit_motor_2"] = joint_limits[1]
        cfg["joint_limit_motor_3"] = joint_limits[2]
        cfg["initial_height"] = init_height
        init_joint_pos = cfg["initial_joint_positions"]
        init_joint_pos["L_F_motor_2/X8_16"] = init_shoulder_pos
        init_joint_pos["L_M_motor_2/X8_16"] = init_shoulder_pos
        init_joint_pos["L_B_motor_2/X8_16"] = init_shoulder_pos
        init_joint_pos["R_F_motor_2/X8_16"] = -init_shoulder_pos
        init_joint_pos["R_M_motor_2/X8_16"] = -init_shoulder_pos
        init_joint_pos["R_B_motor_2/X8_16"] = -init_shoulder_pos
        init_joint_pos["L_F_motor_3/X8_9"] = -init_elbow_pos
        init_joint_pos["L_M_motor_3/X8_9"] = -init_elbow_pos
        init_joint_pos["L_B_motor_3/X8_9"] = -init_elbow_pos
        init_joint_pos["R_F_motor_3/X8_9"] = init_elbow_pos
        init_joint_pos["R_M_motor_3/X8_9"] = init_elbow_pos
        init_joint_pos["R_B_motor_3/X8_9"] = init_elbow_pos
        cfg["initial_joint_positions"] = init_joint_pos

        self.robot_env = DaisyBasicEnv(**cfg)
        self.robot = self.robot_env.robot

    def _get_states(self):
        self.robot_state["robot_timestamp"] += self.dt
        self.robot_state["dt"] = self.dt

        state = self.robot.calc_state()
        # concatenating the joint pos and base state for controllers like ilqr
        # joint_pos should probably be renamed as generalized coordinates or something
        base_pos = self.get_current_base_pos()
        self.robot_state["joint_positions"] = np.hstack(
            [state["j_pos"], base_pos, state["base_ori_euler"]]
        ).tolist()
        base_vel = self.get_current_base_vel()
        self.robot_state["joint_velocities"] = np.hstack(
            [state["j_vel"], base_vel, state["base_ang_vel"]]
        ).tolist()
        self.robot_state["joint_acc"] = state["j_eff"]  # TODO: fill in this field

        self.robot_state["base_pos"] = self.get_current_base_pos().tolist()
        self.robot_state["base_vel"] = self.get_current_base_vel().tolist()

        self.robot_state["base_ori_euler"] = self.get_current_base_ori_euler().tolist()
        self.robot_state["base_ori_quat"] = self.get_current_base_ori_quat().tolist()
        self.robot_state["base_ang_vel"] = state["base_ang_vel"].tolist()

        self.robot_state["torque_cmd"] = state["j_eff"]  # TODO: fill in this field
        return self.robot_state

    def _apply_control(self, control):
        _, _, _ = self.robot_env.step(control)
        return control

    def set_friction_parameters(self, restitution, lateral_friction, damping):
        body_idx = self.robot.robot_body.bodies[0]
        bullet_client = pybullet  # self.robot._p
        numLinks = bullet_client.getNumJoints(body_idx)
        bullet_client.changeDynamics(
            body_idx, -1, restitution=restitution, lateralFriction=lateral_friction
        )
        for joint_idx in range(numLinks):
            bullet_client.changeDynamics(
                body_idx,
                joint_idx,
                restitution=restitution,
                lateralFriction=lateral_friction,
                jointDamping=damping,
                linearDamping=damping,
                angularDamping=damping,
            )

    def reset(self, state=None):
        state_dict, _, _ = self.robot_env.reset()
        return state_dict

    def get_current_joint_state(self):
        state = self.robot.calc_state()
        return np.hstack([state["j_pos"], state["j_vel"]])

    def get_current_joint_pos(self):
        state = self.robot.calc_state()
        return np.array(state["j_pos"])

    def get_current_joint_vel(self):
        state = self.robot.calc_state()
        return np.array(state["j_vel"])

    def get_current_base_pos(self):
        state = self.robot.calc_state()
        return np.hstack(
            [state["base_pos_x"], state["base_pos_y"], state["base_pos_z"]]
        )

    def get_current_base_vel(self):
        state = self.robot.calc_state()
        return np.hstack([state["base_velocity"]])

    def get_current_full_state(self):
        # full state includes joint pos, joint vel, base pos, base ori euler, base vel, base vel
        state = self.robot.calc_state()
        base_pos = self.get_current_base_pos()
        return np.hstack(
            [
                state["j_pos"],
                base_pos,
                state["base_ori_euler"],
                state["j_vel"],
                state["base_velocity"],
                state["base_ang_vel"],
            ]
        )

    def get_current_base_ori_euler(self):
        state = self.robot.calc_state()
        return np.array(state["base_ori_euler"])

    def get_current_base_ori_quat(self):
        state = self.robot.calc_state()
        return np.array(state["base_ori_quat"])

    def get_current_foot_pos(self):
        state = self.robot.calc_state()
        return np.array(state["foot_pose"])[:, :3]

    def reset_then_step(self, joint_des_state, torque):
        curr_state = self.get_current_full_state()

        joint_pos = joint_des_state[: self.n_dofs]
        base_pos = joint_des_state[self.n_dofs : self.n_dofs + 6]
        base_quat_orient = pybullet.getQuaternionFromEuler(base_pos[3:])
        joint_vel = joint_des_state[self.n_dofs + 6 : 2 * self.n_dofs + 6]
        base_velocity = joint_des_state[self.n_dofs * 2 + 6 :]

        self.robot.parts["daisy"].reset_pose(base_pos[:3], base_quat_orient)
        self.robot.parts["daisy"].reset_velocity(base_velocity)

        for idx, joint in enumerate(self.robot.ordered_joints):
            joint.reset_position(joint_pos[idx], joint_vel[idx])

        # add some checks for simulation inaccuracies

        foot_pos = self.get_current_foot_pos()[:, -1]
        if np.any(foot_pos < -0.01):
            # print("foot step under the ground ", foot_pos)
            return curr_state

        self._apply_control(torque)
        next_state = self.get_current_full_state()

        if (
            np.abs(next_state[20] - curr_state[20]) > 0.003
            or np.abs(next_state[21] - curr_state[21]) > 0.005
        ):
            next_state = curr_state
        if np.abs(next_state[22] - curr_state[22]) > 0.001:
            next_state[22] = curr_state[22]
        return next_state

    def set_robot_state(self, robot_state):
        raise NotImplementedError(
            f"Mirror simulation not implemented for {type(self).__name__}"
        )
