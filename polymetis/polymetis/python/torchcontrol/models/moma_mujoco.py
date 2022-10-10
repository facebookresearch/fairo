# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from code provided by https://github.com/AlexanderKhazatsky
# TODO: Is this the right way to acknowledge
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

try:
    import dm_robotics, dm_control
except ImportError:
    log.warning(
        f"If looking for Mujoco-based IK, please install Polymetis using '[mj_ik]'."
    )
    sys.exit(1)
from dm_control import mjcf
from dm_robotics.moma.effectors import arm_effector, cartesian_6d_velocity_effector
from dm_robotics.moma.models.robots.robot_arms import robot_arm


def quat_diff(target, source):
    return (R.from_quat(source).inv() * R.from_quat(target)).as_rotvec()


class MjcfArm(robot_arm.RobotArm):
    def __init__(self, mjcf_model):
        self._mjcf_root = mjcf_model

    def _build(self, model_file):
        # Find MJCF elements that will be exposed as attributes.
        self._joints = self._mjcf_root.find_all("joint")
        self._bodies = self.mjcf_model.find_all("body")
        self._actuators = self.mjcf_model.find_all("actuator")
        self._wrist_site = self.mjcf_model.find("site", "wrist_site")
        self._base_site = self.mjcf_model.find("site", "base_site")

    def name(self) -> str:
        return self._name

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    def update_state(
        self, physics: mjcf.Physics, qpos: np.ndarray, qvel: np.ndarray
    ) -> None:
        physics.bind(self._joints).qpos[:] = qpos
        physics.bind(self._joints).qvel[:] = qvel

    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        physics.bind(self._joints).qpos[:] = qpos

    @property
    def base_site(self):
        return self._base_site

    @property
    def wrist_site(self):
        return self._wrist_site


class MujocoModel:
    def __init__(self, cfg):
        # Load mjcf file
        mjcf_model = mjcf.from_path(cfg.xml_path)

        # Initialize physics
        self._arm = MjcfArm(mjcf_model)
        self._physics = mjcf.Physics.from_mjcf_model(mjcf_model)

        # Set up effector (IK) configs
        effector = arm_effector.ArmEffector(
            arm=self._arm, action_range_override=None, robot_name=cfg.name
        )
        effector_model = cartesian_6d_velocity_effector.ModelParams(
            self._arm.wrist_site, self._arm.joints
        )
        # TODO: put into configs
        control_hz = 20
        scaler = 0.1
        effector_control = cartesian_6d_velocity_effector.ControlParams(
            control_timestep_seconds=1 / control_hz,
            max_lin_vel=1.0,
            max_rot_vel=1.0,
            joint_velocity_limits=np.array([2.075 * scaler] * 4 + [2.51 * scaler] * 3),
            nullspace_gain=0.025,
            # nullspace_joint_position_reference=[0 for i in range(7)], <- does this by default
            regularization_weight=1e-2,  # 1e-2
            enable_joint_position_limits=True,
            minimum_distance_from_joint_position_limit=0.3,  # 0.01
            joint_position_limit_velocity_scale=0.95,
            max_cartesian_velocity_control_iterations=300,
            max_nullspace_control_iterations=300,
        )
        self._cart_effector_6d = (
            cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
                cfg.name, effector, effector_model, effector_control
            )
        )

    def local_inverse_kinematics(
        self,
        ee_pos_desired,
        ee_quat_desired,
        ee_pos_current,
        ee_quat_current,
        joint_pos_current,
        joint_vel_current,
    ):
        lin_vel = ee_pos_desired - ee_pos_current
        rot_vel = quat_diff(ee_quat_desired, ee_quat_current)

        action = np.concatenate([lin_vel, rot_vel])
        self._arm.update_state(self._physics, joint_pos_current, joint_vel_current)
        self._cart_effector_6d.set_control(self._physics, action)
        joint_vel_ctrl = self._physics.bind(self._arm.actuators).ctrl.copy()

        desired_qpos = joint_pos_current + joint_vel_ctrl
        success = np.any(joint_vel_ctrl)  # I think it returns zeros when it fails

        return torch.Tensor(desired_qpos), success
