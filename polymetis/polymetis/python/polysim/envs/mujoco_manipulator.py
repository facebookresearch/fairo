# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, List
import logging
import mujoco
import numpy as np
import os

from omegaconf import DictConfig

from polysim.envs import AbstractControlledEnv
from polymetis.utils.data_dir import get_full_path_to_urdf

log = logging.getLogger(__name__)


class MujocoManipulatorEnv(AbstractControlledEnv):
    """A manipulator environment using MuJoCo.

    Args:
        robot_model_cfg: A Hydra configuration file containing information needed for the
                        robot model, e.g. URDF. For an example, see
                        `polymetis/conf/robot_model/franka_panda.yaml`

                        NB: When specifying the path to a URDF file in
                        `robot_description_path`, ensure an MJCF file exists at the
                        same path and same filename with a .mjcf extension. For an
                        example, see `polymetis/data/franka_panda/panda_arm.[urdf|mjcf]`

        gui: Whether to initialize the PyBullet simulation in GUI mode.

        gui_width: Width of GUI window, default 1200

        gui_height: Height of GUI window, default 900

        use_grav_comp: If True, adds gravity compensation torques to the input torques.

        gravity: Value of gravity, default to 9.81
    """

    def __init__(
        self,
        robot_model_cfg: DictConfig,
        gui: bool = False,
        gui_width: int = 1200,
        gui_height: int = 900,
        use_grav_comp: bool = True,
        gravity: float = 9.81,
    ):
        self.robot_model_cfg = robot_model_cfg
        self.robot_description_path = get_full_path_to_urdf(
            self.robot_model_cfg.robot_description_path
        )
        robot_desc_mjcf_path = (
            os.path.splitext(self.robot_description_path)[0] + ".mjcf"
        )
        assert os.path.exists(
            robot_desc_mjcf_path
        ), f"No MJCF file found. Create an MJCF file at {robot_desc_mjcf_path} to use the MuJoCo simulator."
        self.robot_model = mujoco.MjModel.from_xml_path(robot_desc_mjcf_path)
        self.robot_data = mujoco.MjData(self.robot_model)

        self.controlled_joints = self.robot_model_cfg.controlled_joints
        self.n_dofs = self.robot_model_cfg.num_dofs
        assert (
            len(self.controlled_joints) == self.n_dofs
        ), f"Number of controlled joints ({len(self.controlled_joints)}) != number of DOFs ({self.n_dofs})"
        assert (
            self.robot_model.nu == self.n_dofs
        ), f"Number of actuators ({self.robot_model.nu}) != number of DOFs ({self.n_dofs})"

        self.ee_link_idx = self.robot_model_cfg.ee_link_idx
        self.ee_link_name = self.robot_model_cfg.ee_link_name
        self.rest_pose = self.robot_model_cfg.rest_pose
        self.joint_limits_low = np.array(self.robot_model_cfg.joint_limits_low)
        self.joint_limits_high = np.array(self.robot_model_cfg.joint_limits_high)
        if self.robot_model_cfg.joint_damping is None:
            self.joint_damping = None
        else:
            self.joint_damping = np.array(self.robot_model_cfg.joint_damping)
        if self.robot_model_cfg.torque_limits is None:
            self.torque_limits = np.inf * np.ones(self.n_dofs)
        else:
            self.torque_limits = np.array(self.robot_model_cfg.torque_limits)
        self.use_grav_comp = use_grav_comp

        self.prev_torques_commanded = np.zeros(self.n_dofs)
        self.prev_torques_applied = np.zeros(self.n_dofs)
        self.prev_torques_measured = np.zeros(self.n_dofs)
        self.prev_torques_external = np.zeros(self.n_dofs)

        self.gui = gui
        if self.gui:
            # https://mujoco.readthedocs.io/en/latest/programming.html#visualization
            self.gui_width = gui_width
            self.gui_height = gui_height

            self.gui_camera = mujoco.MjvCamera()
            self.gui_opt = mujoco.MjvOption()

            mujoco.glfw.glfw.init()
            self.gui_window = mujoco.glfw.glfw.create_window(
                self.gui_width, self.gui_height, "Mujoco Simulation", None, None
            )
            mujoco.glfw.glfw.make_context_current(self.gui_window)
            mujoco.glfw.glfw.swap_interval(1)

            mujoco.mjv_defaultCamera(self.gui_camera)
            mujoco.mjv_defaultOption(self.gui_opt)

            self.gui_scene = mujoco.MjvScene(self.robot_model, maxgeom=10000)
            self.gui_context = mujoco.MjrContext(
                self.robot_model, mujoco.mjtFontScale.mjFONTSCALE_150.value
            )

    def reset(self, joint_pos: List[float] = None, joint_vel: List[float] = None):
        """Reset the environment."""
        mujoco.mj_resetData(self.robot_model, self.robot_data)
        if joint_pos is None:
            self.robot_data.qpos = self.rest_pose
        if joint_vel is not None:
            self.robot_data.qvel = joint_vel

    def get_num_dofs(self) -> int:
        """Get the number of degrees of freedom for controlling the simulation.

        Returns:
            int: Number of control input dimensions
        """
        return self.n_dofs

    def get_current_joint_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            np.ndarray: Joint positions
            np.ndarray: Joint velocities
        """
        return (
            self.robot_data.qpos,
            self.robot_data.qvel,
        )

    def get_current_joint_torques(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            np.ndarray: Torques received from apply_joint_torques
            np.ndarray: Torques sent to robot (e.g. after clipping)
            np.ndarray: Torques generated by the actuators (e.g. after grav comp)
            np.ndarray: Torques exerted onto the robot
        """
        return (
            self.prev_torques_commanded,
            self.prev_torques_applied,
            self.prev_torques_measured,
            self.prev_torques_external,  # zeros
        )

    def apply_joint_torques(self, torques: np.ndarray):
        """
        input:
            np.ndarray: Desired torques
        Returns:
            np.ndarray: final applied torque
        """
        self.prev_torques_commanded = torques

        applied_torques = np.clip(torques, -self.torque_limits, self.torque_limits)
        self.prev_torques_applied = applied_torques.copy()

        if self.use_grav_comp:
            applied_torques += self.robot_data.qfrc_bias
        self.prev_torques_measured = applied_torques.copy()

        self.robot_data.ctrl = applied_torques
        mujoco.mj_step(self.robot_model, self.robot_data)

        if self.gui:
            self.render()

        return applied_torques

    def render(self):
        viewport = mujoco.MjrRect(0, 0, self.gui_width, self.gui_height)
        mujoco.mjv_updateScene(
            self.robot_model,
            self.robot_data,
            self.gui_opt,
            None,  # no perturbance
            self.gui_camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.gui_scene,
        )
        mujoco.mjr_render(viewport, self.gui_scene, self.gui_context)
        mujoco.glfw.glfw.swap_buffers(self.gui_window)
        mujoco.glfw.glfw.poll_events()

    def set_robot_state(self, robot_state):
        log.warning(
            "set_robot_state is numerically unstable for mujoco_manipulator, proceed with caution...",
        )
        self.robot_data.qpos = robot_state.joint_positions
        self.robot_data.qvel = robot_state.joint_velocities
        self.robot_data.ctrl = self.robot_data.qfrc_bias
        mujoco.mj_step(self.robot_model, self.robot_data)
        if self.gui:
            self.render()
