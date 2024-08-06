# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple
from omegaconf import OmegaConf
import os
import numpy as np
import torch
import magnum

from omegaconf import DictConfig
import habitat_sim
import magnum as mn

from polysim.envs import AbstractControlledEnv
from polymetis.utils.data_dir import get_full_path_to_urdf

import torchcontrol as toco


"""
Habitat simulation setup helper functions. Based on the URDF prototype branch:
https://github.com/facebookresearch/habitat-sim/blob/f6267cbfe0ad6c8f86d79edc917a49fb26ddbb73/examples/tutorials/URDF_robotics_tutorial.py
"""


def make_configuration(
    habitat_dir: str,
    glb_path: str,
) -> habitat_sim.SimulatorConfiguration:
    """Create a habitat_sim.SimulatorConfiguration object, and populate it
    with the scene from the glb_path.

    Args:

        habitat_dir: the directory containing habitat-sim. Mainly used as root
            of glb_path (if it's not an absolute path) and the physics config
            file (typically `data/default.physics_config.json`).

        glb_path: path to the .glb file. If a relative path, assumed to be
            relative to `habitat_dir`.

    Returns:
        habitat_sim.SimulatorConfiguration object with reasonable values and
            loaded with glb scene.
    """
    if not os.path.isabs(glb_path):
        glb_path = os.path.join(habitat_dir, glb_path)

    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = glb_path
    backend_cfg.enable_physics = True
    backend_cfg.physics_config_file = os.path.join(
        habitat_dir, backend_cfg.physics_config_file
    )

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [540, 720]
    sensors = {
        "rgba_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def place_agent(
    sim: habitat_sim.Simulator,
    agent_pos: list,
    agent_orient: list,
) -> magnum.Matrix4:
    """Sets AgentState to some reasonable values and return a transformation matrix."""
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = agent_pos
    # agent_state.position = [-0.15, -1.6, 1.0]
    agent_state.rotation = np.quaternion(*agent_orient)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def place_robot_from_agent(
    sim: habitat_sim.Simulator,
    robot: habitat_sim._ext.habitat_sim_bindings.ManagedBulletArticulatedObject,
    local_base_pos: list,
    orientation_vector: list,
    angle_correction: float,
) -> None:
    """Moves robot to reasonable transformation relative to agent."""
    local_base_pos = np.array(local_base_pos)
    # place the robot root state relative to the agent
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    base_transform = mn.Matrix4.rotation(
        mn.Rad(angle_correction), mn.Vector3(*orientation_vector)
    )
    base_transform.translation = agent_transform.transform_point(local_base_pos)
    robot.transformation = base_transform


class HabitatManipulatorEnv(AbstractControlledEnv):
    def __init__(
        self,
        robot_model_cfg: DictConfig,
        habitat_dir: str,
        hz: int = 240,
        joint_damping: float = 0.1,
        grav_comp: bool = True,
        gui: bool = False,
        habitat_scene_path: str = "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
        agent_pos: list = [-0.15, -0.1, 1.0],
        agent_orient: list = [-0.83147, 0, 0.55557, 0],
        local_base_pos: list = [0.0, -1.1, -2.0],
        orientation_vector: list = [1.0, 0, 0],
        angle_correction: float = -1.56,
    ):
        """
        A wrapper around habitat-sim which loads an articulated object from a URDF
        and places it into a scene.

        Args:
            robot_model_cfg: a typical configuration for a robot model (e.g. see
                `franka_panda.yaml`).

            habitat_dir: directory containing habitat-sim.

            hz: the rate at which to update the simulation.

            joint_damping: velocity gains damping joint movement towards 0 velocity.

            grav_comp: whether to enable gravity compensation.

            gui: whether to show a GUI window.

            habitat_scene_path: path to the .glb file containing the scene.
        """
        # Save static parameters
        self.hz = hz
        self.dt = 1.0 / self.hz
        self.gui = gui
        self.n_dofs = robot_model_cfg.num_dofs
        self.grav_comp = grav_comp

        # Save robot model configurations
        self.robot_model_cfg = robot_model_cfg
        self.robot_description_path = get_full_path_to_urdf(
            self.robot_model_cfg.robot_description_path
        )

        # Create Pinocchio model (for gravity compensation)
        self.robot_model = toco.models.RobotModelPinocchio(
            self.robot_description_path, self.robot_model_cfg.ee_link_name
        )

        # Start Habitat simulator
        self.habitat_cfg = make_configuration(habitat_dir, glb_path=habitat_scene_path)
        self.sim = habitat_sim.Simulator(self.habitat_cfg)
        place_agent(self.sim, agent_pos=agent_pos, agent_orient=agent_orient)

        # Load robot
        self.robot = (
            self.sim.get_articulated_object_manager().add_articulated_object_from_urdf(
                self.robot_description_path, fixed_base=True
            )
        )
        assert self.robot is not None
        place_robot_from_agent(
            self.sim,
            self.robot,
            local_base_pos=local_base_pos,
            orientation_vector=orientation_vector,
            angle_correction=angle_correction,
        )

        self.robot.auto_clamp_joint_limits = True

        # Set correct joint damping values
        for motor_id in range(self.n_dofs):
            joint_motor_settings = habitat_sim.physics.JointMotorSettings(
                0.0,  # position_target
                0.0,  # position_gain
                0.0,  # velocity_target
                joint_damping,  # velocity_gain
                0.0,  # max_impulse
            )
            self.robot.update_joint_motor(motor_id, joint_motor_settings)

        self.reset()

        self.prev_torques_commanded = np.zeros(self.n_dofs)
        self.prev_torques_applied = np.zeros(self.n_dofs)
        self.prev_torques_measured = np.zeros(self.n_dofs)
        self.prev_torques_external = np.zeros(self.n_dofs)

    def reset(self, joint_pos: List[float] = None, joint_vel: List[float] = None):
        """Reset joint positions / velocities to given values (0s by default)"""
        if joint_pos is None:
            joint_pos = OmegaConf.to_container(self.robot_model_cfg.rest_pose)
        if joint_vel is None:
            joint_vel = [0.0] * self.n_dofs

        self.robot.joint_positions = joint_pos
        self.robot.joint_velocities = joint_vel

    def get_num_dofs(self) -> int:
        """Return number of degrees of freedom."""
        return self.n_dofs

    def get_current_joint_pos_vel(self) -> Tuple[List[float], List[float]]:
        """Return tuple of joint positions and velocities, both of size `num_dofs`."""
        return self.robot.joint_positions, self.robot.joint_velocities

    def get_current_joint_torques(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns torques: [inputted, clipped, added with gravity compensation, and measured externally]"""
        return (
            self.prev_torques_commanded,
            self.prev_torques_applied,
            self.prev_torques_measured,
            self.prev_torques_external,
        )

    def compute_grav_comp(self, pos: List[float], vel: List[float]) -> np.ndarray:
        """Computes gravity compensation torques using Pinocchio robot model."""
        return (
            self.robot_model.inverse_dynamics(
                torch.tensor(pos),
                torch.tensor([0 for _ in range(self.n_dofs)]),
                torch.tensor([0 for _ in range(self.n_dofs)]),
            )
            .detach()
            .cpu()
            .numpy()
        )

    def render(self):
        if self.gui:
            import cv2

            obs = self.sim.get_sensor_observations()
            img = obs["rgba_camera_1stperson"]

            cv2.namedWindow("Habitat", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Habitat", img)
            cv2.waitKey(1)

    def apply_joint_torques(self, torques: List[float]) -> List[float]:
        """Sets joint torques and steps simulation. Returns applied
        torques (possibly clipped & gravity compensated.)"""
        self.prev_torques_commanded = np.array(torques)
        self.prev_torques_applied = (
            self.prev_torques_commanded
        )  # TODO: apply torque clipping

        pos, vel = self.get_current_joint_pos_vel()

        if self.grav_comp:
            grav_comp_torques = self.compute_grav_comp(pos, vel)
            applied_torques = (np.array(torques) + grav_comp_torques).tolist()
        else:
            applied_torques = torques

        curr_torques = self.robot.joint_forces  # should always be 0 at this point
        if curr_torques != [0, 0, 0, 0, 0, 0, 0]:
            # Extremely important; otherwise occasionally the simulation will
            # put the articulated object to sleep (maybe limit violations?)
            self.robot.awake = True

        self.robot.joint_forces = applied_torques
        self.prev_torques_measured = np.array(applied_torques)
        self.sim.step_physics(self.dt)

        self.render()

        return applied_torques

    def set_robot_state(self, robot_state):
        raise NotImplementedError(
            f"Mirror simulation not implemented for {type(self).__name__}"
        )
