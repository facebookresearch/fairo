# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple
import time
import tempfile
import logging
from dataclasses import dataclass

import grpc  # This requires `conda install grpcio protobuf`
import torch

import polymetis
from polymetis.base_interface import BaseRobotInterface
from polymetis_pb2 import LogInterval, RobotState, ControllerChunk, Empty
from polymetis_pb2_grpc import PolymetisControllerServerStub

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T

log = logging.getLogger(__name__)


class RobotInterface(BaseRobotInterface):
    """
    Adds user-friendly helper methods to automatically construct some policies
    with sane defaults.

    Args:
        time_to_go_default: Default amount of time for policies to run, if not given.

        use_grav_comp: If True, assumes that gravity compensation torques are added
                       to the given torques.

    """

    def __init__(
        self,
        time_to_go_default: float = 1.0,
        use_grav_comp: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        with tempfile.NamedTemporaryFile("w+") as urdf_file:
            urdf_file.write(self.metadata.urdf_file)
            urdf_file.flush()
            self.set_robot_model(urdf_file.name, self.metadata.ee_link_name)

        self.set_home_pose(torch.Tensor(self.metadata.rest_pose))

        self.Kq_default = torch.Tensor(self.metadata.default_Kq)
        self.Kqd_default = torch.Tensor(self.metadata.default_Kqd)
        self.Kx_default = torch.Tensor(self.metadata.default_Kx)
        self.Kxd_default = torch.Tensor(self.metadata.default_Kxd)
        self.hz = self.metadata.hz

        self.time_to_go_default = time_to_go_default

        self.use_grav_comp = use_grav_comp

    def _adaptive_time_to_go(self, joint_displacement: torch.Tensor):
        """Compute adaptive time_to_go
        Computes the corresponding time_to_go such that the mean velocity is equal to one-eighth
        of the joint velocity limit:
        time_to_go = max_i(joint_displacement[i] / (joint_velocity_limit[i] / 8))
        (Note 1: The magic number 8 is deemed reasonable from hardware tests on a Franka Emika.)
        (Note 2: In a min-jerk trajectory, maximum velocity is equal to 1.875 * mean velocity.)

        The resulting time_to_go is also clipped to a minimum value of the default time_to_go.
        """
        joint_vel_limits = self.robot_model.get_joint_velocity_limits()
        joint_pos_diff = torch.abs(joint_displacement)
        time_to_go = torch.max(joint_pos_diff / joint_vel_limits * 8.0)
        return max(time_to_go, self.time_to_go_default)

    def solve_inverse_kinematics(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor,
        q0: torch.Tensor,
        tol: float = 1e-3,
    ) -> Tuple[torch.Tensor, bool]:
        """Compute inverse kinematics given desired EE pose"""
        # Call IK
        joint_pos_output = self.robot_model.inverse_kinematics(
            position, orientation, rest_pose=q0
        )

        # Check result
        pos_output, quat_output = self.robot_model.forward_kinematics(joint_pos_output)
        pose_desired = T.from_rot_xyz(R.from_quat(orientation), position)
        pose_output = T.from_rot_xyz(R.from_quat(quat_output), pos_output)
        err = torch.linalg.norm((pose_desired * pose_output.inv()).as_twist())
        ik_sol_found = err < tol

        return joint_pos_output, ik_sol_found

    """
    Setter methods
    """

    def set_home_pose(self, home_pose: torch.Tensor):
        """Sets the home pose for `go_home()` to use."""
        self.home_pose = home_pose

    def set_robot_model(self, robot_description_path: str, ee_link_name: str = None):
        """Loads the URDF as a RobotModelPinocchio."""
        # Create Torchscript Pinocchio model for DynamicsControllers
        self.robot_model = toco.models.RobotModelPinocchio(
            robot_description_path, ee_link_name
        )

    """
    Getter methods
    """

    def get_joint_positions(self) -> torch.Tensor:
        return torch.Tensor(self.get_robot_state().joint_positions)

    def get_joint_velocities(self) -> torch.Tensor:
        return torch.Tensor(self.get_robot_state().joint_velocities)

    """
    End-effector computation methods
    """

    def get_ee_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward kinematics on the current joint angles.

        Returns:
            torch.Tensor: 3D end-effector position
            torch.Tensor: 4D end-effector orientation as quaternion
        """
        joint_pos = self.get_joint_positions()
        pos, quat = self.robot_model.forward_kinematics(joint_pos)
        return pos, quat

    def get_jacobian(joint_angles):
        raise NotImplementedError  # TODO

    """
    Movement methods
    """

    def move_to_joint_positions(
        self,
        positions: torch.Tensor,
        time_to_go: float = None,
        delta: bool = False,
        Kq: torch.Tensor = None,
        Kqd: torch.Tensor = None,
        **kwargs,
    ) -> List[RobotState]:
        """Uses JointGoToPolicy to move to the desired positions with the given gains.
        Args:
            positions: Desired target joint positions.
            time_to_go: Amount of time to execute the motion. Uses an adaptive value if not specified (see `_adaptive_time_to_go` for details).
            delta: Whether the specified `positions` are relative to current pose or absolute.
            Kq: Joint P gains for the tracking controller. Uses default values if not specified.
            Kqd: Joint D gains for the tracking controller. Uses default values if not specified.

        Returns:
            Same as `send_torch_policy`
        """
        assert (
            self.robot_model is not None
        ), "Robot model not assigned! Call 'set_robot_model(<path_to_urdf>, <ee_link_name>)' to enable use of dynamics controllers"

        # Parse parameters
        joint_pos_current = self.get_joint_positions()
        joint_pos_desired = torch.Tensor(positions)
        if delta:
            joint_pos_desired += joint_pos_current

        time_to_go_adaptive = self._adaptive_time_to_go(
            joint_pos_desired - joint_pos_current
        )
        if time_to_go is None:
            time_to_go = time_to_go_adaptive
        elif time_to_go < time_to_go_adaptive:
            log.warn(
                "The specified 'time_to_go' might not be large enough to ensure accurate movement."
            )

        # Plan trajectory
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=joint_pos_current,
            goal=joint_pos_desired,
            time_to_go=time_to_go,
            hz=self.hz,
        )

        # Create & execute policy
        torch_policy = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
            joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
            Kq=self.Kq_default if Kq is None else Kq,
            Kqd=self.Kqd_default if Kqd is None else Kqd,
            Kx=self.Kx_default,
            Kxd=self.Kxd_default,
            robot_model=self.robot_model,
            ignore_gravity=self.use_grav_comp,
        )

        return self.send_torch_policy(torch_policy=torch_policy, **kwargs)

    def go_home(self, *args, **kwargs) -> List[RobotState]:
        """Calls move_to_joint_positions to the current home positions."""
        assert (
            self.home_pose is not None
        ), "Home pose not assigned! Call 'set_home_pose(<joint_angles>)' to enable homing"
        return self.move_to_joint_positions(
            positions=self.home_pose, delta=False, *args, **kwargs
        )

    def move_to_ee_pose(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor = None,
        time_to_go: float = None,
        delta: bool = False,
        Kx: torch.Tensor = None,
        Kxd: torch.Tensor = None,
        op_space_interp: bool = True,
        **kwargs,
    ) -> List[RobotState]:
        """Uses an operational space controller to move to a desired end-effector position (and, optionally orientation).
        Args:
            positions: Desired target end-effector position.
            positions: Desired target end-effector orientation (quaternion).
            time_to_go: Amount of time to execute the motion. Uses an adaptive value if not specified (see `_adaptive_time_to_go` for details).
            delta: Whether the specified `position` and `orientation` are relative to current pose or absolute.
            Kx: P gains for the tracking controller. Uses default values if not specified.
            Kxd: D gains for the tracking controller. Uses default values if not specified.
            op_space_interp: Interpolate trajectory in operational space, resulting in a straight line in 3D space instead of the shortest path in joint movement space.

        Returns:
            Same as `send_torch_policy`
        """
        assert (
            self.robot_model is not None
        ), "Robot model not assigned! Call 'set_robot_model(<path_to_urdf>, <ee_link_name>)' to enable use of dynamics controllers"

        joint_pos_current = self.get_joint_positions()
        ee_pos_current, ee_quat_current = self.get_ee_pose()

        # Parse parameters
        ee_pos_desired = torch.Tensor(position)
        if delta:
            ee_pos_desired += ee_pos_current

        if orientation is None:
            ee_quat_desired = ee_quat_current
        else:
            assert (
                len(orientation) == 4
            ), "Only quaternions are accepted as orientation inputs."
            ee_quat_desired = torch.Tensor(orientation)
            if delta:
                ee_quat_desired = (
                    R.from_quat(ee_quat_desired) * R.from_quat(ee_quat_current)
                ).as_quat()

        # Compute joint space target
        joint_pos_desired, success = self.solve_inverse_kinematics(
            ee_pos_desired, ee_quat_desired, joint_pos_current
        )
        if not success:
            log.warning(
                "Unable to find valid joint target. Skipping move_to_ee_pose command..."
            )
            return []

        # Compute adaptive time_to_go
        if time_to_go is None:
            time_to_go_adaptive = self._adaptive_time_to_go(
                joint_pos_desired - joint_pos_current
            )
            time_to_go = time_to_go_adaptive

        # Generate & run policy
        if op_space_interp:
            # Compute operational space trajectory
            ee_pose_desired = T.from_rot_xyz(
                rotation=R.from_quat(ee_quat_desired), translation=ee_pos_desired
            )
            waypoints = toco.planning.generate_cartesian_target_joint_min_jerk(
                joint_pos_start=joint_pos_current,
                ee_pose_goal=ee_pose_desired,
                time_to_go=time_to_go,
                hz=self.hz,
                robot_model=self.robot_model,
                home_pose=self.home_pose,
            )

            # Create joint tracking policy and run
            torch_policy = toco.policies.JointTrajectoryExecutor(
                joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
                joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
                Kq=self.Kq_default,
                Kqd=self.Kqd_default,
                Kx=self.Kx_default if Kx is None else Kx,
                Kxd=self.Kxd_default if Kxd is None else Kxd,
                robot_model=self.robot_model,
                ignore_gravity=self.use_grav_comp,
            )

            return self.send_torch_policy(torch_policy=torch_policy, **kwargs)

        else:
            # Use joint space controller to move to joint target
            return self.move_to_joint_positions(
                joint_pos_desired, time_to_go=time_to_go
            )

    """
    Continuous control methods
    """

    def start_joint_impedance(self, Kq=None, Kqd=None, adaptive=True, **kwargs):
        """Starts joint position control mode.
        Runs an non-blocking joint impedance controller.
        The desired joint positions can be updated using `update_desired_joint_positions`
        """
        if adaptive:
            torch_policy = toco.policies.HybridJointImpedanceControl(
                joint_pos_current=self.get_joint_positions(),
                Kq=self.Kq_default if Kq is None else Kq,
                Kqd=self.Kqd_default if Kqd is None else Kqd,
                Kx=self.Kx_default,
                Kxd=self.Kxd_default,
                robot_model=self.robot_model,
                ignore_gravity=self.use_grav_comp,
            )
        else:
            torch_policy = toco.policies.JointImpedanceControl(
                joint_pos_current=self.get_joint_positions(),
                Kp=self.Kq_default if Kq is None else Kq,
                Kd=self.Kqd_default if Kqd is None else Kqd,
                robot_model=self.robot_model,
                ignore_gravity=self.use_grav_comp,
            )

        return self.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def start_cartesian_impedance(self, Kx=None, Kxd=None, **kwargs):
        """Starts Cartesian position control mode.
        Runs an non-blocking Cartesian impedance controller.
        The desired EE pose can be updated using `update_desired_ee_pose`
        """
        torch_policy = toco.policies.HybridJointImpedanceControl(
            joint_pos_current=self.get_joint_positions(),
            Kq=self.Kq_default,
            Kqd=self.Kqd_default,
            Kx=self.Kx_default if Kx is None else Kx,
            Kxd=self.Kxd_default if Kxd is None else Kxd,
            robot_model=self.robot_model,
            ignore_gravity=self.use_grav_comp,
        )

        return self.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def update_desired_joint_positions(self, positions: torch.Tensor) -> int:
        """Update the desired joint positions used by the joint position control mode.
        Requires starting a joint impedance controller with `start_joint_impedance` beforehand.
        """
        try:
            update_idx = self.update_current_policy({"joint_pos_desired": positions})
        except grpc.RpcError as e:
            log.error(
                "Unable to update desired joint positions. Use 'start_joint_impedance' to start a joint impedance controller."
            )
            raise e

        return update_idx

    def update_desired_ee_pose(
        self,
        position: torch.Tensor = None,
        orientation: torch.Tensor = None,
    ) -> int:
        """Update the desired EE pose used by the Cartesian position control mode.
        Requires starting a Cartesian impedance controller with `start_cartesian_impedance` beforehand.
        """
        joint_pos_current = self.get_joint_positions()
        ee_pos_current, ee_quat_current = self.get_ee_pose()
        ee_pos_desired = ee_pos_current if position is None else position
        ee_quat_desired = ee_quat_current if orientation is None else orientation

        joint_pos_desired, success = self.solve_inverse_kinematics(
            ee_pos_desired, ee_quat_desired, joint_pos_current
        )
        if not success:
            log.warning(
                "Unable to find valid joint target. Skipping update_desired_ee_pose command..."
            )
            return -1

        return self.update_desired_joint_positions(joint_pos_desired)

    def start_joint_velocity_control(
        self, joint_vel_desired, hz=None, Kq=None, Kqd=None, **kwargs
    ):
        """Starts joint velocity control mode.
        Runs a non-blocking joint velocity controller.
        The desired joint velocities can be updated using `update_desired_joint_velocities`
        """
        torch_policy = toco.policies.JointVelocityControl(
            joint_vel_desired=joint_vel_desired,
            Kp=self.Kq_default if Kq is None else Kq,
            Kd=self.Kqd_default if Kqd is None else Kqd,
            robot_model=self.robot_model,
            hz=self.metadata.hz if hz is None else hz,
            ignore_gravity=self.use_grav_comp,
        )

        return self.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def update_desired_joint_velocities(self, velocities: torch.Tensor):
        """Update the desired joint velocities used by the joint velocities control mode.
        Requires starting a joint velocities controller with `start_joint_velocity_control` beforehand.
        """
        try:
            update_idx = self.update_current_policy({"joint_vel_desired": velocities})
        except grpc.RpcError as e:
            log.error(
                "Unable to update desired joint velocities. Use 'start_joint_velocity_control' to start a joint velocities controller."
            )
            raise e

        return update_idx

    """
    PyRobot backward compatibility methods
    """

    def get_joint_angles(self) -> torch.Tensor:
        """Functionally identical to `get_joint_positions`.
        **This method is being deprecated in favor of `get_joint_positions`.**
        """
        log.warning(
            "The method 'get_joint_angles' is deprecated, use 'get_joint_positions' instead."
        )
        return self.get_joint_positions()

    def pose_ee(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Functionally identical to `get_ee_pose`.
        **This method is being deprecated in favor of `get_ee_pose`.**
        """
        log.warning("The method 'pose_ee' is deprecated, use 'get_ee_pose' instead.")
        return self.get_ee_pose()

    def set_joint_positions(
        self, desired_positions, *args, **kwargs
    ) -> List[RobotState]:
        """Functionally identical to `move_to_joint_positions`.
        **This method is being deprecated in favor of `move_to_joint_positions`.**
        """
        log.warning(
            "The method 'set_joint_positions' is deprecated, use 'move_to_joint_positions' instead."
        )
        return self.move_to_joint_positions(
            positions=desired_positions, *args, **kwargs
        )

    def move_joint_positions(
        self, delta_positions, *args, **kwargs
    ) -> List[RobotState]:
        """Functionally identical to calling `move_to_joint_positions` with the argument `delta=True`.
        **This method is being deprecated in favor of `move_to_joint_positions`.**
        """
        log.warning(
            "The method 'set_joint_positions' is deprecated, use 'move_to_joint_positions' with 'delta=True' instead."
        )
        return self.move_to_joint_positions(
            positions=delta_positions, delta=True, *args, **kwargs
        )

    def set_ee_pose(self, *args, **kwargs) -> List[RobotState]:
        """Functionally identical to `move_to_ee_pose`.
        **This method is being deprecated in favor of `move_to_ee_pose`.**
        """
        log.warning(
            "The method 'set_ee_pose' is deprecated, use 'move_to_ee_pose' instead."
        )
        return self.move_to_ee_pose(*args, **kwargs)

    def move_ee_xyz(
        self, displacement: torch.Tensor, use_orient: bool = True, **kwargs
    ) -> List[RobotState]:
        """Functionally identical to calling `move_to_ee_pose` with the argument `delta=True`.
        **This method is being deprecated in favor of `move_to_ee_pose`.**
        """
        log.warning(
            "The method 'move_ee_xyz' is deprecated, use 'move_to_ee_pose' with 'delta=True' instead."
        )
        return self.move_to_ee_pose(position=displacement, delta=True, **kwargs)
