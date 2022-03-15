# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import numpy
import torch
import torchcontrol as toco
from polymetis import GripperInterface, RobotInterface
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T

DEFAULT_MAX_ITERS = 3

# Sampling params
DEFAULT_GP_RANGE_UPPER = [0.7, 0.1, numpy.pi / 2]
DEFAULT_GP_RANGE_LOWER = [0.4, -0.1, -numpy.pi / 2]

# Grasp params
DEFAULT_REST_POSE = ([0.5, 0.0, 0.7], [1.0, 0.0, 0.0, 0.0])
DEFAULT_PREGRASP_HEIGHT = 0.55
DEFAULT_GRASP_HEIGHT = 0.15
DEFAULT_PLANNER_DT = 0.02
DEFAULT_TIME_TO_GO_SECS = 2.0

DEFAULT_GAIN_MULTIPLIER = 1.0

CARTESIAN_SPACE_CONTROLLER = 1
JOINT_SPACE_CONTROLLER = 2


class ManipulatorSystem:
    def __init__(self, robot_kwargs={}, gripper_kwargs={}):
        ip_address = robot_kwargs.get("ip_address", "localhost")
        enforce_version = robot_kwargs.get("enforce_version", "True") == "True"
        gripper_ip_address = gripper_kwargs.get("gripper_ip_address", "localhost")
        self.arm = RobotInterface(
            ip_address=ip_address, enforce_version=enforce_version
        )
        self.gripper = GripperInterface(ip_address=gripper_ip_address)
        self._rest_pose = DEFAULT_REST_POSE
        self._pregrasp_height = DEFAULT_PREGRASP_HEIGHT
        self._grasp_height = DEFAULT_GRASP_HEIGHT
        self._planner_dt = DEFAULT_PLANNER_DT
        self._gp_range_upper = torch.Tensor(DEFAULT_GP_RANGE_UPPER)
        self._gp_range_lower = torch.Tensor(DEFAULT_GP_RANGE_LOWER)
        self._up_time_seconds = DEFAULT_TIME_TO_GO_SECS
        self._down_time_seconds = DEFAULT_TIME_TO_GO_SECS
        self._gain_multiplier = DEFAULT_GAIN_MULTIPLIER
        self._controller_type = JOINT_SPACE_CONTROLLER
        time.sleep(0.5)

        # Set continuous control policy
        self.reset_policy()

        # Reset to rest pose
        self.rest_pos = torch.Tensor(self._rest_pose[0])
        self.rest_quat = torch.Tensor(self._rest_pose[1])
        self.reset()

    def __del__(self):
        self.arm.terminate_current_policy()

    def reset(self, time_to_go=DEFAULT_TIME_TO_GO_SECS):
        self.move_to(self.rest_pos, self.rest_quat, time_to_go=time_to_go)
        self.open_gripper()

    def reset_policy(self):
        # Go home
        self.arm.go_home()

        # Send PD controller
        if self._controller_type == CARTESIAN_SPACE_CONTROLLER:
            joint_pos_current = self.arm.get_joint_positions()
            policy = toco.policies.CartesianImpedanceControl(
                joint_pos_current=joint_pos_current,
                Kp=(self._gain_multiplier * torch.Tensor(self.arm.metadata.default_Kx)),
                Kd=(
                    self._gain_multiplier * torch.Tensor(self.arm.metadata.default_Kxd)
                ),
                robot_model=self.arm.robot_model,
            )
            self.arm.send_torch_policy(policy, blocking=False)
        elif self._controller_type == JOINT_SPACE_CONTROLLER:
            joint_pos_current = self.arm.get_joint_positions()
            policy = toco.policies.JointImpedanceControl(
                joint_pos_current=joint_pos_current,
                Kp=(self._gain_multiplier * torch.Tensor(self.arm.metadata.default_Kq)),
                Kd=(
                    self._gain_multiplier * torch.Tensor(self.arm.metadata.default_Kqd)
                ),
                robot_model=self.arm.robot_model,
            )
            self.arm.send_torch_policy(policy, blocking=False)
        else:
            raise Exception(f"Invalid controller type {self._controller_type}")

    def move_to(
        self, pos, quat, gather_arm_state_func=None, time_to_go=DEFAULT_TIME_TO_GO_SECS
    ):
        """
        Attempts to move to the given position and orientation by
        planning a Cartesian trajectory (a set of min-jerk waypoints)
        and updating the current policy's target to that
        end-effector position & orientation.

        Returns (num successes, num attempts, robot_states)
        """
        # Plan trajectory
        N = int(time_to_go / self._planner_dt)

        if self._controller_type == CARTESIAN_SPACE_CONTROLLER:
            pos_curr, quat_curr = self.arm.get_ee_pose()
            waypoints = toco.planning.generate_cartesian_space_min_jerk(
                start=T.from_rot_xyz(R.from_quat(quat_curr), pos_curr),
                goal=T.from_rot_xyz(R.from_quat(quat), pos),
                time_to_go=time_to_go,
                hz=1 / self._planner_dt,
            )
        elif self._controller_type == JOINT_SPACE_CONTROLLER:
            joint_pos_current = self.arm.get_joint_positions()
            waypoints = toco.planning.generate_cartesian_target_joint_min_jerk(
                joint_pos_start=joint_pos_current,
                ee_pose_goal=T.from_rot_xyz(R.from_quat(quat), pos),
                time_to_go=time_to_go,
                hz=1 / self._planner_dt,
                robot_model=self.arm.robot_model,
            )
        else:
            raise Exception(f"Invalid controller type {self._controller_type}")

        # Execute trajectory
        t0 = time.time()
        t_target = t0
        successes = 0
        error_detected = False
        robot_states = []
        for i in range(N):
            # Update traj
            try:
                if self._controller_type == CARTESIAN_SPACE_CONTROLLER:
                    ee_pos_desired = waypoints[i]["pose"].translation()
                    ee_quat_desired = waypoints[i]["pose"].rotation().as_quat()
                    # ee_twist_desired = waypoints[i]["twist"]
                    self.arm.update_current_policy(
                        {
                            "ee_pos_desired": ee_pos_desired,
                            "ee_quat_desired": ee_quat_desired,
                            # "ee_vel_desired": ee_twist_desired[:3],
                            # "ee_rvel_desired": ee_twist_desired[3:],
                        }
                    )
                    if gather_arm_state_func:
                        observed_state = gather_arm_state_func(
                            self.arm.get_robot_state()
                        )
                        observed_state["ee_pos_desired"] = ee_pos_desired
                        observed_state["ee_quat_desired"] = ee_quat_desired
                        robot_states.append(observed_state)
                elif self._controller_type == JOINT_SPACE_CONTROLLER:
                    joint_pos_desired = waypoints[i]["position"]
                    self.arm.update_current_policy(
                        {
                            "joint_pos_desired": joint_pos_desired,
                        }
                    )
                    if gather_arm_state_func:
                        observed_state = gather_arm_state_func(
                            self.arm.get_robot_state()
                        )
                        observed_state["joint_pos_desired"] = joint_pos_desired
                        robot_states.append(observed_state)
                else:
                    raise Exception(f"Invalid controller type {self._controller_type}")
            except Exception as e:
                error_detected = True
                print(f"Error updating current policy {str(e)}")

            # Check if policy terminated due to issues
            if self.arm.get_previous_interval().end != -1 or error_detected:
                error_detected = False
                print("Interrupt detected. Reinstantiating control policy...")
                time.sleep(3)
                self.reset_policy()
                break
            else:
                successes += 1

            # Spin once
            t_target += self._planner_dt
            t_remaining = t_target - time.time()
            time.sleep(max(t_remaining, 0.0))

        # Wait for robot to stabilize
        time.sleep(0.2)

        return successes, N, robot_states

    def close_gripper(self):
        self.gripper.grasp(speed=0.1, force=1.0)
        time.sleep(1.0)

        # Check state
        state = self.gripper.get_state()
        assert state.width < state.max_width
        return state

    def open_gripper(self):
        max_width = self.gripper.get_state().max_width
        self.gripper.goto(width=max_width, speed=0.1, force=1.0)
        time.sleep(0.5)

        # Check state
        state = self.gripper.get_state()
        assert state.width > 0.0
        return state

    def grasp_pose_to_pos_quat(self, grasp_pose, z):
        x, y, rz = grasp_pose
        pos = torch.Tensor([x, y, z])
        quat = (
            R.from_rotvec(torch.Tensor([0, 0, rz])) * R.from_quat(self.rest_quat)
        ).as_quat()

        return pos, quat

    def grasp(
        self,
        grasp_pose0,
        grasp_pose1,
        gather_arm_state_func=None,
        gather_gripper_state_func=None,
        reset_at_end=True,
    ):
        results = []
        traj_state = {}

        traj_state["grasp_pose"] = grasp_pose0

        # Move to pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, self._pregrasp_height)
        successes, N, robot_states = self.move_to(
            pos, quat, gather_arm_state_func, time_to_go=self._up_time_seconds
        )
        results.append((successes, N))
        traj_state["move_to_grasp_states"] = robot_states

        # Lower (slower than other motions to prevent sudden collisions)
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, self._grasp_height)
        successes, N, robot_states = self.move_to(
            pos, quat, gather_arm_state_func, time_to_go=self._down_time_seconds
        )
        results.append((successes, N))
        traj_state["lower_to_grasp_states"] = robot_states

        # Grasp
        close_gripper_state = self.close_gripper()
        if gather_gripper_state_func:
            traj_state["close_gripper_state"] = gather_gripper_state_func(
                close_gripper_state
            )

        # Lift to pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, self._pregrasp_height)
        successes, N, robot_states = self.move_to(
            pos, quat, gather_arm_state_func, time_to_go=self._up_time_seconds
        )
        results.append((successes, N))
        traj_state["lift_to_release_states"] = robot_states

        # Move to new pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose1, self._pregrasp_height)
        successes, N, robot_states = self.move_to(
            pos, quat, gather_arm_state_func, time_to_go=self._up_time_seconds
        )
        results.append((successes, N))
        traj_state["move_to_release_states"] = robot_states

        # Read gripper state prior to release, in case object had fallen down
        if gather_gripper_state_func:
            preopen_gripper_state = self.gripper.get_state()
            traj_state["preopen_gripper_state"] = gather_gripper_state_func(
                preopen_gripper_state
            )

        # Release
        open_gripper_state = self.open_gripper()
        if gather_gripper_state_func:
            traj_state["open_gripper_state"] = gather_gripper_state_func(
                open_gripper_state
            )

        traj_state["release_pose"] = grasp_pose1

        # Reset
        if reset_at_end:
            self.reset()

        total_successes = sum([r[0] for r in results])
        total_tries = sum([r[1] for r in results])
        return total_successes, total_tries, traj_state

    def continuously_grasp(self, max_iters=1000):
        # Perform grasping
        i = 0
        total_successes, total_tries = 0, 0
        try:
            while True:
                # Sample grasp
                grasp_pose0 = uniform_sample(self._gp_range_lower, self._gp_range_upper)
                grasp_pose1 = uniform_sample(self._gp_range_lower, self._gp_range_upper)

                # Perform grasp
                print(f"Grasp {i + 1}: grasp={grasp_pose0}, release={grasp_pose1}")
                n_successes, n_tries, _ = self.grasp(grasp_pose0, grasp_pose1)
                total_successes += n_successes
                total_tries += n_tries

                # Loop termination
                i += 1
                if max_iters > 0 and i >= max_iters:
                    break

        except KeyboardInterrupt:
            print("Interrupted by user.")

        return total_successes, total_tries

    @property
    def pregrasp_height(self) -> float:
        return self._pregrasp_height

    @pregrasp_height.setter
    def pregrasp_height(self, pregrasp_height: float):
        self._pregrasp_height = pregrasp_height

    @property
    def grasp_height(self) -> float:
        return self._grasp_height

    @grasp_height.setter
    def grasp_height(self, grasp_height: float):
        self._grasp_height = grasp_height

    @property
    def planner_dt(self) -> float:
        return self._planner_dt

    @planner_dt.setter
    def planner_dt(self, planner_dt: float):
        self._planner_dt = planner_dt

    @property
    def gp_range_upper(self) -> torch.Tensor:
        return self._gp_range_upper

    @gp_range_upper.setter
    def gp_range_upper(self, gp_range_upper: torch.Tensor):
        self._gp_range_upper = gp_range_upper

    @property
    def gp_range_lower(self) -> torch.Tensor:
        return self._gp_range_lower

    @gp_range_lower.setter
    def gp_range_lower(self, gp_range_lower: torch.Tensor):
        self._gp_range_lower = gp_range_lower

    @property
    def up_time_seconds(self) -> float:
        return self._up_time_seconds

    @up_time_seconds.setter
    def up_time_seconds(self, up_time_seconds: float):
        self._up_time_seconds = up_time_seconds

    @property
    def down_time_seconds(self) -> float:
        return self._down_time_seconds

    @down_time_seconds.setter
    def down_time_seconds(self, down_time_seconds: float):
        self._down_time_seconds = down_time_seconds

    @property
    def gain_multiplier(self) -> float:
        return self._gain_multiplier

    @gain_multiplier.setter
    def gain_multiplier(self, gain_multiplier: float):
        self._gain_multiplier = gain_multiplier

    @property
    def controller_type(self) -> int:
        return self._controller_type

    @controller_type.setter
    def controller_type(self, type: int):
        self._controller_type = type


def uniform_sample(lower, upper):
    return lower + (upper - lower) * torch.rand_like(lower)
