# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import a0
import json
import numpy as np
import os
import pickle
import platform
import time
import torch
import torchcontrol as toco

from polymetis import RobotInterface, GripperInterface
from polymetis_pb2 import RobotState, GripperState
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from typing import Dict, List


DEFAULT_MAX_ITERS = 3

# Sampling params
DEFAULT_GP_RANGE_UPPER = [0.7, 0.1, np.pi / 2]
DEFAULT_GP_RANGE_LOWER = [0.4, -0.1, -np.pi / 2]

# Grasp params
DEFAULT_REST_POSE = ([0.5, 0.0, 0.7], [1.0, 0.0, 0.0, 0.0])
DEFAULT_PREGRASP_HEIGHT = 0.55
DEFAULT_GRASP_HEIGHT = 0.15
DEFAULT_PLANNER_DT = 0.02
DEFAULT_TIME_TO_GO_SECS = 2.0

class ManipulatorSystem:
    def __init__(self, robot_kwargs={}, gripper_kwargs={}):
        ip_address = robot_kwargs.get("ip_address", "localhost")
        enforce_version = robot_kwargs.get("enforce_version", "True") == "True"
        gripper_ip_address = gripper_kwargs.get("gripper_ip_address", "localhost")
        self.arm = RobotInterface(ip_address=ip_address, enforce_version=enforce_version)
        self.gripper = GripperInterface(ip_address=gripper_ip_address)
        self.rest_pose = DEFAULT_REST_POSE
        self.pregrasp_height = DEFAULT_PREGRASP_HEIGHT
        self.grasp_height = DEFAULT_GRASP_HEIGHT
        self.planner_dt = DEFAULT_PLANNER_DT
        self.gp_range_upper = torch.Tensor(DEFAULT_GP_RANGE_UPPER)
        self.gp_range_lower = torch.Tensor(DEFAULT_GP_RANGE_LOWER)   
        self.up_time_seconds = DEFAULT_TIME_TO_GO_SECS
        self.down_time_seconds = DEFAULT_TIME_TO_GO_SECS    
        time.sleep(0.5)

        # Set continuous control policy
        self.reset_policy()

        # Reset to rest pose
        self.rest_pos = torch.Tensor(self.rest_pose[0])
        self.rest_quat = torch.Tensor(self.rest_pose[1])
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
        joint_pos_current = self.arm.get_joint_positions()
        policy = toco.policies.CartesianImpedanceControl(
            joint_pos_current=joint_pos_current,
            Kp=(0.5*torch.Tensor(self.arm.metadata.default_Kx)),
            Kd=(0.5*torch.Tensor(self.arm.metadata.default_Kxd)),
            robot_model=self.arm.robot_model,
        )
        self.arm.send_torch_policy(policy, blocking=False)

    def move_to(self, pos, quat, gather_arm_state_func=None, time_to_go=DEFAULT_TIME_TO_GO_SECS):
        """
        Attempts to move to the given position and orientation by
        planning a Cartesian trajectory (a set of min-jerk waypoints)
        and updating the current policy's target to that
        end-effector position & orientation.

        Returns (num successes, num attempts, robot_states)
        """
        # Plan trajectory
        pos_curr, quat_curr = self.arm.get_ee_pose()
        N = int(time_to_go / self.planner_dt)

        waypoints = toco.planning.generate_cartesian_space_min_jerk(
            start=T.from_rot_xyz(R.from_quat(quat_curr), pos_curr),
            goal=T.from_rot_xyz(R.from_quat(quat), pos),
            time_to_go=time_to_go,
            hz=1 / self.planner_dt,
        )

        # Execute trajectory
        t0 = time.time()
        t_target = t0
        successes = 0
        error_detected = False
        robot_states = []
        for i in range(N):
            # Update traj
            ee_pos_desired = waypoints[i]["pose"].translation()
            ee_quat_desired = waypoints[i]["pose"].rotation().as_quat()
            # ee_twist_desired = waypoints[i]["twist"]
            try:
                self.arm.update_current_policy(
                    {
                        "ee_pos_desired": ee_pos_desired,
                        "ee_quat_desired": ee_quat_desired,
                        # "ee_vel_desired": ee_twist_desired[:3],
                        # "ee_rvel_desired": ee_twist_desired[3:],
                    }
                )
                if gather_arm_state_func:
                    observed_state = gather_arm_state_func(self.arm.get_robot_state())
                    observed_state["ee_pos_desired"] = ee_pos_desired
                    observed_state["ee_quat_desired"] = ee_quat_desired
                    robot_states.append(observed_state)
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
            t_target += self.planner_dt
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

    def grasp(self, grasp_pose0, grasp_pose1, gather_arm_state_func=None, gather_gripper_state_func=None, reset_at_end=True):
        results = []
        traj_state = {}

        traj_state["grasp_pose"] = grasp_pose0

        # Move to pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, self.pregrasp_height)
        successes, N, robot_states = self.move_to(pos, quat, gather_arm_state_func, time_to_go=self.up_time_seconds)
        results.append((successes, N))
        traj_state["move_to_grasp_states"] = robot_states

        # Lower (slower than other motions to prevent sudden collisions)
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, self.grasp_height)
        successes, N, robot_states = self.move_to(pos, quat, gather_arm_state_func, time_to_go=self.down_time_seconds)
        results.append((successes, N))
        traj_state["lower_to_grasp_states"] = robot_states       

        # Grasp
        close_gripper_state = self.close_gripper()
        if gather_gripper_state_func:
            traj_state["close_gripper_state"] = gather_gripper_state_func(close_gripper_state)

        # Lift to pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose0, self.pregrasp_height)
        successes, N, robot_states = self.move_to(pos, quat, gather_arm_state_func, time_to_go=self.up_time_seconds)
        results.append((successes, N))
        traj_state["lift_to_release_states"] = robot_states       

        # Move to new pregrasp
        pos, quat = self.grasp_pose_to_pos_quat(grasp_pose1, self.pregrasp_height)
        successes, N, robot_states = self.move_to(pos, quat, gather_arm_state_func, time_to_go=self.up_time_seconds)
        results.append((successes, N))
        traj_state["move_to_release_states"] = robot_states       

        # Read gripper state prior to release, in case object had fallen down
        if gather_gripper_state_func:
            preopen_gripper_state = self.gripper.get_state()
            traj_state["preopen_gripper_state"] = gather_gripper_state_func(preopen_gripper_state)

        # Release
        open_gripper_state = self.open_gripper()
        if gather_gripper_state_func:
            traj_state["open_gripper_state"] = gather_gripper_state_func(open_gripper_state)

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
                grasp_pose0 = uniform_sample(self.gp_range_lower, self.gp_range_upper)
                grasp_pose1 = uniform_sample(self.gp_range_lower, self.gp_range_upper)

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

    def set_pregrasp_height(self, pregrasp_height: float):
        self.pregrasp_height = pregrasp_height  

    def set_grasp_height(self, grasp_height: float):
        self.grasp_height = grasp_height    

    def set_planner_dt(self, planner_dt: float):
        self.planner_dt = planner_dt    

    def set_gp_range_upper(self, x: float, y: float, z: float):
        self.gp_range_upper = torch.Tensor([x, y, z])

    def set_gp_range_lower(self, x: float, y: float, z: float):
        self.gp_range_lower = torch.Tensor([x, y, z])     

    def set_up_time_seconds(self, up_time_seconds: float):
        self.up_time_seconds = up_time_seconds

    def set_down_time_seconds(self, down_time_seconds: float):
        self.down_time_seconds = down_time_seconds   

def uniform_sample(lower, upper):
    return lower + (upper - lower) * torch.rand_like(lower)



    