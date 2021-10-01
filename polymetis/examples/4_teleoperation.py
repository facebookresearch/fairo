# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys
import threading
from enum import Enum

import numpy as np
import sophus as sp
import torch
import getch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from polymetis import RobotInterface, GripperInterface

from oculus_reader import OculusReader

# teleop control frequency
UPDATE_HZ = 30
# ramp up teleop engagement in this number of steps to prevent initial jerk
ENGAGE_STEPS = 10
# low pass filter cutoff frequency
LPF_CUTOFF_HZ = 15


class TeleopMode(Enum):
    KEYBOARD = 1
    OCULUS = 2


class TeleopDevice:
    """Allows for teleoperation using either the keyboard or an Oculus controller

    Keyboard: Control end-effector position with WASD and RF, toggle gripper state with space
    Oculus: Fully press both the trigger and the grip button to engage teleoperation. Hold B to perform grasp.
    """

    def __init__(self, ip_address=None, mode: TeleopMode = TeleopMode.OCULUS):
        self.mode = mode

        if self.mode == TeleopMode.OCULUS:
            self.reader = OculusReader(ip_address=ip_address)
            self.reader.run()

        elif self.mode == TeleopMode.KEYBOARD:
            self.steps = 0
            self.delta_pos = np.zeros(3)
            self.grasp_state = 0

    def get_state(self):
        if self.mode == TeleopMode.OCULUS:
            # Get data from oculus reader
            transforms, buttons = self.reader.get_transformations_and_buttons()

            # Generate output
            if transforms:
                is_active = (
                    buttons["rightGrip"][0] > 0.9 and buttons["rightTrig"][0] > 0.9
                )
                grasp_state = buttons["B"]
                pose_matrix = transforms["r"]
            else:
                is_active = False
                grasp_state = 0
                pose_matrix = np.eye(4)

        elif self.mode == TeleopMode.KEYBOARD:
            # Get data from keyboard
            if self.steps > ENGAGE_STEPS:
                key = getch.getch()
                if key == "w":
                    self.delta_pos[0] += 0.01
                elif key == "s":
                    self.delta_pos[0] -= 0.01
                elif key == "a":
                    self.delta_pos[1] += 0.01
                elif key == "d":
                    self.delta_pos[1] -= 0.01
                elif key == "r":
                    self.delta_pos[2] += 0.01
                elif key == "f":
                    self.delta_pos[2] -= 0.01
                elif key == " ":
                    self.grasp_state = 1 - self.grasp_state

            self.steps += 1

            # Generate output
            is_active = True

            pose_matrix = np.eye(4)
            pose_matrix[:3, -1] = self.delta_pos

            grasp_state = self.grasp_state

        return is_active, pose_matrix, grasp_state


class Robot:
    """ Wrapper around arm and gripper """

    def __init__(self, ip_address="localhost"):
        self.arm = RobotInterface(ip_address=ip_address)
        self.gripper = GripperInterface(ip_address=ip_address)
        time.sleep(0.5)

        # Reset
        self.reset()

    def __del__(self):
        self.arm.terminate_current_policy()

    def reset(self):
        # Go home
        self.arm.go_home()

        # Send PD controller
        joint_pos_current = self.arm.get_joint_angles()
        policy = toco.policies.JointImpedanceControl(
            joint_pos_current=joint_pos_current,
            Kp=self.arm.metadata.default_Kq,
            Kd=self.arm.metadata.default_Kqd,
            robot_model=self.arm.robot_model,
        )
        self.arm.send_torch_policy(policy, blocking=False)

        # Reset gripper
        self._open_gripper()

    def get_ee_pose(self):
        pos_curr, quat_curr = self.arm.pose_ee()
        rotvec = R.from_quat(quat_curr).as_rotvec()
        return sp.SE3(sp.SO3.exp(rotvec).matrix(), pos_curr)

    def update_ee_pose(self, pose_des):
        # Compute desired joint pose
        q_curr = self.arm.get_joint_angles()
        pose_curr = self.get_ee_pose()

        J = self.arm.robot_model.compute_jacobian(q_curr)
        J_pinv = torch.pinverse(J)

        q_des = q_curr + J_pinv @ torch.Tensor((pose_des * pose_curr.inverse()).log())

        # Update policy
        self.arm.update_current_policy({"joint_pos_desired": q_des})

        # Check if policy terminated due to issues and restart
        if self.arm.get_previous_interval().end != -1:
            print("Interrupt detected. Reinstantiating control policy...")
            time.sleep(3)
            self.reset()

    def update_grasp_state(self, is_grasped):
        self.desired_grasp_state = is_grasped

        # Send command if gripper is idle and desired grasp state is different from current grasp state
        gripper_state = self.gripper.get_state()
        if not gripper_state.is_moving:
            if self.grasp_state != self.desired_grasp_state:
                if self.desired_grasp_state:
                    self._close_gripper()
                else:
                    self._open_gripper()

    def _close_gripper(self):
        self.gripper.goto(pos=0.0, vel=0.1, force=1.0, blocking=False)
        self.grasp_state = 1

    def _open_gripper(self):
        self.gripper.goto(pos=0.1, vel=0.1, force=1.0, blocking=False)
        self.grasp_state = 0


def interpolate_pose(pose1, pose2, pct):
    pose_diff = pose2 * pose1.inverse()
    return sp.SE3.exp(pct * pose_diff.log()) * pose1


if __name__ == "__main__":
    # mode = TeleopMode.KEYBOARD
    mode = TeleopMode.OCULUS

    # Initialize interfaces
    print("Connecting to devices...")
    robot = Robot()
    teleop = TeleopDevice(mode=mode)
    print("Connected.")

    # Initialize variables
    vr_pose_ref = sp.SE3()
    arm_pose_ref = sp.SE3()
    engage_pct = 0.0

    t0 = time.time()
    t_target = t0
    t_delta = 1.0 / UPDATE_HZ

    arm_pose_desired_filtered = None
    tmp = 2 * np.pi * LPF_CUTOFF_HZ / UPDATE_HZ
    lpf_alpha = tmp / (tmp + 1)

    # Start teleop loop
    print("======================== TELEOP START =========================")
    try:
        while True:
            # Obtain info from teleop device
            is_active, pose_matrix, grasp_state = teleop.get_state()

            # Update arm
            if is_active:
                # Hack to prevent unorthodox matrices
                r = R.from_matrix(torch.Tensor(pose_matrix[:3, :3]))
                vr_pose_curr = sp.SE3(
                    sp.SO3.exp(r.as_rotvec()).matrix(), pose_matrix[:3, -1]
                )

                # Update reference pose through a gradual engaging process
                if engage_pct < 1.0:
                    arm_pose_curr = robot.get_ee_pose()

                    vr_pose_ref = interpolate_pose(
                        vr_pose_curr, vr_pose_ref, engage_pct
                    )
                    arm_pose_ref = interpolate_pose(
                        arm_pose_curr, arm_pose_ref, engage_pct
                    )

                    engage_pct += 1.0 / ENGAGE_STEPS

                # Determine pose
                arm_pose_desired = (vr_pose_curr * vr_pose_ref.inverse()) * arm_pose_ref
                if arm_pose_desired_filtered is None:
                    arm_pose_desired_filtered = arm_pose_desired
                else:
                    arm_pose_desired_filtered = interpolate_pose(
                        arm_pose_desired_filtered, arm_pose_desired, lpf_alpha
                    )

                # Update
                robot.update_ee_pose(arm_pose_desired_filtered)
                robot.update_grasp_state(grasp_state)

            else:
                arm_pose_desired_filtered = None
                engage_pct = 0.0

            # Spin once
            t_target += t_delta
            t_remaining = t_target - time.time()
            time.sleep(max(t_remaining, 0.0))

    except KeyboardInterrupt:
        print("Session ended by user.")
