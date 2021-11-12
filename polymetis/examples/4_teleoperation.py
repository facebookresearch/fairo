# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
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
UPDATE_HZ = 60
# low pass filter cutoff frequency
LPF_CUTOFF_HZ = 15

# controller gains (modified from libfranka example)
KP_DEFAULT = torch.Tensor([300.0, 300.0, 300.0, 30.0, 30.0, 30.0])
KD_DEFAULT = 2 * torch.sqrt(KP_DEFAULT)


class TeleopMode(Enum):
    KEYBOARD = 1
    OCULUS = 2


class TeleopDevice:
    """Allows for teleoperation using either the keyboard or an Oculus controller

    Keyboard: Control end-effector position with WASD and RF, toggle gripper state with space
    Oculus: Using the right controller, fully press the grip button (middle finger) to engage teleoperation. Hold B to perform grasp.
    """

    def __init__(self, ip_address=None, mode: TeleopMode = TeleopMode.OCULUS):
        self.mode = mode

        if self.mode == TeleopMode.OCULUS:
            self.reader = OculusReader(ip_address=ip_address)
            self.reader.run()

        elif self.mode == TeleopMode.KEYBOARD:
            self.steps = 0
            self.delta_pos = np.zeros(3)
            self.delta_rot = np.zeros(3)
            self.grasp_state = 0

        # LPF filter
        self.vr_pose_filtered = None
        tmp = 2 * np.pi * LPF_CUTOFF_HZ / UPDATE_HZ
        self.lpf_alpha = tmp / (tmp + 1)

    def get_state(self):
        if self.mode == TeleopMode.OCULUS:
            # Get data from oculus reader
            transforms, buttons = self.reader.get_transformations_and_buttons()

            # Generate output
            if transforms:
                is_active = buttons["rightGrip"][0] > 0.9
                grasp_state = buttons["B"]
                pose_matrix = np.linalg.pinv(transforms["l"]) @ transforms["r"]
            else:
                is_active = False
                grasp_state = 0
                pose_matrix = np.eye(4)
                self.vr_pose_filtered = None

            # Create transform (hack to prevent unorthodox matrices)
            r = R.from_matrix(torch.Tensor(pose_matrix[:3, :3]))
            vr_pose_curr = sp.SE3(
                sp.SO3.exp(r.as_rotvec()).matrix(), pose_matrix[:3, -1]
            )

            # Filter transform
            if self.vr_pose_filtered is None:
                self.vr_pose_filtered = vr_pose_curr
            else:
                self.vr_pose_filtered = interpolate_pose(
                    self.vr_pose_filtered, vr_pose_curr, self.lpf_alpha
                )
            pose = self.vr_pose_filtered

        elif self.mode == TeleopMode.KEYBOARD:
            # Get data from keyboard
            key = getch.getch()
            if key == "w":  # Translation
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
            elif key == "z":  # Rotation
                self.delta_rot[0] += 0.05
            elif key == "Z":
                self.delta_rot[0] -= 0.05
            elif key == "x":
                self.delta_rot[1] += 0.05
            elif key == "X":
                self.delta_rot[1] -= 0.05
            elif key == "c":
                self.delta_rot[2] += 0.05
            elif key == "C":
                self.delta_rot[2] -= 0.05
            elif key == " ":  # Gripper toggle
                self.grasp_state = 1 - self.grasp_state

            self.steps += 1

            # Generate output
            is_active = True

            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = (
                sp.SO3.exp(self.delta_rot).matrix() @ pose_matrix[:3, :3]
            )
            pose_matrix[:3, -1] = self.delta_pos
            pose = sp.SE3(pose_matrix)

            grasp_state = self.grasp_state

        return is_active, pose, grasp_state


class Robot:
    """Wrapper around arm and gripper"""

    def __init__(self, ip_address="localhost", use_gripper=False):
        self._use_gripper = use_gripper

        self.arm = RobotInterface(ip_address=ip_address)
        if self._use_gripper:
            self.gripper = GripperInterface(ip_address=ip_address)
        time.sleep(0.5)

        # Move arm to home
        self.arm.go_home()

        # Reset (initialize) continuous control
        self.reset()

    def __del__(self):
        self.arm.terminate_current_policy()

    def reset(self):
        # Send PD controller
        joint_pos_current = self.arm.get_joint_angles()
        """
        policy = toco.policies.JointImpedanceControl(
            joint_pos_current=joint_pos_current,
            Kp=self.arm.metadata.default_Kq,
            Kd=self.arm.metadata.default_Kqd,
            robot_model=self.arm.robot_model,
        )
        """
        policy = toco.policies.CartesianImpedanceControl(
            joint_pos_current=joint_pos_current,
            Kp=KP_DEFAULT,
            Kd=KD_DEFAULT,
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
        """
        # Compute desired joint pose
        q_curr = self.arm.get_joint_angles()
        pose_curr = self.get_ee_pose()

        J = self.arm.robot_model.compute_jacobian(q_curr)
        J_pinv = torch.pinverse(J)

        q_des = q_curr + J_pinv @ torch.Tensor((pose_des * pose_curr.inverse()).log())

        # Update policy
        self.arm.update_current_policy({"joint_pos_desired": q_des})
        """
        # Compute desired pos & quat
        ee_pos_desired = torch.Tensor(pose_des.translation())
        ee_quat_desired = R.from_matrix(
            torch.Tensor(pose_des.rotationMatrix())
        ).as_quat()

        # Update policy
        self.arm.update_current_policy(
            {"ee_pos_desired": ee_pos_desired, "ee_quat_desired": ee_quat_desired}
        )

        # Check if policy terminated due to issues and restart
        if self.arm.get_previous_interval().end != -1:
            print("Interrupt detected. Reinstantiating control policy...")
            time.sleep(1)
            self.reset()

    def update_grasp_state(self, is_grasped):
        if not self._use_gripper:
            return

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
        if self._use_gripper:
            self.gripper.grasp(speed=0.1, force=1.0, blocking=False)
            self.grasp_state = 1

    def _open_gripper(self):
        if self._use_gripper:
            self.gripper.goto(width=0.1, speed=0.1, force=1.0, blocking=False)
            self.grasp_state = 0


def interpolate_pose(pose1, pose2, pct):
    pose_diff = pose1.inverse() * pose2
    return pose1 * sp.SE3.exp(pct * pose_diff.log())


def pose_elementwise_diff(pose1, pose2):
    return sp.SE3(
        (pose2.so3() * pose1.so3().inverse()).matrix().T,
        pose2.translation() - pose1.translation(),
    )


def pose_elementwise_apply(delta_pose, pose):
    return sp.SE3(
        (pose.so3() * delta_pose.so3()).matrix(),
        delta_pose.translation() + pose.translation(),
    )


def main(args):
    if args.keyboard:
        print("Running in keyboard mode.")
        mode = TeleopMode.KEYBOARD
    else:
        print("Running in Oculus mode.")
        mode = TeleopMode.OCULUS

    # Initialize interfaces
    print("Connecting to devices...")
    robot = Robot(ip_address=args.ip, use_gripper=args.use_gripper)
    print("Connected to robot.")
    teleop = TeleopDevice(mode=mode)
    print("Connected to teleop device.")

    # Initialize variables
    vr_pose_ref = sp.SE3()
    arm_pose_ref = sp.SE3()
    init_ref = True

    t0 = time.time()
    t_target = t0
    t_delta = 1.0 / UPDATE_HZ

    # Start teleop loop
    print("======================== TELEOP START =========================")
    try:
        while True:
            # Obtain info from teleop device
            is_active, vr_pose_curr, grasp_state = teleop.get_state()

            # Update arm
            if is_active:
                # Update reference pose
                if init_ref:
                    vr_pose_ref = vr_pose_curr
                    arm_pose_ref = robot.get_ee_pose()
                    init_ref = False

                # Determine pose
                vr_pose_diff = pose_elementwise_diff(vr_pose_ref, vr_pose_curr)
                arm_pose_desired = pose_elementwise_apply(vr_pose_diff, arm_pose_ref)

                # Update
                robot.update_ee_pose(arm_pose_desired)
                robot.update_grasp_state(grasp_state)

            else:
                arm_pose_desired_filtered = None
                init_ref = True

            # Spin once
            t_target += t_delta
            t_remaining = t_target - time.time()
            time.sleep(max(t_remaining, 0.0))

    except KeyboardInterrupt:
        print("Session ended by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--ip", type=str, default="localhost", help="IP address of NUC."
    )
    parser.add_argument(
        "-k", "--keyboard", action="store_true", help="Teleop with keyboard mode."
    )
    parser.add_argument(
        "-g", "--use_gripper", action="store_true", help="Run with gripper enabled."
    )
    args = parser.parse_args()

    main(args)
