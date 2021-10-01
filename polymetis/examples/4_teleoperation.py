# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys
import threading

import numpy as np
import sophus as sp
import torch
import getch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from polymetis import RobotInterface, GripperInterface

# from oculus_reader import OculusReader


UPDATE_HZ = 30  # teleop control frequency
ENGAGE_STEPS = (
    10  # ramp up teleop engagement in this number of steps to prevent initial jerk
)


class Robot:
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


class TeleopDevice:
    """Allows for teleoperation using either the keyboard or an Oculus controller

    Keyboard: Control end-effector position with WASD and RF, toggle gripper state with space
    Oculus: Fully press both the trigger and the grip button to engage teleoperation. Hold B to perform grasp.
    """

    def __init__(self, ip_address=None, mode="oculus"):
        self.mode = mode

        if self.mode == "oculus":
            self.reader = OculusReader()
            self.reader.run()

        elif self.mode == "keyboard":
            self.steps = 0
            self.delta_pos = np.zeros(3)
            self.grasp_state = 0

    def get_state(self):
        if self.mode == "oculus":
            # Get data from oculus reader
            transforms, buttons = self.reader.get_transformations_and_buttons()

            # Generate output
            is_active = buttons["rightGrip"] > 0.9 and buttons["rightTrig"] > 0.9
            grasp_state = buttons["B"]
            pose_matrix = transforms["r"]

        elif self.mode == "keyboard":
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


def interpolate_pose(pose1, pose2, pct):
    return sp.SE3.exp((1.0 - pct) * pose1.log() + pct * pose2.log())


if __name__ == "__main__":
    # Initialize interfaces
    robot = Robot()
    teleop = TeleopDevice(mode="keyboard")

    # Start teleop loop
    vr_pose_ref = sp.SE3()
    arm_pose_ref = sp.SE3()
    engage_pct = 0.0

    t0 = time.time()
    t_target = t0
    t_delta = 1.0 / UPDATE_HZ

    try:
        while True:
            # Obtain info from teleop device
            is_active, pose_matrix, grasp_state = teleop.get_state()

            # Update arm
            if is_active:
                vr_pose_curr = sp.SE3(pose_matrix)

                # Update reference pose through a gradual engaging process
                if engage_pct < 1.0:
                    arm_pose_curr = robot.get_ee_pose()
                    arm_pose_ref = interpolate_pose(
                        arm_pose_curr, arm_pose_ref, engage_pct
                    )
                    vr_pose_ref = interpolate_pose(
                        vr_pose_curr, vr_pose_ref, engage_pct
                    )

                    engage_pct += 1.0 / ENGAGE_STEPS

                # Determine pose
                pose_desired = (vr_pose_curr * vr_pose_ref.inverse()) * arm_pose_ref

                # Update
                robot.update_ee_pose(pose_desired)
                robot.update_grasp_state(grasp_state)

            else:
                engage_pct = 0.0

            # Spin once
            t_target += t_delta
            t_remaining = t_target - time.time()
            time.sleep(max(t_remaining, 0.0))

    except KeyboardInterrupt:
        print("Session ended by user.")
