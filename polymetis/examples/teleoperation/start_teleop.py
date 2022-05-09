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
import hydra

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from polymetis import RobotInterface, GripperInterface

# teleop control frequency
UPDATE_HZ = 60
# low pass filter cutoff frequency
LPF_CUTOFF_HZ = 15

# controller gains (modified from libfranka example)
KP_DEFAULT = torch.Tensor([300.0, 300.0, 300.0, 30.0, 30.0, 30.0])
KD_DEFAULT = 2 * torch.sqrt(KP_DEFAULT)


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


@hydra.main(config_path="conf", config_name="teleop_config")
def main(cfg):
    # Initialize interfaces
    print("Connecting to devices...")
    robot = Robot(ip_address=cfg.nuc_ip, use_gripper=cfg.use_gripper)
    print("Connected to robot.")
    teleop = hydra.utils.instantiate(cfg.control_device)
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
    main()
