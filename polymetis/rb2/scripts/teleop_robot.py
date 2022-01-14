# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
import sophus as sp
import torch

import polymetis
import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T


# controller gains (modified from libfranka example)
KP_DEFAULT = torch.Tensor([300.0, 300.0, 300.0, 30.0, 30.0, 30.0])
KD_DEFAULT = 2 * torch.sqrt(KP_DEFAULT)


class TeleopRobot:
    """Wrapper around arm and gripper"""

    def __init__(self, ip_address="localhost", use_gripper=True, home_pose=None):
        self._use_gripper = use_gripper

        self.arm = polymetis.RobotInterface(ip_address=ip_address, enforce_version=False)
        if self._use_gripper:
            self.gripper = polymetis.GripperInterface(ip_address=ip_address)
        time.sleep(0.5)

        # Move arm to home
        if home_pose is not None:
            self.arm.set_home_pose(torch.Tensor(home_pose))

        # Reset (initialize) continuous control
        self.reset()

    def __del__(self):
        self.arm.terminate_current_policy()
        
    def reset(self):
        self.arm.go_home()
        self.reinit_policy()

    def reinit_policy(self):
        raise NotImplementedError
        
    def update_ee_pose(self, pose_des: sp.SE3):
        raise NotImplementedError
        
    def get_joint_pos(self):
        return self.arm.get_joint_angles()

    def get_ee_pose(self):
        pos_curr, quat_curr = self.arm.pose_ee()
        rotvec = R.from_quat(quat_curr).as_rotvec()
        return sp.SE3(sp.SO3.exp(rotvec).matrix(), pos_curr)

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
            
    def _check_disconnect_and_recover(self):
        """ Check if policy terminated due to issues and restart"""
        if self.arm.get_previous_interval().end != -1:
            print("Interrupt detected. Reinstantiating control policy...")
            time.sleep(1)
            self.reinit_policy()
            
            
class TeleopRobotCartesianControl(TeleopRobot):    
    def reinit_policy(self):
        # Send PD controller
        joint_pos_current = self.arm.get_joint_angles()
        policy = toco.policies.CartesianImpedanceControl(
            joint_pos_current=joint_pos_current,
            Kp=KP_DEFAULT,
            Kd=KD_DEFAULT,
            robot_model=self.arm.robot_model,
        )

        self.arm.send_torch_policy(policy, blocking=False)

        # Reset gripper
        self._open_gripper()

    def update_ee_pose(self, pose_des: sp.SE3):
        # Compute desired pos & quat
        ee_pos_desired = torch.Tensor(pose_des.translation())
        ee_quat_desired = R.from_matrix(
            torch.Tensor(pose_des.rotationMatrix())
        ).as_quat()

        # Update policy
        self.arm.update_current_policy(
            {"ee_pos_desired": ee_pos_desired, "ee_quat_desired": ee_quat_desired}
        )
        self._check_disconnect_and_recover()
            
class TeleopRobotJointControl(TeleopRobot):    
    def reinit_policy(self):
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

    def update_ee_pose(self, pose_des: sp.SE3):
        # Compute desired joint pose
        q_curr = self.arm.get_joint_angles()
        pose_curr = self.get_ee_pose()
        J = self.arm.robot_model.compute_jacobian(q_curr)
        J_pinv = torch.pinverse(J)
        
        pos_err = torch.Tensor(pose_des.translation() - pose_curr.translation())
        ori_err = torch.Tensor((pose_des.so3() * pose_curr.so3().inverse()).log())
        q_des = q_curr + J_pinv @ torch.concat([pos_err, ori_err])
        
        # Update policy
        self.arm.update_current_policy({"joint_pos_desired": q_des})
        self._check_disconnect_and_recover()

    def update_joint_pos(self, q_des: torch.Tensor):
        self.arm.update_current_policy({"joint_pos_desired": q_des})
        self._check_disconnect_and_recover()
        