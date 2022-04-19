"""polymetis.RobotInterface combined with GripperInterface, with an additional `grasp` method."""

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import open3d as o3d

import hydra
import graspnetAPI
import polymetis


def compute_des_pose(best_grasp):
    grasp_point = best_grasp.translation

    grasp_approach_delta = best_grasp.rotation_matrix @ np.array([-0.3, 0.0, 0])
    grasp_approach_delta_plus = best_grasp.rotation_matrix @ np.array([-0.3, 0.1, 0])
    grasp_approach_delta_minus = best_grasp.rotation_matrix @ np.array([-0.3, -0.1, 0])
    bx = -grasp_approach_delta
    by = grasp_approach_delta_plus - grasp_approach_delta_minus
    bx = bx / np.linalg.norm(bx)
    by = by / np.linalg.norm(by)
    bz = np.cross(bx, by)
    plane_rot = R.from_matrix(np.vstack([bx, by, bz]).T)

    des_ori = plane_rot * R.from_euler("x", 90, degrees=True) * R.from_euler("y", 90, degrees=True)
    des_ori_quat = des_ori.as_quat()

    return grasp_point, grasp_approach_delta, des_ori_quat


def grasp_to_pose(grasp: graspnetAPI.Grasp):
    return grasp.translation, R.from_matrix(grasp.rotation_matrix).as_quat()


class GraspingRobotInterface(polymetis.RobotInterface):
    def __init__(self, gripper: polymetis.GripperInterface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gripper = hydra.utils.instantiate(gripper)

    def gripper_open(self):
        self.gripper.goto(1, 1, 1)

    def gripper_close(self):
        self.gripper.goto(0, 1, 1)

    def _move_until_success(self, position, orientation, time_to_go, max_attempts=5):
        states = []
        for _ in range(max_attempts):
            states += self.move_to_ee_pose(
                position=position, orientation=orientation, time_to_go=time_to_go
            )
            curr_ee_pos, curr_ee_ori = self.get_ee_pose()

            print(f"Dist to goal: {torch.linalg.norm(curr_ee_pos - position)}")

            if torch.linalg.norm(curr_ee_pos - position) < 0.25:  # TODO: orientation diff
                break
        return states

    def select_grasp(
        self, grasps: graspnetAPI.GraspGroup, scene_pcd: o3d.geometry.PointCloud
    ) -> graspnetAPI.Grasp:
        # TODO: do something smarter than this
        return grasps[int(input("Choose grasp index (1-indexed):")) - 1]

    def grasp(self, grasp: graspnetAPI.Grasp):
        states = []
        grasp_point, grasp_approach_delta, des_ori_quat = compute_des_pose(grasp)

        offset = np.array([0.1, 0, 0.1])

        self.gripper_open()
        states += self._move_until_success(
            position=torch.Tensor(grasp_point + grasp_approach_delta * 2.0 + offset),
            orientation=torch.Tensor(des_ori_quat),
            time_to_go=3,
        )

        grip_ee_pos = torch.Tensor(grasp_point + grasp_approach_delta * 0.8 + offset)

        states += self._move_until_success(
            position=grip_ee_pos, orientation=torch.Tensor(des_ori_quat), time_to_go=3
        )

        self.gripper_close()

        success = self.gripper.get_state().width > 0.001

        return states, success
