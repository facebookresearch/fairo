"""polymetis.RobotInterface combined with GripperInterface, with an additional `grasp` method."""

import time
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
import torch

import hydra
import graspnetAPI
import polymetis

import ikpy.chain
import tempfile

from . import compute_quat_dist

log = logging.getLogger(__name__)


def compute_des_pose(best_grasp):
    """Convert between GraspNet coordinates to robot coordinates."""

    # Grasp point
    grasp_point = best_grasp.translation

    # Compute plane of rotation through three orthogonal vectors on plane of rotation
    grasp_approach_delta = best_grasp.rotation_matrix @ np.array([-0.3, 0.0, 0])
    grasp_approach_delta_plus = best_grasp.rotation_matrix @ np.array([-0.3, 0.1, 0])
    grasp_approach_delta_minus = best_grasp.rotation_matrix @ np.array([-0.3, -0.1, 0])
    bx = -grasp_approach_delta
    by = grasp_approach_delta_plus - grasp_approach_delta_minus
    bx = bx / np.linalg.norm(bx)
    by = by / np.linalg.norm(by)
    bz = np.cross(bx, by)
    plane_rot = R.from_matrix(np.vstack([bx, by, bz]).T)

    # Convert between GraspNet neutral orientation to robot neutral orientation
    des_ori = plane_rot * R.from_euler("y", 90, degrees=True)
    des_ori_quat = des_ori.as_quat()

    return grasp_point, grasp_approach_delta, des_ori_quat


def grasp_to_pose(grasp: graspnetAPI.Grasp):
    return grasp.translation, R.from_matrix(grasp.rotation_matrix).as_quat()


def min_dist_grasp(default_ee_quat, grasps):
    """Find the grasp with minimum orientation distance to the reference grasp"""
    with torch.no_grad():
        rots_as_quat = [
            torch.Tensor(R.from_matrix(grasp.rotation_matrix).as_quat())
            for grasp in grasps
        ]
        dists = [compute_quat_dist(rot, default_ee_quat) for rot in rots_as_quat]
        i = torch.argmin(torch.Tensor(dists)).item()
    log.info(f"Grasp {i} has angle {dists[i]} from reference.")
    return grasps[i], i


def min_dist_grasp_no_z(default_ee_quat, grasps):
    """
    Find the grasp with minimum orientation distance to the reference grasp
    disregarding orientation about z axis
    """
    with torch.no_grad():
        rots_as_R = [R.from_quat(compute_des_pose(grasp)[2]) for grasp in grasps]
        default_r = R.from_quat(default_ee_quat)
        dists = [
            np.linalg.norm((rot.inv() * default_r).as_rotvec()[:2]) for rot in rots_as_R
        ]
        i = torch.argmin(torch.Tensor(dists)).item()
    log.info(f"Grasp {i} has angle {dists[i]} from reference.")
    return grasps[i], i


class GraspingRobotInterface(polymetis.RobotInterface):
    def __init__(
        self,
        gripper: polymetis.GripperInterface,
        k_approach=1.5,
        k_grasp=0.72,
        gripper_max_width=0.085,
        # ikpy params:
        base_elements=("panda_link0",),
        soft_limits=(
            (-2.70, 2.70),
            (-1.56, 1.56),
            (-2.7, 2.7),
            (-2.87, -0.07),
            (-2.7, 2.7),
            (-0.02, 3.55),
            (-2.7, 2.7),
        ),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gripper = hydra.utils.instantiate(gripper)

        self.default_ee_quat = torch.Tensor([1, 0, 0, 0])
        self.k_approach = k_approach
        self.k_grasp = k_grasp
        self.gripper_max_width = gripper_max_width

        with tempfile.NamedTemporaryFile(mode="w+") as f:
            f.write(self.metadata.urdf_file)
            f.seek(0)
            self.robot_model_ikpy = ikpy.chain.Chain.from_urdf_file(
                f.name,
                base_elements=base_elements,
            )
        for i in range(len(soft_limits)):
            self.robot_model_ikpy.links[i + 1].bounds = soft_limits[i]

    def ik(self, position, orientation=None):
        curr_joint_pos = [0] + self.get_joint_positions().numpy().tolist() + [0]
        des_homog_transform = np.eye(4)
        if orientation is not None:
            des_homog_transform[:3, :3] = R.from_quat(orientation).as_matrix()
        des_homog_transform[:3, 3] = position
        try:
            joint_pos_ikpy = self.robot_model_ikpy.inverse_kinematics_frame(
                target=des_homog_transform,
                orientation_mode="all",
                no_position=False,
                initial_position=curr_joint_pos,
            )
            return joint_pos_ikpy[1:-1]
        except ValueError as e:
            log.warning(f"Can't find IK solution! {e}")
            return None

    def gripper_open(self):
        """Open the gripper. (Assumes Robotiq griper)"""
        self.gripper.goto(1, 1, 1)

    def gripper_close(self):
        """Closes the gripper. (Assumes Robotiq gripper)"""
        self.gripper.goto(0, 1, 1)

    def move_until_success(
        self, position, orientation, time_to_go, max_attempts=5, success_dist=0.05
    ):
        states = []
        for _ in range(max_attempts):
            joint_pos = self.ik(position, orientation)
            states += self.move_to_joint_positions(joint_pos, time_to_go=time_to_go)
            curr_ee_pos, curr_ee_ori = self.get_ee_pose()

            xyz_diff = torch.linalg.norm(curr_ee_pos - position)
            ori_diff = (
                R.from_quat(curr_ee_ori).inv() * R.from_quat(orientation)
            ).magnitude()
            log.info(f"Dist to goal: xyz diff {xyz_diff}, ori diff {ori_diff}")

            if (
                states
                and states[-1].prev_command_successful
                and xyz_diff < success_dist
                and ori_diff < 0.2
            ):
                break
        return states

    def check_feasibility(self, point: np.ndarray):
        return self.ik(point) is not None

    def select_grasp(
        self, grasps: graspnetAPI.GraspGroup, num_grasp_choices=5
    ) -> graspnetAPI.Grasp:
        with torch.no_grad():
            feasible_i = []
            for i, grasp in enumerate(grasps):
                log.info(f"checking feasibility {i}/{len(grasps)}")

                if grasp.width > self.gripper_max_width:
                    continue

                grasp_point, grasp_approach_delta, des_ori_quat = compute_des_pose(
                    grasp
                )
                point_a = grasp_point + self.k_approach * grasp_approach_delta
                point_b = grasp_point + self.k_grasp * grasp_approach_delta

                if self.check_feasibility(point_a) and self.check_feasibility(point_b):
                    feasible_i.append(i)

                if len(feasible_i) == num_grasp_choices:
                    if i >= num_grasp_choices:
                        log.info(
                            f"Kinematically filtered {i + 1 - num_grasp_choices} grasps"
                            " to get 5 feasible positions"
                        )
                    break

            # Choose the grasp closest to the neutral position
            filtered_grasps = grasps[feasible_i]
            grasp, i = min_dist_grasp_no_z(self.default_ee_quat, filtered_grasps)
            log.info(f"Closest grasp to ee ori, within top 5: {i + 1}")
            return filtered_grasps, i

    def grasp(
        self,
        grasp: graspnetAPI.Grasp,
        time_to_go=3,
        gripper_width_success_threshold=0.0005,
    ):
        """
        Given a graspnetAPI.Grasp object, the robot opens the gripper, moves the end effector
        close to object, and closes the gripper.

        Returns the trajectory and success/failure based on gripper width threshold.
        """
        states = []
        grasp_point, grasp_approach_delta, des_ori_quat = compute_des_pose(grasp)

        self.gripper_open()
        states += self.move_until_success(
            position=torch.Tensor(grasp_point + grasp_approach_delta * self.k_approach),
            orientation=torch.Tensor(des_ori_quat),
            time_to_go=time_to_go,
        )

        grip_ee_pos = torch.Tensor(grasp_point + grasp_approach_delta * self.k_grasp)

        states += self.move_until_success(
            position=grip_ee_pos,
            orientation=torch.Tensor(des_ori_quat),
            time_to_go=time_to_go,
        )
        self.gripper_close()

        log.info(f"Waiting for gripper to close...")
        while self.gripper.get_state().is_moving:
            time.sleep(0.2)

        gripper_state = self.gripper.get_state()
        width = gripper_state.width
        log.info(f"Gripper width: {width}")

        success = width > gripper_width_success_threshold

        return states, success
