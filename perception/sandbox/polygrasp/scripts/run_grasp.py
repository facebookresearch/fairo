#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

import random

import hydra
from polygrasp.pointcloud_rpc import PointCloudClient
from polygrasp.grasp_rpc import GraspClient


@hydra.main(config_path="../conf", config_name="run_grasp")
def main(cfg):
    # Initialize robot & gripper
    robot = hydra.utils.instantiate(cfg.robot)
    robot.gripper_open()
    # robot.go_home()

    # Initialize cameras
    cfg.camera_sub.intrinsics_file = hydra.utils.to_absolute_path(cfg.camera_sub.intrinsics_file)
    cfg.camera_sub.extrinsics_file = hydra.utils.to_absolute_path(cfg.camera_sub.extrinsics_file)
    cameras = hydra.utils.instantiate(cfg.camera_sub)
    camera_intrinsics = cameras.get_intrinsics()
    camera_extrinsics = cameras.get_extrinsics()

    # Connect to grasp candidate selection and pointcloud processor
    pcd_client = PointCloudClient(camera_intrinsics, camera_extrinsics)
    grasp_client = GraspClient(view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path))

    num_iters = 1
    for i in range(num_iters):
        print(f"Grasp {i + 1} / num_iters")

        # Get RGBD & pointcloud
        rgbd = cameras.get_rgbd()

        scene_pcd = pcd_client.get_pcd(rgbd)
        grasp_group = grasp_client.get_grasps(scene_pcd)

        grasp_client.visualize_grasp(
            scene_pcd, grasp_group, render=False, save_view=False, plot=True
        )

        # # Get grasps per object
        # obj_to_pcd = pcd_client.segment_pcd(scene_pcd)
        # obj_to_grasps = {obj: grasp_client.get_grasps(pcd) for obj, pcd in obj_to_pcd.items()}

        # # Pick a random object to grasp
        # curr_obj, curr_grasps = random.choice(list(obj_to_grasps.items()))
        # print(f"Picking object with ID {curr_obj}")

        # Choose a grasp for this object
        # TODO: scene-aware motion planning for grasps
        curr_grasps = grasp_group
        chosen_grasp = robot.select_grasp(curr_grasps, scene_pcd)

        # Execute grasp
        # traj, success = robot.grasp(chosen_grasp)
        # print(f"Grasp success: {success}")

        # if success:
        #     print(f"Moving end-effector up and down")
        #     curr_pose, curr_ori = robot.get_ee_pose()
        #     robot.move_to_ee_pose(torch.Tensor([0, 0, 0.1]), delta=True)
        #     robot.move_to_ee_pose(torch.Tensor([0, 0, -0.1]), delta=True)


if __name__ == "__main__":
    main()
