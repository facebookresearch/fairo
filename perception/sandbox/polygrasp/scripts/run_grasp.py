#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

import time
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import hydra
import omegaconf

from polygrasp.pointcloud_rpc import SegmentationPointCloudClient
from polygrasp.grasp_rpc import GraspClient

def save_rgbd_masked(rgbd, rgbd_masked):
    num_cams = rgbd.shape[0]
    f, axarr = plt.subplots(2, num_cams)

    for i in range(num_cams):
        axarr[0, i].imshow(rgbd[i, :, :, :3].astype(np.uint8))
        axarr[1, i].imshow(rgbd_masked[i, :, :, :3].astype(np.uint8))

    f.savefig("rgbd_masked")


@hydra.main(config_path="../conf", config_name="run_grasp")
def main(cfg):
    print(f"Config: {omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    print(f"Current working directory: {os.getcwd()}")
    # Initialize robot & gripper
    robot = hydra.utils.instantiate(cfg.robot)
    robot.gripper_open()
    robot.go_home()

    # Initialize cameras
    cfg.camera_sub.intrinsics_file = hydra.utils.to_absolute_path(cfg.camera_sub.intrinsics_file)
    cfg.camera_sub.extrinsics_file = hydra.utils.to_absolute_path(cfg.camera_sub.extrinsics_file)
    cameras = hydra.utils.instantiate(cfg.camera_sub)
    camera_intrinsics = cameras.get_intrinsics()
    camera_extrinsics = cameras.get_extrinsics()

    # masks = np.ones([3, 480, 640])

    masks = np.zeros([3, 480, 640])
    # masks[0][70:320, 60:400] = 1
    # masks[2][130:350, 240:480] = 1
    masks[0][20:460, :440] = 1
    masks[1][100:480, 180:600] = 1

    # Connect to grasp candidate selection and pointcloud processor
    pcd_client = SegmentationPointCloudClient(camera_intrinsics, camera_extrinsics, masks=masks)
    grasp_client = GraspClient(view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path))

    root_dir = os.getcwd()

    for outer_i in range(0, 10):
        cam_i = outer_i % 2
        print(f"=== Starting outer loop with cam {cam_i} ===")

        if cam_i == 0:
            hori_offset = torch.Tensor([0, 0.4, 0])
            grasp_offset = np.array([0.1, 0.0, 0.05])
            time_to_go = 2
        else:
            hori_offset = torch.Tensor([0, -0.4, 0])
            grasp_offset = np.array([0.1, 0.1, 0.1])
            time_to_go = 3

        num_iters = 10
        for i in range(num_iters):
            os.chdir(root_dir)
            timestamp = int(time.time())
            os.makedirs(f"{timestamp}")
            os.chdir(f"{timestamp}")

            print(f"Grasp {i + 1} / num_iters, in {os.getcwd()}")

            print("Getting rgbd and pcds..")
            rgbd = cameras.get_rgbd()
            rgbd_masked = rgbd * masks[:, :, :, None]
            pcds = pcd_client.get_pcd(rgbd)

            save_rgbd_masked(rgbd, rgbd_masked)

            scene_pcd = pcds[0]
            for pcd in pcds[1:]:
                scene_pcd += pcd

            # Get RGBD & pointcloud
            print("Segmenting image...")
            labels = pcd_client.segment_img(rgbd_masked[cam_i])

            obj_to_grasps = {}
            num_objs = int(labels.max())
            print(f"Number of objs: {num_objs}")
            min_mask_size = 2500
            for obj_i in range(1, num_objs + 1):
                obj_mask = labels == obj_i
                obj_mask_size = obj_mask.sum()

                if obj_mask_size < min_mask_size:
                    continue

                obj_masked_rgbd = rgbd_masked[cam_i] * obj_mask[:, :, None]

                plt.imshow(obj_masked_rgbd[:, :, :3])
                plt.title(f"Object {obj_i}, mask size {obj_mask_size}")
                plt.savefig(f"object_{obj_i}_masked")
                plt.close()
                # plt.show()

                print(f"Getting obj {obj_i} pcd...")
                pcd = pcd_client.get_pcd_i(obj_masked_rgbd, cam_i)
                print(f"Getting obj {obj_i} grasp...")
                grasp_group = grasp_client.get_grasps(pcd)
                filtered_grasp_group = grasp_client.get_collision(grasp_group, scene_pcd)
                if len(filtered_grasp_group) > 0:
                    obj_to_grasps[obj_i] = filtered_grasp_group
                    break

            if len(obj_to_grasps) == 0:
                print(f"Failed to find any objects with mask size > {min_mask_size}!")
                break

            curr_grasps = grasp_group

            # Choose a grasp for this object
            grasp_client.visualize_grasp(pcds[cam_i], curr_grasps)
            chosen_grasp = robot.select_grasp(curr_grasps, scene_pcd)

            # Execute grasp
            traj, success = robot.grasp(chosen_grasp, offset=grasp_offset, time_to_go=time_to_go)
            print(f"Grasp success: {success}")

            if success:
                print(f"Moving end-effector up")
                curr_pose, curr_ori = robot.get_ee_pose()
                states = robot.move_until_success(position=curr_pose + torch.Tensor([0, 0, 0.2]), orientation=curr_ori, time_to_go=time_to_go)
                states = robot.move_until_success(position=curr_pose + torch.Tensor([0, 0.0, 0.2]) + hori_offset, orientation=curr_ori, time_to_go=time_to_go)
                states = robot.move_until_success(position=curr_pose + torch.Tensor([0, 0.0, 0.05]) + hori_offset, orientation=curr_ori, time_to_go=time_to_go)

            robot.gripper_open()
            curr_pose, curr_ori = robot.get_ee_pose()
            states = robot.move_until_success(position=curr_pose + torch.Tensor([0, 0.0, 0.2]), orientation=curr_ori, time_to_go=time_to_go)
            robot.go_home()


if __name__ == "__main__":
    main()
