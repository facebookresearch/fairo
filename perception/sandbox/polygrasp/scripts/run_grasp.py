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

from polygrasp.segmentation_rpc import SegmentationClient
from polygrasp.grasp_rpc import GraspClient
from polygrasp.serdes import load_bw_img


def save_rgbd_masked(rgbd, rgbd_masked):
    num_cams = rgbd.shape[0]
    f, axarr = plt.subplots(2, num_cams)

    for i in range(num_cams):
        axarr[0, i].imshow(rgbd[i, :, :, :3].astype(np.uint8))
        axarr[1, i].imshow(rgbd_masked[i, :, :, :3].astype(np.uint8))

    f.savefig("rgbd_masked.png")
    plt.close(f)


def save_obj_masked(obj_masked_rgbd, obj_i, obj_mask_size):
    plt.imshow(obj_masked_rgbd[:, :, :3])
    plt.title(f"Object {obj_i}, mask size {obj_mask_size}")
    plt.savefig(f"object_{obj_i}_masked")
    plt.close()


@hydra.main(config_path="../conf", config_name="run_grasp")
def main(cfg):
    print(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    print(f"Current working directory: {os.getcwd()}")

    print("Initialize robot & gripper")
    robot = hydra.utils.instantiate(cfg.robot)
    robot.gripper_open()
    robot.go_home()

    print("Initializing cameras")
    cfg.cam.intrinsics_file = hydra.utils.to_absolute_path(cfg.cam.intrinsics_file)
    cfg.cam.extrinsics_file = hydra.utils.to_absolute_path(cfg.cam.extrinsics_file)
    cameras = hydra.utils.instantiate(cfg.cam)

    print("Loading camera workspace masks")
    masks_1 = np.array(
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_1], dtype=np.float64
    )
    masks_2 = np.array(
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_2], dtype=np.float64
    )

    print("Connect to grasp candidate selection and pointcloud processor")
    segmentation_client = SegmentationClient()
    grasp_client = GraspClient(view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path))

    root_working_dir = os.getcwd()
    for outer_i in range(1, 10):
        cam_i = outer_i % 2
        print(f"=== Starting outer loop with cam {cam_i} ===")

        if cam_i == 0:
            masks = masks_1
            hori_offset = torch.Tensor([0, 0.4, 0])
            grasp_offset = np.array([0.1, 0.0, 0.05])
            time_to_go = 2
        else:
            masks = masks_2
            hori_offset = torch.Tensor([0, -0.4, 0])
            grasp_offset = np.array([0.1, 0.1, 0.1])
            time_to_go = 3

        num_iters = 10
        for i in range(num_iters):
            # Create directory for current grasp iteration
            os.chdir(root_working_dir)
            timestamp = int(time.time())
            os.makedirs(f"{timestamp}")
            os.chdir(f"{timestamp}")
            print(f"Grasp {i + 1}/{num_iters}, logging to {os.getcwd()}")

            print("Getting rgbd and pcds..")
            rgbd = cameras.get_rgbd()
            rgbd_masked = rgbd * masks[:, :, :, None]
            scene_pcd = cameras.get_pcd(rgbd)
            cam_pcd = cameras.get_pcd_i(rgbd[cam_i], cam_i)
            save_rgbd_masked(rgbd, rgbd_masked)

            # Get RGBD & pointcloud
            print("Segmenting image...")
            obj_masked_rgbds, obj_masks = segmentation_client.segment_img(
                rgbd_masked[cam_i], min_mask_size=cfg.min_mask_size
            )
            if len(obj_masked_rgbds) == 0:
                print(f"Failed to find any objects with mask size > {cfg.min_mask_size}!")
                break

            for obj_i, (obj_masked_rgbd, obj_mask) in enumerate(zip(obj_masked_rgbds, obj_masks)):
                obj_mask_size = obj_mask.sum()
                save_obj_masked(obj_masked_rgbd, obj_i, obj_mask_size)

                print(f"Getting obj {obj_i} pcd...")
                pcd = cameras.get_pcd_i(obj_masked_rgbd, cam_i)
                print(f"Getting obj {obj_i} grasp...")
                grasp_group = grasp_client.get_grasps(pcd)
                filtered_grasp_group = grasp_client.get_collision(grasp_group, cam_pcd)
                if len(filtered_grasp_group) < len(grasp_group):
                    print(
                        f"Filtered {len(grasp_group) - len(filtered_grasp_group)}/{len(grasp_group)} grasps due to collision."
                    )
                if len(filtered_grasp_group) > 0:
                    break

            # Choose a grasp for this object
            grasp_client.visualize_grasp(scene_pcd, filtered_grasp_group)
            chosen_grasp = robot.select_grasp(filtered_grasp_group)

            # Execute grasp
            traj, success = robot.grasp(chosen_grasp, offset=grasp_offset, time_to_go=time_to_go)
            print(f"Grasp success: {success}")

            if success:
                print("Placing object in hand")
                curr_pose, curr_ori = robot.get_ee_pose()
                states = robot.move_until_success(
                    position=curr_pose + torch.Tensor([0, 0, 0.2]),
                    orientation=curr_ori,
                    time_to_go=time_to_go,
                )
                states = robot.move_until_success(
                    position=curr_pose + torch.Tensor([0, 0.0, 0.2]) + hori_offset,
                    orientation=curr_ori,
                    time_to_go=time_to_go,
                )
                states = robot.move_until_success(
                    position=curr_pose + torch.Tensor([0, 0.0, 0.05]) + hori_offset,
                    orientation=curr_ori,
                    time_to_go=time_to_go,
                )

            print("Opening gripper")
            robot.gripper_open()
            curr_pose, curr_ori = robot.get_ee_pose()
            states = robot.move_until_success(
                position=curr_pose + torch.Tensor([0, 0.0, 0.2]),
                orientation=curr_ori,
                time_to_go=time_to_go,
            )

            print("Going home")
            robot.go_home()


if __name__ == "__main__":
    main()
