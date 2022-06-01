#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

from scipy.spatial.transform import Rotation as R

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
        if num_cams > 1:
            ax1, ax2 = axarr[0, i], axarr[1, i]
        else:
            ax1, ax2 = axarr
        ax1.imshow(rgbd[i, :, :, :3].astype(np.uint8))
        ax2.imshow(rgbd_masked[i, :, :, :3].astype(np.uint8))

    f.savefig("rgbd_masked.png")
    plt.close(f)



def get_obj_grasps(grasp_client, obj_masked_rgbds, obj_pcds, scene_pcd):
    for obj_i, obj_pcd in enumerate(obj_pcds):
        print(f"Getting obj {obj_i} grasp...")
        grasp_group = grasp_client.get_grasps(obj_pcd)
        filtered_grasp_group = grasp_client.get_collision(grasp_group, scene_pcd)
        if len(filtered_grasp_group) < len(grasp_group):
            print(
                f"Filtered {len(grasp_group) - len(filtered_grasp_group)}/{len(grasp_group)} grasps due to collision."
            )
        if len(filtered_grasp_group) > 0:
            return obj_i, filtered_grasp_group
    raise Exception(
        f"Unable to find any grasps after filtering, for any of the {len(obj_masked_rgbds)} objects"
    )


def execute_grasp(robot, chosen_grasp, grasp_offset, hori_offset, time_to_go):
    traj, success = robot.grasp(chosen_grasp, offset=grasp_offset, time_to_go=time_to_go, gripper_width_success_threshold=0.001)
    print(f"Grasp success: {success}")

    if success:
        print("Placing object in hand")
        curr_pose, curr_ori = robot.get_ee_pose()
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0, 0.2]),
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0.0, 0.2]) + hori_offset,
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0.0, 0.05]) + hori_offset,
            orientation=curr_ori,
            time_to_go=time_to_go,
        )

    print("Opening gripper")
    robot.gripper_open()
    curr_pose, curr_ori = robot.get_ee_pose()
    traj += robot.move_until_success(
        position=curr_pose + torch.Tensor([0, 0.0, 0.2]),
        orientation=curr_ori,
        time_to_go=time_to_go,
    )

    return traj


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
    for outer_i in range(cfg.num_bin_shifts):
        cam_i = outer_i % 2
        # cam_i = 1
        print(f"=== Starting bin shift with cam {cam_i} ===")

        if cam_i == 0:
            masks = masks_1
            hori_offset = torch.Tensor([0, -0.4, 0])
            grasp_offset = np.array([0.0, 0.0, 0.00])
        else:
            masks = masks_2
            hori_offset = torch.Tensor([0, 0.4, 0])
            grasp_offset = np.array([0.0, 0.0, 0.0])
        time_to_go = 3
        
        for i in range(cfg.num_grasps_per_bin_shift):
            # Create directory for current grasp iteration
            os.chdir(root_working_dir)
            timestamp = int(time.time())
            os.makedirs(f"{timestamp}")
            os.chdir(f"{timestamp}")

            print(
                f"=== Grasp {i + 1}/{cfg.num_grasps_per_bin_shift}, logging to {os.getcwd()} ==="
            )

            print("Getting rgbd and pcds..")
            rgbd = cameras.get_rgbd()

            rgbd_masked = rgbd * masks[:, :, :, None]
            scene_pcd = cameras.get_pcd(rgbd)
            save_rgbd_masked(rgbd, rgbd_masked)

            print("Segmenting image...")
            obj_masked_rgbds, obj_masks = segmentation_client.segment_img(rgbd_masked[cam_i], min_mask_size=cfg.min_mask_size)
            obj_masked_pcds = [cameras.get_pcd_i(obj_masked_rgbd, cam_i) for obj_masked_rgbd in obj_masked_rgbds]
            obj_pcd_centers = [pcd.get_center() for pcd in obj_masked_pcds]
            if len(obj_masked_rgbds) == 0:
                print(f"Failed to find any objects with mask size > {cfg.min_mask_size}!")
                break

            cam_i_other = 1 - cam_i
            obj_masked_rgbds_2, obj_masks_2 = segmentation_client.segment_img(rgbd_masked[cam_i_other], min_mask_size=cfg.min_mask_size)
            obj_masked_pcds_2 = [cameras.get_pcd_i(obj_masked_rgbd, cam_i_other) for obj_masked_rgbd in obj_masked_rgbds_2]
            obj_pcd_centers_2 = [pcd.get_center() for pcd in obj_masked_pcds_2]

            obj_pcds = []
            for i, center_1 in enumerate(obj_pcd_centers):
                dists = [np.linalg.norm(center_1[:2] - center_2[:2]) for center_2 in obj_pcd_centers_2]
                print(f"Dists for obj {i}: {dists}")
                if len(dists) == 0:
                    obj_pcds.append(obj_masked_pcds[i])
                else:
                    j = np.argmin(dists)
                    dist = dists[j]
                    if dist < 0.1:
                        obj_pcds.append(obj_masked_pcds[i] + obj_masked_pcds_2[j])
                    else:
                        obj_pcds.append(obj_masked_pcds[i])

            print("Getting grasps per object...")
            obj_i, filtered_grasp_group = get_obj_grasps(
                grasp_client, obj_masked_rgbds, obj_pcds, scene_pcd
            )

            print("Choosing a grasp for the object")
            chosen_grasp, final_filtered_grasps = robot.select_grasp(filtered_grasp_group)
            grasp_client.visualize_grasp(scene_pcd, final_filtered_grasps)
            grasp_client.visualize_grasp(obj_pcds[obj_i], final_filtered_grasps, name="obj")

            traj = execute_grasp(robot, chosen_grasp, grasp_offset, hori_offset, time_to_go)

            print("Going home")
            robot.go_home()


if __name__ == "__main__":
    main()
