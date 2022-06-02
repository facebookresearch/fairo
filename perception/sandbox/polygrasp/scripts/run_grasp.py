#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

import time
import os

import numpy as np
import sklearn
import matplotlib.pyplot as plt

import torch
import hydra
import omegaconf

import polygrasp
from polygrasp.segmentation_rpc import SegmentationClient
from polygrasp.grasp_rpc import GraspClient
from polygrasp.serdes import load_bw_img

import fairotag


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


def get_obj_grasps(grasp_client, obj_pcds, scene_pcd):
    for obj_i, obj_pcd in enumerate(obj_pcds):
        print(f"Getting obj {obj_i} grasp...")
        grasp_group = grasp_client.get_grasps(obj_pcd)
        filtered_grasp_group = grasp_client.get_collision(grasp_group, scene_pcd)
        if len(filtered_grasp_group) < len(grasp_group):
            print(
                "Filtered"
                f" {len(grasp_group) - len(filtered_grasp_group)}/{len(grasp_group)} grasps"
                " due to collision."
            )
        if len(filtered_grasp_group) > 0:
            return obj_i, filtered_grasp_group
    raise Exception(
        "Unable to find any grasps after filtering, for any of the"
        f" {len(obj_pcds)} objects"
    )


def merge_pcds(pcds, eps=0.1, min_samples=2):
    """Cluster object pointclouds from different cameras based on centroid using DBSCAN; merge when within eps"""
    xys = np.array([pcd.get_center()[:2] for pcd in pcds])
    cluster_labels = (
        sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(xys).labels_
    )

    # Logging
    total_n_objs = len(xys)
    total_clusters = cluster_labels.max() + 1
    unclustered_objs = (cluster_labels < 0).sum()
    print(
        f"Clustering objects from all cameras: {total_clusters} clusters, plus"
        f" {unclustered_objs} non-clustered objects; went from {total_n_objs} to"
        f" {total_clusters + unclustered_objs} objects"
    )

    # Cluster label == -1 when unclustered, otherwise cluster label >=0
    final_pcds = []
    cluster_to_pcd = dict()
    for cluster_label, pcd in zip(cluster_labels, pcds):
        if cluster_label >= 0:
            if cluster_label not in cluster_to_pcd:
                cluster_to_pcd[cluster_label] = pcd
            else:
                cluster_to_pcd[cluster_label] += pcd
        else:
            final_pcds.append(pcd)

    return list(cluster_to_pcd.values()) + final_pcds


def execute_grasp(robot, chosen_grasp, hori_offset, time_to_go):
    traj, success = robot.grasp(
        chosen_grasp, time_to_go=time_to_go, gripper_width_success_threshold=0.001
    )
    print(f"Grasp success: {success}")

    if success:
        print("Placing object in hand to desired pose...")
        curr_pose, curr_ori = robot.get_ee_pose()
        print("Moving up")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0, 0.2]),
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        print("Moving horizontally")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0.0, 0.2]) + hori_offset,
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        print("Moving down")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0.0, 0.05]) + hori_offset,
            orientation=curr_ori,
            time_to_go=time_to_go,
        )

    print("Opening gripper")
    robot.gripper_open()
    curr_pose, curr_ori = robot.get_ee_pose()
    print("Moving up")
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
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_1],
        dtype=np.float64,
    )
    masks_2 = np.array(
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_2],
        dtype=np.float64,
    )

    print("Connect to grasp candidate selection and pointcloud processor")
    segmentation_client = SegmentationClient()
    grasp_client = GraspClient(
        view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path)
    )

    root_working_dir = os.getcwd()
    for outer_i in range(cfg.num_bin_shifts):
        cam_i = outer_i % 2
        print(f"=== Starting bin shift with cam {cam_i} ===")

        # Define some parameters for each workspace.
        if cam_i == 0:
            masks = masks_1
            hori_offset = torch.Tensor([0, -0.4, 0])
        else:
            masks = masks_2
            hori_offset = torch.Tensor([0, 0.4, 0])
        time_to_go = 3

        for i in range(cfg.num_grasps_per_bin_shift):
            # Create directory for current grasp iteration
            os.chdir(root_working_dir)
            timestamp = int(time.time())
            os.makedirs(f"{timestamp}")
            os.chdir(f"{timestamp}")

            print(
                f"=== Grasp {i + 1}/{cfg.num_grasps_per_bin_shift}, logging to"
                f" {os.getcwd()} ==="
            )

            print("Getting rgbd and pcds..")
            rgbd = cameras.get_rgbd()

            rgbd_masked = rgbd * masks[:, :, :, None]

            frt_cams = [fairotag.CameraModule() for _ in range(cameras.n_cams)]
            for frt, intrinsics in zip(frt_cams, cameras.intrinsics):
                frt.set_intrinsics(intrinsics)
            MARKER_LENGTH = 0.05
            MARKER_ID = [0, 1, 2]
            for i in MARKER_ID:
                for frt in frt_cams:
                    frt.register_marker_size(i, MARKER_LENGTH)

            uint_rgbs = rgbd_masked[:,:,:,:3].astype(np.uint8)
            id_to_pose = {}
            for frt, uint_rgb, extrinsics in zip(frt_cams, uint_rgbs, cameras.extrinsic_transforms):
                # import cv2
                # uint_rgb = cv2.imread("/private/home/yixinlin/dev/fairo/perception/sandbox/polygrasp/data/example_markers.png")

                markers = frt.detect_markers(uint_rgb)
                for marker in markers:
                    if marker.pose:
                        homog_translation = np.ones(4)
                        homog_translation[:3] = marker.pose.translation()
                        transformed_pos = extrinsics @ homog_translation
                        id_to_pose[marker.id] = transformed_pos[:3]
            
            import pdb; pdb.set_trace()

            scene_pcd = cameras.get_pcd(rgbd)
            # save_rgbd_masked(rgbd, rgbd_masked)

            print("Segmenting image...")
            unmerged_obj_pcds = []
            for i in range(cameras.n_cams):
                obj_masked_rgbds, obj_masks = segmentation_client.segment_img(
                    rgbd_masked[i], min_mask_size=cfg.min_mask_size
                )
                unmerged_obj_pcds += [
                    cameras.get_pcd_i(obj_masked_rgbd, i)
                    for obj_masked_rgbd in obj_masked_rgbds
                ]
            obj_pcds = merge_pcds(unmerged_obj_pcds)
            if len(obj_pcds) == 0:
                print(
                    f"Failed to find any objects with mask size > {cfg.min_mask_size}!"
                )
                break

            print("Getting grasps per object...")
            obj_i, filtered_grasp_group = get_obj_grasps(
                grasp_client, obj_pcds, scene_pcd
            )

            print("Choosing a grasp for the object")
            chosen_grasp, final_filtered_grasps = robot.select_grasp(
                filtered_grasp_group
            )
            grasp_client.visualize_grasp(scene_pcd, final_filtered_grasps)
            grasp_client.visualize_grasp(
                obj_pcds[obj_i], final_filtered_grasps, name="obj"
            )

            traj = execute_grasp(robot, chosen_grasp, hori_offset, time_to_go)

            print("Going home")
            robot.go_home()


if __name__ == "__main__":
    main()
