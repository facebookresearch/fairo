#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

import random
import torch

import hydra
from polygrasp.pointcloud_rpc import PointCloudClient, RgbdFeaturesPointCloudClient
from polygrasp.grasp_rpc import GraspClient


@hydra.main(config_path="../conf", config_name="run_grasp")
def main(cfg):
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

    import numpy as np
    # masks = np.ones([3, 480, 640])

    masks = np.zeros([3, 480, 640])
    masks[0][70:320, 60:400] = 1
    masks[1][100:350, 300:600] = 1
    # masks[2][130:350, 240:480] = 1

    # Connect to grasp candidate selection and pointcloud processor
    pcd_client = RgbdFeaturesPointCloudClient(camera_intrinsics, camera_extrinsics, masks=masks)
    grasp_client = GraspClient(view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path))

    num_iters = 10
    for i in range(num_iters):
        print(f"Grasp {i + 1} / num_iters")

        print("Getting rgbd and pcds..")
        rgbd = cameras.get_rgbd()
        rgbd_masked = rgbd * masks[:, :, :, None]
        pcds = pcd_client.get_pcd(rgbd)

        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(2, 3)
        # for i in range(3):
        #     axarr[0, i].imshow(rgbd[i, :, :, :3].astype(np.uint8))
        #     axarr[1, i].imshow(rgbd_masked[i, :, :, :3].astype(np.uint8))
        # f.show()
        # import pdb; pdb.set_trace()

        scene_pcd = pcds[0]
        for pcd in pcds[1:]:
            scene_pcd += pcd

        # import pdb; pdb.set_trace()
        # Get RGBD & pointcloud
        print("Segmenting image...")
        labels = pcd_client.segment_img(rgbd_masked[0])
        from matplotlib import pyplot as plt
        obj_to_pcd = {}
        obj_to_grasps = {}
        num_objs = int(labels.max())
        print(f"Number of objs: {num_objs}")
        # for i in range(1, int(num_objs + 1)):
        # if num_objs > 0:
        #     i = 1
        min_mask_size = 2000
        for i in range(1, num_objs + 1):
            obj_mask = labels == i
            obj_mask_size = obj_mask.sum()
            if obj_mask_size < min_mask_size:
                continue

            obj_masked_rgbd = rgbd[0] * obj_mask[:, :, None]

            # plt.imshow(obj_masked_rgbd[:, :, :3])
            # plt.title(f"Object {i}, mask size {obj_mask_size}")
            # plt.show()

            print(f"Getting obj {i} pcd...")
            pcd = pcd_client.get_pcd_i(obj_masked_rgbd, 0)
            print(f"Getting obj {i} grasp...")
            grasp_group = grasp_client.get_grasps(pcd)

            obj_to_pcd[i] = pcd
            obj_to_grasps[i] = grasp_group
            break

            # grasp_client.visualize_grasp(scene_pcd, grasp_group, plot=True)
        if len(obj_to_pcd) == 0:
            print(f"Failed to find any objects with mask size > {min_mask_size}!")
            continue

        # import pdb; pdb.set_trace()

        # vis = grasp_client.visualize(scene_pcd, render=True)
        # import pdb; pdb.set_trace()
        # rgb, d, intrinsics = grasp_client.get_rgbd(vis)


        # grasp_group = grasp_client.get_grasps(pcd)
        # grasp_client.visualize_grasp(
        #     scene_pcd, grasp_group, render=True, save_view=False, plot=True
        # )

        # Get grasps per object
        # obj_to_pcd = pcd_client.segment_pcd(scene_pcd)
        # obj_to_grasps = {obj: grasp_client.get_grasps(pcd) for obj, pcd in obj_to_pcd.items()}

        # Pick a random object to grasp
        # curr_obj, curr_grasps = random.choice(list(obj_to_grasps.items()))
        # curr_obj, curr_grasps = 1, obj_to_grasps[1]
        # print(f"Picking object with ID {curr_obj}")
        curr_grasps = grasp_group

        # Choose a grasp for this object
        # TODO: scene-aware motion planning for grasps
        grasp_client.visualize_grasp(scene_pcd, curr_grasps, plot=True)
        chosen_grasp = robot.select_grasp(curr_grasps, scene_pcd)

        # Execute grasp
        traj, success = robot.grasp(chosen_grasp)
        print(f"Grasp success: {success}")

        if success:
            print(f"Moving end-effector up")
            curr_pose, curr_ori = robot.get_ee_pose()
            states = robot._move_until_success(position=curr_pose + torch.Tensor([0, 0, 0.2]), orientation=curr_ori, time_to_go=3)
            states = robot._move_until_success(position=curr_pose + torch.Tensor([0, 0.4, 0.2]), orientation=curr_ori, time_to_go=3)
            states = robot._move_until_success(position=curr_pose + torch.Tensor([0, 0.4, 0.05]), orientation=curr_ori, time_to_go=3)

        robot.gripper_open()
        curr_pose, curr_ori = robot.get_ee_pose()
        states = robot._move_until_success(position=curr_pose + torch.Tensor([0, 0.0, 0.2]), orientation=curr_ori, time_to_go=3)
        robot.go_home()
            # robot.move_to_ee_pose(torch.Tensor([0, 0, 0.1]), delta=True)
            # robot.move_to_ee_pose(torch.Tensor([0, 0, -0.1]), delta=True)


if __name__ == "__main__":
    main()
