#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

import rospy
import time
import os

# Robot planning tools
from home_robot.hardware.stretch_ros import HelloStretchROSInterface
from home_robot.motion.robot import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.ros.path import get_package_path
from home_robot.ros.camera import RosCamera
import home_robot.utils.image as hrimg
import trimesh
import trimesh.transformations as tra

import numpy as np
import sklearn
import matplotlib.pyplot as plt

import torch
import hydra
import omegaconf

from polygrasp.segmentation_rpc import SegmentationClient
from polygrasp.grasp_rpc import GraspClient
from polygrasp.serdes import load_bw_img

from polygrasp.robot_interface import GraspingRobotInterface
import graspnetAPI
import open3d as o3d
from typing import List

import cv2
import matplotlib.pyplot as plt

"""
Manual installs needed for:
    tracikpy
    home_robot
"""


def init_robot(visualize=False):
    # Create the robot
    print("Create ROS interface")
    rob = HelloStretchROSInterface(visualize_planner=visualize,
                                   root=get_package_path())
    print("Wait...")
    rospy.sleep(0.5)  # Make sure we have time to get ROS messages
    for i in range(1):
        q = rob.update()
        print(rob.get_base_pose())
    print("--------------")
    print("We have updated the robot state. Now test goto.")

    home_q = STRETCH_HOME_Q
    model = rob.get_model()
    # q = model.update_look_front(home_q.copy())
    # rob.goto(q, move_base=False, wait=True)

    # Robot - look at the object because we are switching to grasping mode
    # Send robot to home_q + wait
    q = model.update_look_at_ee(home_q.copy())
    rob.goto(q, move_base=False, wait=True)
    # rob.look('tool')
    # Send it to lift pose + wait
    #q, _ = rob.update()
    q[HelloStretchIdx.ARM] = 0.06
    q[HelloStretchIdx.LIFT] = 0.35
    rob.goto(q, move_base=False, wait=True, verbose=False)
    return rob

@hydra.main(config_path="../conf", config_name="run_grasp")
def main(cfg):
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node('hello_stretch_ros_test')

    # Create the robot
    rob = init_robot()
    model = rob.get_model()  # get the planning model in case we need it

    # Get a couple camera listeners
    rgb_cam = RosCamera('/camera/color')
    dpt_cam = RosCamera('/camera/aligned_depth_to_color', buffer_size=5)
    dpt_cam.far_val = 1.5

    # Get images from the robot
    rgb_cam.wait_for_image()
    dpt_cam.wait_for_image()
    rospy.sleep(1.)

    print("rgb frame =", rgb_cam.get_frame())
    print("dpt frame =", dpt_cam.get_frame())
    camera_pose = rob.get_pose(rgb_cam.get_frame())
    print("camera rgb pose:")
    print(camera_pose)

    # Now get the images for each one
    rgb = rgb_cam.get()
    dpt = dpt_cam.get_filtered()
    xyz = dpt_cam.depth_to_xyz(dpt_cam.fix_depth(dpt))
    # Get xyz in base coords for later
    rgb, dpt, xyz = [np.rot90(np.fliplr(np.flipud(x))) for x in [rgb, dpt, xyz]]
    H, W = rgb.shape[:2]
    xyz = xyz.reshape(-1, 3)
    # Rotate the sretch camera so that top of image is "up"
    R_stretch_camera = tra.euler_matrix(0, 0, -np.pi/2)[:3, :3]
    xyz = xyz @ R_stretch_camera
    xyz = xyz.reshape(H, W, 3)

    show_imgs = False
    show_pcs = False
    show_masks = False
    if show_imgs:
        plt.figure()
        plt.subplot(1,3,1); plt.imshow(rgb)
        plt.subplot(1,3,2); plt.imshow(dpt)
        plt.subplot(1,3,3); plt.imshow(xyz)
        plt.show()

    # TODO remove debug code
    # Use to show the point cloud if you want to see it
    if show_pcs:
        # base_xyz = base_xyz @ tra.euler_matrix(0, 0, np.pi/2)[:3, :3]
        # hrimg.show_point_cloud(base_xyz, rgb / 255., orig=np.zeros(3))
        # TODO remove dead code
        # Convert into base coords
        xyz = xyz.reshape(-1, 3)
        xyz = tra.transform_points(xyz @ R_stretch_camera.T, camera_pose)
        hrimg.show_point_cloud(xyz, rgb / 255., orig=np.zeros(3))
        xyz = xyz.reshape(H, W, 3)

    print("Connect to grasp candidate selection and pointcloud processor")
    segmentation_client = SegmentationClient()
    grasp_client = GraspClient(
        view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path)
    )
    rgbd = np.concatenate([rgb, xyz], axis=-1)
    print("RGBD image of shape:", rgbd.shape)
    rgbd = cv2.resize(rgbd, [int(W / 2), int(H / 2)], interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(dpt, [int(W / 2), int(H / 2)], interpolation=cv2.INTER_NEAREST)
    rgb, xyz = rgbd[:, :, :3], rgbd[:, :, 3:]
    print("Resized to", rgbd.shape)

    min_points = 200
    segment = True
    print("Segment...")
    obj_pcds = []
    obj_masked_rgbds, obj_masks = segmentation_client.segment_img(rgbd,
                                                                  min_mask_size=cfg.min_mask_size)
    print("Found:", len(obj_masked_rgbds))

    #mask_scene = np.ones((int(H / 2), int(W / 2)))
    mask_valid = depth > dpt_cam.near_val  # remove bad points
    mask_scene = mask_valid  # initial mask has to be good
    mask_scene = mask_scene.reshape(-1)
    # Convert XYZ into world frame
    xyz = xyz.reshape(-1, 3)
    xyz = trimesh.transform_points(xyz @ R_stretch_camera.T, camera_pose)
    rgb = rgb.reshape(-1, 3) / 255.

    # Loop to get masks
    for rgbd, mask in zip(obj_masked_rgbds, obj_masks):
        # Smooth mask over valid pixels only
        mask1, mask2 = hrimg.smooth_mask(np.bitwise_and(mask_valid, mask))
        mask_scene = np.bitwise_and(mask_scene > 0, mask1.reshape(-1) == 0)
        # Make sure enough points are observed
        if np.sum(mask2) < min_points: continue
        # Correct hte mask and display - convert to pt cloud
        masked_rgb = (rgbd[:, :, :3] / 255.) * mask2[:, :, None].repeat(3, axis=-1)
        # o3d.visualization.draw_geometries([obj_pcd])
        obj_pcd = hrimg.to_o3d_point_cloud(xyz, rgb / 255., mask2)

        obj_pcds.append(obj_pcd)
        # o3d.visualization.draw_geometries([obj_pcd])
        if show_masks:
            plt.figure()
            plt.subplot(231); plt.imshow(mask) # rgbd[:, :, :3])
            plt.subplot(232); plt.imshow(mask2)
            plt.subplot(233); plt.imshow(mask_scene) # rgbd[:, :, 3:])
            plt.subplot(235); plt.imshow(masked_rgb)
            plt.subplot(236); plt.imshow(rgb) # rgbd[:, :, 3:])
            plt.show()

    # Apply the mask - no more bad poitns
    xyz = xyz[mask_scene]
    rgb = rgb[mask_scene]
    scene_pcd = hrimg.to_o3d_point_cloud(xyz, rgb)

    obj_i, filtered_grasp_group = grasp_client.get_obj_grasps(
        obj_pcds, scene_pcd
    )

    # Transform point clouds with help from trimesh
    geoms = [scene_pcd] + obj_pcds
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3))
    geoms.append(coords)
    o3d.visualization.draw_geometries(geoms)
    pose_xyz = []
    pose_rgb = []

    offset = np.eye(4)
    for i, grasp in enumerate(filtered_grasp_group):
        print(i, grasp.translation)
        # import pdb; pdb.set_trace()
        pose = np.eye(4)
        pose[:3, :3] = grasp.rotation_matrix
        pose[:3, 3] = grasp.translation
        # pose = camera_pose @ pose @ T_fix_stetch_camera
        M = 10
        for j in range(1, M + 1):
            offset[2, 3] = -0.005 * j
            pose = pose @ offset
            pose_xyz.append(pose[:3, 3])
            pose_rgb.append(np.array([1., 1 - (float(j) / M), 0]))
            #xyz = np.concatenate([xyz, pose[:3, 3][None]], axis=0)
            #rgb = np.concatenate([rgb, np.array([[1., 1 - (float(j) / M), 0]])], axis=0) # red

    # pose_xyz = trimesh.transform_points(pose_xyz @ R_stretch_camera.T, camera_pose)
    pose_xyz = np.array(pose_xyz)
    pose_rgb = np.array(pose_rgb)

    #scene_pcd.points = o3d.utility.Vector3dVector(xyz)
    #scene_pcd.colors = o3d.utility.Vector3dVector(rgb)
    grasp_pcd = hrimg.to_o3d_point_cloud(pose_xyz, pose_rgb)
    geoms.append(grasp_pcd)
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()

