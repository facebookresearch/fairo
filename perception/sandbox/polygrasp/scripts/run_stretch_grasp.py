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


def init_robot():
    # Create the robot
    print("Create ROS interface")
    rob = HelloStretchROSInterface(visualize_planner=False, root=get_package_path())
    print("Wait...")
    rospy.sleep(0.5)  # Make sure we have time to get ROS messages
    for i in range(1):
        q = rob.update()
        print(rob.get_base_pose())
    print("--------------")
    print("We have updated the robot state. Now test goto.")

    home_q = STRETCH_HOME_Q
    model = rob.get_model()
    q = model.update_look_front(home_q.copy())
    rob.goto(q, move_base=False, wait=True)

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
    rgb_cam = RosCamera('/camera/color', flipxy=False)
    dpt_cam = RosCamera('/camera/aligned_depth_to_color', flipxy=False)

    # Get images from the robot
    rgb_cam.wait_for_image()
    dpt_cam.wait_for_image()
    print("rgb frame =", rgb_cam.get_frame())
    print("dpt frame =", dpt_cam.get_frame())
    pose = rob.get_pose(rgb_cam.get_frame())
    print("rgb pose:")
    print(pose)

    # Now get the images for each one
    rgb = rgb_cam.get()
    dpt = dpt_cam.fix_depth(dpt_cam.get())
    xyz = dpt_cam.depth_to_xyz(dpt)
    rgb, dpt, xyz = [np.rot90(np.fliplr(np.flipud(x))) for x in [rgb, dpt, xyz]]
    H, W = rgb.shape[:2]
    xyz = xyz.reshape(-1, 3)
    xyz = xyz @ tra.euler_matrix(0, 0, -np.pi/2)[:3, :3]
    xyz = xyz.reshape(H, W, 3)

    show_imgs = False
    show_pcs = False
    if show_imgs:
        plt.figure()
        plt.subplot(1,3,1); plt.imshow(rgb)
        plt.subplot(1,3,2); plt.imshow(dpt)
        plt.subplot(1,3,3); plt.imshow(xyz)
        plt.show()

    # TODO remove debug code
    # Use to show the point cloud if you want to see it
    if show_pcs:
        hrimg.show_point_cloud(xyz, rgb / 255., orig=np.zeros(3))
        # import pdb; pdb.set_trace()

    print("Connect to grasp candidate selection and pointcloud processor")
    segmentation_client = SegmentationClient()
    grasp_client = GraspClient(
        view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path)
    )
    rgbd = np.concatenate([rgb, xyz], axis=-1)
    print("RGBD image of shape:", rgbd.shape)
    rgbd = cv2.resize(rgbd, [int(W / 2), int(H / 2)])
    print("Resized to", rgbd.shape)

    min_points = 50
    segment = True
    if segment:
        print("Segment...")
        obj_masked_rgbds, obj_masks = segmentation_client.segment_img(rgbd, min_mask_size=cfg.min_mask_size)
        for rgbd, mask in zip(obj_masked_rgbds, obj_masks):
            mask2 = hrimg.smooth_mask(mask)
            if np.sum(mask2) < min_points: continue
            masked_rgb = (rgbd[:, :, :3] / 255.) * mask2[:, :, None].repeat(3, axis=-1)
            plt.figure()
            plt.subplot(221); plt.imshow(mask) # rgbd[:, :, :3])
            plt.subplot(222); plt.imshow(mask2)
            plt.subplot(223); plt.imshow(masked_rgb)
            plt.subplot(224); plt.imshow(rgb) # rgbd[:, :, 3:])
            plt.show()
    obj_i, filtered_grasp_group = grasp_client.get_obj_grasps(
        obj_pcds, scene_pcd
    )
    

if __name__ == "__main__":
    main()

