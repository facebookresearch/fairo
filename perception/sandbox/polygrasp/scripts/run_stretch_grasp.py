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
from home_robot.utils.pose import to_pos_quat
from home_robot.utils.numpy import to_npy_file
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
# import a simple ROS grasp client for now
from home_robot.ros.grasp_helper import GraspClient as RosGraspClient


USE_POLYGRASP = False


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
    # rob.goto(q, move_base=False, wait=True)
    # rob.look('tool')
    # Send it to lift pose + wait
    #q, _ = rob.update()
    q[HelloStretchIdx.ARM] = 0.06
    #q[HelloStretchIdx.LIFT] = 0.35
    q[HelloStretchIdx.LIFT] = 0.5
    model.update_gripper(q, open=True)
    rob.goto(q, move_base=False, wait=False, verbose=False)
    return rob, q


@hydra.main(config_path="../conf", config_name="run_grasp")
def main(cfg):
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node('hello_stretch_ros_test')

    # Create the robot
    visualize = False
    rob, q = init_robot(visualize=visualize)
    model = rob.get_model()  # get the planning model in case we need it

    # Get a couple camera listeners
    rgb_cam = rob.rgb_cam
    dpt_cam = rob.dpt_cam
    dpt_cam.far_val = 1.5
    min_grasp_score = 0.

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

    H0 = int(H / 4)
    # H1 = int(3 * H / 4)
    H1 = int(H0 + W)
    xyz = xyz[H0:H1]
    rgb = rgb[H0:H1]
    dpt = dpt[H0:H1]
    H2 = xyz.shape[0]

    show_imgs = False
    show_pcs = False
    show_masks = False
    show_grasps = True
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
        xyz2 = xyz.reshape(-1, 3)
        xyz2 = tra.transform_points(xyz2 @ R_stretch_camera.T, camera_pose)
        hrimg.show_point_cloud(xyz2, rgb / 255., orig=np.zeros(3))
        # xyz = xyz.reshape(H, W, 3)

    print("Connect to grasp candidate selection and pointcloud segmenter...")
    segmentation_client = SegmentationClient()
    print("... segmentation connected")
    if USE_POLYGRASP:
        grasp_client = GraspClient(
            view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path)
        )
        print("... grasps connected")
    else:
        grasp_client = RosGraspClient()
    rgbd = np.concatenate([rgb, xyz], axis=-1)
    seg_only = False
    if not seg_only:
        print("Image was too big for your tiny little GPU!")
        print(" - RGBD image of shape:", rgbd.shape)
        seg_rgbd = cv2.resize(rgbd, [int(H2 / 4), int(W / 4)], interpolation=cv2.INTER_NEAREST)
        # seg_depth = cv2.resize(dpt, [int(W / 4), int(H / 4)], interpolation=cv2.INTER_NEAREST)
        print(" - Resized to", seg_rgbd.shape)
        resized = True
    else:
        seg_rgbd = rgbd
        resized = False
        # seg_depth = dpt
    seg_rgb, seg_xyz = rgbd[:, :, :3], rgbd[:, :, 3:]

    min_points = 200
    segment = True
    print("Segment...")
    print(" - segmented image shape =", seg_rgbd.shape)
    obj_pcds = []
    obj_masked_rgbds, obj_masks = segmentation_client.segment_img(seg_rgbd,
                                                                  min_mask_size=cfg.min_mask_size)
    print("Found:", len(obj_masked_rgbds))

    #mask_scene = np.ones((int(H / 2), int(W / 2)))
    mask_valid = dpt > dpt_cam.near_val  # remove bad points
    mask_scene = mask_valid  # initial mask has to be good
    mask_scene = mask_scene.reshape(-1)

    orig_xyz = xyz.copy()
    orig_rgb = rgb.copy()

    # Convert XYZ into world frame
    xyz = xyz.reshape(-1, 3)
    # xyz = trimesh.transform_points(xyz @ R_stretch_camera.T, camera_pose)
    rgb = rgb.reshape(-1, 3) / 255.

    # Loop to get masks
    seg = np.zeros((rgbd.shape[0], rgbd.shape[1]), dtype=np.uint16)
    #for i, (rgbd, mask) in enumerate(zip(obj_masked_rgbds, obj_masks)):
    for i, mask in enumerate(obj_masks):
        # Upscale the mask back to full size
        mask = mask.astype(np.uint8)  # needs to be non-boolean for opencv
        mask = cv2.resize(mask, [H2, W], interpolation=cv2.INTER_NEAREST)
        seg[mask > 0] = i + 1
        # Smooth mask over valid pixels only
        mask1, mask2 = hrimg.smooth_mask(np.bitwise_and(mask_valid, mask))
        mask_scene = np.bitwise_and(mask_scene > 0, mask1.reshape(-1) == 0)
        # Make sure enough points are observed
        if np.sum(mask2) < min_points: continue
        # Correct hte mask and display - convert to pt cloud
        masked_rgb = (rgbd[:, :, :3] / 255.) * mask2[:, :, None].repeat(3, axis=-1)
        # o3d.visualization.draw_geometries([obj_pcd])
        obj_pcd = hrimg.to_o3d_point_cloud(xyz, rgb, mask2)

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

    # Save some data
    to_npy_file('stretch', xyz=orig_xyz, rgb=orig_rgb, depth=dpt, xyz_color=orig_rgb, seg=seg, K=rgb_cam.get_K())
    if seg_only:
        return

    # Apply the mask - no more bad poitns
    xyz = xyz[mask_scene]
    rgb = rgb[mask_scene]
    scene_pcd = hrimg.to_o3d_point_cloud(xyz, rgb)

    # Transform point clouds with help from trimesh
    geoms = [scene_pcd] + obj_pcds
    pose_xyz = []
    pose_rgb = []

    print("Get grasps...")
    if USE_POLYGRASP:
        obj_i, predicted_grasps = grasp_client.get_obj_grasps(
            obj_pcds, scene_pcd
        )
        # print("Visualize...")
        # grasp_client.visualize_grasp(obj_pcds[obj_i], predicted_grasps, n=len(predicted_grasps), render=False, save_view=False)
        # print("...done.")
        scores = [grasp.score for grasp in predicted_grasps]
    else:
        predicted_grasps = grasp_client.request(orig_xyz, orig_rgb, seg, frame=rgb_cam.get_frame())

    T_fix_camera = np.eye(4)
    T_fix_camera[:3, :3] = R_stretch_camera
    offset = np.eye(4)
    grasps = []
    for i, (score, grasp) in enumerate(zip(scores, predicted_grasps)):
        # import pdb; pdb.set_trace()
        if USE_POLYGRASP:
            pose = np.eye(4)
            pose[:3, :3] = grasp.rotation_matrix
            pose[:3, 3] = grasp.translation
        else:
            pose = grasp
        if score < min_grasp_score:
            continue

        # camera setup
        #R_camera = camera_pose[:3, :3]
        #R_cam_to_grasp = grasp.rotation_matrix @ R_camera
        #angle_dist = np.abs(angles)
        #if angle_dist[0] > 1.5 or angle_dist[1] > 1.5:
        #    # angle relative to camera too big to trust it
        #    continue

        # Get angles in world frame
        pose = camera_pose @ T_fix_camera @ pose
        # angles = tra.euler_from_matrix(pose)
        # z direction for grasping
        dirn = pose[:3, 2]
        axis = np.array([0, 0, 1])
        # angle between these two is...
        theta = np.abs(np.arccos(dirn @ axis / (np.linalg.norm(dirn)))) / np.pi
        print(i, "score =", grasp.score, theta) #, "orientation =", angles)
        # Reject grasps that arent top down for now
        # if theta < 0.75: continue
        grasps.append(pose)

        # pose = camera_pose @ pose @ T_fix_stetch_camera
        M = 10
        for j in range(1, M + 1):
            offset[2, 3] = (-0.005 * j) + -0.2
            _pose = pose @ offset
            pose_xyz.append(_pose[:3, 3])
            pose_rgb.append(np.array([1., 1 - (float(j) / M), theta]))
            #xyz = np.concatenate([xyz, pose[:3, 3][None]], axis=0)
            #rgb = np.concatenate([rgb, np.array([[1., 1 - (float(j) / M), 0]])], axis=0) # red

    for i, geom in enumerate(geoms):
        # Fade out the rgb on the scene
        if i == 0:
            rgb = np.asarray(geom.colors)
            rgb = rgb * 0.5
            rgb[:, 0] = 1
            geom.colors = o3d.utility.Vector3dVector(rgb)
        xyz = np.asarray(geom.points)
        xyz = trimesh.transform_points(xyz @ R_stretch_camera.T, camera_pose)
        #xyz = (xyz @ R_stretch_camera.T)
        geom.points = o3d.utility.Vector3dVector(xyz)

    # Add final things to visualize point cloud problems
    pose_xyz = np.array(pose_xyz)
    pose_rgb = np.array(pose_rgb)
    grasp_pcd = hrimg.to_o3d_point_cloud(pose_xyz, pose_rgb)
    geoms.append(grasp_pcd)
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.zeros(3))
    geoms.append(coords)
    if show_grasps:
        o3d.visualization.draw_geometries(geoms)

    grasp_offset = np.eye(4)
    grasp_offset[2, 3] = -0.28
    for i, grasp in enumerate(grasps):
        grasps[i] = grasp @ grasp_offset

    # Now execute planning
    offset = np.eye(4)
    offset[2, 3] = -0.1

    print("=========== grasps =============")
    return
    print("find a grasp that doesnt move the base...")
    for grasp in grasps:
        grasp_pose = to_pos_quat(grasp)
        standoff_pose = to_pos_quat(grasp @ offset)
        qi = model.static_ik(grasp_pose, q)
        if qi is not None:
            model.set_config(qi)
            input('-- full body --')
        q0 = np.pi / 2
        q1 = model.static_ik(standoff_pose, q)
        if q1 is not None:
            q2 = model.static_ik(grasp_pose, q1)
            if q2 is not None:
                q1[HelloStretchIdx.ARM] -= 0.04
                q2[HelloStretchIdx.ARM] -= 0.04
                model.set_config(q1)
                # TODO - what is wrong with base corrections
                eq1 = q1[HelloStretchIdx.BASE_THETA] - q0
                eq2 = q2[HelloStretchIdx.BASE_THETA] - q0 # q[HelloStretchIdx.BASE_THETA]
                print("theta 1 =", eq1)
                # input('-- static - standoff --')
                model.set_config(q2)
                print("theta 2 =", eq2)
                # input('-- static - grasp --')
                if np.abs(eq1) < 0.075 and np.abs(eq2) < 0.075:
                    # go to the grasp and try it
                    q[HelloStretchIdx.LIFT] = 1.0
                    rob.goto(q, move_base=False, wait=True, verbose=False)
                    input('--> go high')
                    q_pre = q.copy()
                    q_pre[5:] = q1[5:]
                    q_pre = model.update_gripper(q_pre, open=True)
                    rob.goto(q_pre, move_base=False, wait=False, verbose=False)
                    input('--> gripper ready')
                    q1 = model.update_gripper(q1, open=True)
                    rob.goto(q1, move_base=False, wait=False, verbose=False)
                    input('--> go standoff')
                    q2 = model.update_gripper(q2, open=True)
                    rob.goto(q2, move_base=False, wait=False, verbose=False)
                    input('--> go grasp')
                    q2 = model.update_gripper(q2, open=False)
                    rob.goto(q2, move_base=False, wait=False, verbose=False)
                    rospy.sleep(2.)
                    q = model.update_gripper(q, open=False)
                    rob.goto(q, move_base=False, wait=False, verbose=False)
                    input('--> go high again')
                    break
        

if __name__ == "__main__":
    main()

