import os
import copy
import time
import shutil
import json
import math
import sys
import cv2
import numpy as np
import open3d as o3d
from droidlet import dashboard
from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover
import droidlet.lowlevel.hello_robot.o3d_utils as o3d_utils
from droidlet.lowlevel.hello_robot.remote.remote_hello_saver import HelloLogger
from droidlet.lowlevel.hello_robot.remote.slam_service import SLAM
from droidlet.dashboard.o3dviz import O3DViz

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    dashboard.start()
    if sys.platform == "darwin":
        webrtc_streaming=False
    else:
        webrtc_streaming=True
    o3dviz = O3DViz(webrtc_streaming)
    o3dviz.start()


    root = '/Users/soumith/hello-lidar-slam/hello_data_log_1649893783.6101801/1'
    replay = HelloLogger(root, replay=True)
    intrinsic_mat = replay.load_cam_intrinsic()
    CH = 480
    CW = 640

    fx = intrinsic_mat[0, 0]
    fy = intrinsic_mat[1, 1]
    ppx = intrinsic_mat[0, 2]
    ppy = intrinsic_mat[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(CW, CH, fx, fy, ppx, ppy)

from droidlet.event import sio

def load_data(idx):
    frame_id, timestamp, lidar, rgb, depth, base_state, cam_pan, cam_tilt, cam_transform, pose_dict = replay.load(idx)
    
    # rgb_depth = HelloRobotMover.compute_pcd(rgb, depth, rot, trans, base_state, uv_one_in_cam)
    opcd = o3d_utils.compute_pcd(rgb, depth,
                                 cam_transform,
                                 base_state, intrinsic,
                                 compressed=True,)

    return rgb, depth, opcd, base_state, cam_transform

from droidlet.lowlevel.robot_coordinate_utils import (
    xyz_pyrobot_to_canonical_coords,
    base_canonical_coords_to_pyrobot_coords,
)

def visual_registration_rgbd(prev_rgb, prev_depth, cur_rgb, cur_depth, base_state, cam_transform):
    source_rgbd_image = o3d_utils.to_o3d_rgbd(prev_rgb, prev_depth)
    target_rgbd_image = o3d_utils.to_o3d_rgbd(cur_rgb, cur_depth)
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    [success_color_term, trans_color_term, info] = \
        o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image,
            intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            option)
    [success_hybrid_term, trans_hybrid_term, info] = \
        o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option)
    print(trans_color_term, trans_hybrid_term, info)
    # extrinsic = o3d_utils.compute_extrinsic(base_state, cam_transform)

def visual_registration_pcd(prev_pcd, cpcd):
    source = prev_pcd
    target = cpcd

    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    # print("Colored point cloud registration ...\n")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])
        
        # print("1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        
        # print("2. Estimate normal")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        
        # print("3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
        current_transformation = result_icp.transformation
        # print(result_icp, "\n")
    new_target = copy.deepcopy(target)
    new_target.transform(np.linalg.inv(current_transformation))
    return new_target

if __name__ == "__main__":
    opcd = o3d.geometry.PointCloud()
    robot_height = 141  # cm
    min_z = 20  # because of the huge spatial variance in realsense readings
    max_z = robot_height + 5  # cm
    slam = SLAM(
        None,
        obstacle_threshold=10,
        agent_min_z=min_z,
        agent_max_z=max_z,
        robot_rad = 1,
        resolution=5,
    )

    prev_rgb, prev_depth, prev_pcd = None, None, None
    for i in range(0, 1900, 10):
        tm = time.time()
        rgb, depth, cpcd, base_state, cam_transform = load_data(i)

        if prev_rgb is not None:
            odometry = visual_registration_rgbd(prev_rgb, prev_depth,
                                           rgb, depth,
                                           base_state, cam_transform)
        if prev_pcd is not None:
            try:
                new_pcd = visual_registration_pcd(prev_pcd, cpcd)
                cpcd = new_pcd
            except:
                print("Failed visual registration for frame ", i)
                pass
            
        # prev_rgb = rgb # uncomment this for rgbd based odometry refinement
        prev_depth = depth
        # prev_pcd = cpcd # uncomment this for pcd based odometry refinement

        odometry = base_state
        
        opcd += cpcd
        opcd = opcd.voxel_down_sample(0.05)
        o3dviz.put('pointcloud', opcd)

        o3dviz.add_robot(odometry, canonical=False, base=False)

        slam.update_map(pcd=np.asarray(cpcd.points))
        x, y, yaw = odometry.tolist()
        cordinates_in_robot_frame = slam.get_map()
        cordinates_in_standard_frame = [
            xyz_pyrobot_to_canonical_coords(list(c) + [0.0]) for c in cordinates_in_robot_frame
        ]
        cordinates_in_standard_frame = [(c[0], c[2]) for c in cordinates_in_standard_frame]
        
        sio.emit(
            "map",
            {"x": -y, "y": x, "yaw": yaw, "map": cordinates_in_standard_frame},
        )
        
        print("Frame {}, {} ms".format(i, round((time.time() - tm) * 1000)))


    o3dviz.join()
