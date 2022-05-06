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
    opcd = o3d_utils.compute_pcd(
        rgb, depth,
        cam_transform,
        base_state, intrinsic,
        compressed=True,
        in_base_frame=True,
    )

    return rgb, depth, opcd, base_state, cam_transform

from droidlet.lowlevel.robot_coordinate_utils import (
    xyz_pyrobot_to_canonical_coords,
    base_canonical_coords_to_pyrobot_coords,
)
from droidlet.perception.robot.visual_registration import pointcloud_registration_global
from droidlet.perception.robot.visual_registration import SLAM as DSLAM

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

    rgb, depth, cpcd, base_state, cam_transform = load_data(0)
    odepth = o3d.t.geometry.Image(o3d.core.Tensor(depth))
    dslam = DSLAM(intrinsic, odepth)

    prev_rgb, prev_depth, prev_pcd = None, None, None
    for i in range(0, 1900, 2):
        tm = time.time()
        rgb, depth, cpcd, base_state, cam_transform = load_data(i)

        rgb_u8 = np.ascontiguousarray(rgb[:, :, [2, 1, 0]], dtype=np.uint8)

        orgb = o3d.t.geometry.Image(o3d.core.Tensor(rgb_u8))
        odepth = o3d.t.geometry.Image(o3d.core.Tensor(depth))
        orgbd = o3d.t.geometry.RGBDImage(orgb, odepth)
        intrinsic_ = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                                    o3d.core.Dtype.Float64)
        
        # opcdn = o3d.t.geometry.PointCloud.create_from_rgbd_image(orgbd,
        #                                                          intrinsic_,
        #                                                          # extrinsics=(with default value),
        #                                                          depth_scale=1000.0,
        #                                                          depth_max=10.0,
        #                                                          stride=1,
        #                                                          with_normals=False)
        # o3dviz.put("pcd", opcdn.to_legacy())
        volume, poses, config = dslam.update_map(orgb, odepth, first=(i==0))

        if i % 100 == 0:
            mesh = volume.extract_triangle_mesh(
                weight_threshold=config.surface_weight_thr)
            mesh = mesh.to_legacy()
            o3dviz.put("mesh", mesh)
        # new_pcd = volume.extract_point_cloud(weight_threshold=config.surface_weight_thr).to_legacy()
        # o3dviz.add_robot(base_state, canonical=False, base=False)
        # o3dviz.put('pointcloud', new_pcd)
        # cpcd.transform(np.linalg.inv(poses[-1]))


        # if prev_rgb is not None:
        #     odometry = visual_registration_rgbd(prev_rgb, prev_depth,
        #                                         rgb, depth,
        #                                         base_state, cam_transform)
        # if prev_pcd is not None:
        #     new_pcd = pointcloud_registration_global(cpcd, prev_pcd)
        #     cpcd = new_pcd
        #     try:
        #         1+1
        #     except:
        #         print("Failed visual registration for frame ", i)
        #         pass
            
        # # prev_rgb = rgb # uncomment this for rgbd based odometry refinement
        # # prev_depth = depth
        # # prev_pcd = cpcd # uncomment this for pcd based odometry refinement

        # odometry = base_state
        
        # opcd += cpcd
        # opcd = opcd.voxel_down_sample(0.05)
        # # o3dviz.put('pointcloud', opcd)

        # o3dviz.add_robot(odometry, canonical=False, base=False)

        # slam.update_map(pcd=np.asarray(cpcd.points))
        # x, y, yaw = odometry.tolist()
        # cordinates_in_robot_frame = slam.get_map()
        # cordinates_in_standard_frame = [
        #     xyz_pyrobot_to_canonical_coords(list(c) + [0.0]) for c in cordinates_in_robot_frame
        # ]
        # cordinates_in_standard_frame = [(c[0], c[2]) for c in cordinates_in_standard_frame]
        
        # sio.emit(
        #     "map",
        #     {"x": -y, "y": x, "yaw": yaw, "map": cordinates_in_standard_frame},
        # )
        
        print("Frame {}, {} ms".format(i, round((time.time() - tm) * 1000)))


    o3dviz.join()
