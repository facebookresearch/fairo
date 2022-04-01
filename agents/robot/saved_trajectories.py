import os
import copy
import time
import shutil
import json
import math
import cv2
import numpy as np
import open3d as o3d
from droidlet import dashboard
from droidlet.dashboard.o3dviz import o3dviz
from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover

if __name__ == "__main__":
    o3dviz.start()

height, width = 480, 640

def compute_uvone(height, width):
    # 640 / 480, so ppx and ppy are the center of the camera
    # realsense DEPTH_UNITS is 0.001, i.e. it's depth units are in mm
    fx, fy = 605.2880249, 605.65637207
    cx, cy = 319.11114502, 239.48382568
    intrinsic_mat = np.array([[  fx, 0., cx],
                              [  0., fy, cy],
                              [  0., 0., 1.]])
    pintrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    uv_one_in_cam = HelloRobotMover.compute_uvone(intrinsic_mat, height, width)
    return uv_one_in_cam, pintrinsic

uv_one_in_cam, intrinsic = compute_uvone(height, width)


def load_data(idx):
    # root = "/home/soumith/appended/baseline/"
    root = "/home/soumith/collision/hello_data_log_1637169407.7004237/17"
    # root = "/home/soumith/collision/hello_data_log_1638331896.0773966/1"
    # root = "/home/soumith/collision/hello_data_log_1638338130.298805/2"

    rgb_path = os.path.join(root, "rgb", str(idx) + ".jpg")
    depth_path = os.path.join(root, "depth", str(idx) + ".npy")
    odo_path = os.path.join(root, "data.json")

    with open(odo_path) as f:
        data = json.load(f)

    odometry = data[str(idx)]
    base_state = np.array(odometry["base_xyt"])
    cam_transform = np.array(odometry["cam_transform"])
    cam_pan_tilt = np.array(odometry["cam_pan_tilt"])
    print("cam pan_tilt", cam_pan_tilt)

    rgb = cv2.imread(rgb_path)
    depth = np.load(depth_path)

    # unrotate to compute point-cloud
    rgb = np.rot90(rgb, k=-1, axes=(1,0))
    depth = np.rot90(depth, k=-1, axes=(1,0))

    # compute point-cloud
    rgb_u8 = np.ascontiguousarray(rgb[:, :, [2, 1, 0]], dtype=np.uint8)
    depth_u16 = np.ascontiguousarray(depth, dtype=np.float32)

    orgb = o3d.geometry.Image(rgb_u8)
    odepth = o3d.geometry.Image(depth_u16)
    orgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(orgb, odepth, depth_trunc=10.0, convert_rgb_to_intensity=False)
    roty90 = o3d.geometry.get_rotation_matrix_from_axis_angle([0, math.pi / 2, 0])
    rotxn90 = o3d.geometry.get_rotation_matrix_from_axis_angle([-math.pi / 2, 0, 0])
    rotz = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, base_state[2]])
    extrinsic = cam_transform
    rot_cam = extrinsic[:3, :3]
    trans_cam = extrinsic[:3, 3]
    final_rotation = rotz @ rot_cam @ rotxn90 @ roty90
    final_translation = [trans_cam[0] + base_state[0], trans_cam[1] + base_state[1], trans_cam[2] + 0]
    final_transform = extrinsic.copy()
    final_transform[:3, :3] = final_rotation
    final_transform[:3, 3] = final_translation
    extrinsic = np.linalg.inv(final_transform)
    opcd = o3d.geometry.PointCloud.create_from_rgbd_image(orgbd, intrinsic, extrinsic)
    
    rot = cam_transform[:3, :3]
    trans = cam_transform[:3, 3]
    rgb_depth = None
    # rgb_depth = HelloRobotMover.compute_pcd(rgb, depth, rot, trans, base_state, uv_one_in_cam)

    return rgb_depth, opcd, base_state, cam_transform, extrinsic


opcd = o3d.geometry.PointCloud()

for i in range(0, 1, 1):
# for i in range(0, 51, 10):
# for i in range(1900, 2100, 10):
    rgb_depth, cpcd, base_state, cam_transform, extrinsic = load_data(i)
    print(i)
    time.sleep(1)

    opcd += cpcd
    # opcd = opcd.voxel_down_sample(0.05)
    o3dviz.put('pointcloud', opcd)

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    eye = - R.T @ t
    up = -extrinsic[1, :3]     # Y camera axis in world frame
    front = -extrinsic[2, :3]  # Z camera axis in world frame
    center = eye - front       # any point on the ray through the camera center
    o3dviz.set_camera(look_at=center, position=eye, y_axis=up)
    
    o3dviz.add_robot(base_state, canonical=False, base=False)

    x, y, yaw = base_state.tolist()


o3dviz.join()
