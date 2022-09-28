# Code from https://github.com/facebookresearch/fairo/blob/main/droidlet/lowlevel/hello_robot/hello_robot_mover.py

import numpy as np


def compute_uvone(intrinsic_mat, height, width):
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    img_pixs = np.mgrid[0:height:1, 0:width:1]
    img_pixs = img_pixs.reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)
    return uv_one_in_cam

def get_pcd_in_cam(depth, intrinsic_mat):
    uv_one_in_cam = compute_uvone(intrinsic_mat, height=640, width=480)
    depth = depth.reshape(-1)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    pts = pts_in_cam.T
    return pts
