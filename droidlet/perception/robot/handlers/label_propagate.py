"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .core import AbstractHandler
import numpy as np
import os
import cv2
import json
import glob
from droidlet.lowlevel.robot_mover_utils import transform_pose
from numba import njit
from math import ceil, floor

def compute_uvone(height, width):
    intrinsic_mat = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
    rot = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    trans = np.array([0, 0, 0.6])
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    img_resolution = (height, width)
    img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
    img_pixs = img_pixs.reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)
    return uv_one_in_cam, intrinsic_mat, rot, trans

def convert_depth_to_pcd(depth, pose, uv_one_in_cam, rot, trans):
    # point cloud in camera frame
    depth = (depth.astype(np.float32) / 1000.0).reshape(-1)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
    # point cloud in robot base frame
    pts_in_base = pts_in_cam[:3, :].T
    pts_in_base = np.dot(pts_in_base, rot.T)
    pts_in_base = pts_in_base + trans.reshape(-1)
    # point cloud in world frame (pyrobot)
    pts_in_world = transform_pose(pts_in_base, pose)
    return pts_in_world

@njit
def get_annot(height, width, pts_in_cur_img, src_label):
    annot_img = np.zeros((height, width))
    for x in range(len(pts_in_cur_img)):
        r = int(x/width)
        c = int(x - r*width)
        x, y, _ = pts_in_cur_img[x]
        
        # Take ceil and floor combination to fix quantization errors
        if floor(x) >= 0 and ceil(x) < height and floor(y) >=0 and ceil(y) < width:
            annot_img[ceil(y)][ceil(x)] = src_label[r][c]
            annot_img[floor(y)][floor(x)] = src_label[r][c]
            annot_img[ceil(y)][floor(x)] = src_label[r][c]
            annot_img[floor(y)][ceil(x)] = src_label[r][c]
    
    return annot_img

class LabelPropagate(AbstractHandler):
    def __call__(self,    
        src_img,
        src_depth,
        src_label,
        src_pose,
        base_pose,
        cur_depth,
    ):
        """Gets point cloud -> Transpose the point cloud based on robot location -> Project the point cloud back the images
        Args:
            src_img (np.ndarray): source image to propagte from
            src_depth (np.ndarray): source depth to propagte from
            src_label (np.ndarray): source semantic map to propagte from
            src_pose (np.ndarray): (x,y,theta) of the source image
            base_pose (np.ndarray): (x,y,theta) of current image
            cur_depth (np.ndarray): current depth
        """

        height, width, _ = src_img.shape
        uv_one_in_cam, intrinsic_mat, rot, trans = compute_uvone(height, width)
        
        pts_in_world = convert_depth_to_pcd(src_depth, src_pose, uv_one_in_cam, rot, trans)
        
        # TODO: can use cur_pts_in_world for filtering. Not needed for baseline.
        # cur_pts_in_world = convert_depth_to_pcd(cur_depth, base_pose, uv_one_in_cam, rot, trans)
        
        # convert pts_in_world to current base
        pts_in_cur_base = transform_pose(pts_in_world, (-base_pose[0], -base_pose[1], 0))
        pts_in_cur_base = transform_pose(pts_in_cur_base, (0.0, 0.0, -base_pose[2]))

        # conver point from current base to current camera frame
        pts_in_cur_cam = pts_in_cur_base - trans.reshape(-1)
        pts_in_cur_cam = np.dot(pts_in_cur_cam, rot)

        # conver pts in current camera frame into 2D pix values
        pts_in_cur_img = np.matmul(intrinsic_mat, pts_in_cur_cam.T).T
        pts_in_cur_img /= pts_in_cur_img[:, 2].reshape([-1, 1])
        
        return get_annot(height, width, pts_in_cur_img, src_label)
