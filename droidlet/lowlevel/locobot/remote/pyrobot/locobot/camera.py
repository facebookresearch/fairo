# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import threading
import yaml
from copy import deepcopy

# import message_filters
import numpy as np
from ..utils import util as prutil
# import rospy

from ..core import Camera
# from sensor_msgs.msg import CameraInfo
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64
# from tf import TransformListener

from ..utils.util import try_cv2_import

cv2 = try_cv2_import()


def constrain_within_range(value, MIN, MAX):
    return min(max(value, MIN), MAX)


def is_within_range(value, MIN, MAX):
    return (value <= MAX) and (value >= MIN)


class DepthImgProcessor:
    """
    This class transforms the depth image and rgb image to point cloud
    """

    def __init__(
        self,
        subsample_pixs=1,
        depth_threshold=(0, 1.5),
        cfg_filename="realsense_d435.yaml",
    ):
        """
        The constructor for :class:`DepthImgProcessor` class.

        :param subsample_pixs: sample rows and columns for the images
        :param depth_threshold: minimum and maximum of valid depth values
        :param cfg_filename: configuration file name for ORB-SLAM2

        :type subsample_pixs: int
        :type depth_threshold: tuple
        :type cfg_filename: string
        """
        assert (type(depth_threshold) is tuple and 0 < len(depth_threshold) < 3) or (
            depth_threshold is None
        )
        self.subsample_pixs = subsample_pixs
        self.depth_threshold = depth_threshold
        self.cfg_data = self.read_cfg(cfg_filename)
        self.intrinsic_mat = self.get_intrinsic()
        self.intrinsic_mat_inv = np.linalg.inv(self.intrinsic_mat)

        img_pixs = np.mgrid[
            0 : self.cfg_data["Camera.height"] : subsample_pixs,
            0 : self.cfg_data["Camera.width"] : subsample_pixs,
        ]
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        self.uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(self.intrinsic_mat_inv, self.uv_one)

    def get_pix_3dpt(self, depth_im, rs, cs):
        """
        :param depth_im: depth image (shape: :math:`[H, W]`)
        :param rs: rows of interest. It can be a list or 1D numpy array
                   which contains the row indices. The default value is None,
                   which means all rows.
        :param cs: columns of interest. It can be a list or 1D numpy array
                   which contains the column indices.
                   The default value is None,
                   which means all columns.
        :type depth_im: np.ndarray
        :type rs: list or np.ndarray
        :type cs: list or np.ndarray

        :return: 3D point coordinates of the pixels in
                 camera frame (shape: :math:`[4, N]`)
        :rtype np.ndarray
        """
        pts_in_cam = prutil.pix_to_3dpt(depth_im, rs, cs, self.intrinsic_mat, 1.0)
        return pts_in_cam

    def get_pcd_ic(self, depth_im, rgb_im=None):
        """
        Returns the point cloud (filtered by minimum
        and maximum depth threshold)
        in camera's coordinate frame

        :param depth_im: depth image (shape: :math:`[H, W]`)
        :param rgb_im: rgb image (shape: :math:`[H, W, 3]`)

        :type depth_im: np.ndarray
        :type rgb_im: np.ndarray

        :returns: tuple (pts_in_cam, rgb_im)

                  pts_in_cam: point coordinates in
                              camera frame (shape: :math:`[4, N]`)

                  rgb: rgb values for pts_in_cam (shape: :math:`[N, 3]`)
        :rtype tuple(np.ndarray, np.ndarray)
        """
        # pcd in camera from depth
        depth_im = depth_im[0 :: self.subsample_pixs, 0 :: self.subsample_pixs]
        rgb_im = rgb_im[0 :: self.subsample_pixs, 0 :: self.subsample_pixs]
        depth = depth_im.reshape(-1)
        rgb = None
        if rgb_im is not None:
            rgb = rgb_im.reshape(-1, 3)
        if self.depth_threshold is not None:
            valid = depth > self.depth_threshold[0]
            if len(self.depth_threshold) > 1:
                valid = np.logical_and(valid, depth < self.depth_threshold[1])
            uv_one_in_cam = self.uv_one_in_cam[:, valid]
            depth = depth[valid]
            rgb = rgb[valid]
        else:
            uv_one_in_cam = self.uv_one_in_cam
        pts_in_cam = np.multiply(uv_one_in_cam, depth)
        pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
        return pts_in_cam, rgb

    def get_pcd_iw(self, pts_in_cam, extrinsic_mat):
        """
        Returns the point cloud in the world coordinate frame

        :param pts_in_cam: point coordinates in
               camera frame (shape: :math:`[4, N]`)
        :param extrinsic_mat: extrinsic matrix for
               the camera (shape: :math:`[4, 4]`)

        :type pts_in_cam: np.ndarray
        :type extrinsic_mat: np.ndarray

        :return: point coordinates in
                 ORB-SLAM2's world frame (shape: :math:`[N, 3]`)
        :rtype: np.ndarray
        """
        # pcd in world
        pts_in_world = np.dot(extrinsic_mat, pts_in_cam)
        pts_in_world = pts_in_world[:3, :].T
        return pts_in_world

    def read_cfg(self, cfg_filename):
        """
        Reads the configuration file

        :param cfg_filename: configuration file name for ORB-SLAM2

        :type cfg_filename: string

        :return: configurations in the configuration file
        :rtype: dict
        """
        # ./robots/LoCoBot/locobot_navigation/orb_slam2_ros/cfg/realsense_habitat.yaml
        cfg_path = os.path.join(os.path.dirname(__file__), cfg_filename)
        # cfg_path = os.path.join(slam_pkg_path, "cfg", cfg_filename)
        with open(cfg_path, "r") as f:
            for i in range(1):
                f.readline()
            cfg_data = yaml.load(f, Loader=yaml.FullLoader)
        return cfg_data

    def get_intrinsic(self):
        """
        Returns the instrinsic matrix of the camera

        :return: the intrinsic matrix (shape: :math:`[3, 3]`)
        :rtype: np.ndarray
        """
        fx = self.cfg_data["Camera.fx"]
        fy = self.cfg_data["Camera.fy"]
        cx = self.cfg_data["Camera.cx"]
        cy = self.cfg_data["Camera.cy"]
        Itc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return Itc
