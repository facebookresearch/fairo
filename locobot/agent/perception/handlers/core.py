"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
import sys
import os
import open3d as o3d

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
from abc import abstractmethod
from PIL import Image
from locobot.agent.perception import get_color_tag
from locobot.agent.locobot_mover_utils import xyz_pyrobot_to_canonical_coords


class AbstractHandler:
    """Interface for implementing perception handlers.

    Each handler must implement the handle and an optional _debug_draw method.
    """

    @abstractmethod
    def handle(self, *input):
        """Implement this method to hold the core execution logic."""
        pass

    def __call__(self, *input):
        return self.handle(*input)

    def _debug_draw(self, *input):
        """Implement this method to hold visualization details useful in
        debugging."""
        raise NotImplementedError


class WorldObject:
    def __init__(self, label, center, rgb_depth, mask=None, xyz=None):
        self.label = label
        self.center = center
        self.rgb_depth = rgb_depth
        self.mask = mask
        self.xyz = xyz if xyz else rgb_depth.get_coords_for_point(self.center)
        self.eid = None
        self.feature_repr = None
        self.bounds = rgb_depth.get_bounds_for_mask(self.mask) 

    def get_xyz(self):
        """returns xyz in canonical world coordinates."""
        return {"x": self.xyz[0], "y": self.xyz[1], "z": self.xyz[2]}
    
    def get_bounds(self):
        """returns bounding box as dict."""
        return (self.bounds[0], self.bounds[1], self.bounds[2],
            self.bounds[3], self.bounds[4], self.bounds[5])

    def get_masked_img(self):
        raise NotImplementedError

    def save_to_memory(self):
        raise NotImplementedError

    def to_struct(self):
        raise NotImplementedError


class RGBDepth:
    """Class for the current RGB, depth and point cloud fetched from the robot.

    Args:
        rgb (np.array): RGB image fetched from the robot
        depth (np.array): depth map fetched from the robot
        pts (list[(x,y,z)]): list of x,y,z coordinates of the pointcloud corresponding 
        to the rgb and depth maps.
    """
    rgb: np.array
    depth: np.array
    ptcloud: list

    def __init__(self, rgb, depth, pts):
        self.rgb = rgb
        self.depth = depth
        self.ptcloud = pts

    def get_pillow_image(self):
        return Image.fromarray(self.rgb, "RGB")

    def get_bounds_for_mask(self, mask):
        """for all points in the mask, returns the bounds as an axis-aligned bounding box.
        """
        indices = zip(*np.where(mask == True))
        points = [self.get_coords_for_point((x,y)) for x,y in indices]
        points = o3d.utility.Vector3dVector(points)
        obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
        return np.concatenate([obb.get_min_bound(), obb.get_max_bound()])

    def get_coords_for_point(self, point):
        """fetches xyz from the point cloud in pyrobot coordinates and converts it to
        canonical world coordinates.
        """
        xyz_p = self.ptcloud[point[1] * self.rgb.shape[0] + point[0]]
        return xyz_pyrobot_to_canonical_coords(xyz_p)

    def to_struct(self, size=None, quality=10):
        import base64

        rgb = self.rgb
        depth = self.depth

        if size is not None:
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_LINEAR)

        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth = 255 - depth

        # webp seems to be better than png and jpg as a codec, in both compression and quality
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        fmt = ".webp"

        _, rgb_data = cv2.imencode(fmt, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encode_param)
        _, depth_data = cv2.imencode(fmt, depth, encode_param)
        return {
            "rgb": base64.b64encode(rgb_data).decode("utf-8"),
            "depth": base64.b64encode(depth_data).decode("utf-8"),
        }
