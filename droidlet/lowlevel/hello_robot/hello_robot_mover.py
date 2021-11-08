"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys
import math
import copy
import time
import logging
from collections.abc import Iterable
from prettytable import PrettyTable
import Pyro4
import numpy as np

import cv2
from droidlet.shared_data_structs import ErrorWithResponse
from agents.argument_parser import ArgumentParser
from droidlet.shared_data_structs import RGBDepth

from ..robot_mover import MoverInterface
from ..robot_mover_utils import (
    get_camera_angles,
    angle_diff,
    MAX_PAN_RAD,
    CAMERA_HEIGHT,
    ARM_HEIGHT,
    transform_pose,
    base_canonical_coords_to_pyrobot_coords,
    xyz_pyrobot_to_canonical_coords,
)
from tenacity import retry, stop_after_attempt, wait_fixed

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION=2
def roty(a):
    ar = float(a) * math.pi / 180.
    cos = math.cos
    sin = math.sin
    return np.array([[cos(ar), 0, sin(ar)],
                     [0, 1, 0],
                     [-sin(ar), 0, cos(ar)]])
def rotx(a):
    ar = float(a) * math.pi / 180.
    cos = math.cos
    sin = math.sin
    return np.array([
        [1, 0, 0],
        [0, cos(ar), -sin(ar)],
        [0, sin(ar), cos(ar)]
    ])
def rotz(a):
    ar = float(a) * math.pi / 180.
    cos = math.cos
    sin = math.sin
    return np.array([
        [cos(ar), -sin(ar), 0],
        [sin(ar), cos(ar), 0],
        [0, 0, 1],
    ])

def safe_call(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Pyro4.errors.ConnectionClosedError as e:
        msg = "{} - {}".format(f._RemoteMethod__name, e)
        raise ErrorWithResponse(msg)
    except Exception as e:
        print("Pyro traceback:")
        print("".join(Pyro4.util.getPyroTraceback()))
        raise e

class HelloRobotMover(MoverInterface):
    """Implements methods that call the physical interfaces of the Robot.

    Arguments:
        ip (string): IP of the Robot.
    """

    def __init__(self, ip=None):
        self.bot = Pyro4.Proxy("PYRONAME:hello_robot@" + ip)
        self.bot._pyroAsync()
        self.is_moving = self.bot.is_moving()
        self.cam = Pyro4.Proxy("PYRONAME:hello_realsense@" + ip)

        self.data_logger = Pyro4.Proxy("PYRONAME:hello_data_logger@" + ip)
        self.data_logger._pyroAsync()
        _ = safe_call(self.data_logger.ready)
        self.curr_look_dir = np.array([0, 0, 1])  # initial look dir is along the z-axis

        intrinsic_mat = safe_call(self.cam.get_intrinsics)
        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        img_resolution = safe_call(self.cam.get_img_resolution, rotate=False)
        img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)

    def log_data(self, seconds):
        self.data_logger.save_batch(seconds)

    def bot_step(self):
        pass

    def look_at(self):
        pass

    def stop(self):
        """immediately stop the robot."""
        return self.bot.stop()

    def unstop(self):
        """remove a runstop flag via software"""
        return self.bot.remove_runstop()

    def get_pan(self):
        """get yaw in radians."""
        return self.bot.get_pan().value

    def get_tilt(self):
        """get pitch in radians."""
        return self.bot.get_tilt().value

    def reset_camera(self):
        """reset the camera to 0 pan and tilt."""
        return self.bot.reset()

    def move_relative(self, xyt_positions, blocking=True):
        """Command to execute a relative move.

        Args:
            xyt_positions: a list of relative (x,y,yaw) positions for the bot to execute.
            x,y,yaw are in the pyrobot's coordinates.
        """
        if not isinstance(next(iter(xyt_positions)), Iterable):
            # single xyt position given
            xyt_positions = [xyt_positions]
        for xyt in xyt_positions:
            self.is_moving.wait()
            self.is_moving = safe_call(self.bot.go_to_relative, xyt)
            if blocking:
                self.is_moving.wait()

    def move_absolute(self, xyt_positions, blocking=True):
        """Command to execute a move to an absolute position.

        It receives positions in canonical world coordinates and converts them to pyrobot's coordinates
        before calling the bot APIs.

        Args:
            xyt_positions: a list of (x_c,y_c,yaw) positions for the bot to move to.
            (x_c,y_c,yaw) are in the canonical world coordinates.
        """
        if not isinstance(next(iter(xyt_positions)), Iterable):
            # single xyt position given
            xyt_positions = [xyt_positions]
        for xyt in xyt_positions:
            logging.info("Move absolute in canonical coordinates {}".format(xyt))
            global_coords = base_canonical_coords_to_pyrobot_coords(xyt)
            self.is_moving.wait()
            self.is_moving = self.bot.go_to_absolute(global_coords)
            if blocking:
                self.is_moving.wait()
        return "finished"

    def get_base_pos_in_canonical_coords(self):
        """get the current robot position in the canonical coordinate system
       
        the standard coordinate systems:
          Camera looks at (0, 0, 1),
          its right direction is (1, 0, 0) and
          its up-direction is (0, 1, 0)

         return:
         (x, z, yaw) of the robot base in standard coordinates
        """
        future = safe_call(self.bot.get_base_state)
        x_global, y_global, yaw = future.value
        x_standard = -y_global
        z_standard = x_global
        return np.array([x_standard, z_standard, yaw])

    def get_current_pcd(self, in_cam=False, in_global=False):
        """Gets the current point cloud"""
        return self.cam.get_current_pcd()
        
    def get_rgb_depth(self):
        """Fetches rgb, depth and pointcloud in pyrobot world coordinates.

        Returns:
            an RGBDepth object
        """
        rgb, depth, rot, trans = self.cam.get_pcd_data(rotate=False)

        rgb = np.asarray(rgb).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = depth.astype(np.float32)

        thres = 8000
        depth[depth > thres] = thres

        depth_copy = np.copy(depth)

        # get_pcd_data multiplies depth by 1000 to convert to mm, so reverse that
        depth /= 1000.0
        depth = depth.reshape(rgb.shape[0] * rgb.shape[1])

        # normalize by the camera's intrinsic matrix
        pts_in_cam = np.multiply(self.uv_one_in_cam, depth)
        pts = pts_in_cam.T

        # Now, the points are in camera frame.
        # In camera frame
        # z is positive into the camera
        # (larger the z, more into the camera)
        # x is positive to the right
        # (larger the x, more right of the origin)
        # y is positive to the bottom
        # (larger the y, more to the bottom of the origin)
        #                                 /
        #                                /
        #                               / z-axis
        #                              /
        #                             /_____________ x-axis (640)
        #                             |
        #                             |
        #                             | y-axis (480)
        #                             |
        #                             |


        # We now need to transform this to pyrobot frame, where
        # x is into the camera, y is positive to the left,
        # z is positive upwards
        # https://pyrobot.org/docs/navigation
        #                            |    /
        #                 z-axis     |   /
        #                            |  / x-axis
        #                            | /
        #  y-axis        ____________|/
        #
        # If you hold the first configuration in your right hand, and
        # visualize the transformations needed to get to the second
        # configuration, you'll see that
        # you have to rotate 90 degrees anti-clockwise around the y axis, and then
        # 90 degrees clockwise around the x axis.
        # This results in the final configuration
        rotyt = roty(90)
        pts = np.dot(pts, rotyt.T)

        rotxt = rotx(-90)
        pts = np.dot(pts, rotxt.T)

        # next, rotate and translate pts by
        # the robot pose and location
        pts = np.dot(pts, rot.T)
        pts = pts + trans.reshape(-1)
        pts = transform_pose(pts, self.bot.get_base_state().value)

        # now rewrite the ordering of pts so that the colors (rgb_rotated)
        # match the indices of pts
        pts = pts.reshape((rgb.shape[0], rgb.shape[1], 3))
        pts = np.rot90(pts, k=1, axes=(1, 0))
        pts = pts.reshape(rgb.shape[0] * rgb.shape[1], 3)

        depth_rotated = np.rot90(depth_copy, k=1, axes=(1,0))
        rgb_rotated = np.rot90(rgb, k=1, axes=(1,0))

        return RGBDepth(rgb_rotated, depth_rotated, pts)

    def turn(self, yaw):
        """turns the bot by the yaw specified.

        Args:
            yaw: the yaw to execute in degree.
        """
        turn_rad = yaw * math.pi / 180
        self.bot.rotate_by(turn_rad)

    def get_obstacles_in_canonical_coords(self):
        cordinates_in_robot_frame = self.bot.get_map()
        cordinates_in_standard_frame = [
            xyz_pyrobot_to_canonical_coords(list(c) + [0.0]) for c in cordinates_in_robot_frame
        ]
        cordinates_in_standard_frame = [(c[0], c[2]) for c in cordinates_in_standard_frame]
        return cordinates_in_standard_frame

    def explore(self):
        return self.bot.explore()

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("HelloRobot", base_path)
    opts = parser.parse()
    mover = BotMover(ip=opts.ip)
    if opts.check_controller:
        mover.check()
