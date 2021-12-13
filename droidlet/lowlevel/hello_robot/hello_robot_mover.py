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
#from agents.argument_parser import ArgumentParser
from droidlet.shared_data_structs import RGBDepth

from droidlet.lowlevel.robot_coordinate_utils import base_canonical_coords_to_pyrobot_coords

from droidlet.lowlevel.robot_mover import MoverInterface
from droidlet.lowlevel.robot_mover_utils import (
    get_camera_angles,
    angle_diff,
    transform_pose,
)

from droidlet.lowlevel.hello_robot.rotation import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
)

from tenacity import retry, stop_after_attempt, wait_fixed

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION=2

MAX_PAN_RAD = math.pi/4

# TODO/FIXME: state machines.  state machines everywhere

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
        self.camera_transform = self.bot.get_camera_transform().value
        self.camera_height = self.camera_transform[2, 3]
        self.cam = Pyro4.Proxy("PYRONAME:hello_realsense@" + ip)

        self.data_logger = Pyro4.Proxy("PYRONAME:hello_data_logger@" + ip)
        self.data_logger._pyroAsync()
        _ = safe_call(self.data_logger.ready)

        intrinsic_mat = safe_call(self.cam.get_intrinsics)
        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        img_resolution = safe_call(self.cam.get_img_resolution, rotate=False)
        img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)

    def log_data_start(self, seconds):
        self.data_logger.save_batch(seconds)

    def log_data_stop(self):
        self.data_logger.stop()

    def bot_step(self):
        f = not self.bot.is_moving().value
        return f

    def relative_pan_tilt(self, dpan, dtilt, turn_base=True):
        """ 
        move the head so its new tilt is current_tilt + dtilt 
        and pan is current_pan + dpan 
        
        Args: 
            dpan (float): angle in radians to turn head left-right.  
                positive is right
            dtilt (float): angle in radians to turn head up-down.  
                positive is up
        """
        # FIXME handle out-of-range values properly
        dtilt = dtilt or 0
        dpan = dpan or 0
        
        new_tilt = self.get_tilt() + dtilt
        
        # FIXME: make a safe_base_turn method
        if np.abs(dpan) > MAX_PAN_RAD and turn_base:
            dyaw = np.sign(dpan) * (np.abs(dpan) - MAX_PAN_RAD)
            self.turn(dyaw * 180 / math.pi)
            pan_rad = np.sign(dpan) * MAX_PAN_RAD
        else:
            pan_rad = dpan
        new_pan = self.get_pan() + pan_rad
        self.bot.set_pan_tilt(new_pan, new_tilt)
        return "finished"
        

    def set_look(self, pan_rad, tilt_rad, turn_base=True, world=False):
        """
        Sets the agent to look at a specified absolute pan and tilt.
        These are  "absolute" w.r.t. robot current base, if world==False
        and absolute w.r.t. world coords if world=True

        Args: 
            pan_rad (float): angle in radians to turn head left-right.  
                positive is right
            tilt_rad (float): angle in radians to to turn head up-down.
                positive is down. 
        """
        tilt_rad = tilt_rad or self.get_tilt()
        # TODO handle out-of-range properly
        dtilt = angle_diff(self.get_tilt(), tilt_rad)
        if not world:
            pan_rad = pan_rad or self.get_pan()
            dpan = angle_diff(self.get_pan(), pan_rad)
        else:
            base_pan = self.get_base_pos_in_canonical_coords()[2]
            pan_rad = pan_rad or base_pan + self.get_pan()
            dpan = angle_diff(base_pan + self.get_pan(), pan_rad)
        return self.relative_pan_tilt(dpan, dtilt, turn_base=turn_base)

    
    def look_at(self, target, turn_base=True, face=False):
        """
        Executes "look at" by setting the pan, tilt of the camera
        or turning the base if required.
        Uses both the base state and object coordinates in 
        canonical world coordinates to calculate expected yaw and pitch.
        if face == True will move body so head yaw is 0 

        Args:
            target (list): object coordinates as saved in memory.
            turn_base: if False, will try to look at point only by moving camera and not base
        """
        old_pan = self.get_pan()
        old_tilt = self.get_tilt()
        pos = self.get_base_pos_in_canonical_coords()
        cam_transform = self.bot.get_camera_transform().value
        cam_pos = cam_transform[0:3, 3]
        # convert cam_pos to canonical co-ordinates
        # FIXME !  do this properly in a util, current functions don't do it bc
        #          we are mixing base pos already converted and cam height
        cam_pos = [pos[0], cam_pos[2], pos[1]]

        logging.info(f"Current base state (x, z, yaw): {pos}, camera state (x, y, z): {cam_pos}")
        logging.info(f"looking at x,y,z: {target}")
        
        pan_rad, tilt_rad = get_camera_angles(cam_pos, target)
#        pan_res = angle_diff(pos[2], pan_rad)
        # For the Hello camera, negative tilt seems to be up, and positive tilt is down
        # For the locobot camera, it is the opposite
        # TODO: debug this further, and make things across robots consistent
        logging.info(f"Returned new pan and tilt angles (radians): ({pan_rad}, {tilt_rad})")
        if face:
            # TODO less blocking, make me into state machine
            dpan = angle_diff(self.get_base_pos_in_canonical_coords()[2], pan_rad)
            self.turn(dpan * 180 / math.pi)
            return self.set_look(0, tilt_rad)
        else:
            self.set_look(pan_rad, tilt_rad, turn_base=True, world=True)


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
            xzt_positions: a list of relative (x,y,yaw) positions for the bot to execute.
            x,y,yaw are in the pyrobot's coordinates.
        """
        if not isinstance(next(iter(xyt_positions)), Iterable):
            # single xyt position given
            xzt_positions = [xzt_positions]
        for xyt in xyt_positions:
            self.is_moving.wait()
            self.is_moving = safe_call(self.bot.go_to_relative, xyt)
            if blocking:
                self.is_moving.wait()

    def move_absolute(self, xzt_positions, blocking=True):
        """Command to execute a move to an absolute position.

        It receives positions in canonical world coordinates and converts them to pyrobot's coordinates
        before calling the bot APIs.

        Args:
            xzt_positions: a list of (x_c,z_c,yaw) positions for the bot to move to.
            (x_c,z_c,yaw) are in the canonical world coordinates.
        """
        if not isinstance(next(iter(xzt_positions)), Iterable):
            # single xzt position given
            xzt_positions = [xzt_positions]
        for xzt in xzt_positions:
            logging.info("Move absolute in canonical coordinates {}".format(xzt))
            self.is_moving.wait()
            robot_coords = base_canonical_coords_to_pyrobot_coords(xzt)
            self.is_moving = self.bot.go_to_absolute(robot_coords)
            if blocking:
                self.is_moving.wait()
        return "finished"

    def get_base_pos_in_canonical_coords(self):
        """get the current robot position in the canonical coordinate system
       
        the canonical coordinate systems:                           
        from the origin, at yaw=0, front is (x, y, z) = (0, 0, 1),
        its right direction is (x,y,z) = (1, 0, 0) 
        its up direction is (x,y,z) = (0, 1, 0)
        yaw is + counterclockwise

        return:
        (x, z, yaw) of the robot base in canonical coordinates
        """
        future = safe_call(self.bot.get_base_state)
        x_robot, y_robot, yaw = future.value
        z_canonical = x_robot
        x_canonical = -y_robot
        return np.array([x_canonical, z_canonical, yaw])

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

        # the realsense pointcloud seems to produce some spurious points
        # really far away. So, limit the depth to 8 metres
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
        rotyt = rotation_matrix_y(90)
        pts = np.dot(pts, rotyt.T)

        rotxt = rotation_matrix_x(-90)
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

    def explore(self):
        return self.bot.explore()

if __name__ == "__main__":
    import argparse
    #    parser = ArgumentParser("HelloRobot", base_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="")
    opts = parser.parse_args()
    base_path = os.path.dirname(__file__)
    mover = HelloRobotMover(ip=opts.ip)
