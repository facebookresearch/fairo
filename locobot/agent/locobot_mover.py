"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import Pyro4
import logging
import numpy as np
import cv2
import os
import math
import copy
from perception import RGBDepth
from objects import Marker, Pos
from locobot_mover_utils import (
    get_camera_angles,
    angle_diff,
    MAX_PAN_RAD,
    CAMERA_HEIGHT,
    ARM_HEIGHT,
    transform_pose,
    base_canonical_coords_to_pyrobot_coords,
    xyz_pyrobot_to_canonical_coords,
)

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")


class LoCoBotMover:
    """Implements methods that call the physical interfaces of the Locobot.

    Arguments:
        ip (string): IP of the Locobot.
        backend (string): backend where the Locobot lives, either "habitat" or "locobot"
    """

    def __init__(self, ip=None, backend="locobot"):
        self.bot = Pyro4.Proxy("PYRONAME:remotelocobot@" + ip)
        self.close_loop = False if backend == "habitat" else True
        self.curr_look_dir = np.array([0, 0, 1])  # initial look dir is along the z-axis

        intrinsic_mat = self.bot.get_intrinsics()
        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        img_resolution = self.bot.get_img_resolution()
        img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)
        self.backend = backend

    # TODO/FIXME!  instead of just True/False, return diagnostic messages
    # so e.g. if a grip attempt fails, the task is finished, but the status is a failure
    def bot_step(self):
        try:
            f = self.bot.command_finished()
        except:
            # do better here?
            f = True
        return f

    def get_pan(self):
        """get yaw in radians."""
        return self.bot.get_pan()

    def get_tilt(self):
        """get pitch in radians."""
        return self.bot.get_tilt()

    def reset_camera(self):
        """reset the camera to 0 pan and tilt."""
        return self.bot.reset()

    def move_relative(self, xyt_positions):
        """Command to execute a relative move.

        Args:
            xyt_positions: a list of relative (x,y,yaw) positions for the bot to execute.
            x,y,yaw are in the pyrobot's coordinates.
        """
        for xyt in xyt_positions:
            self.bot.go_to_relative(xyt, close_loop=self.close_loop)
            while not self.bot.command_finished():
                print(self.bot.get_base_state("odom"))

    def move_absolute(self, xyt_positions):
        """Command to execute a move to an absolute position.

        It receives positions in canonical world coordinates and converts them to pyrobot's coordinates
        before calling the bot APIs.

        Args:
            xyt_positions: a list of (x_c,y_c,yaw) positions for the bot to move to.
            (x_c,y_c,yaw) are in the canonical world coordinates.
        """
        for xyt in xyt_positions:
            logging.info("Move absolute {}".format(xyt))
            self.bot.go_to_absolute(
                base_canonical_coords_to_pyrobot_coords(xyt), close_loop=self.close_loop
            )
            start_base_state = self.get_base_pos()
            while not self.bot.command_finished():
                print(self.get_base_pos())

            end_base_state = self.get_base_pos()
            logging.info(
                "start {}, end {}, diff {}".format(
                    start_base_state,
                    end_base_state,
                    [b - a for a, b in zip(start_base_state, end_base_state)],
                )
            )
        return "finished"

    def look_at(self, obj_pos, yaw_deg, pitch_deg):
        """Executes "look at" by setting the pan, tilt of the camera or turning the base if required.

        Uses both the base state and object coordinates in canonical world coordinates to calculate
        expected yaw and pitch.

        Args:
            obj_pos (list): object coordinates as saved in memory.
            yaw_deg (float): yaw in degrees
            pitch_deg (float): pitch in degrees

        Returns:
            string "finished"
        """
        pan_rad, tilt_rad = 0.0, 0.0
        old_pan = self.get_pan()
        old_tilt = self.get_tilt()
        pos = self.get_base_pos_in_canonical_coords()
        logging.debug(f"Current Locobot state (x, z, yaw): {pos}")
        if yaw_deg:
            pan_rad = old_pan - float(yaw_deg) * np.pi / 180
        if pitch_deg:
            tilt_rad = old_tilt - float(pitch_deg) * np.pi / 180
        if obj_pos is not None:
            logging.info(f"looking at x,y,z: {obj_pos}")
            pan_rad, tilt_rad = get_camera_angles([pos[0], CAMERA_HEIGHT, pos[1]], obj_pos)
            logging.debug(f"Returned new pan and tilt angles (radians): ({pan_rad}, {tilt_rad})")

        # FIXME!!! more async; move head and body at the same time
        head_res = angle_diff(pos[2], pan_rad)
        if np.abs(head_res) > MAX_PAN_RAD:
            dyaw = np.sign(head_res) * (np.abs(head_res) - MAX_PAN_RAD)
            self.turn(dyaw)
            pan_rad = np.sign(head_res) * MAX_PAN_RAD
        else:
            pan_rad = head_res
        logging.info(f"Camera new pan and tilt angles (radians): ({pan_rad}, {tilt_rad})")
        self.bot.set_pan_tilt(pan_rad, np.clip(tilt_rad, tilt_rad, 0.9))
        logging.debug(f"locobot pan and tilt now: ({self.bot.get_camera_state()})")

        return "finished"

    def point_at(self, target_pos):
        pos = self.get_base_pos_in_canonical_coords()
        yaw_rad, pitch_rad = get_camera_angles([pos[0], ARM_HEIGHT, pos[1]], target_pos)
        states = [
            [yaw_rad, 0.0, pitch_rad, 0.0, 0.0],
            [yaw_rad, 0.0, pitch_rad, -0.2, 0.0],
            [0.0, -math.pi / 4.0, math.pi / 2.0, 0.0, 0.0],  # reset joint position
        ]
        for state in states:
            self.bot.set_joint_positions(state, plan=False)
            while not self.bot.command_finished():
                pass
        return "finished"

    def get_base_pos_in_canonical_coords(self):
        """get the current Locobot position in the canonical coordinate system
        instead of the Locobot's global coordinates as stated in the Locobot
        documentation: https://www.pyrobot.org/docs/navigation.

        the standard coordinate systems:
          Camera looks at (0, 0, 1),
          its right direction is (1, 0, 0) and
          its up-direction is (0, 1, 0)

         return:
         (x, z, yaw) of the Locobot base in standard coordinates
        """

        x_global, y_global, yaw = self.bot.get_base_state("odom")
        x_standard = -y_global
        z_standard = x_global
        return np.array([x_standard, z_standard, yaw])

    def get_base_pos(self):
        """Return Locobot (x, y, yaw) in the robot base coordinates as
        illustrated in the docs:

        https://www.pyrobot.org/docs/navigation
        """
        return self.bot.get_base_state("odom")

    def get_rgb_depth(self):
        """Fetches rgb, depth and pointcloud in pyrobot world coordinates.

        Returns:
            an RGBDepth object
        """
        rgb, depth, rot, trans = self.bot.get_pcd_data()
        depth = depth.astype(np.float32)
        d = copy.deepcopy(depth)
        depth /= 1000.0
        depth = depth.reshape(-1)
        pts_in_cam = np.multiply(self.uv_one_in_cam, depth)
        pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
        pts = pts_in_cam[:3, :].T
        pts = np.dot(pts, rot.T)
        pts = pts + trans.reshape(-1)
        if self.backend == "habitat":
            ros_to_habitat_frame = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
            pts = ros_to_habitat_frame.T @ pts.T
            pts = pts.T
        pts = transform_pose(pts, self.bot.get_base_state("odom"))
        logging.info("Fetched all camera sensor input.")
        return RGBDepth(rgb, d, pts)

    def dance(self):
        self.bot.dance()

    def turn(self, yaw):
        """turns the bot by the yaw specified.

        Args:
            yaw: the yaw to execute in degree.
        """
        turn_rad = yaw * math.pi / 180
        self.bot.go_to_relative([0, 0, turn_rad], close_loop=self.close_loop)

    def grab_nearby_object(self, bounding_box=[(240, 480), (100, 540)]):
        """
        :param bounding_box: region in image to search for grasp
        """
        return self.bot.grasp(bounding_box)

    def is_object_in_gripper(self):
        return self.bot.get_gripper_state() == 2

    def explore(self):
        return self.bot.explore()

    def drop(self):
        return self.bot.open_gripper()
