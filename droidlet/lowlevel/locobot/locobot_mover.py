"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys
import math
import copy
import time
import random
import logging
from collections.abc import Iterable
from prettytable import PrettyTable
import Pyro4
import numpy as np

from droidlet.shared_data_structs import ErrorWithResponse
from agents.argument_parser import ArgumentParser
from droidlet.shared_data_structs import RGBDepth
from droidlet.dashboard.o3dviz import deserialize as o3d_unpickle
from droidlet.lowlevel.pyro_utils import safe_call


from ..robot_mover_utils import (
    get_camera_angles,
    angle_diff,
    MAX_PAN_RAD,
    CAMERA_HEIGHT,
    ARM_HEIGHT,
    transform_pose,
)

from droidlet.lowlevel.robot_coordinate_utils import (
    xyz_pyrobot_to_canonical_coords,
    base_canonical_coords_to_pyrobot_coords,
)

random.seed(0)
np.random.seed(0)
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


class LoCoBotMover:
    """
    Implements methods that call the physical interfaces of the Locobot.

    Arguments:
        ip (string): IP of the Locobot.
        backend (string): backend where the Locobot lives, either "habitat" or "locobot"
    """

    def __init__(self, ip=None, backend="habitat"):
        self.bot = Pyro4.Proxy("PYRONAME:remotelocobot@" + ip)
        self.slam = Pyro4.Proxy("PYRONAME:slam@" + ip)
        self.nav = Pyro4.Proxy("PYRONAME:navigation@" + ip)
        # spin once synchronously
        self.nav.is_busy()
        # put in async mode
        self.nav._pyroAsync()
        self.nav_result = self.nav.is_busy()
        self.curr_look_dir = np.array([0, 0, 1])  # initial look dir is along the z-axis

        intrinsic_mat = safe_call(self.bot.get_intrinsics)
        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        img_resolution = safe_call(self.bot.get_img_resolution)
        img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)
        self.backend = backend

    def is_obstacle_in_front(self, return_viz=False):
        ret = safe_call(self.bot.is_obstacle_in_front, return_viz)
        if return_viz:
            obstacle, cpcd, crop, bbox, rest = ret

            cpcd = o3d_unpickle(cpcd)
            crop = o3d_unpickle(crop)
            bbox = o3d_unpickle(bbox)
            rest = o3d_unpickle(rest)
            return obstacle, cpcd, crop, bbox, rest
        else:
            obstacle = ret
            return obstacle

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

    def move_relative(self, xyt_positions, blocking=True):
        """
        Command to execute a relative move.

        Args:
            xyt_positions: a list of relative (x,y,yaw) positions for the bot to execute.
            x,y,yaw are in the pyrobot's coordinates.
            blocking (boolean): If True, waits for navigation to complete.
        """
        if not isinstance(next(iter(xyt_positions)), Iterable):
            # single xyt position given
            xyt_positions = [xyt_positions]
        for xyt in xyt_positions:
            self.nav_result.wait()  # wait for the previous navigation command to finish
            self.nav_result = safe_call(self.nav.go_to_relative, xyt)
            if blocking:
                self.nav_result.wait()

    def move_absolute(self, xyt_positions, use_map=False, blocking=True):
        """
        Command to execute a move to an absolute position.
        It receives positions in canonical world coordinates and converts them to pyrobot's coordinates
        before calling the bot APIs.

        Args:
            xyt_positions: a list of (x_c,y_c,yaw) positions for the bot to move to.
            (x_c,y_c,yaw) are in the canonical world coordinates.
            blocking (boolean): If True, waits for navigation to complete.
        """
        if not isinstance(next(iter(xyt_positions)), Iterable):
            # single xyt position given
            xyt_positions = [xyt_positions]
        for xyt in xyt_positions:
            logging.info("Move absolute in canonical coordinates {}".format(xyt))
            self.nav_result.wait()  # wait for the previous navigation command to finish
            robot_coords = base_canonical_coords_to_pyrobot_coords(xyt)
            self.nav_result = safe_call(self.nav.go_to_absolute, robot_coords)
            if blocking:
                self.nav_result.wait()
            start_base_state = self.get_base_pos_in_canonical_coords()
            while not self.bot.command_finished():
                print(self.get_base_pos_in_canonical_coords())

            end_base_state = self.get_base_pos_in_canonical_coords()
            logging.info(
                "start {}, end {}, diff {}".format(
                    start_base_state,
                    end_base_state,
                    [b - a for a, b in zip(start_base_state, end_base_state)],
                )
            )
        return "finished"

    def move_to_object(
        self, object_goal: str, episode_id: str, exploration_method: str, blocking=True
    ):
        """Command to execute a move to an object category.

        Args:
            object_goal: supported COCO object category
            exploration_method: learned or frontier
        """
        if self.nav_result.ready:
            self.nav_result.wait()
            self.nav_result = self.nav.go_to_object(object_goal, episode_id, exploration_method)
            if blocking:
                self.nav_result.wait()
        else:
            print("navigator executing another call right now")
        return self.nav_result
    
    def collect_data(
        self, episode_id: str, exploration_method: str, blocking=True
    ):
        """Active learning - Command to explore the scene to gather training data.
        
        Args:
            exploration_method: learned, frontier or SEAL
        """
        if self.nav_result.ready:
            self.nav_result.wait()
            self.nav_result = self.nav.collect_data(episode_id, exploration_method)
            if blocking:
                self.nav_result.wait()
        else:
            print("navigator executing another call right now")
        return self.nav_result

    def look_at(self, obj_pos, yaw_deg, pitch_deg):
        """
        Executes "look at" by setting the pan, tilt of the camera or turning the base if required.
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
        logging.info(f"Current Locobot state (x, z, yaw): {pos}")
        if yaw_deg:
            pan_rad = old_pan - float(yaw_deg) * np.pi / 180
        if pitch_deg:
            tilt_rad = old_tilt - float(pitch_deg) * np.pi / 180
        if obj_pos is not None:
            logging.info(f"looking at x,y,z: {obj_pos}")
            pan_rad, tilt_rad = get_camera_angles([pos[0], CAMERA_HEIGHT, pos[1]], obj_pos)
            logging.info(f"Returned new pan and tilt angles (radians): ({pan_rad}, {tilt_rad})")

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
        """
        Executes pointing the arm at the specified target pos.

        Args:
            target_pos ([x,y,z]): canonical coordinates to point to.

        Returns:
            string "finished"
        """
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
        """
        get the current Locobot position in the canonical coordinate system
        instead of the Locobot's global coordinates as stated in the Locobot
        documentation: https://www.pyrobot.org/docs/navigation.
        The standard coordinate systems:
        Camera looks at (0, 0, 1),
        its right direction is (1, 0, 0) and
        its up-direction is (0, 1, 0)

         return:
         (x, z, yaw) of the Locobot base in standard coordinates
        """

        x_global, y_global, yaw = safe_call(self.bot.get_base_state)
        x_standard = -y_global
        z_standard = x_global
        return np.array([x_standard, z_standard, yaw])

    def get_base_pos(self):
        """
        Return Locobot (x, y, yaw) in the robot base coordinates as
        illustrated in the docs:
        https://www.pyrobot.org/docs/navigation
        """
        return self.bot.get_base_state()

    def get_rgb_depth(self):
        """
        Fetches rgb, depth and pointcloud in pyrobot world coordinates.

        Returns:
            an RGBDepth object
        """
        rgb, depth, rot, trans, base_state = self.bot.get_pcd_data()
        depth = depth.astype(np.float32)
        d = copy.deepcopy(depth)
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
        pts = transform_pose(pts, base_state)

        return RGBDepth(rgb, d, pts)

    def get_rgb_depth_segm(self):
        if self.backend != "habitat":
            return None
        return self.bot.get_rgb_depth_segm()

    def get_current_pcd(self, in_cam=False, in_global=False):
        """Gets the current point cloud"""
        return self.bot.get_current_pcd(in_cam=in_cam, in_global=in_global)

    def is_busy(self):
        return self.nav.is_busy().value and self.bot.is_busy()

    def stop(self):
        """immediately stop the robot."""
        self.nav.stop()
        return self.bot.stop()

    def stop(self):
        """immediately stop the robot."""
        self.nav.stop()
        return self.bot.stop()

    def dance(self):
        self.bot.dance()

    def turn(self, yaw):
        """
        turns the bot by the yaw specified.

        Args:
            yaw: the yaw to execute in degree.
        """
        turn_rad = yaw * math.pi / 180
        self.bot.go_to_relative([0, 0, turn_rad])

    def grab_nearby_object(self, bounding_box=[(240, 480), (100, 540)]):
        """

        :param bounding_box: region in image to search for grasp
        """
        return self.bot.grasp(bounding_box)

    def explore(self, goal):
        if self.nav_result.ready:
            self.nav_result = safe_call(self.nav.explore, goal)
        else:
            print("navigator executing another call right now")
        return self.nav_result

    def is_done_exploring(self):
        return self.nav.is_done_exploring().value

    def get_obstacles_in_canonical_coords(self):
        """
        get the positions of obtacles position in the canonical coordinate system
        instead of the Locobot's global coordinates as stated in the Locobot
        documentation: https://www.pyrobot.org/docs/navigation or
        https://github.com/facebookresearch/pyrobot/blob/master/docs/website/docs/ex_navigation.md
        the standard coordinate systems:
        Camera looks at (0, 0, 1),
        its right direction is (1, 0, 0) and
        its up-direction is (0, 1, 0)

        return:
         list[(x, z)] of the obstacle location in standard coordinates
        """
        cordinates_in_robot_frame = self.slam.get_map()
        cordinates_in_standard_frame = [
            xyz_pyrobot_to_canonical_coords(list(c) + [0.0]) for c in cordinates_in_robot_frame
        ]
        cordinates_in_standard_frame = [(c[0], c[2]) for c in cordinates_in_standard_frame]
        return cordinates_in_standard_frame


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Locobot", base_path)
    opts = parser.parse()
    mover = LoCoBotMover(ip=opts.ip, backend=opts.backend)
    if opts.check_controller:
        mover.check()
