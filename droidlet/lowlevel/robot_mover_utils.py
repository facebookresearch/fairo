"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import logging
from scipy.spatial.transform import Rotation
import math
import os
import shutil
import cv2
import json
from copy import deepcopy as copy
from pathlib import Path

from .rotation import yaw_pitch

MAX_PAN_RAD = np.pi / 4
CAMERA_HEIGHT = 0.6
ARM_HEIGHT = 0.5


def angle_diff(a, b):
    r = b - a
    r = r % (2 * np.pi)
    if r > np.pi:
        r = r - 2 * np.pi
    return r


def get_camera_angles(camera_pos, look_at_pnt):
    """get the new yaw/pan and pitch/tilt angle values and update the camera's
    new look direction."""
    logging.debug(f"look_at_point: {np.array(look_at_pnt)}")
    logging.debug(f"camera_position: {np.array(camera_pos)}")
    logging.debug(f"difference: {np.array(look_at_pnt) - np.array(camera_pos)}")
    look_dir = np.array(look_at_pnt) - np.array(camera_pos)
    logging.debug(f"Un-normalized look direction: {look_dir}")
    if np.linalg.norm(look_dir) < 0.01:
        return 0.0, 0.0
    look_dir = look_dir / np.linalg.norm(look_dir)
    return yaw_pitch(look_dir)


def get_arm_angle(locobot_pos, marker_pos):
    H = 0.2
    dir_xy_vect = np.array(marker_pos)[:2] - np.array(locobot_pos)[:2]
    angle = -np.arctan((marker_pos[2] - H) / np.linalg.norm(dir_xy_vect))
    return angle


def get_bot_angle(locobot_pos, marker_pos):
    dir_xy_vect = np.array(marker_pos)[:2] - np.array(locobot_pos)[:2]
    angle = np.arctan(dir_xy_vect[1] / dir_xy_vect[0])
    return angle


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
    current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape((-1, 3))
    XYZ[:, 0] = XYZ[:, 0] + current_pose[0]
    XYZ[:, 1] = XYZ[:, 1] + current_pose[1]
    return XYZ


def get_move_target_for_point(base_pos, target, eps=0.5):
    """
    For point, we first want to move close to the object and then point to it.

    Args:
        base_pos ([x,z,yaw]): robot base in canonical coords
        target ([x,y,z]): point target in canonical coords

    Returns:
        move_target ([x,z,yaw]): robot base move target in canonical coords
    """

    dx = target[0] - base_pos[0]
    signx = 1 if dx > 0 else -1

    dz = target[2] - base_pos[1]
    signz = 1 if dz > 0 else -1

    targetx = base_pos[0] + signx * (abs(dx) - eps)
    targetz = base_pos[1] + signz * (abs(dz) - eps)

    yaw, _ = get_camera_angles([targetx, CAMERA_HEIGHT, targetz], target)

    return [targetx, targetz, yaw]

def get_step_target_for_straightline_move(base_pos, target, step_size=0.1):
    """
    Heuristic to get step target of step_size for going to from base_pos to target
    in a straight line. 
    Args:
        base_pos ([x,z,yaw]): robot base in canonical coords
        target ([x,y,z]): point target in canonical coords
    
    Returns:
        move_target ([x,z,yaw]): robot base move target in canonical coords 
    """

    dx = target[0] - base_pos[0]
    dz = target[2] - base_pos[1]

    if dx == 0: # vertical line 
        theta = math.radians(90)
    else:
        theta = math.atan(abs(dz/dx))
    
    signx = 1 if dx >= 0 else -1
    signz = 1 if dz >= 0 else -1
    
    targetx = base_pos[0] + signx * step_size * math.cos(theta)
    targetz = base_pos[1] + signz * step_size * math.sin(theta)

    yaw, _ = get_camera_angles([targetx, CAMERA_HEIGHT, targetz], target)
    
    return [targetx, targetz, yaw]

"""
Co-ordinate transform utils. Read more at https://github.com/facebookresearch/fairo/blob/main/locobot/coordinates.MD
"""

pyrobot_to_canonical_frame = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def base_canonical_coords_to_pyrobot_coords(xyt):
    """converts the robot's base coords from canonical to pyrobot coords."""
    return [xyt[1], -xyt[0], xyt[2]]


def xyz_pyrobot_to_canonical_coords(xyz):
    """converts 3D coords from pyrobot to canonical coords."""
    return xyz @ pyrobot_to_canonical_frame


def xyz_canonical_coords_to_pyrobot_coords(xyz):
    """converts 3D coords from canonical to pyrobot coords."""
    return xyz @ np.linalg.inv(pyrobot_to_canonical_frame)


class TrajectoryDataSaver:
    def __init__(self, root):
        print(f'TrajectoryDataSaver saving to {root}')
        self.save_folder = root
        self.img_folder = os.path.join(self.save_folder, "rgb")
        self.img_folder_dbg = os.path.join(self.save_folder, "rgb_dbg")
        self.depth_folder = os.path.join(self.save_folder, "depth")
        self.seg_folder = os.path.join(self.save_folder, "seg")
        self.trav_folder = os.path.join(self.save_folder, "trav")

        if os.path.exists(self.save_folder):
            print(f'rmtree {self.save_folder}')
            shutil.rmtree(self.save_folder)

        print(f'trying to create {self.save_folder}')
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)

        for x in [self.img_folder, self.img_folder_dbg, self.depth_folder, self.seg_folder, self.trav_folder]:
            self.create(x)

        self.pose_dict = {}
        self.img_count = 0
        self.dbg_str = "None"

    def create(self, d):
        if not os.path.isdir(d):
            os.makedirs(d)

    def set_dbg_str(self, x):
        self.dbg_str = x
    
    def get_total_frames(self):
        return self.img_count

    def save(self, rgb, depth, seg, pos):
        print(f'saving {rgb.shape, depth.shape, seg.shape}')
        # store the images and depth
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(
            self.img_folder + "/{:05d}.jpg".format(self.img_count),
            rgb,
        )

        cv2.putText(rgb, self.dbg_str, (40,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

        # robot_dbg_str = 'robot_pose ' + str(np.round(self.get_robot_global_state(), 3))
        # cv2.putText(rgb, robot_dbg_str, (40,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

        cv2.imwrite(
            self.img_folder_dbg + "/{:05d}.jpg".format(self.img_count),
            rgb,
        )

        # store depth in mm
        depth *= 1e3
        depth[depth > np.power(2, 16) - 1] = np.power(2, 16) - 1
        depth = depth.astype(np.uint16)
        np.save(self.depth_folder + "/{:05d}.npy".format(self.img_count), depth)

        # store seg
        np.save(self.seg_folder + "/{:05d}.npy".format(self.img_count), seg)

        # store pos
        self.pose_dict[self.img_count] = copy(pos)
        self.img_count += 1
        
        # print(f"img_count {self.img_count}, #active {self.active_count}, self.goal_loc {self.goal_loc}, base_pos {pos}")
        with open(os.path.join(self.save_folder, "data.json"), "w") as fp:
            json.dump(self.pose_dict, fp)