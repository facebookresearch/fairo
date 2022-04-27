"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import logging
from scipy.spatial.transform import Rotation
import math
import os
import shutil
import glob
import sys

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import json
from copy import deepcopy as copy
from pathlib import Path
import matplotlib.pyplot as plt

from droidlet.shared_data_struct.rotation import yaw_pitch

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

    if dx == 0:  # vertical line
        theta = math.radians(90)
    else:
        theta = math.atan(abs(dz / dx))

    signx = 1 if dx >= 0 else -1
    signz = 1 if dz >= 0 else -1

    targetx = base_pos[0] + signx * step_size * math.cos(theta)
    targetz = base_pos[1] + signz * step_size * math.sin(theta)

    yaw, _ = get_camera_angles([targetx, CAMERA_HEIGHT, targetz], target)

    return [targetx, targetz, yaw]


def get_straightline_path_to(target, robot_pos):
    pts = []
    cur_pos = robot_pos
    while np.linalg.norm(target[:2] - cur_pos[:2]) > 0.5:
        t = get_step_target_for_move(cur_pos, [target[0], CAMERA_HEIGHT, target[1]], step_size=0.5)
        pts.append(t)
        cur_pos = t
    return np.asarray(pts)


def get_circle(r, n=10):
    return [
        [math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r]
        for x in range(0, n + 1)
    ]


def get_circular_path(target, robot_pos, radius, num_points):
    """
    get a circular path with num_points of radius from x
    xyz
    """
    pts = get_circle(radius, num_points)  # these are the x,z

    def get_xyyaw(p, target):
        targetx = p[0] + target[0]
        targetz = p[1] + target[2]
        yaw, _ = get_camera_angles([targetx, CAMERA_HEIGHT, targetz], target)
        return [targetx, targetz, yaw]

    pts = np.asarray([get_xyyaw(p, target) for p in pts])

    # find nearest pt to robot_pos as starting point
    def find_nearest_indx(pts, robot_pos):
        idx = np.asarray(
            [np.linalg.norm(np.asarray(p[:2]) - np.asarray(robot_pos[:2])) for p in pts]
        ).argmin()
        return idx

    idx = find_nearest_indx(pts, robot_pos)
    # rotate the pts to begin at idx
    pts = np.concatenate((pts[idx:, :], pts[:idx, :]), axis=0)

    # TODO: get step-wise move targets to nearest point? or capture move data?
    # spath = get_straightline_path_to(pts[0], robot_pos)
    # if spath.size > 0:
    #     pts = np.concatenate((spath, pts), axis = 0)

    return pts


class TrajectoryDataSaver:
    def __init__(self, root):
        print(f"TrajectoryDataSaver saving to {root}")
        self.save_folder = root
        self.img_folder = os.path.join(self.save_folder, "rgb")
        self.img_folder_dbg = os.path.join(self.save_folder, "rgb_dbg")
        self.depth_folder = os.path.join(self.save_folder, "depth")
        self.seg_folder = os.path.join(self.save_folder, "seg")
        self.trav_folder = os.path.join(self.save_folder, "trav")

        if os.path.exists(self.save_folder):
            print(f"rmtree {self.save_folder}")
            shutil.rmtree(self.save_folder)

        print(f"trying to create {self.save_folder}")
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)

        for x in [
            self.img_folder,
            self.img_folder_dbg,
            self.depth_folder,
            self.seg_folder,
            self.trav_folder,
        ]:
            self.create(x)

        self.pose_dict = {}
        self.pose_dict_hab = {}
        self.img_count = 0
        self.dbg_str = "None"
        self.init_logger()

    def init_logger(self):
        self.logger = logging.getLogger("trajectory_saver")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(filename)s:%(lineno)s - %(funcName)s(): %(message)s")
        # Enable filehandler to debug logs
        fh = logging.FileHandler(f"trajectory_saver.log", "a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

    def create(self, d):
        if not os.path.isdir(d):
            os.makedirs(d)

    def set_dbg_str(self, x):
        self.dbg_str = x

    def get_total_frames(self):
        return self.img_count

    def save(self, rgb, depth, seg, pos, habitat_pos, habitat_rot):
        self.img_count = len(glob.glob(self.img_folder + "/*.jpg"))
        self.logger.info(f"Saving to {self.save_folder}, {self.img_count}, {self.dbg_str}")
        print(f"saving {rgb.shape, depth.shape, seg.shape}")
        # store the images and depth
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.img_folder + "/{:05d}.jpg".format(self.img_count), rgb)

        cv2.putText(
            rgb,
            str(self.img_count) + " " + self.dbg_str,
            (40, 40),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
        )

        # robot_dbg_str = 'robot_pose ' + str(np.round(self.get_robot_global_state(), 3))
        # cv2.putText(rgb, robot_dbg_str, (40,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

        cv2.imwrite(self.img_folder_dbg + "/{:05d}.jpg".format(self.img_count), rgb)

        # store depth
        np.save(self.depth_folder + "/{:05d}.npy".format(self.img_count), depth)

        # store seg
        np.save(self.seg_folder + "/{:05d}.npy".format(self.img_count), seg)

        # store pos
        if os.path.isfile(os.path.join(self.save_folder, "data.json")):
            with open(os.path.join(self.save_folder, "data.json"), "r") as fp:
                self.pose_dict = json.load(fp)

        self.pose_dict[self.img_count] = copy(pos)

        with open(os.path.join(self.save_folder, "data.json"), "w") as fp:
            json.dump(self.pose_dict, fp)

        # store habitat pos
        if os.path.isfile(os.path.join(self.save_folder, "data_hab.json")):
            with open(os.path.join(self.save_folder, "data_hab.json"), "r") as fp:
                self.pose_dict_hab = json.load(fp)

        self.pose_dict_hab[self.img_count] = {
            "position": copy(habitat_pos),
            "rotation": copy(habitat_rot),
        }

        with open(os.path.join(self.save_folder, "data_hab.json"), "w") as fp:
            json.dump(self.pose_dict_hab, fp)


def visualize_examine(agent, robot_poses, object_xyz, label, obstacle_map, save_path, gt_pts=None):
    traj_visual_dir = os.path.join(save_path, "traj_visual")
    if not os.path.isdir(traj_visual_dir):
        os.makedirs(traj_visual_dir)
    vis_count = len(glob.glob(traj_visual_dir + "/*.jpg"))
    if vis_count == 0:
        plt.figure()

    plt.title("Examine Visual")
    # visualize obstacle map
    if len(obstacle_map) > 0:
        obstacle_map = np.asarray([list(x) for x in obstacle_map])
        plt.plot(obstacle_map[:, 1], obstacle_map[:, 0], "b+")

    # visualize object
    if object_xyz is not None:
        plt.plot(object_xyz[0], object_xyz[2], "y*")
        plt.text(object_xyz[0], object_xyz[2], label)

    # visualize robot pose
    if len(robot_poses) > 0:
        robot_poses = np.asarray(robot_poses)
        plt.plot(robot_poses[:, 0], robot_poses[:, 1], "r--")

    if gt_pts is not None:
        pts = np.asarray(gt_pts)
        plt.plot(pts[:, 0], pts[:, 1], "y--")

    # TODO: visualize robot heading

    plt.savefig("{}/{:04d}.jpg".format(traj_visual_dir, vis_count))
    vis_count += 1
