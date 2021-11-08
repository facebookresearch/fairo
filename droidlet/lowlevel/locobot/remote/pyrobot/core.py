# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import copy
import importlib
import os
import sys
import threading
import time
from abc import ABCMeta, abstractmethod

import numpy as np
from .utils import util as prutil
from .utils.util import try_cv2_import

cv2 = try_cv2_import()

class Robot:
    """
    This is the main interface class that is composed of
    key robot modules (base, arm, gripper, and camera).
    This class builds robot specific objects by reading a
    configuration and instantiating the necessary robot module objects.

    """

    def __init__(
        self,
        robot_name,
        use_arm=True,
        use_base=True,
        use_camera=True,
        use_gripper=True,
        arm_config={},
        base_config={},
        camera_config={},
        gripper_config={},
        common_config={},
    ):
        """
        Constructor for the Robot class

        :param robot_name: robot name
        :param use_arm: use arm or not
        :param use_base: use base or not
        :param use_camera: use camera or not
        :param use_gripper: use gripper or not
        :param arm_config: configurations for arm
        :param base_config: configurations for base
        :param camera_config: configurations for camera
        :param gripper_config: configurations for gripper

        :type robot_name: string
        :type use_arm: bool
        :type use_base: bool
        :type use_camera: bool
        :type use_gripper: bool
        :type arm_config: dict
        :type base_config: dict
        :type camera_config: dict
        :type gripper_config: dict
        """
        root_path = os.path.dirname(os.path.realpath(__file__))
        cfg_path = os.path.join(root_path, "cfg")
        robot_pool = []
        for f in os.listdir(cfg_path):
            if f.endswith("_config.py"):
                robot_pool.append(f[: -len("_config.py")])
        root_node = "pyrobot."
        self.configs = None
        this_robot = None
        for srobot in robot_pool:
            if srobot in robot_name:
                this_robot = srobot
                mod = importlib.import_module(
                    root_node + "cfg." + "{:s}_config".format(srobot)
                )
                cfg_func = getattr(mod, "get_cfg")
                if srobot == "locobot" and "lite" in robot_name:
                    self.configs = cfg_func("create")
                else:
                    self.configs = cfg_func()
        if self.configs is None:
            raise ValueError(
                "Invalid robot name provided, only the following"
                " are currently available: {}".format(robot_pool)
            )
        self.configs.freeze()

        root_node += this_robot
        root_node += "."
        if self.configs.HAS_COMMON:
            mod = importlib.import_module(root_node + self.configs.COMMON.NAME)
            common_class = getattr(mod, self.configs.COMMON.CLASS)
            setattr(
                self,
                self.configs.COMMON.NAME,
                common_class(self.configs, **common_config),
            )
        if self.configs.HAS_ARM and use_arm:
            mod = importlib.import_module(root_node + "arm")
            arm_class = getattr(mod, self.configs.ARM.CLASS)
            if self.configs.HAS_COMMON:
                arm_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.arm = arm_class(self.configs, **arm_config)
        if self.configs.HAS_BASE and use_base:
            mod = importlib.import_module(root_node + "base")
            base_class = getattr(mod, self.configs.BASE.CLASS)
            if self.configs.HAS_COMMON:
                base_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.base = base_class(self.configs, **base_config)
        if self.configs.HAS_CAMERA and use_camera:
            mod = importlib.import_module(root_node + "camera")
            camera_class = getattr(mod, self.configs.CAMERA.CLASS)
            if self.configs.HAS_COMMON:
                camera_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.camera = camera_class(self.configs, **camera_config)
        if self.configs.HAS_GRIPPER and use_gripper and use_arm:
            mod = importlib.import_module(root_node + "gripper")
            gripper_class = getattr(mod, self.configs.GRIPPER.CLASS)
            if self.configs.HAS_COMMON:
                gripper_config[self.configs.COMMON.NAME] = getattr(
                    self, self.configs.COMMON.NAME
                )
            self.gripper = gripper_class(self.configs, **gripper_config)

        # sleep some time for tf listeners in subclasses
        # rospy.sleep(2)


class Base(object):
    """
    This is a parent class on which the robot
    specific Base classes would be built.
    """

    def __init__(self, configs):
        """
        The consturctor for Base class.

        :param configs: configurations for base
        :type configs: YACS CfgNode
        """
        self.configs = configs
        self.ctrl_pub = rospy.Publisher(
            configs.BASE.ROSTOPIC_BASE_COMMAND, Twist, queue_size=1
        )

    def stop(self):
        """
        Stop the base
        """
        msg = Twist()
        msg.linear.x = 0
        msg.angular.z = 0
        self.ctrl_pub.publish(msg)

    def set_vel(self, fwd_speed, turn_speed, exe_time=1):
        """
        Set the moving velocity of the base

        :param fwd_speed: forward speed
        :param turn_speed: turning speed
        :param exe_time: execution time
        """
        fwd_speed = min(fwd_speed, self.configs.BASE.MAX_ABS_FWD_SPEED)
        fwd_speed = max(fwd_speed, -self.configs.BASE.MAX_ABS_FWD_SPEED)
        turn_speed = min(turn_speed, self.configs.BASE.MAX_ABS_TURN_SPEED)
        turn_speed = max(turn_speed, -self.configs.BASE.MAX_ABS_TURN_SPEED)

        msg = Twist()
        msg.linear.x = fwd_speed
        msg.angular.z = turn_speed

        start_time = rospy.get_time()
        self.ctrl_pub.publish(msg)
        while rospy.get_time() - start_time < exe_time:
            self.ctrl_pub.publish(msg)
            rospy.sleep(1.0 / self.configs.BASE.BASE_CONTROL_RATE)

    def go_to_relative(self, xyt_position, use_map, close_loop, smooth):
        """
        Moves the robot to the robot to given
        goal state relative to its initial pose.

        :param xyt_position: The  relative goal state of the form (x,y,t)
        :param use_map: When set to "True", ensures that controler is
                        using only free space on the map to move the robot.
        :param close_loop: When set to "True", ensures that controler is
                           operating in open loop by
                           taking account of odometry.
        :param smooth: When set to "True", ensures that the motion
                       leading to the goal is a smooth one.

        :type xyt_position: list
        :type use_map: bool
        :type close_loop: bool
        :type smooth: bool

        :return: True if successful; False otherwise (timeout, etc.)
        :rtype: bool
        """
        raise NotImplementedError

    def go_to_absolute(self, xyt_position, use_map, close_loop, smooth):
        """
        Moves the robot to the robot to given goal state in the world frame.

        :param xyt_position: The goal state of the form (x,y,t)
                             in the world (map) frame.
        :param use_map: When set to "True", ensures that controler is using
                        only free space on the map to move the robot.
        :param close_loop: When set to "True", ensures that controler is
                           operating in open loop by
                           taking account of odometry.
        :param smooth: When set to "True", ensures that the motion
                       leading to the goal is a smooth one.

        :type xyt_position: list
        :type use_map: bool
        :type close_loop: bool
        :type smooth: bool

        :return: True if successful; False otherwise (timeout, etc.)
        :rtype: bool
        """
        raise NotImplementedError

    def track_trajectory(self, states, controls, close_loop):
        """
        State trajectory that the robot should track.

        :param states: sequence of (x,y,t) states that the robot should track.
        :param controls: optionally specify control sequence as well.
        :param close_loop: whether to close loop on the
                           computed control sequence or not.

        :type states: list
        :type controls: list
        :type close_loop: bool

        :return: True if successful; False otherwise (timeout, etc.)
        :rtype: bool
        """
        raise NotImplementedError

    def get_state(self, state_type):
        """
        Returns the requested base pose in the (x,y, yaw) format.

        :param state_type: Requested state type. Ex: Odom, SLAM, etc
        :type state_type: string
        :return: pose: pose of the form [x, y, yaw]
        :rtype: list
        """
        raise NotImplementedError


class Camera(object):
    """
    This is a parent class on which the robot
    specific Camera classes would be built.
    """

    __metaclass__ = ABCMeta

    def __init__(self, configs):
        """
        Constructor for Camera parent class.

        :param configs: configurations for camera
        :type configs: YACS CfgNode
        """
        self.configs = configs
        self.cv_bridge = CvBridge()
        self.camera_info_lock = threading.RLock()
        self.camera_img_lock = threading.RLock()
        self.rgb_img = None
        self.depth_img = None
        self.camera_info = None
        self.camera_P = None
        rospy.Subscriber(
            self.configs.CAMERA.ROSTOPIC_CAMERA_INFO_STREAM,
            CameraInfo,
            self._camera_info_callback,
        )

        rgb_topic = self.configs.CAMERA.ROSTOPIC_CAMERA_RGB_STREAM
        self.rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_topic = self.configs.CAMERA.ROSTOPIC_CAMERA_DEPTH_STREAM
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        img_subs = [self.rgb_sub, self.depth_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            img_subs, queue_size=10, slop=0.2
        )
        self.sync.registerCallback(self._sync_callback)

    def _sync_callback(self, rgb, depth):
        self.camera_img_lock.acquire()
        try:
            self.rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb, "bgr8")
            self.rgb_img = self.rgb_img[:, :, ::-1]
            self.depth_img = self.cv_bridge.imgmsg_to_cv2(depth, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
        self.camera_img_lock.release()

    def _camera_info_callback(self, msg):
        self.camera_info_lock.acquire()
        self.camera_info = msg
        self.camera_P = np.array(msg.P).reshape((3, 4))
        self.camera_info_lock.release()

    def get_rgb(self):
        """
        This function returns the RGB image perceived by the camera.

        :rtype: np.ndarray or None
        """
        self.camera_img_lock.acquire()
        rgb = copy.deepcopy(self.rgb_img)
        self.camera_img_lock.release()
        return rgb

    def get_depth(self):
        """
        This function returns the depth image perceived by the camera.

        The depth image is in meters.

        :rtype: np.ndarray or None
        """
        self.camera_img_lock.acquire()
        depth = copy.deepcopy(self.depth_img)
        self.camera_img_lock.release()
        depth = depth / self.configs.CAMERA.DEPTH_MAP_FACTOR
        return depth

    def get_rgb_depth(self):
        """
        This function returns both the RGB and depth
        images perceived by the camera.

        The depth image is in meters.

        :rtype: np.ndarray or None
        """
        self.camera_img_lock.acquire()
        rgb = copy.deepcopy(self.rgb_img)
        depth = copy.deepcopy(self.depth_img)
        depth = depth / self.configs.CAMERA.DEPTH_MAP_FACTOR
        self.camera_img_lock.release()
        return rgb, depth

    def get_intrinsics(self):
        """
        This function returns the camera intrinsics.

        :rtype: np.ndarray
        """
        if self.camera_P is None:
            return self.camera_P
        self.camera_info_lock.acquire()
        P = copy.deepcopy(self.camera_P)
        self.camera_info_lock.release()
        return P[:3, :3]


