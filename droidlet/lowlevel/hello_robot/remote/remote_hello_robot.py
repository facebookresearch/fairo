"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# python -m Pyro4.naming -n <MYIP>
import select
import logging
import os
import json
import time
import copy
from math import *

import Pyro4
from stretch_body.robot import Robot
from colorama import Fore, Back, Style
import stretch_body.hello_utils as hu
hu.print_stretch_re_use()
import numpy as np
import cv2
from droidlet.lowlevel.hello_robot.remote.utils import transform_global_to_base, goto


Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True

def val_in_range(val_name, val,vmin, vmax):
    p=val <=vmax and val>=vmin
    if p:
        print(Fore.GREEN +'[Pass] ' + val_name + ' with ' + str(val))
    else:
        print(Fore.RED +'[Fail] ' + val_name + ' with ' +str(val)+ ' out of range ' +str(vmin) + ' to ' + str(vmax))

# #####################################################

@Pyro4.expose
class RemoteHelloRobot(object):
    """Hello Robot interface"""

    def __init__(self):
        self._robot = Robot()
        self._robot.startup()
        if not self._robot.is_calibrated():
            self._robot.home()
        self._robot.stow()
        self._done = True
        # Read battery maintenance guide https://docs.hello-robot.com/battery_maintenance_guide/
        self._check_battery()
        self._load_urdf()
    
    def _check_battery(self):
        p = self._robot.pimu
        p.pull_status()
        val_in_range('Voltage',p.status['voltage'], vmin=p.config['low_voltage_alert'], vmax=14.0)
        val_in_range('Current',p.status['current'], vmin=0.1, vmax=p.config['high_current_alert'])
        val_in_range('CPU Temp',p.status['cpu_temp'], vmin=15, vmax=80)
        print(Style.RESET_ALL)
    
    def _load_urdf(self):
        import os
        urdf_path = os.path.join(os.getenv("HELLO_FLEET_PATH"), os.getenv("HELLO_FLEET_ID"), "exported_urdf", "stretch.urdf")

        from pytransform3d.urdf import UrdfTransformManager
        import pytransform3d.transformations as pt
        import pytransform3d.visualizer as pv
        import numpy as np

        self.tm = UrdfTransformManager()
        with open(urdf_path, "r") as f:
            urdf = f.read()
            self.tm.load_urdf(urdf)
        
    def get_camera_transform(self):
        s = self._robot.get_status()
        head_pan = s['head']['head_pan']['pos']
        head_tilt = s['head']['head_tilt']['pos']

        # Get Camera transform
        self.tm.set_joint("joint_head_pan", head_pan)
        self.tm.set_joint("joint_head_pan", head_pan)
        camera_transform = self.tm.get_transform('camera_link', 'base_link')

        return camera_transform

    def get_status(self):
        return self._robot.get_status()

    def get_base_state(self):
        s = self._robot.get_status()
        return (s['base']['x'], s['base']['y'], s['base']['theta'])

    def get_pan(self):
        s = self._robot.get_status()
        return s['head']['head_pan']['pos']

    def get_tilt(self):
        s = self._robot.get_status()
        return s['head']['head_tilt']['pos']

    def set_pan(self, pan):
        self._robot.head.move_to('head_pan', pan)

    def set_tilt(self, tilt):
        self._robot.head.move_to('head_tilt', tilt)
    
    def reset_camera(self):
        self.set_pan(0)
        self.set_tilt(0)

    def set_pan_tilt(self, pan, tilt):
        """Sets both the pan and tilt joint angles of the robot camera  to the
        specified values.

        :param pan: value to be set for pan joint in radian
        :param tilt: value to be set for the tilt joint in radian

        :type pan: float
        :type tilt: float
        :type wait: bool
        """
        self._robot.head.move_to('head_pan', pan)
        self._robot.head.move_to('head_tilt', tilt)

    def test_connection(self):
        print("Connected!!")  # should print on server terminal
        return "Connected!"  # should print on client terminal

    def home(self):
        self._robot.home()

    def stow(self):
        self._robot.stow()

    def push_command(self):
        self._robot.push_command()

    def translate_by(self, x_m):
        self._robot.base.translate_by(x_m)
        self._robot.push_command()

    def rotate_by(self, x_r):
        self._robot.base.rotate_by(x_r)
        self._robot.push_command()

    def go_to_absolute(
        self,
        xyt_position,
        use_map=False,
        close_loop=True,
        smooth=False,
        use_dslam=False,
    ):
        """Moves the robot base to given goal state in the world frame.

        :param xyt_position: The goal state of the form (x,y,yaw)
                             in the world (map) frame.
        :param use_map: When set to "True", ensures that controller is
                        using only free space on the map to move the robot.
        :param close_loop: When set to "True", ensures that controller
                           is operating in open loop by taking
                           account of odometry.
        :param smooth: When set to "True", ensures that the motion
                       leading to the goal is a smooth one.
        :param use_dslam: When set to "True", the robot uses slam for
                          the navigation.

        :type xyt_position: list or np.ndarray
        :type use_map: bool
        :type close_loop: bool
        :type smooth: bool
        """
        assert(use_map == False)
        assert(close_loop == True)
        assert(smooth == False)
        assert(use_dslam == False)
        if self._done:
            self._done = False
            global_xyt = xyt_position
            base_state = self.get_base_state()
            base_xyt = transform_global_to_base(global_xyt, base_state)
            goto(self._robot, list(base_xyt), dryrun=False)
            self._done = True

    def go_to_relative(
        self,
        xyt_position,
        use_map=False,
        close_loop=True,
        smooth=False,
        use_dslam=False,
    ):
        """Moves the robot base to the given goal state relative to its current
        pose.

        :param xyt_position: The  relative goal state of the form (x,y,yaw)
        :param use_map: When set to "True", ensures that controller is
                        using only free space on the map to move the robot.
        :param close_loop: When set to "True", ensures that controller is
                           operating in open loop by taking
                           account of odometry.
        :param smooth: When set to "True", ensures that the
                       motion leading to the goal is a smooth one.
        :param use_dslam: When set to "True", the robot uses slam for
                          the navigation.

        :type xyt_position: list or np.ndarray
        :type use_map: bool
        :type close_loop: bool
        :type smooth: bool
        """
        assert(use_map == False)
        assert(close_loop == True)
        assert(smooth == False)
        assert(use_dslam == False)
        if self._done:
            self._done = False
            goto(self._robot, list(xyt_position), dryrun=False)
            self._done = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 192.168.0.0",
        type=str,
        default="0.0.0.0",
    )

    args = parser.parse_args()

    np.random.seed(123)

    with Pyro4.Daemon(args.ip) as daemon:
        robot = RemoteHelloRobot()
        robot_uri = daemon.register(robot)
        with Pyro4.locateNS() as ns:
            ns.register("remotehellorobot", robot_uri)

        print("Server is started...")
        daemon.requestLoop()
