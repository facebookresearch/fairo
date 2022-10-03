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

from rich import print

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


def val_in_range(val_name, val, vmin, vmax):
    p = val <= vmax and val >= vmin
    if p:
        print(Fore.GREEN + "[Pass] " + val_name + " with " + str(val))
    else:
        print(
            Fore.RED
            + "[Fail] "
            + val_name
            + " with "
            + str(val)
            + " out of range "
            + str(vmin)
            + " to "
            + str(vmax)
        )


# #####################################################


@Pyro4.expose
class RemoteHelloRobot(object):
    """Hello Robot interface"""

    def __init__(self, ip):
        self._ip = ip
        self._robot = Robot()
        self._robot.startup()
        if not self._robot.is_calibrated():
            self._robot.home()
        self._robot.stow()
        self._done = True
        self.cam = None
        # Read battery maintenance guide https://docs.hello-robot.com/battery_maintenance_guide/
        self._check_battery()
        self._load_urdf()
        self.tilt_correction = 0.0

    def _check_battery(self):
        p = self._robot.pimu
        p.pull_status()
        val_in_range("Voltage", p.status["voltage"], vmin=p.config["low_voltage_alert"], vmax=14.0)
        val_in_range("Current", p.status["current"], vmin=0.1, vmax=p.config["high_current_alert"])
        val_in_range("CPU Temp", p.status["cpu_temp"], vmin=15, vmax=80)
        print(Style.RESET_ALL)

    def _load_urdf(self):
        import os

        urdf_path = os.path.join(
            os.getenv("HELLO_FLEET_PATH"),
            os.getenv("HELLO_FLEET_ID"),
            "exported_urdf",
            "stretch.urdf",
        )

        from pytransform3d.urdf import UrdfTransformManager
        import pytransform3d.transformations as pt
        import pytransform3d.visualizer as pv
        import numpy as np

        self.tm = UrdfTransformManager()
        with open(urdf_path, "r") as f:
            urdf = f.read()
            self.tm.load_urdf(urdf)

    def set_tilt_correction(self, angle):
        """
        angle in radians
        """
        print(
            "[hello-robot] Setting tilt correction " "to angle: {} degrees".format(degrees(angle))
        )

        self.tilt_correction = angle

    def get_camera_transform(self):
        s = self._robot.get_status()
        head_pan = s["head"]["head_pan"]["pos"]
        head_tilt = s["head"]["head_tilt"]["pos"]

        if self.tilt_correction != 0.0:
            head_tilt += self.tilt_correction

        # Get Camera transform
        self.tm.set_joint("joint_head_pan", head_pan)
        self.tm.set_joint("joint_head_tilt", head_tilt)
        camera_transform = self.tm.get_transform("camera_color_frame", "base_link")

        # correct for base_link's z offset from the ground
        # at 0, the correction is -0.091491526943
        # at 90, the correction is +0.11526719 + -0.091491526943
        # linear interpolate the correction of 0.023775
        interp_correction = 0.11526719 * abs(head_tilt) / radians(90)
        # print('interp_correction', interp_correction)

        camera_transform[2, 3] += -0.091491526943 + interp_correction

        return camera_transform

    def get_status(self):
        return self._robot.get_status()

    def get_base_state(self):
        s = self._robot.get_status()
        return (s["base"]["x"], s["base"]["y"], s["base"]["theta"])

    def get_pan(self):
        s = self._robot.get_status()
        return s["head"]["head_pan"]["pos"]

    def get_tilt(self):
        s = self._robot.get_status()
        return s["head"]["head_tilt"]["pos"]

    def set_pan(self, pan):
        self._robot.head.move_to("head_pan", pan)

    def set_tilt(self, tilt):
        self._robot.head.move_to("head_tilt", tilt)

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
        self._robot.head.move_to("head_pan", pan)
        self._robot.head.move_to("head_tilt", tilt)

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

    def initialize_cam(self):
        if self.cam is None:
            # wait for realsense service to be up and running
            time.sleep(2)
            with Pyro4.Daemon(self._ip) as daemon:
                cam = Pyro4.Proxy("PYRONAME:hello_realsense@" + self._ip)
            self.cam = cam

    def go_to_absolute(self, xyt_position):
        """Moves the robot base to given goal state in the world frame.

        :param xyt_position: The goal state of the form (x,y,yaw)
                             in the world (map) frame.
        """
        status = "SUCCEEDED"
        if self._done:
            self.initialize_cam()
            self._done = False
            global_xyt = xyt_position
            base_state = self.get_base_state()
            base_xyt = transform_global_to_base(global_xyt, base_state)

            def obstacle_fn():
                return self.cam.is_obstacle_in_front()

            status = goto(self._robot, list(base_xyt), dryrun=False, obstacle_fn=obstacle_fn)
            self._done = True
        return status

    def go_to_relative(self, xyt_position):
        """Moves the robot base to the given goal state relative to its current
        pose.

        :param xyt_position: The  relative goal state of the form (x,y,yaw)
        """
        status = "SUCCEEDED"

        if self._done:
            self.initialize_cam()
            self._done = False

            def obstacle_fn():
                return self.cam.is_obstacle_in_front()

            status = goto(self._robot, list(xyt_position), dryrun=False, obstacle_fn=obstacle_fn)
            self._done = True
        return status

    def is_busy(self):
        return not self.is_moving()

    def is_moving(self):
        return not self._done

    def stop(self):
        robot.stop()
        robot.push_command()

    def remove_runstop(self):
        if robot.pimu.status["runstop_event"]:
            robot.pimu.runstop_event_reset()
            robot.push_command()


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
        robot = RemoteHelloRobot(ip=args.ip)
        robot_uri = daemon.register(robot)
        with Pyro4.locateNS() as ns:
            ns.register("hello_robot", robot_uri)

        print("Hello Robot Server is started...")
        daemon.requestLoop()
