#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

import rospy
import time
import os

# Robot planning tools
from home_robot.hardware.stretch_ros import HelloStretchROSInterface
from home_robot.motion.robot import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.ros.path import get_package_path

import numpy as np
import sklearn
import matplotlib.pyplot as plt

import torch
import hydra
import omegaconf

from polygrasp.segmentation_rpc import SegmentationClient
from polygrasp.grasp_rpc import GraspClient
from polygrasp.serdes import load_bw_img

from polygrasp.robot_interface import GraspingRobotInterface
import graspnetAPI
import open3d as o3d
from typing import List

import cv2

"""
Manual installs needed for:
    tracikpy
    home_robot
"""


def main():
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node('hello_stretch_ros_test')
    print("Create ROS interface")
    rob = HelloStretchROSInterface(visualize_planner=False, root=get_package_path())
    print("Wait...")
    rospy.sleep(0.5)  # Make sure we have time to get ROS messages
    for i in range(1):
        q = rob.update()
        print(rob.get_base_pose())
    print("--------------")
    print("We have updated the robot state. Now test goto.")

    home_q = STRETCH_HOME_Q
    model = rob.get_model()
    q = model.update_look_front(home_q.copy())
    rob.goto(q, move_base=False, wait=True)

    # Robot - look at the object because we are switching to grasping mode
    # Send robot to home_q + wait
    q = model.update_look_at_ee(home_q.copy())
    rob.goto(q, move_base=False, wait=True)
    # rob.look('tool')
    # Send it to lift pose + wait
    #q, _ = rob.update()
    q[HelloStretchIdx.ARM] = 0.06
    q[HelloStretchIdx.LIFT] = 0.85
    rob.goto(q, move_base=False, wait=True, verbose=False)


if __name__ == "__main__":
    main()

