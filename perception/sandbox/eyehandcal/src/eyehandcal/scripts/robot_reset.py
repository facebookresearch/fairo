#!/usr/bin/env python

from ipaddress import ip_address
from turtle import position
from polymetis import RobotInterface, GripperInterface
import time
import torch

robot = RobotInterface(ip_address = "172.16.0.1", enforce_version=False)
gripper = GripperInterface(ip_address = "172.16.0.1")

timeRobot = 5
def robotRy():
    robot.go_home()
    print(" - Going home")
    gripper.goto(1.0,0.1,0.1)
    print(" - Gripper open")
    time.sleep(2)
    print(" - Robot moving")
    state_log = robot.move_to_ee_pose(position=torch.Tensor([0.5,0.3,0.22]), orientation=None, time_to_go=timeRobot)
    print(" - Gripper close")
    gripper.grasp(speed=0.1, force=0.1)
    time.sleep(2)
    print(" - Going home")
    robot.go_home()
    time.sleep(2)
    print(" - Robot moving")
    state_log = robot.move_to_ee_pose(position=torch.Tensor([0.5,-0.3,0.22]), orientation=None, time_to_go=timeRobot)
    print(" - Gripper open")
    gripper.goto(1.0,0.1,0.1)
    time.sleep(2)
    gripper.grasp(speed=0.1, force=1)
    print(" - Going home")
    robot.go_home()
count = 1
while True:
    print (count)
    robotRy()
    count += 1