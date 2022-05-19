#!/usr/bin/env python

'''
This script is a test script for Franka robot arm and robotiq gripper attached to the NUC

@author Jay D Vakil
@organization Meta - FAIR pitt 4720
'''

from ipaddress import ip_address
from turtle import position
from polymetis import RobotInterface, GripperInterface
import time
import torch
import argparse

#Allows the user to send commands and change the IP according to their system
parser = argparse.ArgumentParser()
parser.add_argument("--robot-ip", default="172.16.0.1", help="robot ip address")
parser.add_argument("--gripper-ip", default="172.16.0.1", help="gripper ip address")
parser.add_argument("--time", default=5, help="time to move the robot")
args = parser.parse_args()

#Initialize the robot and gripper interface
robot = RobotInterface(ip_address = args.robot_ip, enforce_version=False)
gripper = GripperInterface(ip_address = args.gripper_ip)

#time to move the robot
timeRobot = args.time

"""
This function moves the robot in this trajectory: home -> location 1 -> home -> location 2 -> home
and tests the gripper by opening it and closing it. 
"""
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

#This loop keeps the robot moving until interrupted by the user
count = 1
while True:
    print (count)
    robotRy()
    count += 1