#!/usr/bin/env python
import torch
import json
import time
from polymetis import RobotInterface


hand = RobotInterface(ip_address="localhost")
arm = RobotInterface(ip_address="192.168.0.10", enforce_version=False)
arm.go_home()

def move_hand(pose, time_to_go=2.0, blocking=False):
    with open(pose) as f:
        pos=torch.tensor(json.load(f))
    hand.move_to_joint_positions(
        pos, 
        time_to_go=time_to_go,
        blocking=blocking)

def move_arm(pose, time_to_go=2.0, blocking=False, offset=[0., 0., 0.]):
    with open(pose) as f:
        armpose = json.load(f)
    pos = torch.tensor(armpose['position']) + torch.tensor(offset)
    print('armpose', pos)
    arm.move_to_ee_pose(
        position=pos,
        orientation=torch.tensor(armpose['orientation']),
        time_to_go=time_to_go,
        blocking=blocking)


zoffset=0.00
print('ready')
move_hand('poses/mid.json', 3.0)
move_arm('armposes/ready.json', 3.0, blocking=True, offset=[-0.1, 0, zoffset])

def touch(offset):
    print('touch')
    move_hand('poses/pointer.json', 2.0)
    move_arm('armposes/pointer_touch.json', 3.0, blocking=True, offset=offset)
    move_arm('armposes/pointer_touch_near.json', 3.0, blocking=True, offset=offset)
    print('lift')
    move_hand('poses/mid.json', 3.0)
    move_arm('armposes/ready.json', 3.0, blocking=True, offset=offset)


touch([-0.1,  0.05, zoffset])
touch([-0.22,  0.05, zoffset])
touch([-0.1, -0.05, zoffset])
touch([-0.22, -0.05, zoffset])
