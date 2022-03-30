#!/usr/bin/env python
import torch
import json
import time
from polymetis import RobotInterface
from utils import ExtRobotInterface

hand = ExtRobotInterface(ip_address="localhost")
arm = RobotInterface(ip_address="192.168.0.10", enforce_version=False)

def move_arm_joint(pose):
    with open(pose) as f:
        armpose = json.load(f)
        q_home=torch.tensor(armpose['joints'])
        print('homing to', q_home)
        arm.move_to_joint_positions(q_home, time_to_go=4.)


def move_hand(pose, time_to_go=2.0, time_to_hold=0, blocking=False):
    with open(pose) as f:
        pos=torch.tensor(json.load(f))
    hand.move_to_joint_positions(
        pos, 
        time_to_go=time_to_go,
        time_to_hold=time_to_hold,
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

print('homing, please wait')
move_hand('poses/prec_pre_grasp.json', 3.0)
move_arm_joint('armposes/grasp_ready.json')
print('ready')


print('pregrasp')
move_hand('poses/prec_pre_grasp.json', 3.0)
#move_arm_joint('armposes/grasp_pre_grasp.json')
move_arm('armposes/grasp_pre_grasp.json', time_to_go=3.0, blocking=True, offset=[0., 0., -0.01])

print('grasp')
move_hand('poses/prec_grasp.json', time_to_go=1.0, time_to_hold=4.0, blocking=False)
time.sleep(1)

print('lift')
move_arm('armposes/grasp_pre_grasp.json', time_to_go=2.0, blocking=True, offset=[0., -0.1, 0.25])
move_arm('armposes/grasp_pre_grasp.json', time_to_go=1.0, blocking=True, offset=[0., -0.2, 0.25])

#print('put back')
#move_arm('armposes/grasp_pre_grasp.json', time_to_go=3.0, blocking=True, offset=[0., 0., 0.0])
move_hand('poses/prec_pre_grasp.json', 2.0, blocking=True)

#print('return')
#move_arm('armposes/grasp_pre_grasp.json', time_to_go=3.0, blocking=True, offset=[0., 0., 0.2])


