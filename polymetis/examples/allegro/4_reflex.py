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
        q_home = torch.tensor(armpose["joints"])
        print("homing to", q_home)
        arm.move_to_joint_positions(q_home, time_to_go=4.0)


def move_hand(pose, time_to_go=2.0, time_to_hold=0, blocking=False, Kq_multiplier=1):
    with open(pose) as f:
        pos = torch.tensor(json.load(f))

    hand.move_to_joint_positions(
        pos,
        time_to_go=time_to_go,
        time_to_hold=time_to_hold,
        Kq=hand.Kq_default * Kq_multiplier,
        blocking=blocking,
    )


print("homing, please wait")
move_hand("poses/pointer.json", 3.0)
move_arm_joint("armposes/reflex_ready.json")
print("ready")

#
for i in range(15):
    move_hand("poses/reflex_touch.json", 2.0, blocking=True)
    move_hand("poses/pointer.json", time_to_go=0.1, time_to_hold=3, blocking=True)
