#!/usr/bin/env python
import torch
import json
import time
import math
from polymetis import RobotInterface
from utils import ExtRobotInterface, MyPDPolicy

hand = ExtRobotInterface(ip_address="localhost")
arm = RobotInterface(ip_address="192.168.0.10", enforce_version=False)


def move_arm_joint(pose):
    with open(pose) as f:
        armpose = json.load(f)
        q_home = torch.tensor(armpose["joints"])
        print("homing to", q_home)
        arm.move_to_joint_positions(q_home, time_to_go=4.0)


def move_hand(pose, time_to_go=2.0, time_to_hold=0, blocking=False):
    with open(pose) as f:
        pos = torch.tensor(json.load(f))
    hand.move_to_joint_positions(
        pos, time_to_go=time_to_go, time_to_hold=time_to_hold, blocking=blocking
    )
    return pos


def move_arm(pose, time_to_go=2.0, blocking=False, offset=[0.0, 0.0, 0.0]):
    with open(pose) as f:
        armpose = json.load(f)
    pos = torch.tensor(armpose["position"]) + torch.tensor(offset)
    print("armpose", pos)
    arm.move_to_ee_pose(
        position=pos,
        orientation=torch.tensor(armpose["orientation"]),
        time_to_go=time_to_go,
        blocking=blocking,
    )


print("homing, please wait")
move_hand("poses/pre_pinch.json", 3.0, blocking=False)
move_arm(
    "armposes/pen_pre_spin.json",
    time_to_go=3.0,
    blocking=True,
    offset=[0.0, -0.1, -0.10],
)
# move_arm_joint('armposes/grasp_ready.json')
print("ready")


move_arm(
    "armposes/pen_pre_spin.json",
    time_to_go=3.0,
    blocking=True,
    offset=[0.0, 0.0, -0.10],
)

print("pinch")
qdes = move_hand("poses/pinch.json", time_to_go=1.0, blocking=False)
policy = MyPDPolicy(qdes, kq=hand.Kq_default, kqd=hand.Kqd_default)
hand.send_torch_policy(policy, blocking=False)

print("Starting sine motion updates...")
q_desired = torch.tensor(qdes)

time_to_go = 10.0
m = 0.2  # magnitude of sine wave (rad)
T = 0.5  # period of sine wave
hz = 50  # update frequency
joint1 = 0
joint2 = 12
for i in range(int(time_to_go * hz)):

    q_desired[joint1] = qdes[joint1] + m * math.sin(math.pi * i / (T * hz))
    q_desired[joint2] = qdes[joint2] + m * math.sin(math.pi * i / (T * hz))
    hand.update_current_policy({"q_desired": q_desired})
    time.sleep(1 / hz)

hand.terminate_current_policy()
move_hand("poses/pre_pinch.json", 3.0, blocking=True)
move_arm(
    "armposes/pen_pre_spin.json",
    time_to_go=3.0,
    blocking=True,
    offset=[0.0, -0.1, -0.10],
)


# print('lift')
# move_arm('armposes/grasp_pre_grasp.json', time_to_go=2.0, blocking=True, offset=[0., 0., 0.25])
