#  conda create -n eyehandcal polymetis librealsense opencv -c fair-robotics -c conda-forge
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from polymetis import RobotInterface
from math import pi
import numpy as np
import torch
import time
import cv2
import os 
import pickle

from realsense_wrapper import RealsenseAPI

def realsense_images():
    rs = RealsenseAPI()
    num_cameras = rs.get_num_cameras()
    assert num_cameras > 0, "no camera found"

    intrinsics = rs.get_intrinsics()
    while True:

        imgs0 = rs.get_images()
        imgs1 = rs.get_images()
        diff=0
        for i in range(num_cameras):
            diff += np.abs(imgs0[0].astype(np.int32)-imgs1[0].astype(np.int32)).mean()
        diff /= 3
        if diff > 5:
            print('robot moving', diff)
            time.sleep(1)
            continue
        yield imgs1, intrinsics



def sample_poses():
    hand_mount_yaw_offset = -pi/4
    for x in np.linspace(0.3, 0.5, 3):
        for y in np.linspace(-0.2, 0.2, 3):
            for z in np.linspace(0.2, 0.4, 3):
                for yaw in np.linspace(-pi/8, pi/8, 3):
                    pos_sampled = torch.Tensor([x, y, z])
                    ori_sampled = R.from_rotvec(torch.Tensor([0, 0, hand_mount_yaw_offset + yaw]))*R.from_rotvec(torch.Tensor([pi, 0, 0]))
                    yield pos_sampled, ori_sampled


def robot_poses(ip_address):
    # Initialize robot interface
    robot = RobotInterface(
        ip_address=ip_address,
    )

    # Get reference state
    robot.go_home()
    for i, (pos_sampled, ori_sampled) in enumerate(sample_poses()):
        print( f"Moving to pose ({i}): pos={pos_sampled}, quat={ori_sampled.as_quat()}")
        state_log = robot.set_ee_pose(
            position=pos_sampled,
            orientation=ori_sampled.as_quat(),
            time_to_go = 3
        )
        pos, quat = robot.pose_ee()
        print(f"Current pose  pos={pos}, quat={quat}")
        yield pos, quat



data = []
for i, (pos,ori), (imgs,intrinsics) in enumerate(zip(robot_poses('100.96.135.68'), realsense_images())):
    cv2.imwrite(f'debug_{i}.jpg', imgs[0])
    data.append({
        'pos': pos,
        'ori': ori,
        'imgs': imgs,
        'intrinsics': intrinsics
    })

with open('caldata.pkl', 'wb') as f:
    pickle.dump(f, data)