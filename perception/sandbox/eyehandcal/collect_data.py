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

    raw_intrinsics = rs.get_intrinsics()
    intrinsics = []
    for intrinsics_param in raw_intrinsics:
        intrinsics.append(
            dict([(p, getattr(intrinsics_param, p)) for p in dir(intrinsics_param) if not p.startswith('__')])
        )


    for i in range(30*5):
        rs.get_images()
    count=0
    while True:
        imgs0 = rs.get_images()
        for i in range(30):
            imgs1 = rs.get_images()
        pixel_diff=[]
        for i in range(num_cameras):
            pixel_diff.append(np.abs(imgs0[i].astype(np.int32)-imgs1[i].astype(np.int32)).reshape(-1))
        diff = np.concatenate(pixel_diff)
        if diff.max() > 50:
            print(f'image moving pixeldiff max: {diff.max()} p95: {np.percentile(diff, 95)}')
            continue
        print('get image', count)
        count+=1
        yield [img.copy() for img in imgs1], intrinsics



def sample_poses(high_camera=False):
    hand_mount_yaw_offset = -pi/4
    for x in np.linspace(0.3, 0.5, 3):
        for y in np.linspace(-0.2, 0.2, 3):
            for z in np.linspace(0.2, 0.4, 3):
                for yaw in np.linspace(-pi/8, pi/8, 3):
                    if high_cameras:
                        pos_sampled = torch.Tensor([x, y, z+.5])
                        ori_sampled = R.from_rotvec(torch.Tensor([pi/2, 0, 0])) * R.from_rotvec(torch.Tensor([0, 0, -pi + hand_mount_yaw_offset + yaw]))
                    else:
                        pos_sampled = torch.Tensor([x, y, z])
                        ori_sampled = R.from_rotvec(torch.Tensor([0, 0, hand_mount_yaw_offset + yaw]))
                    yield pos_sampled, ori_sampled


def robot_poses(ip_address):
    # Initialize robot interface
    robot = RobotInterface(
        ip_address=ip_address,
        enforce_version=False
    )

    # Get reference state
    robot.go_home()
    for i, (pos_sampled, ori_sampled) in enumerate(sample_poses()):
        print( f"Moving to pose ({i}): pos={pos_sampled}, quat={ori_sampled.as_quat()}")
        state_log = robot.move_to_ee_pose(
            position=pos_sampled,
            orientation=ori_sampled.as_quat(),
            time_to_go = 3
        )
        while True:
            pos0, quat0 = robot.get_ee_pose()
            time.sleep(1)
            pos1, quat1 = robot.get_ee_pose()
            diffpos = (pos0-pos1).norm()
            if diffpos < 0.01:
                break
            print(f'robot moving diffpos={diffpos}')

        print(f"Current pose  pos={pos0}, quat={quat0}")
        yield pos1, quat1

data = []
img_gen=realsense_images()
for i, (pos,ori) in enumerate(robot_poses('100.96.135.66')):
    imgs, intrinsics=next(img_gen)
    print(f'write {i}')
    cv2.imwrite(f'debug_{i}.jpg', imgs[1])
    data.append({
        'pos': pos,
        'ori': ori,
        'imgs': imgs,
        'intrinsics': intrinsics
    })

with open('caldata.pkl', 'wb') as f:
    pickle.dump(data, f)
