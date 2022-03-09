#  conda create -n eyehandcal polymetis librealsense opencv tqdm -c fair-robotics -c conda-forge
from torchcontrol.transform import Rotation as R
from polymetis import RobotInterface
from math import pi
import numpy as np
import torch
import time
import cv2
import pickle
from tqdm import tqdm

from realsense_wrapper import RealsenseAPI

def intrinsics2dict(intrinsics):
    if isinstance(intrinsics, list):
        return [intrinsics2dict(x) for x in intrinsics]
    return {
        "coeffs": intrinsics.coeffs,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "height": intrinsics.height,
        "model": intrinsics.model,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "width": intrinsics.width,
    }


def realsense_images():
    rs = RealsenseAPI()
    num_cameras = rs.get_num_cameras()
    assert num_cameras > 0, "no camera found"

    intrinsics = intrinsics2dict(rs.get_intrinsics())
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
        enforce_version=False,
    )

    # Get reference state
    robot.go_home()
    time_to_go = 5

    sampled_poses = list(sample_poses())
    for i, (pos_sampled, ori_sampled) in enumerate(tqdm(sampled_poses)):
        while True:
            print( f"Moving to pose ({i}): pos={pos_sampled}, quat={ori_sampled.as_quat()}")
            state_log = robot.move_to_ee_pose(
                position=pos_sampled,
                orientation=ori_sampled.as_quat(),
                time_to_go = time_to_go
            )
            print(f"Length of state_log: {len(state_log)}")
            if len(state_log) == time_to_go * robot.hz:
                pos, quat = robot.get_ee_pose()
                print(f"Current pose  pos={pos}, quat={quat}")
                yield pos, quat
                break
            else:
                print(f"State log incorrect length. Trying again...")


if __name__ == "__main__":
    data = []
    ip_address = "192.168.1.65"
    poses = robot_poses(ip_address)
    images = realsense_images()
    for i, ((pos, ori), (imgs, intrinsics)) in enumerate(zip(poses, images)):
        cv2.imwrite(f'debug/debug_{i}.jpg', imgs[0])
        data.append({
            'pos': pos,
            'ori': ori,
            'imgs': imgs,
            'intrinsics': intrinsics
        })
    
    with open('caldata.pkl', 'wb') as f:
        pickle.dump(data, f)
