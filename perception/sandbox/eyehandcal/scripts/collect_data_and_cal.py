#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



from math import pi
import time
import os 
import pickle

import numpy as np
import torch
import cv2

from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from polymetis import RobotInterface
from realsense_wrapper import RealsenseAPI

from eyehandcal.utils import detect_corners, quat2rotvec, build_proj_matrix, mean_loss, find_parameter, rotmat
   


def realsense_images(max_pixel_diff=100):
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
        if diff.max() > max_pixel_diff:
            print(f'image moving pixeldiff max: {diff.max()} p95: {np.percentile(diff, 95)}')
            continue
        count+=1
        yield [img.copy() for img in imgs1], intrinsics



def sample_poses(overhead_cameras=True):
    hand_mount_yaw_offset = -pi/4
    for x in np.linspace(0.3, 0.5, 2):
        for y in np.linspace(-0.2, 0.2, 2):
            for z in np.linspace(0.2, 0.4, 2):
                for yaw in np.linspace(-pi/4, pi/4, 3):
                    if overhead_cameras:
                        pos_sampled = torch.Tensor([x, y, z+.1])
                        ori_sampled = R.from_rotvec(torch.Tensor([pi/2, 0, 0])) * R.from_rotvec(torch.Tensor([0, 0, hand_mount_yaw_offset + yaw]))
                    else:
                        pos_sampled = torch.Tensor([x, y, z])
                        ori_sampled = R.from_rotvec(torch.Tensor([0, 0, hand_mount_yaw_offset + yaw]))*R.from_rotvec(torch.Tensor([pi, 0, 0]))
                    yield pos_sampled, ori_sampled


def robot_poses(ip_address, pose_generator):
    # Initialize robot interface
    robot = RobotInterface(
        ip_address=ip_address,
        enforce_version=False
    )

    # Get reference state
    robot.go_home()
    for i, (pos_sampled, ori_sampled) in enumerate(pose_generator):
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


# helper function
def extract_obs_data_std(data, camera_index):
    obs_data_std = []
    for d in data:
        if d['corners'][camera_index] is not None:
            obs_data_std.append((
                torch.tensor(d['corners'][camera_index], dtype=torch.float64),
                d['pos'].double(),
                quat2rotvec(d['ori'].double())
            ))

    ic = data[0]['intrinsics'][camera_index]
    K=build_proj_matrix(
        fx=ic['fx'],
        fy=ic['fy'],
        ppx=ic['ppx'],
        ppy=ic['ppy'])
    return obs_data_std, K



if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()

    parser.add_argument('-o', '--overheadcam', default=False, action='store_true')
    parser.add_argument('--ip', default='100.96.135.68', help="robot ip address")
    parser.add_argument('--datafile', default='caldata.pkl', help="file to either load or save camera data")
    parser.add_argument('--overwrite', default=False, action='store_true', help="overwrite existing datafile, if it exists")
    parser.add_argument('--target-marker-id', default=9, type=int, help="ID of the ARTag marker in the image")
    parser.add_argument('--calibration-file', default='calibration.pkl', help="file to save final calibration data")
    parser.add_argument('--imagedir', default=None, help="folder to save debug images")

    args=parser.parse_args()
    print(f"Config: {args}")

    if os.path.exists(args.datafile) and not args.overwrite:
        print(f"Warning: datafile {args.datafile} already exists. Loading data instead of collecting data...")
        data = pickle.load(open(args.datafile, 'rb'))
    else:
        if os.path.exists(args.datafile):
            print(f"Warning: datafile {args.datafile} already exists. Overwriting...")
        print(f"Collecting data and saving to {args.datafile}...")

        data = []
        img_gen=realsense_images()
        pose_gen=sample_poses(overhead_cameras=args.overheadcam)
        for i, (pos,ori) in enumerate(robot_poses(args.ip, pose_gen)):
            imgs, intrinsics=next(img_gen)

            if args.imagedir is not None:
                os.makedirs(args.imagedir, exist_ok=True)
                for j, img in enumerate(imgs):
                    img_path=f'{args.imagedir}/capture_{i}_camera_{j}.jpg'
                    cv2.imwrite(img_path, img)
                    print(f'save debug images to {img_path}')

            data.append({
                'pos': pos,
                'ori': ori,
                'imgs': imgs,
                'intrinsics': intrinsics
            })

        with open(args.datafile, 'wb') as f:
            pickle.dump(data, f)

    print(f"Done. Data has {len(data)} poses.")

    corner_data = detect_corners(data, target_idx=args.target_marker_id)

    num_of_camera=len(corner_data[0]['intrinsics'])
    params=[]
    for i in range(num_of_camera):
        print(f'Solve camera {i} pose')
        obs_data_std, K = extract_obs_data_std(corner_data, i)
        print('number of images with keypoint', len(obs_data_std))
        param=torch.zeros(9, dtype=torch.float64, requires_grad=True)
        L = lambda param: mean_loss(obs_data_std, param, K)
        param_star=find_parameter(param, obs_data_std, K)

        print('found param loss (mean pixel err)', L(param_star).item())
        params.append(param_star)
    
    with torch.no_grad():
        param_list = []
        for i, param in enumerate(params):
            camera_base_ori = rotmat(param[:3])
            result = {
                "camera_base_ori": camera_base_ori.cpu().numpy().tolist(),
                "camera_base_pos": param[3:6].cpu().numpy().tolist(),
                "p_marker_ee": param[6:9].cpu().numpy().tolist(),
            }
            param_list.append(result)
            print(f"Camera {i} calibration: {result}")
        
        with open(args.calibration_file, 'wb') as f:
            print(f"Saving calibrated parameters to {args.calibration_file}")
            pickle.dump(param_list, f)
