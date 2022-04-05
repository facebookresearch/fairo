#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import pi
import time
import os 
import pickle
import json

import numpy as np
import torch
import cv2

from torchcontrol.transform import Rotation as R
from polymetis import RobotInterface
from realsense_wrapper import RealsenseAPI

from eyehandcal.utils import detect_corners, quat2rotvec, build_proj_matrix, mean_loss, find_parameter, rotmat, dist_in_hull


def realsense_images(max_pixel_diff=200):
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

def sample_poses_from_data(xyz_points, orient_points, num_points):
    points = dist_in_hull(points=xyz_points, n=num_points)
    for point in points:
        for orient in orient_points:
            pos = torch.Tensor(point)
            ori = R.from_quat(torch.Tensor(orient))
            yield pos, ori

def robot_poses(ip_address, pose_generator, time_to_go=3):
    # Initialize robot interface
    robot = RobotInterface(
        ip_address=ip_address,
        enforce_version=False
    )

    # Get reference state
    robot.go_home()
    robot.start_cartesian_impedance()
    for i, (pos_sampled, ori_sampled) in enumerate(pose_generator):
        while True:
            print( f"Moving to pose ({i}): pos={pos_sampled}, quat={ori_sampled.as_quat()}")

            state_log = robot.move_to_ee_pose(position=pos_sampled,orientation=ori_sampled.as_quat(),time_to_go = time_to_go)
            print(f"Length of state_log: {len(state_log)}")
            if len(state_log) != time_to_go * robot.hz:
                print(f"State log incorrect length. Trying again...")
            else:
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
                break
    robot.go_home()


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

    parser.add_argument('--ip', default='100.96.135.68', help="robot ip address")
    parser.add_argument('--datafile', default='caldata.pkl', help="file to either load or save camera data")
    parser.add_argument('--overwrite', default=False, action='store_true', help="overwrite existing datafile, if it exists")
    parser.add_argument('--marker-id', default=9, type=int, help="ID of the ARTag marker in the image")
    parser.add_argument('--calibration-file', default='calibration.json', help="file to save final calibration data")
    parser.add_argument('--points-file', default='calibration_points.json', help="file to load convex hull to sample points from")
    parser.add_argument('--num-points', default=20, type=int, help="number of points to sample from convex hull")
    parser.add_argument('--time-to-go', default=3, type=float, help="time_to_go in seconds for each movement")
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
        points = json.load(open(args.points_file, 'r'))
        xyz_points = np.array(points["xyz"])
        orient_points = np.array(points["quat"])
        pose_gen = sample_poses_from_data(xyz_points, orient_points, num_points=args.num_points)
        poses = robot_poses(args.ip, pose_gen, args.time_to_go)
        for i, (pos,ori) in enumerate(poses):
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

    corner_data = detect_corners(data, target_idx=args.marker_id)

    num_of_camera=len(corner_data[0]['intrinsics'])
    params=[]
    for i in range(num_of_camera):
        print(f'Solve camera {i} pose')
        obs_data_std, K = extract_obs_data_std(corner_data, i)
        print('number of images with keypoint', len(obs_data_std))
        param=torch.zeros(9, dtype=torch.float64, requires_grad=True)
        L = lambda param: mean_loss(obs_data_std, param, K)
        try:
            param_star=find_parameter(param, obs_data_std, K)
        except:
            continue

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
        
        with open(args.calibration_file, 'w') as f:
            print(f"Saving calibrated parameters to {args.calibration_file}")
            json.dump(param_list, f)
