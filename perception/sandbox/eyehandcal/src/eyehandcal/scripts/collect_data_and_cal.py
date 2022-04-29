#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentError
from math import pi
import time
import os 
import pickle
import json
import sys
from eyehandcal.calibrator import solveEyeHandCalibration

import numpy as np
import torch
import cv2

from torchcontrol.transform import Rotation as R
from polymetis import RobotInterface
from realsense_wrapper import RealsenseAPI

from eyehandcal.utils import detect_corners, rotmat, dist_in_hull, uncompress_image, proj_funcs


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
        rs.get_rgbd()
    count=0
    while True:
        imgs0 = rs.get_rgbd()[:, :, :, :3]
        for i in range(30):
            imgs1 = rs.get_rgbd()[:, :, :, :3]
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
    for i, (pos_sampled, ori_sampled) in enumerate(pose_generator):
        print( f"Moving to pose ({i}): pos={pos_sampled}, quat={ori_sampled.as_quat()}")

        state_log = robot.move_to_ee_pose(position=pos_sampled,orientation=ori_sampled.as_quat(),time_to_go = time_to_go)
        print(f"Length of state_log: {len(state_log)}")
        if len(state_log) != time_to_go * robot.hz:
            print(f"warning: log incorrect length. {len(state_log)} != {time_to_go * robot.hz}")
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
    robot.go_home()


def save_result(json_calibration_file, cal_results):
    with torch.no_grad():
        param_list = []
        for i, cal in enumerate(cal_results):
            result = cal._asdict().copy()
            del result['param'] #pytorch vector
            if cal.param is not None:
                if cal.proj_func == "world_marker_proj_hand_camera":
                    camera_ee_ori_rotvec = cal.param[:3]
                    camera_ee_ori = rotmat(camera_ee_ori_rotvec)
                    result.update({
                        "camera_ee_ori": camera_ee_ori.numpy().tolist(),
                        "camera_ee_ori_rotvec": camera_ee_ori_rotvec.numpy().tolist(),
                        "camera_ee_pos" : cal.param[3:6].numpy().tolist(),
                        "marker_base_pos": cal.param[6:9].numpy().tolist()
                    })
                elif cal.proj_func == "hand_marker_proj_world_camera":
                    camera_base_ori_rotvec = cal.param[:3]
                    camera_base_ori = rotmat(camera_base_ori_rotvec)
                    result.update({
                        "camera_base_ori": camera_base_ori.cpu().numpy().tolist(),
                        "camera_base_ori_rotvec": camera_base_ori_rotvec.cpu().numpy().tolist(),
                        "camera_base_pos": cal.param[3:6].cpu().numpy().tolist(),
                        "p_marker_ee": cal.param[6:9].cpu().numpy().tolist(),
                    })
                else:
                    raise ArgumentError("shouldn't reach here")

            param_list.append(result)
            print(f"Camera {i} calibration: {result}")
        
        with open(json_calibration_file, 'w') as f:
            print(f"Saving calibrated parameters to {json_calibration_file}")
            json.dump(param_list, f, indent=4)


def collect_data(polymetis_server_ip, polymetis_time_to_go, img_gen, pose_gen, debug_image_dir):
    data = []
    poses = robot_poses(polymetis_server_ip, pose_gen, polymetis_time_to_go)
    for i, (pos,ori) in enumerate(poses):
        imgs, intrinsics=next(img_gen)

        if debug_image_dir is not None:
            os.makedirs(debug_image_dir, exist_ok=True)
            for j, img in enumerate(imgs):
                img_path=f'{debug_image_dir}/capture_{i}_camera_{j}.jpg'
                cv2.imwrite(img_path, img)
                print(f'save debug images to {img_path}')

        data.append({
                'pos': pos,
                'ori': ori,
                'imgs': imgs,
                'intrinsics': intrinsics
            })
        
    return data


def create_pose_generator(points_file, num_points):
    points = json.load(open(points_file, 'r'))
    xyz_points = np.array(points["xyz"])
    orient_points = np.array(points["quat"])
    pose_gen = sample_poses_from_data(xyz_points, orient_points, num_points=num_points)
    return pose_gen


def main(argv):
    import argparse
    parser=argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int, help="random seed for initializing solution")
    parser.add_argument('--ip', default='100.96.135.68', help="robot ip address")
    parser.add_argument('--datafile', default='caldata.pkl', help="file to either load or save camera data")
    parser.add_argument('--overwrite', default=False, action='store_true', help="overwrite existing datafile, if it exists")
    parser.add_argument('--marker-id', default=9, type=int, help="ID of the ARTag marker in the image")
    parser.add_argument('--calibration-file', default='calibration.json', help="file to save final calibration data")
    parser.add_argument('--points-file', default='calibration_points.json', help="file to load convex hull to sample points from")
    parser.add_argument('--num-points', default=4, type=int, help="number of xyz points to sample from convex hull")
    parser.add_argument('--time-to-go', default=3, type=float, help="time_to_go in seconds for each movement")
    parser.add_argument('--imagedir', default=None, help="folder to save debug images")
    parser.add_argument('--pixel-tolerance', default=2.0, type=float, help="mean pixel error tolerance (stage 2)")
    parser.add_argument('--proj-func', choices=list(proj_funcs.keys()), default = list(proj_funcs.keys())[0])

    args=parser.parse_args(argv)
    print(f"Config: {args}")

 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.datafile) and not args.overwrite:
        print(f"Warning: datafile {args.datafile} already exists. Loading data instead of collecting data...")
        data = pickle.load(open(args.datafile, 'rb'))
        uncompress_image(data)
    else:
        if os.path.exists(args.datafile):
            print(f"Warning: datafile {args.datafile} already exists. Overwriting...")
        print(f"Collecting data and saving to {args.datafile}...")

        img_gen=realsense_images()
        pose_gen = create_pose_generator(args.points_file, args.num_points)
        data = collect_data(args.ip, args.time_to_go, img_gen, pose_gen, args.imagedir)

        with open(args.datafile, 'wb') as f:
            pickle.dump(data, f)

    print(f"Done. Data has {len(data)} poses.")

    corner_data = detect_corners(data, target_idx=args.marker_id)
    cal_results = solveEyeHandCalibration(corner_data, args.proj_func, args.pixel_tolerance)
    save_result(args.calibration_file, cal_results)




if __name__ == '__main__':
    main(sys.argv[1:])
