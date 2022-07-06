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
from collections import namedtuple
from eyehandcal.utils import uncompress_image

import numpy as np
import torch
import cv2

from torchcontrol.transform import Rotation as R
from polymetis import RobotInterface
from realsense_wrapper import RealsenseAPI

from eyehandcal.utils import detect_corners, quat2rotvec, build_proj_matrix, mean_loss, find_parameter, rotmat, dist_in_hull, \
    hand_marker_proj_world_camera, world_marker_proj_hand_camera


def realsense_images(max_pixel_diff=200):
    rs = RealsenseAPI()
    num_cameras = rs.get_num_cameras()
    assert num_cameras > 0, "no camera found"

    intrinsics = rs.get_intrinsics_dict()


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

    ic = list(data[0]['intrinsics'].values())[camera_index]
    K=build_proj_matrix(
        fx=ic['fx'],
        fy=ic['fy'],
        ppx=ic['ppx'],
        ppy=ic['ppy'])
    return obs_data_std, K



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
    parser.add_argument('--num-points', default=20, type=int, help="number of points to sample from convex hull")
    parser.add_argument('--time-to-go', default=3, type=float, help="time_to_go in seconds for each movement")
    parser.add_argument('--imagedir', default=None, help="folder to save debug images")
    parser.add_argument('--pixel-tolerance', default=2.0, type=float, help="mean pixel error tolerance (stage 2)")
    proj_funcs = {'hand_marker_proj_world_camera' :hand_marker_proj_world_camera, 
                  'world_marker_proj_hand_camera' :world_marker_proj_hand_camera,
                  'wrist_camera': world_marker_proj_hand_camera,
                  'world_camera': hand_marker_proj_world_camera}
    parser.add_argument('--proj-func', choices=list(proj_funcs.keys()), default = list(proj_funcs.keys())[0])

    args=parser.parse_args(argv)
    print(f"Config: {args}")

    proj_func = proj_funcs[args.proj_func]

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

    intrinsics = corner_data[0]['intrinsics']
    num_of_camera=len(intrinsics)
    CalibrationResult = namedtuple('CalibrationResult',
                                    field_names=['num_marker_seen', 'stage2_retry', 'pixel_error', 'param', 'proj_func'],
                                    defaults=[None]*5)
    cal_results = []
    for i in range(num_of_camera):
        print(f'Solve camera {i}/{num_of_camera} pose')
        obs_data_std, K = extract_obs_data_std(corner_data, i)
        print('number of images with keypoint', len(obs_data_std))
        if len(obs_data_std) < 3:
            print('too few keypoint found for this camera, skip this camera')
            cal_results.append(CalibrationResult(num_marker_seen=len(obs_data_std)))
            continue

        # stage 1 - assuming marker is attach to EE origin, solve camera pose first
        if args.proj_func == "hand_marker_proj_world_camera":
            p3d = torch.stack([p[1] for p in obs_data_std]).detach().numpy()
        elif args.proj_func == "world_marker_proj_hand_camera":
            p3d = torch.stack([rotmat(-p[2]).matmul(-p[1]) for p in obs_data_std]).detach().numpy()

        p2d = torch.stack([p[0] for p in obs_data_std]).detach().numpy()
        retval, rvec, tvec = cv2.solvePnP(p3d, p2d, K.numpy(), distCoeffs=None, flags=cv2.SOLVEPNP_SQPNP)
        rvec_cam = torch.tensor(-rvec.reshape(-1))
        tvec_cam = -rotmat(rvec_cam).matmul(torch.tensor(tvec.reshape(-1)))
        pixel_error = mean_loss(obs_data_std, torch.cat([rvec_cam, tvec_cam, torch.zeros(3)]), K, proj_func).item()
        print('stage 1 mean pixel error', pixel_error)

        # stage 2 - allow marker to move, joint optimize camera pose and marker
        max_stage2_retry = 10
        stage2_retry_count = 0
        
        while True :

            stage2_retry_count += 1
            if stage2_retry_count > max_stage2_retry:
                cal_results.append(CalibrationResult(num_marker_seen=len(obs_data_std),
                                                     stage2_retry=stage2_retry_count,
                                                     param=param_star,
                                                     pixel_error=pixel_error,
                                                     proj_func=args.proj_func))
                print('Maximum stage2 retry execeeded, bailing out')
                break

            marker_max_displacement = 0.1 #meter
            param=torch.cat([rvec_cam, tvec_cam, torch.randn(3)*marker_max_displacement]).clone().detach()
            param.requires_grad=True
            L = lambda param: mean_loss(obs_data_std, param, K, proj_func)
            try:
                param_star=find_parameter(param, L)
            except Exception as e:
                print(e)
                continue

            pixel_error = L(param_star).item()
            print('stage 2 mean pixel error', pixel_error)
            if pixel_error > args.pixel_tolerance:
                print(f"Try again {stage2_retry_count}/{max_stage2_retry} because of poor solution {pixel_error} > {args.pixel_tolerance}")
            else:
                print(f"Good solution {pixel_error} <= {args.pixel_tolerance}")
                cal_results.append(CalibrationResult(num_marker_seen=len(obs_data_std),
                                                     stage2_retry=stage2_retry_count,
                                                     param=param_star,
                                                     pixel_error=pixel_error,
                                                     proj_func=args.proj_func))
                break

    with torch.no_grad():
        param_list = []
        for i, cal in enumerate(cal_results):
            result = cal._asdict().copy()
            result.update({
                "camera_serial_number": list(intrinsics.keys())[i],
                "intrinsics":list(intrinsics.values())[i]
            })
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
                elif args.proj_func == "hand_marker_proj_world_camera":
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
        
        with open(args.calibration_file, 'w') as f:
            print(f"Saving calibrated parameters to {args.calibration_file}")
            json.dump(param_list, f, indent=4)

if __name__ == '__main__':
    main(sys.argv[1:])
