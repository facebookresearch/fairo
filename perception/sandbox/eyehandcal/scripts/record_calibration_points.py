#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from matplotlib import pyplot as plt
import numpy as np

import fairotag as frt
from realsense_wrapper import RealsenseAPI

from polymetis import RobotInterface


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="100.96.135.68", help="robot ip address")
    parser.add_argument("--marker-id", type=int, default=9, help="ARTag marker id")
    parser.add_argument("--marker-length", type=float, default=0.05, help="ARTag length in meters")
    args = parser.parse_args()
    print(f"Config: {args}")

    print(f"Connecting to camera...")
    rs = RealsenseAPI()
    num_cams = rs.get_num_cameras()

    print(f"Registering marker ID {args.marker_id} with length {args.marker_length} using FairoTag...")
    fairotag_cams = [frt.CameraModule() for _ in range(num_cams)]
    for intrinsics, fairotag_cam in zip(rs.get_intrinsics(), fairotag_cams):
        fairotag_cam.register_marker_size(args.marker_id, args.marker_length)
        fairotag_cam.set_intrinsics(fx=intrinsics.fx, fy=intrinsics.fy, ppx=intrinsics.ppx, ppy=intrinsics.ppy, coeffs=np.array(intrinsics.coeffs))

    def view_imgs():
        fig, axes = plt.subplots(num_cams, 2, squeeze=False)
        fig.suptitle(f"Camera images (press `q` to exit)")
        rgbds  = rs.get_rgbd()
        for i, (fairotag_cam, rgbd, ax) in enumerate(zip(fairotag_cams, rgbds, axes)):
            img = rgbd[:, :, :3].astype(np.uint8)
            depth = rgbd[:, :, 3]
            markers = fairotag_cam.detect_markers(img)
            img_render = fairotag_cam.render_markers(img, markers=markers)
            ax[0].imshow(img_render)
            ax[0].set_title(f"Camera {i} image (with FairoTag)")
            ax[1].imshow(depth)
            ax[1].set_title(f"Camera {i} depth")
        plt.show()

    robot = RobotInterface(ip_address=args.ip, enforce_version=False)

    print("In readonly mode, move the robot to a new pose.")
    xyz_poses = []
    while True:
        xyz_pose = robot.get_ee_pose()[0].numpy().squeeze().tolist()
        view_imgs()
        result = input(f"New xyz pose: {xyz_pose}. Press enter to save, `c` to skip, or `exit` to exit.")
        if result == "c":
            continue
        elif result == "exit":
            break
        else:
            xyz_poses.append(xyz_pose)

    quat_poses = []
    while True:
        quat_pose = robot.get_ee_pose()[1].numpy().squeeze().tolist()
        view_imgs()
        result = input(f"New orientation: {quat_pose}. Press enter to save, `c` to skip, or `exit` to exit.")
        if result == "c":
            continue
        elif result == "exit":
            break
        quat_poses.append(quat_pose)

    print(f"Saving poses to calibration_points.json...")
    with open("calibration_points.json", "w") as f:
        json.dump({"xyz": xyz_poses, "quat": quat_poses}, f)
    print("Success.")
