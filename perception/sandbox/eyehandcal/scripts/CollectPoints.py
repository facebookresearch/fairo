
from ast import For
from logging import RootLogger
from os import posix_fadvise
import torch
import json
import numpy as np
import os
import sys
from realsense_wrapper import RealsenseAPI
import cv2

from polymetis import RobotInterface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0", help="robopen IP address")
    parser.add_argument("--num-poses", default="8", help="Number of XYZ and quat coordinates")
    args =parser.parse_args()
    num_pose = int(args.num_poses)
    print(f"config: {args.ip}")
    # Initialize robot interface
    robot = RobotInterface(
        ip_address=args.ip,
    )
    xyz = []
    count = 0
    rs = RealsenseAPI()
    while count < num_pose:
        xyz_pose = robot.get_ee_pose()[0].numpy().squeeze().tolist()
        result = input(f"New XYZ pose: {xyz_pose}. Press 's' to save, 'c' to skip, 'e' to exit: ")
        if result == 'c':
            continue
        elif result == 'e':
            break
        elif result == 's':
            xyz.append(xyz_pose)
            count += 1
            ## save img too
            imgs = rs.get_rgbd()
            for j, img in enumerate(imgs):
                rgb = img[:,:,:3]
                imgPath = f'/mnt/tmp_nfs_clientshare/jaydv/fairo/perception/sandbox/eyehandcal/scripts/debug/{count}_cam{j}.jpg'
                cv2.imwrite(imgPath, rgb[:,:,::-1])
                imgPath = f'/mnt/tmp_nfs_clientshare/jaydv/fairo/perception/sandbox/eyehandcal/scripts/debug/{count}_depth_cam{j}.jpg'
                dimg = img[:,:,3]
                cv2.imwrite(imgPath, dimg.astype(np.uint8))
                print(f"Image {count} saved to {imgPath}")
            print("XYZ pose saved")
        else:
            print ("Invalid selection, please try again")

    quat = []
    countQuat = 0
    while countQuat < num_pose:
        quat_pose = robot.get_ee_pose()[1].numpy().squeeze().tolist()
        result = input(f"New quat pose: {quat_pose}. Press 's' to save, 'c' to skip, 'e' to exit: ")
        if result == 'c':
            continue
        elif result == 'e':
            break
        elif result == 's':
            quat.append(quat_pose)
            countQuat += 1
            print("Quat pose saved")
        else:
            print ("Invalid selection, please try again")

    if os.path.exists("poses.json"):
        result = input("File with the same name found, enter 'r' to rename the file, 'd' to delete the file, 'e' to exit the program: ")
        if result == 'r':
            new_file = input("What would you like to name the file? enter without '.json': ")
            new_file = new_file + ".json"
            print("renaming existing pose.json to " + new_file)
            os.rename("pose.json", new_file)
        if result == 'd':
            print("Deleting existing pose.json")
            os.remove("poses.json")
        if result == 'e':
            sys.exit("User selected exit, XYZ and quat coordinates are not saved")
    print("saving files to poses.json...")
    with open("poses.json", "w") as f:
        json.dump({"xyz": xyz, "quat": quat}, f)

    print("Success.")


##################################################################################################################################
