
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ast import For
from logging import RootLogger
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

from polymetis import RobotInterface



if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="172.16.0.1",
    )
    points = open("poses.json")
    p = json.load(points)
    robot.go_home()

    def getEEpose():
        ee_pos, ee_quat = robot.get_ee_pose()
        print(f"Current ee position: {ee_pos}")
        print(f"Current ee orientation: {ee_quat}  (xyzw)")

    def getEEBpos():
        ee_pos, _ = robot.get_ee_pose()
        return ee_pos

    def getEEBQ():
        _, ee_quat = robot.get_ee_pose()
        return ee_quat
    def getCPos():
        cam = open("calibrationEEcam.json")
        c = json.load(cam)
        return c['camera_ee_pos']
    def getCO():
        cam = open("calibrationEEcam.json")
        c = json.load(cam)
        i = []
        for x in c['camera_ee_ori_rotvec']:
            i = np.asarray(x)
        return i

    def liveloc():
        r_ee_b = getEEBQ()
        t_ee_b = getEEBpos()
        r_c_ee = getCO()
        t_c_ee = getCPos()
        rot = R.from_rotvec(r_c_ee)
        rot_matrix = rot.as_matrix()
        cam_pose = rot_matrix.dot(t_ee_b) + t_c_ee
        return cam_pose
        

    liveloc()
    '''
    for i in p['xyz']:
        ee_pos_desired = torch.Tensor(i)
        for j in p['quat']:
            ee_quat_desired = torch.Tensor(j)
            state_log = robot.move_to_ee_pose(
                position=ee_pos_desired, orientation=ee_quat_desired, time_to_go=10
            )
            getEEpose()
            print()

    '''