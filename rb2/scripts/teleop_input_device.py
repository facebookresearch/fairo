# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sophus as sp
import torch

from torchcontrol.transform import Rotation as R

from oculus_reader import OculusReader


# Low pass filter cutoff frequency
LPF_CUTOFF_HZ = 15

# Helper function
def interpolate_pose(pose1, pose2, pct):
    pose_diff = pose1.inverse() * pose2
    return pose1 * sp.SE3.exp(pct * pose_diff.log())


class OculusTeleop:
    """Allows for teleoperation using an Oculus controller"""

    def __init__(self, query_hz, ip_address=None):
        self.reader = OculusReader(ip_address=ip_address)
        self.reader.run()

        # LPF filter
        self.vr_pose_filtered = None
        tmp = 2 * np.pi * LPF_CUTOFF_HZ / query_hz
        self.lpf_alpha = tmp / (tmp + 1)

    def get_state(self):
        # Get data from oculus reader
        transforms, buttons = self.reader.get_transformations_and_buttons()

        # Generate output
        if transforms:
            is_active = buttons["rightGrip"][0] > 0.9
            grasp_state = buttons["B"]
            pose_matrix = np.linalg.pinv(transforms["l"]) @ transforms["r"]
        else:
            is_active = False
            grasp_state = 0
            pose_matrix = np.eye(4)
            self.vr_pose_filtered = None

        # Create transform (hack to prevent unorthodox matrices)
        r = R.from_matrix(torch.Tensor(pose_matrix[:3, :3]))
        vr_pose_curr = sp.SE3(
            sp.SO3.exp(r.as_rotvec()).matrix(), pose_matrix[:3, -1]
        )

        # Filter transform
        if self.vr_pose_filtered is None:
            self.vr_pose_filtered = vr_pose_curr
        else:
            self.vr_pose_filtered = interpolate_pose(
                self.vr_pose_filtered, vr_pose_curr, self.lpf_alpha
            )
        pose = self.vr_pose_filtered

        return is_active, pose, grasp_state