import numpy as np
import sophus as sp
import torch

from oculus_reader import OculusReader
from torchcontrol.transform import Rotation as R

from .base import TeleopDeviceReader


class OculusQuestReader(TeleopDeviceReader):
    """Allows for teleoperation using an Oculus controller
    Using the right controller, fully press the grip button (middle finger) to engage teleoperation. Hold B to perform grasp.
    """

    def __init__(self, lpf_cutoff_hz, control_hz):
        self.reader = OculusReader()
        self.reader.run()

        # LPF filter
        self.vr_pose_filtered = None
        tmp = 2 * np.pi * lpf_cutoff_hz / control_hz
        self.lpf_alpha = tmp / (tmp + 1)

        print("Oculus Quest teleop reader instantiated.")

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
        vr_pose_curr = sp.SE3(sp.SO3.exp(r.as_rotvec()).matrix(), pose_matrix[:3, -1])

        # Filter transform
        if self.vr_pose_filtered is None:
            self.vr_pose_filtered = vr_pose_curr
        else:
            self.vr_pose_filtered = self._interpolate_pose(
                self.vr_pose_filtered, vr_pose_curr, self.lpf_alpha
            )
        pose = self.vr_pose_filtered

        return is_active, pose, grasp_state

    @staticmethod
    def _interpolate_pose(pose1, pose2, pct):
        pose_diff = pose1.inverse() * pose2
        return pose1 * sp.SE3.exp(pct * pose_diff.log())
