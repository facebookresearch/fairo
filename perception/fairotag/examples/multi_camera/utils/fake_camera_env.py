import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R


class FakeCameraEnv:
    def __init__(self, n_cams, ws_height, deviation=0.0, noise=0.0):
        self.n_cams = n_cams
        self.ws_height = ws_height
        self.noise = noise

        # Initialize cameras
        self.cam_poses = []
        for i in range(n_cams):
            theta = 2 * np.pi * (i / n_cams)
            pos = ws_height * np.array([np.cos(theta), np.sin(theta), 1.0])
            ori = R.from_rotvec([0.0, 0.0, theta]) * R.from_rotvec([0.0, -3 * np.pi / 4, 0.0])

            pos = pos + deviation * np.random.randn(3)
            ori = ori * R.from_rotvec(deviation * np.random.randn(3))

            self.cam_poses.append(sp.SE3(ori.as_matrix(), pos))

    def sample_marker(self, pose=None, return_pose=False, misdetect_prob=0.0):
        if pose is None:
            pose = self.get_random_pose(0.5 * self.ws_height)

        cam_transforms = []
        for cam_pose in self.cam_poses:
            rel_pose = cam_pose.inverse() * pose

            pos_noise = self.noise * np.random.randn(3)
            ori_noise = R.from_rotvec(self.noise * np.random.randn(3))
            est_pose = rel_pose * sp.SE3(ori_noise.as_matrix(), pos_noise)

            if np.random.uniform(0, 1) < misdetect_prob:
                est_pose = None

            cam_transforms.append(est_pose)

        if return_pose:
            return pose, cam_transforms
        else:
            return cam_transforms

    def get_cam_poses(self):
        return self.cam_poses

    def get_random_pose(self, workspace_width=1):
        pos = workspace_width * np.random.uniform(-1, 1, 3)
        ori = R.random()
        return sp.SE3(ori.as_matrix(), pos)
