import hydra
import numpy as np
import torch

from polygrasp.cam_pub_sub import PointCloudSubscriber


class MockedGraspingRobotInterface:
    def __init__(self):
        pass

    def gripper_open(self, *args, **kwargs):
        pass

    def gripper_close(self, *args, **kwargs):
        pass

    def go_home(self, *args, **kwargs):
        pass

    def move_until_success(self, *args, **kwargs):
        return []

    def grasp(self, *args, **kwargs):
        return [], True

    def get_ee_pose(self):
        return torch.zeros(3), None

    def select_grasp(self, grasp_group):
        return grasp_group, 0


class MockedCam(PointCloudSubscriber):
    def __init__(self, rgbd_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgbd_img = np.load(hydra.utils.to_absolute_path(rgbd_path))

    def get_rgbd(self):
        return self.rgbd_img
