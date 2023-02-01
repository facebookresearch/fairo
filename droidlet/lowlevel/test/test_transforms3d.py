import unittest
import os
from droidlet.test_utils import skipIfOfflineDecorator
import cv2
import open3d as o3d
import numpy as np
import json
from functools import partial
from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover


test_assets = [
    "pcd_test_1.jpg",
    "pcd_test_1.npy",
    "pcd_test_1.json",
    "pcd_test_intrinsic.json",
]
folder_prefix = os.path.join(os.path.dirname(__file__), "test_assets")
url_prefix = "test_assets/"

skipIfOffline = skipIfOfflineDecorator(test_assets, folder_prefix, url_prefix)


def load_intrinsic(name="pcd_test_intrinsic.json", prefix=folder_prefix):
    data_path = os.path.join(prefix, name)
    with open(data_path, "r") as fp:
        data = json.load(fp)
    intrinsic = np.asarray(data["cam_intrinsic"], dtype=np.float32)
    resolution = data["cam_resolution"]
    return intrinsic, resolution


def load_ground_truth_pcd(name="pcd_test_1", key=0, prefix=folder_prefix):
    rgb_path = os.path.join(prefix, name + ".jpg")
    depth_path = os.path.join(prefix, name + ".npy")
    data_path = os.path.join(prefix, name + ".json")
    rgb = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    with open(data_path, "r") as fp:
        pose_dict = json.load(fp)
    data = pose_dict[str(key)]
    base_xyt = data["base_xyt"]
    cam_transform = np.asarray(data["cam_transform"], dtype=np.float32)
    return rgb, depth, base_xyt, cam_transform


class TransformsTest(unittest.TestCase):
    @skipIfOffline
    def test_native_pcd_transform(self):
        intrinsic_mat, resolution = load_intrinsic()
        height, width = resolution
        uv_one_in_cam = HelloRobotMover.compute_uvone(intrinsic_mat, height, width)

        rgb, depth, base_xyt, cam_transform = load_ground_truth_pcd()
        rot = cam_transform[:3, :3]
        trans = cam_transform[:3, 3]

        rgb_depth = HelloRobotMover.compute_pcd(rgb, depth, rot, trans, base_xyt, uv_one_in_cam)
        points, colors = rgb_depth.ptcloud.reshape(-1, 3), rgb_depth.rgb.reshape(-1, 3)
        colors = colors / 255.0

        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(points)
        opcd.colors = o3d.utility.Vector3dVector(colors)

        return rgb_depth

    @skipIfOffline
    def test_open3d_pcd_transform(self):
        assert True


if __name__ == "__main__":
    unittest.main()
