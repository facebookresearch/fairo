from typing import List

import numpy as np
import open3d as o3d

import pyrealsense2


class PointCloudClient:
    def __init__(
        self,
        camera_intrinsics: List[pyrealsense2.pyrealsense2.intrinsics],
        camera_extrinsics: np.ndarray,
    ):
        assert len(camera_intrinsics) == len(camera_extrinsics)
        self.n_cams = len(camera_intrinsics)

        # Convert to open3d intrinsics
        self.o3_intrinsics = [
            o3d.camera.PinholeCameraIntrinsic(
                width=intrinsic.width,
                height=intrinsic.height,
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.ppx,
                cy=intrinsic.ppy,
            )
            for intrinsic in camera_intrinsics
        ]

        # Convert to numpy homogeneous transforms
        self.extrinsic_transforms = np.empty([self.n_cams, 4, 4])
        for i, calibration in enumerate(camera_extrinsics):
            self.extrinsic_transforms[i] = np.eye(4)
            self.extrinsic_transforms[i, :3, :3] = calibration["camera_base_ori"]
            self.extrinsic_transforms[i, :3, 3] = calibration["camera_base_pos"]

    def get_pcd(self, rgbds: np.ndarray) -> o3d.geometry.PointCloud:
        scene_pcd = o3d.geometry.PointCloud()
        for rgbd, intrinsic, transform in zip(
            rgbds, self.o3_intrinsics, self.extrinsic_transforms
        ):
            # The specific casting here seems to be very important, even though
            # rgbd should already be in np.uint16 type...
            img = rgbd[:, :, :3].astype(np.uint8)
            depth = rgbd[:, :, 3].astype(np.uint16)

            o3d_img = o3d.cuda.pybind.geometry.Image(img)
            o3d_depth = o3d.cuda.pybind.geometry.Image(depth)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_img,
                o3d_depth,
                convert_rgb_to_intensity=False,
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            pcd.transform(transform)
            scene_pcd += pcd

        return scene_pcd

    def segment_pcd(self):
        raise NotImplementedError
