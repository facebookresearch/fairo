import logging
import json
from types import SimpleNamespace

import numpy as np
import open3d as o3d

import a0
from polygrasp import serdes

log = logging.getLogger(__name__)
topic = "cams/rgbd"


class CameraSubscriber:
    def __init__(self, intrinsics_file, extrinsics_file):
        with open(intrinsics_file, "r") as f:
            intrinsics_json = json.load(f)
            self.intrinsics = [SimpleNamespace(**d) for d in intrinsics_json]

        with open(extrinsics_file, "r") as f:
            self.extrinsics = json.load(f)

        assert len(self.intrinsics) == len(self.extrinsics)
        self.n_cams = len(self.intrinsics)

        self.sub = a0.SubscriberSync(topic, a0.INIT_MOST_RECENT, a0.ITER_NEWEST)
        self.recent_rgbd = None

    def get_rgbd(self):
        if self.sub.can_read():
            self.recent_rgbd = serdes.bytes_to_np(self.sub.read().payload)
        return self.recent_rgbd


class PointCloudSubscriber(CameraSubscriber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = self.intrinsics[0].width
        self.height = self.intrinsics[0].height

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
            for intrinsic in self.intrinsics
        ]

        # Convert to numpy homogeneous transforms
        self.extrinsic_transforms = np.empty([self.n_cams, 4, 4])
        for i, calibration in enumerate(self.extrinsics):
            self.extrinsic_transforms[i] = np.eye(4)
            self.extrinsic_transforms[i, :3, :3] = calibration["camera_base_ori"]
            self.extrinsic_transforms[i, :3, 3] = calibration["camera_base_pos"]

    def get_pcd_i(self, rgbd: np.ndarray, cam_i: int, mask: np.ndarray = None):
        if mask is None:
            mask = np.ones_like(rgbd[:, :, 0])

        intrinsic = self.o3_intrinsics[cam_i]
        transform = self.extrinsic_transforms[cam_i]

        # The specific casting here seems to be very important, even though
        # rgbd should already be in np.uint16 type...
        img = (rgbd[:, :, :3] * mask[:, :, None]).astype(np.uint8)
        depth = (rgbd[:, :, 3] * mask).astype(np.uint16)

        o3d_img = o3d.cuda.pybind.geometry.Image(img)
        o3d_depth = o3d.cuda.pybind.geometry.Image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_img,
            o3d_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform(transform)

        return pcd

    def get_pcd(
        self, rgbds: np.ndarray, masks: np.ndarray = None
    ) -> o3d.geometry.PointCloud:
        if masks is None:
            masks = np.ones([self.n_cams, self.height, self.width])
        pcds = [self.get_pcd_i(rgbds[i], i, masks[i]) for i in range(len(rgbds))]
        result = pcds[0]
        for pcd in pcds[1:]:
            result += pcd
        return result


if __name__ == "__main__":
    import argparse
    import realsense_wrapper

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intrinsics",
        type=str,
        default="conf/intrinsics.json",
        help="JSON file to overwrite with current intrinsics.",
    )
    args = parser.parse_args()
    cameras = realsense_wrapper.RealsenseAPI()

    intrinsics = cameras.get_intrinsics()
    intrinsics_py = [
        dict(
            coeffs=x.coeffs,
            fx=x.fx,
            fy=x.fy,
            height=x.height,
            ppx=x.ppx,
            ppy=x.ppy,
            width=x.width,
        )
        for x in intrinsics
    ]
    with open(args.intrinsics, "w") as f:
        json.dump(intrinsics_py, f, indent=4)

    rgbd_pub = a0.Publisher(topic)

    log.info(f"Starting camera logger with {cameras.get_num_cameras()} cameras...")
    while True:
        img_bytes = serdes.np_to_bytes(cameras.get_rgbd())
        rgbd_pub.pub(img_bytes)
