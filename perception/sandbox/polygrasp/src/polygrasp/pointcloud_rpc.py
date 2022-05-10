import logging
from typing import List
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import a0
from polygrasp import serdes

log = logging.getLogger(__name__)


topic_key = "pcd_server"


class PointCloudClient:
    def __init__(
        self,
        camera_intrinsics: List[SimpleNamespace],
        camera_extrinsics: np.ndarray,
        masks: np.ndarray = None,
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

        if masks is None:
            intrinsic = camera_intrinsics[0]
            self.masks = np.ones([self.n_cams, intrinsic.width, intrinsic.height])
        else:
            self.masks = masks

    def get_pcd(self, rgbds: np.ndarray) -> o3d.geometry.PointCloud:
        pcds = []
        for rgbd, intrinsic, transform, mask in zip(
            rgbds, self.o3_intrinsics, self.extrinsic_transforms, self.masks
        ):
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
            pcds.append(pcd)

        return pcds

    def get_pcd_i(self, rgbd, i):
        intrinsic = self.o3_intrinsics[i]
        transform = self.extrinsic_transforms[i]
        mask = self.masks[i]

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


class PointCloudServer:
    def _get_segmentations(self, rgbd):
        raise NotImplementedError

    def start(self):
        def onrequest(req):
            log.info("Got request; computing segmentations...")

            payload = req.pkt.payload
            rgbd = serdes.capnp_to_rgbd(payload)
            result = self._get_segmentations(rgbd)

            log.info("Done. Replying with serialized segmentations...")
            req.reply(serdes.rgbd_to_capnp(result).to_bytes())

        server = a0.RpcServer(topic_key, onrequest, None)
        log.info("Starting server...")
        while True:
            pass


class SegmentationPointCloudClient(PointCloudClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = a0.RpcClient(topic_key)

    def segment_img(self, rgbd):
        state = []

        def onreply(pkt):
            state.append(pkt.payload)

        bits = serdes.rgbd_to_capnp(rgbd).to_bytes()
        self.client.send(bits, onreply)

        while not state:
            pass

        return serdes.capnp_to_rgbd(state.pop())
