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
    ):
        assert len(camera_intrinsics) == len(camera_extrinsics)
        self.n_cams = len(camera_intrinsics)

        self.width = camera_intrinsics[0].width
        self.height = camera_intrinsics[0].height

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

    def get_pcd_i(self, rgbd: np.ndarray, cam_i: int, mask: np.ndarray = None):
        if mask is None:
            mask = np.ones([self.height, self.width])

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

    def get_pcd(self, rgbds: np.ndarray, masks: np.ndarray = None) -> o3d.geometry.PointCloud:
        if masks is None:
            masks = np.ones([self.n_cams, self.height, self.width])
        pcds = [self.get_pcd_i(rgbds[i], i, masks[i]) for i in range(len(rgbds))]
        result = pcds[0]
        for pcd in pcds[1:]:
            result += pcd
        return result


class SegmentationPointCloudClient(PointCloudClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = a0.RpcClient(topic_key)

    def segment_img(self, rgbd, min_mask_size=2500):
        state = []

        def onreply(pkt):
            state.append(pkt.payload)

        bits = serdes.rgbd_to_capnp(rgbd).to_bytes()
        self.client.send(bits, onreply)

        while not state:
            pass

        obj_masked_rgbds = []
        obj_masks = []

        labels = serdes.capnp_to_rgbd(state.pop())
        num_objs = int(labels.max())
        for obj_i in range(1, num_objs + 1):
            obj_mask = labels == obj_i

            obj_mask_size = obj_mask.sum()
            if obj_mask_size < min_mask_size:
                continue
            obj_masked_rgbd = rgbd * obj_mask[:, :, None]
            obj_masked_rgbds.append(obj_masked_rgbd)
            obj_masks.append(obj_mask)

        return obj_masked_rgbds, obj_masks


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
