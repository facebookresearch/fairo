from typing import Iterator, List
import logging
from concurrent import futures
import functools

import numpy as np
import open3d as o3d

import grpc
from polygrasp import polygrasp_pb2
from polygrasp import polygrasp_pb2_grpc
import pyrealsense2


log = logging.getLogger(__name__)


class PointCloudServer(polygrasp_pb2_grpc.PointCloudServer):
    def GetPointcloud(self, request_iterator: Iterator[polygrasp_pb2.Image], context) -> Iterator[polygrasp_pb2.PointCloud]:
        raise NotImplementedError
    
    def SegmentPointcloud(self, request_iterator: Iterator[polygrasp_pb2.PointCloud], context) -> Iterator[polygrasp_pb2.ObjectMask]:
        raise NotImplementedError

class PointCloudClient:
    def __init__(self, camera_intrinsics: List[pyrealsense2.pyrealsense2.intrinsics], camera_extrinsics: np.ndarray):
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
        for rgbd, intrinsic, transform in zip(rgbds, self.o3_intrinsics, self.extrinsic_transforms):
            # The specific casting here seems to be very important, even though
            # rgbd should already be in np.uint16 type...
            img = rgbd[:, :, :3].astype(np.uint8)
            depth = rgbd[:, :, 3].astype(np.uint16)

            o3d_img = o3d.cuda.pybind.geometry.Image(img)
            o3d_depth = o3d.cuda.pybind.geometry.Image(depth)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_img,o3d_depth,convert_rgb_to_intensity=False,)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            pcd.transform(transform)
            scene_pcd += pcd

        # o3d.visualization.draw_geometries([scene_pcd])
        return scene_pcd

    def segment_pcd(self):
        raise NotImplementedError

def serve(port=50054, max_workers=10, *args, **kwargs):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    polygrasp_pb2_grpc.add_PointCloudServerServicer_to_server(PointCloudServer(*args, **kwargs), server)
    server.add_insecure_port(f"[::]:{port}")
    log.info(f"=== Starting server... ===")
    server.start()
    log.info(f"=== Done. Server running at port {port}. ===")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
