from typing import Iterator
import logging
from concurrent import futures

import numpy as np
import open3d

import grpc
from polygrasp import polygrasp_pb2
from polygrasp import polygrasp_pb2_grpc


log = logging.getLogger(__name__)


class PointCloudServer(polygrasp_pb2_grpc.PointCloudServer):
    def GetPointcloud(self, request_iterator: Iterator[polygrasp_pb2.Image], context) -> Iterator[polygrasp_pb2.PointCloud]:
        raise NotImplementedError
    
    def SegmentPointcloud(self, request_iterator: Iterator[polygrasp_pb2.PointCloud], context) -> Iterator[polygrasp_pb2.ObjectMask]:
        raise NotImplementedError

class PointCloudClient:
    def __init__(self, camera_intrinsics: np.ndarray, camera_extrinsics: np.ndarray):
        pass

    def get_pcd(self) -> open3d.geometry.PointCloud:
        raise NotImplementedError

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
