from typing import Iterator
import logging
from concurrent import futures

import numpy as np
import open3d

import grpc
from polygrasp import polygrasp_pb2
from polygrasp import polygrasp_pb2_grpc


log = logging.getLogger(__name__)


class GraspServer(polygrasp_pb2_grpc.GraspServer):
    def _get_grasps(self, pcd: open3d.geometry.PointCloud) -> np.ndarray:
        raise NotImplementedError

    def GetGrasps(self, request_iterator: Iterator[polygrasp_pb2.PointCloud], context) -> Iterator[polygrasp_pb2.GraspGroup]:
        raise NotImplementedError


class GraspClient:
    def get_grasps(pcd: open3d.geometry.PointCloud) -> np.ndarray:
        raise NotImplementedError

def serve(port=50053, max_workers=10, *args, **kwargs):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    polygrasp_pb2_grpc.add_GraspServerServicer_to_server(GraspServer(*args, **kwargs), server)
    server.add_insecure_port(f"[::]:{port}")
    log.info(f"=== Starting server... ===")
    server.start()
    log.info(f"=== Done. Server running at port {port}. ===")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
