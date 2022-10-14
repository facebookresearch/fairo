# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import threading
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from record3d import Record3DStream, IntrinsicMatrixCoeffs

log = logging.getLogger(__name__)


@dataclass
class CameraMetadata:
    intrinsic_mat: np.ndarray
    device_type: int  # TrueDepth = 0, LiDAR = 1


@dataclass
class R3dFrame:
    pose_mat: np.ndarray
    pose_pos: np.ndarray
    pose_quat: np.ndarray
    color_img: Optional[np.ndarray]
    depth_img: Optional[np.ndarray]


class iPhoneReader:
    def __init__(self, device_idx: int = 0, retrieve_imgs: bool = True):
        # Initialize
        self._retrieve_imgs = retrieve_imgs

        self._dev = None
        self._session = None

        self._frame_event = threading.Event()
        self._frame_callback = None
        self._latest_frame = None

        self._camera_coeffs = None
        self._dev_type = -1

        self._origin_pose = np.eye(4)
        self._pose_reset = True

        # Search for device
        log.info("Searching for devices")
        devs = Record3DStream.get_connected_devices()
        log.info("{} device(s) found".format(len(devs)))
        for dev in devs:
            log.info("\tID: {}\n\tUDID: {}\n".format(dev.product_id, dev.udid))

        if len(devs) <= device_idx:
            raise RuntimeError(
                "Cannot connect to device #{}, try different index.".format(device_idx)
            )

        self._dev = devs[device_idx]

        # Create session
        self._session = Record3DStream()
        self._session.on_new_frame = self._on_new_frame
        self._session.on_stream_stopped = self._on_stream_stopped

    def _on_stream_stopped(self):
        log.info("Stream stopped")

    def _on_new_frame(self):
        camera_pose = self._session.get_camera_pose()
        # NOTE: quat & pos - camera_pose.[qx|qy|qz|qw|tx|ty|tz])
        pose_raw = np.eye(4)
        pose_raw[:3, 3] = np.array([camera_pose.tx, camera_pose.ty, camera_pose.tz])
        pose_raw[:3, :3] = R.from_quat(
            [camera_pose.qx, camera_pose.qy, camera_pose.qz, camera_pose.qw]
        ).as_matrix()

        # Reset pose
        if self._pose_reset:
            self._origin_pose = pose_raw
            self._pose_reset = False

        # Get pose
        pose_curr = np.linalg.pinv(self._origin_pose) @ pose_raw
        pos_curr = pose_curr[:3, 3]
        quat_curr = R.from_matrix(pose_curr[:3, :3]).as_quat()

        # Get images
        img_color = None
        img_depth = None
        if self._retrieve_imgs:
            img_color = self._session.get_rgb_frame()
            img_depth = self._session.get_depth_frame()

        # Save frame
        frame = R3dFrame(pose_curr, pos_curr, quat_curr, img_color, img_depth)
        self._latest_frame = frame
        self._frame_event.set()

        # Callback
        if self._frame_callback is not None:
            self._frame_callback(frame)

    def recenter_pose(self):
        self._pose_reset = True

    def wait_for_frame(self, timeout=None) -> R3dFrame:
        self._frame_event.clear()
        self._frame_event.wait(timeout)
        return self._latest_frame

    @property
    def metadata(self):
        return CameraMetadata(
            intrinsic_mat=np.array(
                [
                    [self._camera_coeffs.fx, 0, self._camera_coeffs.tx],
                    [0, self._camera_coeffs.fy, self._camera_coeffs.ty],
                    [0, 0, 1],
                ]
            ),
            device_type=self._dev_type,
        )

    def start(self, frame_callback: Optional[Callable[[R3dFrame], None]] = None):
        self._frame_callback = frame_callback

        # Initiate connection and start capturing
        self._session.connect(self._dev)

        # Get metadata
        self._camera_coeffs = self._session.get_intrinsic_mat()
        self._dev_type = self._session.get_device_type()
