# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import threading
from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np

import iphone_reader
from iphone_reader import Record3dReader, R3dFrame


class FakeStream:
    """Fake Record3D Stream that acts as an connected iPhone"""

    def __init__(self):
        self.on_new_frame = None
        self.on_stream_stopped = None

    @staticmethod
    def get_connected_devices():
        DeviceInfo = namedtuple("DeviceInfo", "product_id, udid")
        return [DeviceInfo(0, 0)]

    def connect(self, dev):
        pass

    def get_intrinsic_mat(self):
        return np.eye(3)

    def get_device_type(self):
        return 0

    def generate_frame(self):
        self.on_new_frame()

    def get_camera_pose(self):
        Pose = namedtuple("Pose", "tx, ty, tz, qx, qy, qz, qw")
        return Pose(0, 0, 0, 0, 0, 0, 1)

    def get_rgb_frame(self):
        return np.zeros([10, 10, 3])

    def get_depth_frame(self):
        return np.zeros([10, 10])


@pytest.fixture(params=[True, False])
def is_img_enabled(request):
    return request.param


@pytest.fixture
def reader(is_img_enabled, monkeypatch):
    monkeypatch.setattr(iphone_reader.api, "Record3DStream", FakeStream)
    return Record3dReader(retrieve_imgs=is_img_enabled)


def test_polling(reader, is_img_enabled):
    # Start reader without callback
    reader.start()

    # Wait for frame in separate thread
    def get_frame(frame_cache):
        frame = reader.wait_for_frame(timeout=1)
        frame_cache.append(frame)

    frames_out = []
    thr = threading.Thread(target=get_frame, args=(frames_out,))
    thr.start()

    # Generate frame and wait for reader to process frame
    reader._session.generate_frame()
    thr.join(timeout=1)

    # Check output frame
    assert len(frames_out) == 1
    frame = frames_out[0]
    assert type(frame) is R3dFrame

    # Check if imgs are queried
    assert (frame.color_img is not None) == is_img_enabled
    assert (frame.depth_img is not None) == is_img_enabled


def test_callback(reader, is_img_enabled):
    # Run reader with fake callback
    fake_callback = MagicMock()
    reader.start(frame_callback=fake_callback)

    # Generate frame
    reader._session.generate_frame()

    # Check that frame generation causes the callback to be called
    assert fake_callback.called_once()
