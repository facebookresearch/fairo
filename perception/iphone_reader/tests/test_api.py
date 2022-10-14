import pytest
import threading
from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np

import iphone_reader
from iphone_reader import iPhoneReader, R3dFrame


class FakeStream:
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
    return iPhoneReader(retrieve_imgs=is_img_enabled)


def test_polling(reader, is_img_enabled):
    reader.start()

    def get_frame(frame_cache):
        frame = reader.wait_for_frame(timeout=1)
        frame_cache.append(frame)

    frames_out = []
    thr = threading.Thread(target=get_frame, args=(frames_out,))
    thr.start()

    reader._session.generate_frame()
    thr.join(timeout=1)

    assert len(frames_out) == 1
    assert type(frames_out[0]) is R3dFrame


def test_callback(reader, is_img_enabled):
    fake_callback = MagicMock()
    reader.start(frame_callback=fake_callback)
    reader._session.generate_frame()

    assert fake_callback.called_once()
