import time

import numpy as np
import pyrealsense2 as rs


class RealsenseAPI:
    """ Wrapper that implements boilerplate code for RealSense cameras """

    def __init__(self):
        # Identify devices
        self.device_ls = []
        for c in rs.context().query_devices():
            self.device_ls.append(c.get_info(rs.camera_info(1)))

        # Start stream
        print(f"Connecting to RealSense cameras ({len(self.device_ls)} found) ...")
        self.pipes = []
        self.profiles = []
        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()

            config.enable_device(device_id)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            self.pipes.append(pipe)
            self.profiles.append(pipe.start(config))

            print(f"Connected to camera {i+1} ({device_id}).")
            time.sleep(1)

        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(60):
            self._get_frames()

    def _get_frames(self):
        return [pipe.wait_for_frames() for pipe in self.pipes]

    def get_intrinsics(self):
        intrinsics_ls = []
        for profile in self.profiles:
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()

            camera_matrix = np.eye(3)
            camera_matrix[0, 0] = intrinsics.fx
            camera_matrix[1, 1] = intrinsics.fy
            camera_matrix[0, 2] = intrinsics.ppx
            camera_matrix[1, 2] = intrinsics.ppy

            dist_coeffs = np.array(intrinsics.coeffs)

            intrinsics_ls.append((camera_matrix, dist_coeffs))

        return intrinsics_ls

    def get_num_cameras(self):
        return len(self.device_ls)

    def get_images(self):
        framesets = self._get_frames()
        imgs = []
        for frameset in framesets:
            frame = frameset.get_color_frame()
            img = np.asanyarray(frame.get_data())
            imgs.append(img)

        return imgs
