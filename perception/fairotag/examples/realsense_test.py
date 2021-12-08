import time

import numpy as np
import sophus as sp
import torch
import cv2
import pyrealsense2 as rs

import fairotag as frt


class RealSenseCamera:
    """ Wrapper that implements boilerplate code for RealSense cameras """

    def __init__(self):
        # Identify devices
        devices = rs.context().query_devices()
        self.cameras = [c for c in devices]

        # Start stream
        print(f"Connecting to RealSense cameras ({len(self.cameras)} found) ...")
        self.pipes = []
        self.profiles = []
        for i, camera in enumerate(self.cameras):
            pipe = rs.pipeline()
            config = rs.config()

            # config.enable_device(camera.get_info(rs.camera_info(1)))
            # config.enable_stream(rs.stream.fisheye, 848, 800, rs.format.y8, 30)
            profile = pipe.start()

            self.pipes.append(pipe)
            self.profiles.append(profile)

            print(f"Connected to camera {i+1}.")

        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(60):
            self._get_frames()

    def _get_frames(self):
        return [pipe.wait_for_frames() for pipe in self.pipes]

    def get_intrinsics(self):
        intrinsics_ls = []
        for profile in self.profiles:
            stream = profile.get_streams()[0]
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
        return len(self.cameras)

    def get_images(self):
        framesets = self._get_frames()
        imgs = []
        for frameset in framesets:
            frame = frameset.get_fisheye_frame()
            img = np.asanyarray(frame.get_data())
            imgs.append(img)

        return imgs


if __name__ == "__main__":
    camera = RealSenseCamera()
    imgs = camera.get_images()
    import ipdb

    ipdb.set_trace()
