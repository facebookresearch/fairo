import numpy as np
import pyrealsense2 as rs


class RealsenseAPI:
    """Wrapper that implements boilerplate code for RealSense cameras"""

    def __init__(self, height=480, width=640, fps=30, warm_start=60):
        self.height = height
        self.width = width
        self.fps = fps

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
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            self.pipes.append(pipe)
            self.profiles.append(pipe.start(config))

            print(f"Connected to camera {i+1} ({device_id}).")

        self.align = rs.align(rs.stream.color)
        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(warm_start):
            self._get_frames()
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        self.spatial.set_option(rs.option.holes_fill, 3)

    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_intrinsics(self):
        intrinsics_ls = []
        for profile in self.profiles:
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()

            intrinsics_ls.append(intrinsics)

        return intrinsics_ls

    def get_num_cameras(self):
        return len(self.device_ls)

    def get_rgbd(self):
        """Returns a numpy array of [n_cams, height, width, RGBD]"""
        num_cams = self.get_num_cameras()

        rgbd = np.empty([num_cams, self.height, self.width, 4], dtype=np.uint16)

        for i, frameset in enumerate(self._get_frames()):
            color_frame = frameset.get_color_frame()
            rgbd[i, :, :, :3] = np.asanyarray(color_frame.get_data())

            depth_frame = frameset.get_depth_frame()
            # breakpoint() # apply post-processing filters
            # print('I was in realsense wrapper!!!!!!!')
            
            # depth_frame = self.decimation.process(depth_frame)
            filtered_depth = self.spatial.process(depth_frame)
            
            rgbd[i, :, :, 3] = np.asanyarray(filtered_depth.get_data())
            # rgbd[i, :, :, 3] = np.asanyarray(depth_frame.get_data())

        return rgbd

    # def visualize_depth(self, )

if __name__ == "__main__":
    cams = RealsenseAPI()

    print(f"Num cameras: {cams.get_num_cameras()}")
    rgbd = cams.get_rgbd()
