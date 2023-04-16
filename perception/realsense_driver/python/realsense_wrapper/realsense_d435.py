import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict


class RealsenseAPI:
    """Wrapper that implements boilerplate code for RealSense cameras"""

    def __init__(self, height=480, width=640, fps=30, warm_start=60, depth_preset: str="Default"):
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
        self.profiles = OrderedDict()
        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()

            config.enable_device(device_id)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            self.pipes.append(pipe)
            self.profiles[device_id]=pipe.start(config)

            if depth_preset:
                depth_sensor=self.profiles[device_id].get_device().first_depth_sensor()
                RealsenseAPI._set_visual_preset(depth_sensor, depth_preset)
            print(f"Connected to camera {i+1} ({device_id}).")

        self.align = rs.align(rs.stream.color)
        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(warm_start):
            self._get_frames()


    @staticmethod
    def _set_visual_preset(depth_sensor, preset_desc: str):
        preset_option = None
        options = []
        for j in range(int(depth_sensor.get_option_range(rs.option.visual_preset).max)):
            desc=depth_sensor.get_option_value_description(rs.option.visual_preset, j)
            options.append(desc)
            if preset_desc in desc:
                preset_option = j
                print(f'{preset_desc}  preset is: {preset_option}')
                break

        if preset_option is None:
            raise RuntimeWarning(f"{preset_desc} not available, please choose one from {options}")
        else:
            depth_sensor.set_option(rs.option.visual_preset, preset_option)




    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_intrinsics(self):
        intrinsics_ls = []
        for profile in self.profiles.values():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()

            intrinsics_ls.append(intrinsics)

        return intrinsics_ls

    def get_intrinsics_dict(self):
        intrinsics_ls = OrderedDict()
        for device_id, profile in self.profiles.items():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()
            param_dict = dict([(p, getattr(intrinsics, p)) for p in dir(intrinsics) if not p.startswith('__')])
            param_dict['model'] = param_dict['model'].name

            intrinsics_ls[device_id] = param_dict

        return intrinsics_ls
    
    def get_num_cameras(self):
        return len(self.device_ls)

    def get_rgbd(self):
        """Returns a numpy array of [n_cams, height, width, RGBD]"""
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        rgbd = np.empty([num_cams, self.height, self.width, 4], dtype=np.uint16)

        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            rgbd[i, :, :, :3] = np.asanyarray(color_frame.get_data())

            depth_frame = frameset.get_depth_frame()
            rgbd[i, :, :, 3] = np.asanyarray(depth_frame.get_data())

        return rgbd


if __name__ == "__main__":
    cams = RealsenseAPI()

    print(f"Num cameras: {cams.get_num_cameras()}")
    rgbd = cams.get_rgbd()
