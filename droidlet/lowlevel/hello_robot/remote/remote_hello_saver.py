import os
import time
import shutil
import json
import cv2
import numpy as np
import Pyro4

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION=2

def safe_call(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Pyro4.errors.ConnectionClosedError as e:
        msg = "{} - {}".format(f._RemoteMethod__name, e)
        raise ErrorWithResponse(msg)
    except Exception as e:
        print("Pyro traceback:")
        print("".join(Pyro4.util.getPyroTraceback()))
        raise e

@Pyro4.expose
class LabelPropSaver:
    def __init__(self, root, bot, cam):
        self.bot = bot
        self.cam = cam
        
        self.save_folder = root
        self.save_frequency = 1 # save every 10 frames
        self.skip_frame_count = 0 # internal counter
        self.dbg_str = "None"
        self.save_id = 0
        self._stop = False

    def return_paths(self, id_):
        id_ = str(id_)
        img_folder = os.path.join(self.save_folder, id_, "rgb")
        img_folder_dbg = os.path.join(self.save_folder, id_, "rgb_dbg")
        depth_folder = os.path.join(self.save_folder, id_, "depth")
        data_file = os.path.join(self.save_folder, id_, "data.json")
        return img_folder, img_folder_dbg, depth_folder, data_file

    def create_dirs(self, id_):

        img_folder, img_folder_dbg, depth_folder, data_file = self.return_paths(id_)

        for x in [img_folder, img_folder_dbg, depth_folder]:
            os.makedirs(x, exist_ok=True)

    def stop(self):
        self._stop = True

    def save_batch(self, seconds):
        print("Logging data for {} seconds".format(seconds), end='', flush=True)
        self._stop = False
        pose_dict = {}
        self.save_id += 1
        self.create_dirs(self.save_id)
        start_time = time.time()
        frame_count = 0
        end_time = seconds
        while time.time() - start_time <= seconds :
            rgb, depth = safe_call(self.cam.get_rgb_depth)
            base_pos = safe_call(self.bot.get_base_state)
            cam_pan = safe_call(self.bot.get_pan)
            cam_tilt = safe_call(self.bot.get_tilt)
            cam_transform = safe_call(self.bot.get_camera_transform)

            name = "{}".format(frame_count)
            self.save(self.save_id, name, rgb, depth, base_pos, cam_pan, cam_tilt, cam_transform, pose_dict)
            frame_count += 1
            print('.', end='', flush=True)
            if self._stop:
                end_time = time.time() - start_time
                print('pre-emptively stopped after {} seconds', round(end_time, 1))
                break
        print(' {} frames at {} fps'.format(frame_count, round(float(frame_count) / end_time, 1)))

    def ready(self):
        return True
            

    def save(self, id_, name, rgb, depth, pos, cam_pan, cam_tilt, cam_transform, pose_dict):
        img_folder, img_folder_dbg, depth_folder, data_file = self.return_paths(id_)

        self.skip_frame_count += 1
        if self.skip_frame_count % self.save_frequency == 0:
            # store the images and depth
            cv2.imwrite(
                img_folder + "/{}.jpg".format(name),
                rgb,
            )

            cv2.putText(rgb, self.dbg_str, (40,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

            cv2.imwrite(
                img_folder_dbg + "/{}.jpg".format(name),
                rgb,
            )

            # convert depth to milimetres
            depth *= 1e3

            # saturate maximum depth to 65,535mm or 65.53cm
            max_depth = np.power(2, 16) - 1
            depth[depth > max_depth] = max_depth
            
            depth = depth.astype(np.uint16)
            np.save(depth_folder + "/{}.npy".format(name), depth)

            # store pos
            if pos is not None:
                pose_dict[name] = {
                    'base_xyt': pos,
                    'cam_pan_tilt': [cam_pan, cam_tilt],
                    'cam_transform': cam_transform.tolist(),
                    }
            
            with open(data_file, "w") as fp:
                json.dump(pose_dict, fp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 192.168.0.0",
        type=str,
        default="0.0.0.0",
    )

    args = parser.parse_args()

    np.random.seed(123)

    with Pyro4.Daemon(args.ip) as daemon:
        bot = Pyro4.Proxy("PYRONAME:hello_robot@" + args.ip)
        cam = Pyro4.Proxy("PYRONAME:hello_realsense@" + args.ip)
        data_logger = LabelPropSaver('hello_data_log_' + str(time.time()), bot, cam)
        data_logger_uri = daemon.register(data_logger)
        with Pyro4.locateNS() as ns:
            ns.register("hello_data_logger", data_logger_uri)

        print("Server is started...")
        daemon.requestLoop()
