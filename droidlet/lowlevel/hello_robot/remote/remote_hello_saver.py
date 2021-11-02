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
        self.img_folder = os.path.join(self.save_folder, "rgb")
        self.img_folder_dbg = os.path.join(self.save_folder, "rgb_dbg")
        self.depth_folder = os.path.join(self.save_folder, "depth")

        if os.path.exists(self.save_folder):
            shutil.rmtree(self.save_folder)

        for x in [self.save_folder, self.img_folder, self.img_folder_dbg, self.depth_folder]:
            os.makedirs(x, exist_ok=True)
        
        self.pose_dict = {}
        self.save_frequency = 1 # save every 10 frames
        self.skip_frame_count = 0 # internal counter
        self.dbg_str = "None"
        self.save_id = 0

    def save_batch(self, seconds):
        self.save_id += 1
        print("Logging data for {} seconds".format(seconds), end='', flush=True)
        start_time = time.time()
        frame_count = 0
        self.pose_dict = {}
        while time.time() - start_time <= seconds :
            rgb, depth = safe_call(self.cam.get_rgb_depth)
            base_pos = safe_call(self.bot.get_base_state)
            name = "{}_{}".format(self.save_id, frame_count)
            self.save(self.save_id, name, rgb, depth, base_pos)
            frame_count += 1
            print('.', end='', flush=True)
        print(' {} frames at {} fps'.format(frame_count, round(float(frame_count) / seconds, 1)))

    def ready(self):
        return True
            

    def save(self, id_, name, rgb, depth, pos):

        self.skip_frame_count += 1
        if self.skip_frame_count % self.save_frequency == 0:
            # store the images and depth
            cv2.imwrite(
                self.img_folder + "/{}.jpg".format(name),
                rgb,
            )

            cv2.putText(rgb, self.dbg_str, (40,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

            cv2.imwrite(
                self.img_folder_dbg + "/{}.jpg".format(name),
                rgb,
            )

            # convert depth to milimetres
            depth *= 1e3

            # saturate maximum depth to 65,535mm or 65.53cm
            max_depth = np.power(2, 16) - 1
            depth[depth > max_depth] = max_depth
            
            depth = depth.astype(np.uint16)
            np.save(self.depth_folder + "/{}.npy".format(name), depth)

            # store pos
            if pos is not None:
                self.pose_dict[name] = pos
            
            with open(os.path.join(self.save_folder, "{}_data.json".format(id_)), "w") as fp:
                json.dump(self.pose_dict, fp)


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
