import logging
import json
from types import SimpleNamespace

import a0
import realsense_wrapper
from polygrasp import serdes

log = logging.getLogger(__name__)
topic = "cams/rgbd"


class CameraSubscriber:
    def __init__(self, intrinsics_file, extrinsics_file):
        with open(intrinsics_file, "r") as f:
            intrinsics_json = json.load(f)
            self.intrinsics = [SimpleNamespace(**d) for d in intrinsics_json]

        with open(extrinsics_file, "r") as f:
            self.extrinsics = json.load(f)

        self.sub = a0.SubscriberSync(a0.PubSubTopic(topic), a0.INIT_MOST_RECENT)

    def get_intrinsics(self):
        return self.intrinsics

    def get_extrinsics(self):
        return self.extrinsics

    def get_rgbd(self):
        return serdes.bytes_to_np(self.sub.read().payload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intrinsics",
        type=str,
        default="conf/intrinsics.json",
        help="JSON file to overwrite with current intrinsics.",
    )
    args = parser.parse_args()
    cameras = realsense_wrapper.RealsenseAPI()

    intrinsics = cameras.get_intrinsics()
    intrinsics_py = [
        dict(
            coeffs=x.coeffs, fx=x.fx, fy=x.fy, height=x.height, ppx=x.ppx, ppy=x.ppy, width=x.width
        )
        for x in intrinsics
    ]
    with open(args.intrinsics, "w") as f:
        json.dump(intrinsics_py, f)

    rgbd_pub = a0.Publisher(topic)

    log.info(f"Starting camera logger with {cameras.get_num_cameras()} cameras...")
    while True:
        img_bytes = serdes.np_to_bytes(cameras.get_rgbd())
        rgbd_pub.pub(a0.Packet(img_bytes))
