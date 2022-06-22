import logging
import a0
import numpy as np
import signal

from polygrasp import serdes
from . import wait_until_a0_server_ready, start_a0_server_heartbeat

topic_key = "segmentation"

log = logging.getLogger(__name__)


class SegmentationClient:
    def __init__(self):
        wait_until_a0_server_ready(topic_key)
        self.client = a0.RpcClient(topic_key)

    def segment_img(self, rgbd, min_mask_size=2500):
        bits = serdes.rgbd_to_capnp(rgbd).to_bytes()
        result_bits = self.client.send_blocking(bits).payload
        labels = serdes.capnp_to_rgbd(result_bits)

        num_objs = int(labels.max())
        obj_masked_rgbds = []
        obj_masks = []

        for obj_i in range(1, num_objs + 1):
            obj_mask = labels == obj_i

            obj_mask_size = obj_mask.sum()
            if obj_mask_size < min_mask_size:
                continue
            obj_masked_rgbd = rgbd * obj_mask[:, :, None]
            obj_masked_rgbds.append(obj_masked_rgbd)
            obj_masks.append(obj_mask)

        return obj_masked_rgbds, obj_masks


class SegmentationServer:
    def _get_segmentations(self, rgbd: np.ndarray):
        raise NotImplementedError

    def start(self):
        def onrequest(req):
            log.info("Got request; computing segmentations...")

            payload = req.pkt.payload
            rgbd = serdes.capnp_to_rgbd(payload)
            result = self._get_segmentations(rgbd)

            log.info("Done. Replying with serialized segmentations...")
            req.reply(serdes.rgbd_to_capnp(result).to_bytes())

        log.info("Starting server...")
        server = a0.RpcServer(topic_key, onrequest, None)
        start_a0_server_heartbeat(topic_key)
        signal.pause()
