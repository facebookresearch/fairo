"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import torch
import numpy as np
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData
from droidlet.perception.craftassist.voxel_models.semantic_segmentation.vision import SemSegWrapper
from droidlet.lowlevel.minecraft.iglu_util import IGLU_BLOCK_MAP
from .utils.vision_logger import VisionLogger
from droidlet.lowlevel.minecraft.pyworld.world_config import opts as world_opts
from droidlet.event import sio

RADIUSX = 6
RADIUSY = 5
RADIUSZ = 6


class DetectionWrapper:
    """Detect objects using voxel and language input and return updates.

    Args:
        agent (LocoMCAgent): reference to the minecraft Agent
        model_path (str): path to the segmentation model
    """

    def __init__(self, model=None, agent=None, threshold=None, radius=(RADIUSX, RADIUSY, RADIUSZ)):
        self.cuda = torch.cuda.is_available()
        if type(model) is str or type(model) is dict:
            self.model = load_torch_detector(model, self.cuda)
        else:
            self.model = model
        self.agent = agent
        self.threshold = self.model.threshold if not threshold else threshold
        self.radius = radius
        self.vision_err_cnt = 0

        self.VisionErrorLogger = VisionLogger(
            "vision_error_details.csv",
            [
                "command",
                "action_dict",
                "time",
                "vision_error",
                "ref_obj_text_span" "world_snapshot",
            ],
        )

        @sio.on("saveErrorDetailsToCSV")
        def save_vision_error_details(sid, data):
            logging.info("Saving vision error details: %r" % (data))
            if "vision_error" not in data or "msg" not in data:
                logging.info("Could not save error details due to error in dashboard backend.")
                return
            is_vision_error = data["vision_error"]
            ref_obj_text_span = data["ref_obj_text_span"]
            if is_vision_error:
                sl = world_opts.SL
                h = world_opts.H
                blocks = self.agent.get_blocks(
                    int(sl / 4),
                    int(3 * sl // 4 - 1),
                    0,
                    int(h // 2 - 1),
                    int(sl // 4),
                    int(3 * sl // 4 - 1),
                )
                # print(("vision error blocks: %r" % (blocks)))
                # logging.info("vision error blocks: %r" % (blocks))
                vision_err_fname = f"vision_err_{self.vision_err_cnt}.npy"
                self.VisionErrorLogger.log_dialogue_outputs(
                    [
                        data["msg"],
                        data["action_dict"],
                        None,
                        True,
                        ref_obj_text_span,
                        vision_err_fname,
                    ]
                )
                np.save(vision_err_fname, blocks)
                self.vision_err_cnt += 1

    def perceive(self, blocks, text_spans=[], offset=(0, 0, 0)):
        """
        Run the detection classifier and get the resulting voxel predictions

        Args:
            text_spans (list[str]): list of text spans of reference objects from logical form. This is
                             used to detect a specific reference object.
            blocks: WxHxWx2 numpy array
                    corresponding to x, y, z, bid, meta
            offset: (x, y, z) offsets.  these will be _added_ back to the locations
                    of each instance segmentation block

        """
        out = CraftAssistPerceptionData()
        if not text_spans:
            return out
        if self.model is None:
            return out
        # masks should be a list of HxWxW torch arrays with values between 0 and 1
        print(blocks.shape)
        if len(blocks.shape) == 4:
            blocks = np.apply_along_axis(
                lambda x: IGLU_BLOCK_MAP[(int(x[0]), int(x[1]))], 3, blocks
            )
        assert len(blocks.shape) == 3
        masks = self.model.perceive(blocks, text_spans)
        for i in range(len(text_spans)):
            t = text_spans[i]
            seg = masks[i]
            out.labeled_blocks[t] = [
                (l[0] + offset[0], l[1] + offset[1], l[2] + offset[2]) for l in seg
            ]
        return out


def load_torch_detector(model_data, cuda=False):
    model = SemSegWrapper(model=model_data, cuda=cuda)
    return model


if __name__ == "__main__":
    model_path = "/checkpoint/yuxuans/models/hitl_vision/v5.pt"
    mw = DetectionWrapper(model_path)
    blocks = np.zeros((17, 13, 17, 2))
    for ix in range(2, 5):
        for iy in range(2, 5):
            for iz in range(2, 5):
                blocks[ix, iy, iz] = (35, 14)
    out = mw.perceive(blocks, ["red cube", "blue sphere"])
    print(out)
