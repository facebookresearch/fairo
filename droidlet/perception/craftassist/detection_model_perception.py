"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import torch
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData

RADIUSX = 6
RADIUSY = 5
RADIUSZ = 6


class DetectionWrapper:
    """Detect objects using voxel and language input and return updates.

    Args:
        agent (LocoMCAgent): reference to the minecraft Agent
        model_path (str): path to the segmentation model
    """

    def __init__(self, model=None, threshold=0.5, radius=(RADIUSX, RADIUSY, RADIUSZ)):
        if type(model) is str or type(model) is dict:
            self.model = load_torch_detector(model)
        else:
            self.model = model
        self.threshold = threshold
        self.radius = radius

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
        masks = self.model(text_spans, blocks)
        for i in range(len(text_spans)):
            t = text_spans[i]
            mask = masks[i]
            seg = torch.nonzero(mask > self.threshold).tolist()
            out.labeled_blocks[t] = [
                (l[0] + offset[0], l[1] + offset[1], l[2] + offset[2]) for l in seg
            ]
        return out


def load_torch_detector(model_data):
    raise NotImplementedError
