"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from .voc import VOCDetection
from typing import Iterable
import to_coco_api


VOC_PATH = "/datasets01/VOC/060817/"


class VOCDetection2012(VOCDetection):
    def __init__(self, image_set: str = "train", transforms: Iterable = None):
        super(VOCDetection, self).__init__(
            VOC_PATH, image_set=image_set, year="2012", download=False
        )
        self.prepare = to_coco_api.PrepareInstance()
        self._transforms = transforms


from .voc import make_voc_transforms


def build(image_set, args):
    # if we only use voc2012, then we need to adapt trainval and test to
    # VOC2012 constraints
    if image_set == "test":
        image_set = "val"
    if image_set == "trainval":
        image_set = "train"
    return VOCDetection2012(
        image_set=image_set, transforms=make_voc_transforms(image_set, args.remove_difficult)
    )
