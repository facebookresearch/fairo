"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import torchvision
from typing import Iterable

import to_coco_api
import transforms as T

VOC_PATH = "/checkpoint/szagoruyko/detection_transformer_shared/datasets01"


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, image_set: str = "train", transforms: Iterable = None):
        super().__init__(VOC_PATH, image_set=image_set, year="2007", download=False)
        self.prepare = to_coco_api.PrepareInstance()
        self._transforms = transforms

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target

    def get_in_coco_format(self, idx: int):
        image, target = super().__getitem__(idx)
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


def make_voc_transforms(image_set, remove_difficult):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    transform_train = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize([400, 500, 600], max_size=1000),
                T.Compose(
                    [
                        T.RandomResize([400, 500, 600]),
                        T.RandomCrop((384, 384)),
                        T.RandomResize([400, 500, 600], max_size=1000),
                    ]
                ),
            ),
            normalize,
            T.RemoveDifficult(remove_difficult),
        ]
    )

    transform_val = T.Compose([T.RandomResize([600], max_size=1000), normalize])

    transforms = {
        "train": transform_train,
        "trainval": transform_train,
        "val": transform_val,
        "test": transform_val,
    }

    return transforms[image_set]


def build(image_set, args):
    return VOCDetection(
        image_set=image_set, transforms=make_voc_transforms(image_set, args.remove_difficult)
    )
