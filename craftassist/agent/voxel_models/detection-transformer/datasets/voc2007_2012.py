"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_in_coco_format(self, idx: int):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_in_coco_format(sample_idx)


from .voc import VOCDetection
from .voc2012 import VOCDetection2012

from .voc import make_voc_transforms


def build(image_set, args):
    ds_2007 = VOCDetection(
        image_set=image_set, transforms=make_voc_transforms(image_set, args.remove_difficult)
    )

    if image_set == "test":
        return ds_2007

    ds_2012 = VOCDetection2012(
        image_set=image_set, transforms=make_voc_transforms(image_set, args.remove_difficult)
    )

    return ConcatDataset([ds_2007, ds_2012])
