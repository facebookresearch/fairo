"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import random
import torch
import torch.utils.data
from shape_dataset import ShapePairData, ShapePieceData
from inst_seg_dataset import InstanceSegData


# Returns three tensors: 32x32x32 context, 8x8x8 segment, 1 target
class CombinedData(torch.utils.data.Dataset):
    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        ratios={"shape": 1.0},
        extra_params={},
        config=None,
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []

        if config is None:
            self.ds_names = [k for k, p in ratios.items() if p > 0]
            self.ds_probs = [ratios[name] for name in self.ds_names]
            self.datasets = dict(
                [(name, [self._get_dataset(name, extra_params)]) for name in self.ds_names]
            )
        else:
            self.ds_names = []
            self.ds_probs = []
            self.datasets = {}
            for name, params in config.items():
                for i, p in enumerate(params):
                    pname = "{}_{}".format(name, i)
                    self.ds_names.append(pname)
                    self.ds_probs.append(p["prob"])
                    self.datasets[pname] = [self._get_dataset(name, p)]

        if sum(self.ds_probs) != 1.0:
            raise Exception("Sum of probs must equal 1.0")

        print("Datasets")
        for name, dss in self.datasets.items():
            print("   ", name, len(dss))

    def _get_dataset(self, name, extra_params):
        if name == "inst_seg":
            drop_perc = extra_params.get("drop_perc", 0.0)
            ground_type = extra_params.get("ground_type", None)
            random_ground_height = extra_params.get("random_ground_height", False)
            return InstanceSegData(
                nexamples=self.num_examples,
                drop_perc=drop_perc,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                ground_type=ground_type,
                random_ground_height=random_ground_height,
            )
        if name == "shape_pair":
            shape_type = extra_params.get("shape_type", "random")
            fixed_size = extra_params.get("fixed_size", None)
            max_shift = extra_params.get("max_shift", 0)
            ground_type = extra_params.get("ground_type", None)
            random_ground_height = extra_params.get("random_ground_height", False)
            return ShapePairData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                ground_type=ground_type,
                max_shift=max_shift,
                fixed_size=fixed_size,
                shape_type=shape_type,
                random_ground_height=random_ground_height,
            )
        if name == "shape_piece":
            ground_type = extra_params.get("ground_type", None)
            random_ground_height = extra_params.get("random_ground_height", False)
            return ShapePieceData(
                nexamples=self.num_examples,
                context_side_length=self.c_sl,
                seg_side_length=self.s_sl,
                useid=self.useid,
                ground_type=ground_type,
                random_ground_height=random_ground_height,
            )
        raise Exception("No dataset with name {}".format(name))

    def _get_example(self):
        ds_name = np.random.choice(self.ds_names, p=self.ds_probs)
        dataset = random.choice(self.datasets[ds_name])
        return dataset._get_example()

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    from visualization_utils import GeoscorerDatasetVisualizer

    dataset = CombinedData(
        nexamples=3, useid=False, ratios={"shape_pair": 1.0}, extra_params={"shape_type": "cube"}
    )
    vis = GeoscorerDatasetVisualizer(dataset)
    for n in range(len(dataset)):
        vis.visualize()
