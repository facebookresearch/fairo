"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import json

""" An example:
dataset_config = {
    "inst_seg": [
        {"drop_perc": -1.0, "ground_type": None, "random_ground_height": False, "prob": 0.1},
        {"drop_perc": -1.0, "ground_type": "flat", "random_ground_height": False, "prob": 0.1},
    ],
    "shape_piece": [
        {"ground_type": None, "random_ground_height": False, "prob": 0.1},
        {"ground_type": "flat", "random_ground_height": True, "prob": 0.1},
        {"ground_type": "hilly", "random_ground_height": True, "prob": 0.1},
    ],
    "shape_pair": [
        {
            "shape_type": "random",
            "fixed_size": None,
            "max_shift": 6,
            "ground_type": "flat",
            "random_ground_height": True,
            "prob": 0.5,
        },
    ],
}
filename = "dataset_configs/all_datasets_base.json"
"""

dataset_config = {
    "shape_pair": [
        {
            "shape_type": "same",
            "fixed_size": 3,
            "max_shift": 0,
            "ground_type": "flat",
            "random_ground_height": False,
            "prob": 0.05,
        },
        {
            "shape_type": "same",
            "fixed_size": 4,
            "max_shift": 0,
            "ground_type": "flat",
            "random_ground_height": False,
            "prob": 0.05,
        },
        {
            "shape_type": "same",
            "fixed_size": 5,
            "max_shift": 0,
            "ground_type": "flat",
            "random_ground_height": False,
            "prob": 0.05,
        },
        {
            "shape_type": "same",
            "fixed_size": 6,
            "max_shift": 0,
            "ground_type": "flat",
            "random_ground_height": False,
            "prob": 0.05,
        },
        {
            "shape_type": "random",
            "fixed_size": None,
            "max_shift": 0,
            "ground_type": "flat",
            "random_ground_height": True,
            "prob": 0.1,
        },
        {
            "shape_type": "random",
            "fixed_size": None,
            "max_shift": 0,
            "ground_type": "hilly",
            "random_ground_height": True,
            "prob": 0.1,
        },
        {
            "shape_type": "random",
            "fixed_size": None,
            "max_shift": 5,
            "ground_type": "flat",
            "random_ground_height": True,
            "prob": 0.1,
        },
        {
            "shape_type": "random",
            "fixed_size": None,
            "max_shift": 5,
            "ground_type": "hilly",
            "random_ground_height": True,
            "prob": 0.1,
        },
    ],
    "shape_piece": [{"ground_type": "flat", "random_ground_height": True, "prob": 0.2}],
    "inst_seg": [{"ground_type": "flat", "random_ground_height": False, "prob": 0.2}],
}

filename = "dataset_configs/all_good_split.json"
with open(filename, "w+") as f:
    json.dump(dataset_config, f)
print("dumped", filename)
