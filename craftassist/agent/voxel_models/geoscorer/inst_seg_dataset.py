"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import math
import os
import random
import pickle
import torch
import torch.utils.data
from collections import defaultdict
import spatial_utils as su
import directional_utils as du

################################################################################
# Utils for Parsing InstSeg Data File
################################################################################


# IMPORTANT NOTE: Changes to this function only take effect if you remove the cached
# preparsed file.
def parse_instance_data(inst_data):
    parsed_instance_data = []
    for h in inst_data:
        S, segs, labels, _ = h
        segs = segs.astype("int32")
        blocks = list(zip(*S.nonzero()))
        # First convert the schematic into sparse segment info
        offsets = [[i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
        sizes = [len(segs), len(segs[0]), len(segs[0][0])]
        instances = defaultdict(list)
        touching = defaultdict(set)
        for b in blocks:
            b_w_id = [b[0], b[1], b[2]]
            b_w_id.append(S[b])
            seg_id = segs[b]
            instances[seg_id].append(b_w_id)
            for off in offsets:
                nc = tuple([a + b for a, b in zip(b, off)])
                # check out of bounds
                if not all(e > 0 for e in nc) or not all([nc[i] < sizes[i] for i in range(3)]):
                    continue
                if segs[nc] != 0 and segs[nc] != seg_id:
                    touching[seg_id].add(segs[nc])

        # Then get the width/height/depth metadata
        metadata = {}
        for i, blocks in instances.items():
            maxs = [0, 0, 0]
            mins = [sizes[i] for i in range(3)]
            for b in blocks:
                maxs = [max(b[i], maxs[i]) for i in range(3)]
                mins = [min(b[i], mins[i]) for i in range(3)]
            metadata[i] = {"size": [x - n + 1 for n, x in zip(mins, maxs)], "label": labels[i]}
        # For now remove houses where there are no touching components
        # this is only one house
        if len(touching) == 0:
            continue
        parsed_instance_data.append(
            {"segments": instances, "touching": touching, "metadata": metadata}
        )
    return parsed_instance_data


def parse_segments_into_file(dpath, save_path):
    inst_data = pickle.load(open(dpath, "rb"))
    parsed_instance_data = parse_instance_data(inst_data)
    pickle.dump(parsed_instance_data, open(save_path, "wb+"))


def convert_tuple_to_block(b):
    if b[3] < 0 or b[3] > 255:
        raise Exception("block id out of bounds")
    return ((b[0], b[1], b[2]), (b[3], 0))


################################################################################
# Utils for Creating InstSeg Dataset
################################################################################


def get_context_seg_sparse(seg_data, drop_perc, rand_drop=True):
    # first choose a house
    sd = random.choice(seg_data)

    # then drop some segs
    if drop_perc < 0:
        drop_perc = random.randint(0, 80) * 1.0 / 100
    seg_ids = list(sd["segments"].keys())
    random.shuffle(seg_ids)
    num_segs = len(seg_ids)
    to_keep = math.ceil(num_segs - num_segs * drop_perc)
    keep_ids = seg_ids[:to_keep]

    # choose a remaining seg to get a connected one
    conn_to_target_id = random.choice(keep_ids)
    if conn_to_target_id not in sd["touching"]:
        conn_to_target_id = random.choice(list(sd["touching"].keys()))
        keep_ids.append(conn_to_target_id)

    # get a connected seg as target
    target_seg_id = random.choice(list(sd["touching"][conn_to_target_id]))
    keep_ids = [k for k in keep_ids if k != target_seg_id]

    # make segment out of blocks from target_seg
    seg_sparse = [convert_tuple_to_block(b) for b in sd["segments"][target_seg_id]]

    # make context out of blocks from keep_ids
    context_sparse = []
    for i in set(keep_ids):
        context_sparse += [convert_tuple_to_block(b) for b in sd["segments"][i]]

    return context_sparse, seg_sparse


# Note that 1/7 segments are larger than 8x8x8
# Only 1/70 are larger than 16x16x16, maybe move to this size seg
class InstanceSegData(torch.utils.data.Dataset):
    """
    The goal of the dataset is to take a context voxel, a segment voxel,
    and a direction and predict the correct location to put the min corner
    of the segment voxel in the context space to combine them correctly.

    This dataset specifically uses the inst_seg house data that contains
    handmade houses with each segment labeled.  This dataset chooses one
    of the segments of the house, removes it from the context, and uses
    its actual position as the target for reconstruction.

    Each element contains:
        "context": CxCxC context voxel
        "segment": SxSxS segment voxel
        "dir_vec": 6 element direction vector
        "viewer_pos": the position of the viewer
        "viewer_look": the center of the context object, possibly different
            from the space center if the context is larger than the space.
    """

    def __init__(
        self,
        data_dir="/checkpoint/drotherm/minecraft_dataset/vision_training/training3/",
        nexamples=10000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        drop_perc=0.8,
        ground_type=None,
        random_ground_height=False,
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.drop_perc = drop_perc
        self.ground_type = ground_type
        self.random_ground_height = random_ground_height

        # Load the parsed data
        parsed_file = os.path.join(data_dir, "training_parsed.pkl")
        if not os.path.exists(parsed_file):
            print(">> Redo inst seg parse")
            data_path = os.path.join(data_dir, "training_data.pkl")
            parse_segments_into_file(data_path, parsed_file)
        self.seg_data = pickle.load(open(parsed_file, "rb"))

    def _get_example(self):
        # Get the raw context and seg
        context_sparse, seg_sparse = get_context_seg_sparse(self.seg_data, self.drop_perc)
        # Convert into an example
        example = su.sparse_context_seg_in_space_to_example(
            context_sparse,
            seg_sparse,
            self.c_sl,
            self.s_sl,
            self.useid,
            self.ground_type,
            self.random_ground_height,
        )

        # Add the direction info
        target_coord = torch.tensor(su.index_to_coord(example["target"], self.c_sl))
        example["viewer_pos"], example["dir_vec"] = du.get_random_vp_and_max_dir_vec(
            example["viewer_look"], target_coord, self.c_sl
        )
        return example

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return abs(self.num_examples)


if __name__ == "__main__":
    import argparse
    from visualization_utils import GeoscorerDatasetVisualizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=3, help="num examples to visualize")
    parser.add_argument("--useid", action="store_true", help="should we use the block id")
    parser.add_argument("--drop_perc", type=float, default=0.8, help="should we use the block id")
    parser.add_argument("--ground_type", type=str, default=None, help="ground type")
    opts = parser.parse_args()

    dataset = InstanceSegData(
        nexamples=opts.n_examples,
        drop_perc=opts.drop_perc,
        useid=opts.useid,
        ground_type=opts.ground_type,
    )
    vis = GeoscorerDatasetVisualizer(dataset)
    for n in range(len(dataset)):
        vis.visualize()
