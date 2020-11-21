"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import argparse
import training_utils as tu
import spatial_utils as su
import torch


class ContextSegmentMergerWrapper(object):
    """
    Wrapper for the geoscorer
    """

    def __init__(self, models_path):
        if models_path is None:
            raise Exception("Geoscorer wrapper requires a model path")

        # Taken from defaults in training_utils
        # TODO: actually make this match non-manually
        opts = {
            "hidden_dim": 65,
            "num_layers": 3,
            "blockid_embedding_dim": 8,
            "context_sidelength": 32,
            "useid": True,
            "num_words": 256,
        }

        self.tms = tu.get_context_segment_trainer_modules(
            opts=opts, checkpoint_path=models_path, backup=False, verbose=False
        )
        print("loaded model opts", self.tms["opts"])
        self.context_sl = 32
        self.seg_sl = 8
        tu.set_modules(tms=self.tms, train=False)

    def segment_context_to_pos(self, segment, context, dir_vec, viewer_pos):
        batch = {
            "context": context.unsqueeze(0),
            "seg": segment.unsqueeze(0),
            "dir_vec": dir_vec.unsqueeze(0),
            "viewer_pos": viewer_pos.unsqueeze(0),
            "viewer_look": torch.tensor([16.0, 16.0, 16.0]).unsqueeze(0),
        }
        scores = tu.get_scores_from_datapoint(self.tms, batch, self.tms["opts"])
        index = scores[0].flatten().max(0)[1]
        target_coord = su.index_to_coord(index.item(), self.context_sl)
        seg_origin = su.get_seg_origin_from_target_coord(segment, target_coord)
        return target_coord, seg_origin


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", type=str, help="path to geoscorer models")
    args = parser.parse_args()

    geoscorer = ContextSegmentMergerWrapper(args.models_path)
