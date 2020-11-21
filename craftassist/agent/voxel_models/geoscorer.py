"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import sys
import os
import torch

GEOSCORER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "geoscorer/")
sys.path.append(GEOSCORER_DIR)

from geoscorer_wrapper import ContextSegmentMergerWrapper
import spatial_utils as su
import directional_utils as du


class Geoscorer(object):
    """
    A model class that provides geoscorer functionality.
    This is distinct from the wrapper itself because I see the wrapper
    becoming more specialized as we add more functionality and this object
    possible becoming a process or holding multiple wrappers.
    """

    #       "AWAY": torch.tensor([0, 1, 0, 1, 0]),
    #       "FRONT": torch.tensor([0, 1, 0, 1, 0]),
    #       "BACK": torch.tensor([0, 1, 0, 0, 1]),
    #       "LEFT": torch.tensor([1, 0, 0, 0, 1]),
    #       "RIGHT": torch.tensor([1, 0, 0, 1, 0]),
    #       "DOWN": torch.tensor([0, 0, 1, 0, 1]),
    #       "UP": torch.tensor([0, 0, 1, 1, 0]),

    def __init__(self, merger_model_path=None):
        if merger_model_path is not None:
            logging.info("Geoscorer using  merger_model_path={}".format(merger_model_path))
            self.merger_model = ContextSegmentMergerWrapper(merger_model_path)
        else:
            raise Exception("specify a geoscorer model")
        self.radius = self.merger_model.context_sl // 2
        self.seg_sl = self.merger_model.seg_sl
        self.blacklist = ["BETWEEN", "INSIDE", "AWAY", "NEAR"]

    # Define the circumstances where we can use geoscorer
    def use(self, steps, rel_dir):
        if steps is not None:
            return False

        if rel_dir is None or rel_dir in self.blacklist:
            return False
        return True

    def produce_object_positions(self, objects, context, minc, reldir, player_pos):
        segments = [self._process_segment(o[0]) for o in objects]
        context_t = self._process_context(context)
        dir_vec = du.get_direction_embedding(reldir)
        viewer_pos = torch.tensor(self._process_viewer_pos(player_pos, minc)).float()

        mincs = []
        for seg in segments:
            local_target_coord, global_origin_coord = self._seg_context_processed_to_coord(
                seg, context_t, minc, dir_vec, viewer_pos
            )
            context_t = su.combine_seg_context(seg, context_t, local_target_coord)
            mincs.append(global_origin_coord)
        origin = mincs[0]
        offsets = [[p[i] - origin[i] for i in range(3)] for p in mincs]
        return origin, offsets

    def _seg_context_processed_to_coord(self, segment, context, min_corner, dir_vec, viewer_pos):
        local_target_coord, local_origin_coord = self.merger_model.segment_context_to_pos(
            segment, context, dir_vec, viewer_pos
        )
        global_coord = self._local_to_global(local_origin_coord, min_corner)
        return local_target_coord, global_coord

    def _process_viewer_pos(self, global_vp, minc):
        return self._global_to_local(global_vp, minc)

    # Get blocks returns (y z x) so conver to (x y z)
    def _process_context(self, context):
        c_tensor = torch.from_numpy(context[:, :, :, 0]).long().to(device="cuda")
        c_tensor = c_tensor.permute(2, 0, 1)
        c_tensor = c_tensor.contiguous()
        return c_tensor

    def _process_segment(self, segment):
        """
        Takes a segment, described as a list of tuples of the form:
            ((x, y, z), (block_id, ?))
        Returns an 8x8x8 block with the segment shifted to the origin its bounds.
        """
        shifted_seg, _ = su.shift_sparse_voxel_to_origin(segment)

        sl = self.seg_sl
        c = self.seg_sl // 2
        p, _ = su.densify(shifted_seg, [sl, sl, sl], center=[c, c, c], useid=True)
        s_tensor = torch.from_numpy(p).long().to(device="cuda")
        return s_tensor

    def _local_to_global(self, local_vp, min_c):
        return [sum(x) for x in zip(local_vp, min_c)]

    def _global_to_local(self, global_vp, min_c):
        return [x[0] - x[1] for x in zip(global_vp, min_c)]
