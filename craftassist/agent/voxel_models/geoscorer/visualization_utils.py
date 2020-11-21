"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys
import visdom
import torch

VOXEL_MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(VOXEL_MODELS_DIR)
import plot_voxels as pv
import spatial_utils as su
import training_utils as tu


class GeoscorerVisualizer:
    def __init__(self):
        self.vis = visdom.Visdom(server="http://localhost")
        self.sp = pv.SchematicPlotter(self.vis)

    def visualize(self, tensor):
        self.sp.drawGeoscorerPlotly(tensor)

    def visualize_combined(self, context, segment, origin, vp=None, vl=None):
        combined_voxel = su.combine_seg_context(segment, context, origin, seg_mult=3)
        if vl is not None:
            vl = vl.long()
            combined_voxel[vl[0], vl[1], vl[2]] = 5
        if vp is not None:
            vp = vp.long()
            combined_voxel[vp[0], :, vp[2]] = 10
        self.sp.drawGeoscorerPlotly(combined_voxel)


class GeoscorerDatasetVisualizer:
    def __init__(self, dataset):
        self.vis = visdom.Visdom(server="http://localhost")
        self.sp = pv.SchematicPlotter(self.vis)
        self.dataset = dataset
        self.vis_index = 0
        self.model = None
        self.opts = None

    def set_model(self, model, opts=None):
        self.model = model
        if opts:
            self.opts = opts

    def visualize(self, use_model=False, verbose=False):
        if self.vis_index == len(self.dataset):
            raise Exception("No more examples to visualize in dataset")
        b = self.dataset[self.vis_index]
        c_sl = b["context"].size()[0]
        if verbose:
            print("viewer_pos", b["viewer_pos"])
            print("viewer_look", b["viewer_look"])
            print("target_coord", su.index_to_coord(b["target"].item(), c_sl))
            print("dir_vec", b["dir_vec"])
            print("-----------\n")
        if "schematic" in b:
            self.sp.drawGeoscorerPlotly(b["schematic"])
        self.vis_index += 1
        self.sp.drawGeoscorerPlotly(b["context"])
        self.sp.drawGeoscorerPlotly(b["seg"])
        target_coord = su.index_to_coord(b["target"].item(), c_sl)
        combined_voxel = su.combine_seg_context(b["seg"], b["context"], target_coord, seg_mult=3)
        # Add in the viewer pos and look
        vp = b["viewer_pos"].long()
        vl = b["viewer_look"].long()
        combined_voxel[vl[0], vl[1], vl[2]] = 5
        combined_voxel[vp[0], :, vp[2]] = 10
        self.sp.drawGeoscorerPlotly(combined_voxel)

        if use_model:
            b = {k: t.unsqueeze(0) for k, t in b.items()}
            targets, scores = tu.get_scores_and_target_from_datapoint(self.model, b, self.opts)
            max_ind = torch.argmax(scores, dim=1)
            pred_coord = su.index_to_coord(max_ind, c_sl)
            b = {k: t.squeeze(0) for k, t in b.items()}
            predicted_voxel = su.combine_seg_context(
                b["seg"], b["context"], pred_coord, seg_mult=3
            )
            self.sp.drawGeoscorerPlotly(predicted_voxel)
