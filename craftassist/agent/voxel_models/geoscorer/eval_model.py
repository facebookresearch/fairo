"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import torch
from tqdm import tqdm
from collections import defaultdict
import directional_utils as du
import spatial_utils as su
import training_utils as tu
from visualization_utils import GeoscorerVisualizer


class EvalMetrics:
    def __init__(self, keys, wrong_keys=[]):
        self.keys = keys
        self.overall_counts = {k: 0 for k in keys}
        self.batch_counts = {k: 0 for k in keys}
        self.overall_updates = 0
        self.batch_updates = 0
        self.wrong_values = {k: defaultdict(int) for k in wrong_keys}

    def update_batch_elem(self, updates):
        self.batch_updates += 1
        for k, v in updates.items():
            if k in self.batch_counts:
                self.batch_counts[k] += v

    def add_to_wrong_values(self, key, val):
        self.wrong_values[key][val] += 1

    def print_wrong_values(self):
        for k, vdict in self.wrong_values.items():
            print("  ", k)
            for vk, vc in vdict.items():
                print("    ", vk, vc)

    def get_batch_avg(self):
        if self.batch_updates == 0:
            return {}
        out = {k: v / self.batch_updates for k, v in self.batch_counts.items()}
        return out

    def reset_batch(self):
        self.overall_updates += self.batch_updates
        for k, v in self.batch_counts.items():
            self.overall_counts[k] += v
        self.batch_updates = 0
        self.batch_counts = {k: 0 for k in self.keys}

    def get_overall_avg(self):
        if self.overall_updates == 0:
            return {}
        out = {k: v / self.overall_updates for k, v in self.overall_counts.items()}
        return out

    def print_metrics(self, batch, overall=False):
        if not overall:
            avgs = self.get_batch_avg()
            outstr = "Batch {} ntot {} ".format(batch, self.batch_updates)
            counts = self.batch_counts
        else:
            avgs = self.get_overall_avg()
            outstr = "Overall ntot {} ".format(self.overall_updates)
            counts = self.overall_counts

        for k, v in avgs.items():
            outstr += " {} n {:<3} avg {:<6.2f}".format(k, counts[k], v)
        print(outstr)


def dir_correct(viewer_pos, viewer_look, dir_vec, predicted_coord):
    predicted_coord = torch.tensor(predicted_coord)
    pred_dir_vec = du.get_max_direction_vec(viewer_pos, viewer_look, predicted_coord)
    if (~pred_dir_vec.eq(dir_vec)).int().sum() == 0:
        return True
    return False


def eval_loop(tms, DL, opts, vis=True, wrong_counts=False):
    tu.set_modules(tms, train=False)
    dlit = iter(DL)
    c_sl = opts["context_sidelength"]
    max_allowed_key = "min_dist_above_{}".format(opts["max_allowed_dist"])

    wrong_keys = []
    if wrong_counts:
        wrong_keys = (["dir_vec", "vpvl", "vpvl_dir_vec"],)

    metrics = EvalMetrics(
        [
            "overlap_count",
            "out_of_bounds_count",
            max_allowed_key,
            "dir_wrong_count",
            "dist_from_target",
        ],
        wrong_keys=wrong_keys,
    )
    n = len(dlit)
    if vis:
        viz = GeoscorerVisualizer()

    for j in range(n):
        metrics.reset_batch()
        batch = dlit.next()
        targets, scores = tu.get_scores_and_target_from_datapoint(tms, batch, opts)
        max_ind = torch.argmax(scores, dim=1)

        if vis:
            it = range(3)
        else:
            it = range(batch["context"].size()[0])

        if opts["tqdm"]:
            it = tqdm(it)

        for i in it:
            context = batch["context"][i]
            seg = batch["seg"][i]
            vp = batch["viewer_pos"][i]
            vl = batch["viewer_look"][i]
            dir_vec = batch["dir_vec"][i]

            predicted_ind = max_ind[i]
            target_ind = targets[i]
            predicted_coord = su.index_to_coord(predicted_ind.cpu().item(), c_sl)
            target_coord = su.index_to_coord(target_ind.cpu().item(), c_sl)

            if vis:
                viz.visualize(context)
                viz.visualize(seg)
                viz.visualize_combined(context, seg, target_coord, vp, vl)
                viz.visualize_combined(context, seg, predicted_coord)

            # TODO: some of these calcs can be dramatically sped up using tensors
            c_tuple = set([s[0] for s in su.sparsify_voxel(context)])
            s_tuple = set([s[0] for s in su.sparsify_voxel(seg)])
            predicted_seg_voxel = su.shift_sparse_voxel(
                [(s, (0, 0)) for s in s_tuple],
                predicted_coord,
                min_b=[0, 0, 0],
                max_b=[c_sl, c_sl, c_sl],
            )
            predicted_seg_tuple = set([v[0] for v in predicted_seg_voxel])

            res = {
                "overlap_count": len(predicted_seg_tuple & c_tuple),
                "out_of_bounds_count": len(s_tuple) - len(predicted_seg_tuple),
                "min_dist_above_8": 0,
                "dir_wrong_count": 0,
                "dist_from_target": su.euclid_dist(predicted_coord, target_coord),
            }

            min_dist = su.get_min_block_pair_dist(predicted_seg_tuple, c_tuple)
            if min_dist > opts["max_allowed_dist"]:
                res[max_allowed_key] = 1
            if vis:
                print("batch info:")
                print("  viewer_pos", vp)
                print("  viewer_look", vl)
                print("  target_coord", target_coord)
                print("  predicted_coord", predicted_coord)
                print("  dir_vec", dir_vec)
                print("-------------\n")
            if not dir_correct(vp, vl, dir_vec, predicted_coord):
                res["dir_wrong_count"] = 1
                if wrong_counts:
                    dvl = dir_vec.tolist()
                    vpl = vp[[0, 2]].tolist() + vl[[0, 2]].tolist()
                    metrics.add_to_wrong_values("dir_vec", tuple(dvl))
                    metrics.add_to_wrong_values("vpvl", tuple(vpl))
                    metrics.add_to_wrong_values("vpvl_dir_vec", tuple(vpl + dvl))
            metrics.update_batch_elem(res)

        metrics.print_metrics(j)
        metrics.print_wrong_values()
        if vis:
            input("Press enter to continue.")
    # To include the last batch in the overall aggregates
    metrics.reset_batch()
    metrics.print_metrics(0, overall=True)
    metrics.print_wrong_values()


if __name__ == "__main__":
    parser = tu.get_train_parser()
    parser.add_argument(
        "--max_allowed_dist", type=int, default=8, help="threshold for min dist above metric"
    )
    parser.add_argument("--tqdm", action="store_true", help="use tqdm")
    parser.add_argument("--vis_eval", action="store_true")
    parser.add_argument("--wrong_counts", action="store_true")
    opts = vars(parser.parse_args())

    # Setup the data, models and optimizer
    dataset, dataloader = tu.setup_dataset_and_loader(opts)
    tms = tu.get_context_segment_trainer_modules(
        opts, opts["checkpoint"], backup=opts["backup"], verbose=True, use_new_opts=True
    )
    if opts["cuda"] == 1:
        tms["score_module"].cuda()
        tms["lfn"].cuda()

    eval_loop(tms, dataloader, opts, vis=opts["vis_eval"], wrong_counts=opts["wrong_counts"])
