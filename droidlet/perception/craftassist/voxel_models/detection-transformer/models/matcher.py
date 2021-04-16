"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from itertools import zip_longest

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from box_ops import generalized_box_iou, box_cxcyczwhd_to_xyzxyz


def prepare_outputs(outputs):
    """
    Change convention from outputs = {scores[N], boxes[N]}
    into a [{scores[0], boxes[0]}, ..., {scores[N], boxes[N]}]
    """
    return [dict(zip_longest(outputs, t)) for t in zip_longest(*outputs.values())]


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class, cost_bbox, cost_giou):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        outputs = outputs.copy()
        outputs["pred_scores"] = outputs["pred_logits"].softmax(dim=-1)
        outputs = prepare_outputs(outputs)
        for out, tgt in zip(outputs, targets):
            cost = self._get_cost_matrix(out, tgt)
            src_idx, tgt_idx = linear_sum_assignment(cost.cpu())
            src_idx, tgt_idx = torch.as_tensor(src_idx), torch.as_tensor(tgt_idx)
            indices.append((src_idx, tgt_idx))
        return indices

    def _get_cost_matrix(self, out, tgt):
        out_prob, out_bbox = out["pred_scores"], out["pred_boxes"]
        tgt_ids, tgt_bbox = tgt["labels"], tgt["boxes"]
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcyczwhd_to_xyzxyz(out_bbox), box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )
        cost = (
            self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        )
        return cost


class SequentialMatcher(nn.Module):
    def forward(self, outputs, targets):
        return [(torch.arange(len(tgt["labels"])),) * 2 for tgt in targets]


class LexicographicalMatcher(nn.Module):
    def __init__(self, lexic="acb"):
        super().__init__()
        self.lexic = lexic

    def forward(self, outputs, targets):
        indices = []
        for tgt in targets:
            tgt_cls, tgt_box = tgt["labels"], tgt["boxes"]
            area = tgt_box[:, 2] * tgt_box[:, 3]
            if self.lexic == "acb":
                search_list = [
                    (-a, cl, b)
                    for cl, a, b in zip(tgt_cls.tolist(), area.tolist(), tgt_box.tolist())
                ]
            else:
                search_list = [
                    (cl, -a, b)
                    for cl, a, b in zip(tgt_cls.tolist(), area.tolist(), tgt_box.tolist())
                ]
            # argsort from https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
            j = sorted(range(len(search_list)), key=search_list.__getitem__)
            j = torch.as_tensor(j, dtype=torch.int64)
            i = torch.arange(len(j), dtype=j.dtype)
            indices.append((i, j))
        return indices


def build_matcher(args):
    if args.set_loss == "sequential":
        matcher = SequentialMatcher()
    elif args.set_loss == "hungarian":
        matcher = HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
        )
    elif args.set_loss == "lexicographical":
        matcher = LexicographicalMatcher()
    else:
        raise ValueError(
            f"Only sequential, lexicographical and hungarian accepted, got {args.set_loss}"
        )
    return matcher
