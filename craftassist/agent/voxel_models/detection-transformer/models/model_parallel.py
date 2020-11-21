"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import misc as misc_ops

import box_ops

# TODO need to do proper packaging as this is getting confusing
from utils import NestedTensor, accuracy, get_world_size, is_dist_avail_and_initialized

from .backbone import build_backbone
from .common import MLP
from .detr import build_transformer
from .loss_utils import dice_loss, sigmoid_focal_loss
from .mask_heads import DETRmask
from .matcher import build_matcher


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        print("... DTER Forwarding ... ")
        print(samples.tensors.shape)
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        # (6, bs, num_queries, hidden_dim)
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        print("---- hs size ----")
        print(hs.shape)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def match(self, outputs, targets):
        assert len(outputs["pred_logits"]) == len(targets)
        return self.matcher(outputs, targets)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # class loss
        target_classes_o = [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        # TODO optimize this
        for k, (I, _) in enumerate(indices):
            target_classes[k][I] = target_classes_o[k]

        loss_ce = F.cross_entropy(
            src_logits.flatten(0, 1), target_classes.flatten(0, 1), self.empty_weight
        )

        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in
            # this one here
            idx = self._get_src_permutation_idx(indices)
            ordered_src_logits = src_logits[idx]
            target_classes_o = torch.cat(target_classes_o)
            losses["class_error"] = (
                100 - accuracy(ordered_src_logits.detach(), target_classes_o)[0]
            )
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Not really a loss, but well :-)
        No gradients anyway
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        # print('------ outputs ---------')
        # print(outputs['pred_logits'].shape)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / (num_boxes * 4)

        if "loss_giou" in self.weight_dict:
            loss_giou = 1 - torch.diag(
                box_ops.generalized_box_iou(
                    box_ops.box_cxcyczwhd_to_xyzxyz(src_boxes),
                    box_ops.box_cxcyczwhd_to_xyzxyz(target_boxes),
                )
            )
            losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        # print('---- loss masks ----')

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        # print('---- src masks ----')
        # print(src_masks[0][0])
        # print('---- targets ----')
        # print(len(targets))
        # print(targets[0]['masks'].shape)
        # print(targets[0]['labels'].shape)
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        src_masks = misc_ops.interpolate(
            src_masks[:, None], size=target_masks.shape[-3:], mode="trilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full((len(src),), i, dtype=torch.int64) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full((len(tgt),), i, dtype=torch.int64) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum([len(t["labels"]) for t in targets])
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    def __init__(self, rescale_to_orig_size=False, threshold=0.3):
        super().__init__()
        self.rescale_to_orig_size = rescale_to_orig_size
        self.threshold = threshold

    def forward(self, outputs, targets):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        # convert to [x0, y0, x1, y1, z0, z1] format
        boxes = []
        field = "orig_size" if self.rescale_to_orig_size else "size"
        out_bbox = box_ops.box_cxcyczwhd_to_xyzxyz(out_bbox)
        for b, t in zip(out_bbox, targets):
            img_d, img_h, img_w = t[field].tolist()
            b = b * torch.tensor(
                [img_w, img_h, img_d, img_w, img_h, img_d], dtype=torch.float32, device=b.device
            )
            boxes.append(b)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        results = [
            {"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)
        ]

        if "pred_masks" in outputs:
            max_h = max([tgt["size"][0] for tgt in targets])
            max_w = max([tgt["size"][1] for tgt in targets])
            outputs_masks = outputs["pred_masks"]
            outputs_masks = outputs_masks.squeeze(2)
            outputs_masks = F.interpolate(
                outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
            ).sigmoid()
            outputs_masks = (outputs_masks > self.threshold).byte().cpu().detach()

            out_masks = outputs_masks
            for i, (cur_mask, t) in enumerate(zip(out_masks, targets)):
                img_h, img_w = t["size"][0], t["size"][1]
                results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
                if self.rescale_to_orig_size:
                    results[i]["masks"] = F.interpolate(
                        results[i]["masks"].float(),
                        size=tuple(t["orig_size"].tolist()),
                        mode="nearest",
                    ).byte()

        return results


def build(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "lvis":
        num_classes = 1235
    if args.dataset_file == "coco_panoptic":
        num_classes = 250  # TODO: what is correct number? would be nice to refactor this anyways
    device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.mask_model != "none":
        model = DETRmask(model, mask_head=args.mask_model)
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    if args.giou_loss_coef:
        weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        # print(aux_weight_dict)
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
    )
    criterion.to(device)
    postprocessor = PostProcess().to(device)

    return model, criterion, postprocessor
