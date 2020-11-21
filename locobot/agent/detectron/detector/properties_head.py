# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List
import torch
import numpy as np
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, Linear, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.registry import Registry

logger = logging.getLogger(__name__)

_TOTAL_SKIPPED = 0

ROI_PROPERTY_HEAD_REGISTRY = Registry("ROI_PROPERTY_HEAD")
ROI_PROPERTY_HEAD_REGISTRY.__doc__ = """
Registry for properties heads, which make property predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_properties_head(cfg, input_shape):
    """Build a properties head from `cfg.MODEL.ROI_PROPERTY_HEAD.NAME`."""
    name = cfg.MODEL.ROI_PROPERTY_HEAD.NAME
    return ROI_PROPERTY_HEAD_REGISTRY.get(name)(cfg, input_shape)


def property_rcnn_loss(pred_logits, instances, num_classes):
    gt_prop_labels = []
    for i in instances:
        # logger.info("instance gt_props {}".format(i.gt_props))
        gt_props = torch.zeros(i.gt_props.size(0), num_classes, device=i.gt_props.device).scatter_(
            1, i.gt_props, 1
        )
        for x in gt_props:
            x[0] = 0
        gt_prop_labels.append(gt_props)

    # https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573

    target = cat(gt_prop_labels, dim=0)
    loss = torch.nn.BCEWithLogitsLoss()(pred_logits, target)
    return {"loss_properties": loss}


def property_rcnn_inference(pred_logits, pred_instances):
    pred_prob = pred_logits.sigmoid()
    pred_instance = pred_instances[0]

    topk = []
    # FIXME @anuragprat1k picking top 3 properties for now.
    max_props = 3

    for x in pred_prob:
        curk = []
        for i in range(x.shape[0]):
            if x[i].item() > 0.5:
                curk.append((x[i].item(), i))
        curk.sort(key=lambda t: t[0])
        ik = [k[1] for k in curk[-min(max_props, len(curk)) :]]
        if not ik:
            ik = [0 for x in range(max_props)]
        topk.append(ik)

    pred_instance.pred_props = torch.tensor(topk)


@ROI_PROPERTY_HEAD_REGISTRY.register()
class BasicPropertiesRCNNHead(nn.Module):
    """Implement the basic properties R-CNN losses and inference logic."""

    def __init__(self, cfg, input_shape: ShapeSpec):
        """The following attributes are parsed from config:

        num_conv, num_fc: the number of conv/fc layers
        conv_dim/fc_dim: the dimension of the conv/fc layers
        norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_PROPERTY_HEAD.NUM_CLASSES
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        self.num_classes = cfg.MODEL.ROI_PROPERTY_HEAD.NUM_CLASSES
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances: List[Instances]):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))  # Apparently [Batch Size] * [Num classes]
        if self.training:
            return property_rcnn_loss(x, instances, self.num_classes)
        else:
            property_rcnn_inference(x, instances)
            return instances

    @property
    def output_size(self):
        return self._output_size
