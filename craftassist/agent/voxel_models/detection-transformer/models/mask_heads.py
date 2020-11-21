"""
Copyright (c) Facebook, Inc. and its affiliates.

This file provides the definition of the convolutional heads used to predict masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NestedTensor


class DETRmask(nn.Module):
    def __init__(self, detr, mask_head="v2"):
        super().__init__()
        self.detr = detr

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)

        if mask_head == "smallconv":
            maskHead = MaskHeadSmallConv
            mask_dim = hidden_dim + nheads
        elif mask_head == "v2":
            maskHead = MaskHeadV2
            mask_dim = hidden_dim
        else:
            raise RuntimeError(f"Unknown mask model {mask_head}")
        self.mask_head = maskHead(mask_dim, [256], hidden_dim)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[-1].tensors])
        outputs_seg_masks = seg_masks.view(
            bs,
            self.detr.num_queries,
            seg_masks.shape[-3],
            seg_masks.shape[-2],
            seg_masks.shape[-1],
        )

        out["pred_masks"] = outputs_seg_masks
        return out


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64,
        ]
        self.lay1 = torch.nn.Conv3d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv3d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.out_lay = torch.nn.Conv3d(inter_dims[1], 1, 3, padding=1)

        self.dim = dim

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1, 1).flatten(0, 1)

        # print(' @@@@ bbox mask')
        # print(bbox_mask.shape)
        # print(' @@@@ before maskhead size')
        # print(x.shape)
        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        x = self.out_lay(x)
        # print(' @@@@ after maskhead size')
        # print(x.shape)
        return x


class MaskHeadV2(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # inter_dims = [dim, context_dim // 4, context_dim // 16, context_dim // 64, context_dim // 128]
        inter_dims = [context_dim // 4, context_dim // 16, context_dim // 64, context_dim // 128]

        blocks = []
        adapters = []
        refiners = []
        in_dim = dim
        for i in range(2):
            out_dim = inter_dims[i]
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 3, padding=1),
                    # nn.GroupNorm(8, out_dim),
                    # nn.ReLU()
                )
            )
            adapters.append(nn.Conv2d(fpn_dims[i], out_dim, 1))
            refiners.append(
                nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GroupNorm(8, out_dim), nn.ReLU()
                )
            )
            in_dim = out_dim

        self.blocks = nn.ModuleList(blocks)
        self.adapters = nn.ModuleList(adapters)
        self.refiners = nn.ModuleList(refiners)
        self.out_lay = nn.Conv2d(in_dim, 1, 3, padding=1)

        self.dim = dim

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        # bbox_mask = bbox_mask.mean(2)
        bs, num_queries, num_heads = bbox_mask.shape[:3]
        for fpn, block, adapter, refiner in zip(fpns, self.blocks, self.adapters, self.refiners):
            x = block(x)
            adapted_fpn = adapter(fpn)
            x = F.interpolate(x, size=adapted_fpn.shape[-2:], mode="nearest")
            x = x.reshape((bs, -1) + x.shape[1:]) + adapted_fpn[:, None]
            mask = F.interpolate(bbox_mask.flatten(1, 2), size=x.shape[-2:], mode="bilinear")
            mask = mask.reshape((bs, num_queries, num_heads) + mask.shape[-2:])
            x = x.reshape((bs, -1, num_heads, x.shape[2] // num_heads) + x.shape[3:])
            x = x * mask[:, :, :, None]
            x = x.flatten(2, 3)
            x = x.flatten(0, 1)
            x = refiner(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        # q: (bs, num_queries, hidden_dim)
        # k: (bs, hiddem_dim, h, w, d)
        q = self.q_linear(q)
        k = F.conv3d(
            k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), self.k_linear.bias
        )
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(
            k.shape[0],
            self.num_heads,
            self.hidden_dim // self.num_heads,
            k.shape[-3],
            k.shape[-2],
            k.shape[-1],
        )
        weights = torch.einsum("bqnc,bnchwd->bqnhwd", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights
