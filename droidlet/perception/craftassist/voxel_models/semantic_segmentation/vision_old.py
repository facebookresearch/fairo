"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import torch
import torch.nn as nn
from data_loaders import make_example_from_raw

from droidlet.lowlevel.minecraft.small_scenes_with_shapes import SL, H

BERT_HIDDEN_DIM = 1  # 768


class SemSegNet(nn.Module):
    """Semantic Segmentation Neural Network"""

    def __init__(self, opts, classes=None):
        super(SemSegNet, self).__init__()
        if opts.load:
            if opts.load_model != "":
                print(f"Loading pretrained semseg model...")
                self.load(opts.load_model)
            else:
                raise ("loading from file specified but no load_filepath specified")
        else:
            self.opts = opts
            self._build()
            self.classes = classes

    def _build(self):
        opts = self.opts
        try:
            embedding_dim = opts.embedding_dim
        except:
            embedding_dim = 8
        try:
            num_words = opts.num_words
        except:
            num_words = 3
        try:
            num_layers = opts.num_layers
        except:
            num_layers = 4
        try:
            hidden_dim = opts.hidden_dim
        except:
            hidden_dim = 64

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(embedding_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.linear = nn.Linear(hidden_dim + BERT_HIDDEN_DIM, 1)
        # self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        szs = list(x.size())  # B x SL x SL x SL
        B = szs[0]
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()

        for i in range(self.num_layers):
            z = self.layers[i](z)

        t = (
            t.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, SL, SL, SL)
        )  # B x TE x SL x SL x SL
        TE = t.size(1)
        H = z.size(1)
        # print(f"voxel embed norm: {torch.norm(z[0])}, bert embed norm: {torch.norm(t[0])}")
        z = torch.cat((z, t), 1)  # B x (TE + H) x SL x SL x SL

        z = z.permute(0, 2, 3, 4, 1).contiguous()  # B x SL x SL x SL x (TE + H)
        z = self.linear(z).permute(0, 4, 1, 2, 3)  # B x (TE + H) x SL x SL x SL
        z = self.sigmoid(z).squeeze()
        return z

    def save(self, filepath):
        self.cpu()
        sds = {}
        sds["opts"] = self.opts
        sds["classes"] = self.classes
        sds["state_dict"] = self.state_dict()
        torch.save(sds, filepath)
        if self.opts.cuda:
            self.cuda()

    def load(self, filepath):
        sds = torch.load(filepath)
        self.opts = sds["opts"]
        print("loading from file, using opts")
        print(self.opts)
        self._build()
        self.load_state_dict(sds["state_dict"], strict=False)
        self.zero_grad()
        self.classes = sds["classes"]


class Opt:
    pass


class SemSegWrapper:
    """Wrapper for Semantic Segmentation Net"""

    def __init__(self, model, threshold=-1.0, blocks_only=True, cuda=False):
        if type(model) is str:
            opts = Opt()
            opts.load = True
            opts.load_model = model
            model = SemSegNet(opts)
        self.model = model
        self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        self.classes = model.classes
        # threshold for relevance; unused rn
        self.threshold = threshold
        # if true only label non-air blocks
        self.blocks_only = blocks_only
        # this is used by the semseg_process
        i2n = self.classes["idx2name"]
        self.tags = [(c, self.classes["name2count"][c]) for c in i2n]
        assert self.classes["name2idx"]["none"] == 0

    @torch.no_grad()
    def segment_object(self, blocks):
        self.model.eval()
        blocks = torch.from_numpy(blocks)[:, :, :, 0]
        blocks, _, o = make_example_from_raw(blocks)
        blocks = blocks.unsqueeze(0)
        if self.cuda:
            blocks = blocks.cuda()
        y = self.model(blocks)
        _, mids = y.squeeze().max(0)
        locs = mids.nonzero()
        locs = locs.tolist()
        if self.blocks_only:
            return {
                tuple(np.subtract(l, o)): mids[l[0], l[1], l[2]].item()
                for l in locs
                if blocks[0, l[0], l[1], l[2]] > 0
            }
        else:
            return {tuple(ll for ll in l): mids[l[0], l[1], l[2]].item() for l in locs}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of hidden dim in fc layer"
    )
    parser.add_argument("--embedding_dim", type=int, default=16, help="size of blockid embedding")
    parser.add_argument("--num_words", type=int, default=256, help="number of blocks")
    parser.add_argument("--num_classes", type=int, default=20, help="number of blocks")

    args = parser.parse_args()

    N = SemSegNet(args)
