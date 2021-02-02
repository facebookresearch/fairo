"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import torch
import torch.nn as nn
from data_loaders import make_example_from_raw


def conv3x3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3x3 convolution with padding
    """
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv3x3x3up(in_planes, out_planes, bias=True):
    """3x3x3 convolution with padding
    """
    return nn.ConvTranspose3d(
        in_planes, out_planes, stride=2, kernel_size=3, padding=1, output_padding=1
    )


def convbn(in_planes, out_planes, stride=1, bias=True):
    """3x3x3 convolution with batch norm and relu
    """
    return nn.Sequential(
        (conv3x3x3(in_planes, out_planes, stride=stride, bias=bias)),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True),
    )


def convbnup(in_planes, out_planes, bias=True):
    """3x3x3 convolution with batch norm and relu
    """
    return nn.Sequential(
        (conv3x3x3up(in_planes, out_planes, bias=bias)),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True),
    )


class InstSegNet(nn.Module):
    """Basic Instance Segmentation Neural Network
    """

    def __init__(self, opts):
        super(InstSegNet, self).__init__()
        if opts.load:
            if opts.load_model != "":
                self.load(opts.load_model)
            else:
                raise ("loading from file specified but no load_filepath specified")
        else:
            self.opts = opts
            self._build()

    def forward(self, x):
        raise NotImplementedError

    def save(self, filepath):
        self.cpu()
        sds = {}
        sds["opts"] = self.opts
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
        self.load_state_dict(sds["state_dict"])
        self.zero_grad()


class FlatInstSegNet(InstSegNet):
    """Flat Instance Segmentation Neural Network
    """

    def __init__(self, opts):
        super(FlatInstSegNet, self).__init__(opts)

    def _build(self):
        opts = self.opts
        embedding_dim = getattr(opts, "embedding_dim", 8)
        num_words = getattr(opts, "num_words", 255)
        num_layers = getattr(opts, "num_layers", 4)
        hidden_dim = getattr(opts, "hidden_dim", 64)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(embedding_dim + 1, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        )
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dim + 1, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.out = nn.Conv3d(hidden_dim, 1, kernel_size=5, padding=2)

    def forward(self, x, seed_oh):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        for i in range(self.num_layers):
            z = torch.cat((z, seed_oh), 1)
            z = self.layers[i](z)
        return self.out(z).squeeze()


# num_scales = 3:
#                   o --> o
#                   ^     |
#                   |     v
#             o --> o --> o
#             ^           |
#             |           v
#       o --> o --> o --> o
#       ^                 |
#       |                 v
# o --> o --> o --> o --> o --> o --> o --> o --> o
# *     *     *     *     *
#


class MsInstSegNet(InstSegNet):
    """Multi-scale Instance Segmentation Neural Network
    """

    def __init__(self, opts):
        super(MsInstSegNet, self).__init__(opts)

    def _build(self):
        opts = self.opts
        embedding_dim = getattr(opts, "embedding_dim", 8)
        num_words = getattr(opts, "num_words", 255)
        num_layers_per_scale = getattr(opts, "num_layers_per_scale", 1)
        hidden_dim = getattr(opts, "hidden_dim", 64)
        num_scales = getattr(opts, "num_scales", 3)
        num_cleanup = getattr(opts, "num_cleanup", 3)

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.start = convbn(embedding_dim + 1, hidden_dim)
        self.scales = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.cleanup = nn.ModuleList()
        for i in range(num_scales):
            scale = nn.ModuleList()
            if i != 0:
                self.downsamplers.append(convbn(hidden_dim, hidden_dim, stride=2))
                self.upsamplers.append(convbnup(hidden_dim, hidden_dim))
            for j in range(num_layers_per_scale * (num_scales - i)):
                d = hidden_dim
                e = d
                if i == 0:
                    e = e + 1  # keep the seed around
                scale.append(convbn(e, d))
            self.scales.append(scale)

        for i in range(num_cleanup):
            self.cleanup.append(convbn(hidden_dim, hidden_dim))

        self.out = nn.Conv3d(hidden_dim, 1, kernel_size=5, padding=2)

    def forward(self, x, seed_oh):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        nscales = len(self.scales)
        szs = list(x.size())
        x = x.view(-1)
        z = self.embedding.weight.index_select(0, x)
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        z = self.start(torch.cat((z, seed_oh), 1))
        scales = []
        v = 0  # flake...
        for i in range(nscales):
            if i > 0:
                u = self.downsamplers[i - 1](v)
            else:
                u = z
            for j in range(len(self.scales[i])):
                m = self.scales[i][j]
                if i == 0:
                    u = torch.cat((u, seed_oh), 1)
                u = m(u)
                if j == 0:
                    v = u.clone()
            scales.append(u)
        for i in range(nscales - 2, -1, -1):
            scales[i] = scales[i] + self.upsamplers[i](scales[i + 1])
        z = scales[0]
        for m in self.cleanup:
            z = m(z)
        return self.out(z).squeeze()


class Opt:
    pass


############################NOT DONE!!!!!
class InstSegWrapper:
    """Wrapper for Instance Segmentation Net
    """

    def __init__(self, model, threshold=-1.0, blocks_only=True, cuda=False):
        if type(model) is str:
            opts = Opt()
            opts.load = True
            opts.load_model = model
            model = InstSegNet(opts)
        self.model = model
        self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()

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
    args.load = False

    N = MsInstSegNet(args)
