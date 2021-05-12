"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import torch
import torch.nn as nn


def underdirt(schematic, labels=None, max_shift=0, nothing_id=0):
    # todo fancier dirt!
    # FIXME!!!! label as ground where appropriate
    shift = torch.randint(max_shift + 1, (1,)).item()
    if shift > 0:
        new_schematic = torch.LongTensor(schematic.size())
        new_schematic[:, shift:, :] = schematic[:, :-shift, :]
        new_schematic[:, :shift, :] = 3
        new_labels = None
        if labels is not None:
            new_labels = torch.LongTensor(labels.size())
            new_labels[:, shift:, :] = labels[:, :-shift, :]
            new_labels[:, :shift, :] = nothing_id
        return new_schematic, new_labels
    else:
        return schematic, labels


def flip_rotate(c, l=None, idx=None):
    """
    Randomly transform the cube for more data.
    The transformation is chosen from:
        0. original
        1. x-z plane rotation 90
        2. x-z plane rotation 180
        3. x-z plane rotation 270
        4. x-axis flip
        5. z-axis flip
    """
    idx = np.random.choice(range(6)) if (idx is None) else idx
    l_ = l
    if idx == 0:
        c_ = c
        l_ = l
    elif idx >= 1 and idx <= 3:  # rotate
        npc = c.numpy()
        npc = np.rot90(npc, idx, axes=(0, 2))  # rotate on the x-z plane
        c_ = torch.from_numpy(npc.copy())
        if l is not None:
            npl = l.numpy()
            npl = np.rot90(npl, idx, axes=(0, 2))  # rotate on the x-z plane
            l_ = torch.from_numpy(npl.copy())
    else:  # flip
        npc = c.numpy()
        npc = np.flip(npc, axis=(idx - 4) * 2)  # 0 or 2
        c_ = torch.from_numpy(npc.copy())
        if l is not None:
            npl = l.numpy()
            npl = np.flip(npl, axis=(idx - 4) * 2)  # 0 or 2
            l_ = torch.from_numpy(npl.copy())
    return c_, l_, idx


def pad_to_sidelength(schematic, labels=None, nothing_id=0, sidelength=32):
    szs = list(schematic.size())
    szs = np.add(szs, -sidelength)
    pad = []
    # this is all backwards bc pytorch pad semantics :(
    for s in szs:
        if s >= 0:
            pad.append(0)
        else:
            pad.append(-s)
        pad.append(0)
    schematic = torch.nn.functional.pad(schematic, pad[::-1])
    if labels is not None:
        labels = torch.nn.functional.pad(labels, pad[::-1], value=nothing_id)
    return schematic, labels


# TODO simplify
def fit_in_sidelength(schematic, labels=None, nothing_id=0, sl=32, max_shift=0):
    schematic, labels = pad_to_sidelength(
        schematic, labels=labels, nothing_id=nothing_id, sidelength=sl
    )
    nz = schematic.nonzero()
    m, _ = nz.median(0)
    min_y, _ = nz.min(0)
    min_y = min_y[1]
    xshift = max(torch.randint(-max_shift, max_shift + 1, (1,)).item() - m[0].item() + sl // 2, 0)
    zshift = max(torch.randint(-max_shift, max_shift + 1, (1,)).item() - m[2].item() + sl // 2, 0)
    new_schematic = torch.LongTensor(sl, sl, sl).fill_(1)
    new_schematic[xshift:, : sl - min_y, zshift:] = schematic[
        : sl - xshift, min_y:sl, : sl - zshift
    ]
    new_labels = None
    if labels is not None:
        new_labels = torch.LongTensor(sl, sl, sl).fill_(nothing_id)
        new_labels[xshift:, : sl - min_y, zshift:] = labels[: sl - xshift, min_y:sl, : sl - zshift]
    return new_schematic, new_labels, (xshift, -min_y, zshift)


def make_example_from_raw(schematic, labels=None, augment={}, nothing_id=0, sl=32):

    max_shift = augment.get("max_shift", 0)
    s, l, o = fit_in_sidelength(
        schematic, labels=labels, nothing_id=nothing_id, max_shift=max_shift
    )
    if len(augment) > 0:
        if augment.get("flip_rotate", False):
            s, l, _ = flip_rotate(s, l=l)
        m = augment.get("underdirt")
        if m is not None:
            # really should fix offset here.....TODO
            s, l = underdirt(s, labels=l, max_shift=m, nothing_id=nothing_id)
    s[s == 0] = 1
    s -= 1
    return s, l, o


class SemSegNet(nn.Module):
    def __init__(self, classes=None):
        super(SemSegNet, self).__init__()
        # if opts.load:
        #     if opts.load_model != "":
        #         self.load(opts.load_model)
        #     else:
        #         raise ("loading from file specified but no load_filepath specified")
        # else:
        #     self._build()
        #     self.classes = classes
        self._build()
        self.classes = classes

    def _build(self):
        try:
            embedding_dim = 4
        except:
            embedding_dim = 8
        try:
            num_words = 256
        except:
            num_words = 3
        try:
            num_layers = 4
        except:
            num_layers = 4  # 32x32x32 input
        try:
            hidden_dim = 128
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
            if i == 0:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                        nn.BatchNorm3d(hidden_dim),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                        nn.BatchNorm3d(hidden_dim),
                        nn.ReLU(inplace=True),
                    )
                )
        # self.out = nn.Conv3d(hidden_dim, opts.num_classes, kernel_size=1)
        # self.lsm = nn.LogSoftmax(dim=1)

    # def forward(self, x):
    #     shape = list(x.size())
    #     shape.insert(1, 128)
    #     ret = torch.zeros(shape).cuda() + 0.5
    #     return ret

    def forward(self, x):
        # FIXME when pytorch is ready for this, embedding
        # backwards is soooooo slow
        # z = self.embedding(x)
        # print('x size==> {}'.format(x.size()))
        szs = list(x.size())
        x = x.view(-1)
        # print('x view size==> {}'.format(x.size()))
        # print('embed size==> {}'.format(self.embedding.weight.size()))
        z = self.embedding.weight.index_select(0, x)
        # print('z size==> {}'.format(z.size()))
        szs.append(self.embedding_dim)
        z = z.view(torch.Size(szs))
        # print('z view size==> {}'.format(z.size()))
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        # print('z permute size==> {}'.format(z.size()))
        for i in range(self.num_layers):
            z = self.layers[i](z)
            # print('layer {} : z fc after size==> {}'.format(i, z.size()))
        # out = self.out(z)
        # print('out size==> {}'.format(out.size()))
        # rtr = self.lsm(out)
        # print('return size==> {}'.format(z.size()))
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
        self.load_state_dict(sds["state_dict"])
        self.zero_grad()
        self.classes = sds["classes"]


class Opt:
    pass


class SemSegWrapper:
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
