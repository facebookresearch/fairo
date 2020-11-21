"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import pickle
import numpy as np
import torch
from torch.utils import data as tds
import random


def get_rectanguloid_mask(y, fat=1):
    """Get a rectanguloid mask of the data
    """
    M = y.nonzero().max(0)[0].tolist()
    m = y.nonzero().min(0)[0].tolist()
    M = [min(M[i] + fat, y.shape[i] - 1) for i in range(3)]
    m = [max(v - fat, 0) for v in m]
    mask = torch.zeros_like(y)
    mask[m[0] : M[0], m[1] : M[1], m[2] : M[2]] = 1
    return mask


def underdirt(schematic, labels=None, max_shift=0, nothing_id=0):
    """Convert schematic to underdirt
    """
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
    """Randomly transform the cube for more data.

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
    """Add padding to schematics to sidelength
    """
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


# TODO cut outliers

# FIXME this should be better
def fit_in_sidelength(
    schematic, center_on_labels=False, labels=None, nothing_id=0, sl=32, max_shift=0
):
    """Adjust schematics to the center of the padded one
    """
    schematic, labels = pad_to_sidelength(
        schematic, labels=labels, nothing_id=nothing_id, sidelength=sl
    )
    if center_on_labels:
        nz = labels.nonzero()
    else:
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


def make_example_from_raw(
    schematic, labels=None, center_on_labels=False, augment={}, nothing_id=0, sl=32
):
    """Preprocess raw data and make good examples out of it
    """
    max_shift = augment.get("max_shift", 0)
    s, l, o = fit_in_sidelength(
        schematic,
        labels=labels,
        center_on_labels=center_on_labels,
        nothing_id=nothing_id,
        max_shift=max_shift,
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


class InstSegData(tds.Dataset):
    """Instance Segmentation Dataset out of raw data
    """
    def __init__(
        self,
        data_path,
        nexamples=-1,
        sidelength=32,
        classes=None,
        augment={},
        min_inst_size=1,
        mask_fat=1,
        useid=True,
    ):
        self.sidelength = sidelength
        self.useid = useid
        self.examples = []
        self.inst_data = pickle.load(open(data_path, "rb"))
        self.nexamples = nexamples
        self.augment = augment
        self.mask_fat = mask_fat
        if self.nexamples < 0:
            self.nexamples = len(self.inst_data)
        else:
            self.nexamples = min(len(self.inst_data), self.nexamples)

    def __getitem__(self, index):
        x = self.inst_data[index]
        has_label = [i for i in range(len(x[2])) if x[2][i] != "none"]
        i = random.choice(has_label)
        labels = (x[1] == i).astype("uint8")
        labels = torch.from_numpy(labels)
        s, l, o = make_example_from_raw(
            torch.from_numpy(x[0]),
            labels=labels,
            sl=self.sidelength,
            augment=self.augment,
            center_on_labels=True,
        )
        seed = random.choice(l.nonzero().tolist())
        seed_oh = l.clone().zero_()
        seed_oh[seed[0], seed[1], seed[2]] = 1
        mask = get_rectanguloid_mask(l, fat=self.mask_fat)
        return s, seed_oh, l, mask

    def __len__(self):
        return self.nexamples


# def drawme(s, islabel=False):
#    ss = s.clone()
#    if not islabel:
#        ss += 1
#        ss[ss == 1] = 0
#    else:
#        # fixme (4), also need to swap
#        ss[ss == 4] = 0
#    fig, ax = sp.draw((torch.stack([ss, ss.clone().zero_()], 3)).numpy(), 4, "yo")


if __name__ == "__main__":
    #    import sys
    #    import visdom

    #    sys.path.append("/private/home/aszlam/fairinternal/minecraft/python/craftassist/geoscorer/")
    #    import plot_voxels

    S = InstSegData("/checkpoint/aszlam/minecraft/segmentation_data/training_data.pkl")
#    viz = visdom.Visdom(server="http://localhost")
#    sp = plot_voxels.SchematicPlotter(viz)

#    def plot(i):
#        h = S[i]
#        z = torch.zeros(h[0].size()).long()
#        schematic = torch.stack([h[0], z], 3)
#        fig, ax = sp.draw(schematic.numpy(), 4, "yo")
#        return fig, ax, h
