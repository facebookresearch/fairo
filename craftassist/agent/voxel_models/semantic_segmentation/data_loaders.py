"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import pickle
import numpy as np
import torch
from torch.utils import data as tds
from copy import deepcopy


def underdirt(schematic, labels=None, max_shift=0, nothing_id=0):
    """Convert schematic to underdirt"""
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
    """Add padding to schematics to sidelength"""
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

# TODO simplify
def fit_in_sidelength(schematic, labels=None, nothing_id=0, sl=32, max_shift=0):
    """Adjust schematics to the center of the padded one"""
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
    """Preprocess raw data and make good examples out of it"""
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


def swallow_classes(classes, predator, prey_classes, class_map):
    """Swallow classes"""
    new_classes = deepcopy(classes)
    apex = class_map.get(predator, predator)
    for prey in prey_classes:
        class_map[prey] = apex
        new_classes["name2count"][predator] += new_classes["name2count"][prey]
        del new_classes["name2count"][prey]
    for prey in prey_classes:
        for s, t in class_map.items():
            if t == prey:
                class_map[s] = apex
    return new_classes, class_map


def organize_classes(classes, min_occurence):
    """Organize classes"""
    class_map = {}
    new_classes = deepcopy(classes)
    for cname in classes["name2count"]:
        # hacky, should stem this properly
        if cname[-1] == "s" and classes["name2count"].get(cname[:-1]) is not None:
            new_classes, class_map = swallow_classes(new_classes, cname[:-1], [cname], class_map)
    small_classes = []
    for cname, count in new_classes["name2count"].items():
        if count < min_occurence:
            small_classes.append(cname)
    new_classes, class_map = swallow_classes(new_classes, "none", small_classes, class_map)
    new_classes, class_map = swallow_classes(new_classes, "none", ["nothing"], class_map)
    counts = sorted(list(new_classes["name2count"].items()), key=lambda x: x[1], reverse=True)
    new_classes["name2idx"]["none"] = 0
    new_classes["idx2name"].append("none")
    for i in range(len(counts)):
        cname = counts[i][0]
        if cname != "none":
            new_classes["name2idx"][cname] = i
            new_classes["idx2name"].append(cname)

    return new_classes, class_map


class SemSegData(tds.Dataset):
    """Semantic Segmentation Dataset out of raw data"""

    def __init__(
        self,
        data_path,
        nexamples=-1,
        sidelength=32,
        classes=None,
        augment={},
        min_class_occurence=250,
        useid=True,
    ):
        self.sidelength = sidelength
        self.useid = useid
        self.examples = []
        self.inst_data = pickle.load(open(data_path, "rb"))
        self.nexamples = nexamples
        self.augment = augment
        if self.nexamples < 0:
            self.nexamples = len(self.inst_data)
        else:
            self.nexamples = min(len(self.inst_data), self.nexamples)
        if classes is None:
            classes = {"name2idx": {}, "idx2name": [], "name2count": {}}
            for i in range(len(self.inst_data)):
                for cname in self.inst_data[i][2]:
                    if classes["name2count"].get(cname) is None:
                        classes["name2count"][cname] = 1
                    else:
                        classes["name2count"][cname] += 1
        if classes["name2count"].get("none") is None:
            classes["name2count"]["none"] = 1
        merged_classes, class_map = organize_classes(classes, min_class_occurence)
        for cname in merged_classes["name2idx"]:
            class_map[cname] = cname
        self.classes = merged_classes
        # this should be 0...
        self.nothing_id = self.classes["name2idx"]["none"]

        c = self.classes["name2idx"]
        for i in range(len(self.inst_data)):
            self.inst_data[i] = list(self.inst_data[i])
            x = self.inst_data[i]
            x[0] = torch.from_numpy(x[0]).long()
            x[1] = torch.from_numpy(x[1]).long()
            x[1].apply_(lambda z: c[class_map[x[2][z]]] if z > 0 else self.nothing_id)

    def get_classes(self):
        return self.classes

    def set_classes(self, classes):
        self.classes = classes

    def __getitem__(self, index):
        x = self.inst_data[index]
        s, l, _ = make_example_from_raw(
            x[0], labels=x[1], nothing_id=self.nothing_id, sl=self.sidelength, augment=self.augment
        )
        return s, l

    def __len__(self):
        return self.nexamples
