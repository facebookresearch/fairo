"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import csv
import numpy as np


SHAPENET_PATH = ""  # add path to shapenet csv here


def read_shapenet_csv(csvpath=SHAPENET_PATH + "metadata.csv"):
    keyword_to_id = {}
    id_to_keyword = {}
    with open(csvpath, "r") as cfile:
        cvs_metadata = csv.reader(cfile)
        next(cvs_metadata)
        for l in cvs_metadata:
            bin_id = l[0][4:]
            keywords = l[3].split(",")
            if keywords[0] == "":
                continue
            id_to_keyword[bin_id] = keywords
            for k in keywords:
                if keyword_to_id.get(k) is None:
                    keyword_to_id[k] = []
                else:
                    keyword_to_id[k].append(bin_id)
    return keyword_to_id, id_to_keyword


def to_relative_pos(block_list):
    """Convert absolute block positions to their relative positions

    Find the "origin", i.e. the minimum (x, y, z), and subtract this from all
    block positions.

    Args:
    - block_list: a list of ((x,y,z), (id, meta))

    Returns:
    - a block list where positions are shifted by `origin`
    - `origin`, the (x, y, z) offset by which the positions were shifted
    """
    try:
        locs, idms = zip(*block_list)
    except ValueError:
        raise ValueError("to_relative_pos invalid input: {}".format(block_list))

    locs = np.array([loc for (loc, idm) in block_list])
    origin = np.min(locs, axis=0)
    locs -= origin
    S = [(tuple(loc), idm) for (loc, idm) in zip(locs, idms)]
    if type(block_list) is not list:
        S = tuple(S)
    if type(block_list) is frozenset:
        S = frozenset(S)
    return S, origin


def blocks_list_to_npy(blocks, xyz=False):
    """Convert a list of blockid meta (x, y, z), (id, meta) to numpy"""
    xyzbm = np.array([(x, y, z, b, m) for ((x, y, z), (b, m)) in blocks])
    mx, my, mz = np.min(xyzbm[:, :3], axis=0)
    Mx, My, Mz = np.max(xyzbm[:, :3], axis=0)

    npy = np.zeros((My - my + 1, Mz - mz + 1, Mx - mx + 1, 2), dtype="int32")

    for x, y, z, b, m in xyzbm:
        npy[y - my, z - mz, x - mx] = (b, m)

    offsets = (my, mz, mx)

    if xyz:
        npy = np.swapaxes(np.swapaxes(npy, 1, 2), 0, 1)
        offsets = (mx, my, mz)

    return npy, offsets


def blocks_list_add_offset(blocks, origin):
    """Offset all blocks in block list by a constant xyz

    Args:
      blocks: a list[(xyz, idm)]
      origin: xyz

    Returns list[(xyz, idm)]
    """
    ox, oy, oz = origin
    return [((x + ox, y + oy, z + oz), idm) for ((x, y, z), idm) in blocks]
