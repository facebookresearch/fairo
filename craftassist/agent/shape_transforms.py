"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from collections import Counter
from build_utils import blocks_list_to_npy  # , npy_to_blocks_list

##############################################
# WARNING: all npy arrays in this file are xyz
# not yzx


def maybe_convert_to_npy(blocks):
    """Convert a list of blocks to numpy array"""
    if type(blocks) is list:
        blocks, _ = blocks_list_to_npy(blocks, xyz=True)
        return blocks
    else:
        assert blocks.shape[-1] == 2
        assert len(blocks.shape) == 4
        return blocks.copy()


def maybe_convert_to_list(blocks):
    """Convert blocks to a list"""
    if type(blocks) is list:
        return blocks.copy()
    else:
        nz = np.transpose(blocks[:, :, :, 0].nonzero())
        return [(tuple(loc), tuple(blocks[tuple(loc)])) for loc in nz]


def flint(x):
    return int(np.floor(x))


def ceint(x):
    return int(np.ceil(x))


def check_boundary(p, m, M):
    if (
        p[0] == m[0]
        or p[0] == M[0] - 1
        or p[1] == m[1]
        or p[1] == M[1] - 1
        or p[2] == m[2]
        or p[2] == M[2] - 1
    ):
        return True
    else:
        return False


def reshift(shape):
    m = np.min([l[0] for l in shape], axis=0)
    return [((b[0][0] - m[0], b[0][1] - m[1], b[0][2] - m[2]), b[1]) for b in shape]


def moment_at_center(npy, sl):
    """
    shifts the object in the 4d numpy array so that the center of mass is at sl//2
    in a sl x sl x sl x 2 array
    warning, this will cut anything that is too big to fit in sl x sl x sl
    and then the moment might not actually be in center.
    """
    nz = np.transpose(npy[:, :, :, 0].nonzero())
    mins = np.min(nz, axis=0)
    shifted_nz = nz - mins
    com = np.floor(np.array(shifted_nz.mean(axis=0)))
    # this will fail if com is bigger than sl.
    assert all(com < sl)
    npy_out_center = np.array((sl // 2, sl // 2, sl // 2))
    shifted_nz = (shifted_nz - com + npy_out_center).astype("int32")
    npy_out = np.zeros((sl, sl, sl, 2), dtype="int32")
    for i in range(nz.shape[0]):
        if all(shifted_nz[i] >= 0) and all(shifted_nz[i] - sl < 0):
            npy_out[tuple(shifted_nz[i])] = npy[tuple(nz[i])]
    return npy_out


#############################################
## THICKEN
#############################################

# this doesn't preserve corners.  should it?
# separate deltas per dim?
def thicker_blocks(blocks, delta=1):
    """Takes a list of blocks and thickens them
    by an amount equal to delta"""
    newblocks = {l: idm for (l, idm) in blocks}
    for b in blocks:
        for dx in range(-delta, delta + 1):
            for dy in range(-delta, delta + 1):
                for dz in range(-delta, delta + 1):
                    l = b[0]
                    newblocks[(l[0] + dx, l[1] + dy, l[2] + dz)] = b[1]
    return list(newblocks.items())


def thicker(blocks, delta=1):
    """
    Returns:
        numpy array of blocks thickened with an amount delta
    """
    blocks = maybe_convert_to_list(blocks)
    newblocks = thicker_blocks(blocks, delta=delta)
    npy, _ = blocks_list_to_npy(newblocks, xyz=True)
    return npy


#############################################
## SCALE
#############################################


def get_loc_weight(idx, cell_size):
    """compute the scaled indices and amount in 1d they
    extend on either side of the block boundary
    """
    left = idx * cell_size
    right = (idx + 1) * cell_size
    lidx = int(np.floor(left))
    ridx = int(np.floor(right))
    if ridx > lidx:
        right_weight = right - ridx
        left_weight = ridx - left
    else:
        right_weight = 0
        left_weight = 1
    return (lidx, ridx), (left_weight, right_weight)


def get_cell_weights(idxs, cell_szs):
    """compute the amount of the cell in each of
    the 8 cubes it might touch"""
    index = []
    dw = []
    for k in range(3):
        i, w = get_loc_weight(idxs[k], cell_szs[k])
        index.append(i)
        dw.append(w)
    cell_weights = np.zeros((2, 2, 2))
    best_cell = None
    big_weight = 0.0
    total_weight = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = dw[0][i] * dw[1][j] * dw[2][k]
                cell_weights[i, j, k] = w
                total_weight += w
                if w > big_weight:
                    big_weight = w
                    best_cell = (index[0][i], index[1][j], index[2][k])
    cell_weights = cell_weights / total_weight
    return best_cell, index, cell_weights


def scale(blocks, lams=(1.0, 1.0, 1.0)):
    """scales the blockobject in the ith direction with factor lams[i]
    algorithm is to first scale the blocks up (so each minecraft cube has
    size lams), and then for each 1x1x1 block arranged in place assign it
    the id, meta of the big block it most intersects
    """
    assert lams[0] >= 1.0  # eventually FIXME?
    assert lams[1] >= 1.0  # eventually FIXME?
    assert lams[2] >= 1.0  # eventually FIXME?
    inp = maybe_convert_to_npy(blocks)
    szs = np.array(inp.shape[:3])
    big_szs = np.ceil(szs * lams)
    cell_szs = szs / big_szs
    big_szs = big_szs.astype("int32")
    big = np.zeros(tuple(big_szs) + (2,)).astype("int32")
    for i in range(big_szs[0]):
        for j in range(big_szs[1]):
            for k in range(big_szs[2]):
                best_cell, _, _ = get_cell_weights((i, j, k), cell_szs)
                big[i, j, k, :] = inp[best_cell]
    return big


def scale_sparse(blocks, lams=(1.0, 1.0, 1.0)):
    """scales the blockobject in the ith direction with factor lams[i]
    algorithm is to first scale the blocks up (so each minecraft cube has
    size lams), and then for each 1x1x1 block arranged in place assign it
    the id, meta of the big block it most intersects
    """
    assert lams[0] >= 1.0  # eventually FIXME?
    assert lams[1] >= 1.0  # eventually FIXME?
    assert lams[2] >= 1.0  # eventually FIXME?
    inp = maybe_convert_to_list(blocks)
    locs = [l for (l, idm) in inp]
    m = np.min(locs, axis=0)
    inp_dict = {(l[0] - m[0], l[1] - m[1], l[2] - m[2]): idm for (l, idm) in inp}
    szs = np.max(locs, axis=0) - np.min(locs, axis=0) + 1
    big_szs = np.ceil(szs * lams)
    cell_szs = szs / big_szs
    big_szs = big_szs.astype("int32")
    big = np.zeros(tuple(big_szs) + (2,)).astype("int32")
    for (x, y, z) in inp_dict.keys():
        for i in range(flint(x * lams[0]), ceint(x * lams[0]) + 2):
            for j in range(flint(y * lams[1]), ceint(y * lams[1]) + 2):
                for k in range(flint(z * lams[2]), ceint(z * lams[2]) + 2):
                    if i < big_szs[0] and j < big_szs[1] and k < big_szs[2]:
                        best_cell, _, _ = get_cell_weights((i, j, k), cell_szs)
                        idm = inp_dict.get(best_cell)
                        if idm:
                            big[i, j, k, :] = idm
                        else:
                            big[i, j, k, :] = (0, 0)
    return big


def shrink_sample(blocks, lams):
    """Shrink the blocks with dimensions in lams"""
    assert lams[0] <= 1.0
    assert lams[1] <= 1.0
    assert lams[2] <= 1.0
    blocks = maybe_convert_to_npy(blocks)
    szs = blocks.shape
    xs = np.floor(np.arange(0, szs[0], 1 / lams[0])).astype("int32")
    ys = np.floor(np.arange(0, szs[1], 1 / lams[1])).astype("int32")
    zs = np.floor(np.arange(0, szs[2], 1 / lams[2])).astype("int32")
    small = np.zeros((len(xs), len(ys), len(zs), 2), dtype="int32")
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(zs)):
                small[i, j, k] = blocks[xs[i], ys[j], zs[k]]
    return small


#############################################
## ROTATE
#############################################


def rotate(blocks, angle=0, mirror=-1, plane="xz"):
    """Rotate a list of blocks by an angle 'angle' along
    the plane given by 'plane'.
    If 'mirror' is > 0, a mirror image of the blocks is returned
    Returns:
        A rotated list of blocks
    """
    inp = maybe_convert_to_npy(blocks)
    if mirror > 0:
        inp = np.flip(inp, mirror)
    # maybe generalize?
    assert angle % 90 == 0
    i = angle // 90
    if i < 0:
        i = i % 4
    if plane == "xz" or plane == "zx":
        return np.rot90(inp, i, axes=(0, 2))
    elif plane == "xy" or plane == "yx":
        return np.rot90(inp, i, axes=(0, 1))
    else:
        return np.rot90(inp, i, axes=(1, 2))


#############################################
## REPLACE
#############################################


def hash_idm(npy):
    return npy[:, :, :, 0] + 1000 * npy[:, :, :, 1]


def unhash_idm(npy):
    npy = npy.astype("int32")
    b = npy % 1000
    m = (npy - b) // 1000
    return np.stack((b, m), axis=3)


# TODO current_idm should be a list
def replace_by_blocktype(blocks, new_idm=(0, 0), current_idm=None, every_n=1, replace_every=False):
    """replace some blocks with a different kind
    note that it is allowed that new_idm is (0,0)
    """
    if current_idm is not None:  # specifying a transformation of one blocktype to another
        blocks = maybe_convert_to_npy(blocks)
        h = hash_idm(blocks)
        u = h.copy()
        old_idm_hash = current_idm[0] + 1000 * current_idm[1]
        new_idm_hash = new_idm[0] + 1000 * new_idm[1]
        u[u == old_idm_hash] = new_idm_hash
        out = unhash_idm(u)
    else:  # TODO FIXME need better algorithm here
        if every_n == 1 and not replace_every:
            lblocks = maybe_convert_to_list(blocks)
            mode = Counter([idm for loc, idm in lblocks]).most_common(1)[0][0]
            for b in lblocks:
                if b[1] == mode:
                    b[1] = new_idm
            return maybe_convert_to_npy(lblocks)

        blocks = maybe_convert_to_npy(blocks)
        out = blocks.copy()
        if type(every_n) is int:
            every_n = (every_n, every_n, every_n)
        nzmask = blocks[:, :, :, 0] > 0
        every_n_mask = nzmask.copy()
        every_n_mask[:] = False
        every_n_mask[:: every_n[0], :: every_n[0], :: every_n[0]] = True
        mask = np.logical_and(every_n_mask, nzmask)
        out_b = out[:, :, :, 0]
        out_b[mask] = new_idm[0]
        out_m = out[:, :, :, 1]
        out_m[mask] = new_idm[1]
    return out


def replace_by_halfspace(blocks, new_idm=(0, 0), geometry=None, replace_every=False):
    """replace some blocks with a different kind, chosen by
    block at (i,j,k) is changed if
    geometry['v']@((i,j,k) - geometry['offset']) > geometry['threshold']
    note that it is allowed that new_idm is (0, 0)
    """
    if not replace_every:
        lblocks = maybe_convert_to_list(blocks)
        mode = Counter([idm for loc, idm in lblocks]).most_common(1)[0][0]
    else:
        mode = None
    blocks = maybe_convert_to_npy(blocks)
    if not geometry:
        geometry = {"v": np.ndarray((0, 1, 0)), "threshold": 0, "offset": blocks.shape // 2}

    out = blocks.copy()
    szs = blocks.shape
    for i in range(szs[0]):
        for j in range(szs[1]):
            for k in range(szs[2]):
                if blocks[i, j, k, 0] > 0:
                    if (np.array((i, j, k)) - geometry["offset"]) @ geometry["v"] > geometry[
                        "threshold"
                    ]:
                        if replace_every or tuple(out[i, j, k, :]) == mode:
                            out[i, j, k, :] = new_idm

    return out


#############################################
## FILL
#############################################


def maybe_update_extreme_loc(extremes, loc, axis):
    """given a non-air index loc into a block list, updates
    a list of max and min values along axis at projections
    given by the non-axis entries in loc
    """
    other_indices = list(range(3))[:axis] + list(range(3))[axis + 1 :]
    loc_in = (loc[other_indices[0]], loc[other_indices[1]])
    loc_out = loc[axis]
    current = extremes.get(loc_in)
    if current is None:
        extremes[loc_in] = (loc_out, loc_out)
    else:
        small, big = current
        if loc_out < small:
            small = loc_out
        if loc_out > big:
            big = loc_out
        extremes[loc_in] = (small, big)


def get_index_from_proj(proj, proj_axes, i):
    """returns a tuple that can be used to index the rank-3 npy array
    from the proj coords, proj_axes, and coordinate in the remaining dimension
    """
    index = [0, 0, 0]
    index[proj_axes[0]] = proj[0]
    index[proj_axes[1]] = proj[1]
    index[(set([0, 1, 2]) - set(proj_axes)).pop()] = i
    return tuple(index)


def maybe_fill_line(old_npy, new_npy, proj, extremes, proj_axes, fill_material=None):
    """fill the line from starting from extremes[0] to extremes[1].
    proj_axes is the gives the axes of the non-extreme coordinates
    proj gives the coordinates in the proj_axes coordinates
    if the extremes are different block types changes in the middle
    """
    if (
        extremes[0] < extremes[1] + 1
    ):  # else no need to fill- extremes are the same block or touching
        old_small = old_npy[get_index_from_proj(proj, proj_axes, extremes[0])]
        old_big = old_npy[get_index_from_proj(proj, proj_axes, extremes[1])]
        med = (extremes[1] + extremes[0]) // 2
        for i in range(extremes[0], med + 1):
            new_npy[get_index_from_proj(proj, proj_axes, i)] = old_small
        for i in range(med + 1, extremes[1]):
            new_npy[get_index_from_proj(proj, proj_axes, i)] = old_big


# TODO specify a slice or direction
def fill_flat(blocks, fill_material=None):
    """attempts to fill areas in a shape.  basic algorithm:  if two blocks in the shape
    are connected by an x, y, or z only line, fills that line
    """
    blocks = maybe_convert_to_npy(blocks)
    szs = blocks.shape

    # find the max and min in each plane
    # this is kind of buggy, should instead find inner edge of most outer connected component...
    # instead of plain extremes
    xz_extremes = {}
    yz_extremes = {}
    xy_extremes = {}
    for i in range(szs[0]):
        for j in range(szs[1]):
            for k in range(szs[2]):
                if blocks[i, j, k, 0] > 0:
                    maybe_update_extreme_loc(xz_extremes, (i, j, k), 1)
                    maybe_update_extreme_loc(yz_extremes, (i, j, k), 0)
                    maybe_update_extreme_loc(xy_extremes, (i, j, k), 2)

    old = blocks.copy()
    # fill in the between blocks:
    for proj, extremes in xz_extremes.items():
        maybe_fill_line(old, blocks, proj, extremes, (0, 2), fill_material=fill_material)
    for proj, extremes in yz_extremes.items():
        maybe_fill_line(old, blocks, proj, extremes, (1, 2), fill_material=fill_material)
    for proj, extremes in xy_extremes.items():
        maybe_fill_line(old, blocks, proj, extremes, (0, 1), fill_material=fill_material)
    return blocks


# fixme deal with 2D
def hollow(blocks):
    """
    Args:
        blocks: list of blocks
    Returns:
        list of blocks that have been hollowed out
    """
    # this is not the inverse of fill
    filled_blocks = fill_flat(blocks)
    schematic = filled_blocks.copy()
    schematic[:] = 0
    szs = filled_blocks.shape
    for i in range(szs[0]):
        for j in range(szs[1]):
            for k in range(szs[2]):
                idm = filled_blocks[i, j, k]
                if check_boundary((i, j, k), (0, 0, 0), szs) and idm[0] != 0:
                    schematic[i, j, k] = idm
                    continue
                airtouching = False
                for r in [-1, 0, 1]:
                    for s in [-1, 0, 1]:
                        for t in [-1, 0, 1]:
                            if filled_blocks[i + r, j + s, k + t, 0] == 0:
                                airtouching = True
                if airtouching and idm[0] != 0:
                    schematic[i, j, k] = idm
    return schematic


if __name__ == "__main__":
    import torch
    import visdom
    from shapes import *
    from voxel_models.plot_voxels import SchematicPlotter

    vis = visdom.Visdom(server="http://localhost")
    sp = SchematicPlotter(vis)
    b = hollow_triangle(size=5, thickness=1)
    #    b = hollow_rectangle(size = (10,7),thickness = 2)
    a = torch.LongTensor(15, 15)
    a.zero_()
    for i in b:
        a[i[0][0], i[0][1]] = 1
    print(a)
    c = reshift(thicker(b))
    a.zero_()
    for i in c:
        a[i[0][0], i[0][1]] = 1
    print(a)

    b = hollow_cube(size=5, thickness=1)
    Z = scale(b, (1.0, 1.5, 1.7))
