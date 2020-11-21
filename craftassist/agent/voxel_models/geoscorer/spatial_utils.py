"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import torch
import os
import sys
import random

CRAFTASSIST_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
TEST_DIR = os.path.join(CRAFTASSIST_DIR, "../test/")
sys.path.append(CRAFTASSIST_DIR)
sys.path.append(TEST_DIR)

from world import World, Opt, flat_ground_generator

"""
Generic Spatial Utils
"""


def euclid_dist(a, b):
    """Return euclidean distance between a and b"""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def avg_dist_between_target_indices(a_batch, b_batch, sl):
    avg_dist = 0
    for i in range(a_batch.size(0)):
        a_coord = index_to_coord(a_batch[i].item(), sl)
        b_coord = index_to_coord(b_batch[i].item(), sl)
        avg_dist += euclid_dist(a_coord, b_coord)
    return avg_dist / a_batch.size(0)


def get_bounds(sparse_voxel):
    """
    Voxel should either be a schematic, a list of ((x, y, z), (block_id, ?)) objects
    or a list of coordinates.
    Returns a list of inclusive bounds.
    """
    if len(sparse_voxel) == 0:
        return [0, 0, 0, 0, 0, 0]

    # A schematic
    if len(sparse_voxel[0]) == 2 and len(sparse_voxel[0][0]) == 3 and len(sparse_voxel[0][1]) == 2:
        x, y, z = list(zip(*list(zip(*sparse_voxel))[0]))
    # A list or coordinates
    elif len(sparse_voxel[0]) == 3:
        x, y, z = list(zip(*sparse_voxel))
    else:
        raise Exception("Unknown schematic format")
    return min(x), max(x), min(y), max(y), min(z), max(z)


def get_min_corner(sparse_voxel):
    """
    Voxel should either be a schematic, a list of ((x, y, z), (block_id, ?)) objects
    or a list of coordinates.
    Returns the minimum corner of the bounding box around the voxel.
    """
    if len(sparse_voxel) == 0:
        return [0, 0, 0]

    # A schematic
    if len(sparse_voxel[0]) == 2 and len(sparse_voxel[0][0]) == 3 and len(sparse_voxel[0][1]) == 2:
        x, y, z = list(zip(*list(zip(*sparse_voxel))[0]))
    # A list or coordinates
    elif len(sparse_voxel[0]) == 3:
        x, y, z = list(zip(*sparse_voxel))
    else:
        raise Exception("Unknown schematic format")
    return min(x), min(y), min(z)


def get_min_corner_from_bounds(bounds):
    return [bounds[0], bounds[2], bounds[4]]


def get_side_lengths(bounds):
    """
    Bounds should be a list of [min_x, max_x, min_y, max_y, min_z, max_z].
    Returns a list of the side lengths.
    """
    return [x + 1 for x in (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])]


def get_bounds_and_sizes(sparse_voxel):
    bounds = get_bounds(sparse_voxel)
    side_lengths = get_side_lengths(bounds)
    return bounds, side_lengths


def get_voxel_bounds(voxel):
    nz = torch.nonzero(voxel, as_tuple=False)
    maxs = nz.max(dim=0).values
    mins = nz.min(dim=0).values
    return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]


def get_min_block_pair_dist(sparse_a, sparse_b):
    min_dist = float("inf")
    for a in sparse_a:
        for b in sparse_b:
            min_dist = min(euclid_dist(a, b), min_dist)
    return min_dist


def shift_sparse_voxel(sparse_v, shift_vec, min_b=None, max_b=None):
    """
    Takes a voxel and shifts it by the shift_vec, dropping all blocks that
    fall outside min_b (inclusive) and max_b (exclusive) if they are specified.
    """
    new_v = []
    for v in sparse_v:
        block = v[0]
        nb = tuple(b + s for b, s in zip(block, shift_vec))
        if min_b is not None:
            in_bounds = [b >= mn for b, mn in zip(nb, min_b)]
            if not all(in_bounds):
                continue
        if max_b is not None:
            in_bounds = [b < mx for b, mx in zip(nb, max_b)]
            if not all(in_bounds):
                continue
        new_v.append((nb, v[1]))
    return new_v


def coord_to_index(coord, sl):
    """
    Takes a 3D coordinate in a cube and the cube side length.
    Returns index in flattened 3D array.
    """
    return coord[0] * sl * sl + coord[1] * sl + coord[2]


def index_to_coord(index, sl):
    """
    Takes an index into a flattened 3D array and its side length.
    Returns the coordinate in the cube.
    """
    coord = []
    two_d_slice_size = sl * sl
    coord.append(index // two_d_slice_size)
    remaining = index % two_d_slice_size
    coord.append(remaining // sl)
    coord.append(remaining % sl)
    return coord


def shift_sparse_voxel_to_origin(sparse_voxel):
    """
    Takes a segment, described as a list of tuples of the form:
        ((x, y, z), (block_id, ?))
    Returns the segment in the same form, shifted to the origin, and the shift vec
    """
    bounds = get_bounds(sparse_voxel)
    shift_zero_vec = [-bounds[0], -bounds[2], -bounds[4]]
    new_voxel = shift_sparse_voxel(sparse_voxel, shift_zero_vec)
    return new_voxel, shift_zero_vec


def shift_sparse_voxel_to_space_center(sparse_voxel, sl, min_shift=None, max_shift=None):
    bounds, sizes = get_bounds_and_sizes(sparse_voxel)
    vcenter = [s // 2 for s in sizes]
    scenter = [sl // 2 for _ in sizes]
    shift_vec = [sc - vc for sc, vc in zip(scenter, vcenter)]
    if max_shift is not None:
        shift_vec = [min(sv, ms) for sv, ms in zip(shift_vec, max_shift)]
    if min_shift is not None:
        shift_vec = [max(sv, ms) for sv, ms in zip(shift_vec, min_shift)]

    new_voxel = shift_sparse_voxel(sparse_voxel, shift_vec)
    return new_voxel, shift_vec


def bound_and_trim_sparse_voxel(sparse_voxel, max_sl):
    """
    Takes a sparse voxel, bounds it to a max_side_length and then shifts
    the voxel to the origin (effectively trimming it).
    """
    bounds, sizes = get_bounds_and_sizes(sparse_voxel)
    bounded_sparse_voxel = []
    # Its possible that some versions of trimming will remove all blocks
    # but this should be very rare.
    while len(bounded_sparse_voxel) == 0:
        mins = [bounds[0], bounds[2], bounds[4]]
        maxs = [bounds[1], bounds[3], bounds[5]]
        for dim in range(3):
            to_remove = sizes[dim] - max_sl
            if to_remove <= 0:
                continue
            if random.choice([True, False]):
                mins[dim] += to_remove
            else:
                maxs[dim] -= to_remove
        for b in sparse_voxel:
            use_block = True
            for dim in range(3):
                if b[0][dim] < mins[dim] or b[0][dim] > maxs[dim]:
                    use_block = False
                    break
            if use_block:
                bounded_sparse_voxel.append(b)

    trimmed_voxel, _ = shift_sparse_voxel_to_origin(bounded_sparse_voxel)
    return trimmed_voxel


def sparsify_voxel(voxel):
    vs = []
    sizes = voxel.size()
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            for k in range(sizes[2]):
                if voxel[i, j, k] != 0:
                    vs.append(((i, j, k), (voxel[i, j, k], 0)))
    return vs


# outputs a dense voxel rep (np array) from a sparse one.
# size should be a tuple of (H, W, D) for the desired voxel representation
# useid=True puts the block id into the voxel representation,
#    otherwise put a 1
def densify(blocks, size, center=(0, 0, 0), useid=False):
    V = np.zeros((size[0], size[1], size[2]), dtype="int32")

    offsets = (size[0] // 2 - center[0], size[1] // 2 - center[1], size[2] // 2 - center[2])
    for b in blocks:
        x = b[0][0] + offsets[0]
        y = b[0][1] + offsets[1]
        z = b[0][2] + offsets[2]
        if x >= 0 and y >= 0 and z >= 0 and x < size[0] and y < size[1] and z < size[2]:
            if type(b[1]) is int:
                V[x, y, z] = b[1]
            else:
                V[x, y, z] = b[1][0]
    if not useid:
        V[V > 0] = 1
    return V, offsets


def get_dense_array_from_sl(sparse_shape, sl, useid):
    center = [sl // 2, sl // 2, sl // 2]
    dense_lists, _ = densify(sparse_shape, [sl, sl, sl], center=center, useid=useid)
    shape_dense = np.asarray(dense_lists)
    return shape_dense


"""
Geoscorer Specific Spatial Utils
"""


def shift_sparse_context_seg_to_space_center(context, seg, sl):
    """
    Takes a context and segment (in its target position) and shifts the
    center of their combined bounding box as close to the center of the
    overall space as possible without moving the min corner of the seg
    out of the space.
    """
    s_bounds, s_sizes = get_bounds_and_sizes(seg)
    s_origin = get_min_corner_from_bounds(s_bounds)
    s_center = [m + s // 2 for m, s in zip(s_origin, s_sizes)]
    max_shift = [sl - sc - 1 for sc in s_center]
    min_shift = [-sc for sc in s_center]

    centered_context, shift_vec = shift_sparse_voxel_to_space_center(
        context, sl, min_shift=min_shift, max_shift=max_shift
    )
    c_bounds, c_sizes = get_bounds_and_sizes(centered_context)
    c_min_corner = get_min_corner_from_bounds(c_bounds)
    context_center = [s // 2 + mc for s, mc in zip(c_sizes, c_min_corner)]
    centered_seg = shift_sparse_voxel(seg, shift_vec)
    return centered_context, centered_seg, context_center


def get_seg_origin_from_target_coord(seg, target_coord):
    seg_bounds = get_voxel_bounds(seg)
    seg_sizes = get_side_lengths(seg_bounds)
    target_seg_origin = [t - s // 2 for t, s in zip(target_coord, seg_sizes)]
    return target_seg_origin


def combine_seg_context(seg, context, target_coord, seg_mult=1):
    """
    The target is the center of the seg (s_bound // 2), convert to origin of
    the seg voxel (because the seg is shifted to the origin of its voxel) to
    combine the seg and context using the target.
    """
    # Calculate the shift vec
    seg_shift = get_seg_origin_from_target_coord(seg, target_coord)

    c_sl = context.size()[0]
    s_sl = seg.size()[0]
    completed_context = context.clone()
    # Calculate the region to copy over, sometimes the segment
    #   falls outside the range of the context bounding box
    cs = [slice(s, min(s + s_sl, c_sl)) for s in seg_shift]
    ss = [slice(0, s_sl - max(0, s + s_sl - c_sl)) for s in seg_shift]
    completed_context[cs] = seg_mult * seg[ss] + context[cs]
    return completed_context


def sparse_context_seg_to_target_and_origin_seg(context_sparse, seg_sparse, c_sl, s_sl):
    origin_seg_sparse, shift_vec = shift_sparse_voxel_to_origin(seg_sparse)
    _, seg_size = get_bounds_and_sizes(origin_seg_sparse)
    target_coord = [-sh + s // 2 for sh, s in zip(shift_vec, seg_size)]
    return target_coord, shift_vec, origin_seg_sparse


def sparse_context_seg_target_to_example(
    context_sparse, shifted_seg_sparse, target_coord, c_sl, s_sl, useid, schem_sparse=None
):
    context_dense = get_dense_array_from_sl(context_sparse, c_sl, useid)
    seg_dense = get_dense_array_from_sl(shifted_seg_sparse, s_sl, useid)
    target_index = coord_to_index(target_coord, c_sl)
    example = {
        "context": torch.from_numpy(context_dense),
        "seg": torch.from_numpy(seg_dense),
        "target": torch.tensor([target_index]),
    }
    if schem_sparse:
        schem_dense = get_dense_array_from_sl(schem_sparse, c_sl, useid)
        example["schematic"] = torch.from_numpy(schem_dense)
    return example


def add_ground_to_context(context_sparse, target_coord, flat=True, random_height=True):
    min_y = min([c[0][1] for c in context_sparse] + [target_coord[1]])
    max_ground_depth = min_y - 1
    if max_ground_depth <= 0:
        return
    if random_height:
        ground_depth = random.randint(1, max_ground_depth)
    else:
        ground_depth = max_ground_depth

    pos_z = 63

    shift = (-16, pos_z - 1 - ground_depth, -16)
    spec = {
        "players": [],
        "item_stacks": [],
        "mobs": [],
        "agent": {"pos": (0, pos_z, 0)},
        "coord_shift": shift,
    }
    world_opts = Opt()
    # TODO: this maybe should be a variable??
    world_opts.sl = 32

    if flat or max_ground_depth == 1:
        spec["ground_generator"] = flat_ground_generator
        spec["ground_args"] = {"ground_depth": ground_depth}
    else:
        world_opts.avg_ground_height = max_ground_depth // 2
        world_opts.hill_scale = max_ground_depth // 2
    world = World(world_opts, spec)

    ground_blocks = []
    for l, d in world.blocks_to_dict().items():
        shifted_l = tuple([l[i] - shift[i] for i in range(3)])
        ground_blocks.append((shifted_l, d))
    context_sparse += ground_blocks


def sparse_context_seg_in_space_to_example(
    context, seg, c_sl, s_sl, use_id, ground_type, random_ground_height
):
    """
    Takes a context and seg where both are in their correct position in space
    and converts them into an example, with gorund if specified.
    """
    # First center the context in the c_sl sized space
    c_context, c_seg, c_coord = shift_sparse_context_seg_to_space_center(context, seg, c_sl)

    # Then calculate the target coord from the new seg position
    target_coord, _, origin_seg = sparse_context_seg_to_target_and_origin_seg(
        c_context, c_seg, c_sl, s_sl
    )

    # Add ground if required
    if ground_type is not None:
        add_ground_to_context(
            c_context,
            target_coord,
            flat=True if ground_type == "flat" else False,
            random_height=random_ground_height,
        )

    # Convert it all into an example
    schem_sparse = c_seg + c_context
    example = sparse_context_seg_target_to_example(
        c_context, origin_seg, target_coord, c_sl, s_sl, use_id, schem_sparse
    )
    example["viewer_look"] = torch.tensor(c_coord).float()
    return example
