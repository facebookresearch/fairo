"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
import os
import sys
import random
import math
import torch
import torch.utils.data

CRAFTASSIST_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(CRAFTASSIST_DIR)

import shapes
import shape_helpers as sh
import spatial_utils as su
import directional_utils as du

################################################################################
# Utils for Creating Shapes
################################################################################

# subshapes by everything in a l1 or l2 ball from a point.
# put pairs + triples of shapes in frame, sometimes one partially built


PERM = torch.randperm(256)
r = np.arange(0, 256) / 256
CMAP = np.stack((r, np.roll(r, 80), np.roll(r, 160)))
MIN_SIZE = 4


def get_shape(name="random", max_size=20, fixed_size=False, opts=None):
    if name != "random" and name not in SHAPENAMES:
        print(">> Shape name {} not in dict, choosing randomly".format(name))
        name = "random"
    if name == "random":
        name = random.choice(SHAPENAMES)
    if not opts:
        opts = SHAPE_HELPERS[name](max_size, fixed_size)
    opts["labelme"] = False
    return GEOSCORER_SHAPEFNS[name](**opts), opts, name


def options_cube(max_size, fixed_size=False):
    if fixed_size:
        return {"size": max_size}
    return {"size": np.random.randint(MIN_SIZE, max_size + 1)}


def options_hollow_cube(max_size, fixed_size=False):
    opts = {}
    if fixed_size:
        opts["size"] = max_size
    else:
        opts["size"] = np.random.randint(MIN_SIZE, max_size + 1)

    if opts["size"] < 5:
        opts["thickness"] = 1
    else:
        opts["thickness"] = np.random.randint(1, opts["size"] - 3)
    return opts


def options_rectanguloid(max_size, fixed_size=False):
    if fixed_size:
        return {"size": np.array([max_size for i in range(3)])}
    return {"size": np.random.randint(MIN_SIZE, max_size + 1, size=3)}


def options_hollow_rectanguloid(max_size, fixed_size=False):
    opts = {}
    if fixed_size:
        opts["size"] = np.array([max_size for i in range(3)])
    else:
        opts["size"] = np.random.randint(MIN_SIZE, max_size + 1, size=3)
    ms = min(opts["size"])
    if ms <= 4:
        opts["thickness"] = 1
    else:
        opts["thickness"] = np.random.randint(1, ms - 3 + 1)
    return opts


def options_sphere(max_size, fixed_size=False):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    if fixed_size:
        return {"radius": max_r}
    return {"radius": np.random.randint(min_r, max_r + 1)}


def options_spherical_shell(max_size, fixed_size=False):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    opts = {}
    if fixed_size:
        opts["radius"] = max_r
    elif max_r <= 5:
        opts["radius"] = np.random.randint(min_r, max_r + 1)
    else:
        opts["radius"] = np.random.randint(5, max_r + 1)

    if opts["radius"] <= 5:
        opts["thickness"] = 1
    else:
        opts["thickness"] = np.random.randint(1, opts["radius"] - 3)
    return opts


def options_square_pyramid(max_size, fixed_size=False):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    opts = {}
    if fixed_size:
        opts["radius"] = max_r
    else:
        opts["radius"] = np.random.randint(min_r, max_r + 1)
    opts["slope"] = np.random.rand() * 0.4 + 0.8
    return opts


def options_square(max_size, fixed_size=False):
    if fixed_size:
        size = max_size
    else:
        size = np.random.randint(MIN_SIZE, max_size + 1)

    return {"size": size, "orient": sh.orientation3()}


def options_rectangle(max_size, fixed_size=False):
    if fixed_size:
        size = np.array([max_size for _ in range(2)])
    else:
        size = np.random.randint(MIN_SIZE, max_size + 1, size=2)
    return {"size": size, "orient": sh.orientation3()}


def options_circle(max_size, fixed_size=False):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    if fixed_size:
        radius = max_r
    else:
        radius = np.random.randint(min_r, max_r + 1)
    return {"radius": radius, "orient": sh.orientation3()}


def options_disk(max_size, fixed_size=False):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    if fixed_size:
        radius = max_r
    else:
        radius = np.random.randint(min_r, max_r + 1)
    return {"radius": radius, "orient": sh.orientation3()}


def options_triangle(max_size, fixed_size=False):
    if fixed_size:
        size = max_size
    else:
        size = np.random.randint(MIN_SIZE, max_size + 1)
    return {"size": size, "orient": sh.orientation3()}


def options_dome(max_size, fixed_size=False):
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    if fixed_size:
        radius = max_r
    else:
        radius = np.random.randint(min_r, max_r + 1)
    return {"radius": radius}


def options_arch(max_size, fixed_size=False):
    ms = max(MIN_SIZE + 1, max_size * 2 // 3)
    mh = max_size // 2 - 1
    if fixed_size or mh < 2:
        size = ms
        distance = 2 * (max_size // 2) - 1
    else:
        size = np.random.randint(MIN_SIZE, ms)
        distance = 2 * np.random.randint(2, mh) + 1
    return {"size": size, "distance": distance}


def options_ellipsoid(max_size, fixed_size=False):
    # these sizes are actually radiuses
    min_r = MIN_SIZE // 2
    max_r = max_size // 2
    if fixed_size:
        radius = np.array([max_r for _ in range(3)])
    else:
        radius = np.random.randint(min_r, max_r + 1, size=3)
    return {"size": radius}


def options_tower(max_size, fixed_size=False):
    if fixed_size:
        height = max_size
    else:
        height = np.random.randint(3, max_size + 1)
    return {"height": height, "base": np.random.randint(-4, 6)}


def options_hollow_triangle(max_size, fixed_size=False):
    if fixed_size:
        size = max_size
    else:
        size = np.random.randint(MIN_SIZE, max_size + 1)
    return {"size": size, "orient": sh.orientation3()}


def options_hollow_rectangle(max_size, fixed_size=False):
    opts = {}
    if fixed_size:
        opts["size"] = np.array([max_size for i in range(3)])
    else:
        opts["size"] = np.random.randint(MIN_SIZE, max_size + 1, size=3)
    ms = min(opts["size"])
    if ms <= 4:
        opts["thickness"] = 1
    else:
        opts["thickness"] = np.random.randint(1, ms - 3 + 1)
    return opts


def options_rectanguloid_frame(max_size, fixed_size=False):
    opts = {}
    if fixed_size:
        opts["size"] = np.array([max_size for i in range(3)])
    else:
        opts["size"] = np.random.randint(MIN_SIZE, max_size + 1, size=3)
    ms = min(opts["size"])
    if ms <= 4:
        opts["thickness"] = 1
    else:
        opts["thickness"] = np.random.randint(1, ms - 3 + 1)
    return opts


# eventually put ground blocks, add 'floating', 'hill', etc.
# TODO hollow is separate tag
GEOSCORER_SHAPENAMES = [n for n in sh.SHAPE_NAMES]
GEOSCORER_SHAPENAMES.append("TOWER")

GEOSCORER_SHAPEFNS = {k: v for k, v in sh.SHAPE_FNS.items()}
GEOSCORER_SHAPEFNS["TOWER"] = shapes.tower

SHAPE_HELPERS = {
    "CUBE": options_cube,
    "HOLLOW_CUBE": options_hollow_cube,
    "RECTANGULOID": options_rectanguloid,
    "HOLLOW_RECTANGULOID": options_hollow_rectanguloid,
    "SPHERE": options_sphere,
    "SPHERICAL_SHELL": options_spherical_shell,
    "PYRAMID": options_square_pyramid,
    "SQUARE": options_square,
    "RECTANGLE": options_rectangle,
    "CIRCLE": options_circle,
    "DISK": options_disk,
    "TRIANGLE": options_triangle,
    "DOME": options_dome,
    "ARCH": options_arch,
    "ELLIPSOID": options_ellipsoid,
    "TOWER": options_tower,
    "HOLLOW_TRIANGLE": options_hollow_triangle,
    "HOLLOW_RECTANGLE": options_hollow_rectangle,
    "RECTANGULOID_FRAME": options_rectanguloid_frame,
}

################################################################################
# Utils for the Shape Piece Dataset
################################################################################


def check_l1_dist(a, b, d):
    return abs(b[0] - a[0]) <= d[0] and abs(b[1] - a[1]) <= d[1] and abs(b[2] - a[2]) <= d[2]


def get_rectanguloid_subsegment(S, c, max_chunk=10):
    bounds, segment_sizes = su.get_bounds_and_sizes(S)
    max_dists = []
    for s in segment_sizes:
        max_side_len = min(s - 1, max_chunk)
        max_dist = int(max(max_side_len / 2, 1))
        max_dists.append(random.randint(1, max_dist))

    return [check_l1_dist(c, b[0], max_dists) for b in S]


def get_random_shape_pt(shape, side_length=None):
    sl = side_length
    p = random.choice(shape)[0]
    if not side_length:
        return p
    while p[0] >= sl or p[1] >= sl or p[2] >= sl:
        p = random.choice(shape)[0]
    return p


def get_shape_segment(max_chunk=10, side_length=None):
    """
    Takes a shape, chooses a rectanguloid subset as the segment.
    """
    shape, _, _ = get_shape()
    p = get_random_shape_pt(shape, side_length)

    seg = get_rectanguloid_subsegment(shape, p, max_chunk=max_chunk)
    s_tries = 0
    while all(item for item in seg):
        p = get_random_shape_pt(shape, side_length)
        seg = get_rectanguloid_subsegment(shape, p, max_chunk=max_chunk)
        s_tries += 1
        # Get new shape
        if s_tries > 3:
            shape, _, _ = get_shape()
            p = get_random_shape_pt(shape, side_length)
            seg = get_rectanguloid_subsegment(shape, p, max_chunk=max_chunk)
            s_tries = 0

    seg_inds = set([i for i, use in enumerate(seg) if use])
    context_sparse = [b for i, b in enumerate(shape) if i not in seg_inds]
    seg_sparse = [shape[i] for i in seg_inds]
    return context_sparse, seg_sparse


class ShapePieceData(torch.utils.data.Dataset):
    """
    The goal of the dataset is to take a context voxel, a segment voxel,
    and a direction and predict the correct location to put the min corner
    of the segment voxel in the context space to combine them correctly.

    This dataset specifically uses shapes as created above and then selects
    a rectanguloid segment from the shape to use as the segment.  The actual
    position of the segment is the target for reconstruction.

    Each element contains:
        "context": CxCxC context voxel
        "segment": SxSxS segment voxel
        "dir_vec": 6 element direction vector
        "viewer_pos": the position of the viewer
        "viewer_look": the center of the context object, possibly different
            from the space center if the context is larger than the space.
    """

    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        ground_type=None,
        random_ground_height=False,
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.ground_type = ground_type
        self.random_ground_height = random_ground_height

    def _get_example(self):
        # Get the raw context and seg
        uncentered_context_sparse, uncentered_seg_sparse = get_shape_segment(
            max_chunk=self.s_sl - 1, side_length=self.c_sl
        )
        # Convert into an example, without direction info
        example = su.sparse_context_seg_in_space_to_example(
            uncentered_context_sparse,
            uncentered_seg_sparse,
            self.c_sl,
            self.s_sl,
            self.useid,
            self.ground_type,
            self.random_ground_height,
        )

        # Add the direction info
        target_coord = torch.tensor(su.index_to_coord(example["target"], self.c_sl))
        example["viewer_pos"], example["dir_vec"] = du.get_random_vp_and_max_dir_vec(
            example["viewer_look"], target_coord, self.c_sl
        )
        return example

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


################################################################################
# Utils for the Shape Pair Dataset
################################################################################


def get_two_shape_sparse(c_sl, s_sl, shape_type="random", max_shift=0, fixed_size=None):
    # Get the two shapes that are supposed to be the right size
    max_s = s_sl
    max_c = c_sl - 2 * max_s

    if shape_type == "same":
        shape_type = random.choice(GEOSCORER_SHAPENAMES)
    elif shape_type not in ["random", "empty"]:
        shape_type = shape_type.upper()

    if fixed_size is not None and fixed_size < s_sl:
        sparse_context, _, _ = get_shape(shape_type, fixed_size, True)
        sparse_seg, _, _ = get_shape(shape_type, fixed_size, True)
        max_s = fixed_size
        max_c = fixed_size
    else:
        sparse_context, _, _ = get_shape(shape_type, max_c, False)
        sparse_seg, _, _ = get_shape(shape_type, max_s, False)

    # Trim both and ensure they are within the correct bound
    t_context = su.bound_and_trim_sparse_voxel(sparse_context, max_c)
    t_seg = su.bound_and_trim_sparse_voxel(sparse_seg, max_s)

    # Find shift_vec to match centers
    c_bounds, c_sizes = su.get_bounds_and_sizes(t_context)
    s_bounds, s_sizes = su.get_bounds_and_sizes(t_seg)
    c_min = su.get_min_corner_from_bounds(c_bounds)
    s_min = su.get_min_corner_from_bounds(s_bounds)
    c_center = [s // 2 + mn for s, mn in zip(c_sizes, c_min)]
    s_center = [s // 2 + mn for s, mn in zip(s_sizes, s_min)]
    init_shift_vec = [c - s for c, s in zip(c_center, s_center)]

    # Choose a dim along which to offset, and a direction to offset
    dim = random.choice([0, 1, 2])
    dr = random.choice([-1, 1])

    # Calculate base shift
    base_shift = (c_sizes[dim] // 2) + 1
    if dr > 0:
        if c_sizes[dim] % 2 == 0:
            base_shift -= 1
    base_shift += s_sizes[dim] // 2
    if dr < 0:
        if s_sizes[dim] % 2 == 0:
            base_shift -= 1

    # Determine how much we can additionally shift in this direction
    remaining_halfspace = int((c_sl // 2) - math.ceil(c_sizes[dim] * 1.0 / 2) - s_sizes[dim])
    max_shift = min(max_shift, remaining_halfspace)
    extra_shift = 0
    if max_shift > 0:
        extra_shift = np.random.randint(0, max_shift + 1)

    # Then shift the segment in the correct direction by t
    shift_vec = [init_shift_vec[i] for i in range(3)]
    shift_vec[dim] += dr * (base_shift + extra_shift)
    seg = su.shift_sparse_voxel(t_seg, shift_vec)
    return t_context, seg


class ShapePairData(torch.utils.data.Dataset):
    def __init__(
        self,
        nexamples=100000,
        context_side_length=32,
        seg_side_length=8,
        useid=False,
        ground_type=None,
        random_ground_height=False,
        shape_type="random",
        max_shift=0,
        fixed_size=None,
    ):
        self.c_sl = context_side_length
        self.s_sl = seg_side_length
        self.num_examples = nexamples
        self.useid = useid
        self.examples = []
        self.ground_type = ground_type
        self.random_ground_height = random_ground_height
        self.shape_type = shape_type
        self.max_shift = max_shift
        self.fixed_size = fixed_size

    def _get_example(self):
        # Get the raw context and seg
        context_sparse, seg_sparse = get_two_shape_sparse(
            self.c_sl, self.s_sl, self.shape_type, self.max_shift, self.fixed_size
        )
        # Convert into an example, without direction info
        example = su.sparse_context_seg_in_space_to_example(
            context_sparse,
            seg_sparse,
            self.c_sl,
            self.s_sl,
            self.useid,
            self.ground_type,
            self.random_ground_height,
        )

        # Add the direction info
        target_coord = torch.tensor(su.index_to_coord(example["target"], self.c_sl))
        example["viewer_pos"], example["dir_vec"] = du.get_random_vp_and_max_dir_vec(
            example["viewer_look"], target_coord, self.c_sl
        )
        return example

    def __getitem__(self, index):
        return self._get_example()

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    import argparse
    from visualization_utils import GeoscorerDatasetVisualizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=3, help="num examples to visualize")
    parser.add_argument(
        "--shape_type", type=str, default="random", help="set both shapes to same type"
    )
    parser.add_argument("--fixed_size", type=int, default=None, help="fix the size of the shapes")
    parser.add_argument(
        "--max_shift", type=int, default=0, help="max separation between context and seg"
    )
    parser.add_argument(
        "--ground_type", type=str, default=None, help="ground type to use (None|flat|hilly)"
    )
    parser.add_argument("--random_ground_height", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="pair")
    opts = parser.parse_args()

    if opts.dataset_type == "pair":
        dataset = ShapePairData(
            nexamples=opts.n_examples,
            ground_type=opts.ground_type,
            shape_type=opts.shape_type,
            max_shift=opts.max_shift,
            fixed_size=opts.fixed_size,
            random_ground_height=opts.random_ground_height,
        )
    else:
        dataset = ShapePieceData(
            nexamples=opts.n_examples,
            ground_type=opts.ground_type,
            random_ground_height=opts.random_ground_height,
        )

    vis = GeoscorerDatasetVisualizer(dataset)
    use_model = opts.checkpoint is not None
    if use_model:
        import training_utils as tu

        tms = tu.get_context_segment_trainer_modules(
            vars(opts), opts.checkpoint, backup=False, verbose=True, use_new_opts=False
        )
        if tms["opts"]["cuda"] == 1:
            tms["score_module"].cuda()
            tms["lfn"].cuda()
        vis.set_model(tms, opts=tms["opts"])
    for n in range(len(dataset)):
        vis.visualize(use_model=use_model, verbose=True)
