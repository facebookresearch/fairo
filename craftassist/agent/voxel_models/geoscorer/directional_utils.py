"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import torch
import os
import sys

CRAFTASSIST_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
sys.path.append(CRAFTASSIST_DIR)
import rotation

UNIQUE_DIRECTIONS = ["FRONT", "BACK", "LEFT", "RIGHT", "UP", "DOWN"]


def get_direction_embedding(dir_name):
    """Takes a direction name from rotation.DIRECTIONS and produces the
    correct direction vector, first three values encode the dimension in
    a one hot vector, the last two encode the direction in a one hot.
    UP:    +y (0, 1, 0, 1, 0)
    RIGHT: -x (1, 0, 0, 0, 1)
    FRONT: +z (0, 0, 1, 1, 0)
    """
    np_vec = rotation.DIRECTIONS.get(dir_name, rotation.DIRECTIONS["UP"]).copy()
    if np.sum(np_vec) > 0:
        dir_vec = torch.tensor([np_vec[0], np_vec[1], np_vec[2], 1, 0])
    else:
        np_vec *= -1
        dir_vec = torch.tensor([np_vec[0], np_vec[1], np_vec[2], 0, 1])
    return dir_vec.float()


def get_dim_from_dir_name(dir_name):
    """Returns the dimension index of a given direction name. For example, UP
    is +y, so the return value is 1.
    """
    np_vec = np.abs(rotation.DIRECTIONS.get(dir_name, rotation.DIRECTIONS["UP"])).copy()
    for i in range(3):
        if np_vec[i] == 1:
            return i


def get_dir_from_dir_name(dir_name):
    """Returns the direction on the relevant axis of the direction name.
    UP, LEFT, FRONT are all positive. DOWN, RIGHT, BACK are negative.
    """
    np_vec = rotation.DIRECTIONS.get(dir_name, rotation.DIRECTIONS["UP"]).copy()
    if np_vec.sum() > 0:
        return 1
    return -1


def get_dim_dir_from_dir_name(dir_name):
    return get_dim_from_dir_name(dir_name), get_dir_from_dir_name(dir_name)


def get_random_viewer_info(sl, viewer_look=None):
    """
    Returns a viewer_pos that isn't the same as the viewer_look.
    """
    viewer_pos = torch.tensor(np.random.randint(0, sl, 3))
    if viewer_look is None:
        viewer_look = torch.tensor([sl // 2 for _ in range(3)])

    if viewer_pos.eq(viewer_look).sum() == viewer_pos.size(0):
        if viewer_pos[0] < sl + 1:
            viewer_pos[0] += 1
        else:
            viewer_pos[0] -= 1
    return viewer_pos


def get_vector(start, end):
    return end - start


def dim_to_vec(dim):
    return [(1 if i == dim else 0) for i in range(3)]


def dir_vec_to_dim(dir_vec):
    for i in range(3):
        if dir_vec[i] == 1:
            return i
    raise Exception("dir vec has no dimension")


def dr_to_vec(dr):
    return [1, 0] if dr == 1 else [0, 1]


def dim_dir_to_dir_tensor(dim, dr):
    dim_l = dim_to_vec(dim)
    dir_l = dr_to_vec(dr)
    return torch.tensor(dim_l + dir_l, dtype=torch.long)


"""
COORD SWITCH UTILS
"""


def get_dir_dist(viewer_pos, viewer_look, target_coord):
    """
    Takes the viewer pos and the block the viewer is looking at (viewer_look)
    and determines the offset of the target_coord from the viewer_look in the
    coordinate system where the look vec (viewer_pos to viewer_look) is the +z
    axis. Returns the offsets as distance and direction for each dimension.
    """
    unbatched = viewer_pos.dim() == 1
    look_vec = get_vector(viewer_pos, viewer_look)
    source_vec = get_vector(viewer_look, target_coord)
    if unbatched:
        look_vec = look_vec.unsqueeze(0)
        source_vec = source_vec.unsqueeze(0)
    r_source_vec = rotation.batched_rotate(look_vec, source_vec)
    if unbatched:
        r_source_vec = r_source_vec.squeeze(0)
    dist = r_source_vec.abs()
    direction = r_source_vec.gt(0).double() - r_source_vec.lt(0).double()
    return direction, dist


def create_xyz_tensor(sl):
    """
    Return an 32 x 32 x 32 x 3 tensor where each len 3 inner tensor is
    the xyz coordinates of that position
    """
    incr_t = torch.arange(sl, dtype=torch.float64)
    z = incr_t.expand(sl, sl, sl).unsqueeze(3)
    y = incr_t.unsqueeze(1).expand(sl, sl, sl).unsqueeze(3)
    x = incr_t.unsqueeze(1).unsqueeze(2).expand(sl, sl, sl).unsqueeze(3)
    xyz = torch.cat([x, y, z], 3)
    return xyz


def get_xyz_viewer_look_coords_batched(viewer_pos, viewer_look, batched_target_coords):
    """
    This takes a batch of target coords and then transforms them into a space
    where the viewer_look is the origin and the vector from viewer_pos to
    viewer_look is aligned with the positive z axis.

    N: batch size in training
    D: num target coord per element
    viewer_pos, viewer_look: N x 3 tensors
    batched_target_coords: N x D x 3 tensor
    Returns: N x D x 3 tensor
    """
    # First verify the sizing and unsqueeze if necessary
    unbatched = viewer_pos.dim() == 1
    if unbatched:
        viewer_pos = viewer_pos.unsqueeze(0)
        viewer_look = viewer_look.unsqueeze(0)
        batched_target_coords = batched_target_coords.unsqueeze(0)
    n = batched_target_coords.size(0)
    d = batched_target_coords.size(1)
    assert viewer_pos.size(1) == 3
    assert batched_target_coords.size(2) == 3

    vp_expanded = viewer_pos.unsqueeze(1).expand(n, d, 3).contiguous().view(n * d, 3)
    vl_expanded = viewer_look.unsqueeze(1).expand(n, d, 3).contiguous().view(n * d, 3)
    tc_flatten = batched_target_coords.reshape(n * d, 3)

    dr, dist = get_dir_dist(vp_expanded, vl_expanded, tc_flatten)
    out = dr * dist
    out_reshaped = out.view(n, d, 3)
    if unbatched:
        out_reshaped = out.view(d, 3)
    return out_reshaped


def get_sampled_direction_vec(viewer_pos, viewer_look, target_coord):
    directions, dists = get_dir_dist(viewer_pos, viewer_look, target_coord)
    dists = dists.squeeze()
    directions = directions.squeeze()
    if sum(dists) == 0:
        dim = np.random.choice(3)
    else:
        ndists = dists / sum(dists)
        dim = np.random.choice(3, p=ndists)
    direction = directions[dim].item()
    return dim_dir_to_dir_tensor(dim, direction)


def float_equals(a, b, epsilon):
    return True if abs(a - b) < epsilon else False


def get_argmax_list(vals, epsilon=0.0001, minlist=False, maxlen=None):
    """
    Provides a deterministic max for all lists of floats.
    """
    mult = -1 if minlist else 1
    max_ind = []
    for i, v in enumerate(vals):
        if not max_ind or float_equals(max_ind[0][1], v, epsilon):
            if maxlen and len(max_ind) == maxlen:
                continue
            max_ind.append((i, v))
        elif mult * (v - max_ind[0][1]) > 0:
            max_ind = [(i, v)]
    return max_ind


def get_max_direction_vec(viewer_pos, viewer_look, target_coord):
    directions, dists = get_dir_dist(viewer_pos, viewer_look, target_coord)
    dists = dists.squeeze()
    directions = directions.squeeze()
    if sum(dists) == 0:
        dim = 1
    else:
        ndists = dists / sum(dists)
        dim = get_argmax_list(ndists, maxlen=1)[0][0]
    direction = directions[dim].item()
    return dim_dir_to_dir_tensor(dim, direction)


def get_random_vp_and_max_dir_vec(viewer_look, target_coord, c_sl):
    # Choose a random viewer pos and the direction vector that corresponds to the combo
    # of viewer pos and target coord
    viewer_pos = get_random_viewer_info(c_sl, viewer_look)
    dir_vec = get_max_direction_vec(viewer_pos, viewer_look, target_coord)
    return viewer_pos, dir_vec


if __name__ == "__main__":
    print("Test get_dir_dist on all axis directions:")
    viewer_look = torch.tensor(
        [[16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16], [0, 0, 0]]
    ).float()
    viewer_pos = torch.tensor(
        [[0, 16, 16], [31, 16, 16], [16, 16, 0], [16, 16, 31], [0, 0, 31]]
    ).float()

    offsets = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0]]).float()

    # Using the rotation matrix from a vl, vp pair on all of the inputs before
    # trying the next vl, vp pair.
    expected_outputs = torch.tensor(
        [
            # vp: [0, 16, 16], vl: [16, 16, 16]
            [-1, 0, 0],  # source: [ 0, 0,  1]
            [0, 0, 1],  # source: [ 1, 0,  0]
            [1, 0, 0],  # source: [ 0, 0, -1]
            [0, 0, -1],  # source: [-1, 0,  0]
            # vp: [31, 16, 16], vl: [16, 16, 16]
            [1, 0, 0],  # source: [ 0, 0,  1]
            [0, 0, -1],  # source: [ 1, 0,  0]
            [-1, 0, 0],  # source: [ 0, 0, -1]
            [0, 0, 1],  # source: [-1, 0,  0]
            # vp: [16, 16, 0], vl: [16, 16, 16]
            [0, 0, 1],  # source: [ 0, 0,  1]
            [1, 0, 0],  # source: [ 1, 0,  0]
            [0, 0, -1],  # source: [ 0, 0, -1]
            [-1, 0, 0],  # source: [-1, 0,  0]
            # vp: [16, 16, 31], vl: [16, 16, 16]
            [0, 0, -1],  # source: [ 0, 0,  1]
            [-1, 0, 0],  # source: [ 1, 0,  0]
            [0, 0, 1],  # source: [ 0, 0, -1]
            [1, 0, 0],  # source: [-1, 0,  0]
            # vp: [0, 0, 31], vl: [0, 0, 0]
            [0, 0, -1],  # source: [ 0, 0,  1]
            [-1, 0, 0],  # source: [ 1, 0,  0]
            [0, 0, 1],  # source: [ 0, 0, -1]
            [1, 0, 0],  # source: [-1, 0,  0]
        ]
    ).float()

    vl_expanded = viewer_look.unsqueeze(1).expand(5, 4, 3).contiguous().view(20, 3)
    vp_expanded = viewer_pos.unsqueeze(1).expand(5, 4, 3).contiguous().view(20, 3)
    off_expanded = offsets.unsqueeze(0).expand(5, 4, 3).contiguous().view(20, 3)
    targets = vl_expanded + off_expanded
    dr, dist = get_dir_dist(vp_expanded, vl_expanded, targets)
    output = dr * dist
    assert (~expected_outputs.eq(output)).int().sum() == 0
    print(">> TEST PASSED\n")

    def test_get_max(dn, vp, vl, t):
        expected = get_direction_embedding(dn)
        dir_vec = get_max_direction_vec(vp, vl, t)
        print(">> vp", vp, "vl", vl, "target", t, "dir_name", dn)
        assert (~expected.eq(dir_vec)).int().sum() == 0
        print(">> TEST PASSED\n")

    print("Test get_max_direction_vec on a few examples:")
    test_get_max(
        "RIGHT", torch.tensor([16, 16, 0]), torch.tensor([16, 16, 16]), torch.tensor([12, 16, 16])
    )
    test_get_max(
        "RIGHT", torch.tensor([16, 16, 31]), torch.tensor([16, 16, 16]), torch.tensor([20, 16, 16])
    )
    test_get_max(
        "LEFT", torch.tensor([0, 16, 16]), torch.tensor([16, 16, 16]), torch.tensor([16, 16, 10])
    )
    test_get_max(
        "LEFT", torch.tensor([31, 16, 16]), torch.tensor([16, 16, 16]), torch.tensor([16, 16, 25])
    )
    test_get_max(
        "BACK", torch.tensor([0, 16, 16]), torch.tensor([16, 16, 16]), torch.tensor([9, 16, 16])
    )
    test_get_max(
        "FRONT", torch.tensor([16, 16, 31]), torch.tensor([16, 16, 16]), torch.tensor([16, 16, 5])
    )
    test_get_max(
        "UP", torch.tensor([16, 16, 16]), torch.tensor([16, 16, 16]), torch.tensor([16, 20, 16])
    )
    test_get_max(
        "DOWN", torch.tensor([16, 16, 16]), torch.tensor([16, 16, 16]), torch.tensor([16, 2, 16])
    )

    print("Test the xyz grid setup:")
    sl = 4
    xyz = create_xyz_tensor(sl)

    def test_xyz(vp, vl, c, eout):
        out_xyz = get_xyz_viewer_look_coords_batched(vp, vl, xyz.view(-1, 3)).view(sl, sl, sl, 3)
        print(">> vp", vp, "vl", vl, "c", c)
        assert (~c.eq(xyz[c[0], c[1], c[2]])).int().sum() == 0
        print(">> out coord", out_xyz[c[0], c[1], c[2]])
        assert (~eout.eq(out_xyz[c[0], c[1], c[2]])).int().sum() == 0
        print(">> TEST PASS\n")

    test_xyz(
        vp=torch.tensor([0, 0, -1]),
        vl=torch.tensor([0, 0, 0]),
        c=torch.tensor([0, 0, 0]),
        eout=torch.tensor([0, 0, 0]),
    )

    test_xyz(
        vp=torch.tensor([sl - 1, 0, sl]),
        vl=torch.tensor([sl - 1, sl - 1, sl - 1]),
        c=torch.tensor([0, 0, 0]),
        eout=torch.tensor([sl - 1, -(sl - 1), sl - 1]),
    )

    test_xyz(
        vp=torch.tensor([-1, sl - 1, sl - 1]),
        vl=torch.tensor([0, sl - 1, sl - 1]),
        c=torch.tensor([0, 0, 0]),
        eout=torch.tensor([(sl - 1), -(sl - 1), 0]),
    )

    test_xyz(
        vp=torch.tensor([-1, sl - 1, sl - 1]),
        vl=torch.tensor([0, sl - 1, sl - 1]),
        c=torch.tensor([sl - 1, 0, 0]),
        eout=torch.tensor([sl - 1, -(sl - 1), sl - 1]),
    )

    test_xyz(
        vp=torch.tensor([2, 2, 1]),
        vl=torch.tensor([2, 2, 2]),
        c=torch.tensor([0, 0, 0]),
        eout=torch.tensor([-2, -2, -2]),
    )

    test_xyz(
        vp=torch.tensor([2, 2, 1]),
        vl=torch.tensor([2, 2, 2]),
        c=torch.tensor([2, 2, 2]),
        eout=torch.tensor([0, 0, 0]),
    )

    print("Test the batched xyz setup:")
    vp = torch.tensor([2, 2, 1]).unsqueeze(0)
    vl = torch.tensor([2, 2, 2]).unsqueeze(0)
    out_xyz = get_xyz_viewer_look_coords_batched(vp, vl, xyz.view(1, -1, 3))
    assert out_xyz.dim() == 3
    print(">> TEST PASS\n")
