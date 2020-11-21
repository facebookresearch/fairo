"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import numpy as np
from numpy import sin, cos, deg2rad
import torch

DIRECTIONS = {
    "AWAY": np.array([0, 0, 1]),
    "FRONT": np.array([0, 0, 1]),
    "BACK": np.array([0, 0, -1]),
    "LEFT": np.array([1, 0, 0]),
    "RIGHT": np.array([-1, 0, 0]),
    "DOWN": np.array([0, -1, 0]),
    "UP": np.array([0, 1, 0]),
}


def transform(coords, yaw, pitch, inverted=False):
    """Coordinate transforms with respect to yaw/pitch of the viewer
       coords should be relative to the viewer *before* pitch/yaw transform
       If we want to transform any of DIRECTIONS back, then it would be inverted=True
    """
    # our yaw and pitch are clockwise in the standard coordinate system
    theta = deg2rad(-yaw)
    gamma = deg2rad(-pitch)

    # standard 3d coordinate system as in:
    # http://planning.cs.uiuc.edu/node101.html#fig:yawpitchroll
    # http://planning.cs.uiuc.edu/node102.html
    #    ^ z
    #    |
    #    |-----> y
    #   /
    #  V x
    rtheta = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

    rgamma = np.array([[cos(gamma), 0, sin(gamma)], [0, 1, 0], [-sin(gamma), 0, cos(gamma)]])

    # Minecraft world:
    #         ^ y
    #         |  ^ z
    #         | /
    #  x <----/
    x, y, z = coords
    # a b c is the standard coordinate system
    if not inverted:
        trans_mat = np.linalg.inv(rtheta @ rgamma)
    else:
        trans_mat = rtheta @ rgamma
    a, b, c = trans_mat @ [-z, -x, y]
    # transform back to Minecraft world
    return [-b, c, -a]


def look_vec(yaw, pitch):
    """Takes yaw and pitch and 
    Returns:
        the look vector representing the coordinates of where the 
    entity is looking"""
    yaw = deg2rad(yaw)
    pitch = deg2rad(pitch)
    x = -cos(pitch) * sin(yaw)
    y = -sin(pitch)
    z = cos(pitch) * cos(yaw)
    return np.array([x, y, z])


def batch_normalize(batched_vector):
    """Batched normalization of vectors."""
    vec = batched_vector.double()
    norm = torch.norm(vec, dim=1)
    # Set norm to 1 if it's 0
    norm = norm + norm.eq(0).double()
    expanded_norm = norm.unsqueeze(1).expand(-1, vec.size()[1])
    return torch.div(vec, expanded_norm)


# def get_batched_rotation_matrix(viewer_pos, viewer_look):
def get_batched_rotation_matrix(look_vec):
    """Produces a batched rotation matrix to transform any batch of 2D vectors
    #(a, b) into a space where viewer_pos is (0, 0) and viewer_look is (0, 1).
    (a, b) into a space where the look vec is aligned with (0, 1).
    Takes a batch of look vectors of length 2 and returns a batch of 2x2 tensor.
    If the look vecs are already (0, 1) the identity matrix is returned.
    #Takes two batches of tensors of length 2 and returns a batch of 2x2 tensor.
    #If viewer_pos and viewer_look are already (0, 0) and (0, 1) respectively,
    #the rotation matrix is the identity matrix.

    Rotation Matrix Calcuation:
        #First convert viewer_pos, viewer_look into a look vec (lx, ly).  Then
        #normalize it to (nlx, nly).

        First normalize the look vec (lx, ly) into (nlx, nly).

        Our goal is to rotate (nlx, nly) to (0, 1).  Given the angle, theta,
        between these two vectors the rotation matrix would be:
        [  cos(theta), sin(theta) ]
        [ -sin(theta), cos(theta) ]

        If you consider the length one vector to (nlx, nly) to be the
        hypotenuse of a triangle with the (0, 1) axis then:
        cos(theta) = nly / 1
        theta = arccos(nly)

        Also note that: sin(arccos(x)) = sqrt( 1 - x^2 )
        And plugging this in to the rotation matrix we get:
        [  nly              , sqrt( 1 - nly^2 )]
        [ -sqrt( 1 - nly^2 ), nly              ]

    A few tricks:
        - The range of arccos is only half of the space so switch the sign
          of sin(theta) when nlx is negative.  Done in a strange way below
          to avoid if statements because of batching.
        - Raising 0 to the power of 0 gives a nan, so instead use a mask to
          set the value manually to 0 as intended.
    """
    vp_to_vl = look_vec
    # vp_to_vl = viewer_look - viewer_pos
    normalized_look_vec = batch_normalize(vp_to_vl)
    nly = normalized_look_vec[:, 1]
    nlx = normalized_look_vec[:, 0]

    # nlx_modifier corrects for the range of acrcos
    nlx_modifier = nlx.gt(0).double() - (nlx.lt(0).double() + nlx.eq(0).double())

    # Take care of nans created by raising 0 to a power
    # and then masking the sin theta to 0 as intended
    base = 1 - nly * nly
    nan_mask = torch.isnan(torch.pow(base, 0.5)).double()
    base = base + nan_mask
    sin_theta = nlx_modifier * nan_mask.eq(0).double() * torch.pow(base, 0.5)

    nly = nly.unsqueeze(1)
    sin_theta = sin_theta.unsqueeze(1)
    rm_pt1 = torch.cat([nly, sin_theta], 1).unsqueeze(1)
    rm_pt2 = torch.cat([-sin_theta, nly], 1).unsqueeze(1)
    rm = torch.cat([rm_pt1, rm_pt2], 1)
    return rm


def batched_rotate(look_vec, source_vec):
    """
    Takes a N x 3 look_vec batch and a N x 3 target_vec batch and returns
    a N x 3 rotated target_vec batch where the look_vec was rotated in
    XZ space to align with the (0, X, 1) axis.

    This is the batched equivalent of the transform fxn above with inverted
    set to False and pitch=0.
    """
    look_vec = look_vec.double()
    source_vec = source_vec.double()
    # N x 2
    look_xz = look_vec[:, [0, 2]]
    # N x 2
    source_xz = source_vec[:, [0, 2]]
    # N x 2 x 2
    rms = get_batched_rotation_matrix(look_xz)
    # N x 1 x 2 bmm N x 2 x 2 => N x 1 x 2 ==> N x 2
    rotated_source_xz = torch.bmm(source_xz.unsqueeze(1), rms).squeeze(1)
    # N x 3
    rotated_source = torch.cat(
        [rotated_source_xz[:, [0]], source_vec[:, [1]], rotated_source_xz[:, [1]]], dim=1
    )
    return rotated_source


if __name__ == "__main__":
    print("Compare batched and non-batched transform test:")
    inp = DIRECTIONS["RIGHT"]
    yaw = 45
    pitch = 0
    print(">> input_vec:", inp)
    print(">> yaw:", yaw, "pitch:", pitch)
    lv = look_vec(yaw, pitch)
    print(">> look_vec:", lv)

    inverted_out = transform(inp, yaw, pitch, inverted=True)
    out = transform(inp, yaw, pitch, inverted=False)
    print(">> output of transform, inv=True:", inverted_out)
    print(">> output of transform, inv=False:", out)

    lv_batched = torch.from_numpy(lv).unsqueeze(0)
    inp_batched = torch.from_numpy(inp).unsqueeze(0)
    output_batched = batched_rotate(lv_batched, inp_batched)
    print(">> batched out:", output_batched)
    diff = (output_batched - torch.tensor(out)).abs()
    assert diff.ge(0.0001).int().sum() == 0
    print(">> TEST PASSED\n")

    print("Validate batched rotation outputs on all axis directions test:")
    look_vecs = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]])
    look_vecs_expanded = look_vecs.unsqueeze(1).expand(4, 4, 3).contiguous().view(16, 3)

    input_vecs = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0]])
    input_vecs_expanded = input_vecs.unsqueeze(0).expand(4, 4, 3).contiguous().view(16, 3)

    expected_outputs = torch.tensor(
        [
            # lv: [1, 0, 0]
            [-1, 0, 0],  # source: [ 0, 0,  1]
            [0, 0, 1],  # source: [ 1, 0,  0]
            [1, 0, 0],  # source: [ 0, 0, -1]
            [0, 0, -1],  # source: [-1, 0,  0]
            # lv: [-1, 0, 0]
            [1, 0, 0],  # source: [ 0, 0,  1]
            [0, 0, -1],  # source: [ 1, 0,  0]
            [-1, 0, 0],  # source: [ 0, 0, -1]
            [0, 0, 1],  # source: [-1, 0,  0]
            # lv: [0, 0, 1]
            [0, 0, 1],  # source: [ 0, 0,  1]
            [1, 0, 0],  # source: [ 1, 0,  0]
            [0, 0, -1],  # source: [ 0, 0, -1]
            [-1, 0, 0],  # source: [-1, 0,  0]
            # lv: [0, 0, -1]
            [0, 0, -1],  # source: [ 0, 0,  1]
            [-1, 0, 0],  # source: [ 1, 0,  0]
            [0, 0, 1],  # source: [ 0, 0, -1]
            [1, 0, 0],  # source: [-1, 0,  0]
        ]
    )

    batched_rotated_coords = batched_rotate(look_vecs_expanded, input_vecs_expanded)
    assert (~batched_rotated_coords.eq(expected_outputs)).int().sum() == 0
    print(">> TEST PASSED")
