"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np

"""
Co-ordinate transform utils. Read more at https://github.com/facebookresearch/fairo/blob/main/locobot/coordinates.MD
"""
# FIXME! find a new home for this file!
pyrobot_to_canonical_frame = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def base_canonical_coords_to_pyrobot_coords(xyt):
    """converts the robot's base coords from canonical to pyrobot coords."""
    return [xyt[1], -xyt[0], xyt[2]]


def xyz_pyrobot_to_canonical_coords(xyz):
    """converts 3D coords from pyrobot to canonical coords."""
    return xyz @ pyrobot_to_canonical_frame


def xyz_canonical_coords_to_pyrobot_coords(xyz):
    """converts 3D coords from canonical to pyrobot coords."""
    return xyz @ np.linalg.inv(pyrobot_to_canonical_frame)
