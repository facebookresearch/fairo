"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import math
import numpy as np
from numpy import sin, cos

# CANONICAL WORLD COORDS
# coords are (x, y, z)
# 0 yaw is x axis
#                 z+
#                 |  
#                 |
#        +yaw     |   -yaw
#                 |   
#    x-___________|___________x+
#                 |
#                 | 
#                 z-
# 
#         ^ y+
#         |     z+
#         |   /
#         | /
#         0 -----> x+
#
#             y+  
#             |
#             |   -pitch
#             |   
#    z-_______|________z+
#             |   
#             |   
#             |   +pitch
#             |     
#             y-


DIRECTIONS = {
    "AWAY":  np.array([ 0,  0,  1]),
    "FRONT": np.array([ 0,  0,  1]),
    "BACK":  np.array([ 0,  0, -1]),
    "LEFT":  np.array([-1,  0,  0]),
    "RIGHT": np.array([ 1,  0,  0]),
    "DOWN":  np.array([ 0, -1,  0]),
    "UP":    np.array([ 0,  1,  0]),
}

# FIXME add the xz_only option for mc also, shouldn't use yaw for determining "up"
def transform(direction, yaw, pitch, inverted=False, xz_only=False):
    """Coordinate transforms with respect to current yaw/pitch of the viewer direction

    inverted=True finds the vector in canonical coords pointing towards
    the vector <direction> where <direction> is specified as if you were 
    facing yaw, pitch. (i.e. viewer-->canonical)

    inverted=False takes a <direction> in canonical
    coordinates and converts to coordinates where FRONT (z+) 
    is yaw=0 and pitch=0.  (i.e. canonical-->viewer)
    
    yaw is assumed to be in the range [-pi, pi], and
    increasing yaw moves *counterclockwise*.
    
    pitch is assumed to be in
    the range [-pi/2, pi/2].  pi/2 is down, -pi/2 is up.
    """

    # fmt: off
    ryaw = np.array([[cos(yaw), 0,    sin(yaw)],
                   [0,          1,    0        ],
                   [-sin(yaw),   0,    cos(yaw) ]])

    rpitch = np.array([[1,   0,             0          ],
                       [0,   cos(pitch),   sin(pitch)],
                       [0,   -sin(pitch),  cos(pitch)]])
    # fmt: on

    if not inverted:
        trans_mat = rpitch @ ryaw
    else:
        trans_mat = np.linalg.inv(rpitch @ ryaw)
    return trans_mat @ direction


def yaw_pitch(look_vec):
    """
    returns the yaw and pitch from the input look_vec.
    the look_vec should be input according to the diagram above
    the yaw, pitch is returned according to the diagram above
    """
    xz_dir = np.array([look_vec[0], look_vec[2]])
    xz_dir = xz_dir / np.linalg.norm(xz_dir)
    yaw = np.arctan2(-xz_dir[0], xz_dir[1])
    
    # get the pitch value/tilt angle
    pitch = np.arctan2(look_vec[1], np.sqrt(look_vec[0] ** 2 + look_vec[2] ** 2))
    
    yaw = yaw % (2 * np.pi)
    if yaw > np.pi:
        yaw = yaw - 2 * np.pi
        return yaw, pitch
    return yaw, pitch
    
    
def look_vec(yaw, pitch):
    """
    returns look_vec from input pitch and yaw. 
    this should invert yaw_pitch (up to norm)
    the look_vec is returned according to the diagram above
    the yaw, pitch should be input according to the diagram above
    """
    #    yaw = deg2rad(yaw)
    #    pitch = deg2rad(pitch)
    x = cos(pitch) * sin(-yaw)
    y = -sin(pitch)
    z = cos(pitch) * cos(yaw)
    return np.array([x, y, z])


def rotation_matrix_x(a):
    ar = float(a) * math.pi / 180.
    cos = math.cos
    sin = math.sin
    return np.array([
        [1, 0, 0],
        [0, cos(ar), -sin(ar)],
        [0, sin(ar), cos(ar)]
    ])


def rotation_matrix_y(a):
    ar = float(a) * math.pi / 180.
    cos = math.cos
    sin = math.sin
    return np.array([[cos(ar), 0, sin(ar)],
                     [0, 1, 0],
                     [-sin(ar), 0, cos(ar)]])

def rotation_matrix_z(a):
    ar = float(a) * math.pi / 180.
    cos = math.cos
    sin = math.sin
    return np.array([
        [cos(ar), -sin(ar), 0],
        [sin(ar), cos(ar), 0],
        [0, 0, 1],
    ])


if __name__ == "__main__":
    A = (4, 0, 1)
    B = (4, 4, 4)
    print(transform(DIRECTIONS["RIGHT"], 45, 0, inverted=True))
    print("yaw_pitch(look_vec(3.1, -1.0))")
    print(yaw_pitch(look_vec(3.1, -1.0)))
    print("look_vec(*yaw_pitch(np.array((-2,1,1))))")
    print(look_vec(*yaw_pitch(np.array((-2, 1, 1)))))
