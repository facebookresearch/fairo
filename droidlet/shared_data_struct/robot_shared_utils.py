from collections import namedtuple


Pos = namedtuple("pos", ["x", "y", "z"])
Marker = namedtuple("Marker", "markerId pos color category properties")
RobotPerceptionData = namedtuple(
    "perception",
    ["new_objects", "updated_objects", "humans", "obstacle_map", "self_pose"],
    defaults=[None, None, None, [], None],
)
