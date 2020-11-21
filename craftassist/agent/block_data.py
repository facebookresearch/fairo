"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

NORMAL_BLOCKS_N = 454

PASSABLE_BLOCKS = (0, 8, 9, 31, 37, 38, 39, 40, 55, 106, 171, 175)
BORING_BLOCKS = (0, 1, 2, 3, 6, 12, 31, 32, 37, 38, 39, 40)
REMOVABLE_BLOCKS = BORING_BLOCKS + (106,)

# don't bother trying to remove vines, since they grow
BUILD_IGNORE_BLOCKS = (106,)

# don't care about the difference between grass and dirt; grass grows and
# messes things up
BUILD_INTERCHANGEABLE_PAIRS = [(2, 3)]

BUILD_BLOCK_REPLACE_MAP = {
    8: (0, 0),  # replace still water with air
    9: (0, 0),  # replace flowing water with air
    13: (1, 0),  # replace gravel with stone
}


# TODO map with all properties of blocks, and function to search
# e.g. see through/translucent, color, material (wood vs stone vs metal), etc.
"""Mapping from colour to blockid/ meta"""
COLOR_BID_MAP = {
    "aqua": [(35, 3), (35, 9), (95, 3), (57, 0)],
    "black": [(35, 15), (49, 0), (95, 15)],
    "blue": [(35, 11), (22, 0), (57, 0), (79, 0)],
    "fuchsia": [(35, 2)],
    "green": [(133, 0), (35, 13), (95, 13), (159, 13)],
    "gray": [(1, 0), (1, 6)],
    "lime": [(159, 5), (95, 5)],
    "maroon": [(112, 0)],
    "navy": [(159, 11)],
    "olive": [(159, 13)],
    "purple": [(95, 10), (35, 10)],
    "red": [(215, 0), (152, 0), (35, 14)],
    "silver": [(155, 0)],
    "teal": [(57, 0)],
    "white": [(43, 7), (42, 0)],
    "yellow": [(35, 4), (41, 0)],
    "orange": [(35, 1), (95, 1), (159, 1)],
    "brown": [(5, 5), (95, 12), (159, 12)],
    "pink": [(35, 6), (95, 6), (159, 6)],
    "gold": [(41, 0)],
}

"""Grouping of blocks to block types"""
BLOCK_GROUPS = {
    "building_block": [
        1,
        5,
        24,
        43,
        44,
        45,
        48,
        98,
        112,
        125,
        126,
        133,
        152,
        155,
        168,
        179,
        181,
        182,
        201,
        202,
        204,
        205,
        206,
        214,
        215,
        216,
        251,
    ],
    "plant": [37, 38, 39, 40, 175],
    "terracotta": [235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250],
    "water": [8, 9],
    "lava": [10, 11],
    "stairs": [53, 108, 109, 128, 134, 135, 136, 156, 163, 164, 180, 114, 203],
    "door": [64, 71, 193, 194, 195, 196, 197, 324, 330, 427, 428, 429, 430, 431],
    "ore": [14, 15, 16, 21, 56, 73, 74, 129, 153],
    "torch": [50, 75, 76],
    "wood": [17, 162],
    "shulkerbox": [219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234],
    "glass": [20, 95, 102, 160],
    "gate": [107, 183, 184, 185, 186, 187],
    "fence": [85, 113, 188, 189, 190, 191, 192],
    "disc": [2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267],
}
