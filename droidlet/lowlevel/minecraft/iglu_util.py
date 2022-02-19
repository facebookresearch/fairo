"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains IGLU related utility functions
"""


"""
Mapping between minecraft (id, meta) and IGLU single number texture index
"""
IGLU_BLOCK_MAP = {
    (0, 0): 0,  # Air
    (1, 0): 1,  # Stone
    (1, 1): 2,  # Granite
    (1, 2): 3,  # Polished Granite
    (1, 3): 4,  # Diorite
    (1, 4): 5,  # Polished Diorite
    (1, 5): 6,  # Andesite
    (1, 6): 7,  # Polished Andesite
    (2, 0): 8,  # Grass
    (3, 0): 9,  # Dirt
    (3, 1): 10,  # Coarse Dirt
    (3, 2): 11,  # Podzol
    (4, 0): 12,  # Cobblestone
    (5, 0): 13,  # Oak Wood Plank
    (5, 1): 14,  # Spruce Wood Plank
    (5, 2): 15,  # Birch Wood Plank
    (5, 3): 16,  # Jungle Wood Plank
    (5, 4): 17,  # Acacia Wood Plank
    (5, 5): 18,  # Dark Oak Wood Plank
    (6, 0): 19,  # Oak Sapling
    (6, 1): 20,  # Spruce Sapling
    (6, 2): 21,  # Birch Sapling
    (6, 3): 22,  # Jungle Sapling
    (6, 4): 23,  # Acacia Sapling
    (6, 5): 24,  # Dark Oak Sapling
    (7, 0): 25,  # Bedrock
    (8, 0): 26,  # Flowing Water
    (9, 0): 27,  # Still Water
    (12, 0): 28,  # Sand
    (12, 1): 29,  # Red Sand
    (13, 0): 30,  # Gravel
    (17, 0): 31,  # Oak Wood
    (17, 1): 32,  # Spruce Wood
    (17, 2): 33,  # Birch Wood
    (17, 3): 34,  # Jungle Wood
    (18, 0): 35,  # Oak Leaves
    (18, 1): 36,  # Spruce Leaves
    (18, 2): 37,  # Birch Leaves
    (18, 3): 38,  # Jungle Leaves
    (24, 0): 39,  # Sandstone
    (24, 1): 40,  # Chiseled Sandstone
    (24, 2): 41,  # Smooth Sandstone
    (31, 0): 42,  # Dead Shrub
    (31, 1): 43,  # Grass
    (31, 2): 44,  # Fern
    (32, 0): 45,  # Dead Bush
    (35, 0): 46,  # White Wool
    (35, 1): 47,  # Orange Wool
    (35, 2): 48,  # Magenta Wool
    (35, 3): 49,  # Light Blue Wool
    (35, 4): 50,  # Yellow Wool
    (35, 5): 51,  # Lime Wool
    (35, 6): 52,  # Pink Wool
    (35, 7): 53,  # Gray Wool
    (35, 8): 54,  # Light Gray Wool
    (35, 9): 55,  # Cyan Wool
    (35, 10): 56,  # Purple Wool
    (35, 11): 57,  # Blue Wool
    (35, 12): 58,  # Brown Wool
    (35, 13): 59,  # Green Wool
    (35, 14): 60,  # Red Wool
    (35, 15): 61,  # Black Wool
    (37, 0): 62,  # Dandelion
    (38, 0): 63,  # Poppy
    (39, 0): 64,  # Brown Mushroom
    (40, 0): 65,  # Red Mushroom
}
