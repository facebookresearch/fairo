// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

const BLOCK_MAP = {
    1: 0xbcbcbc,  // Stone
    2: 0xbcbcbc,  // Granite
    3: 0xbcbcbc,  // Polished Granite
    4: 0xbcbcbc,  // Diorite
    5: 0xbcbcbc,  // Polished Diorite
    6: 0xbcbcbc,  // Andesite
    7: 0xbcbcbc,  // Polished Andesite
    8: 0x8fce00,  // Grass
    9: 0xdd992c,  // Dirt
    10: 0xdd992c,  // Coarse Dirt
    11: 0xbcbcbc,  // Podzol
    12: 0xbcbcbc,  // Cobblestone
    13: 0xdd992c,  // Oak Wood Plank
    14: 0xdd992c,  // Spruce Wood Plank
    15: 0xdd992c,  // Birch Wood Plank
    16: 0xdd992c,  // Jungle Wood Plank
    17: 0xdd992c,  // Acacia Wood Plank
    18: 0xdd992c,  // Dark Oak Wood Plank
    19: 0xdd992c,  // Oak Sapling
    20: 0xdd992c,  // Spruce Sapling
    21: 0xdd992c,  // Birch Sapling
    22: 0xdd992c,  // Jungle Sapling
    23: 0xdd992c,  // Acacia Sapling
    24: 0xdd992c,  // Dark Oak Sapling
    25: 0xbcbcbc,  // Bedrock
    26: 0x75bdff,  // Flowing Water
    27: 0x75bdff,  // Still Water
    28: 0xffa500,  // Sand
    29: 0xff0000,  // Red Sand
    30: 0xdd992c,  // Gravel
    31: 0xdd992c,  // Oak Wood
    32: 0xdd992c,  // Spruce Wood
    33: 0xdd992c,  // Birch Wood
    34: 0xdd992c,  // Jungle Wood
    35: 0x8fce00,  // Oak Leaves
    36: 0x8fce00,  // Spruce Leaves
    37: 0x8fce00,  // Birch Leaves
    38: 0x8fce00,  // Jungle Leaves
    39: 0xdd992c,  // Sandstone
    40: 0xdd992c,  // Chiseled Sandstone
    41: 0xdd992c,  // Smooth Sandstone
    42: 0xdd992c,  // Dead Shrub
    43: 0x8fce00,  // Grass
    44: 0x8fce00,  // Fern
    45: 0xdd992c,  // Dead Bush

    46: 0xffffff, // White Wool
    47: 0xffa500, // Orange Wool
    48: 0xff00ff, // Magenta Wool
    49: 0x75bdff, // Light Blue Wool
    50: 0xffff00, // Yellow Wool
    51: 0x00ff00, // Lime Wool
    52: 0xffc0cb, // Pink Wool
    53: 0x5b5b5b, // Gray Wool
    54: 0xbcbcbc, // Light Gray Wool
    55: 0x00ffff, // Cyan Wool
    56: 0x800080, // Purple Wool
    57: 0x2986cc, // Blue Wool
    58: 0xdd992c, // Brown Wool
    59: 0x8fce00, // Green Wool
    60: 0xff0000, // Red Wool
    61: 0x1a1a1a, // Black Wool

    62: 0x8fce00,  // Dandelion
    63: 0x8fce00,  // Poppy
    64: 0xdd992c,  // Brown Mushroom
    65: 0xff0000,  // Red Mushroom

    67: 0xffa500, // Orange Hole
    68: 0xff00ff, // Magenta Hole
    69: 0x75bdff, // Light Blue Hole
    70: 0xffff00, // Yellow Hole
    71: 0x00ff00, // Lime Hole
    72: 0xffc0cb, // Pink Hole
    73: 0x5b5b5b, // Gray Hole
    74: 0xbcbcbc, // Light Gray Hole
    75: 0x00ffff, // Cyan Hole
    76: 0x800080, // Purple Hole
    77: 0x2986cc, // Blue Hole
    78: 0xdd992c, // Brown Hole
    79: 0x8fce00, // Green Hole
    80: 0xff0000, // Red Hole
    81: 0x1a1a1a, // Black Hole
};

export { BLOCK_MAP };