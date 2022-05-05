// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

const MINECRAFT_BLOCK_MAP = {
    "0,0": 0, // Air
    "1,0": 1, // Stone
    "1,1": 2, // Granite
    "1,2": 3, // Polished Granite
    "1,3": 4, // Diorite
    "1,4": 5, // Polished Diorite
    "1,5": 6, // Andesite
    "1,6": 7, // Polished Andesite
    "2,0": 8, // Grass
    "3,0": 9, // Dirt
    "3,1": 10, // Coarse Dirt
    "3,2": 11, // Podzol
    "4,0": 12, // Cobblestone
    "5,0": 13, // Oak Wood Plank
    "5,1": 14, // Spruce Wood Plank
    "5,2": 15, // Birch Wood Plank
    "5,3": 16, // Jungle Wood Plank
    "5,4": 17, // Acacia Wood Plank
    "5,5": 18, // Dark Oak Wood Plank
    "6,0": 19, // Oak Sapling
    "6,1": 20, // Spruce Sapling
    "6,2": 21, // Birch Sapling
    "6,3": 22, // Jungle Sapling
    "6,4": 23, // Acacia Sapling
    "6,5": 24, // Dark Oak Sapling
    "7,0": 25, // Bedrock
    "8,0": 26, // Flowing Water
    "9,0": 27, // Still Water
    "12,0": 28, // Sand
    "12,1": 29, // Red Sand
    "13,0": 30, // Gravel
    "17,0": 31, // Oak Wood
    "17,1": 32, // Spruce Wood
    "17,2": 33, // Birch Wood
    "17,3": 34, // Jungle Wood
    "18,0": 35, // Oak Leaves
    "18,1": 36, // Spruce Leaves
    "18,2": 37, // Birch Leaves
    "18,3": 38, // Jungle Leaves
    "24,0": 39, // Sandstone
    "24,1": 40, // Chiseled Sandstone
    "24,2": 41, // Smooth Sandstone
    "31,0": 42, // Dead Shrub
    "31,1": 43, // Grass
    "31,2": 44, // Fern
    "32,0": 45, // Dead Bush
    "35,0": 46, // White Wool
    "35,1": 47, // Orange Wool
    "35,2": 48, // Magenta Wool
    "35,3": 49, // Light Blue Wool
    "35,4": 50, // Yellow Wool
    "35,5": 51, // Lime Wool
    "35,6": 52, // Pink Wool
    "35,7": 53, // Gray Wool
    "35,8": 54, // Light Gray Wool
    "35,9": 55, // Cyan Wool
    "35,10": 56, // Purple Wool
    "35,11": 57, // Blue Wool
    "35,12": 58, // Brown Wool
    "35,13": 59, // Green Wool
    "35,14": 60, // Red Wool
    "35,15": 61, // Black Wool
    "37,0": 62, // Dandelion
    "38,0": 63, // Poppy
    "39,0": 64, // Brown Mushroom
    "40,0": 65, // Red Mushroom
    "95,4": 66, // Yellow Stained Glass
  };

const VW_ITEM_MAP = {
    // 1: null, // Stone
    // 8: null, // Grass
    // 9: null, // Dirt
    // 12: null, // Cobblestone
    // 13: null, // Oak Plank
    // 25: null, // Bedrock
    // 28: null, // Sand

    // 46: 0xffffff, // White Wool
    // 47: 0xffa500, // Orange Wool
    // 48: 0xff00ff, // Magenta Wool
    // 49: 0x75bdff, // Light Blue Wool
    // 50: 0xffff00, // Yellow Wool
    // 51: 0x00ff00, // Lime Wool
    // 52: 0xffc0cb, // Pink Wool
    // 53: 0x5b5b5b, // Gray Wool
    // 54: 0xbcbcbc, // Light Gray Wool
    // 55: 0x00ffff, // Cyan Wool
    // 56: 0x800080, // Purple Wool
    // 57: 0x2986cc, // Blue Wool
    // 58: 0xdd992c, // Brown Wool
    // 59: 0x8fce00, // Green Wool
    // 60: 0xff0000, // Red Wool
    // 61: 0x1a1a1a, // Black Wool

    // 66: null, // Yellow Glass

    "white wool": 0xffffff, // White Wool
    "orange wool": 0xffa500, // Orange Wool
    "magenta wool": 0xff00ff, // Magenta Wool
    "light blue wool": 0x75bdff, // Light Blue Wool
    "yellow wool": 0xffff00, // Yellow Wool
    "lime wool": 0x00ff00, // Lime Wool
    "pink wool": 0xffc0cb, // Pink Wool
    "gray wool": 0x5b5b5b, // Gray Wool
    "light gray wool": 0xbcbcbc, // Light Gray Wool
    "cyan wool": 0x00ffff, // Cyan Wool
    "pruple wool": 0x800080, // Purple Wool
    "blue wool": 0x2986cc, // Blue Wool
    "brown wool": 0xdd992c, // Brown Wool
    "green wool": 0x8fce00, // Green Wool
    "red wool": 0xff0000, // Red Wool
    "black wool": 0x1a1a1a, // Black Wool
};

const VW_MOB_MAP = {
    // 50: "creeper",
    // 51: "skeleton",
    // 52: "spider",
    // 53: "giant",
    // 54: "zombie",
    // 55: "slime",
    // 56: "ghast",
    // 57: "pig zombie",
    // 58: "enderman",
    // 59: "cave spider",
    // 60: "silverfish",
    // 61: "blaze",
    // 62: "lava slime",
    // 63: "ender dragon",
    // 64: "wither boss",
    // 65: "bat",
    // 66: "witch",
    // 68: "guardian",
    // 90: "pig",
    // 91: "sheep",
    // 92: "cow",
    // 93: "chicken",
    // 94: "squid",
    // 95: "wolf",
    // 96: "mushroom cow",
    // 97: "snow man",
    // 98: "ozelot",
    // 99: "villager golem",
    // 100: "entity horse",
    // 101: "rabbit",
    // 120: "villager",

    "bat": null,
    "cat": null,
    "chicken": {
        "model_folder": "chicken_model/",
        "model_file": "chicken.gltf",
        "default_scale": 90.0,
        "rotation_offset": [0, 0, 0],
        "position_offset": [11, 0, 23] 
    },
    "cow": null,
    "donkey": null,
    "fox": null,
    "horse": null,
    "parrot": null,
    "pig": null,
    "rabbit": null,
    "sheep": null,
    "wolf": null,
};

export { MINECRAFT_BLOCK_MAP, VW_ITEM_MAP, VW_MOB_MAP };