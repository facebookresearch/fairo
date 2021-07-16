// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <set>
#include "types.h"

// Data from:
//
// https://github.com/PrismarineJS/minecraft-data/blob/master/data/pe/1.0/blocks.json
//
// which is MIT licensed.
//
// Changes:
// - Added 38 (brown mushroom) and 39 (red mushroom) to WALKTHROUGH_BLOCKS
// - Added 55 (torch) and 171 (carpet) to WALKTHROUGH_BLOCKS

const std::set<uint16_t> WALKTHROUGH_BLOCKS = {
    0,   6,   8,   9,   10,  11,  27,  28,  30,  31,  32,  37,  38,  39,  40,  50,  51,
    55,  59,  63,  66,  68,  69,  70,  72,  75,  76,  77,  83,  90,  92,  104, 105, 106,
    115, 119, 131, 132, 141, 142, 143, 147, 148, 157, 171, 175, 176, 177, 207, 209,
};

const Block BLOCK_AIR = {0, 0};

const uint8_t BLOCK_ID_YELLOW_FLOWER = 37;
const uint8_t BLOCK_ID_RED_FLOWER = 38;
